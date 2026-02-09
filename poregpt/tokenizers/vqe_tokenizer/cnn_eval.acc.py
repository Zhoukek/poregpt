# -*- coding: utf-8 -*-
"""
Feature Extraction Pipeline (Multi-GPU Accelerated)
使用 Hugging Face Accelerate 进行多卡并行推理，显著提升特征提取速度。
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed

# 导入本地模块
from .cnn_model import NanoporeCNNModel
from .dataset import NanoporeSignalDataset


def load_trained_cnn(checkpoint_path: str, cnn_type: int):
    """
    加载预训练模型（移除了device参数，由Accelerator统一管理）
    """
    model = NanoporeCNNModel(cnn_type=cnn_type)
    
    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location="cpu",weights_only=False)  # 先加载到 CPU
    state_dict = ckpt['model_state_dict']
    
    # 兼容 DDP 训练保存的权重 (去除 'module.' 前缀)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式
    
    return model


def cnn_eval(
    input_shards_dir: str,
    output_shard_dir: str,
    checkpoint_path: str,
    shard_size: int = 1_000_000,
    feature_dim: int = 64,
    batch_size: int = 128,
    num_workers: int = 8,
    cnn_type: int = 1,
    device: str = 'cuda',  # 保留参数接口，但实际由 Accelerator 决定
):
    """
    主特征提取函数 (支持多GPU分布式推理)
    """
    # === 1. 初始化 Accelerator ===
    # mixed_precision 可选 'no', 'fp16', 'bf16'。推理任务通常用 'no' 保证精度。
    accelerator = Accelerator(mixed_precision='no', device_placement=True)
    
    # 为了保持原有参数接口，这里覆盖 device 变量
    device = accelerator.device
    print(f"🚀 当前进程 [{accelerator.process_index}] 使用设备: {device}")

    # === 2. 创建输出目录 (仅主进程) ===
    # 避免多个进程同时创建目录导致报错
    if accelerator.is_main_process:
        os.makedirs(output_shard_dir, exist_ok=True)
    # 同步所有进程，确保目录已创建
    accelerator.wait_for_everyone()

    # === 3. 准备数据集与分布式采样器 ===
    dataset = NanoporeSignalDataset(shards_dir=input_shards_dir)
    
    # 使用 DistributedSampler 切分数据。每个 GPU 只处理数据集的一个子集。
    sampler = DistributedSampler(
        dataset, 
        num_replicas=accelerator.num_processes, 
        rank=accelerator.process_index, 
        shuffle=False, 
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=4096,
        sampler=sampler,  # 使用 sampler 代替 shuffle
        num_workers=num_workers,
        pin_memory=True,  # 加速数据传输
        drop_last=False
    )

    total_samples = len(dataset)
    print(f"📊 总样本数: {total_samples:,} (当前进程处理约 {len(dataloader.dataset)} 个)")

    # === 4. 模型加载与准备 ===
    model = load_trained_cnn(checkpoint_path, cnn_type)
    
    # 将模型和数据加载器交给 Accelerator 包装
    # 它会自动处理 .to(device) 和 DistributedDataParallel 包装
    model, dataloader = accelerator.prepare(model, dataloader)

    # === 5. 探测输出维度 T (仅主进程执行，避免重复计算) ===
    T = None
    if accelerator.is_main_process:
        # 临时探测
        with torch.no_grad():
            temp_batch = next(iter(dataloader))[:1].to(device)
            feat = model.module.encode(temp_batch)
            T = feat.shape[2]
            assert feat.shape[1] == feature_dim, "通道数不匹配"
    
    # 将探测到的 T 广播给所有进程
    T = accelerator.gather(T).max().item() if not accelerator.is_main_process else T
    accelerator.wait_for_everyone()

    total_tokens = total_samples * T
    print(f"🔍 探测到时间步长 T: {T}, 总 Token 数: {total_tokens:,}")

    # === 6. 特征写入逻辑 (仅主进程) ===
    # 只有主进程 (process_index == 0) 负责写入文件，防止文件冲突
    buffer = []
    shard_index = 0
    global_token_idx = 0
    shards_info = []

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_tokens, desc="提取特征中...")

    def _flush_buffer():
        nonlocal buffer, shard_index, global_token_idx, shards_info
        
        # 如果不是主进程，或者 buffer 为空，直接返回
        if not accelerator.is_main_process or not buffer:
            return
            
        current_shard_size = len(buffer)
        shard_file = f"shard_{shard_index:05d}.npy"
        shard_path = os.path.join(output_shard_dir, shard_file)

        shard_data = np.stack(buffer, axis=0).astype(np.float32)
        
        # 使用 memmap 写入
        memmap = np.memmap(shard_path, dtype='float32', mode='w+', shape=shard_data.shape)
        memmap[:] = shard_data
        del memmap

        shards_info.append({
            "shard_file": shard_file,
            "start_token_index": global_token_idx,
            "num_tokens": current_shard_size,
            "shape": [current_shard_size, feature_dim]
        })

        print(f"🆕 已创建分片: {shard_path} ({current_shard_size} 个 Token)")
        global_token_idx += current_shard_size
        buffer.clear()
        shard_index += 1

    # === 7. 主推理循环 ===
    with torch.no_grad():
        for batch in dataloader:
            # 前向传播
            feats = model.module.encode(batch)          # [B, C, T]
            feats = feats.permute(0, 2, 1)       # [B, T, C]
            feats = feats.reshape(-1, feature_dim) # [B*T, C]
            
            # 将 Tensor 转换为 NumPy (仅在主进程需要)
            if accelerator.is_main_process:
                feats_np = accelerator.gather_for_metrics(feats).cpu().numpy()
                # 注意：使用 gather_for_metrics 而不是 gather，避免在拼接维度上出错
                # 如果 batch_size 很小，可以直接用 .cpu().numpy()
                for i in range(feats_np.shape[0]):
                    buffer.append(feats_np[i])
                    pbar.update(1)
                    if len(buffer) >= shard_size:
                        _flush_buffer()
            else:
                # 非主进程只需占位，确保进程同步
                # 如果不执行 gather，进程间会因为数据量不同而卡住
                _ = accelerator.gather_for_metrics(feats)

    # === 8. 收尾工作 (仅主进程) ===
    if accelerator.is_main_process:
        # 写入剩余 buffer
        if buffer:
            _flush_buffer()
        
        # 保存元数据
        shards_json_path = os.path.join(output_shard_dir, "shards.json")
        with open(shards_json_path, 'w') as f:
            json.dump({
                "total_samples": total_samples,
                "tokens_per_sample": T,
                "total_tokens": total_tokens,
                "feature_dim": feature_dim,
                "shard_size_max_tokens": shard_size,
                "shards": shards_info
            }, f, indent=2)
        
        print(f"\n✅ 特征提取完成！保存路径: {output_shard_dir}")
        print(f"📄 元数据文件: {shards_json_path}")

    # 确保所有进程在此处同步完成
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多卡加速版特征提取器")
    parser.add_argument("--input_shards_dir", type=str, required=True)
    parser.add_argument("--output_shard_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=1_000_000)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cnn_type", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda") # 保留接口，实际由 accelerate 决定

    args = parser.parse_args()
    
    # 使用 notebook_launcher 启动，它会自动处理多进程分发
    # 如果是命令行直接运行，Accelerator 会自动检测环境
    # cpu=True 表示允许在 CPU 上运行（如果没检测到 GPU）
    cnn_eval(**vars(args))
