# -*- coding: utf-8 -*-
"""
Feature Extraction Pipeline for Nanopore Signal Tokenization

Purpose:
    从原始纳米孔电信号中提取基于 CNN 的 token-level 特征（embeddings），
    每个输入样本将被转换为 T 个 64 维特征向量（T 由模型结构决定，如 2400），
    所有特征按 token 粒度分片存储为 .npy 文件，并附带结构化元数据（shards.json）。

Design Philosophy:
    - 支持大规模数据（TB 级）：通过内存映射（memmap）避免 OOM；
    - 可恢复性：每个 shard 独立，支持中断后续跑；
    - 元数据完备：记录样本数、token 数、维度、分片边界等，便于下游加载；
    - 与训练/建模流程解耦：仅依赖 checkpoint 和输入 shard 目录。

Input:
    - input_shards_dir: 原始信号分片目录（由 dataset.NanoporeSignalDataset 读取）
    - checkpoint_path: 训练好的 CNN 模型权重路径

Output:
    - output_shard_dir/
        ├── shard_00000.npy   # shape: [N, 64], float32
        ├── shard_00001.npy
        └── shards.json       # 元数据索引文件

Note:
    模型输出格式为 [B, C, T]（PyTorch Conv1d 标准 NCL 格式），
    本脚本将其转置为 [B, T, C] 后展平为 [B*T, C]，以符合 token 序列惯例。
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 导入本地模块：CNN 模型定义 + 信号数据集
from .cnn_model import NanoporeCNNModel
from .dataset import NanoporeSignalDataset


def load_trained_cnn(checkpoint_path: str, cnn_type: int, device: str):
    """
    加载预训练的 Nanopore CNN 模型（支持单卡 / DDP 权重兼容）

    Args:
        checkpoint_path (str): 模型 checkpoint 路径，需包含 'model_state_dict'
        cnn_type (int): 模型配置类型（用于初始化 NanoporeCNNModel）
        device (str): 推理设备（'cpu' 或 'cuda'）

    Returns:
        torch.nn.Module: 已加载权重并切换至 eval 模式的模型

    Notes:
        - 自动处理 DDP 训练保存的权重（key 前缀 'module.'）
        - 不加载优化器等无关状态，仅加载模型参数
    """
    # 初始化模型结构（必须与训练时一致）
    model = NanoporeCNNModel(cnn_type=cnn_type)

    # 安全加载 checkpoint（weights_only=False 兼容旧版 PyTorch）
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # 兼容 DDP（DistributedDataParallel）训练保存的权重
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        # 移除 'module.' 前缀以适配单卡推理
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 加载权重（严格匹配）
    model.load_state_dict(state_dict)

    # 切换到评估模式（关闭 dropout / BN 更新）
    model.to(device).eval()

    print(f"✅ Loaded model from {checkpoint_path}")
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
    device: str = 'cuda',
):
    """
    主特征提取函数：逐 batch 提取 token-level embeddings 并分片持久化

    Workflow:
        1. 加载模型与数据集
        2. 推断时间步长 T（从模型输出动态获取）
        3. 遍历数据集，对每个 batch：
            a. 前向传播得到 [B, C, T] 特征
            b. 转置为 [B, T, C] → 展平为 [B*T, C]
            c. 逐 token 写入 memmap 分片文件
        4. 生成 shards.json 元数据

    Args:
        input_shards_dir (str): 输入信号分片目录（需符合 NanoporeSignalDataset 格式）
        output_shard_dir (str): 输出特征分片目录（自动创建）
        checkpoint_path (str): 模型权重路径
        shard_size (int): 每个分片最多容纳的 token 数量（非样本数！）
        feature_dim (int): 期望的特征维度（应与模型输出通道数一致）
        batch_size (int): 推理批大小（影响 GPU 显存和吞吐）
        num_workers (int): DataLoader 子进程数（加速 I/O）
        cnn_type (int): 模型类型标识（用于初始化）
        device (str): 推理设备

    Side Effects:
        - 在 output_shard_dir 下创建 .npy 分片文件和 shards.json
        - 打印进度日志到 stdout
    """
    # 确保输出目录存在
    os.makedirs(output_shard_dir, exist_ok=True)

    # === Step 1: 加载模型与数据集 ===
    model = load_trained_cnn(checkpoint_path, cnn_type, device)
    dataset = NanoporeSignalDataset(shards_dir=input_shards_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # 必须为 False 以保证样本顺序可复现
        prefetch_factor=64,
        num_workers=num_workers,
        pin_memory=True,        # 加速 GPU 数据传输
        drop_last=False         # 保留最后不完整 batch
    )

    total_samples = len(dataset)
    print(f"📊 Total samples: {total_samples:,}")

    # === Step 2: 动态探测模型输出的时间步长 T ===
    # 注意：此处仅用第一个 batch 的第一个样本探测，避免重复计算
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            feat = model.encode(x[:1])  # shape: [1, C, T]
            # 验证输出为 3D 且通道数匹配
            assert feat.dim() == 3, f"Expected 3D output from model.encode(), got {feat.shape}"
            assert feat.shape[1] == feature_dim, (
                f"Channel dimension mismatch: expected {feature_dim}, got {feat.shape[1]}"
            )
            T = feat.shape[2]  # 时间步长（如 2400）
            break  # 仅需一次探测

    total_tokens = total_samples * T
    print(f"🔍 Feature dim: {feature_dim}, Time steps per sample: {T}")
    print(f"🔢 Total tokens to extract: {total_tokens:,}")

    # === Step 3: 初始化缓冲与分片状态 ===
    buffer = []                 # List of [C,] arrays, will be stacked when flushed
    shard_index = 0
    global_token_idx = 0
    shards_info = []

    pbar = tqdm(total=total_tokens, desc="Extracting features (by token)")

    def _flush_buffer():
        """将当前 buffer 一次性写入新 shard 文件"""
        nonlocal buffer, shard_index, global_token_idx, shards_info

        if not buffer:
            return

        current_shard_size = len(buffer)
        shard_file = f"shard_{shard_index:05d}.npy"
        shard_path = os.path.join(output_shard_dir, shard_file)

        # Stack into [N, C]
        shard_data = np.stack(buffer, axis=0).astype(np.float32)

        # Write via memmap (efficient and compatible)
        memmap = np.memmap(
            shard_path,
            dtype='float32',
            mode='w+',
            shape=shard_data.shape
        )
        memmap[:] = shard_data
        del memmap  # Ensure flush

        # Record metadata
        shards_info.append({
            "shard_file": shard_file,
            "start_token_index": global_token_idx,
            "num_tokens": current_shard_size,
            "shape": [current_shard_size, feature_dim]
        })

        print(f"🆕 Created: {shard_path} ({current_shard_size} tokens)")
        global_token_idx += current_shard_size
        buffer.clear()
        shard_index += 1

    # === Step 4: 主推理与写入循环 ===
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            feats = model.encode(x)                     # [B, C, T]
            feats = feats.permute(0, 2, 1)              # [B, T, C]
            feats = feats.reshape(-1, feature_dim)      # [B*T, C]
            feats_np = feats.cpu().numpy().astype(np.float32)
            num_tokens = feats_np.shape[0]

            # Add all tokens to buffer
            for i in range(num_tokens):
                buffer.append(feats_np[i])
                pbar.update(1)

                # Flush if buffer reaches shard_size
                if len(buffer) >= shard_size:
                    _flush_buffer()

        # Flush remaining tokens
        if buffer:
            _flush_buffer()

    # === Step 5: 保存全局元数据 ===
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

    # === Final Summary ===
    print(f"\n✅ Done! Features saved to: {output_shard_dir}/")
    print(f"📄 Index file: {shards_json_path}")
    print(f"🔢 Total shards: {len(shards_info)}")
    print(f"🧩 Each sample contributes {T} tokens of dim {feature_dim}")


if __name__ == "__main__":
    """
    命令行入口：解析参数并启动特征提取
    """
    parser = argparse.ArgumentParser(
        description="Extract token-level CNN features from nanopore signal shards."
    )
    parser.add_argument("--input_shards_dir", type=str, required=True,
                        help="Directory containing input signal shards (NanoporeSignalDataset format)")
    parser.add_argument("--output_shard_dir", type=str, required=True,
                        help="Output directory for feature shards")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained CNN checkpoint (.pth)")
    parser.add_argument("--shard_size", type=int, default=1_000_000,
                        help="Max number of tokens per output shard (default: 1M)")
    parser.add_argument("--feature_dim", type=int, default=64,
                        help="Expected feature dimension (must match model output channels)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size (adjust based on GPU memory)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of DataLoader workers for I/O parallelism")
    parser.add_argument("--cnn_type", type=int, default=1,
                        help="Model configuration identifier (passed to NanoporeCNNModel)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference ('cuda' or 'cpu')")

    args = parser.parse_args()
    cnn_eval(**vars(args))
