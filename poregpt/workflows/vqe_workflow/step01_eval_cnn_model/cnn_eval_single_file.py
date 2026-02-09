# -*- coding: utf-8 -*-
"""
Feature Extraction from a Single .npy File (for Nanopore Signal Tokenization)

Purpose:
    从单个 .npy 文件（包含多个信号 chunk）中提取 CNN token-level 特征，
    每个样本输出 T 个 64 维特征向量，按 token 粒度分片保存为 {basename}_partXX.npy，
    所有输出文件存入指定 output_dir。

Input:
    - input_npy_file: 单个 .npy 文件，shape 应为 [N, L]，其中 N 是样本数，L 是信号长度

Output:
    - output_dir/{basename}_part{index:02d}.npy: 每个文件包含最多 shard_size 个 token，shape [M, 64]

Note:
    - 全部数据加载到内存（适合中小规模）
    - 不生成 shards.json
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from poregpt.tokenizers.vqe_tokenizer.cnn_model import NanoporeCNNModel


def load_trained_cnn(checkpoint_path: str, cnn_type: int, device: str):
    model = NanoporeCNNModel(cnn_type=cnn_type)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    return model


def cnn_eval_single_file(
    input_npy_file: str,
    output_dir: str,
    checkpoint_path: str,
    shard_size: int = 1_000_000,
    feature_dim: int = 64,
    batch_size: int = 128,
    cnn_type: int = 1,
    device: str = 'cuda',
):
    """
    从单个 .npy 文件提取特征，分片保存为 {basename}_partXX.npy 到 output_dir
    """
    # === Step 1: 加载输入数据 ===
    print(f"📥 Loading data from {input_npy_file}...")
    signals = np.load(input_npy_file)

    # 确保形状为 [N, 1, L]
    if signals.ndim == 2:
        signals = signals[:, np.newaxis, :]  # [N, L] -> [N, 1, L]
    elif signals.ndim == 3:
        if signals.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {signals.shape[1]} in shape {signals.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {signals.shape}")

    total_samples = signals.shape[0]
    signal_length = signals.shape[2]
    print(f"📊 Total samples: {total_samples:,}, Signal length: {signal_length}")

    # === Step 2: 创建输出目录 ===
    os.makedirs(output_dir, exist_ok=True)

    # === Step 3: 加载模型 ===
    model = load_trained_cnn(checkpoint_path, cnn_type, device)

    # === Step 4: 探测 T（时间步长）===
    with torch.no_grad():
        dummy_input = torch.from_numpy(signals[:1]).to(device).float()
        feat = model.encode(dummy_input)  # [1, C, T]
        assert feat.dim() == 3 and feat.shape[1] == feature_dim
        T = feat.shape[2]
    print(f"🔍 Time steps per sample: {T}, Feature dim: {feature_dim}")

    total_tokens = total_samples * T
    print(f"🔢 Total tokens to extract: {total_tokens:,}")

    # === Step 5: 准备输出文件名前缀 ===
    base_name = os.path.splitext(os.path.basename(input_npy_file))[0]

    # === Step 6: 初始化缓冲区 ===
    buffer = []  # 存储 token 向量
    part_index = 1

    pbar = tqdm(total=total_tokens, desc="Extracting features")

    def _flush_buffer():
        nonlocal buffer, part_index
        if not buffer:
            return
        shard_data = np.stack(buffer, axis=0).astype(np.float32)  # [M, 64]
        out_path = os.path.join(output_dir, f"{base_name}_part{part_index:02d}.npy")
        np.save(out_path, shard_data)
        print(f"🆕 Saved: {out_path} ({len(buffer)} tokens)")
        buffer.clear()
        part_index += 1

    # === Step 7: 主推理循环（按 batch 处理）===
    model.eval()
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_signals = signals[i:i + batch_size]  # [B, L]
            batch_tensor = torch.from_numpy(batch_signals).to(device).float()

            feats = model.encode(batch_tensor)          # [B, C, T]
            feats = feats.permute(0, 2, 1)              # [B, T, C]
            feats = feats.reshape(-1, feature_dim)      # [B*T, C]
            feats_np = feats.cpu().numpy().astype(np.float32)

            # 添加到 buffer
            for token in feats_np:
                buffer.append(token)
                pbar.update(1)
                if len(buffer) >= shard_size:
                    _flush_buffer()

    # === Step 8: 写入剩余数据 ===
    if buffer:
        _flush_buffer()

    print(f"\n✅ Done! Output files saved in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CNN features from a single .npy signal file."
    )
    parser.add_argument("--input_npy_file", type=str, required=True,
                        help="Path to input .npy file (shape: [N, signal_length])")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output part files")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained CNN checkpoint (.pth)")
    parser.add_argument("--shard_size", type=int, default=1_000_000,
                        help="Max number of tokens per output part file")
    parser.add_argument("--feature_dim", type=int, default=64,
                        help="Expected feature dimension")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size")
    parser.add_argument("--cnn_type", type=int, default=1,
                        help="Model configuration identifier")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference ('cuda' or 'cpu')")

    args = parser.parse_args()
    cnn_eval_single_file(**vars(args))
