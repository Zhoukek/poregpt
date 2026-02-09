# -*- coding: utf-8 -*-
"""
根据特征目录下的所有 .npy 文件自动生成 shards.json
- 自动探测 feature_dim
- 按文件名排序确保顺序正确
- 仅依赖 numpy 和标准库
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path


def generate_shards_json(feat_dir: str):
    feat_dir = Path(feat_dir)
    if not feat_dir.is_dir():
        raise ValueError(f"Directory not found: {feat_dir}")

    # 获取所有 *_part*.npy 文件，并按名称排序（确保 part01, part02... 顺序）
    npy_files = sorted(feat_dir.glob("*_part*.npy"))
    if not npy_files:
        raise ValueError(f"No files matching *_part*.npy found in {feat_dir}")

    print(f"🔍 Found {len(npy_files)} .npy files.")

    # 自动探测 feature_dim：从第一个非空文件读取
    feature_dim = None
    for f in npy_files:
        data = np.load(f, mmap_mode='r')
        if data.size == 0:
            continue
        if data.ndim != 2:
            raise ValueError(f"File {f} is not 2D (shape: {data.shape})")
        feature_dim = data.shape[1]
        break

    if feature_dim is None:
        raise ValueError("All .npy files are empty or invalid.")

    print(f"🧠 Auto-detected feature_dim = {feature_dim}")

    # 遍历所有文件，构建 shards 列表
    shards_info = []
    total_tokens = 0

    for npy_file in npy_files:
        data = np.load(npy_file, mmap_mode='r')
        if data.ndim != 2 or data.shape[1] != feature_dim:
            raise ValueError(
                f"Inconsistent shape in {npy_file}: expected (?, {feature_dim}), got {data.shape}"
            )
        num_tokens = data.shape[0]
        shards_info.append({
            "shard_file": npy_file.name,
            "start_token_index": total_tokens,
            "num_tokens": num_tokens,
            "shape": [num_tokens, feature_dim]
        })
        total_tokens += num_tokens
        print(f"📄 {npy_file.name}: {num_tokens:,} tokens")

    # 构建最终 JSON
    shards_json = {
        "total_tokens": total_tokens,
        "feature_dim": feature_dim,
        "shard_size_max_tokens": max(s["num_tokens"] for s in shards_info),
        "shards": shards_info
    }

    # 保存
    output_path = feat_dir / "shards.json"
    with open(output_path, 'w') as f:
        json.dump(shards_json, f, indent=2)

    print(f"\n✅ Saved shards.json to: {output_path}")
    print(f"📊 Total tokens: {total_tokens:,}")
    print(f"🧩 Feature dimension: {feature_dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-generate shards.json from *_part*.npy files (feature_dim auto-detected)."
    )
    parser.add_argument("--feat_dir", type=str, required=True,
                        help="Directory containing *_partXX.npy files")

    args = parser.parse_args()
    generate_shards_json(args.feat_dir)
