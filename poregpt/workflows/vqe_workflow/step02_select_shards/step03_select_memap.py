#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 sample_ratio 对 shards.json 中的 shard 条目进行随机采样，
生成新的 shards_sampled_XXp.json，不移动/复制 .npy 文件。
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a sampled shards.json without moving .npy files.")
    parser.add_argument("--feature_shards_dir", type=str, required=True,
                        help="Directory containing original shards.json and shard_*.npy files")
    parser.add_argument("--sample_ratio", type=float, default=0.5,
                        help="Probability to keep each shard (default: 0.5)")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Suffix for output file, e.g., '50p'. If not given, auto from ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    feature_shards_dir = Path(args.feature_shards_dir)
    shards_json_path = feature_shards_dir / "shards.json"
    assert shards_json_path.exists(), f"❌ shards.json not found at {shards_json_path}"

    with open(shards_json_path, 'r') as f:
        meta = json.load(f)

    original_shards = meta["shards"]
    feature_dim = meta["feature_dim"]

    # Sample shards by index
    selected_shards = []
    total_tokens = 0

    for shard in original_shards:
        if np.random.rand() < args.sample_ratio:
            selected_shards.append(shard)
            total_tokens += shard["num_tokens"]

    if not selected_shards:
        raise RuntimeError("⚠️ No shards selected! Reduce sample_ratio or check data.")

    # Determine output filename
    suffix = args.output_suffix or f"{int(args.sample_ratio * 100)}p"
    output_file = feature_shards_dir / f"shards_sampled_{suffix}.json"

    # Build new meta
    new_meta = {
        "total_tokens": total_tokens,
        "feature_dim": feature_dim,
        "shards": selected_shards
    }

    with open(output_file, 'w') as f:
        json.dump(new_meta, f, indent=2)

    print(f"✅ Sampled {len(selected_shards)} / {len(original_shards)} shards")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
