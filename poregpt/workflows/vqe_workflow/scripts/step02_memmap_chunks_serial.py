#!/usr/bin/env python3
"""
将目录下每个 .npy 文件（含 dict）转换为同名的 memmap 友好 .npy，
并生成 shards.json（按 num_samples 降序排列）。
单进程版本，防止内存爆炸。
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
from tqdm import tqdm


def convert_single_file(input_path, output_path, chunk_size, dtype):
    """
    转换单个 .npy 文件。
    返回: (filename, num_valid_chunks)
    """
    try:
        # 加载原始 pickle-based .npy（全量加载，但单进程可控）
        data = np.load(input_path, allow_pickle=True)
        valid_chunks = []
        for item in data:
            if isinstance(item, dict) and 'chunk_data' in item:
                chunk_data = item['chunk_data']
                if hasattr(chunk_data, 'shape') and len(chunk_data.shape) == 1 and chunk_data.shape[0] == chunk_size:
                    # 转换为指定 dtype 并脱离原引用
                    valid_chunks.append(np.array(chunk_data, dtype=dtype))

        if valid_chunks:
            arr = np.stack(valid_chunks, axis=0)
            np.save(output_path, arr)
            num_samples = len(valid_chunks)
        else:
            # 保存空数组
            arr = np.empty((0, chunk_size), dtype=dtype)
            np.save(output_path, arr)
            num_samples = 0

        # 👇 显式删除大对象，帮助 GC
        del data, valid_chunks, arr
        return (os.path.basename(output_path), num_samples)

    except Exception as e:
        print(f"❌ Failed to process {input_path}: {e}", file=sys.stderr)
        return (os.path.basename(output_path), 0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert each .npy file to memmap-friendly format and generate shards.json (sorted by num_samples descending)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with original .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for converted .npy files")
    parser.add_argument("--chunk_size", type=int, default=12000, help="Expected chunk length (default: 12000)")
    parser.add_argument("--dtype", type=str, default="float32", help="Output dtype (default: float32)")
    args = parser.parse_args()

    # 解析 dtype
    try:
        dtype = getattr(np, args.dtype) if isinstance(args.dtype, str) else args.dtype
    except AttributeError:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    # 获取输入文件列表（排序以保证可重现）
    input_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.npy')
    ])
    if not input_files:
        raise ValueError(f"No .npy files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 构建任务列表（仅路径）
    tasks = [
        (
            os.path.join(args.input_dir, fname),
            os.path.join(args.output_dir, fname),
            args.chunk_size,
            dtype
        )
        for fname in input_files
    ]

    print(f"ParallelGrouping {len(tasks)} files in single process (with memory cleanup)...")

    results = []
    for task in tqdm(tasks, desc="Converting files"):
        result = convert_single_file(*task)
        results.append(result)
        # 👇 关键：每处理完一个文件就触发垃圾回收
        gc.collect()

    # 构建 shard_info 并按 num_samples 降序排序
    result_dict = dict(results)
    shard_info = []
    for fname in input_files:
        num_samples = result_dict.get(fname, 0)
        shard_info.append({
            "path": fname,
            "num_samples": num_samples
        })

    # 🔻 按样本数降序排列（从大到小）🔻
    shard_info.sort(key=lambda x: x["num_samples"], reverse=True)

    total_samples = sum(info["num_samples"] for info in shard_info)

    # 保存 shards.json
    meta_path = os.path.join(args.output_dir, "shards.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "total_samples": total_samples,
            "chunk_size": args.chunk_size,
            "dtype": np.dtype(dtype).name,
            "shards": shard_info
        }, f, indent=2)

    print(f"\n🎉 Conversion completed!")
    print(f"   Total files processed: {len(input_files)}")
    print(f"   Total samples:         {total_samples}")
    print(f"   Largest shard:         {shard_info[0]['num_samples']} samples")
    print(f"   Smallest shard:        {shard_info[-1]['num_samples']} samples")
    print(f"   Metadata saved to:     {meta_path}")


if __name__ == "__main__":
    main()
