#!/usr/bin/env python3
"""
将 fast5 文件直接转换为 memmap 友好的 .npy 格式，并生成 shards.json。
整合了 fast5_to_trank.py 和 step02_memmap_chunks.py 的功能。
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from poregpt.utils.signal import nanopore_process_signal


def process_single_fast5_and_save(args):
    """
    处理单个 fast5 文件，将其信号数据切分并直接保存为 memmap 友好的 .npy 格式。

    Args:
        args (tuple): 包含以下参数的元组
            - fast5_path (str): 输入 fast5 文件路径
            - output_dir (str): 输出目录路径
            - nanopore_signal_process_strategy (str): 信号处理策略
            - signal_chunk_size (int): 信号块大小
            - signal_chunk_overlap_size (int): 重叠大小
            - expected_chunk_size (int): 期望的块大小（用于过滤）
            - dtype (numpy.dtype): 输出数据类型

    Returns:
        dict: 包含 shard 信息的字典 {'path': filename, 'num_samples': count}
    """
    (fast5_path, output_dir, nanopore_signal_process_strategy, 
     signal_chunk_size, signal_chunk_overlap_size, expected_chunk_size, dtype,clip_value) = args

    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 存储处理后的数据块
        processed_chunks = []

        # 读取 fast5 文件
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in f5.get_reads():
                try:
                    # 从 fast5 文件中提取原始信号数据
                    channel_info = read.handle[read.global_key + 'channel_id'].attrs
                    offset = int(channel_info['offset'])
                    scaling = channel_info['range'] / channel_info['digitisation']
                    raw = read.handle[read.raw_dataset_name][:]
                    signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)

                    # 应用信号处理策略
                    signal_processed = nanopore_process_signal(signal_raw, nanopore_signal_process_strategy)

                    # 将处理后的信号按指定大小切分
                    L = len(signal_processed)
                    step_size = signal_chunk_size - signal_chunk_overlap_size

                    # 确保信号长度足够处理
                    if L >= signal_chunk_size:
                        start = 0
                        chunk_idx = 0

                        # 按步长滑动窗口切分信号
                        while start + signal_chunk_size <= L:
                            chunk = signal_processed[start : start + signal_chunk_size]
                            start += step_size
                            # 检查是否存在小于-3或大于3的元素
                            if clip_value > 0.000001 and np.any((chunk < 0 - clip_value) | (chunk > 0 + clip_value)):
                                continue
                            # 将当前块信息添加到列表中
                            processed_chunks.append({
                                'read_id': read.read_id,
                                'chunk_idx': chunk_idx,
                                'chunk_data': chunk  # 注意这里改为 'chunk_data'
                            })
                            chunk_idx += 1
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id} in {fast5_path}: {e}")
                    continue

        # 生成输出文件名（将 fast5 扩展名替换为 npy）
        base_name = Path(fast5_path).stem
        output_file = os.path.join(output_dir, f"{base_name}.npy")

        # 过滤并保存数据
        valid_chunks = []
        for item in processed_chunks:
            # 检查是否有 'chunk_data' 键且形状正确
            if 'chunk_data' in item and item['chunk_data'].shape[0] == expected_chunk_size:
                valid_chunks.append(item['chunk_data'].astype(dtype))

        if not valid_chunks:
            # 如果没有有效块，创建一个空数组，但保持正确的形状
            arr = np.empty((0, expected_chunk_size), dtype=dtype)
        else:
            arr = np.stack(valid_chunks, axis=0)

        np.save(output_file, arr)

        # 返回 shard 信息
        num_samples = len(valid_chunks)
        return {
            'path': f"{base_name}.npy",
            'num_samples': num_samples
        }

    except Exception as e:
        error_msg = f"❌ Error processing {fast5_path}: {str(e)}"
        print(error_msg, file=sys.stderr)
        # 即使出错也返回一个有效的 shard 信息（0 个样本）
        base_name = Path(fast5_path).stem
        return {
            'path': f"{base_name}.npy",
            'num_samples': 0
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert fast5 files directly to memmap-friendly .npy format and generate shards.json"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing fast5 files (searches recursively)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for converted .npy files")
    parser.add_argument("--strategy", type=str, default='apple', choices=['apple', 'stone', 'lemon','tango','mongo'],
                        help="Nanopore signal processing strategy (default: apple)")
    parser.add_argument("--chunk_size", type=int, default=40000, help="Size of each signal chunk (default: 40000)")
    parser.add_argument("--overlap_size", type=int, default=10000, help="Overlap size between chunks (default: 10000)")
    parser.add_argument("--dtype", type=str, default="float32", help="Output dtype (default: float32)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: auto)")
    parser.add_argument("--clip_value", type=int, default=30, help="clip value")



    args = parser.parse_args()
    print(args.clip_value)

    clip_value = args.clip_value/10

    # 解析 dtype
    try:
        dtype = getattr(np, args.dtype) if isinstance(args.dtype, str) else args.dtype
    except AttributeError:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    # 查找所有 fast5 文件（包括 .fast5 和 .fasta5 扩展名）
    input_path = Path(args.input_dir)
    fast5_files = list(input_path.rglob("*.fast5")) + list(input_path.rglob("*.fasta5"))

    if not fast5_files:
        print(f"⚠️ No fast5 files found in {args.input_dir}")
        return

    print(f"Found {len(fast5_files)} fast5 files")
    num_workers = args.num_workers or min(cpu_count(), len(fast5_files))
    print(f"Using {num_workers} processes with {args.strategy}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 准备参数列表用于多进程处理
    fast5_args_list = [
        (str(fast5_file), args.output_dir, args.strategy, args.chunk_size, 
         args.overlap_size, args.chunk_size, dtype,clip_value)
        for fast5_file in fast5_files
    ]

    # 并行处理所有 fast5 文件，直接保存并返回 shard 信息
    print("\n🔍 Processing fast5 files and saving to memmap format...")
    with Pool(processes=num_workers) as pool:
        shard_info = list(tqdm(
            pool.imap_unordered(process_single_fast5_and_save, fast5_args_list),
            total=len(fast5_args_list),
            desc="Processing and saving"
        ))

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
    print(f"   Total files processed: {len(fast5_files)}")
    print(f"   Total samples:         {total_samples}")
    print(f"   Largest shard:         {shard_info[0]['num_samples']} samples")
    print(f"   Smallest shard:        {shard_info[-1]['num_samples']} samples")
    print(f"   Output directory:      {args.output_dir}")
    print(f"   Metadata saved to:     {meta_path}")


if __name__ == "__main__":
    main()
