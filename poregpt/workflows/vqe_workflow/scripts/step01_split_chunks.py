#!/usr/bin/env python3

import os
import numpy as np
import glob
import shutil
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Split a directory of .npy chunk files into larger batched .npy files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .npy chunk files (each is a list/array of chunks)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged batch .npy files."
    )
    parser.add_argument(
        "--chunks_per_file",
        type=int,
        default=100000,
        help="Number of chunks to pack into each output .npy file. Default: 100000."
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=12000,
        help="Expected length of each chunk (used for filtering). Default: 12000."
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    chunks_per_file = args.chunks_per_file
    chunk_length = args.chunk_length

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有有效 chunks
    all_chunks = []
    print(f"Loading all chunks from {input_dir} (this will take memory; ensure sufficient RAM)...")
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not npy_files:
        print(f"⚠️  No .npy files found in {input_dir}")
        return

    for f in tqdm(npy_files):
        try:
            data = np.load(f, allow_pickle=True)
            # 确保 data 是 iterable（list 或 array）
            for item in data:
                if 'chunk_data' in item and item['chunk_data'].shape[0] == chunk_length:
                    all_chunks.append(item)
        except Exception as e:
            print(f"⚠️  Skipping corrupted file {f}: {e}")

    print(f"Total valid chunks (length={chunk_length}): {len(all_chunks)}")

    if not all_chunks:
        print("❌ No valid chunks found. Exiting.")
        return

    # 拆分并保存
    num_batches = 0
    for i in range(0, len(all_chunks), chunks_per_file):
        batch = all_chunks[i:i + chunks_per_file]
        out_path = os.path.join(output_dir, f"chunk_batch_{num_batches:06d}.npy")
        np.save(out_path, batch)
        num_batches += 1

    print(f"✅ Saved {num_batches} batch files to {output_dir}")

if __name__ == "__main__":
    main()
