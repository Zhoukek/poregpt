import argparse
import numpy as np
from pathlib import Path

def main():
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="Synchronously shuffle chunks, references, and lengths.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the _all.npy files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the shuffled .npy files.")
    args = parser.parse_args()

    # 2. 获取并检查路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 自动创建输出文件夹（如果不存在的话），防止 np.save 时报错
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_npy = input_dir / "chunks_all.npy"
    ref_npy = input_dir / "references_all.npy"
    ref_len = input_dir / "reference_lengths_all.npy"

    # 3. 加载数据
    print(f"Loading data from: {input_dir}")
    chunks = np.load(chunk_npy)
    references = np.load(ref_npy)
    reference_lengths = np.load(ref_len)

    # 4. 动态获取数据量 N（再也不用手动写死了！）
    N = chunks.shape[0]
    print(f"Detected {N} samples. Generating permutation...")

    # 安全检查：确保三个文件的样本数量完全一致
    assert references.shape[0] == N, "Error: Mismatch in chunks and references count!"
    assert reference_lengths.shape[0] == N, "Error: Mismatch in chunks and reference_lengths count!"

    # 5. 生成随机索引并同步打乱
    perm = np.random.permutation(N)

    print("Shuffling arrays...")
    chunks = chunks[perm]
    references = references[perm]
    reference_lengths = reference_lengths[perm]

    # 6. 保存打乱后的数据
    out_chunk = output_dir / "chunks.npy"
    out_ref = output_dir / "references.npy"
    out_len = output_dir / "reference_lengths.npy"

    print(f"Saving shuffled data to: {output_dir}")
    np.save(out_chunk, chunks)
    np.save(out_ref, references)
    np.save(out_len, reference_lengths)

    print("Shuffle and save complete!")

if __name__ == "__main__":
    main()