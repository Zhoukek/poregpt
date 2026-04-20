import argparse
import numpy as np
from pathlib import Path
import pandas as pd

def merge_npy(files):
    arrays = []
    for f in files:
        if not f.exists():
            print(f"Warning: File not found: {f}")
            continue
        arrays.append(np.load(f))
    
    if not arrays:
        raise ValueError("No arrays found to merge. Please check your paths and CSV.")
    
    return np.concatenate(arrays, axis=0)

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Merge NPY files based on processing_summary.csv")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the .npy files and where merged files will be saved.")
    parser.add_argument("--summary_csv", type=str, required=True, 
                        help="Path to the processing_summary.csv file.")
    
    args = parser.parse_args()

    # 2. 获取路径
    data_dir = Path(args.data_dir)
    summary_csv = Path(args.summary_csv)

    print(f"Data directory: {data_dir}")
    print(f"Summary CSV: {summary_csv}")

    # 3. 读取 CSV 获取文件列表
    summary_data = pd.read_csv(summary_csv)
    file_list = [str(i).split(".")[0] for i in summary_data['fast5_name']]
    print(f"Found {len(file_list)} files in summary CSV.")

    # 4. 动态构建待合并的文件路径列表
    chunk_files = [data_dir / f"{i}_chunks.npy" for i in file_list]
    ref_files = [data_dir / f"{i}_references.npy" for i in file_list]
    len_files = [data_dir / f"{i}_reference_lengths.npy" for i in file_list]

    # 5. 合并
    print("Merging chunks...")
    chunks = merge_npy(chunk_files)
    
    print("Merging references...")
    references = merge_npy(ref_files)
    
    print("Merging reference lengths...")
    reference_lengths = merge_npy(len_files)

    print("\nMerged shapes:")
    print("chunks:", chunks.shape)
    print("references:", references.shape)
    print("reference_lengths:", reference_lengths.shape)

    # 6. 保存
    np.save(data_dir / "chunks_all.npy", chunks)
    np.save(data_dir / "references_all.npy", references)
    np.save(data_dir / "reference_lengths_all.npy", reference_lengths)

    print(f"\nMerge finished! Files saved in: {data_dir}")

if __name__ == "__main__":
    main()