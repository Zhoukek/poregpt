import os
import gzip
import json
import re
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import tqdm
def count_tokens_in_text(text):
    return len(re.findall(r"<\|bwav:[^|>]+\|>", text))

def process_file(filepath):
    """处理单个 .jsonl.gz 文件，返回该文件中所有样本的 token 数列表"""
    counts = []
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if text:
                        cnt = count_tokens_in_text(text)
                        counts.append(cnt)
                except Exception as e:
                    # 可选：记录错误，但不中断
                    pass
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
    return counts

def main(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl.gz')]
    files.sort()
    print(f"Found {len(files)} .jsonl.gz files. Using {cpu_count()} CPU cores.")

    # 并行处理
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, files)

    # 合并所有结果
    token_counts = []
    for res in results:
        token_counts.extend(res)

    print(f"Total samples processed: {len(token_counts)}")

    if not token_counts:
        print("No valid samples found!")
        return

    # 绘图（y轴使用对数刻度）
    plt.figure(figsize=(10, 6))
    max_val = max(token_counts)
    bins = range(0, max_val + 2)  # 每个整数一个 bin
    plt.hist(token_counts, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of <|bwav:...|> Token Counts per Sample (Log Scale)')
    plt.xlabel('Number of <|bwav:...|> Tokens')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')  # 👈 关键修改：启用 y 轴对数刻度
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('token_distribution_log10.png', dpi=150)  # 文件名也改为 _log
    plt.show()

    # 打印统计信息
    arr = np.array(token_counts)
    print(f"Min: {arr.min()}, Max: {arr.max()}")
    print(f"Mean: {arr.mean():.2f}, Median: {np.median(arr):.2f}")
    print(f"95th percentile: {np.percentile(arr, 95):.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_token_dist_mp.py <data_dir>")
        sys.exit(1)
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory.")
        sys.exit(1)
    main(data_dir)
