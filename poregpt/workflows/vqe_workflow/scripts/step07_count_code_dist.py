import os
import re
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import argparse

# 全局配置
TOKEN_PATTERN = re.compile(r'<\|bwav:(\d+)\|>')
CODEBOOK_SIZE = 8192


def process_file(filepath):
    """处理单个 .jsonl.gz 文件，返回局部频次数组"""
    counts = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    matches = TOKEN_PATTERN.findall(text)
                    for mid_str in matches:
                        token_id = int(mid_str)
                        if 0 <= token_id < CODEBOOK_SIZE:
                            counts[token_id] += 1
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
        return np.zeros(CODEBOOK_SIZE, dtype=np.int64)
    return counts


def main(data_dir, num_workers=8, plot_path="token_freq_triple.png"):
    # 扫描文件
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl.gz')]
    print(f"📁 Found {len(files)} .jsonl.gz files in: {data_dir}")
    if not files:
        print("❌ No .jsonl.gz files found!")
        return

    # 并行统计
    total_counts = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for counts in pool.imap_unordered(process_file, files):
                total_counts += counts
                pbar.update()

    # 统计摘要
    total_tokens = total_counts.sum()
    used_tokens = np.count_nonzero(total_counts)
    max_count = np.max(total_counts)
    max_id = int(np.argmax(total_counts))

    print(f"\n✅ Total tokens: {total_tokens:,}")
    print(f"✅ Unique tokens used: {used_tokens} / {CODEBOOK_SIZE} ({100 * used_tokens / CODEBOOK_SIZE:.2f}%)")
    print(f"✅ Most frequent token: ID={max_id}, count={max_count:,}")

    # 🔍 新增：打印 Top 1, 2, 3, 9, 10 的 token 频率及占比（精确到小数点后六位）
    # 先获取按频次降序排列的 token ID 列表
    sorted_indices = np.argsort(total_counts)[::-1]  # 从高到低排序

    # 指定要打印的排名（1-based）
    target_ranks = [1, 2, 3,4,5,6,7,8, 9, 10]

    print("\n🏆 Top Token Frequencies (with percentage of total tokens):")
    print(f"{'Rank':>4} {'Token ID':>8} {'Count':>12} {'Percentage':>14}")
    print("-" * 45)

    for rank in target_ranks:
        # 转为 0-based 索引
        idx = rank - 1
        if idx < len(sorted_indices):
            token_id = sorted_indices[idx]
            count = total_counts[token_id]
            # 计算百分比，保留小数点后六位
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0.0
            print(f"{rank:>4} {token_id:>8} {count:>12,} {percentage:>13.6f}%")
        else:
            print(f"{rank:>4}        N/A          N/A            N/A")

    # 保存数据
    np.save("token_frequencies.npy", total_counts)
    print("\n💾 Token frequencies saved to: token_frequencies.npy")

    # === 三子图绘图 ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8), sharex=False)

    x_ids = np.arange(CODEBOOK_SIZE)

    # 上图：by token ID
    ax1.plot(x_ids, total_counts, linewidth=0.6, alpha=0.85, color='black')
    ax1.fill_between(x_ids, total_counts, alpha=0.3, color='steelblue')
    ax1.set_title('Token Frequency by ID (0 to 8191)', fontsize=14)
    ax1.set_ylabel('Frequency (count)', fontsize=12)
    ax1.set_xlim(0, CODEBOOK_SIZE - 1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.annotate(
        f'Max: ID={max_id}\nCount={max_count:,}',
        xy=(max_id, max_count),
        xytext=(max_id + 800, max_count * 0.7),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5)
    )

    # 中图：ranked (linear)
    sorted_counts = np.sort(total_counts)[::-1]  # descending
    x_rank = np.arange(CODEBOOK_SIZE)
    ax2.plot(x_rank, sorted_counts, linewidth=0.8, color='darkorange')
    ax2.fill_between(x_rank, sorted_counts, alpha=0.4, color='orange')
    ax2.set_title('Token Frequency Ranked (High to Low, Linear Scale)', fontsize=14)
    ax2.set_ylabel('Frequency (count)', fontsize=12)
    ax2.set_xlim(0, CODEBOOK_SIZE - 1)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 标注未使用 token 数量
    unused = CODEBOOK_SIZE - used_tokens
    if unused > 0:
        ax2.text(0.02, 0.95, f'{unused} tokens never used', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.6))

    # 下图：ranked (log scale)
    log_sorted_counts = np.log10(sorted_counts + 1)  # +1 to avoid log(0)
    ax3.plot(x_rank, log_sorted_counts, linewidth=0.8, color='purple')
    ax3.fill_between(x_rank, log_sorted_counts, alpha=0.4, color='violet')
    ax3.set_title('Token Frequency Ranked (Log₁₀ Scale)', fontsize=14)
    ax3.set_xlabel('Rank (0 = most frequent)', fontsize=12)
    ax3.set_ylabel('Log₁₀ (Frequency + 1)', fontsize=12)
    ax3.set_xlim(0, CODEBOOK_SIZE - 1)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Triple-plot saved to: {plot_path}")
