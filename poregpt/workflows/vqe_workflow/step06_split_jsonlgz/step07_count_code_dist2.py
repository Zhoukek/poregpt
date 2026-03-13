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

    # 找最小非零频次
    nonzero_mask = total_counts > 0
    if used_tokens > 0:
        min_count = np.min(total_counts[nonzero_mask])
        min_id = int(np.where(total_counts == min_count)[0][0])
        sorted_indices = np.argsort(-total_counts)
        rank_of_min = int(np.where(sorted_indices == min_id)[0][0])
    else:
        min_count = 0
        min_id = -1
        rank_of_min = -1

    unused = CODEBOOK_SIZE - used_tokens  # ← 保留 dead code 统计！

    print(f"\n✅ Total tokens: {total_tokens:,}")
    print(f"✅ Unique tokens used: {used_tokens} / {CODEBOOK_SIZE} ({100 * used_tokens / CODEBOOK_SIZE:.2f}%)")
    print(f"✅ Most frequent token: ID={max_id}, count={max_count:,}")
    if used_tokens > 0:
        print(f"✅ Least frequent (non-zero) token: ID={min_id}, count={min_count:,}, rank={rank_of_min}")
        # ======================================================================
    # ✅ 新增需求：打印 Top 1, 2, 3, 9, 10 的 token 频次及占比（6位小数）
    # 目标：不修改原有逻辑，仅追加输出
    # ======================================================================
    if total_tokens > 0:
        # 获取按频次降序排列的 token ID（从高到低）
        sorted_indices = np.argsort(-total_counts)  # 等价于 argsort()[::-1]，但更高效

        target_ranks = [1, 2, 3, 4,5,6,7,8,9, 10]
        print("\n🏆 Top Token Frequencies (Rank, Token ID, Count, Percentage):")
        print(f"{'Rank':>4} {'Token ID':>8} {'Count':>12} {'Percentage':>14}")
        print("-" * 48)

        for rank in target_ranks:
            idx = rank - 1  # 转为 0-based 索引
            token_id = sorted_indices[idx]
            count = total_counts[token_id]
            percentage = (count / total_tokens) * 100  # 转为百分比
            # 格式化输出：count 加千分位逗号，percentage 保留6位小数
            print(f"{rank:>4} {token_id:>8} {count:>12,} {percentage:>13.6f}%")
    else:
        print("\n⚠️  No tokens found — skipping top-k frequency report.")
    # ======================================================================
    np.save("token_frequencies.npy", total_counts)
    print("💾 Token frequencies saved to: token_frequencies.npy")

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

    # 标注最大值
    ax1.annotate(
        f'Max: ID={max_id}\nCount={max_count:,}',
        xy=(max_id, max_count),
        xytext=(max_id + 800, max_count * 0.7),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5)
    )

    # ✅ 新增：标注最小非零值（仅当存在）
    if used_tokens > 0 and min_id != max_id:
        ax1.annotate(
            f'Min (non-zero):\nID={min_id}\nCount={min_count:,}',
            xy=(min_id, min_count),
            xytext=(min_id - 800 if min_id > 4096 else min_id + 800, min_count * 2 + 10),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
            fontsize=9,
            color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
        )

    # 准备 ranked 数据
    sorted_counts = np.sort(total_counts)[::-1]
    x_rank = np.arange(CODEBOOK_SIZE)
    cumsum_counts = np.cumsum(sorted_counts)
    total_sum = total_tokens if total_tokens > 0 else 1

    # ✅ 中图：竖线按 x 轴固定百分比位置（5%, 10%, ..., 90%）
    percentiles_x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    vertical_lines = []
    for p_x in percentiles_x:
        idx = int(p_x * CODEBOOK_SIZE)  # 固定 x 位置
        if idx >= CODEBOOK_SIZE:
            idx = CODEBOOK_SIZE - 1
        cum_ratio = cumsum_counts[idx] / total_sum if total_sum > 0 else 0
        vertical_lines.append((idx, cum_ratio))

    # 中图：ranked (linear)
    ax2.plot(x_rank, sorted_counts, linewidth=0.8, color='darkorange')
    ax2.fill_between(x_rank, sorted_counts, alpha=0.4, color='orange')
    ax2.set_title('Token Frequency Ranked (High to Low, Linear Scale)', fontsize=14)
    ax2.set_ylabel('Frequency (count)', fontsize=12)
    ax2.set_xlim(0, CODEBOOK_SIZE - 1)
    ax2.grid(True, linestyle='--', alpha=0.5)

    if unused > 0:
        ax2.text(0.02, 0.75, f'{unused} tokens never used', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.6))

    # ✅ 中图：在固定 x 百分位处画竖线，并横排标注累计比例（无 "cum="）
    y_max = sorted_counts.max() if sorted_counts.max() > 0 else 1
    for idx, cum_ratio in vertical_lines:
        ax2.axvline(x=idx, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        label = f'{cum_ratio:.1%}'  # ← 去掉 "cum="
        # 横排文字：放在竖线上方，居中或略偏右
        ax2.text(idx, y_max * 0.95, label,
                 rotation=0,  # ← 横着写
                 verticalalignment='top',
                 horizontalalignment='center',  # 居中对齐竖线
                 fontsize=8,
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # ✅ 新增：右上角显示前 1% 和前 5% 的累计比例
    top1_idx = int(0.01 * CODEBOOK_SIZE)
    top5_idx = int(0.05 * CODEBOOK_SIZE)
    cum_top1 = cumsum_counts[top1_idx] / total_sum if total_sum > 0 else 0
    cum_top5 = cumsum_counts[top5_idx] / total_sum if total_sum > 0 else 0
    summary_text = f"Top 1% tokens: {cum_top1:.1%}\nTop 5% tokens: {cum_top5:.1%}"
    ax2.text(0.98, 0.55, summary_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    # 标注最小非零（中图）
    if used_tokens > 0:
        ax2.annotate(
            f'Min (non-zero):\nID={min_id}\nCount={min_count:,}',
            xy=(rank_of_min, min_count),
            xytext=(rank_of_min - 300, min_count * 5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
            fontsize=9,
            color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
        )

    # 下图：log scale —— ✅ 移除所有竖线！
    log_sorted_counts = np.log10(sorted_counts + 1)
    ax3.plot(x_rank, log_sorted_counts, linewidth=0.8, color='purple')
    ax3.fill_between(x_rank, log_sorted_counts, alpha=0.4, color='violet')
    ax3.set_title('Token Frequency Ranked (Log₁₀ Scale)', fontsize=14)
    ax3.set_xlabel('Rank (0 = most frequent)', fontsize=12)
    ax3.set_ylabel('Log₁₀ (Frequency + 1)', fontsize=12)
    ax3.set_xlim(0, CODEBOOK_SIZE - 1)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # ✅ 不再绘制任何竖线或标签

    # 标注最小非零（log 图）
    if used_tokens > 0:
        log_min_count = np.log10(min_count + 1)
        ax3.annotate(
            f'Min: {min_count:,}',
            xy=(rank_of_min, log_min_count),
            xytext=(rank_of_min - 300, log_min_count + 0.8),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
            fontsize=9,
            color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
        )

    # === 在下图 (ax3) 上叠加 Zipf 理论分布（log10 scale）===
    # === 在下图 (ax3) 上叠加 Zipf 理论分布（log10 scale）===
    if total_tokens > 0 and used_tokens > 0:
        # 排名从 1 开始（Zipf 定律要求 r >= 1）
        ranks = np.arange(1, CODEBOOK_SIZE + 1)  # shape: (8192,)
        
        # 使用 s=1 的标准 Zipf 分布
        s = 1.0
        harmonic_sum = np.sum(1.0 / (ranks ** s))
        C = total_tokens / harmonic_sum  # 归一化常数，使总和 ≈ total_tokens
        
        # 理论频次（按排名顺序，与 sorted_counts 对齐）
        zipf_freq = C / (ranks ** s)  # 第1名对应 rank=1
        
        # 转换为 log10(freq + 1)，与实际数据一致
        log_zipf_freq = np.log10(zipf_freq + 1)
        
        ax3.plot(x_rank, log_zipf_freq, 
                 linewidth=1.2, 
                 linestyle='--', 
                 color='red', 
                 alpha=0.8,
                 label='Zipf (s=1) theoretical')
        
        # 添加图例
        ax3.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)

        # ✅ 新增：在右上角显示 C 值
        ax3.text(0.98, 0.7, f'C = {C:.2e}', 
                 transform=ax3.transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Triple-plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token frequency analysis with fixed-percentile lines and min/max annotations.")
    parser.add_argument("data_dir", type=str, help="Directory containing .jsonl.gz files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="token_freq_triple.png", help="Output plot filename")
    args = parser.parse_args()
    main(args.data_dir, num_workers=args.workers, plot_path=args.output)
