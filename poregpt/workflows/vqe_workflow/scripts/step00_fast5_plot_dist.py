# scripts/step00_fast5_plot_dist.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
import warnings
from multiprocessing import Pool, cpu_count
import random

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def extract_signal_from_read(read):
    try:
        channel_info = read.handle[read.global_key + 'channel_id'].attrs
        offset = float(channel_info['offset'])
        digitisation = float(channel_info['digitisation'])
        range_val = float(channel_info['range'])
        scaling = range_val / digitisation
        raw = read.handle[read.raw_dataset_name][:]
        # ❗ 注意：这里保留你原代码的 + offset（虽然物理上应为 -，但保持一致）
        signal_pA = scaling * (raw.astype(np.float32) + offset)
        return signal_pA
    except Exception as e:
        return None


def process_single_fast5(args):
    """
    处理单个 .fast5 文件，返回采样信号和极端 read 信息。
    Args:
        args: (f5_path_str, max_points_per_file)
    Returns:
        dict: {
            'points': list of floats (sampled),
            'extreme_reads': list of dicts,
            'read_count': int,
            'total_points': int
        }
    """
    f5_path_str, max_points_per_file = args
    f5_path = Path(f5_path_str)
    sampled_points = []
    extreme_reads = []
    read_count = 0
    total_points = 0

    try:
        with get_fast5_file(f5_path, mode="r") as f5:
            reads = list(f5.get_reads())
            for read in reads:
                signal = extract_signal_from_read(read)
                if signal is None or signal.size == 0:
                    continue

                read_count += 1
                n = len(signal)
                total_points += n

                local_min, local_max = signal.min(), signal.max()
                if local_max > 1e5 or local_min < -1e5:
                    extreme_reads.append({
                        'file': str(f5_path),
                        'read_id': read.read_id,
                        'min': float(local_min),
                        'max': float(local_max)
                    })

                if len(sampled_points) < max_points_per_file:
                    needed = max_points_per_file - len(sampled_points)
                    if needed >= n:
                        sampled_points.extend(signal.tolist())
                    else:
                        idxs = np.random.choice(n, size=needed, replace=False)
                        sampled_points.extend(signal[idxs].tolist())

    except Exception as e:
        print(f"❌ Error in worker processing {f5_path}: {e}")
        return {
            'points': [],
            'extreme_reads': [],
            'read_count': 0,
            'total_points': 0
        }

    return {
        'points': sampled_points,
        'extreme_reads': extreme_reads,
        'read_count': read_count,
        'total_points': total_points
    }


def analyze_fast5_directory_parallel(root_dir: str, output_hist: str = "fast5_signal_hist.png",
                                    max_total_points: int = 5_000_000, n_jobs: int = -1):
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Directory not found: {root_dir}")

    fast5_files = list(root.rglob("*.fast5"))
    if not fast5_files:
        raise FileNotFoundError(f"No .fast5 files found in {root_dir}")

    print(f"🔍 Found {len(fast5_files)} .fast5 files under {root_dir}")

    # 随机取最多 200 个文件（如果总数不足 200，则取全部）
    sample_size = min(100, len(fast5_files))
    random_fast5_files = random.sample(fast5_files, sample_size)
    print(f"🎲 Randomly selected {sample_size} files for analysis.")

    if n_jobs == -1:
        n_jobs = cpu_count()

    max_per_file = max(1000, max_total_points // max(1, sample_size // 10 + 1))
    args_list = [(str(f), max_per_file) for f in random_fast5_files]

    print(f"ParallelGroup: using {n_jobs} processes")
    print(f"Sampling up to ~{max_total_points:,} total points (max {max_per_file:,} per file)")

    all_values = []
    total_reads = 0
    total_points = 0
    all_extreme_reads = []

    with Pool(processes=n_jobs) as pool:
        results = pool.imap_unordered(process_single_fast5, args_list)
        for i, res in enumerate(results, 1):
            all_values.extend(res['points'])
            total_reads += res['read_count']
            total_points += res['total_points']
            all_extreme_reads.extend(res['extreme_reads'])

            if len(all_values) > max_total_points:
                all_values = all_values[:max_total_points]

            if i % 10 == 0 or i == len(random_fast5_files):
                print(f"  Processed {i}/{len(random_fast5_files)} sampled files | "
                      f"reads: {total_reads}, points: {total_points:,}, sampled: {len(all_values):,}")

    if not all_values:
        raise RuntimeError("No valid signal data extracted!")

    all_values = np.array(all_values, dtype=np.float32)
    global_min, global_max = all_values.min(), all_values.max()
    mean_val, std_val = all_values.mean(), all_values.std()

    print(f"\n✅ Summary (sampled {len(all_values):,} points from {total_reads} reads):")
    print(f"   Min: {global_min:.3f} pA")
    print(f"   Max: {global_max:.3f} pA")
    print(f"   Mean: {mean_val:.3f} pA")
    print(f"   Std: {std_val:.3f} pA")

    if all_extreme_reads:
        print(f"\n❗ {len(all_extreme_reads)} reads have extreme values (>100k or <-100k pA):")
        for er in all_extreme_reads[:5]:
            print(f"   File: {er['file']}, Read: {er['read_id']}, Min: {er['min']:.1f}, Max: {er['max']:.1f}")
        if len(all_extreme_reads) > 5:
            print(f"   ... and {len(all_extreme_reads) - 5} more.")

    # === 绘图：双子图（线性 + log10 x/y）===
    global_min, global_max = all_values.min(), all_values.max()
    mean_val, std_val = all_values.mean(), all_values.std()

    # 计算 ±5σ 范围外的比例
    lower_5sigma = mean_val - 5 * std_val
    upper_5sigma = mean_val + 5 * std_val
    outside_5sigma = np.sum((all_values < lower_5sigma) | (all_values > upper_5sigma))
    ratio_outside_5sigma = outside_5sigma / len(all_values)

    # 构造统计文本
    stats_text = (
        f"Min: {global_min:.3f} pA\n"
        f"Max: {global_max:.3f} pA\n"
        f"Mean: {mean_val:.3f} pA\n"
        f"Std: {std_val:.3f} pA\n"
        f"Outliers (>±5σ): {ratio_outside_5sigma:.6f}\n"
        f"  ({outside_5sigma:,} / {len(all_values):,})"
    )

    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- 上图：线性尺度 ---
    lower_clip, upper_clip = np.percentile(all_values, [0.1, 99.9])
    clipped_linear = all_values[(all_values >= lower_clip) & (all_values <= upper_clip)]
    ax1.hist(clipped_linear, bins=200, color='teal', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax1.set_title("Distribution of Nanopore Raw Signal (pA) — Linear Scale (0.1%–99.9%)", fontsize=14)
    ax1.set_xlabel("Current (pA)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 添加统计文本框到上图右上角
    ax1.text(
        0.98, 0.98, stats_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8)
    )

    # --- 下图：X = log10(signal), Y = log10(frequency) ---
    # --- 下图：X = log10(signal), Y = log10(frequency) ---
    positive_vals = all_values[all_values > 0]
    if len(positive_vals) > 0:
        log_vals = np.log10(positive_vals)
        # 直接绘制所有正信号的对数，不进行裁剪
        ax2.hist(log_vals, bins=200, color='steelblue', alpha=0.8,
                 edgecolor='black', linewidth=0.3, log=True)
        ax2.set_xlabel("Log₁₀(Current (pA))", fontsize=12)
        ax2.set_ylabel("Log₁₀(Frequency)", fontsize=12)
    else:
        ax2.text(0.5, 0.5, "No positive signal values for log scale", 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel("Log₁₀(Current (pA))", fontsize=12)
        ax2.set_ylabel("Log₁₀(Frequency)", fontsize=12)
    ax2.set_title("Distribution of Nanopore Raw Signal (pA) — Log₁₀ X and Y Axes", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_hist, dpi=200, bbox_inches='tight')
    print(f"\n🖼️ Histogram saved to: {os.path.abspath(output_hist)}")

def main():
    parser = argparse.ArgumentParser(description="Parallel analysis of signal range in recursive .fast5 files")
    parser.add_argument("fast5_root", help="Root directory containing .fast5 files (recursive)")
    parser.add_argument("--output", "-o", default="fast5_signal_hist_parallel.png", help="Output histogram path")
    parser.add_argument("--max-points", type=int, default=5_000_000, help="Max total signal points to sample")
    parser.add_argument("--jobs", "-j", type=int, default=-1, help="Number of parallel jobs (-1 = all CPUs)")
    args = parser.parse_args()

    analyze_fast5_directory_parallel(
        root_dir=args.fast5_root,
        output_hist=args.output,
        max_total_points=args.max_points,
        n_jobs=args.jobs
    )


if __name__ == "__main__":
    main()
