import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_distribution_summary(summary_json_path):
    """加载分布统计摘要文件"""
    with open(summary_json_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return summary

def load_and_aggregate_csv_files(csv_files, bin_centers):
    """加载所有CSV文件并聚合分布数据"""
    print(f"📊 开始聚合 {len(csv_files)} 个CSV文件...")

    total_counts = np.zeros(len(bin_centers), dtype=np.int64)
    total_points_processed = 0

    for csv_file in tqdm(csv_files, desc="加载CSV文件"):
        try:
            df = pd.read_csv(csv_file)

            if not all(col in df.columns for col in ['value', 'count', 'ratio']):
                print(f"⚠️ 警告: {csv_file} 缺少必要列，跳过")
                continue

            if len(df) != len(bin_centers):
                print(f"⚠️ 警告: {csv_file} 行数不匹配，跳过")
                continue

            if not np.allclose(df['value'].values, bin_centers, rtol=1e-5):
                print(f"⚠️ 警告: {csv_file} value列不匹配，跳过")
                continue

            current_counts = df['count'].values.astype(np.int64)
            total_counts += current_counts
            total_points_processed += current_counts.sum()

        except Exception as e:
            print(f"⚠️ 警告: 无法读取 {csv_file}, 错误: {e}")
            continue

    if total_points_processed > 0:
        total_ratios = total_counts.astype(np.float64) / total_points_processed
    else:
        print("❌ 错误: 没有有效的数据点")
        return None, 0

    print(f"📈 聚合完成，成功处理了 {total_points_processed:,} 个数据点")
    return total_ratios, total_points_processed

def plot_ratio_distributions_old(bin_centers, total_ratios, total_points, input_dir, bin_width):
    """绘制比例分布图及其log10版本"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 获取输入目录名（去掉路径分隔符）
    input_dir_name = input_dir 
    
    # 绘制常规比例分布图
    ax1.plot(bin_centers, total_ratios, linewidth=1.0, alpha=0.8, color='blue', label='Signal Ratio Distribution')
    ax1.set_ylabel('Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Signal Ratio Distribution\nInput Directory: {input_dir_name}\nTotal Points: {total_points:,}')
    
    # 绘制log10版本的比例分布图
    ax2.plot(bin_centers, total_ratios, linewidth=1.0, alpha=0.8, color='orange', label='Log-Scale Signal Ratio Distribution')
    ax2.set_xlabel('Signal Value')
    ax2.set_ylabel('Log(Ratio)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(f'Log-Scale Signal Ratio Distribution\nInput Directory: {input_dir_name}\nTotal Points: {total_points:,}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    output_path = Path(input_dir) / 'combined_signal_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 分布图已保存至: {output_path}")
    plt.show()

from scipy.stats import norm

# 在plot_ratio_distributions函数中进行修改
def plot_ratio_distributions(bin_centers, total_ratios, total_points, input_dir, bin_width):
    """绘制比例分布图及其log10版本"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # 获取输入目录名（去掉路径分隔符）
    input_dir_name = input_dir

    # 绘制常规比例分布图
    ax1.plot(bin_centers, total_ratios, linewidth=1.0, alpha=0.8, color='blue', label='Signal Ratio Distribution')
    
    # 添加标准正态分布曲线
    normal_x = np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), 1000)
    normal_y = norm.pdf(normal_x)
    # 缩放正态分布曲线以适应数据范围
    scale_factor = max(total_ratios) / max(normal_y)
    normal_y_scaled = normal_y * scale_factor
    ax1.plot(normal_x, normal_y_scaled, linewidth=1.0, alpha=0.8, color='red', linestyle='--', label='Standard Normal Distribution (scaled)')
    
    ax1.set_ylabel('Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Signal Ratio Distribution\nInput Directory: {input_dir_name}\nTotal Points: {total_points:,}')

    # 绘制log10版本的比例分布图
    ax2.plot(bin_centers, total_ratios, linewidth=1.0, alpha=0.8, color='orange', label='Log-Scale Signal Ratio Distribution')
    ax2.set_xlabel('Signal Value')
    ax2.set_ylabel('Log(Ratio)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(f'Log-Scale Signal Ratio Distribution\nInput Directory: {input_dir_name}\nTotal Points: {total_points:,}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    output_path = Path(input_dir) / 'combined_signal_distribution_with_normal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 分布图已保存至: {output_path}")
    plt.show()

def main(input_dir):
    input_dir = Path(input_dir)

    # 寻找distribution_summary.json文件
    summary_json_path = input_dir / "distribution_summary.json"
    if not summary_json_path.exists():
        print(f"❌ 错误: 在目录 {input_dir} 中找不到 distribution_summary.json 文件")
        return

    # 加载摘要文件
    print("📂 加载分布统计摘要...")
    summary = load_distribution_summary(summary_json_path)

    # 获取CSV文件列表
    csv_files = list(input_dir.glob("*_dist.csv"))
    print(f"🔍 找到 {len(csv_files)} 个CSV分布文件")

    if not csv_files:
        print("❌ 未找到任何_dist.csv文件")
        return

    # 获取bin信息
    bin_range = summary['bin_range']
    num_bins = summary['num_bins']
    bin_width = summary['bin_width']

    # 计算bin中心值
    bin_centers = np.linspace(bin_range[0], bin_range[1], num_bins + 1, dtype=np.float32)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2  # 中心值

    # 聚合所有CSV文件的数据
    total_ratios, total_points = load_and_aggregate_csv_files(csv_files, bin_centers)

    if total_ratios is None:
        print("❌ 数据加载失败，程序退出")
        return

    # 使用新函数绘图
    plot_ratio_distributions(bin_centers, total_ratios, total_points, input_dir, bin_width)

    # 保存聚合数据
    agg_data = pd.DataFrame({
        'value': bin_centers,
        'total_ratio': total_ratios
    })

    agg_csv_path = input_dir / 'aggregated_signal_distribution.csv'
    agg_data.to_csv(agg_csv_path, index=False)
    print(f"💾 聚合数据已保存至: {agg_csv_path}")

    print("🎉 信号分布分析完成!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python script.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    main(input_directory)
