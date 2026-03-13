#!/usr/bin/env python3
"""
信号分布统计生成器
为每个.npy文件生成同名的dist.csv文件，包含value-count-ratio统计信息
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json


def process_single_npy_file(args):
    """
    处理单个npy文件，生成其信号值分布统计
    
    Args:
        args: (file_path, output_dir, bins_edges)
        
    Returns:
        dict: 处理结果信息
    """
    file_path, output_dir, bin_edges = args
    
    try:
        # 加载numpy数组
        data = np.load(file_path)
        
        if data.size == 0:
            # 创建空的分布统计
            df_empty = create_empty_distribution_df(bin_edges)
            output_csv = Path(output_dir) / f"{file_path.stem}_dist.csv"
            df_empty.to_csv(output_csv, index=False)
            return {
                'file': file_path.name,
                'status': 'success',
                'total_points': 0,
                'output_file': str(output_csv)
            }
        
        # 根据 fast5_to_memap.py 的输出格式处理数据并转换为float32
        if data.ndim == 2:
            signal_values = data.flatten().astype(np.float32)
        elif data.ndim == 1:
            signal_values = data.astype(np.float32)
        else:
            raise ValueError(f"不支持的数组维度: {data.ndim}")
        
        # 检查数据类型和范围
        if not np.issubdtype(signal_values.dtype, np.floating):
            signal_values = signal_values.astype(np.float32)
        
        # 移除无效值（NaN, inf等）
        signal_values = signal_values[np.isfinite(signal_values)]
        
        if len(signal_values) == 0:
            df_empty = create_empty_distribution_df(bin_edges)
            output_csv = Path(output_dir) / f"{file_path.stem}_dist.csv"
            df_empty.to_csv(output_csv, index=False)
            return {
                'file': file_path.name,
                'status': 'success',
                'total_points': 0,
                'output_file': str(output_csv)
            }
        
        # 计算直方图
        counts, _ = np.histogram(signal_values, bins=bin_edges)
        total_count = len(signal_values)
        
        # 计算比例
        ratios = counts.astype(np.float64) / total_count  # 使用float64避免精度丢失
        
        # 创建DataFrame
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 中心值作为value
        df = pd.DataFrame({
            'value': bin_centers.astype(np.float32),  # 保持一致的数据类型
            'count': counts.astype(int),
            'ratio': ratios
        })
        
        # 保存到CSV
        output_csv = Path(output_dir) / f"{file_path.stem}_dist.csv"
        df.to_csv(output_csv, index=False)
        
        return {
            'file': file_path.name,
            'status': 'success',
            'total_points': total_count,
            'output_file': str(output_csv)
        }
        
    except Exception as e:
        error_output = Path(output_dir) / f"{file_path.stem}_dist.csv"
        df_empty = create_empty_distribution_df(bin_edges)
        df_empty.to_csv(error_output, index=False)
        return {
            'file': file_path.name,
            'status': 'error',
            'error': str(e),
            'output_file': str(error_output)
        }


def create_empty_distribution_df(bin_edges):
    """创建空的分布统计DataFrame"""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df = pd.DataFrame({
        'value': bin_centers.astype(np.float32),
        'count': np.zeros(len(bin_centers), dtype=int),
        'ratio': np.zeros(len(bin_centers), dtype=np.float64)
    })
    return df


def analyze_signal_distributions_parallel(root_dir: str, output_dir: str = "distributions", 
                                         num_workers: int = None, bin_range: tuple = (-10, 10), 
                                         num_bins: int = 2000):
    """
    并行分析目录下所有.npy文件的信号分布
    
    Args:
        root_dir: 输入目录路径
        output_dir: 输出目录路径
        num_workers: 并行工作进程数
        bin_range: 分布范围 (min, max)
        num_bins: 分箱数量
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {root_dir}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.npy文件
    npy_files = list(root.rglob("*.npy"))
    print(f"🔍 找到 {len(npy_files)} 个 .npy 文件")
    
    if not npy_files:
        print("⚠️ 未找到任何 .npy 文件")
        return
    
    # 定义bins
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1, dtype=np.float32)
    
    # 准备参数列表
    args_list = [(f, output_dir, bin_edges) for f in npy_files]
    
    # 设置并行进程数
    num_workers = num_workers or min(cpu_count(), len(npy_files))
    print(f"🚀 使用 {num_workers} 个进程并行处理")
    
    # 并行处理
    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_npy_file, args_list),
                          total=len(npy_files), desc="Processing files"):
            results.append(result)
    
    # 统计结果
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\n✅ 成功处理: {len(successful)} 个文件")
    print(f"❌ 失败处理: {len(failed)} 个文件")
    
    if successful:
        total_points = sum(r['total_points'] for r in successful)
        avg_points = total_points / len(successful) if successful else 0
        print(f"📊 总信号点数: {total_points:,}")
        print(f"📊 平均每文件: {avg_points:,.0f} 个点")
    
    if failed:
        print("\n❌ 失败的文件:")
        for f in failed:
            print(f"   {f['file']}: {f['error']}")
    
    # 保存处理摘要
    summary = {
        "total_files_found": len(npy_files),
        "successful_files": len(successful),
        "failed_files": len(failed),
        "total_points_processed": sum(r['total_points'] for r in successful),
        "bin_range": bin_range,
        "num_bins": num_bins,
        "bin_width": float((bin_range[1] - bin_range[0]) / num_bins),
        "results": results
    }
    
    summary_file = output_path / "distribution_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 处理摘要已保存至: {summary_file}")
    print(f"📁 CSV文件已保存至: {output_path.absolute()}")


def analyze_signal_distributions_sequential(root_dir: str, output_dir: str = "distributions", 
                                           bin_range: tuple = (-10, 10), num_bins: int = 2000):
    """
    顺序处理版本（当并行出现问题时的备选方案）
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {root_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(root.rglob("*.npy"))
    print(f"🔍 找到 {len(npy_files)} 个 .npy 文件")
    
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1, dtype=np.float32)
    
    results = []
    for npy_file in tqdm(npy_files, desc="Processing files"):
        try:
            # 加载数据
            data = np.load(npy_file)
            
            if data.size == 0:
                df_empty = create_empty_distribution_df(bin_edges)
                output_csv = Path(output_dir) / f"{npy_file.stem}_dist.csv"
                df_empty.to_csv(output_csv, index=False)
                results.append({
                    'file': npy_file.name,
                    'status': 'success',
                    'total_points': 0,
                    'output_file': str(output_csv)
                })
                continue
            
            # 处理数据并转换为float32
            if data.ndim == 2:
                signal_values = data.flatten().astype(np.float32)
            elif data.ndim == 1:
                signal_values = data.astype(np.float32)
            else:
                raise ValueError(f"不支持的数组维度: {data.ndim}")
            
            # 检查数据类型和范围
            if not np.issubdtype(signal_values.dtype, np.floating):
                signal_values = signal_values.astype(np.float32)
            
            # 移除无效值（NaN, inf等）
            signal_values = signal_values[np.isfinite(signal_values)]
            
            if len(signal_values) == 0:
                df_empty = create_empty_distribution_df(bin_edges)
                output_csv = Path(output_dir) / f"{npy_file.stem}_dist.csv"
                df_empty.to_csv(output_csv, index=False)
                results.append({
                    'file': npy_file.name,
                    'status': 'success',
                    'total_points': 0,
                    'output_file': str(output_csv)
                })
                continue
            
            # 计算分布
            counts, _ = np.histogram(signal_values, bins=bin_edges)
            total_count = len(signal_values)
            
            # 计算比例
            ratios = counts.astype(np.float64) / total_count  # 使用float64避免精度丢失
            
            # 创建DataFrame
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            df = pd.DataFrame({
                'value': bin_centers.astype(np.float32),
                'count': counts.astype(int),
                'ratio': ratios
            })
            
            output_csv = Path(output_dir) / f"{npy_file.stem}_dist.csv"
            df.to_csv(output_csv, index=False)
            
            results.append({
                'file': npy_file.name,
                'status': 'success',
                'total_points': total_count,
                'output_file': str(output_csv)
            })
            
        except Exception as e:
            error_output = Path(output_dir) / f"{npy_file.stem}_dist.csv"
            df_empty = create_empty_distribution_df(bin_edges)
            df_empty.to_csv(error_output, index=False)
            results.append({
                'file': npy_file.name,
                'status': 'error',
                'error': str(e),
                'output_file': str(error_output)
            })
    
    # 保存摘要
    summary = {
        "total_files_found": len(npy_files),
        "successful_files": len([r for r in results if r['status'] == 'success']),
        "failed_files": len([r for r in results if r['status'] == 'error']),
        "results": results
    }
    
    summary_file = output_path / "distribution_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 顺序处理完成")
    print(f"📋 处理摘要已保存至: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="并行分析目录下所有 .npy 文件的信号分布，生成 value-count-ratio 统计CSV文件"
    )
    parser.add_argument("root_dir", type=str, help="包含 .npy 文件的根目录（递归搜索）")
    parser.add_argument("--output", "-o", type=str, default="distributions", 
                       help="输出目录路径（默认: distributions）")
    parser.add_argument("--workers", "-w", type=int, default=None, 
                       help="并行工作进程数（默认: 自动检测CPU核心数）")
    parser.add_argument("--sequential", "-s", action='store_true', 
                       help="使用顺序处理模式（并行有问题时使用）")
    parser.add_argument("--min-val", type=float, default=-10.0, 
                       help="分布最小值（默认: -10.0）")
    parser.add_argument("--max-val", type=float, default=10.0, 
                       help="分布最大值（默认: 10.0）")
    parser.add_argument("--bins", type=int, default=2000, 
                       help="分箱数量（默认: 2000，对应0.01宽度的bins）")

    args = parser.parse_args()
    
    if args.sequential:
        analyze_signal_distributions_sequential(
            root_dir=args.root_dir,
            output_dir=args.output,
            bin_range=(args.min_val, args.max_val),
            num_bins=args.bins
        )
    else:
        analyze_signal_distributions_parallel(
            root_dir=args.root_dir,
            output_dir=args.output,
            num_workers=args.workers,
            bin_range=(args.min_val, args.max_val),
            num_bins=args.bins
        )


if __name__ == "__main__":
    main()
