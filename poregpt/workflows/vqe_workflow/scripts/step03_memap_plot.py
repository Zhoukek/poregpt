import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

# 假设 dataset 模块在上级目录或已加入 PYTHONPATH
# 如果你在项目根目录运行，且 dataset.py 在当前目录，可以这样导入：
from poregpt.tokenizers.vqe_tokenizer import NanoporeSignalDataset


def analyze_data_distribution(npy_dir: str, batch_size: int = 16, num_workers: int = 4, output_plot: str = "data_distribution.png"):
    """
    遍历整个 Nanopore 信号数据集，统计每个 chunk 的 min/max/mean/std，并绘制分布图。
    
    Args:
        npy_dir (str): 包含 .npy 文件的目录路径
        batch_size (int): DataLoader 的批大小
        num_workers (int): DataLoader 的工作进程数
        output_plot (str): 输出图像文件名
    """
    if not os.path.isdir(npy_dir):
        raise ValueError(f"指定的目录不存在: {npy_dir}")

    print(f"📁 加载数据集: {npy_dir}")
    dataset = NanoporeSignalDataset(shards_dir=npy_dir)
    print(f"📊 总共 {len(dataset)} 个 chunks")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    all_min = []
    all_max = []
    all_mean = []
    all_std = []

    print("🔄 开始遍历数据集并收集统计信息...")
    for idx, batch in enumerate(dataloader):
        # batch shape: [B, 1, T] → 转为 numpy 并去掉 channel 维度
        signals = batch.squeeze(1).numpy()  # [B, T]

        # 计算每个样本的统计量（沿时间维度）
        chunk_mins = np.min(signals, axis=1)   # [B,]
        chunk_maxs = np.max(signals, axis=1)
        chunk_means = np.mean(signals, axis=1)
        chunk_stds = np.std(signals, axis=1)

        all_min.extend(chunk_mins.tolist())
        all_max.extend(chunk_maxs.tolist())
        all_mean.extend(chunk_means.tolist())
        all_std.extend(chunk_stds.tolist())

        if (idx + 1) % 20 == 0:
            print(f"  已处理 {idx + 1}/{len(dataloader)} batches")

    print("✅ 数据遍历完成，开始绘图...")

    # 转为 numpy 数组便于分析
    all_min = np.array(all_min)
    all_max = np.array(all_max)
    all_mean = np.array(all_mean)
    all_std = np.array(all_std)

    # 打印全局统计摘要
    print("\n📈 全局统计摘要（基于所有 chunks）:")
    print(f"  Min:  [{all_min.min():.3f}, {all_min.max():.3f}] | mean={all_min.mean():.3f}")
    print(f"  Max:  [{all_max.min():.3f}, {all_max.max():.3f}] | mean={all_max.mean():.3f}")
    print(f"  Mean: [{all_mean.min():.3f}, {all_mean.max():.3f}] | mean={all_mean.mean():.3f}")
    print(f"  Std:  [{all_std.min():.3f}, {all_std.max():.3f}] | mean={all_std.mean():.3f}")

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].hist(all_min, bins=80, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    axs[0, 0].set_title('Min Value Distribution', fontsize=14)
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(all_max, bins=80, color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.5)
    axs[0, 1].set_title('Max Value Distribution', fontsize=14)
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(all_mean, bins=80, color='forestgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    axs[1, 0].set_title('Mean Value Distribution', fontsize=14)
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].hist(all_std, bins=80, color='crimson', alpha=0.8, edgecolor='black', linewidth=0.5)
    axs[1, 1].set_title('Standard Deviation Distribution', fontsize=14)
    axs[1, 1].set_xlabel('Value')
    axs[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  分布图已保存至: {os.path.abspath(output_plot)}")


def main():
    parser = argparse.ArgumentParser(description="分析 Nanopore 信号数据的统计分布")
    parser.add_argument(
        "npy_dir",
        type=str,
        help="包含 .npy 信号文件的目录路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="DataLoader 的批大小 (default: 16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 的工作进程数 (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_distribution.png",
        help="输出图像文件名 (default: data_distribution.png)"
    )

    args = parser.parse_args()

    analyze_data_distribution(
        npy_dir=args.npy_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_plot=args.output
    )


if __name__ == "__main__":
    main()
