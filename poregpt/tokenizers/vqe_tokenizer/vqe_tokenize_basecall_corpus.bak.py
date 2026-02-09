import os
import gzip
import json
import numpy as np
import pandas as pd  # 添加 pandas 导入
from pathlib import Path
from tqdm import tqdm
import argparse
# import yaml  # 移除 yaml 导入，因为我们不再使用 YAML
from ont_fast5_api.fast5_interface import get_fast5_file # 假设 ont_fast5_api 已安装
from .vqe_tokenizer import VQETokenizer
import torch
from ...utils.signal import nanopore_process_signal
import time

def process_single_fast5(fast5_path, csv_path, model_path, device, nanopore_signal_process_strategy="apple"):
    """
    处理单个 FAST5 文件及其对应的 CSV 文件。

    Args:
        fast5_path (str): 单个 FAST5 文件的完整路径。
        csv_path (str): 对应的输入 CSV 文件的路径。
        model_path (str): VQE 模型的路径。
        device (str): 用于 VQE tokenizer 的设备 ('cpu', 'cuda', 'cuda:0', etc.)。
        nanopore_signal_process_strategy (str): 信号处理策略。
    """
    print(f"📖 正在读取 FAST5: {fast5_path}")
    print(f"📖 正在读取 CSV: {csv_path}")

    # 初始化 tokenizer 并指定设备
    tokenizer = VQETokenizer(
        model_ckpt=model_path,
        device=device,
        token_batch_size=8000
    )

    # 检查输入文件是否存在
    if not os.path.exists(fast5_path):
        print(f"❌ 未找到 FAST5 文件: {fast5_path}")
        return
    if not os.path.exists(csv_path):
        print(f"❌ 未找到 CSV 文件: {csv_path}")
        return

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    print(f"📊 从 CSV 加载了 {len(df)} 行。")

    # --- 重要改动：检查 CSV 中的 fast5 文件名是否与传入的 fast5_path 匹配 ---
    unique_fast5_names_in_csv = df['fast5'].unique()
    expected_fast5_filename = os.path.basename(fast5_path)
    if len(unique_fast5_names_in_csv) != 1 or unique_fast5_names_in_csv[0] != expected_fast5_filename:
         print(f"⚠️  CSV 文件中的 fast5 名称 ({unique_fast5_names_in_csv}) 与传入的 FAST5 文件名 ({expected_fast5_filename}) 不匹配或不唯一。")
         print(f"     确保 CSV 文件只包含来自 {expected_fast5_filename} 的数据。")
         return # 或者根据需要抛出异常


    # 获取输出路径 (基于 FAST5 文件路径，替换扩展名为 .jsonl.gz)
    output_jsonl_gz_path = os.path.splitext(fast5_path)[0] + '.jsonl.gz'
    print(f"🔄 正在处理 FAST5: {os.path.basename(fast5_path)} -> {os.path.basename(output_jsonl_gz_path)}")

    results_for_this_fast5 = []

    # 总体进度条
    total_rows = len(df)
    overall_pbar = tqdm(total=total_rows, desc="Processing Chunks", unit="chunk")

    # 按 read_id 分组
    grouped_by_read = df.groupby('read_id')

    with get_fast5_file(fast5_path, mode="r") as f5:
        for read_id, group_df_by_read in grouped_by_read:
            # 对于当前 read_id，只提取和处理一次信号
            try:
                # 在 FAST5 文件中查找特定的 read
                read = f5.get_read(read_id)
                if read is None:
                    print(f"    ⚠️  在 {os.path.basename(fast5_path)} 中未找到 Read ID {read_id}。正在跳过此 read 的所有 chunks。")
                    # 更新进度条，跳过这个 read 的所有 chunks
                    for _ in group_df_by_read.itertuples():
                        overall_pbar.update(1)
                    continue

                # --- 提取原始信号 (仅一次) ---
                channel_info = read.handle[read.global_key + 'channel_id'].attrs
                offset = int(channel_info['offset'])
                scaling = channel_info['range'] / channel_info['digitisation']
                raw = read.handle[read.raw_dataset_name][:]
                signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)

                # --- 应用处理策略 (仅一次) ---
                signal_processed = nanopore_process_signal(signal_raw, nanopore_signal_process_strategy)

                # --- 处理此 read_id 的所有 chunks ---
                for _, row in group_df_by_read.iterrows():
                    chunk_start = int(row['chunk_start']) # 确保为整数
                    chunk_size = int(row['chunk_size'])   # 确保为整数
                    bases = row['bases']

                    # --- 提取片段 (Chunk) ---
                    chunk_end = chunk_start + chunk_size
                    # 确保不超出信号长度范围
                    if chunk_end > len(signal_processed):
                         print(f"    ⚠️  Read {read_id}: 片段 ({chunk_start}:{chunk_end}) 超出信号长度 ({len(signal_processed)})。正在跳过。")
                         overall_pbar.update(1)
                         continue

                    chunk_signal = signal_processed[chunk_start:chunk_end]

                    # --- 标记化片段 ---
                    # time0 = time.time() # 可选：取消注释以测量时间
                    tokens = tokenizer.tokenize_chunk(chunk_signal)
                    text = "".join(tokens)
                    # time1 = time.time() # 可选：取消注释以测量时间
                    # time_cost = time1 - time0 # 可选：取消注释以测量时间
                    # print(f"      🔤 Tokenizing chunk {chunk_start}-{chunk_end} (Len: {len(chunk_signal)}) took {time_cost:.4f}s") # 可选：取消注释以打印时间

                    # --- 准备结果条目 ---
                    result_entry = {
                        "fast5": os.path.basename(fast5_path), # 存储原始文件名而非完整路径
                        "read_id": read_id,
                        "chunk_start": chunk_start,
                        "chunk_size": chunk_size,
                        "bases": bases,
                        "text": text
                    }
                    results_for_this_fast5.append(result_entry)

                    # 更新总体进度条
                    overall_pbar.update(1)


            except Exception as e:
                print(f"    ❌ 处理 {os.path.basename(fast5_path)} 中的 read {read_id} 时出错: {e}")
                # 即使发生错误，也要更新进度条，跳过这个 read 的所有 chunks
                for _ in group_df_by_read.itertuples():
                    overall_pbar.update(1)
                continue # 继续处理下一个 read_id

    # 关闭总体进度条
    overall_pbar.close()

    # --- 将结果写入 JSONL.GZ ---
    print(f"💾 正在将 {len(results_for_this_fast5)} 条结果写入 {os.path.basename(output_jsonl_gz_path)}")
    with gzip.open(output_jsonl_gz_path, 'wt', encoding='utf-8') as gz_file:
        for item in results_for_this_fast5:
            gz_file.write(json.dumps(item) + '\n')

    print("🎉 处理完成！")

def main():
    """
    主函数：解析命令行参数，初始化组件，并执行处理流程。
    """
    parser = argparse.ArgumentParser(description="Tokenize Nanopore signal chunks from a single FAST5 file using a VQE tokenizer.")
    parser.add_argument('--fast5_path', type=str, required=True, help='Path to the single FAST5 file.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the corresponding input CSV file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VQE model checkpoint.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the tokenizer on (e.g., cpu, cuda, cuda:0). Defaults to cuda.')
    parser.add_argument('--signal_strategy', type=str, default='apple', help='Nanopore signal processing strategy. Defaults to apple.')

    args = parser.parse_args()

    # 从命令行参数获取值
    fast5_path = args.fast5_path
    csv_path = args.csv_path
    model_path = args.model_path
    device = args.device
    signal_strategy = args.signal_strategy

    # 打印所有加载的配置参数
    print("--- Loaded Configuration from Command Line Args ---")
    print(f"  FAST5 Path (fast5_path): {fast5_path}")
    print(f"  CSV Path (csv_path): {csv_path}")
    print(f"  Model Path (model_path): {model_path}")
    print(f"  Device (device): {device}")
    print(f"  Signal Strategy (signal_strategy): {signal_strategy}")
    print("--------------------------------------------------")

    # 执行处理流程
    print("🚀 Starting single FAST5 processing...")
    process_single_fast5(
        fast5_path=fast5_path,
        csv_path=csv_path,
        model_path=model_path,
        device=device,
        nanopore_signal_process_strategy=signal_strategy
    )

if __name__ == "__main__":
    main()
