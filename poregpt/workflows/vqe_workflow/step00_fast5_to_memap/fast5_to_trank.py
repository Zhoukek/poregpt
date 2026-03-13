import os
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
from pathlib import Path
from poregpt.utils.signal import nanopore_process_signal
import argparse


def process_single_fast5(args):
    """
    处理单个 fast5 文件，将其信号数据切分并保存为 numpy 数组
    
    Args:
        args (tuple): 包含以下参数的元组
            - fast5_path (str): 输入 fast5 文件路径
            - output_dir (str): 输出目录路径
            - nanopore_signal_process_strategy (str): 信号处理策略
            - signal_chunk_size (int): 信号块大小
            - signal_chunk_overlap_size (int): 重叠大小
    
    Returns:
        str: 处理结果信息字符串
    """
    fast5_path, output_dir, nanopore_signal_process_strategy, signal_chunk_size, signal_chunk_overlap_size = args
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储处理后的数据块
        processed_chunks = []
        
        # 读取 fast5 文件
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in f5.get_reads():
                try:
                    # 从 fast5 文件中提取原始信号数据
                    channel_info = read.handle[read.global_key + 'channel_id'].attrs
                    offset = int(channel_info['offset'])
                    scaling = channel_info['range'] / channel_info['digitisation']
                    raw = read.handle[read.raw_dataset_name][:]
                    signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
                    
                    # 应用信号处理策略
                    signal_processed = nanopore_process_signal(signal_raw, nanopore_signal_process_strategy)
                    
                    # 将处理后的信号按指定大小切分
                    L = len(signal_processed)
                    step_size = signal_chunk_size - signal_chunk_overlap_size
                    
                    # 确保信号长度足够处理
                    if L >= signal_chunk_size:
                        start = 0
                        chunk_idx = 0
                        
                        # 按步长滑动窗口切分信号
                        while start + signal_chunk_size <= L:
                            chunk = signal_processed[start : start + signal_chunk_size]
                            
                            # 将当前块信息添加到列表中
                            processed_chunks.append({
                                'read_id': read.read_id,
                                'chunk_idx': chunk_idx,
                                'chunk': chunk
                            })
                            
                            start += step_size
                            chunk_idx += 1
                    
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id} in {fast5_path}: {e}")
                    continue
        
        # 生成输出文件名（将 fast5 扩展名替换为 npy）
        base_name = Path(fast5_path).stem
        output_file = os.path.join(output_dir, f"{base_name}.npy")
        
        # 保存所有处理后的数据块
        if processed_chunks:
            # 提取所有数据块，忽略元数据信息
            chunks_data = [item['chunk'] for item in processed_chunks]
            np.save(output_file, chunks_data)
        
        return f"✅ Processed {fast5_path} -> {output_file} ({len(processed_chunks)} chunks)"
        
    except Exception as e:
        error_msg = f"❌ Error processing {fast5_path}: {str(e)}\n{traceback.format_exc()}"
        return error_msg


def convert_fast5_to_npy_parallel(
    input_dir,
    output_dir,
    nanopore_signal_process_strategy="apple",
    signal_chunk_size=40000,
    signal_chunk_overlap_size=10000,
    num_processes=None
):
    """
    并行转换目录中所有 fast5 文件为 npy 文件
    
    该函数遍历输入目录及其所有子目录，找到所有 fast5 文件，
    对每个文件应用信号处理策略并按指定大小切分，最终保存为 numpy 数组格式
    
    Args:
        input_dir (str): 输入目录路径，包含 fast5 文件
        output_dir (str): 输出目录路径，用于保存 npy 文件
        nanopore_signal_process_strategy (str): 信号处理策略名称
        signal_chunk_size (int): 每个信号块的大小，默认 40000
        signal_chunk_overlap_size (int): 信号块之间的重叠大小，默认 10000
        num_processes (int, optional): 并行进程数，默认为 CPU 核心数
    
    Returns:
        None
    """
    # 设置默认进程数为 CPU 核心数
    if num_processes is None:
        num_processes = cpu_count()
    
    # 查找所有 fast5 文件（包括 .fast5 和 .fasta5 扩展名）
    input_path = Path(input_dir)
    fast5_files = list(input_path.rglob("*.fast5")) + list(input_path.rglob("*.fasta5"))
    
    # 检查是否找到任何 fast5 文件
    if not fast5_files:
        print(f"⚠️ No fast5 files found in {input_dir}")
        return
    
    print(f"Found {len(fast5_files)} fast5 files")
    print(f"Using {num_processes} processes")
    
    # 准备参数列表用于多进程处理
    args_list = [
        (str(fast5_file), output_dir, nanopore_signal_process_strategy, signal_chunk_size, signal_chunk_overlap_size)
        for fast5_file in fast5_files
    ]
    
    # 使用进程池并行处理所有 fast5 文件
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_fast5, args_list),
            total=len(args_list),
            desc="Processing fast5 files"
        ))
    
    # 统计处理结果
    success_count = sum(1 for r in results if r.startswith("✅"))
    error_count = len(results) - success_count
    
    print(f"\n📊 Summary:")
    print(f"   Success: {success_count}")
    print(f"   Errors:  {error_count}")
    print(f"   Total:   {len(results)}")


def main():
    """
    主函数 - 工业级 fast5 到 npy 转换器入口点
    
    通过命令行参数接收用户配置，执行 fast5 文件到 npy 数组的批量转换。
    支持自定义信号处理策略、块大小、重叠大小和并行进程数。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Convert fast5 files to npy arrays with signal processing and chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/fast5/ -o /path/to/output/
  %(prog)s -i /data/fast5/ -o /data/processed/ -s apple -c 50000 -ov 15000 -p 16
        """
    )
    
    # 添加必需参数
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Input directory containing fast5 files (searches recursively)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory to save npy files'
    )
    
    # 添加可选参数
    parser.add_argument(
        '-s', '--strategy',
        type=str,
        default='apple',
        choices=['apple', 'med_flt', 'lp_flt'],  # 假设这些是支持的策略
        help='Nanopore signal processing strategy (default: apple)'
    )
    
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=40000,
        help='Size of each signal chunk (default: 40000)'
    )
    
    parser.add_argument(
        '-ov', '--overlap-size',
        type=int,
        default=10000,
        help='Overlap size between chunks (default: 10000)'
    )
    
    parser.add_argument(
        '-p', '--processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: number of CPU cores)'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行转换任务
    convert_fast5_to_npy_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        nanopore_signal_process_strategy=args.strategy,
        signal_chunk_size=args.chunk_size,
        signal_chunk_overlap_size=args.overlap_size,
        num_processes=args.processes
    )


if __name__ == "__main__":
    main()
