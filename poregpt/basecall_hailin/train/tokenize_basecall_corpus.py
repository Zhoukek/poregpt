

def process_basecall_corpus(fast5_dir, csv_path, vqetokenizer, nanopore_signal_process_strategy="apple"):
    """
    读取 CSV 文件，从 FAST5 文件中提取片段，对其进行标记化，
    并将结果按 FAST5 文件名分组保存到 JSONL.GZ 文件中。

    Args:
        fast5_dir (str): 包含 FAST5 文件的目录路径。
        csv_path (str): 输入 CSV 文件的路径。
        vqetokenizer: 一个预定义的 VQE tokenizer 类实例，具有 tokenize_chunk 方法。
        nanopore_signal_process_strategy (str): 信号处理策略。
    """

    # 读取 CSV 文件
    print(f"📖 正在读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📊 从 CSV 加载了 {len(df)} 行。")

    # 按 fast5 文件名分组，以便高效处理
    grouped_df = df.groupby('fast5')
    print(f"📁 找到 {len(grouped_df)} 个唯一的 FAST5 文件。")

    for fast5_filename, group in grouped_df:
        fast5_path = os.path.join(fast5_dir, fast5_filename)

        if not os.path.exists(fast5_path):
            print(f"❌ 未找到 FAST5 文件: {fast5_path}")
            continue # 如果文件不存在则跳过

        output_jsonl_gz_path = os.path.join(fast5_dir, f"{os.path.splitext(fast5_filename)[0]}.jsonl.gz")
        print(f"🔄 正在处理 FAST5: {fast5_filename} -> {os.path.basename(output_jsonl_gz_path)}")

        results_for_this_fast5 = []

        # 为该组中的所有 reads 一次性打开 FAST5 文件
        with get_fast5_file(fast5_path, mode="r") as f5:
            for _, row in group.iterrows():
                read_id = row['read_id']
                chunk_start = int(row['chunk_start']) # 确保为整数
                chunk_size = int(row['chunk_size'])   # 确保为整数
                bases = row['bases']

                try:
                    # 在 FAST5 文件中查找特定的 read
                    read = f5.get_read(read_id)
                    if read is None:
                        print(f"    ⚠️  在 {fast5_filename} 中未找到 Read ID {read_id}。正在跳过。")
                        continue

                    # --- 提取原始信号 ---
                    channel_info = read.handle[read.global_key + 'channel_id'].attrs
                    offset = int(channel_info['offset'])
                    scaling = channel_info['range'] / channel_info['digitisation']
                    raw = read.handle[read.raw_dataset_name][:]
                    signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)

                    # --- 应用处理策略 ---
                    signal_processed = nanopore_process_signal(signal_raw, nanopore_signal_process_strategy)

                    # --- 提取片段 (Chunk) ---
                    # 根据开始位置和大小计算结束索引
                    chunk_end = chunk_start + chunk_size
                    # 确保不超出信号长度范围
                    if chunk_end > len(signal_processed):
                         print(f"    ⚠️  片段 ({chunk_start}:{chunk_end}) 超出信号长度 ({len(signal_processed)})，read ID 为 {read_id}。正在跳过。")
                         continue

                    chunk_signal = signal_processed[chunk_start:chunk_end]

                    # --- 标记化片段 ---
                    # 调用提供的 vqetokenizer 实例的 tokenize_chunk 方法
                    print(f"    🔤 正在标记化 read {read_id} 的片段 {chunk_start}-{chunk_end} (长度: {len(chunk_signal)})")
                    text = vqetokenizer.tokenize_chunk(chunk_signal)

                    # --- 准备结果条目 ---
                    result_entry = {
                        "fast5": fast5_filename,
                        "read_id": read_id,
                        "chunk_start": chunk_start,
                        "chunk_size": chunk_size,
                        "bases": bases,
                        "text": text
                    }
                    results_for_this_fast5.append(result_entry)

                except Exception as e:
                    print(f"    ❌ 处理 {fast5_filename} 中的 read {read_id} (片段 {chunk_start}-{chunk_start+chunk_size}) 时出错: {e}")
                    continue # 继续处理此 FAST5 的下一行

        # --- 将此 FAST5 的结果写入 JSONL.GZ ---
        print(f"💾 正在将 {len(results_for_this_fast5)} 条结果写入 {os.path.basename(output_jsonl_gz_path)}")
        with gzip.open(output_jsonl_gz_path, 'wt', encoding='utf-8') as gz_file:
            for item in results_for_this_fast5:
                gz_file.write(json.dumps(item) + '\n')

    print("🎉 所有处理完成！")
