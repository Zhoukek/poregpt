#!/bin/bash

# ==============================
# Nanopore RVQ Tokenizer for Basecall Corpus - Parallel with Skip Existing & Direct Output
# ==============================

# --- 配置区域 ---
FAST5_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/validation"
# 实验的C64K_CNN3
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/validation_vqe25s57500"
MODEL_CKPT="/mnt/nas_syy/default/poregpt/models/vqe_models/pass25_dna37g_scratch_c64k_cnn3_lr5e4_dbsz64_gbsz8192_lc2400/models/porepgt_vqe_tokenizer.step57500.pth"

# 实验的C64K_CNN6
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/validation_vqe44s67000"
MODEL_CKPT="/mnt/nas_syy/default/poregpt/models/vqe_models/pass44_dna37g_scratch_c64k_cnn6-lr5e4_dbsz32_gbsz8192-lc2400/models/porepgt_vqe_tokenizer.step67000.pth"

NUM_GPUS=4
MAX_CONCURRENT=8  # 总并发数
SIGNAL_STRATEGY="lemon"
TOKEN_BATCH_SIZE=1280
# --- 配置区域结束 ---

mkdir -p "$OUTPUT_DIR"

# 获取所有 .fast5 文件（递归）
mapfile -d '' all_files < <(find "$FAST5_DIR" -name "*.fast5" -print0)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .fast5 files found in $FAST5_DIR." >&2
    exit 1
fi

echo "🔍 Found ${#all_files[@]} .fast5 files in $FAST5_DIR. Running up to $MAX_CONCURRENT tasks concurrently..."

task_count=0
total=${#all_files[@]}
processed=0
skipped=0

for ((i=0; i<total; i++)); do
    fast5_path="${all_files[i]}"

    # 推断对应的 .bc.csv 文件路径
    csv_path="${fast5_path%.fast5}.bc.csv"

    # 检查 .bc.csv 文件是否存在
    if [ ! -f "$csv_path" ]; then
        echo "⚠️  Skipping $fast5_path: corresponding .bc.csv file ($csv_path) not found." >&2
        ((skipped++)) # 认为找不到 CSV 的也算跳过
        continue
    fi

    # --- ✅ 修改：构建输出路径 ---
    # 计算 fast5 文件相对于 FAST5_DIR 的相对路径
    relative_path="${fast5_path#$FAST5_DIR/}"
    # 如果 relative_path 为空（意味着 fast5_path 就是 FAST5_DIR），则设置为一个默认文件名
    if [ -z "$relative_path" ]; then
        relative_path="root_output.jsonl.gz"
    else
        # 将 .fast5 替换为 .jsonl.gz
        relative_path="${relative_path%.fast5}.jsonl.gz"
    fi
    
    # 最终的输出文件路径
    output_file="$OUTPUT_DIR/$relative_path"
    
    # 计算输出文件所在的目录
    output_dir_of_file=$(dirname "$output_file")


    # ✅ 如果目标文件已存在，跳过
    if [ -f "$output_file" ]; then
        echo "⏭️  Skipping $output_file due to already existing." >&2
        ((skipped++))
        continue
    fi

    # 确保输出文件的上级目录存在
    mkdir -p "$output_dir_of_file"

    # 分配 GPU
    gpu_id=$(( task_count % NUM_GPUS ))

    # 控制并发
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n # 等待任意一个后台任务结束
    fi

    # 启动任务：✅ 不重定向日志，直接输出
    echo "➡️  Submitting $(basename "$fast5_path") (CSV: $(basename "$csv_path")) to GPU $gpu_id (output: $output_file)" >&2

    # 启动 Python 脚本
    # 注意：Python 脚本中的 --output_dir 参数现在决定了整个输出的根目录
    # Python 脚本内部会根据 fast5_path 的名称生成最终的 jsonl.gz 文件名
    poregpt-vqe-tokenize-basecall-corpus \
        --fast5_path "$fast5_path" \
        --csv_path "$csv_path" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_CKPT" \
        --device "cuda:$gpu_id" \
        --signal_strategy "$SIGNAL_STRATEGY" & # 注意：token_batch_size 是在 Python 脚本内部硬编码的，或者也需要通过命令行传递

    ((task_count++))
    ((processed++)) # 只有提交了任务才算处理
done

# 等待所有后台任务完成
wait

echo "🎉 Done. Processed: $processed, Skipped (missing CSV or already exist): $skipped, Total scanned .fast5: $total" >&2
