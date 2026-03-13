#!/bin/bash

# ==============================
# Nanopore RVQ Tokenizer - Parallel with Skip Existing & Direct Output
# ==============================

FAST5_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655/fast5"
OUTPUT_ROOT="fast5_jsonlgz"
MODEL_CKPT="/mnt/nas_syy/default/poregpt/shared/vqe_models_pass20_init_codebook_cnn_and_finetuned/checkpoints/porepgt_vqe_tokenizer.spoch38000.pth"
MODEL_CKPT="/mnt/gpudisk/dna_shards/vqe_models_pass18_only_init_codebook/nanopore_signal_tokenizer.pth.spoch16000.pth"
NUM_GPUS=4
MAX_CONCURRENT=8  # 总并发数

mkdir -p "$OUTPUT_ROOT"

# 获取所有 .fast5 文件（递归）
mapfile -d '' all_files < <(find "$FAST5_DIR" -name "*.fast5" -print0)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .fast5 files found." >&2
    exit 1
fi

echo "🔍 Found ${#all_files[@]} files. Running up to $MAX_CONCURRENT tasks concurrently..."

task_count=0
total=${#all_files[@]}
skipped=0

for ((i=0; i<total; i++)); do
    fast5="${all_files[i]}"
    
    # 构造输出路径
    rel_path="${fast5#$FAST5_DIR/}"
    output_file="$OUTPUT_ROOT/${rel_path%.fast5}.jsonl.gz"
    output_dir="$(dirname "$output_file")"
    
    # ✅ 如果目标文件已存在，跳过
    if [ -f "$output_file" ]; then
	echo "skiping $output_file due to already existed"
        ((skipped++))
        continue
    fi
    
    mkdir -p "$output_dir"
    
    # 分配 GPU
    gpu_id=$(( task_count % NUM_GPUS ))
    
    # 控制并发
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n
    fi
    
    # 启动任务：✅ 不重定向日志，直接输出
    echo "➡️  Submitting $(basename "$fast5") to GPU $gpu_id (output: $output_file)" >&2
    poregpt-vqe-tokenize-fast5 \
        --fast5_file "$fast5" \
        --output_file "$output_file" \
        --model_ckpt "$MODEL_CKPT" \
        --gpu_id "$gpu_id" \
        --signal_process_strategy "apple" \
	--token_batch_size 8000 & 
    
    ((task_count++))
done

# 等待所有后台任务完成
wait

echo "🎉 Done. Processed: $((total - skipped)), Skipped (already exist): $skipped" >&2
