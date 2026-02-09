#!/bin/bash

# ==============================
# Nanopore RVQ Tokenizer - Single File Version
# ==============================

set -e  # 出错立即退出
which vq-tokenize
# --- 配置 ---
MODEL_CKPT="/mnt/nas_syy/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/w64/models_scratch_cnn3/porepgt_vqe_tokenizer.step22500.pth"

GPU_ID=0
OUTPUT_ROOT="fast5_jsonlgz"


FAST5_FILE="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/train/train_00001.fast5"

if [ ! -f "$FAST5_FILE" ]; then
    echo "❌ Input file not found: $FAST5_FILE"
    exit 1
fi

# --- 构造输出路径 ---
mkdir -p "$OUTPUT_ROOT"
rel_name=$(basename "$FAST5_FILE")
output_file="$OUTPUT_ROOT/${rel_name%.fast5}.jsonl.gz"

# --- 执行 tokenization ---
echo "➡️  Tokenizing: $FAST5_FILE"
echo "📤 Output:      $output_file"
echo "🖥️  GPU:         $GPU_ID"

poregpt-vqe-tokenize-fast5-batched\
    --fast5_file "$FAST5_FILE" \
    --output_file "$output_file" \
    --model_ckpt "$MODEL_CKPT" \
    --gpu_id "$GPU_ID" \
    --signal_process_strategy "apple" \
    --token_batch_size 8000
echo "✅ Done."
