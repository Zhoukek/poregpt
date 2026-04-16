#!/bin/bash

source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt

# # --- Configuration ---
# INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_032g/memap_mongoq30/trank/validation"
# OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g/memap_mongoq30/"
# MODEL_CHECKPOINT="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/models/porepgt_vqe_tokenizer.step25000.pth"
# MODEL_TYPE=25
# NUM_GPUS=1
# BATCH_SIZE=1
# MAX_CONCURRENT=$NUM_GPUS

# --- 你只需要改这里！---
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_032g/memap_mongoq30/trank/validation/validation_00017.npy"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/basecall"

MODEL_CHECKPOINT="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/models/porepgt_vqe_tokenizer.step25000.pth"
MODEL_TYPE=25
DEVICE="cuda:0"
BATCH_SIZE=1

# ------------------------

# 自动生成输出文件名
output_file="$OUTPUT_DIR/$(basename ${INPUT_DIR%.npy}.jsonl.gz)"
mkdir -p "$OUTPUT_DIR"

echo "处理文件：$INPUT_DIR"
echo "输出到：$output_file"

# 直接执行处理
poregpt-vqe-tokenize-trank \
    -i "$INPUT_DIR" \
    -o "$output_file" \
    --model-type "$MODEL_TYPE" \
    --model-ckpt "$MODEL_CHECKPOINT" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

echo "✅ 单个文件处理完成！"