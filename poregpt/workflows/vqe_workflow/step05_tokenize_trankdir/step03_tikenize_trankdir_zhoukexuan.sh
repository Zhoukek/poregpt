#!/bin/bash
# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt
# --- Configuration Section ---
# Modify these variables according to your setup

# Input directory containing .npy files
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/trank"

# Output directory for .jsonl.gz files
OUTPUT_DIR="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_595g"


#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================


# 模型检查点路径
MODEL_CHECKPOINT="/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/models/porepgt_vqe_tokenizer.step25000.pth"

# 运行参数
NUM_GPUS=8            # 使用的 GPU 数量 (0-7)
MAX_CONCURRENT=4      # 最大并发任务数 (通常设为 GPU 数量的 1-2 倍)
BATCH_SIZE=16         # 模型推理批大小
LAYER=0               # Tokenization 使用的层级

# --- End of Configuration ---

# 路径预处理：去除末尾斜杠，确保字符串替换逻辑准确
INPUT_DIR="${INPUT_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

echo "Starting distributed tokenization process..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "Number of GPUs: $NUM_GPUS"
echo "Max Concurrent Tasks: $MAX_CONCURRENT"
echo "Batch Size: $BATCH_SIZE"
echo "----------------------------------------"

# Check if the input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then 
  echo "Error: Input directory '$INPUT_DIR' does not exist!"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get all .npy files recursively
echo "🔍 Finding .npy files..."
mapfile -d '' all_files < <(find "$INPUT_DIR" -name "*.npy" -type f -print0)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .npy files found in $INPUT_DIR." >&2
    exit 1
fi

total=${#all_files[@]}
echo "🔍 Found $total .npy files."

# Initialize counters
task_count=0
processed=0
skipped=0

# Loop through each .npy file
for npy_file in "${all_files[@]}"; do
    if [ -z "$npy_file" ]; then
        continue
    fi

    # 1. 计算相对路径
    rel_path="${npy_file#$INPUT_DIR/}" 
    
    # 2. 构造输出文件路径 (修正后的逻辑：直接基于 OUTPUT_DIR 和 rel_path)
    output_file="$OUTPUT_DIR/${rel_path%.npy}.jsonl.gz"
    
    # 3. 获取输出目录 (用于创建子目录)
    output_dir="$(dirname "$output_file")"

    # ✅ Check if output file already exists, skip if it does
    if [ -f "$output_file" ]; then
        ((skipped++))
        continue
    fi

    # Ensure the output subdirectory exists
    mkdir -p "$output_dir"

    # Determine which GPU to use for this task
    gpu_id=$(( task_count % NUM_GPUS ))

    # Control concurrency
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n 
    fi

    # Log the task submission
    echo "➡️  Submitting $(basename "$npy_file") to GPU $gpu_id" >&2

    # Start the tokenization command in the background
    DEVICE_ARG="cuda:$gpu_id"
    
    # 保持调用命令与你的环境/脚本名一致
    # 注意：确保命令 'poregpt-vqe-tokenize-trank' 在你的环境变量中，或替换为 python3 vqe_tokenize_trank.py
    echo "Executing: poregpt-vqe-tokenize-trank -i '$npy_file' -o '$output_file' --model-ckpt '$MODEL_CHECKPOINT' --device '$DEVICE_ARG' --batch-size $BATCH_SIZE"
    
    poregpt-vqe-tokenize-trank \
         -i "$npy_file" \
         -o "$output_file" \
         --model-ckpt "$MODEL_CHECKPOINT" \
         --device "$DEVICE_ARG" \
         --batch-size $BATCH_SIZE &

    # Increment counters
    ((task_count++))
    ((processed++))
done

# Wait for *all* remaining background tasks to complete
echo "⏳ Waiting for all $processed tasks to finish..."
wait

# Print final summary
final_total=$((processed + skipped))
echo "----------------------------------------"
echo "🎉 Tokenization finished!"
echo "  Total Files Found: $total"
echo "  Submitted for Processing: $processed"
echo "  Skipped (Already Existed): $skipped"
echo "  Final Count (P+S): $final_total"

if [ $final_total -eq $total ]; then
  echo "✅ Counts match. Process completed."
else
  echo "⚠️  Counts don't match total found. Unexpected state."
fi
