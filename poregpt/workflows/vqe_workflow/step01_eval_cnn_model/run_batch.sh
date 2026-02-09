#!/bin/bash

# --- Configuration Section ---
# Modify these variables according to your setup

# Input directory containing .npy files
INPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train"

# Output directory for processed .npy shards
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train_features_cnn6"

# Path to your trained CNN model checkpoint (.pth file)
MODEL_CHECKPOINT="/mnt/nas_syy/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step00_train_cnn_model/train_cnn6/models/nanopore_cnn6.epoch128.pth"

# 硬件资源配置
NUM_GPUS=4
MAX_CONCURRENT=8  # <--- 修改点：总并发任务数，可以大于 NUM_GPUS

# 模型参数
BATCH_SIZE=1024 
CNN_TYPE=6 
FEATURE_DIM=128 
SHARD_SIZE=10000000 

# --- End of Configuration ---

# Validate NUM_GPUS
if [[ $NUM_GPUS -le 0 ]]; then
  echo "Error: NUM_GPUS must be greater than 0!"
  exit 1
fi

echo "Starting distributed CNN evaluation process..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Max Concurrent Tasks: $MAX_CONCURRENT"
echo "----------------------------------------"

# 检查输入目录
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist!"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# 获取所有 .npy 文件（递归）
echo "🔍 Finding .npy files..."
mapfile -d '' all_files < <(find "$INPUT_DIR" -name "*.npy" -type f -print0)

total=${#all_files[@]}
if [ $total -eq 0 ]; then
    echo "❌ No .npy files found in $INPUT_DIR." >&2
    exit 1
fi

# 排序以保证一致性
IFS=$'\n' sorted_files=($(sort <<<"${all_files[*]}"))
unset IFS
all_files=("${sorted_files[@]}")

echo "🔍 Found $total .npy files. Running up to $MAX_CONCURRENT tasks..."

# 初始化计数器
task_count=0
skipped=0

for ((i=0; i<total; i++)); do
    npy_file="${all_files[i]}"

    # 构造输出路径并保持子目录结构
    rel_path="${npy_file#$INPUT_DIR/}"
    output_subdir="$OUTPUT_DIR/$(dirname "$rel_path")"
    base_filename=$(basename "$npy_file")
    output_feature_dir="$output_subdir/features_${base_filename}"

    # 构造锁文件路径
    lock_file="$output_feature_dir/.processing_lock"

    # ✅ 检查是否已完成
    if [ -d "$output_feature_dir" ] && [ -f "$output_feature_dir/shards.json" ]; then
        ((skipped++))
        continue
    fi

    # ✅ 检查是否有锁（避免多进程冲突）
    if [ -f "$lock_file" ]; then
        echo "⚠️  Skipping $(basename "$npy_file") (lock file exists)." >&2
        ((skipped++))
        continue
    fi

    # 准备输出目录
    mkdir -p "$output_subdir"
    touch "$lock_file"

    # --- 核心逻辑：分配 GPU ---
    # 使用当前启动的任务总数对 GPU 数量取模
    gpu_id=$(( task_count % NUM_GPUS ))

    # --- 核心逻辑：控制并发 ---
    # 如果当前正在运行的后台任务数达到 MAX_CONCURRENT，则等待其中任意一个结束
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n
    fi

    echo "➡️  Submitting $(basename "$npy_file") to GPU $gpu_id (Task #$task_count)" >&2

    # 设置设备参数
    DEVICE_ARG="cuda:$gpu_id"

    # 启动后台任务
    python3 cnn_eval_single_file.py \
         --input_npy_file "$npy_file" \
         --checkpoint_path "$MODEL_CHECKPOINT" \
         --output_dir "$output_feature_dir" \
         --shard_size $SHARD_SIZE \
         --batch_size $BATCH_SIZE \
         --cnn_type $CNN_TYPE \
         --feature_dim $FEATURE_DIM \
         --device "$DEVICE_ARG" &

    # 只有真正启动了任务，才增加计数器
    ((task_count++))
done

# 等待所有后台任务完成
echo "⏳ Waiting for remaining background tasks to finish..."
wait

# 清理锁文件
echo "🧹 Cleaning up lock files..."
find "$OUTPUT_DIR" -name ".processing_lock" -delete

# 打印最终统计
processed=$task_count
echo "----------------------------------------"
echo "🎉 CNN evaluation finished!"
echo " Total Files Found: $total"
echo " Successfully Processed: $processed"
echo " Skipped (Already Done): $skipped"
echo " Final Check: $((processed + skipped)) / $total"
