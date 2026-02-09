#!/bin/bash

# ============================================================================
# 脚本名称: step03_extract_cnn_features.sh
# 功能:     使用训练好的 CNN 模型，将预处理的 Nanopore 信号片段（.npy）编码为
#          64 维特征向量，并保存为分片 memmap 格式（每个 shard ≤1M 样本），
#          附带 shards.json 索引文件，用于后续 LLM 训练。
#
# 依赖:
#   - Python 3.8+ 环境已激活（含 PyTorch、nanopore_signal_tokenizer）
#   - 已通过 pip install -e . 安装 nanopore_signal_tokenizer 包
#   - checkpoint 模型文件存在（由 step02_train_cnn_model.py 生成）
#
# 作者:     Your Name
# 日期:     2025-06-10
# ============================================================================

# ----------------------------------------------------------------------------
# 配置区：所有参数集中在此处，便于修改、版本控制和实验管理。
# 每个变量对应 extract_features_sharded.py 的 argparse 参数。
# ----------------------------------------------------------------------------

# 输入目录：包含原始信号 chunk 的 memmap shards（由 step01 生成）
INPUT_SHARDS_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train"

# 输出目录：存放分片的 64-dim 特征 memmap 文件 + shards.json
OUTPUT_SHARD_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train_features_cnn3"

# CNN 模型检查点路径（必须是包含 'model_state_dict' 的 .pth 文件）
CHECKPOINT_PATH="/mnt/nas_syy/default/poregpt/shared/cnn_models/cnn3_apple/nanopore_cnn3.epoch58.pth"

# 每个 shard 最多包含多少样本（默认 10,000,000）
SHARD_SIZE=10000000

# 特征维度（必须为 64，与模型 flatten 后输出一致）
FEATURE_DIM=64

# 推理批大小（越大越快，但受显存限制）
BATCH_SIZE=8192

# 数据加载线程数
NUM_WORKERS=32

# CNN 类型（需与训练时一致）
CNN_TYPE=3

# 设备（通常为 cuda 或 cpu）
DEVICE="cuda:0"

# ----------------------------------------------------------------------------
# 构造完整的 Python 命令
# 使用数组避免空格/引号问题，确保命令安全可靠
# ----------------------------------------------------------------------------
CMD=(
    torchrun
    --nproc_per_node=1        
    --master_port=29506      
    -m poregpt.tokenizers.vqe_tokenizer.cnn_eval 
    --input_shards_dir "$INPUT_SHARDS_DIR"
    --output_shard_dir "$OUTPUT_SHARD_DIR"
    --checkpoint_path "$CHECKPOINT_PATH"
    --shard_size "$SHARD_SIZE"
    --feature_dim "$FEATURE_DIM"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --cnn_type "$CNN_TYPE"
    --device "$DEVICE"
)

# ----------------------------------------------------------------------------
# 打印完整命令（用于调试和复现）
# printf "%q" 会对参数进行 shell 转义，确保输出的命令可直接复制运行
# ----------------------------------------------------------------------------
echo ">>> Running CNN feature extraction command:"
printf "%q " "${CMD[@]}"
echo  # 换行
echo "--------------------------------------------------"

# ----------------------------------------------------------------------------
# 执行命令
# ----------------------------------------------------------------------------
"${CMD[@]}"

# ----------------------------------------------------------------------------
# 成功提示
# ----------------------------------------------------------------------------
echo "✅ Feature extraction completed."
echo "📁 Features saved to: $OUTPUT_SHARD_DIR"
echo "📄 Shard index: $OUTPUT_SHARD_DIR/shards.json"
