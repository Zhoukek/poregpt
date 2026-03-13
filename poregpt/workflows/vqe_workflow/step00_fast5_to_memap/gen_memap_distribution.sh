#!/bin/bash

# =============================================================================
# 工业级纳米孔测序信号分布统计生成脚本
# =============================================================================
# 功能描述:
#   为每个.npy文件生成对应的信号分布统计CSV文件，用于后续LLM训练
#   每个CSV文件包含value(信号值), count(频次), ratio(比例)三列统计信息
#
# 参数配置说明:
#   INPUT_DIR:    输入目录路径 - 包含待处理的.npy信号文件
#   OUTPUT_DIR:   输出目录路径 - 存放生成的*_dist.csv统计文件  
#   WORKERS:      并行进程数量 - 提高大数据集处理效率 (留空则自动检测)
#   MIN_VAL:      信号值范围下限 - 统计范围起始值
#   MAX_VAL:      信号值范围上限 - 统计范围结束值  
#   BINS:         分箱数量 - 将信号值范围划分为多少个区间 (2000个bin = 0.01宽度)
#   SEQUENTIAL:   顺序处理开关 - 设为"true"时使用单进程顺序处理
# =============================================================================

# =========================== 用户配置区 ===========================

DATASET_NAME="human_dna_032g"
MEMAP_NAME="memap_appleq30"
# 输入目录: 修改为你的.npy文件所在目录
INPUT_DIR="/mnt/nas_syy/default/poregpt/dataset/$DATASET_NAME/$MEMAP_NAME/train"

# 输出目录: 统计文件存放位置
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/dataset/$DATASET_NAME/$MEMAP_NAME/train"

# 并行进程数: 大数据集建议设置为CPU核心数的1-2倍 (留空自动检测)
WORKERS=32

# 信号值范围: 信号统计的数值范围
MIN_VAL=-10.0
MAX_VAL=10.0

# 分箱参数: 2000个bin对应0.01的bin宽度 ((10-(-10))/2000 = 0.01)
BINS=2000

# 处理模式: true=顺序处理 false=并行处理
SEQUENTIAL="false"
# ===================================================================

# 检查Python脚本是否存在
if [ ! -f "gen_memap_distribution.py" ]; then
    echo "错误: 找不到 gen_memap_distribution.py 脚本文件"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始处理信号分布统计..."
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "处理模式: $(if [ "$SEQUENTIAL" = "true" ]; then echo "顺序"; else echo "并行($WORKERS进程)"; fi)"
echo "信号范围: [$MIN_VAL, $MAX_VAL], 分箱数: $BINS"

# 构建执行命令
if [ "$SEQUENTIAL" = "true" ]; then
    CMD="python3 gen_memap_distribution.py '$INPUT_DIR' --output '$OUTPUT_DIR' --sequential --min-val $MIN_VAL --max-val $MAX_VAL --bins $BINS"
else
    if [ -n "$WORKERS" ]; then
        CMD="python3 gen_memap_distribution.py '$INPUT_DIR' --output '$OUTPUT_DIR' --workers $WORKERS --min-val $MIN_VAL --max-val $MAX_VAL --bins $BINS"
    else
        CMD="python3 gen_memap_distribution.py '$INPUT_DIR' --output '$OUTPUT_DIR' --min-val $MIN_VAL --max-val $MAX_VAL --bins $BINS"
    fi
fi

# 执行处理
eval $CMD

echo "信号分布统计处理完成!"
