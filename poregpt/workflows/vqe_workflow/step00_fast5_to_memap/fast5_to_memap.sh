#!/bin/bash

# =============================================================================
# Fast5 to Memmap-friendly NPY Converter Script
#
# 描述:
#   此脚本将牛津纳米孔技术(Oxford Nanopore Technologies, ONT)的fast5文件转换为
#   适合内存映射(memmap)使用的numpy数组(.npy)格式。转换过程包括信号处理、
#   数据分块(chunking)以及元数据(shards.json)生成。
#   脚本使用多进程进行高效批量转换。
#
# 配置说明:
#   编辑下方变量来自定义转换行为。
#   脚本将使用预定义值，无需命令行参数。
#
# 依赖要求:
#   - 已安装所需Python包的Python环境
#   - 足够的磁盘空间存储输出文件
#   - 输入目录的读取权限
#   - 输出目录的写入权限
# =============================================================================

# 任何命令失败时立即退出
set -euo pipefail

# 全局变量定义
SCRIPT_NAME=$(basename "$0")                                    # 当前脚本名称
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # 当前脚本所在目录
START_TIME=$(date +%s)                                          # 脚本开始执行的时间戳

# =============================================================================
# 用户配置区
# 编辑这些变量来配置转换行为
# =============================================================================

# 数据集名称
DATASET_NAME="human_min0_max2_read96655_p90"
DATASET_NAME="human_dna_152g"
DATASET_NAME="human_dna_280g"
DATASET_NAME="human_dna_032g"
DATASET_NAME="human_dna_595g"

# 信号处理策略：
# - "apple": Apple算法（通常用于去噪）
# - "stone": Stone算法（另一种信号处理方法）
# - "lemon": Lemon算法（新增策略）
STRATEGY="stone"
# 30表示范围在-3.0到3.0之间的片段才会留存
CLIP=0


# 输入目录数组：包含fast5文件的目录（会递归搜索子目录）
INPUT_DIRS=(
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/fast5/train"
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/fast5/validation"
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/fast5/test"
)

# 输出目录数组：保存npy文件的目录（必须与输入目录一一对应）
OUTPUT_DIRS=(
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/memap_${STRATEGY}q${CLIP}/train"
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/memap_${STRATEGY}q${CLIP}/validation"
    "/mnt/nas_syy/default/poregpt/dataset/${DATASET_NAME}/memap_${STRATEGY}q${CLIP}/test"
)
# 每个信号块的大小（采样点数量）
# 注意：此值需要与下游模型或分析工具兼容
CHUNK_SIZE=12000

# 分块之间的重叠大小（采样点数量）
# 重叠有助于减少分块边界处的信息丢失
OVERLAP_SIZE=600

# 并行处理的进程数
# 建议设置为CPU核心数或稍少一些，避免过度占用系统资源
PROCESSES=64

# 输出数据类型
# 通常使用 "float32" 以平衡精度和内存使用
DTYPE="float32"

# =============================================================================
# 脚本主体
# =============================================================================

# 检查输入和输出目录数组长度是否相等
if [[ ${#INPUT_DIRS[@]} -ne ${#OUTPUT_DIRS[@]} ]]; then
    echo "错误: 输入目录数组和输出目录数组的长度不相等!" >&2
    echo "输入目录数量: ${#INPUT_DIRS[@]}"
    echo "输出目录数量: ${#OUTPUT_DIRS[@]}"
    exit 1
fi

# 打印脚本启动信息
echo "========================================="
echo "启动 Fast5 到 Memmap-friendly NPY 转换脚本"
echo "脚本名称: $SCRIPT_NAME"
echo "执行目录: $SCRIPT_DIR"
echo "开始时间: $(date -d "@$START_TIME" '+%Y-%m-%d %H:%M:%S')"
echo "数据集名称: $DATASET_NAME"
echo "========================================="

# 验证信号处理策略是否有效
if [[ "$STRATEGY" != "apple" && "$STRATEGY" != "stone"  && "$STRATEGY" != "lemon" && "$STRATEGY" != "tango" && "$STRATEGY" != "mongo" ]]; then
    echo "错误: 无效的信号处理策略 '$STRATEGY'. 有效选项为: apple, stone, lemon, tango, mongo" >&2
    exit 1
fi

# 打印当前配置
echo "开始转换..."
echo "信号处理策略: $STRATEGY"
echo "信号块大小: $CHUNK_SIZE"
echo "块间重叠大小: $OVERLAP_SIZE"
echo "并行进程数: $PROCESSES"
echo "输出数据类型: $DTYPE"
echo ""

# 循环处理每个输入-输出目录对
for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIR="${INPUT_DIRS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    
    echo "-----------------------------------------"
    echo "处理第 $((i+1)) 个目录对:"
    echo "输入目录: $INPUT_DIR"
    echo "输出目录: $OUTPUT_DIR"
    echo "-----------------------------------------"
    
    # 验证输入目录是否存在
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "错误: 输入目录不存在: $INPUT_DIR" >&2
        exit 1
    fi

    # 创建输出目录（如果不存在）
    mkdir -p "$OUTPUT_DIR"

    # 运行Python转换脚本
    # 使用 -u 参数确保Python的输出不被缓冲，实时显示进度条
    echo "正在执行Python转换程序..."
    python3 -u fast5_to_memap.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --strategy "$STRATEGY" \
        --chunk_size "$CHUNK_SIZE" \
        --overlap_size "$OVERLAP_SIZE" \
        --clip_value $CLIP \
        --dtype "$DTYPE" \
        --num_workers "$PROCESSES"

    # 检查输出目录中是否成功生成了 shards.json
    if [[ -f "$OUTPUT_DIR/shards.json" ]]; then
        echo "✓ 成功生成 shards.json 元数据文件"
        # 可选：打印总样本数摘要
        if command -v jq &> /dev/null; then
            TOTAL_SAMPLES=$(jq '.total_samples' "$OUTPUT_DIR/shards.json" 2>/dev/null || echo "无法读取样本总数")
            echo "总样本数: $TOTAL_SAMPLES"
        fi
    else
        echo "⚠️  警告: 未找到 shards.json 文件，请检查转换过程是否正常完成。" >&2
        exit 1
    fi
    
    echo "完成处理第 $((i+1)) 个目录对"
done

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# 打印完成信息
echo ""
echo "========================================="
echo "所有转换完成！"
echo "结束时间: $(date -d "@$END_TIME" '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: $ELAPSED_TIME 秒 ($(printf '%dh %dm %ds' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60))))"
echo "数据集名称: $DATASET_NAME"
echo "========================================="
