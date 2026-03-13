#!/bin/bash

# ============================================================================
# 脚本名称: step00_fast5_to_chunks_dna.sh
# 功能:     使用 DATASET_DIR 统一管理数据根目录，显式映射 FAST5 → 输出目录。
# 注意:     依赖已安装的 nanopore_llm_workflow 包（提供 fast5-to-chunks 命令）
# ============================================================================

# ----------------------------------------------------------------------------
# 核心配置：数据集根目录
# ----------------------------------------------------------------------------
DATASET_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655_10p"

# ----------------------------------------------------------------------------
# 输入-输出映射（基于 DATASET_DIR 构建）
#   - 每个 FAST5 子目录对应一个 chunk 输出子目录
#   - 命名约定：train → train2, test → test2（避免覆盖旧数据）
# ----------------------------------------------------------------------------
FAST5_DIRS=(
    "$DATASET_DIR/fast5/train"
    "$DATASET_DIR/fast5/test"
    "$DATASET_DIR/fast5/validation"
)

OUTPUT_DIRS=(
    "$DATASET_DIR/chunk/train"
    "$DATASET_DIR/chunk/test"
    "$DATASET_DIR/chunk/validation"
)

# 检查数组长度是否一致（防止配置错位）
if [ "${#FAST5_DIRS[@]}" -ne "${#OUTPUT_DIRS[@]}" ]; then
    echo "❌ Error: Number of FAST5_DIRS != OUTPUT_DIRS"
    exit 1
fi

# ----------------------------------------------------------------------------
# 信号处理核心参数（与 Fast5Dir.to_chunks() 完全对齐）
# 参考: nanopore_signal_tokenizer/utils/fast5.py
# ----------------------------------------------------------------------------

# 【Chunking】每个信号片段的长度（单位：采样点）
# 华大 DNA 测序默认采样率 5000 Hz，12000 点 ≈ 2.4 秒信号
WINDOW_SIZE=12000

# 【Chunking】滑动窗口步长（单位：采样点）
# STRIDE = WINDOW_SIZE - overlap；此处 overlap = 60 点（≈12ms）
STRIDE=11940

# 【归一化】是否启用 median-MAD 归一化（强烈建议开启）
DO_NORMALIZE="true"

# 【滤波】是否启用中值滤波（kernel=5），用于去除 spike 噪声
# 注意：fast5.py 中 medfilt 是硬编码应用的，但 CLI 仍保留开关（未来可能解耦）
DO_MEDIANFILTER="true"

# 【滤波】是否启用 Butterworth 低通滤波（默认关闭，因可能引入相位失真）
DO_LOWPASSFILTER="false"

# 【设备】默认采样率（Hz），当 FAST5 metadata 缺失时使用
# 华大测序仪标准采样率：5000 Hz
DEFAULT_FS=5000

# 【性能】并行进程数（-1 表示全部 CPU，此处显式设为 64）
# 建议 ≤ 物理核心数，避免 I/O 瓶颈
N_JOBS=64

# 【多头裁剪】最大开头裁剪长度（inclusive），用于生成 stride 对齐的多相位输入
# 例如：若下游 CNN 下采样 stride=4，则 cut_head_all=3 可覆盖 0,1,2,3 四种起始偏移
CUT_HEAD_ALL=3

# 【多头裁剪】裁剪步长，控制相位密度
# step=2 → 生成 head_cut = 0, 2（共 2 种相位）
CUT_HEAD_STEP=2

# 【末尾兜底】触发末尾补 chunk 的最小剩余长度（单位：采样点）
# 若滑动结束后剩余 ≥ TAIL_THRESHOLD，则从末尾强制切出一个完整 window
# 设置过小会导致重复 chunk，过大则浪费尾部信号
TAIL_THRESHOLD=150

# ----------------------------------------------------------------------------
# 信号范围裁剪参数（用于异常值过滤）
# 参考华大工程师反馈：DNA 开孔电流稳定，典型范围 [90, 220] pA
# 但为鲁棒性，放宽至 [1, 220] pA
# ----------------------------------------------------------------------------

# 【原始信号】低于此值（pA）视为异常（如 gating 卡顿）
# 华大建议：DNA 不写死开孔电流，但可设下限为 1 pA（避免负值/零值）
SIGNAL_MIN_VALUE=1

# 【原始信号】高于此值（pA）视为异常（如电极噪声）
# 华大 RNA/DNA 开孔电流典型上限为 220 pA
SIGNAL_MAX_VALUE=220

# ----------------------------------------------------------------------------
# 归一化后信号范围（用于质量控制）
# 归一化后信号理论上应落在 [-5, 5] 或 [-9, 9] 内
# 若超出，说明存在未修复的极端异常，该 read 将被跳过
# ----------------------------------------------------------------------------

# 【归一化信号】允许的最小值（MAD 单位）
NORMAL_MIN_VALUE=-9

# 【归一化信号】允许的最大值（MAD 单位）
NORMAL_MAX_VALUE=9

# ----------------------------------------------------------------------------
# 构建通用参数（转换布尔值为 CLI 参数）
# ----------------------------------------------------------------------------

if [ "$DO_NORMALIZE" = "true" ]; then
    NORMALIZE_ARG="--do_normalize"
else
    NORMALIZE_ARG="--no_normalize"
fi

MEDIANFILTER_ARG=""
[ "$DO_MEDIANFILTER" = "true" ] && MEDIANFILTER_ARG="--do_medianfilter"

LOWPASSFILTER_ARG=""
[ "$DO_LOWPASSFILTER" = "true" ] && LOWPASSFILTER_ARG="--do_lowpassfilter"

# ----------------------------------------------------------------------------
# 循环处理每一对 (fast5_dir, output_dir)
# ----------------------------------------------------------------------------
for i in "${!FAST5_DIRS[@]}"; do
    FAST5_DIR="${FAST5_DIRS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

    if [ ! -d "$FAST5_DIR" ]; then
        echo "⚠️  Warning: FAST5 directory not found: $FAST5_DIR. Skipping..."
        continue
    fi

    echo "================================================================"
    echo "🚀 Processing: $FAST5_DIR"
    echo "📁 Output to:  $OUTPUT_DIR"
    echo "================================================================"

    mkdir -p "$OUTPUT_DIR"

    # 调用全局命令（由 pyproject.toml [project.scripts] 定义）
    # 实际调用: nanopore_llm_workflow.cli.fast5.fast5_to_chunks:main
    CMD=(
        nanopore_llm_workflow-fast5-to-chunks  # ← 注意：命令名是 fast5-to-chunks，不是 nanopore_llm_workflow-fast5-to-chunks
        --fast5_dir "$FAST5_DIR"
        --output_dir "$OUTPUT_DIR"
        --window_size "$WINDOW_SIZE"
        --stride "$STRIDE"
        $NORMALIZE_ARG
        $MEDIANFILTER_ARG
        $LOWPASSFILTER_ARG
        --default_fs "$DEFAULT_FS"
        --n_jobs "$N_JOBS"
        --cut_head_all "$CUT_HEAD_ALL"
        --cut_head_step "$CUT_HEAD_STEP"
        --tail_threshold "$TAIL_THRESHOLD"
        --signal_min_value "$SIGNAL_MIN_VALUE"
        --signal_max_value "$SIGNAL_MAX_VALUE"
        --normal_min_value "$NORMAL_MIN_VALUE"
        --normal_max_value "$NORMAL_MAX_VALUE"
    )

    echo ">>> Running command:"
    printf "%q " "${CMD[@]}"
    echo
    echo "--------------------------------------------------"

    "${CMD[@]}"

    if [ $? -eq 0 ]; then
        echo "✅ Success: $FAST5_DIR → $OUTPUT_DIR"
    else
        echo "❌ Failed: $FAST5_DIR"
        exit 1
    fi
    echo
done

echo "🎉 All preprocessing completed."
