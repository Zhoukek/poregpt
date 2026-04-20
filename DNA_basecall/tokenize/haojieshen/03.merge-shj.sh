#!/bin/bash
# 先加载MACA环境
source /mnt/zzbnew/rnamodel/shenhaojie/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本
export PYTHONPATH=/mnt/zzbnew/rnamodel/shenhaojie/poregpt
# ==========================================
# ==========================================
# 03.merge-shj.sh - 合并分块结果为最终文件
# 这个脚本负责将之前步骤生成的分块文件（chunks, references, reference_lengths）合并成最终的 npy 文件，供后续分析使用。
# 重建的各个文件在 DATA_DIR_1 目录下，最终的合并结果也会输出到同一目录。
DATA_DIR_1="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/recon"
CSV_PATH_1="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/recon/processing_summary.csv"

echo "================ Starting Task 1 ================"
python 03.merge-shj.py \
    --data_dir "$DATA_DIR_1" \
    --summary_csv "$CSV_PATH_1"

