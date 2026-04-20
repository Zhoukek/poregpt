#!/bin/bash
source /mnt/zzbnew/rnamodel/shenhaojie/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本
export PYTHONPATH=/mnt/zzbnew/rnamodel/shenhaojie/poregpt
# ==========================================
# 只需要在这里修改你的路径参数！
# ==========================================
INPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/basecall"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/recon"

# 模型权重如果固定不变，这行就不需要动
MODEL_CKPT="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/encoder"

# ==========================================
# 运行 Python 脚本
# ==========================================
echo "========================================"
echo "开始数据处理..."
echo "Input Dir:  $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

python 03.recon_signal-shj.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_ckpt "$MODEL_CKPT"