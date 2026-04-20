# 先加载MACA环境
source /mnt/zzbnew/rnamodel/shenhaojie/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/shenhaojie/poregpt


# ==========================================
# 只需要在这里修改你的输入和输出路径！
# ==========================================
INPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/recon"
OUTPUT_DIR="/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-kexuanzhou-model-type25-cnn_type13-transfomrmer-teacher_model-distill1-mongo/shuffle_npy"

# ==========================================
# 运行 Python 脚本
# ==========================================
echo "========================================"
echo "开始数据打乱 (Shuffle)..."
echo "Input Dir:  $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

python 02.shuffle-shj.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
