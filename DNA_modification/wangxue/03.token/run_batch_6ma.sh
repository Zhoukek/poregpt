#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic settings
# =========================
PATTERN_FILE="./data/6ma_mask.txt"

STRIDE=5
TARGET_STRAND=0
PAD_SAMPLES=0

TOKEN_STRIDE=5
DEVICE="cuda"

Group_A="LB08"
Group_B="LB09"



TOKENIZER_CKPT="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_20m_DNA_VQE64K_CNN08_V84S286000/encoder"
MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_20m_DNA_VQE64K_CNN08_V84S286000/base"

# Group A
JSONL_A="./data/movetable/${Group_A}_test/data.jsonl"
H5_A="./data/movetable/${Group_A}_test/signal.h5"

# Group B
JSONL_B="./data/movetable/${Group_B}_test/data.jsonl"
H5_B="./data/movetable/${Group_B}_test/signal.h5"

# =========================
# check
# =========================
if [[ ! -f "${PATTERN_FILE}" ]]; then
  echo "ERROR: pattern file not found: ${PATTERN_FILE}"
  exit 1
fi

mkdir -p result

# =========================
# loop over patterns
# =========================
number=0

while IFS= read -r PATTERN || [[ -n "${PATTERN}" ]]; do
  # 去掉首尾空白
  PATTERN="$(echo "${PATTERN}" | tr -d '\r' | xargs)"

  # 跳过空行和注释
  [[ -z "${PATTERN}" ]] && continue
  [[ "${PATTERN}" =~ ^# ]] && continue

  number=$((number + 1))

  OUT_A="./result/${Group_A}_${number}"
  OUT_B="./result/${Group_B}_${number}"

  mkdir -p "${OUT_A}/png" "${OUT_B}/png"

  echo "=================================================="
  echo "Processing pattern #${number}"
  echo "PATTERN=${PATTERN}"
  echo "OUT_A=${OUT_A}"
  echo "OUT_B=${OUT_B}"
  echo "=================================================="


 #### Step1: 原始信号水平分析
 normal="none"

  python3 script/01_extract_motif_signal.py \
    --jsonl_path "${JSONL_A}" \
    --h5_path "${H5_A}" \
    --pattern "${PATTERN}" \
    --stride "${STRIDE}" \
    --pad_samples "${PAD_SAMPLES}" \
    --max_occurrences 1000 \
    --batch_size 100 \
    --out_dir "${OUT_A}/png" \
    --out_txt "${OUT_A}/signal_none.txt" \
    --out_jsonl "${OUT_A}/signal_none.jsonl" \
    --normalize_mode "${normal}" \
    --write_window_signal \
    --write_debug_meta \
    --mark_base_pos 13 32 51 70 89 108 127 \
    --y_lim_mode manual \
    --y_lim -3 3

  python3 script/01_extract_motif_signal.py \
    --jsonl_path "${JSONL_B}" \
    --h5_path "${H5_B}" \
    --pattern "${PATTERN}" \
    --stride "${STRIDE}" \
    --pad_samples "${PAD_SAMPLES}" \
    --max_occurrences 1000 \
    --batch_size 100 \
    --out_dir "${OUT_B}/png" \
    --out_txt "${OUT_B}/signal_none.txt" \
    --out_jsonl "${OUT_B}/signal_none.jsonl" \
    --normalize_mode "${normal}" \
    --write_window_signal \
    --write_debug_meta \
    --mark_base_pos 13 32 51 70 89 108 127 \
    --y_lim_mode manual \
    --y_lim -3 3


  # python3 script/01_extract_between_flanks.py \
  #   --jsonl_path "${JSONL_A}" \
  #   --h5_path "${H5_A}" \
  #   --left_flank ACCTGCCGGAAAGGCCG \
  #   --right_flank GATCCGGAACAGTTTTT \
  #   --pair_mode nearest \
  #   --stride "${STRIDE}" \
  #   --pad_samples "${PAD_SAMPLES}" \
  #   --batch_size 100 \
  #   --out_dir "${OUT_A}/png" \
  #   --out_txt "${OUT_A}/signal_none_between_flanks.txt" \
  #   --out_jsonl "${OUT_A}/signal_none_between_flanks.jsonl" \
  #   --normalize_mode "${normal}" \
  #   --write_window_signal \
  #   --write_debug_meta \
  #   --y_lim_mode manual \
  #   --y_lim -3 3

  # python3 script/01_extract_between_flanks.py \
  #   --jsonl_path "${JSONL_B}" \
  #   --h5_path "${H5_B}" \
  #   --left_flank  ACCTGCCGGAAAGGCCG \
  #   --right_flank GATCCGGAACAGTTTTT \
  #   --pair_mode nearest \
  #   --stride "${STRIDE}" \
  #   --pad_samples "${PAD_SAMPLES}" \
  #   --batch_size 100 \
  #   --out_dir "${OUT_B}/png" \
  #   --out_txt "${OUT_B}/signal_none_between_flanks.txt" \
  #   --out_jsonl "${OUT_B}/signal_none_between_flanks.jsonl" \
  #   --normalize_mode "${normal}" \
  #   --write_window_signal \
  #   --write_debug_meta \
  #   --y_lim_mode manual \
  #   --y_lim -3 3



#  #### Step2: 提取每个碱基的信号均值
#  python3 script/signal_features.py \
#    --in_jsonl "${OUT_A}/signal_none.jsonl" \
#    --out_jsonl "${OUT_A}/signal_features.jsonl" \
#    --out_csv "${OUT_A}/signal_features.csv"

#  python3 script/signal_features.py \
#    --in_jsonl "${OUT_B}/signal_none.jsonl" \
#    --out_jsonl "${OUT_B}/signal_features.jsonl" \
#    --out_csv "${OUT_B}/signal_features.csv"


#######################################################################################
# #### Step1: 提取目标序列信号

#    normal="lemon"
#    python3 script/01_extract_motif_signal.py \
#      --jsonl_path "${JSONL_A}" \
#      --h5_path "${H5_A}" \
#      --pattern "${PATTERN}" \
#      --stride "${STRIDE}" \
#      --pad_samples "${PAD_SAMPLES}" \
#      --max_occurrences 1000 \
#      --batch_size 100 \
#      --out_dir "${OUT_A}/png" \
#      --out_txt "${OUT_A}/signal.txt" \
#      --out_jsonl "${OUT_A}/signal.jsonl" \
#      --normalize_mode "${normal}" \
#      --write_window_signal \
#      --write_debug_meta \
#      --mark_base_pos 13 32 51 70 89 108 127 \
#      --y_lim_mode manual \
#      --y_lim -3 3 \
#      --unify_motif_orientation

#    python3 script/01_extract_motif_signal.py \
#      --jsonl_path "${JSONL_B}" \
#      --h5_path "${H5_B}" \
#      --pattern "${PATTERN}" \
#      --stride "${STRIDE}" \
#      --pad_samples "${PAD_SAMPLES}" \
#      --max_occurrences 1000 \
#      --batch_size 100 \
#      --out_dir "${OUT_B}/png" \
#      --out_txt "${OUT_B}/signal.txt" \
#      --out_jsonl "${OUT_B}/signal.jsonl" \
#      --normalize_mode "${normal}" \
#      --write_window_signal \
#      --write_debug_meta \
#      --mark_base_pos 13 32 51 70 89 108 127 \
#      --y_lim_mode manual \
#      --y_lim -3 3 \
#      --unify_motif_orientation


  # #### Step2: tokenize
  # python3 script/02_tokenize_base_tokens.py \
  #   --input_jsonl "${OUT_A}/signal.jsonl" \
  #   --output_jsonl "${OUT_A}/base_tokens.jsonl" \
  #   --model_ckpt "${TOKENIZER_CKPT}" \
  #   --device "${DEVICE}" \
  #   --map_mode stride --token_stride "${TOKEN_STRIDE}" --ensure_non_empty

  # python3 script/02_tokenize_base_tokens.py \
  #   --input_jsonl "${OUT_B}/signal.jsonl" \
  #   --output_jsonl "${OUT_B}/base_tokens.jsonl" \
  #   --model_ckpt "${TOKENIZER_CKPT}" \
  #   --device "${DEVICE}" \
  #   --map_mode stride --token_stride "${TOKEN_STRIDE}" --ensure_non_empty

  # #### Step3.1: embedding per base
  # python3 script/03_embed_per_base.py \
  #   --model_path "${MODEL_PATH}" \
  #   --input_jsonl "${OUT_A}/base_tokens.jsonl" \
  #   --out_dir "${OUT_A}/base_embedding" \
  #   --label "${PATTERN}" \
  #   --pool mean

  # python3 script/03_embed_per_base.py \
  #   --model_path "${MODEL_PATH}" \
  #   --input_jsonl "${OUT_B}/base_tokens.jsonl" \
  #   --out_dir "${OUT_B}/base_embedding" \
  #   --label "${PATTERN}" \
  #   --pool mean


  #  ### Step3.2: embedding per read
  # python3 script/03_embed_per_read.py \
  #   --model_path "${MODEL_PATH}" \
  #   --input_jsonl "${OUT_A}/base_tokens.jsonl" \
  #   --out_dir "${OUT_A}/base_embedding" \
  #   --label "${PATTERN}" \
  #   --pool mean

  # python3 script/03_embed_per_read.py \
  #   --model_path "${MODEL_PATH}" \
  #   --input_jsonl "${OUT_B}/base_tokens.jsonl" \
  #   --out_dir "${OUT_B}/base_embedding" \
  #   --label "${PATTERN}" \
  #   --pool mean

#   #### Step4: compare
#   python3 script/04_compare_two_groups.py \
#     --emb_a "${OUT_A}/base_embedding/${PATTERN}_base_emb_mean.npy" \
#     --emb_b "${OUT_B}/base_embedding/${PATTERN}_base_emb_mean.npy" \
#     --label_a "${Group_A}" \
#     --label_b "${Group_B}" \
#     --pattern "${PATTERN}" \
#     --out_dir "./result/compare_${number}"

done < "${PATTERN_FILE}"

echo "All patterns finished."
