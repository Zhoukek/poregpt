#!/bin/bash

MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_20m_DNA_VQE64K_CNN08_V84S286000/base"
BASE_DIR="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result"

out_dir="result_qc"
mkdir -p "${out_dir}"
mkdir -p "${out_dir}/signal_level"
mkdir -p "${out_dir}/token_level"
mkdir -p "${out_dir}/read_level"
mkdir -p "${out_dir}/base_level"
mkdir -p "${out_dir}/seq_level"

LB1="LB06"
LB2="LB07"
label1="5mC"
label2="C"
id=1

# #### read level QC

# in1="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result/${LB1}_${id}/signal_none_between_flanks.jsonl"
# python script/plot_compare_to_reference.py \
#   --input_jsonl ${in1} \
#   --reference_seq AATTTGTGAATAANTAGGTCAGCTAGCAGGCTNGATTGAGAAGTCCCTAATNTTACTAGATCTAGCGTAGNGGTAATATAAGCTATCACNACCTCGCCCATCAGAGCTNTCTCATATGCACACTAAGNTACGCTAGG \
#   --out_dir ${out_dir}/seq_level/compare_plots \
#   --min_len 120 \
#   --max_len 150


# #### signal level QC

# in1="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result_0331/${LB1}_${id}/signal_features.csv"
# in2="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result_0331/${LB2}_${id}/signal_features.csv"

# python script/compare_signal_distance.py \
# --input1 ${in1} \
# --input2 ${in2} \
# --output ${out_dir}/signal_level/signal.pdf \
# --highlight 13 32 51 70 89 108 127 \
# --label1 ${label1} \
# --label2 ${label2} \
# --metric mean std dwell_time skewness 


# #### token level QC

in1="${BASE_DIR}/${LB1}_${id}/base_tokens.jsonl"
in2="${BASE_DIR}/${LB2}_${id}/base_tokens.jsonl"


python script/compare_token_overlap.py \
--input1 ${in1} \
--input2 ${in2} \
--highlight 13 32 51 70 89 108 127 \
--label1 ${label1} \
--label2 ${label2} \
--output ${out_dir}/token_level/token_overlap


#### reads embedding level QC

python script/compare_read_embedding_distance.py \
  --group1 ${BASE_DIR}/${LB1}_${id}/base_embedding/read_emb_mean.npy\
  --group2 ${BASE_DIR}/${LB2}_${id}/base_embedding/read_emb_mean.npy\
  --name1 ${label1} \
  --name2 ${label2} \
  --outdir ${out_dir}/read_level \
  --metric euclidean \
  --n_pairs 20000 \
  --n_plot_points 1200 \
  --n_boot 100 \
  --bootstrap_pairs 10000 \
  --pca_max_points 3000


# python script/compare_read_embedding_distance.py \
#   --group1 ${BASE_DIR}/${LB1}_${id}/base_embedding_between_flanks/read_emb_mean.npy\
#   --group2 ${BASE_DIR}/${LB2}_${id}/base_embedding_between_flanks/read_emb_mean.npy\
#   --name1 ${label1} \
#   --name2 ${label2} \
#   --outdir ${out_dir}/read_level_between_flanks \
#   --metric euclidean \
#   --n_pairs 100 \
#   --n_plot_points 1200 \
#   --n_boot 100 \
#   --bootstrap_pairs 10000 \
#   --pca_max_points 3000



# #### base embedding level QC

python script/compare_base_embedding_distance.py\
  --input1 ${BASE_DIR}/${LB1}_${id}/base_embedding/base_emb_mean.npy \
  --input2 ${BASE_DIR}/${LB2}_${id}/base_embedding/base_emb_mean.npy \
  --label1 ${label1} \
  --label2 ${label2} \
  --out_prefix "${out_dir}/base_level/distance" \
  --mod_positions 13 32 51 70 89 108 127 \
  --window_size 1 \




# python script/compare_base_embedding_distance.py\
#   --input1 ${BASE_DIR}/${LB1}_${id}/base_embedding_between_flanks/base_emb_mean.npy \
#   --input2 ${BASE_DIR}/${LB2}_${id}/base_embedding_between_flanks/base_emb_mean.npy \
#   --label1 ${label1} \
#   --label2 ${label2} \
#   --out_prefix "${out_dir}/base_level_between_flanks/distance" \
#   --mod_positions 13 32 51 70 89 108 127 \
#   --window_size 1 \

