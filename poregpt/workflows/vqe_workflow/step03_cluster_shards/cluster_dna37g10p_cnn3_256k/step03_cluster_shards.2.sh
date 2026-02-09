
#!/bin/bash

# 参数变量
NITER=35
NREDO=5
# cluster为16k
MPPC=32768
NUM_CLUSTERS=16384

# cluster为64k
MPPC=8192
NUM_CLUSTERS=65536

# cluster为256k
MPPC=2048
NUM_CLUSTERS=262144


# 输入输出路径（可按需调整）
FEATURE_SHARDS_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train_features_cnn3"
SHARDS_JSON="shards_sampled_10p.json"
OUTPUT_PREFIX="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train_features_cnn3/cluster_10p_256k"
MAX_SAMPLED_TOKENS=2000000000


# 日志文件名包含参数
LOGFILE="step03_cluster_shards_iter${NITER}_redo${NREDO}_k${NUM_CLUSTERS}_mppc${MPPC}.out"

# 后台运行命令
nohup stdbuf -oL -eL python3 -u ../step03_cluster_shards.py \
    --feature_shards_dir "${FEATURE_SHARDS_DIR}" \
    --shards_json "${SHARDS_JSON}" \
    --output_prefix "${OUTPUT_PREFIX}" \
    --num_clusters "${NUM_CLUSTERS}" \
    --niter "${NITER}" \
    --nredo "${NREDO}" \
    --max_points_per_centroid "${MPPC}" \
    --max_sampled_tokens "${MAX_SAMPLED_TOKENS}" \
    2>&1 | stdbuf -i0 -o0 tr '\r' '\n' | tee "${LOGFILE}" &
