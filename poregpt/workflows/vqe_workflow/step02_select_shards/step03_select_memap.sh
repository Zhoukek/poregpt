# 在原目录下生成一个 50% 采样的 shards 文件
python3 step03_select_memap.py \
    --feature_shards_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train_features_cnn6 \
    --sample_ratio 0.1 \
    --seed 42
