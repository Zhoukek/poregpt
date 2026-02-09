stdbuf -oL -eL python3 -u step03_cluster_memap.py \
    --memmap_npy_path /mnt/gpudisk/dataset/memap_dna.npy \
    --output_prefix /mnt/nas_syy/default/huada_signal_llm/dataset/dna/memap/human_min0_max2_read96655/train_features_clustered_20p \
    --num_clusters 16384 \
    --niter 10 \
    --nredo 100 \
    --max_sampled_tokens -1 \
    2>&1 | stdbuf -i0 -o0 tr '\r' '\n' | tee step03_cluster_memap.out
