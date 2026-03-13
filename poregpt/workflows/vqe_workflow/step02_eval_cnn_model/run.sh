python3 cnn_eval_single_file.py \
    --input_npy_file /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train/chunk_batch_000000.npy \
    --checkpoint_path /mnt/nas_syy/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step00_train_cnn_model/train_cnn6/models/nanopore_cnn6.epoch128.pth \
    --output_dir . \
    --shard_size 10000000 \
    --batch_size 4096 \
    --cnn_type 6 \
    --feature_dim 128 \
    --device cuda
