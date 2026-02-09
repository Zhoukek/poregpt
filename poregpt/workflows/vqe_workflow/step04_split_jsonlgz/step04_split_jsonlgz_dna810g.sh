# 或指定进程数（例如 8）
# 或指定进程数（例如 8）
python3 -u step04_split_jsonlgz.py \
  --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/dna814g/jsonlgz_vqe_pass25_c256k_cnn3_step22500/chunk8k \
  --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/dna814g/jsonlgz_vqe_pass25_c256k_cnn3_step22500/split1280_overlap640 \
  --workers 64 \
  --min_chunk_token_count 512 \
  --chunk_window_size 1280 \
  --chunk_overlap_size 640 


