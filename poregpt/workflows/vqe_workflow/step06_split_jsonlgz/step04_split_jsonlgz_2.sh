# 或指定进程数（例如 8）
# 或指定进程数（例如 8）
python3 -u step04_split_jsonlgz.py \
  --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500/test \
  --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500_split1280_overlap1024/test \
  --workers 64 \
  --min_chunk_token_count 512 \
  --chunk_window_size 1280 \
  --chunk_overlap_size 1024 

python3 -u step04_split_jsonlgz.py \
  --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500/train \
  --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500_split1280_overlap1024/train \
  --workers 64 \
  --min_chunk_token_count 512 \
  --chunk_window_size 1280 \
  --chunk_overlap_size 1024 

python3 -u step04_split_jsonlgz.py \
  --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500/validation \
  --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/jsonlgz_vqe_model_pass25_c64k_gstep57500_split1280_overlap1024/validation \
  --workers 64 \
  --min_chunk_token_count 512 \
  --chunk_window_size 1280 \
  --chunk_overlap_size 1024 


