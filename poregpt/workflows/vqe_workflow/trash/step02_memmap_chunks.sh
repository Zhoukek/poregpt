nohup python3 -u step02_memmap_chunks.py \
    --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/chunk_apple/train \
    --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/memap_apple/train \
    --chunk_size 12000 \
    --dtype float32 \
    --num_workers 32 \
    &> step02_memmap_chunks.out &

nohup python3 -u step02_memmap_chunks.py \
    --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/chunk_apple/test \
    --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/memap_apple/test \
    --chunk_size 12000 \
    --dtype float32 \
    --num_workers 32 \
    &>> step02_memmap_chunks.out &

nohup python3 -u step02_memmap_chunks.py \
    --input_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/chunk_apple/validation \
    --output_dir /mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/memap_apple/validation \
    --chunk_size 12000 \
    --dtype float32 \
    --num_workers 32 \
    &>> step02_memmap_chunks.out &
