#!/bin/bash

result=result_0513_mod_v1
input_dir=/mnt/zzbnew/rnamodel/data/DNA_modification/250F601844011/signals
output_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/${result}/corpus/canonical_chunks

python -m script.mod_v1.ccf5_chunk \
  --input_dir ${input_dir} \
  --output_dir ${output_dir} \
  --min_raw_len 14000 \
  --trim_head 2000 \
  --trim_tail 2000 \
  --chunk_len 128 \
  --stride 64 \
  --threads 1 \
  --batchsize 16



input_dir=/mnt/zzbnew/rnamodel/data/DNA_449_110/cuihuihai/250F701586012/signals
output_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/${result}/corpus/native_chunks

python -m script.mod_v1.ccf5_chunk \
  --input_dir ${input_dir} \
  --output_dir ${output_dir} \
  --min_raw_len 14000 \
  --trim_head 2000 \
  --trim_tail 2000 \
  --chunk_len 128 \
  --stride 64 \
  --threads 1 \
  --batchsize 16



input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/result_0331/LB06_2/signal_none.jsonl
output_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/${result}/corpus/synthetic_chunks

python -m script.mod_v1.jsonl_chunk \
  --input_jsonl ${input_dir} \
  --out_dir ${output_dir} \
  --chunk_len 128 \
  --stride 64 \
  --min_len 128
