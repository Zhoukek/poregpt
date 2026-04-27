#!/bin/bash


# name="lemon_signal"

train_data=/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_baseline_model_VQ_8k-lemon/shuffle_npy
epochs=5
train_chunks=500000
lr=0.002
inmodel=/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121/config.toml
outmodel=/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/subtask/tokenize/model-type26-cnn_type13_baseline_model_VQ_8k-lemon

# bonito train --config ${inmodel} --directory ${train_data} --lr ${lr} --epochs ${epochs} -f --chunks ${train_chunks} ${outmodel} --batch 96 --device cuda:0 

# 使用两张GPU，DataParallel或DistributedDataParallel
CUDA_VISIBLE_DEVICES=0,1 nohup bonito train \
    --config ${inmodel} \
    --directory ${train_data} \
    --lr ${lr} \
    --epochs ${epochs} \
    -f \
    --chunks ${train_chunks} \
    ${outmodel} \
    --batch 96 \
    --device cuda:0 \
    > ${outmodel}/train.log 2>&1 &