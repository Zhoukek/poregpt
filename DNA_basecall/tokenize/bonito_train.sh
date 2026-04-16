#!/bin/bash


# name="lemon_signal"

train_data=/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-zhoukexuan-model-type25-cnn_type13-transformer-baseline/shuffle_npy

epochs=5
train_chunks=500000
lr=0.002
inmodel=/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121/config.toml
outmodel=/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/subtask/tokenize/train_model_cnn_type13_distill_0.1

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
    --device cuda:1 \
    > ${outmodel}/train.log 2>&1 &