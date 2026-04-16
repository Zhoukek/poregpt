#!/bin/bash

name="stone_signal"

train_data=/mnt/zzbnew/rnamodel/wangxue/DNA_basecall//tokenize/data/${name}/split_npy
out_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_basecall//tokenize/result/eval/${name}
inmodel=/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121
# inmodel=/mnt/zzbnew/rnamodel/wangxue/DNA_basecall//tokenize/result/train_model/lemon_signal


bonito evaluate  --directory ${train_data} --output_dir ${out_dir} --batch 128 --chunks 4000 --device cuda:1 ${inmodel} --weights 5
