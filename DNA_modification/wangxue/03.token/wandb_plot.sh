#!/bin/bash

export WANDB_API_KEY=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

img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/plot/plots"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/token/figure/signal/5mc/"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/model/model1_out/eval"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/model/model1_out/eval/random_triplets"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/synthetic_model2_out/synthetic_model2_state_mean_waveforms"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/synthetic_read_vis"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/model2_out"

img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/model2_out/native_state_mean_waveforms"
img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/synthetic_read_vis/"
# img_dir="/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/synthetic_read_vis/per_state_chunk_examples"
python script/wandb_plot.py --img_dir ${img_dir}
