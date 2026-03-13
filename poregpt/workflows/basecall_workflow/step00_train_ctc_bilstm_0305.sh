#!/bin/bash 

export PYTHONPATH=/mnt/si003067jezr/default/poregpt:$PYTHONPATH

model_name="HF_20m_DNA_VQE64K_CNN03_V20260203"
nproc_per_node=1
batch_size=64
num_epochs=20
lr="1e-3"
weight_decay="1e-4"
warmup_ratio="0.4"
min_lr="1e-5"
hidden_layer=-1
unfreeze_last_n_layers=4
head_type="ctc"
train_decode="ctc_viterbi"
pre_head_type="bilstm"
feature_source="hidden"
head_output_activation="tanh"
head_output_scale=5



wandb_project="basecall"
wandb_run_name="dna37g_${model_name}_unfreeze${unfreeze_last_n_layers}_bsz${batch_size}"

base_model="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${model_name}/base"
data_root="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${model_name}/basecall"
outdir="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${model_name}/${wandb_run_name}"

mkdir -p "${outdir}"


export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354


torchrun --nproc_per_node="${nproc_per_node}" --nnodes=1 -m poregpt.basecall.train_ddp_multifolder \
  --jsonl_paths  ${data_root} \
  --model_name_or_path ${base_model} \
  --output_dir "${outdir}" \
  --batch_size "${batch_size}" \
  --num_epochs "${num_epochs}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --warmup_ratio "${warmup_ratio}" \
  --min_lr "${min_lr}" \
  --group_by file \
  --find_unused_parameters \
  --freeze_backbone \
  --head_type "${head_type}" \
  --hidden-layer "${hidden_layer}" \
  --pre_head_type "${pre_head_type}" \
  --train_decoder "${train_decode}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --feature_source "${feature_source}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --save_best \
  --use_wandb \
  --wandb_project ${wandb_project} \
  --wandb_run_name ${wandb_run_name} 
