#export CUDA_VISIBLE_DEVICES=1,2
export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
export MCCL_BLOCKING_WAIT=1 
export MCCL_TIMEOUT=1800
torchrun --nproc_per_node=8 --master_port 29501 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml 2>&1 | tee run.log
