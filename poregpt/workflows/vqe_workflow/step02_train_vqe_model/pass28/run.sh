#export CUDA_VISIBLE_DEVICES=1,2
export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
torchrun --nproc_per_node=4 --master_port 29501 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config_c64k_scratch.yaml
