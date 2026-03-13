export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
#CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port 29509 -m poregpt.tokenizers.vqe_tokenizer.cnn_train --config config_train_cnn3.yaml 
torchrun --nproc_per_node=4 --master_port 29501 -m poregpt.tokenizers.vqe_tokenizer.cnn_train --config config_a40.yaml 
