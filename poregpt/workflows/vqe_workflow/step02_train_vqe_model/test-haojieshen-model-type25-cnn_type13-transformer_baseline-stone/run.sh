
# # 先加载MACA环境
# source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

# export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export WANDB_API_KEY=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
# torchrun --nproc_per_node=8 --master_port 29500 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml 2>&1 | tee run.log


# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=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
# torchrun --nproc_per_node=4 --master_port 29502 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml 2>&1 | tee run.log
# export WANDB_API_KEY=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


# 3. 后台运行训练代码并将日志写入 run.log
nohup torchrun --nproc_per_node=4 --master_port 29502 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml > run.log 2>&1 &