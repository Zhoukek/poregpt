
# 先加载MACA环境
source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本


export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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
torchrun --nproc_per_node=2 --master_port 29501 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml 2>&1 | tee run.log

