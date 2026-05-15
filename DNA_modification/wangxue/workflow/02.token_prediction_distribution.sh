source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/olmo_workflow/OLMo:/mnt/zzbnew/rnamodel/zhoukexuan/poregpt:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=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


torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29505 \
    scripts/02.token_prediction_distribution.py
