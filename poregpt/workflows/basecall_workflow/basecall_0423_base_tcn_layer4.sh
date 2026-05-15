#!/usr/bin/env bash
set -euo pipefail

# =========================
# 1) 国产卡 / MACA 运行时环境
# =========================
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}

export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MCPYTORCH_DISABLE_PRINT=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export OMP_NUM_THREADS=1

# =========================
# 2) MCCL / NCCL：稳定优先
#    先不要写死网卡和激进参数
# =========================
unset MCCL_SOCKET_IFNAME
unset MCCL_NET_GDR_LEVEL
unset MCCL_MAX_NCHANNELS
unset MCCL_P2P_LEVEL
unset MCCL_LIMIT_RING_LL_THREADTHRESHOLDS
unset FORCE_ACTIVATE_WAIT
unset SET_DEVICE_NUMA_PREFERRED

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH

# =========================
# 3) 从平台环境变量读取多节点信息
# =========================
# 平台注入优先，没给就退化到单节点
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${POD_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 每个节点多少卡
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# =========================
# 4) 任务参数
# =========================
model_name="HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000"
batch_size=32
num_epochs=500
lr="1e-3"
weight_decay="1e-4"
warmup_ratio="0.4"
min_lr="1e-5"
hidden_layer=-1
unfreeze_last_n_layers=4
head_type="ctc"
train_decode="ctc_viterbi"
pre_head_type="tcn"
feature_source="hidden"
head_output_activation="tanh"
head_output_scale=5

wandb_project="DNA_basecalling"
wandb_run_name="${model_name}_${head_type}_${pre_head_type}_${feature_source}_unfreeze${unfreeze_last_n_layers}_bsz${batch_size}_nnodes${NNODES}_rank${NODE_RANK}_fix1"

base_model="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${model_name}/base"
data_root="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${model_name}/basecall_subset20"
outdir="/mnt/zzbnew/rnamodel/wangxue/DNA_basecall/result/${model_name}/${wandb_run_name}"

mkdir -p "${outdir}"

# 不建议把 key 明文写在脚本里，建议外部 export
export WANDB_API_KEY=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

# =========================
# 5) 训练前打印环境，方便排障
# =========================
echo "========== DISTRIBUTED ENV =========="
echo "HOSTNAME=$(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORLD_SIZE(env)=${WORLD_SIZE:-<unset>}"
echo "RANK(env)=${RANK:-<unset>}"
echo "POD_RANK(env)=${POD_RANK:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "====================================="

python - <<'PY'
import os
import torch

print("Python-side env check:")
for k in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "POD_RANK", "LOCAL_RANK"]:
    print(f"{k} =", os.environ.get(k))

try:
    n = torch.cuda.device_count()
    print("torch.cuda.device_count =", n)
    for i in range(n):
        try:
            print(f"device {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"device {i}: <name unavailable> {e}")
except Exception as e:
    print("CUDA check failed:", e)
PY

# =========================
# 6) 启动 torchrun
# =========================
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m script.basecaller_v8_0420.train_ddp_multifolder \
  --jsonl_paths "${data_root}" \
  --model_name_or_path "${base_model}" \
  --output_dir "${outdir}" \
  --batch_size "${batch_size}" \
  --num_epochs "${num_epochs}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --warmup_ratio "${warmup_ratio}" \
  --min_lr "${min_lr}" \
  --group_by record \
  --freeze_backbone \
  --head_type "${head_type}" \
  --hidden-layer "${hidden_layer}" \
  --pre_head_type "${pre_head_type}" \
  --train_decoder "${train_decode}" \
  --unfreeze_last_n_layers "${unfreeze_last_n_layers}" \
  --feature_source "${feature_source}" \
  --head_output_activation "${head_output_activation}" \
  --head_output_scale "${head_output_scale}" \
  --ddp_backend nccl \
  --save_best \
  --use_wandb \
  --wandb_project "${wandb_project}" \
  --wandb_run_name "${wandb_run_name}" \
  --log_interval 100 \
  --num_workers 0 
