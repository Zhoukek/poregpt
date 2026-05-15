#!/bin/bash

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



input_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result_mod2/corpus/canonical_chunks
output_dir=/mnt/zzbnew/rnamodel/wangxue/DNA_modification/modeling/result/models/model1

mkdir -p ${output_dir}

python -m script.mod_v1.train_model1 \
  --chunks ${input_dir}/*chunks.npy \
  --out_dir ${output_dir} \
  --batch_size 4096 \
  --epochs 20 \
  --num_workers 4 \
  --pin_memory \
  --debug_first_batch \
  --max_chunks 10000000
