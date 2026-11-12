#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
GPU_IDS=${GPU_IDS:-${GPUS:-0}}
NUM_PROCS=${NUM_PROCS:-$(awk -F, '{print NF}' <<< "$GPU_IDS")}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
exec "$PY" -m accelerate.commands.launch --num_processes="$NUM_PROCS" \
  --mixed_precision=bf16 train_finetune.py --config-name "${CONFIG_NAME:-train_city_stratified_50_50}" \
  num_gpu="$NUM_PROCS" batch_size="${BATCH_SIZE:-$NUM_PROCS}" "$@"
