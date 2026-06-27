#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= Runtime configuration =================
GPUS="4,5,6,7"             
NUM_PROCS=4          
PY=${PY:-python3}
# ==========================================

export CUDA_VISIBLE_DEVICES=$GPUS

# Launch training.
accelerate launch \
    --num_machines=1 \
    --dynamo_backend="no" \
    --num_processes=$NUM_PROCS \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/train_pretrain.py"
