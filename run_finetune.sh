#!/usr/bin/env bash

# ================= Environment variables =================
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled  #offline 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ================= Runtime configuration =================
GPUS="2,3,5,6"          
NUM_PROCS=4         
# ==============================================

export CUDA_VISIBLE_DEVICES=$GPUS

echo "==========================================================="
echo "[*] Starting height finetuning..."
echo "[*] GPUs: $GPUS"
echo "[*] Number of processes: $NUM_PROCS"
echo "==========================================================="

# Launch training.
accelerate launch \
    --num_machines=1 \
    --dynamo_backend="no" \
    --num_processes=$NUM_PROCS \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/train_finetune.py" "$@"
