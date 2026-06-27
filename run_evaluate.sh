#!/bin/bash

# ================= Hardware and environment =================
export CUDA_VISIBLE_DEVICES=7

# ================= Task paths =================
CKPT_PATH="./checkpoints/pretrain_whusside_60ep_724/final.pth"
DATA_PATH="/home/shangying/workspace/dataset/WHU-SSIDE/valtest" 
DATASET="whusside"

# ================= Inference hyperparameters =================
BATCH_SIZE=1
ITERS=16

# ================= Saving options =================
#SAVE_VIS="--save_vis" 
SAVE_VIS=" "   # 

# ================= Run evaluation =================
echo -e "\n====================================================================="
echo "[*] Evaluating checkpoint: $CKPT_PATH"
echo "====================================================================="

python evaluate_stereo.py \
    --restore_ckpt "$CKPT_PATH" \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --valid_iters $ITERS \
    --batch_size $BATCH_SIZE \
    --mixed_precision \
    $SAVE_VIS

echo -e "\n[*] Evaluation finished."
