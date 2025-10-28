#!/usr/bin/env bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0,1,2,7       #4,5,6,7     0,1,2,3

accelerate launch --num_processes 4  --mixed_precision bf16 /home/shangying/workspace/project/SDH-Net/train_whustereo.py