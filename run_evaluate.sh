#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0}
exec "$PY" evaluate_stereo.py \
  --restore_ckpt "${STEREO_CHECKPOINT:-checkpoints/pretrain_whusside_60ep_724/final.pth}" \
  --data_path "${DATA_ROOT:-data/WHU-SSIDE/valtest}" --dataset whusside \
  --valid_iters 16 --batch_size 1 --mixed_precision "$@"
