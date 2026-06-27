#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/home/shangying/miniconda3/envs/sd2h/bin/python}

"$PY" tools/prepare_stereo_calibration_protocols.py \
  --base_config config/train_stereo_calib_base.yaml \
  --metrics_csv /home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622/instance_metrics.csv \
  --base_filter_manifest checkpoints/stereo_calib_noise_filter_manifest/instance_filter_manifest.json \
  --output_root checkpoints/stereo_calibration_protocols_seed724 \
  --config_dir config \
  --seed 724 \
  --fewshot_ratios 0.10 0.20 0.50 \
  --fewshot_high_min_train 20 \
  --fewshot_high_min_fraction 0.20 \
  --cross_city_train_cities HZ_DG_058 SZ_DG_038 \
  --batch_size 1 \
  --val_batch_size 1 \
  --num_gpu 1 \
  --write_pooled_fewshot_config
