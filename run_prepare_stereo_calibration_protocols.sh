#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-python3}
DATA_ROOT=${DATA_ROOT:-data/UnifiedBH/overlap15}
CACHE_ROOT=${CACHE_ROOT:-${DATA_ROOT}/disp/sd2h_full9_abs_groundfix_whusside724_20260622}

"$PY" - "$DATA_ROOT" "$CACHE_ROOT" <<'CONFIG'
from pathlib import Path
import sys
from omegaconf import OmegaConf
p = Path('config/train_stereo_calib_base.yaml')
cfg = OmegaConf.load(p)
cfg.train_data_path = str(Path(sys.argv[1]).resolve())
cfg.val_data_path = cfg.train_data_path
cfg.precomputed_disp_delta_root = str(Path(sys.argv[2]).resolve())
OmegaConf.save(cfg, p)
CONFIG

"$PY" tools/prepare_stereo_calibration_protocols.py \
  --base_config config/train_stereo_calib_base.yaml \
  --metrics_csv "$CACHE_ROOT/instance_metrics.csv" \
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
