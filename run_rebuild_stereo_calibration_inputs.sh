#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/home/shangying/miniconda3/envs/sd2h/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/shangying/workspace/dataset/UnifiedBH/overlap15}
CACHE_ROOT=${CACHE_ROOT:-${DATA_ROOT}/disp/sd2h_full9_abs_groundfix_whusside724_20260622}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CITIES=(
  HZ_049 HZ_050 HZ_057 HZ_DG_058
  SZ_060 SZ_061 HK_062 SZ_DG_038 PRE_039
)

echo "[1/4] Sync cached disp/delta/ground files with current left/right/label tiles"
"$PY" tools/sync_disp_cache_to_dataset.py \
  --data_root "$DATA_ROOT" \
  --cache_root "$CACHE_ROOT" \
  --cities "${CITIES[@]}" \
  --strict_counts \
  --apply

echo "[2/4] Rebuild label-dependent delta maps and instance_metrics.csv from cached disparity"
"$PY" tools/rebuild_unifiedbh_delta_metrics_from_disp_cache.py \
  --data_root "$DATA_ROOT" \
  --cache_root "$CACHE_ROOT" \
  --cities "${CITIES[@]}" \
  --label_min_height 3.0 \
  --min_instance_pixels 16 \
  --num_workers "${REBUILD_NUM_WORKERS:-16}" \
  --overwrite

echo "[3/4] Rebuild conservative noisy-instance filter manifest"
rm -rf checkpoints/stereo_calib_noise_filter_manifest
"$PY" tools/build_instance_filter_manifest_from_metrics.py \
  --metrics_csv "$CACHE_ROOT/instance_metrics.csv" \
  --cities "${CITIES[@]}" \
  --out_dir checkpoints/stereo_calib_noise_filter_manifest \
  --seed 724 \
  --min_label_height 3.0 \
  --min_physical_height 0.0 \
  --filter_ratios_by_bin 0.06 0.06 0.06 0.05 0.04 0.03 0.03 0.03 \
  --min_keep_per_city_bin 30

echo "[4/4] Generate leakage-safe training protocols"
./run_prepare_stereo_calibration_protocols.sh

echo "[done] Rebuilt calibration inputs and configs."
