#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/home/shangying/miniconda3/envs/sd2h/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/shangying/workspace/dataset/UnifiedBH/overlap15}
OUT_ROOT=${OUT_ROOT:-${DATA_ROOT}/output}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

COMMON_ARGS=(
  --data_path "$DATA_ROOT"
  --device cuda:0
  --batch_size "${PLOT_BATCH_SIZE:-8}"
  --num_workers "${PLOT_NUM_WORKERS:-8}"
  --valid_iters 0
  --mixed_precision
  --boxplot_ymax 20
)

plot_protocol() {
  local config_name="$1"
  local ckpt_dir="$2"
  local out_dir="$3"
  local prefix="$4"
  local grid_mode="$5"
  local box_color_mode="$6"
  shift 6
  local train_cities=()
  local val_cities=()
  local mode="train"
  for token in "$@"; do
    if [[ "$token" == "--" ]]; then
      mode="val"
      continue
    fi
    if [[ "$mode" == "train" ]]; then
      train_cities+=("$token")
    else
      val_cities+=("$token")
    fi
  done

  local ckpt="${ckpt_dir}/finetune_best_rmse.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] Missing checkpoint: $ckpt"
    return
  fi

  mkdir -p "$out_dir"
  "$PY" plotting/make_unifiedbh_figures.py \
    --config_yaml "config/${config_name}.yaml" \
    --restore_ckpt "$ckpt" \
    --output_directory "$out_dir" \
    --figure_prefix "$prefix" \
    --train_cities "${train_cities[@]}" \
    --val_cities "${val_cities[@]}" \
    --scatter_grid_mode "$grid_mode" \
    --boxplot_color_mode "$box_color_mode" \
    "${COMMON_ARGS[@]}"
}

plot_protocol \
  train_quality_diverse_cross_city \
  checkpoints/quality_diverse_cross_city_hz_dg_058_sz_dg_038_to_7cities \
  "${OUT_ROOT}/quality_diverse_cross_city" \
  quality_diverse_cross_city \
  overall_plus_cities \
  validation \
  HZ_DG_058 SZ_DG_038 -- HZ_049 HZ_050 HZ_057 SZ_060 SZ_061 HK_062 PRE_039

for tag in 10_90 20_80 50_50; do
  plot_protocol \
    "train_city_stratified_${tag}" \
    "checkpoints/city_stratified_${tag}" \
    "${OUT_ROOT}/city_stratified_${tag}" \
    "city_stratified_${tag}" \
    cities_only \
    train \
    HZ_049 HZ_050 HZ_057 HZ_DG_058 SZ_060 SZ_061 HK_062 SZ_DG_038 PRE_039 -- \
    HZ_049 HZ_050 HZ_057 HZ_DG_058 SZ_060 SZ_061 HK_062 SZ_DG_038 PRE_039
done

echo "[done] Figures written under ${OUT_ROOT}"
