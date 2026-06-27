#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/home/shangying/miniconda3/envs/sd2h/bin/python}
GPU_IDS=${GPU_IDS:-3,4}
NUM_PROCS=${NUM_PROCS:-2}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_train() {
  local config_name="$1"
  local save_dir="$2"
  mkdir -p "$save_dir"
  echo "[$(date '+%F %T')] START $config_name on GPUs ${GPU_IDS}"
  "$PY" -m accelerate.commands.launch \
    --num_processes="$NUM_PROCS" \
    --main_process_port="${MAIN_PROCESS_PORT:-29641}" \
    train_finetune.py --config-name "$config_name" > "${save_dir}/train.log" 2>&1
  echo "[$(date '+%F %T')] END $config_name"
}

# Run ./run_prepare_stereo_calibration_protocols.sh before this script.
run_train train_quality_diverse_cross_city checkpoints/quality_diverse_cross_city_hz_dg_058_sz_dg_038_to_7cities
run_train train_city_stratified_10_90 checkpoints/city_stratified_10_90
run_train train_city_stratified_20_80 checkpoints/city_stratified_20_80
run_train train_city_stratified_50_50 checkpoints/city_stratified_50_50
