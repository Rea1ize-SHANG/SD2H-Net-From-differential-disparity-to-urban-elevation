# Reproduction guide

This guide follows the repository's stereo pre-training, cached-disparity preparation, height-adaptation, evaluation, and plotting entry points. Run commands from the repository root.

## 1. Prepare the environment

Follow the [Environment](../README.md#environment) section to create the Python environment and install PyTorch and the project dependencies. Activate that environment before running the commands below.

```bash
python3 reproduction/run_pipeline.py --help
```

## 2. Set dataset and checkpoint paths

Download WHU-SSIDE and UnifiedBH using the [Data Preparation](../README.md#data-preparation) instructions. Place the DepthAnythingV2 weights under `pretrained/` and the stereo/height checkpoints under `checkpoints/`.

Set the paths for your local installation:

```bash
export WHUSSIDE_ROOT=/path/to/WHU-SSIDE
export DATA_ROOT=/path/to/UnifiedBH/overlap15
export CACHE_ROOT="$DATA_ROOT/disp/sd2h_full9_abs_groundfix_whusside724_20260622"
export STEREO_CKPT=checkpoints/pretrain_whusside_60ep_724/final.pth
export HEIGHT_CKPT=checkpoints/city_stratified_50_50/finetune_best_rmse.pth
export OUT_ROOT=outputs/reproduction
```

In `config/train_pretrain.yaml`, set `train_data_path` and `val_data_path` to the WHU-SSIDE training and validation folders. In the chosen height config, set `train_data_path`, `val_data_path`, `precomputed_disp_delta_root`, and `instance_filter_manifest` to the corresponding local paths. Keep each protocol's training and validation manifests with its own checkpoint.

## 3. Train the stereo backbone

```bash
python3 reproduction/run_pipeline.py --stage pretrain --num-processes 1 --gpu 0
```

This invokes `train_pretrain.py` through Accelerate. The training settings are read from `config/train_pretrain.yaml`.

## 4. Precompute disparity and local differential disparity

```bash
python3 reproduction/run_pipeline.py --stage precompute \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" \
  --stereo-checkpoint "$STEREO_CKPT" --gpu 0
```

The cache contains `disp_npy`, the differential-disparity subfolders, `local_ground_npy`, and `instance_metrics.csv`. Building masks define the roof support, and the surrounding non-building annulus supplies the local ground reference.

## 5. Prepare height-adaptation protocols

```bash
python3 reproduction/run_pipeline.py --stage prepare \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT"
```

This calls `run_prepare_stereo_calibration_protocols.sh`. The generated manifests and configs specify the train/validation split for each protocol. To rebuild the cache-dependent inputs, use the existing entry point:

```bash
DATA_ROOT="$DATA_ROOT" CACHE_ROOT="$CACHE_ROOT" PY=python3 \
  bash run_rebuild_stereo_calibration_inputs.sh
```

## 6. Train a height-adaptation protocol

```bash
python3 reproduction/run_pipeline.py --stage train \
  --config-name train_city_stratified_50_50 --num-processes 1 --gpu 0
```

Select `train_city_stratified_10_90`, `train_city_stratified_20_80`, `train_city_stratified_50_50`, or `train_quality_diverse_cross_city` as needed. Height adaptation uses the cached stereo evidence with a frozen stereo backbone. Product-derived height labels supervise the building-level height module.

## 7. Evaluate and save predicted height maps

```bash
python3 reproduction/run_pipeline.py --stage evaluate \
  --data-root "$DATA_ROOT" \
  --height-config config/train_city_stratified_50_50.yaml \
  --height-checkpoint "$HEIGHT_CKPT" \
  --output-root "$OUT_ROOT/evaluation" --gpu 0 --save-vis
```

The evaluation calls `evaluate_height.py` and writes its metrics and visualizations to the selected output directory. Use `--cities` to specify a subset of the nine UnifiedBH regions.

## 8. Generate protocol plots

```bash
python3 reproduction/run_pipeline.py --stage plot \
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" \
  --output-root "$OUT_ROOT/figures" --gpu 0
```

This invokes `run_plot_stereo_calibration_protocols.sh`, which groups the outputs by protocol. Use the corresponding protocol config, checkpoint, and instance manifest throughout evaluation and plotting.

## 9. Inspect commands before execution

Add `--dry-run` to any stage to print the exact command, working directory, and stage-specific environment variables without launching training or inference:

```bash
python3 reproduction/run_pipeline.py --stage evaluate \
  --data-root "$DATA_ROOT" --height-checkpoint "$HEIGHT_CKPT" \
  --height-config config/train_city_stratified_50_50.yaml \
  --output-root "$OUT_ROOT/evaluation" --dry-run
```

## 10. Export and check display figures

```bash
python3 reproduction/verify_assets.py
```

To export the display PNGs again, install Pillow and Poppler, place the matching manuscript figure PDFs in a source directory, and run:

```bash
python3 reproduction/export_figures.py \
  --figures sources/figures --output outputs/display_figures
```

The export script checks source hashes, applies the recorded panel crops, and writes the PNGs to a new output folder. It preserves the scientific values and source PDFs.

## Display figure map

| File | Content |
|---|---|
| `workflow.png` | Conventional and differential-disparity workflows |
| `architecture.png` | Stereo pre-training and height-adaptation architecture |
| `height_maps.png` | GF-7 image context and predicted building heights |
| `regional_benchmark.png` | Paired scene and regional evaluation panels |
| `multiscale_comparison.png` | Local paired maps across spatial supports |
| `kunming_transfer_full.png` | Complete-scene Kunming transfer plots |
| `kunming_transfer_screened.png` | Quality-screened Kunming transfer plots |
