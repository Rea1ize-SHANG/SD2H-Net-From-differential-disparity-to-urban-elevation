# Reproduction commands

Run from the repository root in the SD2H environment. Follow the main [README](../README.md) to obtain data and weights.

```bash
export WHU=/path/to/WHU-SSIDE
export DATA=/path/to/UnifiedBH/overlap15
export CACHE="$DATA/disp/sd2h_full9_abs_groundfix_whusside724_20260622"

# Stereo pre-training.
python reproduction/run_pipeline.py --stage pretrain \
  --data-root "$WHU" --gpu 0,1,2,3 --num-processes 4 --batch-size 4

# Cache fixed stereo evidence.
python reproduction/run_pipeline.py --stage precompute \
  --data-root "$DATA" --cache-root "$CACHE" --gpu 0

# Build filtering, train/validation manifests and calibration coefficients.
python reproduction/run_pipeline.py --stage prepare \
  --data-root "$DATA" --cache-root "$CACHE"

# Train one height protocol.
python reproduction/run_pipeline.py --stage train \
  --data-root "$DATA" --cache-root "$CACHE" \
  --config-name train_city_stratified_50_50 --gpu 0

# Evaluate the matching checkpoint and configuration.
python reproduction/run_pipeline.py --stage evaluate \
  --data-root "$DATA" --height-config config/train_city_stratified_50_50.yaml \
  --height-checkpoint checkpoints/city_stratified_50_50/finetune_best_rmse.pth \
  --output-root outputs/height_evaluation --gpu 0
```

Add `--dry-run` to print a stage command without starting it. Set `--num-processes` and a divisible global `--batch-size` for multi-GPU training. No stage launches another training job implicitly.

For the comparison method, follow the [RPC-DSM workflow](../rpc_dsm/README.md).

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
