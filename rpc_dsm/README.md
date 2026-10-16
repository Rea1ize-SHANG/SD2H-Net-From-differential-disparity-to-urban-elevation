# RPC–DSM building-height pipeline

This folder contains the RPC–DSM comparison workflow: GF-7 pair preparation, epipolar rectification, common cropping, stereo disparity, RPC triangulation, geolocation, projected DSM export, footprint alignment, and roof–ground height extraction.

Run commands from the SD2H-Net repository root on Linux. The workflow uses the stereo implementation already in `core/` and `precompute_unifiedbh_disp_delta.py`.

## Environment

Use the main repository's Python/PyTorch environment. Install GDAL and the additional geospatial packages, and install s2p with its native image-processing tools:

```bash
conda install -c conda-forge gdal rasterio scipy scikit-image numpy
pip install -r rpc_dsm/requirements.txt
python -m rpc_dsm.pipeline --help
```

## Inputs

- A raw-scene root containing GF-7 scene subfolders. Panchromatic pairs use `BPA/FPA` or `BWDPAN/FWDPAN` filenames.
- GeoTIFF inputs with complete RPC metadata in their `RPC` tag namespace.
- A WGS84 building shapefile with a `Height` field, or a single-band georeferenced height raster.
- A locally supplied stereo checkpoint passed with `--checkpoint`.
- A dedicated output directory selected with `--output-root`.

## Complete workflow

```bash
python -m rpc_dsm.pipeline \
  --raw-root data/GF7_raw \
  --label data/buildings.shp \
  --checkpoint checkpoints/pretrain_whusside_60ep_724/final.pth \
  --output-root outputs/rpc_dsm_run \
  --gpus 0 \
  --ground-buffer-m 20 \
  --rpc-residual-threshold 3 \
  --from-stage prepare --until-stage heights
```

Add `--scenes` with the requested original folder names or scene IDs to restrict processing. Use `--gpus 0 1` to shard stereo inference across explicitly selected GPUs. Set `--min-height` and `--max-height` to the required absolute-altitude range for RPC rectification.

The stage order is:

```text
prepare → rectify → crop → disparity → dsm → geolocate → project → labels → heights
```

To validate inputs and inspect the stage range before creating outputs:

```bash
python -m rpc_dsm.pipeline \
  --raw-root data/GF7_raw --label data/buildings.shp \
  --checkpoint checkpoints/pretrain_whusside_60ep_724/final.pth \
  --output-root outputs/rpc_dsm_run --dry-run
```

## Resume a stage range

Keep the same inputs and output root when resuming:

```bash
python -m rpc_dsm.pipeline \
  --raw-root data/GF7_raw --label data/buildings.shp \
  --checkpoint checkpoints/pretrain_whusside_60ep_724/final.pth \
  --output-root outputs/rpc_dsm_run \
  --from-stage dsm --until-stage heights
```

The RPC-only mode is selected when `--tie-root` is omitted. To use existing original-image ENVI tie points, provide `--tie-root data/tie_points` with the scene-specific `.pts` files.

## Building-height extraction

`building_heights.py` aggregates DSM samples within each building mask. Roof values above `Q3 + 1.5 × IQR` are excluded. The predicted height is the maximum accepted roof elevation minus the minimum non-building DSM elevation in the exterior 20 m buffer. The implementation keeps the two-pixel inner exclusion and assigns exterior pixels to their nearest building instance. Repeated tile observations are consolidated by building ID within a scene.

Run the extraction stage directly on existing DSM, geolocation, and label outputs:

```bash
python -m rpc_dsm.building_heights \
  --dsm-root outputs/rpc_dsm_run/dsm \
  --labels-root outputs/rpc_dsm_run/labels \
  --crop-root outputs/rpc_dsm_run/preprocess/crop \
  --geo-root outputs/rpc_dsm_run/geolocation \
  --output-root outputs/rpc_dsm_run/heights \
  --scenes 058 --buffer-m 20
```

## Output layout

```text
outputs/rpc_dsm_run/
├── preprocess/epipolar/    # Rectification configs and homographies
├── preprocess/crop/       # Common stereo crops
├── tiles/                 # Paired 1024-pixel tiles and disparity arrays
├── dsm/                   # RPC-derived elevation rasters
├── geolocation/           # Longitude/latitude grids
├── projected_dsm/         # Georeferenced DSM rasters
├── labels/                # Aligned footprint masks and height labels
├── heights/               # Scene-level building-height CSVs
├── logs/
└── progress.json
```

## Code map and checks

| Module | Task |
|---|---|
| `pipeline.py` | Ordered stages and resume control |
| `inputs.py`, `rectify.py`, `crop.py` | RPC input preparation and rectification |
| `disparity.py` | Stereo inference and tiled disparity cache |
| `triangulate.py`, `gf7_rpc_compat.py` | RPC geometry and surface elevation |
| `geolocate.py`, `project.py` | Geographic grids and projected DSM |
| `vector_labels.py`, `raster_labels.py` | Footprint/height alignment |
| `building_heights.py` | Roof–ground height extraction |
| `dataset_files.py`, `local_dataset.py`, `naming.py`, `watch.py` | Supporting I/O and monitoring |

```bash
python -m unittest rpc_dsm.test_workflow
```

Checkpoint files remain external inputs referenced by command-line paths. This folder contains Python source, dependency declarations, and usage instructions.
