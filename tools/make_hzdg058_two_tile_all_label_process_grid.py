#!/usr/bin/env python3
"""Render all-label intermediate process panels for two HZ_DG_058 tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
import torch
from skimage.measure import label as connected_components
from skimage.segmentation import find_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.height_datasets import GF7_CITY_K_MAPPING, read_image_as_float32, read_npy_as_float32
from precompute_unifiedbh_disp_delta import _build_delta_and_instances
from tools.train_instance_height_calibrator import (
    PhysicsGuidedInstanceCalibrator,
    _add_features,
)


DEFAULT_DATA_ROOT = Path("/home/shangying/workspace/dataset/UnifiedBH/overlap15")
DEFAULT_CACHE_ROOT = Path(
    "/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/"
    "sd2h_full9_abs_groundfix_whusside724_20260622"
)
DEFAULT_CKPT = Path(
    "/home/shangying/workspace/myrepos/SD2H/checkpoints/"
    "instance_calib_city_stratified_20_80_devlocked_conservative/"
    "best_instance_calibrator.pth"
)
DEFAULT_OUT = Path("/home/shangying/workspace/output/hzdg058_all_label_process_current_best")
TILES = [("HZ_DG_058", 6, 3), ("HZ_DG_058", 9, 2)]


def _read_label(path: Path) -> np.ndarray:
    arr = skimage.io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where(np.abs(arr) <= 1.0e6, arr, 0.0)
    return arr.astype(np.float32, copy=False)


def _to_rgb_uint8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(3):
        channel = arr[..., ch].astype(np.float32)
        finite = np.isfinite(channel)
        if not finite.any():
            continue
        lo, hi = np.percentile(channel[finite], [p_low, p_high])
        if hi <= lo:
            hi = lo + 1.0
        out[..., ch] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _normalize(arr: np.ndarray, vmin=None, vmax=None, robust=(1.0, 99.0), valid_mask=None):
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if valid_mask is not None:
        finite &= valid_mask
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(a[finite], robust)
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((a - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    norm[~np.isfinite(norm)] = 0.0
    return norm.astype(np.float32)


def _colorize(arr, cmap, vmin=None, vmax=None, robust=(1.0, 99.0), valid_mask=None, background=(238, 238, 238)):
    norm = _normalize(arr, vmin=vmin, vmax=vmax, robust=robust, valid_mask=valid_mask)
    rgb = (plt.get_cmap(cmap)(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    if valid_mask is not None:
        rgb[~valid_mask] = np.asarray(background, dtype=np.uint8)
    return rgb


def _instance_color(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask[mask > 0].astype(np.int64, copy=False))
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for inst_id in ids:
        rng = np.random.default_rng(int(inst_id) * 7919 + 17)
        out[mask == inst_id] = rng.integers(50, 245, size=3, dtype=np.uint8)
    return out


def _overlay_blue_contours(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base_rgb.copy()
    boundaries = find_boundaries(mask > 0, mode="outer")
    thick = boundaries.copy()
    thick[:-1, :] |= boundaries[1:, :]
    thick[1:, :] |= boundaries[:-1, :]
    thick[:, :-1] |= boundaries[:, 1:]
    thick[:, 1:] |= boundaries[:, :-1]
    out[thick] = np.asarray([0, 120, 255], dtype=np.uint8)
    return out


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), image.astype(np.uint8), check_contrast=False)


def _paths(data_root: Path, cache_root: Path, city: str, row: int, col: int) -> dict:
    suffix = f"{row}_{col}"
    core = f"{city}_{suffix}"
    paths = {
        "core": core,
        "city": city,
        "left": data_root / "left" / city / f"{city}_left_{suffix}.tif",
        "right": data_root / "right" / city / f"{city}_right_{suffix}.tif",
        "label": data_root / "label" / city / f"{city}_label_{suffix}.tif",
        "disp": cache_root / "disp_npy" / city / f"Disp_{core}.npy",
    }
    missing = [str(p) for k, p in paths.items() if k not in {"core", "city"} and not Path(p).exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    return paths


def _make_build_args() -> SimpleNamespace:
    return SimpleNamespace(
        min_instance_pixels=1,
        min_ring_pixels=1,
        ring_inner_radius=3,
        ring_outer_radius=35,
        fallback_ring_radius=80,
        ground_percentile=50.0,
        correct_ground_outliers=True,
        ground_outlier_abs_tol=3.0,
        ground_outlier_mad_scale=3.5,
        ground_outlier_min_instances=3,
        label_min_height=0.0,
    )


def _load_calibrator(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    config = ckpt["config"]
    normalizer = config["normalizer"]
    model = PhysicsGuidedInstanceCalibrator(
        num_features=len(normalizer["feature_names"]),
        num_cities=len(normalizer["city_to_idx"]),
        use_city_embedding=bool(config.get("use_city_embedding", False)),
        init_scale=float(config.get("init_scale", 1.0)),
        init_bias=float(config.get("init_bias", 0.0)),
        hidden_dim=int(config.get("hidden_dim", 48)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
        residual_clip=float(config.get("residual_clip", 12.0)),
        low_residual_clip=float(config.get("low_residual_clip", 3.0)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, normalizer


@torch.no_grad()
def _predict_rows(df: pd.DataFrame, model, normalizer: dict, device: torch.device) -> np.ndarray:
    feat = _add_features(df)
    names = normalizer["feature_names"]
    x = feat[names].to_numpy(dtype=np.float64)
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)
    x = (x - mean) / std

    city_to_idx = normalizer["city_to_idx"]
    unknown_idx = int(city_to_idx.get("<UNK>", max(city_to_idx.values())))
    city_idx = feat["city"].astype(str).map(city_to_idx).fillna(unknown_idx).to_numpy(dtype=np.int64)
    physical = feat["height_physical_mean"].to_numpy(dtype=np.float32)

    pred, _, _ = model(
        torch.from_numpy(x.astype(np.float32)).to(device),
        torch.from_numpy(physical).to(device),
        torch.from_numpy(city_idx).to(device),
    )
    return pred.detach().cpu().numpy().astype(np.float32)


def _tile_panels(paths: dict, model, normalizer: dict, device: torch.device, out_dir: Path) -> dict:
    city = paths["city"]
    core = paths["core"]
    left = read_image_as_float32(str(paths["left"]))
    label_h = _read_label(paths["label"])
    disp = read_npy_as_float32(str(paths["disp"]))
    mask = connected_components(np.isfinite(label_h) & (label_h > 0.0), connectivity=2).astype(np.int64)
    valid = mask > 0

    delta, _signed, _reverse, delta_abs, ground, rows = _build_delta_and_instances(
        disp,
        mask,
        label_h,
        float(GF7_CITY_K_MAPPING[city]),
        _make_build_args(),
    )
    if not rows:
        raise RuntimeError(f"No label instances found for {core}")

    rows_df = pd.DataFrame(rows)
    rows_df["city"] = city
    rows_df["core_name"] = core
    rows_df["k"] = float(GF7_CITY_K_MAPPING[city])
    preds = _predict_rows(rows_df, model, normalizer, device)

    pred_h = np.zeros_like(label_h, dtype=np.float32)
    for inst_id, pred in zip(rows_df["inst_id"].astype(int).to_numpy(), preds):
        pred_h[mask == int(inst_id)] = float(pred)

    left_rgb = _to_rgb_uint8(left)
    panels = {
        "left": left_rgb,
        "label_outline": _overlay_blue_contours(left_rgb, mask),
        "disparity": _colorize(disp, "turbo", robust=(1.0, 99.0)),
        "ground_disparity": _colorize(ground, "turbo", robust=(1.0, 99.0), valid_mask=valid),
        "differential_disparity": _colorize(delta_abs, "inferno", robust=(1.0, 99.5), valid_mask=valid),
        "instance_mask": _instance_color(mask),
        "predicted_height": _colorize(pred_h, "viridis", vmin=0.0, vmax=120.0, valid_mask=valid),
        "reference_height": _colorize(label_h, "viridis", vmin=0.0, vmax=120.0, valid_mask=valid),
    }

    tile_dir = out_dir / core
    for name, image in panels.items():
        _write_png(tile_dir / f"{core}_{name}.png", image)

    summary = {
        "core": core,
        "instances_from_label": int(np.unique(mask[valid]).size),
        "instances_predicted": int(len(rows_df)),
        "reference_mean": float(label_h[valid].mean()) if valid.any() else 0.0,
        "reference_max": float(label_h[valid].max()) if valid.any() else 0.0,
        "predicted_mean": float(pred_h[valid].mean()) if valid.any() else 0.0,
        "predicted_max": float(pred_h[valid].max()) if valid.any() else 0.0,
    }
    (tile_dir / f"{core}_summary.json").write_text(json.dumps(summary, indent=2))
    return {"core": core, "panels": panels, "summary": summary}


def _make_grid(rows: list[dict], out_dir: Path) -> None:
    panel_rows = [
        ("left", "Left image"),
        ("label_outline", "Label outline"),
        ("disparity", "Disparity"),
        ("ground_disparity", "Ground disparity"),
        ("differential_disparity", "Differential disparity"),
        ("instance_mask", "Instance mask"),
        ("predicted_height", "Predicted height"),
        ("reference_height", "Reference height"),
    ]
    fig_rows = len(rows) * 2
    fig_cols = 4
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 22,
            "axes.titlesize": 24,
            "figure.dpi": 220,
        }
    ):
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(17.5, 17.8), squeeze=False)
        for sample_idx, item in enumerate(rows):
            base_row = sample_idx * 2
            for local_idx, (key, title) in enumerate(panel_rows):
                r = base_row + local_idx // 4
                c = local_idx % 4
                ax = axes[r, c]
                ax.imshow(item["panels"][key])
                ax.set_axis_off()
                ax.set_title(title, pad=8)
                if c == 0 and local_idx in (0, 4):
                    ax.text(
                        0.02,
                        0.98,
                        item["core"],
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=20,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.58, edgecolor="none", pad=3.0),
                    )
        fig.subplots_adjust(left=0.008, right=0.995, top=0.965, bottom=0.01, wspace=0.035, hspace=0.14)
        fig.savefig(out_dir / "HZ_DG_058_6_3_9_2_all_label_process_grid_4x4.png", dpi=300)
        fig.savefig(out_dir / "HZ_DG_058_6_3_9_2_all_label_process_grid_4x4.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, normalizer = _load_calibrator(args.checkpoint, device)

    rendered = []
    for city, row, col in TILES:
        p = _paths(args.data_root, args.cache_root, city, row, col)
        print(f"[*] Rendering {p['core']} with all label instances")
        rendered.append(_tile_panels(p, model, normalizer, device, args.output_dir))

    _make_grid(rendered, args.output_dir)
    summary = {
        "checkpoint": str(args.checkpoint),
        "cache_root": str(args.cache_root),
        "tiles": [item["summary"] for item in rendered],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[*] Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
