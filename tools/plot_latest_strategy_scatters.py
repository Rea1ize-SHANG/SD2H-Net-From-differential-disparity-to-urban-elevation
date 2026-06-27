#!/usr/bin/env python3
"""Plot scatter figures for the latest UnifiedBH height-calibration strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.make_unifiedbh_figures import _plot_scatter_grid
from tools.train_instance_height_calibrator import (
    HEIGHT_BINS,
    USE_COLUMNS,
    PhysicsGuidedInstanceCalibrator,
    _add_features,
    _height_bins,
    _metrics,
)


def _load_manifest(path: Path) -> Tuple[set[str], Dict[str, set[int]]]:
    payload = json.loads(path.read_text())
    valid_tiles = {str(x) for x in payload.get("valid_tiles", [])}
    ignore = {
        str(tile): {int(v) for v in ids}
        for tile, ids in payload.get("ignore_instances", {}).items()
    }
    return valid_tiles, ignore


def _load_base_ignore(path: Path) -> Dict[str, set[int]]:
    payload = json.loads(path.read_text())
    raw = payload.get("ignore_instances", payload)
    return {str(tile): {int(v) for v in ids} for tile, ids in raw.items()}


def _apply_manifest(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    valid_tiles, ignore = _load_manifest(manifest_path)
    tile = df["city"].astype(str) + "/" + df["core_name"].astype(str)
    sub = df.loc[tile.isin(valid_tiles)].copy()
    if sub.empty:
        return sub.reset_index(drop=True)
    sub_tile = (sub["city"].astype(str) + "/" + sub["core_name"].astype(str)).to_numpy()
    inst = sub["inst_id"].astype(int).to_numpy()
    keep = np.asarray([int(i) not in ignore.get(str(t), set()) for t, i in zip(sub_tile, inst)], dtype=bool)
    return sub.loc[keep].copy().reset_index(drop=True)


def _apply_base_filter(df: pd.DataFrame, base_manifest_path: Path, cities: Iterable[str]) -> pd.DataFrame:
    ignore = _load_base_ignore(base_manifest_path)
    sub = df[df["city"].isin(list(cities))].copy().reset_index(drop=True)
    tile = (sub["city"].astype(str) + "/" + sub["core_name"].astype(str)).to_numpy()
    inst = sub["inst_id"].astype(int).to_numpy()
    keep = np.asarray([int(i) not in ignore.get(str(t), set()) for t, i in zip(tile, inst)], dtype=bool)
    return sub.loc[keep].copy().reset_index(drop=True)


def _predict_residual_calibrator(df: pd.DataFrame, ckpt_path: Path, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    normalizer = config["normalizer"]
    city_to_idx = normalizer["city_to_idx"]
    unknown_idx = int(city_to_idx.get("<UNK>", max(city_to_idx.values())))

    feat = _add_features(df)
    names = normalizer["feature_names"]
    x = feat[names].to_numpy(dtype=np.float64)
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)
    x = (x - mean) / std
    physical = feat["height_physical_mean"].to_numpy(dtype=np.float32)
    city_idx = feat["city"].astype(str).map(city_to_idx).fillna(unknown_idx).to_numpy(dtype=np.int64)

    model = PhysicsGuidedInstanceCalibrator(
        num_features=x.shape[1],
        num_cities=max(city_to_idx.values()) + 1,
        use_city_embedding=bool(config["use_city_embedding"]),
        init_scale=float(config["init_scale"]),
        init_bias=float(config["init_bias"]),
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
        residual_clip=float(config["residual_clip"]),
        low_residual_clip=float(config["low_residual_clip"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(torch_device)
    preds = []
    with torch.no_grad():
        for start in range(0, len(df), 65536):
            xb = torch.from_numpy(x[start:start + 65536].astype(np.float32)).to(torch_device)
            pb = torch.from_numpy(physical[start:start + 65536]).to(torch_device)
            cb = torch.from_numpy(city_idx[start:start + 65536]).to(torch_device)
            pred, _base, _res = model(xb, pb, cb)
            preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds)


def _predict_affine(df: pd.DataFrame, scale: float, bias: float) -> np.ndarray:
    physical = df["height_physical_mean"].to_numpy(dtype=np.float64)
    return np.clip(scale * physical + bias, 3.0, 260.0)


def _print_metrics(title: str, df: pd.DataFrame, pred: np.ndarray) -> None:
    y = df["gt_mean"].to_numpy(dtype=np.float64)
    print(f"\n[{title}]")
    overall = _metrics(y, pred)
    print(
        f"ALL       N={overall['n']:>7d} MAE={overall['mae']:.3f} "
        f"RMSE={overall['rmse']:.3f} Bias={overall['bias']:+.3f} R2={overall['r2']:.3f}"
    )
    for city, sub in df.reset_index(drop=True).groupby("city", sort=True):
        idx = sub.index.to_numpy()
        m = _metrics(y[idx], pred[idx])
        print(
            f"{city:<9} N={m['n']:>7d} MAE={m['mae']:.3f} "
            f"RMSE={m['rmse']:.3f} Bias={m['bias']:+.3f} R2={m['r2']:.3f}"
        )


def _panels_by_city(df: pd.DataFrame, pred: np.ndarray, include_overall: bool = True) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    df = df.reset_index(drop=True)
    y = df["gt_mean"].to_numpy(dtype=np.float64)
    panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if include_overall:
        panels.append(("All", y, pred.astype(np.float64)))
    for city, sub in df.groupby("city", sort=True):
        idx = sub.index.to_numpy()
        panels.append((str(city), y[idx], pred[idx].astype(np.float64)))
    return panels


def _save_strategy(output_dir: Path, prefix: str, df: pd.DataFrame, pred: np.ndarray, ncols: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_metrics(prefix, df, pred)
    _plot_scatter_grid(output_dir / f"{prefix}_scatter_grid.png", _panels_by_city(df, pred, include_overall=True), max_points=120000, ncols=ncols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, default=Path("/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622/instance_metrics.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/shangying/workspace/output/latest_height_strategy_scatters"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path("/home/shangying/workspace/myrepos/SD2H")
    df = pd.read_csv(args.metrics_csv, usecols=USE_COLUMNS)

    ckpt_path = repo / "checkpoints/instance_calib_city_stratified_20_80_devlocked_conservative/best_instance_calibrator.pth"
    cfg = json.loads((repo / "checkpoints/instance_calib_city_stratified_20_80_devlocked_conservative/config.json").read_text())

    dev20_df = _apply_manifest(df, repo / "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_20_80_val_manifest.json")
    dev20_pred = _predict_residual_calibrator(dev20_df, ckpt_path, args.device)
    _save_strategy(args.output_dir, "dev20_residual_calibrator", dev20_df, dev20_pred, ncols=3)

    locked_df = _apply_base_filter(df, repo / "checkpoints/stereo_calib_noise_filter_manifest/instance_filter_manifest.json", ["HK_062", "SZ_061"])
    locked_affine = _predict_affine(locked_df, float(cfg["init_scale"]), float(cfg["init_bias"]))
    _save_strategy(args.output_dir, "locked_hk062_sz061_affine_physical_only", locked_df, locked_affine, ncols=3)

    locked_residual = _predict_residual_calibrator(locked_df, ckpt_path, args.device)
    _save_strategy(args.output_dir, "locked_hk062_sz061_residual_calibrator", locked_df, locked_residual, ncols=3)

    cross_df = _apply_manifest(df, repo / "checkpoints/dev_protocols_cross_hzdg058_sz038_locked_hk062_sz061/quality_diverse_cross_city_val_manifest.json")
    cross_json = json.loads((repo / "checkpoints/dev_protocols_cross_hzdg058_sz038_locked_hk062_sz061/quick_calibration_cross_dev.json").read_text())
    # The best cross-development model is not checkpointed; use its stable affine baseline for a directly reproducible panel.
    scale = float(cross_json["global_affine"]["scale"])
    bias = float(cross_json["global_affine"]["bias"])
    cross_affine = _predict_affine(cross_df, scale, bias)
    _save_strategy(args.output_dir, "cross_dev_affine_physical_only", cross_df, cross_affine, ncols=3)

    print(f"\nSaved scatter figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
