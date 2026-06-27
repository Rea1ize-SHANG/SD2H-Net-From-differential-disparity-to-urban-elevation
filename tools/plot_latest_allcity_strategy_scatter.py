#!/usr/bin/env python3
"""Plot all-city scatter panels for the latest UnifiedBH calibration strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.make_unifiedbh_figures import _plot_scatter_grid
from tools.plot_latest_strategy_scatters import (
    _apply_base_filter,
    _apply_manifest,
    _predict_affine,
    _predict_residual_calibrator,
)
from tools.train_instance_height_calibrator import USE_COLUMNS, _metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path(
            "/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/"
            "sd2h_full9_abs_groundfix_whusside724_20260622/instance_metrics.csv"
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("/home/shangying/workspace/output/latest_height_strategy_scatters"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path("/home/shangying/workspace/myrepos/SD2H")
    df = pd.read_csv(args.metrics_csv, usecols=USE_COLUMNS)
    ckpt_path = repo / "checkpoints/instance_calib_city_stratified_20_80_devlocked_conservative/best_instance_calibrator.pth"
    cfg = json.loads((repo / "checkpoints/instance_calib_city_stratified_20_80_devlocked_conservative/config.json").read_text())

    dev = _apply_manifest(
        df,
        repo / "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_20_80_val_manifest.json",
    )
    locked = _apply_base_filter(
        df,
        repo / "checkpoints/stereo_calib_noise_filter_manifest/instance_filter_manifest.json",
        ["HK_062", "SZ_061"],
    )
    eval_df = pd.concat([dev, locked], ignore_index=True)
    y = eval_df["gt_mean"].to_numpy(dtype=np.float64)

    raw = eval_df["height_physical_mean"].to_numpy(dtype=np.float64)
    affine = _predict_affine(eval_df, float(cfg["init_scale"]), float(cfg["init_bias"]))
    residual = _predict_residual_calibrator(eval_df, ckpt_path, args.device)

    panels = [
        ("Raw physical", y, raw),
        ("Affine physical only", y, affine),
        ("Residual calibrator", y, residual),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / "latest_allcity_three_strategy_robust_scatter.png"
    _plot_scatter_grid(out_png, panels, max_points=140000, ncols=3)

    rows = []
    for name, pred in [("Raw physical", raw), ("Affine physical only", affine), ("Residual calibrator", residual)]:
        rows.append({"strategy": name, "city": "ALL", **_metrics(y, pred)})
    (args.output_dir / "latest_allcity_three_strategy_robust_scatter_metrics.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_png.with_suffix('.pdf')}")
    for row in rows:
        print(
            f"{row['strategy']}: N={row['n']:,}, MAE={row['mae']:.3f}, "
            f"RMSE={row['rmse']:.3f}, Bias={row['bias']:+.3f}, R2={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
