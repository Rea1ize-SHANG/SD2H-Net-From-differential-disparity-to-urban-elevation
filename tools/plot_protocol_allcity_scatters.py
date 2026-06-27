#!/usr/bin/env python3
"""Plot one all-city scatter figure per UnifiedBH calibration protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.make_unifiedbh_figures import _plot_scatter_grid
from tools.plot_latest_strategy_scatters import _apply_base_filter
from tools.quick_instance_calibration_search import (
    USE_COLUMNS,
    _apply_manifest,
    _fit_huber_affine,
    _make_preprocessor,
    _metrics,
    _predict_city_affine,
    _prepare_features,
    _sample_weights,
)


PROTOCOLS = {
    "city10_90": {
        "title": "Intra-city 10/90",
        "train_manifest": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_10_90_train_manifest.json",
        "val_manifest": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_10_90_val_manifest.json",
        "summary": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/quick_calibration_10_90.json",
    },
    "city20_80": {
        "title": "Intra-city 20/80",
        "train_manifest": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_20_80_train_manifest.json",
        "val_manifest": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/city_stratified_20_80_val_manifest.json",
        "summary": "checkpoints/dev_protocols_seed724_locked_hk062_sz061/quick_calibration_20_80.json",
    },
    "cross_hzdg058_sz038": {
        "title": "Cross-city transfer",
        "train_manifest": "checkpoints/dev_protocols_cross_hzdg058_sz038_locked_hk062_sz061/quality_diverse_cross_city_train_manifest.json",
        "val_manifest": "checkpoints/dev_protocols_cross_hzdg058_sz038_locked_hk062_sz061/quality_diverse_cross_city_val_manifest.json",
        "summary": "checkpoints/dev_protocols_cross_hzdg058_sz038_locked_hk062_sz061/quick_calibration_cross_dev.json",
    },
}


def _dedupe_instances(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["city", "core_name", "inst_id"]).reset_index(drop=True)


def _fit_feature_model(train_df: pd.DataFrame, eval_df: pd.DataFrame, model_name: str) -> np.ndarray:
    include_city = model_name.endswith("_city")
    base_name = model_name.replace("_city", "")
    train_feat, numeric, categorical = _prepare_features(train_df, include_city=include_city)
    eval_feat, _, _ = _prepare_features(eval_df, include_city=include_city)
    y = train_feat["gt_mean"].to_numpy(dtype=np.float64)
    weights = _sample_weights(y)
    feature_cols = numeric + categorical

    if base_name == "ridge_features":
        pre = _make_preprocessor(numeric, categorical)
        model = make_pipeline(pre, Ridge(alpha=4.0, random_state=0))
        model.fit(train_feat[feature_cols], y, ridge__sample_weight=weights)
        return model.predict(eval_feat[feature_cols])

    if base_name == "huber_features":
        pre = _make_preprocessor(numeric, categorical)
        model = make_pipeline(pre, HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=1000))
        model.fit(train_feat[feature_cols], y, huberregressor__sample_weight=weights)
        return model.predict(eval_feat[feature_cols])

    if base_name == "smooth_gbr":
        train_x = train_feat[feature_cols].copy()
        eval_x = eval_feat[feature_cols].copy()
        if categorical:
            train_x = pd.get_dummies(train_x, columns=list(categorical), dummy_na=True)
            eval_x = pd.get_dummies(eval_x, columns=list(categorical), dummy_na=True)
            eval_x = eval_x.reindex(columns=train_x.columns, fill_value=0.0)
        train_x = train_x.replace([np.inf, -np.inf], np.nan)
        eval_x = eval_x.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train_x)
        x_eval = imputer.transform(eval_x)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=45,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=724,
        )
        model.fit(x_train, y, sample_weight=weights)
        return model.predict(x_eval)

    raise ValueError(f"Unsupported model: {model_name}")


def _predict_best(train_df: pd.DataFrame, eval_df: pd.DataFrame, best_model: str) -> np.ndarray:
    if best_model == "raw_physical":
        pred = eval_df["height_physical_mean"].to_numpy(dtype=np.float64)
    elif best_model == "global_huber_affine":
        scale, bias = _fit_huber_affine(
            train_df["height_physical_mean"].to_numpy(dtype=np.float64),
            train_df["gt_mean"].to_numpy(dtype=np.float64),
            _sample_weights(train_df["gt_mean"].to_numpy(dtype=np.float64)),
        )
        pred = scale * eval_df["height_physical_mean"].to_numpy(dtype=np.float64) + bias
    elif best_model == "city_huber_affine":
        pred, _params = _predict_city_affine(train_df, eval_df)
    else:
        pred = _fit_feature_model(train_df, eval_df, best_model)
    return np.clip(np.asarray(pred, dtype=np.float64), 3.0, 260.0)


def _city_panels(eval_df: pd.DataFrame, pred: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    eval_df = eval_df.reset_index(drop=True)
    y = eval_df["gt_mean"].to_numpy(dtype=np.float64)
    panels = []
    for city, sub in eval_df.groupby("city", sort=True):
        idx = sub.index.to_numpy()
        panels.append((str(city), y[idx], pred[idx].astype(np.float64)))
    return panels


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
    parser.add_argument("--output_dir", type=Path, default=Path("/home/shangying/workspace/output/latest_height_strategy_scatters/protocol_allcity"))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS.keys()), choices=list(PROTOCOLS.keys()))
    args = parser.parse_args()

    repo = Path("/home/shangying/workspace/myrepos/SD2H")
    df = pd.read_csv(args.metrics_csv, usecols=USE_COLUMNS)
    locked = _apply_base_filter(
        df,
        repo / "checkpoints/stereo_calib_noise_filter_manifest/instance_filter_manifest.json",
        ["HK_062", "SZ_061"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for tag in args.protocols:
        spec = PROTOCOLS[tag]
        summary = json.loads((repo / spec["summary"]).read_text())
        best_model = str(summary["best_model"])
        train_df = _apply_manifest(df, repo / spec["train_manifest"]).reset_index(drop=True)
        val_df = _apply_manifest(df, repo / spec["val_manifest"]).reset_index(drop=True)
        eval_df = _dedupe_instances(pd.concat([val_df, locked], ignore_index=True))
        pred = _predict_best(train_df, eval_df, best_model)
        y = eval_df["gt_mean"].to_numpy(dtype=np.float64)
        metrics = _metrics(y, pred)
        all_metrics[tag] = {
            "title": spec["title"],
            "best_model": best_model,
            "train_instances": int(len(train_df)),
            "eval_instances": int(len(eval_df)),
            "eval_cities": sorted(eval_df["city"].astype(str).unique().tolist()),
            "metrics": metrics,
        }
        safe_title = f"{spec['title']} ({best_model})"
        out_png = args.output_dir / f"{tag}_{best_model}_allcity_robust_scatter.png"
        _plot_scatter_grid(out_png, [(safe_title, y, pred)], max_points=140000, ncols=1)
        city_grid_png = args.output_dir / f"{tag}_{best_model}_citywise_grid_robust_scatter.png"
        _plot_scatter_grid(city_grid_png, _city_panels(eval_df, pred), max_points=120000, ncols=3)
        print(
            f"{tag}: model={best_model}, N={metrics['n']:,}, MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, Bias={metrics['bias']:+.3f}, R2={metrics['r2']:.3f}"
        )
        print(f"  saved: {out_png}")
        print(f"  saved: {city_grid_png}")

    (args.output_dir / "protocol_allcity_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"Saved metrics: {args.output_dir / 'protocol_allcity_metrics.json'}")


if __name__ == "__main__":
    main()
