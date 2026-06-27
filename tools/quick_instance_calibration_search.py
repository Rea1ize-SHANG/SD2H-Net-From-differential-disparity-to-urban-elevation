#!/usr/bin/env python3
"""Fast instance-level calibration checks for UnifiedBH height adaptation.

This utility is intentionally separate from the tile-level finetuning loop.
It uses the precomputed instance metrics and explicit train/validation
manifests to test whether the stereo differential-disparity prior can be
calibrated from the training instances without leaking validation/test labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "gt_mean",
    "pixel_count",
    "height_physical_mean",
    "height_physical_p50",
    "height_reverse_mean",
    "delta_abs_mean",
    "delta_abs_p50",
    "delta_relu_mean",
    "delta_signed_mean",
    "disp_mean",
    "disp_p50",
    "ground_disp",
    "ground_disp_image_median",
    "ground_disp_image_mad",
    "disp_order_gap",
    "k",
]

USE_COLUMNS = [
    "inst_id",
    "city",
    "core_name",
    "ground_corrected",
    "disp_order_bad",
    *NUMERIC_COLUMNS,
]

HEIGHT_BINS = [12.0, 24.0, 30.0, 50.0, 100.0, 150.0, 300.0]


def _load_manifest(path: Path) -> Tuple[set[str], Dict[str, set[int]], List[str]]:
    payload = json.loads(path.read_text())
    valid_tiles = {str(x) for x in payload.get("valid_tiles", [])}
    ignore = {
        str(tile): {int(v) for v in ids}
        for tile, ids in payload.get("ignore_instances", {}).items()
    }
    cities = [str(x) for x in payload.get("train_cities", payload.get("val_cities", []))]
    return valid_tiles, ignore, cities


def _apply_manifest(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    valid_tiles, ignore, _cities = _load_manifest(manifest_path)
    tile_key = df["city"].astype(str) + "/" + df["core_name"].astype(str)
    keep_tile = tile_key.isin(valid_tiles).to_numpy()
    if not keep_tile.any():
        return df.iloc[0:0].copy()

    sub = df.loc[keep_tile].copy()
    sub_tile = sub["city"].astype(str) + "/" + sub["core_name"].astype(str)
    ignored = np.zeros(len(sub), dtype=bool)
    inst = sub["inst_id"].astype(int).to_numpy()
    for idx, tile in enumerate(sub_tile.to_numpy()):
        ids = ignore.get(tile)
        if ids and int(inst[idx]) in ids:
            ignored[idx] = True
    return sub.loc[~ignored].copy()


def _prepare_features(df: pd.DataFrame, include_city: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()
    out["log_area"] = np.log1p(out["pixel_count"].astype(float).clip(lower=0.0))
    out["sqrt_area"] = np.sqrt(out["pixel_count"].astype(float).clip(lower=0.0))
    out["phys_area"] = out["height_physical_mean"].astype(float) * out["log_area"].astype(float)
    out["phys_delta_ratio"] = out["height_physical_mean"].astype(float) / (
        out["delta_abs_mean"].astype(float).abs() + 1.0
    )
    out["abs_signed_delta"] = out["delta_signed_mean"].astype(float).abs()
    out["ground_minus_median"] = (
        out["ground_disp"].astype(float) - out["ground_disp_image_median"].astype(float)
    )
    out["ground_corrected"] = out["ground_corrected"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    out["disp_order_bad"] = out["disp_order_bad"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)

    numeric = [
        "height_physical_mean",
        "height_physical_p50",
        "height_reverse_mean",
        "delta_abs_mean",
        "delta_abs_p50",
        "delta_relu_mean",
        "delta_signed_mean",
        "abs_signed_delta",
        "disp_mean",
        "disp_p50",
        "ground_disp",
        "ground_disp_image_median",
        "ground_disp_image_mad",
        "ground_minus_median",
        "disp_order_gap",
        "k",
        "log_area",
        "sqrt_area",
        "phys_area",
        "phys_delta_ratio",
        "ground_corrected",
        "disp_order_bad",
    ]
    categorical = ["city"] if include_city else []
    return out, numeric, categorical


def _height_bin_indices(y: np.ndarray, bins: Sequence[float] = HEIGHT_BINS) -> np.ndarray:
    return np.digitize(y.astype(float), np.asarray(bins, dtype=np.float64), right=True)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    bins = _height_bin_indices(y)
    _, counts = np.unique(bins, return_counts=True)
    count_map = dict(zip(np.unique(bins), counts))
    weights = np.asarray([1.0 / math.sqrt(count_map[int(b)]) for b in bins], dtype=np.float64)
    weights /= max(float(np.mean(weights)), 1.0e-6)
    height_boost = 1.0 + 0.02 * np.clip(y - 24.0, 0.0, 80.0)
    return np.clip(weights * height_boost, 0.2, 8.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def _print_metrics(title: str, rows: Iterable[Tuple[str, dict]]) -> None:
    print(f"\n[{title}]")
    print(f"{'model/city':<34} {'N':>9} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'R2':>8}")
    print("-" * 82)
    for name, m in rows:
        print(
            f"{name:<34} {m['n']:>9d} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
            f"{m['bias']:>+8.3f} {m['r2']:>8.3f}"
        )


def _fit_huber_affine(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=1000),
    )
    model.fit(x, y, huberregressor__sample_weight=sample_weight)
    scaler = model.named_steps["standardscaler"]
    huber = model.named_steps["huberregressor"]
    scale = float(huber.coef_[0] / scaler.scale_[0])
    bias = float(huber.intercept_ - huber.coef_[0] * scaler.mean_[0] / scaler.scale_[0])
    return scale, bias


def _predict_city_affine(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    global_scale, global_bias = _fit_huber_affine(
        train_df["height_physical_mean"].to_numpy(),
        train_df["gt_mean"].to_numpy(),
        _sample_weights(train_df["gt_mean"].to_numpy()),
    )
    preds = np.zeros(len(val_df), dtype=np.float64)
    params = {"GLOBAL": (global_scale, global_bias)}
    for city, city_val in val_df.groupby("city", sort=False):
        city_train = train_df[train_df["city"] == city]
        if len(city_train) >= 30:
            scale, bias = _fit_huber_affine(
                city_train["height_physical_mean"].to_numpy(),
                city_train["gt_mean"].to_numpy(),
                _sample_weights(city_train["gt_mean"].to_numpy()),
            )
        else:
            scale, bias = global_scale, global_bias
        params[str(city)] = (float(scale), float(bias))
        preds[city_val.index.to_numpy()] = scale * city_val["height_physical_mean"].to_numpy() + bias
    return preds, params


def _make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    transformers = [
        (
            "num",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            list(numeric),
        )
    ]
    if categorical:
        transformers.append(
            (
                "cat",
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                list(categorical),
            )
        )
    return ColumnTransformer(transformers)


def _fit_models(train_df: pd.DataFrame, val_df: pd.DataFrame, include_city: bool) -> Dict[str, np.ndarray]:
    train_feat, numeric, categorical = _prepare_features(train_df, include_city=include_city)
    val_feat, _, _ = _prepare_features(val_df, include_city=include_city)
    y = train_feat["gt_mean"].to_numpy(dtype=np.float64)
    weights = _sample_weights(y)
    feature_cols = numeric + categorical

    results: Dict[str, np.ndarray] = {}

    pre = _make_preprocessor(numeric, categorical)
    ridge = make_pipeline(pre, Ridge(alpha=4.0, random_state=0))
    ridge.fit(train_feat[feature_cols], y, ridge__sample_weight=weights)
    results["ridge_features_city" if include_city else "ridge_features"] = ridge.predict(val_feat[feature_cols])

    pre = _make_preprocessor(numeric, categorical)
    huber = make_pipeline(pre, HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=1000))
    huber.fit(train_feat[feature_cols], y, huberregressor__sample_weight=weights)
    results["huber_features_city" if include_city else "huber_features"] = huber.predict(val_feat[feature_cols])

    gbr_features = train_feat[feature_cols].copy()
    gbr_val = val_feat[feature_cols].copy()
    if categorical:
        gbr_features = pd.get_dummies(gbr_features, columns=list(categorical), dummy_na=True)
        gbr_val = pd.get_dummies(gbr_val, columns=list(categorical), dummy_na=True)
        gbr_val = gbr_val.reindex(columns=gbr_features.columns, fill_value=0.0)
    gbr_features = gbr_features.replace([np.inf, -np.inf], np.nan)
    gbr_val = gbr_val.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(gbr_features)
    x_val = imputer.transform(gbr_val)
    gbr = HistGradientBoostingRegressor(
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
    gbr.fit(x_train, y, sample_weight=weights)
    results["smooth_gbr_city" if include_city else "smooth_gbr"] = gbr.predict(x_val)
    return results


def _clip_predictions(preds: Mapping[str, np.ndarray], low: float = 3.0, high: float = 260.0) -> Dict[str, np.ndarray]:
    return {name: np.clip(np.asarray(pred, dtype=np.float64), low, high) for name, pred in preds.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv, usecols=USE_COLUMNS, nrows=args.max_rows or None)
    train_df = _apply_manifest(df, args.train_manifest).reset_index(drop=True)
    val_df = _apply_manifest(df, args.val_manifest).reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise SystemExit(f"Empty split: train={len(train_df)} val={len(val_df)}")

    y_train = train_df["gt_mean"].to_numpy(dtype=np.float64)
    y_val = val_df["gt_mean"].to_numpy(dtype=np.float64)

    print(f"Train instances: {len(train_df):,} | Val instances: {len(val_df):,}")
    print(f"Train cities: {sorted(train_df['city'].unique().tolist())}")
    print(f"Val cities:   {sorted(val_df['city'].unique().tolist())}")

    predictions: Dict[str, np.ndarray] = {
        "raw_physical": val_df["height_physical_mean"].to_numpy(dtype=np.float64),
    }
    scale, bias = _fit_huber_affine(
        train_df["height_physical_mean"].to_numpy(),
        y_train,
        _sample_weights(y_train),
    )
    predictions["global_huber_affine"] = scale * val_df["height_physical_mean"].to_numpy(dtype=np.float64) + bias
    city_pred, city_params = _predict_city_affine(train_df, val_df)
    predictions["city_huber_affine"] = city_pred
    predictions.update(_fit_models(train_df, val_df, include_city=False))
    predictions.update(_fit_models(train_df, val_df, include_city=True))
    predictions = _clip_predictions(predictions)

    overall = [(name, _metrics(y_val, pred)) for name, pred in predictions.items()]
    overall = sorted(overall, key=lambda item: (item[1]["rmse"], item[1]["mae"]))
    _print_metrics("Overall development validation", overall)

    best_name = overall[0][0]
    best_pred = predictions[best_name]
    city_rows = []
    for city, city_val in val_df.groupby("city", sort=True):
        idx = city_val.index.to_numpy()
        city_rows.append((str(city), _metrics(y_val[idx], best_pred[idx])))
    _print_metrics(f"City metrics for best model: {best_name}", city_rows)

    bin_rows = []
    bin_idx = _height_bin_indices(y_val)
    for i in range(len(HEIGHT_BINS) + 1):
        mask = bin_idx == i
        if i == 0:
            name = f"0-{HEIGHT_BINS[0]:g}m"
        elif i < len(HEIGHT_BINS):
            name = f"{HEIGHT_BINS[i-1]:g}-{HEIGHT_BINS[i]:g}m"
        else:
            name = f">{HEIGHT_BINS[-1]:g}m"
        bin_rows.append((name, _metrics(y_val[mask], best_pred[mask])))
    _print_metrics(f"Height-bin metrics for best model: {best_name}", bin_rows)

    print("\n[Calibration parameters]")
    print(f"global_huber_affine: scale={scale:.6f}, bias={bias:.6f}")
    for city, (s, b) in sorted(city_params.items()):
        print(f"{city:<12} scale={s:.6f}, bias={b:.6f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "metrics_csv": str(args.metrics_csv),
            "train_instances": int(len(train_df)),
            "val_instances": int(len(val_df)),
            "overall": {name: m for name, m in overall},
            "best_model": best_name,
            "best_city_metrics": {name: m for name, m in city_rows},
            "best_height_bin_metrics": {name: m for name, m in bin_rows},
            "global_affine": {"scale": float(scale), "bias": float(bias)},
            "city_affine": {city: {"scale": float(s), "bias": float(b)} for city, (s, b) in city_params.items()},
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved summary: {args.output_json}")


if __name__ == "__main__":
    main()
