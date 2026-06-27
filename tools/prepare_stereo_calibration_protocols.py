#!/usr/bin/env python3
"""Prepare leakage-safe stereo-to-height calibration protocols.

This script creates train/validation instance manifests and Hydra configs for
two public-facing UnifiedBH adaptation protocols:

1. quality_diverse_cross_city:
   Fit all calibration coefficients from a small set of quality-diverse
   training cities and validate on the remaining cities.
2. city_stratified_ratio:
   Fit calibration coefficients from a stratified subset in each city and
   validate on the remaining instances. Multiple ratios can be generated in one
   pass, e.g. 10/90, 20/80, and 50/50.

The generated configs deliberately avoid carrying calibration coefficients that
were fitted from the full dataset. Validation instances are never used for
scale/bias or instance-feature coefficient estimation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
from omegaconf import OmegaConf


DEFAULT_CITIES = [
    "HZ_049",
    "HZ_050",
    "HZ_057",
    "HZ_DG_058",
    "SZ_060",
    "SZ_061",
    "HK_062",
    "SZ_DG_038",
    "PRE_039",
]

DEFAULT_HEIGHT_EDGES = [12.0, 24.0, 30.0, 50.0, 100.0, 150.0, 300.0]


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(row: Mapping[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                pass
    return float(default)


def _to_int(row: Mapping[str, str], *names: str, default: int = 0) -> int:
    return int(round(_to_float(row, *names, default=default)))


def _tile_key(row: Mapping[str, str]) -> str:
    return f"{row['city']}/{row['core_name']}"


def _instance_key(row: Mapping[str, str]) -> Tuple[str, str, int]:
    return row["city"], row["core_name"], _to_int(row, "inst_id", "instance_id")


def _height_bin(height: float, edges: Sequence[float]) -> int:
    return int(np.digitize([height], np.asarray(edges, dtype=np.float64), right=True)[0])


def _height_bin_name(bin_idx: int, edges: Sequence[float]) -> str:
    lower = 0.0 if bin_idx == 0 else float(edges[bin_idx - 1])
    upper = float(edges[bin_idx]) if bin_idx < len(edges) else math.inf
    if math.isinf(upper):
        return f">{lower:g}m"
    return f"{lower:g}-{upper:g}m"


def _load_base_ignored(base_manifest: Path) -> Dict[str, Set[int]]:
    with base_manifest.open("r") as f:
        payload = json.load(f)
    raw = payload.get("ignore_instances", payload)
    ignored: Dict[str, Set[int]] = {}
    for key, ids in raw.items():
        clean = {int(x) for x in ids if int(x) > 0}
        if clean:
            ignored[str(key)] = clean
    return ignored


def _merge_ignore_instances(*manifests: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
    merged: Dict[str, Set[int]] = defaultdict(set)
    for manifest in manifests:
        for key, ids in manifest.items():
            for inst_id in ids:
                inst_id = int(inst_id)
                if inst_id > 0:
                    merged[str(key)].add(inst_id)
    return {key: sorted(ids) for key, ids in sorted(merged.items()) if ids}


def _load_metrics(metrics_csv: Path, base_ignored: Mapping[str, Set[int]], min_height: float) -> List[dict]:
    rows: List[dict] = []
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = row["city"]
            core = row["core_name"]
            inst_id = _to_int(row, "inst_id")
            gt = _to_float(row, "gt_mean")
            if gt < min_height:
                continue
            if inst_id in base_ignored.get(f"{city}/{core}", set()):
                continue
            row = dict(row)
            row["_inst_id"] = inst_id
            row["_gt"] = gt
            row["_tile_key"] = f"{city}/{core}"
            row["_key"] = (city, core, inst_id)
            rows.append(row)
    return rows


def _select_cross_city(rows: Sequence[dict], train_cities: Sequence[str]) -> Tuple[Set[Tuple[str, str, int]], Set[Tuple[str, str, int]]]:
    train_city_set = set(train_cities)
    train, val = set(), set()
    for row in rows:
        if row["city"] in train_city_set:
            train.add(row["_key"])
        else:
            val.add(row["_key"])
    return train, val


def _select_cross_city_within(
    rows: Sequence[dict],
    cities: Sequence[str],
    train_cities: Sequence[str],
) -> Tuple[Set[Tuple[str, str, int]], Set[Tuple[str, str, int]]]:
    """Select cross-city splits only from an explicit city universe."""
    city_set = set(cities)
    train_city_set = set(train_cities)
    train, val = set(), set()
    for row in rows:
        city = row["city"]
        if city not in city_set:
            continue
        if city in train_city_set:
            train.add(row["_key"])
        else:
            val.add(row["_key"])
    return train, val


def _ratio_tag(ratio: float) -> str:
    train_pct = int(round(float(ratio) * 100.0))
    val_pct = max(0, 100 - train_pct)
    return f"{train_pct}_{val_pct}"


def _city_tag(cities: Sequence[str]) -> str:
    return "_".join(city.lower() for city in cities)


def _select_stratified_fewshot(
    rows: Sequence[dict],
    cities: Sequence[str],
    ratio: float,
    edges: Sequence[float],
    seed: int,
    high_min_train: int,
    high_min_fraction: float,
    min_val_per_bin: int,
    high_min_val: int,
    high_val_fraction: float,
) -> Tuple[Set[Tuple[str, str, int]], Set[Tuple[str, str, int]]]:
    rng = np.random.default_rng(seed)
    by_city_bin: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in rows:
        if row["city"] not in cities:
            continue
        by_city_bin[(row["city"], _height_bin(row["_gt"], edges))].append(row)

    train: Set[Tuple[str, str, int]] = set()
    all_keys = {row["_key"] for row in rows if row["city"] in cities}
    for city in cities:
        for bin_idx in range(len(edges) + 1):
            group = by_city_bin.get((city, bin_idx), [])
            if not group:
                continue
            count = len(group)
            bin_lower = 0.0 if bin_idx == 0 else float(edges[bin_idx - 1])
            if count == 1:
                sample_count = 1 if ratio >= 0.5 else 0
            else:
                sample_count = max(1, int(round(count * ratio)))
                val_floor = max(1, int(min_val_per_bin))
                if bin_lower >= 30.0:
                    sample_count = max(sample_count, min(count, high_min_train))
                    sample_count = max(sample_count, int(math.ceil(count * high_min_fraction)))
                    val_floor = max(val_floor, int(high_min_val))
                    val_floor = max(val_floor, int(math.ceil(count * high_val_fraction)))
                val_floor = min(val_floor, count - 1)
                sample_count = min(sample_count, count - val_floor)
            if sample_count > 0:
                indices = rng.choice(count, size=sample_count, replace=False)
                for idx in indices:
                    train.add(group[int(idx)]["_key"])
    val = all_keys - train
    return train, val


def _make_manifest(
    rows: Sequence[dict],
    keep_keys: Set[Tuple[str, str, int]],
    base_ignored: Mapping[str, Set[int]],
    source_base_manifest: Path,
    source_metrics_csv: Path,
    split_name: str,
    split_type: str,
    train_cities: Sequence[str],
    val_cities: Sequence[str],
    seed: int,
    edges: Sequence[float],
) -> dict:
    ignore: Dict[str, List[int]] = defaultdict(list)
    valid_tiles: Set[str] = set()
    summary = defaultdict(lambda: {"total": 0, "kept": 0, "ignored": 0, "bins": defaultdict(lambda: {"total": 0, "kept": 0})})

    for row in rows:
        city = row["city"]
        key = row["_key"]
        tile = row["_tile_key"]
        inst_id = row["_inst_id"]
        bin_idx = _height_bin(row["_gt"], edges)
        bin_name = _height_bin_name(bin_idx, edges)
        summary[city]["total"] += 1
        summary[city]["bins"][bin_name]["total"] += 1
        if key in keep_keys:
            valid_tiles.add(tile)
            summary[city]["kept"] += 1
            summary[city]["bins"][bin_name]["kept"] += 1
        else:
            ignore[tile].append(inst_id)
            summary[city]["ignored"] += 1

    clean_summary = {}
    for city, data in sorted(summary.items()):
        clean_summary[city] = {
            "total": int(data["total"]),
            "kept": int(data["kept"]),
            "ignored": int(data["ignored"]),
            "bins": {
                name: {"total": int(vals["total"]), "kept": int(vals["kept"])}
                for name, vals in sorted(data["bins"].items())
            },
        }

    split_ignore = {key: sorted(ids) for key, ids in sorted(ignore.items()) if ids}
    merged_ignore = _merge_ignore_instances(base_ignored, split_ignore)

    return {
        "split_name": split_name,
        "split_type": split_type,
        "seed": int(seed),
        "source_base_manifest": str(source_base_manifest),
        "source_metrics_csv": str(source_metrics_csv),
        "train_cities": list(train_cities),
        "val_cities": list(val_cities),
        "height_bin_edges": [float(x) for x in edges],
        "summary": clean_summary,
        "ignore_instances": merged_ignore,
        "split_ignore_instances": split_ignore,
        "base_ignore_instances": {key: sorted(ids) for key, ids in sorted(base_ignored.items()) if ids},
        "valid_tiles": sorted(valid_tiles),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_summary_csv(path: Path, manifest_payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["city", "bin", "total", "kept", "ignored"])
        for city, stats in manifest_payload["summary"].items():
            writer.writerow([city, "ALL", stats["total"], stats["kept"], stats["ignored"]])
            for bin_name, bin_stats in stats["bins"].items():
                writer.writerow([
                    city,
                    bin_name,
                    bin_stats["total"],
                    bin_stats["kept"],
                    bin_stats["total"] - bin_stats["kept"],
                ])


def _write_calibration_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "city",
        "train_instances",
        "high_instances",
        "height_physical_scale",
        "height_physical_bias",
        "config_path",
        "train_manifest",
        "val_manifest",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _poly_terms_from_rows(rows: Sequence[dict], x_clip: float) -> np.ndarray:
    x_safe = np.asarray([_to_float(r, "height_physical_mean") for r in rows], dtype=np.float64)
    x = np.minimum(x_safe, x_clip) if x_clip > 0 else x_safe
    pixel_count = np.asarray([_to_float(r, "pixel_count") for r in rows], dtype=np.float64)
    log_area = np.log1p(np.maximum(pixel_count, 0.0))
    sqrt_area = np.sqrt(np.maximum(pixel_count, 0.0))
    return np.stack(
        [
            x,
            x * x,
            x * x * x,
            np.sqrt(np.maximum(x, 0.0) + 1e-6),
            log_area,
            log_area * log_area,
            x * log_area,
            sqrt_area,
            x * sqrt_area,
            np.ones_like(x),
        ],
        axis=1,
    )


def _poly_base_from_rows(rows: Sequence[dict], coeffs: Sequence[float], x_clip: float) -> np.ndarray:
    terms = _poly_terms_from_rows(rows, x_clip)
    return terms @ np.asarray(coeffs, dtype=np.float64)


def _feature_matrix(
    rows: Sequence[dict],
    poly_base: np.ndarray,
    use_signed_ground: bool = True,
    physical_scale: float = 1.0,
    physical_bias: float = 0.0,
) -> np.ndarray:
    physical = np.asarray([_to_float(r, "height_physical_mean") for r in rows], dtype=np.float64)
    physical = float(physical_scale) * physical + float(physical_bias)
    delta = np.asarray([_to_float(r, "delta_abs_mean", "delta_mean") for r in rows], dtype=np.float64)
    disp = np.asarray([_to_float(r, "disp_mean") for r in rows], dtype=np.float64)
    ground = np.asarray([_to_float(r, "ground_disp") for r in rows], dtype=np.float64)
    signed_delta = ground - disp
    order_bad = (signed_delta < 0).astype(np.float64)
    pixel_count = np.asarray([_to_float(r, "pixel_count") for r in rows], dtype=np.float64)
    log_area = np.log1p(np.maximum(pixel_count, 0.0))
    sqrt_area = np.sqrt(np.maximum(pixel_count, 0.0))
    sqrt_phys = np.sqrt(np.maximum(physical, 0.0) + 1e-6)
    if use_signed_ground:
        values = [
            physical,
            delta,
            poly_base,
            log_area,
            sqrt_area,
            pixel_count,
            physical * log_area,
            delta * log_area,
            physical * sqrt_area,
            delta * sqrt_area,
            physical * physical,
            delta * delta,
            sqrt_phys,
            disp,
            ground,
            signed_delta,
            np.abs(signed_delta),
            order_bad,
            np.zeros_like(physical),
        ]
    else:
        values = [
            physical,
            delta,
            poly_base,
            log_area,
            sqrt_area,
            pixel_count,
            physical * log_area,
            delta * log_area,
            physical * sqrt_area,
            delta * sqrt_area,
            physical * physical,
            delta * delta,
            sqrt_phys,
            disp,
            disp,
            ground,
            ground - disp,
            np.zeros_like(physical),
            np.zeros_like(physical),
        ]
    return np.stack(values, axis=1)


def _weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    ridge: float = 0.0,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if fit_intercept:
        design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    else:
        design = x
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64).clip(min=1.0e-8))
        design = design * w[:, None]
        y = y * w
    reg = float(ridge) * np.eye(design.shape[1], dtype=np.float64)
    if fit_intercept:
        reg[-1, -1] = 0.0
    beta = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    if fit_intercept:
        return beta[:-1], float(beta[-1])
    return beta, 0.0


def _balanced_weights(y: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    bins = np.digitize(y, np.asarray(edges, dtype=np.float64), right=True)
    weights = np.ones_like(y, dtype=np.float64)
    for bin_idx in np.unique(bins):
        keep = bins == bin_idx
        if keep.any():
            weights[keep] = 1.0 / float(keep.sum())
    weights *= len(weights) / max(float(weights.sum()), 1.0e-8)
    return weights


def _fit_calibration(rows: Sequence[dict], keep_keys: Set[Tuple[str, str, int]], base_cfg) -> dict:
    train_rows = [r for r in rows if r["_key"] in keep_keys]
    if len(train_rows) < 20:
        raise RuntimeError(f"Too few training instances for calibration: {len(train_rows)}")

    y = np.asarray([_to_float(r, "gt_mean") for r in train_rows], dtype=np.float64)
    physical = np.asarray([_to_float(r, "height_physical_mean") for r in train_rows], dtype=np.float64)
    scale_vec, bias = _weighted_linear_fit(physical, y, fit_intercept=True)

    poly_x_clip = float(base_cfg.height_poly_area_x_clip)
    poly_terms = _poly_terms_from_rows(train_rows, poly_x_clip)
    poly_coeffs, _ = _weighted_linear_fit(
        poly_terms,
        y,
        weights=_balanced_weights(y, [float(x) for x in base_cfg.height_eval_bins]),
        ridge=1.0e-3,
        fit_intercept=False,
    )
    poly_base = _poly_base_from_rows(train_rows, poly_coeffs, poly_x_clip)
    feats = _feature_matrix(
        train_rows,
        poly_base,
        use_signed_ground=bool(base_cfg.height_instance_feature_use_signed_ground),
        physical_scale=float(scale_vec[0]),
        physical_bias=float(bias),
    )
    weights = _balanced_weights(y, [float(x) for x in base_cfg.height_eval_bins])
    coeffs, feat_bias = _weighted_linear_fit(
        feats,
        y,
        weights=weights,
        ridge=1.0e-3,
        fit_intercept=True,
    )

    high_mask = y > 30.0
    if int(high_mask.sum()) >= 20:
        high_coeffs, high_bias = _weighted_linear_fit(
            feats[high_mask],
            y[high_mask],
            ridge=1.0e-3,
            fit_intercept=True,
        )
    else:
        high_coeffs, high_bias = coeffs, feat_bias

    return {
        "height_physical_scale": float(scale_vec[0]),
        "height_physical_bias": float(bias),
        "height_poly_area_coeffs": [float(x) for x in poly_coeffs],
        "height_instance_feature_linear_coeffs": [float(x) for x in coeffs],
        "height_instance_feature_linear_bias": float(feat_bias),
        "height_instance_feature_high_coeffs": [float(x) for x in high_coeffs],
        "height_instance_feature_high_bias": float(high_bias),
        "calibration_train_instances": int(len(train_rows)),
        "calibration_high_instances": int(high_mask.sum()),
    }


def _apply_calibration_to_cfg(cfg, calibration: Mapping[str, object]) -> None:
    for key in [
        "height_physical_scale",
        "height_physical_bias",
        "height_poly_area_coeffs",
        "height_instance_feature_linear_coeffs",
        "height_instance_feature_linear_bias",
        "height_instance_feature_high_coeffs",
        "height_instance_feature_high_bias",
    ]:
        setattr(cfg, key, calibration[key])


def _write_config(path: Path, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path), resolve=True)


def _make_config(
    base_cfg,
    project_name: str,
    save_path: str,
    train_cities: Sequence[str],
    val_cities: Sequence[str],
    train_manifest: Path,
    val_manifest: Path,
    calibration: Mapping[str, object],
    sampler_key: str,
    batch_size: int,
    val_batch_size: int,
    num_gpu: int,
) -> object:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.project_name = project_name
    cfg.save_path = save_path
    cfg.train_cities = list(train_cities)
    cfg.val_cities = list(val_cities)
    cfg.train_instance_filter_manifest = str(train_manifest)
    cfg.val_instance_filter_manifest = str(val_manifest)
    cfg.sample_weight_cache_key_override = sampler_key
    cfg.batch_size = int(batch_size)
    cfg.val_batch_size = int(val_batch_size)
    cfg.num_gpu = int(num_gpu)
    cfg.mixed_precision = "bf16"
    cfg.use_weighted_sampler = False
    cfg.epochs = 12
    cfg.early_stop_min_epoch = 4
    cfg.early_stop_patience = 3
    cfg.head_lr = 6.0e-5
    cfg.calibration_lr = 1.5e-4
    cfg.max_grad_norm = 0.5

    cfg.weak_label_soft_loss_enabled = True
    cfg.weak_label_soft_abs_margin = 1.5
    cfg.weak_label_soft_rel_margin = 0.12
    cfg.weak_label_soft_margin_max = 10.0
    cfg.weak_label_ordinal_weight = 0.22
    cfg.weak_label_ordinal_min_gap = 6.0
    cfg.weak_label_ordinal_margin = 0.5
    cfg.weak_label_ordinal_max_pairs = 4096
    cfg.weak_label_ordinal_physical_weight = True
    cfg.weak_label_ordinal_physical_min_weight = 0.25
    cfg.weak_label_ordinal_gap_bins = [6.0, 12.0, 24.0, 48.0, 96.0]

    cfg.height_instance_correction_clip = 0.35
    cfg.height_instance_log_ratio_clip = 0.35
    cfg.height_instance_log_ratio_target_clip = 0.35
    cfg.height_instance_low_log_ratio_clip = 0.04
    cfg.height_instance_low_positive_clip = 0.04
    cfg.physical_consistency_min_weight = 0.45
    cfg.global_loss_weight = 0.50
    cfg.stratified_loss_weight = 0.50
    cfg.high_under_weight = 0.85
    cfg.high_bias_weight = 0.25
    cfg.best_r2_score_weight = 3.0
    cfg.height_train_output_keys = [
        "height",
        "height_base",
        "height_physical",
        "height_correction",
        "height_learned_physical_floor_raise",
        "height_learned_physical_floor_fraction",
        "height_pre_learned_physical_floor",
    ]
    cfg.height_instance_feature_map_dtype = "bf16"
    cfg.restore_ckpt = None
    cfg.resume_ckpt = None
    cfg.save_epoch_checkpoints = False
    cfg.save_final_checkpoint = False
    cfg.public_protocol_name = project_name
    cfg.calibration_protocol = "train_split_only"
    cfg.calibration_train_instances = int(calibration["calibration_train_instances"])
    cfg.calibration_high_instances = int(calibration["calibration_high_instances"])
    _apply_calibration_to_cfg(cfg, calibration)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_config", default="config/train_stereo_calib_base.yaml")
    parser.add_argument("--metrics_csv", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622/instance_metrics.csv")
    parser.add_argument("--base_filter_manifest", default="checkpoints/stereo_calib_noise_filter_manifest/instance_filter_manifest.json")
    parser.add_argument("--output_root", default="checkpoints/stereo_calibration_protocols_seed724")
    parser.add_argument("--config_dir", default="config")
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--fewshot_ratio", type=float, default=0.10, help="Backward-compatible single ratio.")
    parser.add_argument(
        "--fewshot_ratios",
        nargs="+",
        type=float,
        default=None,
        help="One or more city-stratified train ratios, e.g. 0.10 0.20 0.50.",
    )
    parser.add_argument("--fewshot_high_min_train", type=int, default=20)
    parser.add_argument("--fewshot_high_min_fraction", type=float, default=0.20)
    parser.add_argument("--fewshot_min_val_per_bin", type=int, default=1)
    parser.add_argument("--fewshot_high_min_val", type=int, default=1)
    parser.add_argument("--fewshot_high_val_fraction", type=float, default=0.25)
    parser.add_argument(
        "--balanced_protocol_names",
        action="store_true",
        help="Write balanced_city_stratified_* configs/manifests for the weak-label protocol.",
    )
    parser.add_argument("--cross_city_train_city", default="HZ_DG_058", help="Backward-compatible single cross-city training city.")
    parser.add_argument(
        "--cross_city_train_cities",
        nargs="+",
        default=None,
        help="Quality-diverse training cities for cross-city transfer.",
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--min_height", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--write_pooled_fewshot_config", action="store_true")
    parser.add_argument("--write_citywise_configs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.base_config)
    metrics_csv = Path(args.metrics_csv)
    base_manifest = Path(args.base_filter_manifest)
    output_root = Path(args.output_root)
    config_dir = Path(args.config_dir)
    edges = [float(x) for x in DEFAULT_HEIGHT_EDGES]

    base_ignored = _load_base_ignored(base_manifest)
    rows = _load_metrics(metrics_csv, base_ignored, min_height=args.min_height)
    cities = list(args.cities)

    cross_train_cities = list(args.cross_city_train_cities or [args.cross_city_train_city])
    unknown_train_cities = [city for city in cross_train_cities if city not in cities]
    if unknown_train_cities:
        raise ValueError(f"Cross-city training cities are not in --cities: {unknown_train_cities}")
    cross_val_cities = [city for city in cities if city not in set(cross_train_cities)]
    cross_train, cross_val = _select_cross_city_within(rows, cities, cross_train_cities)

    fewshot_ratios = args.fewshot_ratios or [args.fewshot_ratio]
    fewshot_splits = []
    for ratio in fewshot_ratios:
        few_train, few_val = _select_stratified_fewshot(
            rows,
            cities=cities,
            ratio=ratio,
            edges=edges,
            seed=args.seed,
            high_min_train=args.fewshot_high_min_train,
            high_min_fraction=args.fewshot_high_min_fraction,
            min_val_per_bin=args.fewshot_min_val_per_bin,
            high_min_val=args.fewshot_high_min_val,
            high_val_fraction=args.fewshot_high_val_fraction,
        )
        fewshot_splits.append((float(ratio), few_train, few_val))

    cross_tag = _city_tag(cross_train_cities)

    protocols = [
        {
            "name": "quality_diverse_cross_city",
            "project": f"QualityDiverseCrossCity_{cross_tag}_To_{len(cross_val_cities)}Cities",
            "train_cities": cross_train_cities,
            "val_cities": cross_val_cities,
            "train_keys": cross_train,
            "val_keys": cross_val,
            "train_manifest": output_root / "quality_diverse_cross_city_train_manifest.json",
            "val_manifest": output_root / "quality_diverse_cross_city_val_manifest.json",
            "config_path": config_dir / "train_quality_diverse_cross_city.yaml",
            "save_path": f"./checkpoints/quality_diverse_cross_city_{cross_tag}_to_{len(cross_val_cities)}cities/",
            "sampler_key": f"quality_diverse_cross_city_{cross_tag}_to_{len(cross_val_cities)}cities",
        }
    ]
    if args.write_pooled_fewshot_config:
        for ratio, few_train, few_val in fewshot_splits:
            tag = _ratio_tag(ratio)
            name_prefix = "balanced_city_stratified" if args.balanced_protocol_names else "city_stratified"
            project_prefix = "BalancedCityStratified" if args.balanced_protocol_names else "CityStratified"
            protocols.append(
                {
                    "name": f"{name_prefix}_{tag}",
                    "project": f"{project_prefix}_{tag}",
                    "train_cities": cities,
                    "val_cities": cities,
                    "train_keys": few_train,
                    "val_keys": few_val,
                    "train_manifest": output_root / f"{name_prefix}_{tag}_train_manifest.json",
                    "val_manifest": output_root / f"{name_prefix}_{tag}_val_manifest.json",
                    "config_path": config_dir / f"train_{name_prefix}_{tag}.yaml",
                    "save_path": f"./checkpoints/{name_prefix}_{tag}/",
                    "sampler_key": f"{name_prefix}_{tag}",
                    "fewshot_ratio": ratio,
                }
            )

    calibration_summary_rows = []
    for protocol in protocols:
        train_payload = _make_manifest(
            rows,
            protocol["train_keys"],
            base_ignored=base_ignored,
            source_base_manifest=base_manifest,
            source_metrics_csv=metrics_csv,
            split_name=f"{protocol['name']}_train",
            split_type="train_only_calibration_manifest",
            train_cities=protocol["train_cities"],
            val_cities=protocol["val_cities"],
            seed=args.seed,
            edges=edges,
        )
        val_payload = _make_manifest(
            rows,
            protocol["val_keys"],
            base_ignored=base_ignored,
            source_base_manifest=base_manifest,
            source_metrics_csv=metrics_csv,
            split_name=f"{protocol['name']}_val",
            split_type="heldout_evaluation_manifest",
            train_cities=protocol["train_cities"],
            val_cities=protocol["val_cities"],
            seed=args.seed,
            edges=edges,
        )
        _write_json(protocol["train_manifest"], train_payload)
        _write_json(protocol["val_manifest"], val_payload)
        _write_summary_csv(output_root / f"{protocol['name']}_train_summary.csv", train_payload)
        _write_summary_csv(output_root / f"{protocol['name']}_val_summary.csv", val_payload)

        calibration = _fit_calibration(rows, protocol["train_keys"], base_cfg)
        cfg = _make_config(
            base_cfg,
            project_name=protocol["project"],
            save_path=protocol["save_path"],
            train_cities=protocol["train_cities"],
            val_cities=protocol["val_cities"],
            train_manifest=protocol["train_manifest"],
            val_manifest=protocol["val_manifest"],
            calibration=calibration,
            sampler_key=protocol["sampler_key"],
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_gpu=args.num_gpu,
        )
        _write_config(protocol["config_path"], cfg)
        calibration_summary_rows.append({
            "protocol": protocol["name"],
            "city": "|".join(protocol["train_cities"]),
            "train_instances": calibration["calibration_train_instances"],
            "high_instances": calibration["calibration_high_instances"],
            "height_physical_scale": calibration["height_physical_scale"],
            "height_physical_bias": calibration["height_physical_bias"],
            "config_path": str(protocol["config_path"]),
            "train_manifest": str(protocol["train_manifest"]),
            "val_manifest": str(protocol["val_manifest"]),
        })

        print(f"\n[{protocol['name']}]")
        print(f"  train manifest: {protocol['train_manifest']}")
        print(f"  val manifest:   {protocol['val_manifest']}")
        print(f"  config:         {protocol['config_path']}")
        print(
            "  calibration instances: "
            f"total={calibration['calibration_train_instances']} "
            f"high>30m={calibration['calibration_high_instances']}"
        )
        print(
            "  physical affine: "
            f"height = {calibration['height_physical_scale']:.6f} * physical "
            f"+ {calibration['height_physical_bias']:.6f}"
        )

    if not args.write_citywise_configs:
        _write_calibration_summary(output_root / "stereo_calibration_coefficients_summary.csv", calibration_summary_rows)
        return

    for city in cities:
        city_ratio, few_train, few_val = fewshot_splits[0]
        city_tag = _ratio_tag(city_ratio)
        city_train = {key for key in few_train if key[0] == city}
        city_val = {key for key in few_val if key[0] == city}
        name = f"city_stratified_{city.lower()}_{city_tag}"
        train_manifest = output_root / f"{name}_train_manifest.json"
        val_manifest = output_root / f"{name}_val_manifest.json"
        config_path = config_dir / f"train_{name}.yaml"
        train_payload = _make_manifest(
            rows,
            city_train,
            base_ignored=base_ignored,
            source_base_manifest=base_manifest,
            source_metrics_csv=metrics_csv,
            split_name=f"{name}_train",
            split_type="citywise_stratified_train_manifest",
            train_cities=[city],
            val_cities=[city],
            seed=args.seed,
            edges=edges,
        )
        val_payload = _make_manifest(
            rows,
            city_val,
            base_ignored=base_ignored,
            source_base_manifest=base_manifest,
            source_metrics_csv=metrics_csv,
            split_name=f"{name}_val",
            split_type="citywise_stratified_heldout_manifest",
            train_cities=[city],
            val_cities=[city],
            seed=args.seed,
            edges=edges,
        )
        _write_json(train_manifest, train_payload)
        _write_json(val_manifest, val_payload)
        _write_summary_csv(output_root / f"{name}_train_summary.csv", train_payload)
        _write_summary_csv(output_root / f"{name}_val_summary.csv", val_payload)

        calibration = _fit_calibration(rows, city_train, base_cfg)
        cfg = _make_config(
            base_cfg,
            project_name=f"CityStratified_{city}_{city_tag}",
            save_path=f"./checkpoints/city_stratified_{city.lower()}_{city_tag}/",
            train_cities=[city],
            val_cities=[city],
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            calibration=calibration,
            sampler_key=f"city_stratified_{city.lower()}_{city_tag}",
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_gpu=args.num_gpu,
        )
        cfg.citywise_protocol = True
        cfg.fewshot_ratio = float(city_ratio)
        _write_config(config_path, cfg)
        calibration_summary_rows.append({
            "protocol": "city_stratified_citywise",
            "city": city,
            "train_instances": calibration["calibration_train_instances"],
            "high_instances": calibration["calibration_high_instances"],
            "height_physical_scale": calibration["height_physical_scale"],
            "height_physical_bias": calibration["height_physical_bias"],
            "config_path": str(config_path),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
        })
        print(f"\n[stereo_calib_fewshot_citywise:{city}]")
        print(f"  train manifest: {train_manifest}")
        print(f"  val manifest:   {val_manifest}")
        print(f"  config:         {config_path}")
        print(
            "  calibration instances: "
            f"total={calibration['calibration_train_instances']} "
            f"high>30m={calibration['calibration_high_instances']}"
        )
        print(
            "  physical affine: "
            f"height = {calibration['height_physical_scale']:.6f} * physical "
            f"+ {calibration['height_physical_bias']:.6f}"
        )

    _write_calibration_summary(output_root / "stereo_calibration_coefficients_summary.csv", calibration_summary_rows)


if __name__ == "__main__":
    main()
