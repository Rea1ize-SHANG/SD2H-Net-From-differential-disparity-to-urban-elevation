#!/usr/bin/env python3
"""Rebuild UnifiedBH delta caches and instance metrics from cached disparity.

This is the fast path used after labels or tile lists change but the stereo
disparity cache is still valid. It avoids reading the large left/right images
and recomputes local-ground disparity, differential disparity, and
instance-level metrics using either label connected components or the building
instance masks consumed by height training.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import numpy as np
import skimage.io
from scipy.ndimage import find_objects
from skimage.measure import label as connected_components
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.height_datasets import GF7_CITY_K_MAPPING, output_core_name, resolve_city_dir
from precompute_unifiedbh_disp_delta import (
    _build_delta_and_instances,
    _height_bin_rows,
    _regression_stats,
    _write_csv,
)


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


def _iter_raster_files(folder: Path) -> Iterable[Path]:
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png"):
        yield from folder.glob(pattern)


def _read_label(path: Path) -> np.ndarray:
    arr = skimage.io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where(np.abs(arr) <= 1.0e6, arr, 0.0)
    return arr.astype(np.float32, copy=False)


def _read_mask(path: Path) -> np.ndarray:
    arr = skimage.io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where((arr >= 0) & (arr <= np.iinfo(np.int32).max), arr, 0)
    return arr.astype(np.int64, copy=False)


def _collect_file_map(data_root: Path, city: str, kind: str) -> dict:
    folder = data_root / kind / resolve_city_dir(city, kind)
    if not folder.exists():
        return {}
    return {output_core_name(str(path)): path for path in _iter_raster_files(folder)}


def _delta_paths(cache_root: Path, city: str, core_name: str) -> dict:
    return {
        "disp": cache_root / "disp_npy" / city / f"Disp_{core_name}.npy",
        "abs": cache_root / "delta_abs_npy" / city / f"DeltaDispAbs_{core_name}.npy",
        "ground": cache_root / "local_ground_npy" / city / f"LocalGroundDisp_{core_name}.npy",
    }


def _safe_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values)]


def _build_instance_rows_from_cached_fast(disp, delta, signed, reverse, abs_delta, ground, mask, height, k_value, cfg):
    records = []
    min_pixels = int(cfg["min_instance_pixels"])
    min_height = float(cfg["label_min_height"])
    for inst_id, bbox in enumerate(find_objects(mask), start=1):
        if bbox is None:
            continue
        local_inst = mask[bbox] == inst_id
        pixel_count = int(local_inst.sum())
        if pixel_count < min_pixels:
            continue

        label_vals = _safe_values(height[bbox][local_inst])
        label_vals = label_vals[label_vals > min_height]
        disp_vals = _safe_values(disp[bbox][local_inst])
        relu_vals = _safe_values(delta[bbox][local_inst])
        signed_vals = _safe_values(signed[bbox][local_inst])
        reverse_vals = _safe_values(reverse[bbox][local_inst])
        abs_vals = _safe_values(abs_delta[bbox][local_inst])
        ground_vals = _safe_values(ground[bbox][local_inst])
        if label_vals.size == 0 or disp_vals.size == 0 or abs_vals.size == 0 or ground_vals.size == 0:
            continue

        ground_disp = float(np.percentile(ground_vals, 50))
        disp_p50 = float(np.percentile(disp_vals, 50))
        records.append({
            "inst_id": inst_id,
            "pixel_count": pixel_count,
            "gt_mean": float(label_vals.mean()),
            "gt_p50": float(np.percentile(label_vals, 50)),
            "gt_p95": float(np.percentile(label_vals, 95)),
            "gt_max": float(label_vals.max()),
            "disp_mean": float(disp_vals.mean()),
            "disp_p50": disp_p50,
            "ground_disp": ground_disp,
            "ground_disp_raw": ground_disp,
            "disp_order_gap_raw": disp_p50 - ground_disp,
            "disp_order_bad_raw": bool((disp_p50 - ground_disp) > 0.0),
            "ground_source": "cached",
            "ground_outlier_tol": 0.0,
            "ground_corrected": False,
            "disp_order_gap": disp_p50 - ground_disp,
            "disp_order_bad": bool((disp_p50 - ground_disp) > 0.0),
            "delta_signed_mean": float(signed_vals.mean()) if signed_vals.size else 0.0,
            "delta_signed_p50": float(np.percentile(signed_vals, 50)) if signed_vals.size else 0.0,
            "delta_relu_mean": float(relu_vals.mean()) if relu_vals.size else 0.0,
            "delta_relu_p50": float(np.percentile(relu_vals, 50)) if relu_vals.size else 0.0,
            "delta_abs_mean": float(abs_vals.mean()),
            "delta_abs_p50": float(np.percentile(abs_vals, 50)),
            "delta_reverse_relu_mean": float(reverse_vals.mean()) if reverse_vals.size else 0.0,
            "height_physical_mean": float(k_value * abs_vals.mean()),
            "height_physical_p50": float(k_value * np.percentile(abs_vals, 50)),
            "height_forward_relu_mean": float(k_value * relu_vals.mean()) if relu_vals.size else 0.0,
            "height_reverse_mean": float(k_value * reverse_vals.mean()) if reverse_vals.size else 0.0,
        })

    grounds = np.asarray([row["ground_disp"] for row in records], dtype=np.float32)
    ground_median = float(np.median(grounds)) if grounds.size else 0.0
    ground_mad = float(np.median(np.abs(grounds - ground_median))) if grounds.size else 0.0
    for row in records:
        row["ground_disp_image_median"] = ground_median
        row["ground_disp_image_mad"] = ground_mad
    return records


def _process_item(payload):
    city, core_name, left_path, right_path, label_path, mask_path, disp_path, cache_root_str, cfg = payload
    cache_root = Path(cache_root_str)
    left_path = Path(left_path)
    right_path = Path(right_path)
    label_path = Path(label_path)
    mask_path = Path(mask_path) if mask_path else None
    disp_path = Path(disp_path)
    paths = _delta_paths(cache_root, city, core_name)

    build_args = SimpleNamespace(**cfg)
    height = _read_label(label_path)
    disp = np.load(disp_path).astype(np.float32)
    if disp.shape != height.shape:
        raise ValueError(f"Shape mismatch for {city}/{core_name}: disp={disp.shape}, label={height.shape}")

    if cfg["instance_source"] == "mask":
        if mask_path is None:
            raise FileNotFoundError(f"Missing instance mask for {city}/{core_name}")
        mask = _read_mask(mask_path)
        if mask.shape != height.shape:
            raise ValueError(f"Shape mismatch for {city}/{core_name}: mask={mask.shape}, label={height.shape}")
    else:
        mask = connected_components(height > float(cfg["label_min_height"]), connectivity=2).astype(np.int64)
    k_value = float(GF7_CITY_K_MAPPING[city])
    if bool(cfg["reuse_cached_products"]):
        abs_delta = np.load(paths["abs"]).astype(np.float32)
        ground = np.load(paths["ground"]).astype(np.float32)
        signed = np.where(np.isfinite(abs_delta), ground - disp, np.nan)
        delta = np.maximum(signed, 0.0)
        reverse = np.maximum(-signed, 0.0)
        rows = _build_instance_rows_from_cached_fast(
            disp,
            delta,
            signed,
            reverse,
            abs_delta,
            ground,
            mask,
            height,
            k_value,
            cfg,
        )
    else:
        delta, signed, reverse, abs_delta, ground, rows = _build_delta_and_instances(
            disp,
            mask,
            height,
            k_value,
            build_args,
        )

    for key, arr in [
        ("abs", abs_delta),
        ("ground", ground),
    ]:
        out_path = paths[key]
        if bool(cfg["overwrite"]) or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, arr.astype(np.float32))

    instance_rows = []
    for row in rows:
        row.update(
            {
                "city": city,
                "core_name": core_name,
                "left_path": str(left_path),
                "right_path": str(right_path),
                "label_path": str(label_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "k": k_value,
                "disp_npy": str(disp_path),
                "delta_npy": str(paths["abs"]),
                "delta_abs_npy": str(paths["abs"]),
                "ground_npy": str(paths["ground"]),
            }
        )
        instance_rows.append(row)

    gt_positive = height[np.isfinite(height) & (height > float(cfg["label_min_height"]))]
    manifest_row = {
        "city": city,
        "core_name": core_name,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "label_path": str(label_path),
        "mask_path": str(mask_path) if mask_path is not None else "",
        "k": k_value,
        "instance_count": len(rows),
        "label_positive_pixels": int(gt_positive.size),
        "label_mean": float(gt_positive.mean()) if gt_positive.size else 0.0,
        "label_max": float(gt_positive.max()) if gt_positive.size else 0.0,
        "disp_npy": str(disp_path),
        "delta_npy": str(paths["abs"]),
        "delta_abs_npy": str(paths["abs"]),
        "ground_npy": str(paths["ground"]),
    }
    return manifest_row, instance_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/UnifiedBH/overlap15")
    parser.add_argument(
        "--cache_root",
        default="data/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622",
    )
    parser.add_argument(
        "--source_disp_cache_root",
        default=None,
        help="Optional read-only source for disp_npy; defaults to --cache_root.",
    )
    parser.add_argument(
        "--instance_source",
        choices=["label", "mask"],
        default="label",
        help="Use label connected components or the file masks consumed by height training.",
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument(
        "--max_items_per_city",
        type=int,
        default=0,
        help="Optional cap on matched tiles per city for rapid sensitivity checks (0 keeps all tiles).",
    )
    parser.add_argument("--label_min_height", type=float, default=3.0)
    parser.add_argument("--min_instance_pixels", type=int, default=16)
    parser.add_argument("--min_ring_pixels", type=int, default=32)
    parser.add_argument("--ring_inner_radius", type=int, default=3)
    parser.add_argument("--ring_outer_radius", type=int, default=35)
    parser.add_argument("--fallback_ring_radius", type=int, default=80)
    parser.add_argument("--ground_percentile", type=float, default=50.0)
    parser.add_argument("--correct_ground_outliers", action="store_true", default=True)
    parser.add_argument("--no_correct_ground_outliers", dest="correct_ground_outliers", action="store_false")
    parser.add_argument("--ground_outlier_abs_tol", type=float, default=3.0)
    parser.add_argument("--ground_outlier_mad_scale", type=float, default=3.5)
    parser.add_argument("--ground_outlier_min_instances", type=int, default=3)
    parser.add_argument("--height_bins", nargs="+", type=float, default=[12, 24, 30, 50, 100, 150, 300])
    parser.add_argument("--overwrite", action="store_true", help="Rewrite delta/ground npy files even if they exist.")
    parser.add_argument(
        "--reuse_cached_products",
        action="store_true",
        help="Aggregate reports from existing disparity, delta, and ground arrays without recomputing local ground sampling.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    source_disp_cache_root = Path(args.source_disp_cache_root) if args.source_disp_cache_root else cache_root
    worker_cfg = dict(
        min_instance_pixels=args.min_instance_pixels,
        min_ring_pixels=args.min_ring_pixels,
        ring_inner_radius=args.ring_inner_radius,
        ring_outer_radius=args.ring_outer_radius,
        fallback_ring_radius=args.fallback_ring_radius,
        ground_percentile=args.ground_percentile,
        correct_ground_outliers=args.correct_ground_outliers,
        ground_outlier_abs_tol=args.ground_outlier_abs_tol,
        ground_outlier_mad_scale=args.ground_outlier_mad_scale,
        ground_outlier_min_instances=args.ground_outlier_min_instances,
        label_min_height=args.label_min_height,
        overwrite=args.overwrite,
        reuse_cached_products=args.reuse_cached_products,
        instance_source=args.instance_source,
    )

    for subdir in ("delta_abs_npy", "local_ground_npy"):
        (cache_root / subdir).mkdir(parents=True, exist_ok=True)

    label_items: List[Tuple[str, str, Path, Path, Path, Path | None, Path]] = []
    missing_disp = []
    for city in args.cities:
        if city not in GF7_CITY_K_MAPPING:
            raise ValueError(f"City missing K value: {city}")
        left_map = _collect_file_map(data_root, city, "left")
        right_map = _collect_file_map(data_root, city, "right")
        label_map = _collect_file_map(data_root, city, "label")
        mask_map = _collect_file_map(data_root, city, "mask")
        valid_keys = set(left_map) & set(right_map) & set(label_map)
        if args.instance_source == "mask":
            valid_keys &= set(mask_map)
        valid_keys = sorted(valid_keys)
        if args.max_items_per_city > 0:
            valid_keys = valid_keys[: args.max_items_per_city]
        print(
            f"[*] {city}: left={len(left_map)} right={len(right_map)} "
            f"label={len(label_map)} mask={len(mask_map)} matched={len(valid_keys)} "
            f"instance_source={args.instance_source}"
        )
        for core_name in valid_keys:
            disp_path = _delta_paths(source_disp_cache_root, city, core_name)["disp"]
            if not disp_path.exists():
                missing_disp.append(str(disp_path))
                continue
            label_items.append(
                (
                    city,
                    core_name,
                    left_map[core_name],
                    right_map[core_name],
                    label_map[core_name],
                    mask_map.get(core_name),
                    disp_path,
                )
            )

    if missing_disp:
        print("[!] Missing cached disparity files:")
        for path in missing_disp[:50]:
            print(f"    {path}")
        if len(missing_disp) > 50:
            print(f"    ... {len(missing_disp) - 50} more")
        raise FileNotFoundError(f"Missing {len(missing_disp)} cached disparity files.")

    payloads = [
        (
            city,
            core_name,
            str(left_path),
            str(right_path),
            str(label_path),
            str(mask_path) if mask_path is not None else "",
            str(disp_path),
            str(cache_root),
            worker_cfg,
        )
        for city, core_name, left_path, right_path, label_path, mask_path, disp_path in label_items
    ]
    manifest_rows = []
    instance_rows = []
    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        iterator = map(_process_item, payloads)
        for manifest_row, rows in tqdm(iterator, total=len(payloads), desc="Rebuilding delta/metrics", dynamic_ncols=True):
            manifest_rows.append(manifest_row)
            instance_rows.extend(rows)
    else:
        print(f"[*] Multiprocessing workers: {num_workers}")
        with mp.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(_process_item, payloads, chunksize=4)
            for manifest_row, rows in tqdm(iterator, total=len(payloads), desc="Rebuilding delta/metrics", dynamic_ncols=True):
                manifest_rows.append(manifest_row)
                instance_rows.extend(rows)

    _write_csv(cache_root / "manifest.csv", manifest_rows)
    _write_csv(cache_root / "instance_metrics.csv", instance_rows)

    summary_rows = []
    bin_rows = []
    if instance_rows:
        cities = sorted(set(r["city"] for r in instance_rows))
        edges = [0.0] + [float(x) for x in args.height_bins]
        for city in ["ALL"] + cities:
            rows = instance_rows if city == "ALL" else [r for r in instance_rows if r["city"] == city]
            y_true = np.array([r["gt_mean"] for r in rows], dtype=np.float64)
            y_pred = np.array([r["height_physical_mean"] for r in rows], dtype=np.float64)
            y_reverse = np.array([r["height_reverse_mean"] for r in rows], dtype=np.float64)
            summary_rows.append({"city": city, "prediction": "height_physical_mean", **_regression_stats(y_true, y_pred)})
            summary_rows.append({"city": city, "prediction": "height_reverse_mean", **_regression_stats(y_true, y_reverse)})
            for bin_row in _height_bin_rows(y_true, y_pred, edges):
                bin_rows.append({"city": city, "prediction": "height_physical_mean", **bin_row})
            for bin_row in _height_bin_rows(y_true, y_reverse, edges):
                bin_rows.append({"city": city, "prediction": "height_reverse_mean", **bin_row})

    _write_csv(cache_root / "physical_metrics_summary.csv", summary_rows)
    _write_csv(cache_root / "physical_metrics_by_bin.csv", bin_rows)

    print(f"[*] Rebuilt samples: {len(manifest_rows)}")
    print(f"[*] Rebuilt instance rows: {len(instance_rows)}")
    print(f"[*] Output: {cache_root}")
    for row in summary_rows:
        if row.get("city") == "ALL":
            print(
                f"    {row['prediction']}: MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} "
                f"R2={row['r2']:.3f} Bias={row['bias']:+.3f} N={row['count']}"
            )


if __name__ == "__main__":
    main()
