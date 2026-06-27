#!/usr/bin/env python3
"""Rebuild UnifiedBH delta caches and instance metrics from cached disparity.

This is the fast path used after labels or tile lists change but the stereo
disparity cache is still valid. It avoids reading the large left/right images
and recomputes label-dependent local-ground disparity, differential disparity,
and instance-level metrics directly from the current labels.
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


def _collect_file_map(data_root: Path, city: str, kind: str) -> dict:
    folder = data_root / kind / resolve_city_dir(city, kind)
    if not folder.exists():
        return {}
    return {output_core_name(str(path)): path for path in _iter_raster_files(folder)}


def _delta_paths(cache_root: Path, city: str, core_name: str) -> dict:
    return {
        "disp": cache_root / "disp_npy" / city / f"Disp_{core_name}.npy",
        "delta": cache_root / "delta_relu_npy" / city / f"DeltaDisp_{core_name}.npy",
        "signed": cache_root / "delta_signed_npy" / city / f"DeltaDispSigned_{core_name}.npy",
        "reverse": cache_root / "delta_reverse_relu_npy" / city / f"DeltaDispReverse_{core_name}.npy",
        "abs": cache_root / "delta_abs_npy" / city / f"DeltaDispAbs_{core_name}.npy",
        "ground": cache_root / "local_ground_npy" / city / f"LocalGroundDisp_{core_name}.npy",
    }


def _process_item(payload):
    city, core_name, left_path, right_path, label_path, cache_root_str, cfg = payload
    cache_root = Path(cache_root_str)
    left_path = Path(left_path)
    right_path = Path(right_path)
    label_path = Path(label_path)
    paths = _delta_paths(cache_root, city, core_name)

    build_args = SimpleNamespace(**cfg)
    height = _read_label(label_path)
    disp = np.load(paths["disp"]).astype(np.float32)
    if disp.shape != height.shape:
        raise ValueError(f"Shape mismatch for {city}/{core_name}: disp={disp.shape}, label={height.shape}")

    mask = connected_components(height > float(cfg["label_min_height"]), connectivity=2).astype(np.int64)
    k_value = float(GF7_CITY_K_MAPPING[city])
    delta, signed, reverse, abs_delta, ground, rows = _build_delta_and_instances(
        disp,
        mask,
        height,
        k_value,
        build_args,
    )

    for key, arr in [
        ("delta", delta),
        ("signed", signed),
        ("reverse", reverse),
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
                "k": k_value,
                "disp_npy": str(paths["disp"]),
                "delta_npy": str(paths["abs"]),
                "delta_relu_npy": str(paths["delta"]),
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
        "k": k_value,
        "instance_count": len(rows),
        "label_positive_pixels": int(gt_positive.size),
        "label_mean": float(gt_positive.mean()) if gt_positive.size else 0.0,
        "label_max": float(gt_positive.max()) if gt_positive.size else 0.0,
        "disp_npy": str(paths["disp"]),
        "delta_npy": str(paths["abs"]),
        "delta_relu_npy": str(paths["delta"]),
        "delta_signed_npy": str(paths["signed"]),
        "delta_reverse_npy": str(paths["reverse"]),
        "delta_abs_npy": str(paths["abs"]),
        "ground_npy": str(paths["ground"]),
    }
    return manifest_row, instance_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15")
    parser.add_argument(
        "--cache_root",
        default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622",
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
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
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
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
    )

    for subdir in ("delta_relu_npy", "delta_signed_npy", "delta_reverse_relu_npy", "delta_abs_npy", "local_ground_npy"):
        (cache_root / subdir).mkdir(parents=True, exist_ok=True)

    label_items: List[Tuple[str, str, Path, Path, Path]] = []
    missing_disp = []
    for city in args.cities:
        if city not in GF7_CITY_K_MAPPING:
            raise ValueError(f"City missing K value: {city}")
        left_map = _collect_file_map(data_root, city, "left")
        right_map = _collect_file_map(data_root, city, "right")
        label_map = _collect_file_map(data_root, city, "label")
        valid_keys = sorted(set(left_map) & set(right_map) & set(label_map))
        print(
            f"[*] {city}: left={len(left_map)} right={len(right_map)} "
            f"label={len(label_map)} matched={len(valid_keys)}"
        )
        for core_name in valid_keys:
            paths = _delta_paths(cache_root, city, core_name)
            if not paths["disp"].exists():
                missing_disp.append(str(paths["disp"]))
                continue
            label_items.append((city, core_name, left_map[core_name], right_map[core_name], label_map[core_name]))

    if missing_disp:
        print("[!] Missing cached disparity files:")
        for path in missing_disp[:50]:
            print(f"    {path}")
        if len(missing_disp) > 50:
            print(f"    ... {len(missing_disp) - 50} more")
        raise FileNotFoundError(f"Missing {len(missing_disp)} cached disparity files.")

    payloads = [
        (city, core_name, str(left_path), str(right_path), str(label_path), str(cache_root), worker_cfg)
        for city, core_name, left_path, right_path, label_path in label_items
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
