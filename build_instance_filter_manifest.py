import argparse
import csv
import json
import math
import os
import os.path as osp
from glob import glob

import numpy as np
import skimage.io
from skimage.measure import label as connected_components
from tqdm import tqdm

from core.height_datasets import (
    GF7_CITY_K_MAPPING,
    delta_filename_for_subdir,
    output_core_name,
    read_npy_as_float32,
    resolve_city_dir,
)


def read_label(path):
    arr = skimage.io.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where(np.abs(arr) <= 1.0e6, arr, 0.0)
    return arr.astype(np.float32, copy=False)


def bin_name_for_height(height, edges):
    lower = 0.0
    for edge in edges:
        if height <= edge:
            return f"{lower:g}-{edge:g}m"
        lower = edge
    return f">{lower:g}m"


def bin_index_for_height(height, edges):
    return int(np.searchsorted(np.asarray(edges, dtype=np.float32), height, side="left"))


def collect_label_paths(data_root, cities):
    paths = []
    for city in cities:
        label_dir = osp.join(data_root, "label", resolve_city_dir(city, "label"))
        city_paths = []
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png"):
            city_paths.extend(glob(osp.join(label_dir, ext)))
        for path in sorted(city_paths):
            paths.append((city, path))
    return paths


def robust_z(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(values)
    return (values - med) / scale


def npy_path(cache_root, subdir, city, prefix, core_name):
    return osp.join(cache_root, subdir, city, f"{prefix}_{core_name}.npy")


def instance_means(flat_ids, values):
    values = np.asarray(values).reshape(-1)
    valid_values = np.isfinite(values)
    valid = (flat_ids > 0) & valid_values
    if not valid.any():
        return np.zeros(int(flat_ids.max()) + 1, dtype=np.float32)
    sums = np.bincount(flat_ids[valid], weights=values[valid])
    counts = np.bincount(flat_ids[valid])
    out = np.zeros_like(sums, dtype=np.float32)
    keep = counts > 0
    out[keep] = (sums[keep] / counts[keep]).astype(np.float32)
    return out


def instance_medians(flat_ids, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = (flat_ids > 0) & np.isfinite(values)
    max_id = int(flat_ids.max()) if flat_ids.size else 0
    out = np.zeros(max_id + 1, dtype=np.float32)
    if not valid.any():
        return out

    ids = flat_ids[valid].astype(np.int64, copy=False)
    vals = values[valid]
    order = np.lexsort((vals, ids))
    ids = ids[order]
    vals = vals[order]

    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], ids.size]
    unique_ids = ids[starts]
    lengths = ends - starts
    lo = starts + (lengths - 1) // 2
    hi = starts + lengths // 2
    out[unique_ids] = ((vals[lo] + vals[hi]) * 0.5).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build a stratified instance ignore manifest for noisy building-height labels.")
    parser.add_argument("--data_root", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15")
    parser.add_argument("--cities", nargs="+", default=["HZ_049", "HZ_050", "HZ_057"])
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--delta_subdir", default="delta_relu_npy")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--bin_edges", nargs="+", type=float, default=[12, 24, 30, 50, 100, 150, 300])
    parser.add_argument(
        "--filter_ratios_by_bin",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-height-bin fixed filter ratios. Length should match number of bins.",
    )
    parser.add_argument("--filter_ratio_low", type=float, default=0.12)
    parser.add_argument("--filter_ratio_mid", type=float, default=0.10)
    parser.add_argument("--filter_ratio_high", type=float, default=0.06)
    parser.add_argument("--min_keep_per_city_bin", type=int, default=30)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--min_label_height", type=float, default=0.0)
    parser.add_argument("--min_physical_height", type=float, default=0.0)
    parser.add_argument("--high_phys_ratio", type=float, default=0.35)
    parser.add_argument("--low_phys_ratio", type=float, default=3.5)
    parser.add_argument("--disp_order_margin", type=float, default=0.0)
    parser.add_argument(
        "--no_force_bad_disp_order",
        action="store_true",
        help="Do not mark instances bad when median building disparity is larger than local ground disparity.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    label_paths = collect_label_paths(args.data_root, args.cities)
    for city, label_path in tqdm(label_paths, desc="Collecting instances"):
        height = read_label(label_path)
        inst_threshold = max(0.0, float(args.min_label_height))
        inst = connected_components(height > inst_threshold, connectivity=2).astype(np.int32)
        ids = np.unique(inst)
        ids = ids[ids > 0]
        if ids.size == 0:
            continue

        core_name = output_core_name(label_path)
        delta_path = osp.join(
            args.cache_root,
            args.delta_subdir,
            city,
            delta_filename_for_subdir(args.delta_subdir, core_name),
        )
        if not osp.exists(delta_path):
            raise FileNotFoundError(delta_path)
        delta = read_npy_as_float32(delta_path)
        if delta.shape != height.shape:
            raise ValueError(f"Shape mismatch: {label_path} label={height.shape}, delta={delta.shape}")
        disp_path = npy_path(args.cache_root, "disp_npy", city, "Disp", core_name)
        ground_path = npy_path(args.cache_root, "local_ground_npy", city, "LocalGroundDisp", core_name)
        if not osp.exists(disp_path):
            raise FileNotFoundError(disp_path)
        if not osp.exists(ground_path):
            raise FileNotFoundError(ground_path)
        disp = read_npy_as_float32(disp_path)
        ground = read_npy_as_float32(ground_path)
        if disp.shape != height.shape:
            raise ValueError(f"Shape mismatch: {disp_path} disp={disp.shape}, label={height.shape}")
        if ground.shape != height.shape:
            raise ValueError(f"Shape mismatch: {ground_path} ground={ground.shape}, label={height.shape}")

        k_value = float(GF7_CITY_K_MAPPING[city])
        physical = np.maximum(delta, 0.0) * k_value

        flat_inst = inst.reshape(-1)
        valid = flat_inst > 0
        flat_ids = flat_inst[valid]
        counts = np.bincount(flat_ids)
        h_sums = np.bincount(flat_ids, weights=height.reshape(-1)[valid])
        d_sums = np.bincount(flat_ids, weights=delta.reshape(-1)[valid])
        p_sums = np.bincount(flat_ids, weights=physical.reshape(-1)[valid])
        disp_means = instance_means(flat_inst, disp)
        disp_medians = instance_medians(flat_inst, disp)
        ground_means = instance_means(flat_inst, ground)

        for instance_id in np.flatnonzero(counts):
            if instance_id == 0:
                continue
            count = int(counts[instance_id])
            gt_mean = float(h_sums[instance_id] / max(count, 1))
            phys_mean = float(p_sums[instance_id] / max(count, 1))
            delta_mean = float(d_sums[instance_id] / max(count, 1))
            disp_mean = float(disp_means[instance_id]) if instance_id < disp_means.shape[0] else 0.0
            disp_p50 = float(disp_medians[instance_id]) if instance_id < disp_medians.shape[0] else 0.0
            ground_disp = float(ground_means[instance_id]) if instance_id < ground_means.shape[0] else 0.0
            disp_order_gap = disp_p50 - ground_disp
            disp_order_bad = bool(disp_order_gap > float(args.disp_order_margin))
            rows.append(
                {
                    "city": city,
                    "core": core_name,
                    "key": f"{city}/{core_name}",
                    "instance_id": int(instance_id),
                    "pixels": count,
                    "gt_mean": gt_mean,
                    "phys_mean": phys_mean,
                    "delta_mean": delta_mean,
                    "disp_mean": disp_mean,
                    "disp_p50": disp_p50,
                    "ground_disp": ground_disp,
                    "disp_order_gap": disp_order_gap,
                    "disp_order_bad": disp_order_bad,
                    "abs_gap": abs(gt_mean - phys_mean),
                    "under_gap": max(0.0, gt_mean - phys_mean),
                    "over_gap": max(0.0, phys_mean - gt_mean),
                    "bin_idx": bin_index_for_height(gt_mean, args.bin_edges),
                    "bin": bin_name_for_height(gt_mean, args.bin_edges),
                }
            )

    if not rows:
        raise RuntimeError("No building instances were found.")

    total = len(rows)

    by_city_bin = {}
    for i, row in enumerate(rows):
        by_city_bin.setdefault((row["city"], row["bin_idx"]), []).append(i)

    scores = np.zeros(total, dtype=np.float32)
    force_bad = np.zeros(total, dtype=bool)
    for (city, bin_idx), indices in by_city_bin.items():
        gt = np.array([rows[i]["gt_mean"] for i in indices], dtype=np.float32)
        phys = np.array([rows[i]["phys_mean"] for i in indices], dtype=np.float32)
        pixels = np.array([rows[i]["pixels"] for i in indices], dtype=np.float32)
        under = np.array([rows[i]["under_gap"] for i in indices], dtype=np.float32)
        over = np.array([rows[i]["over_gap"] for i in indices], dtype=np.float32)

        under_z = robust_z(under)
        over_z = robust_z(over)
        small_z = robust_z(-np.log1p(pixels))
        bin_scores = 0.75 * np.maximum(under_z, over_z) + 0.25 * small_z

        high = gt >= 30.0
        low = gt <= 12.0
        ratio = phys / np.maximum(gt, 1e-6)
        force = np.zeros_like(high, dtype=bool)
        force |= gt < float(args.min_label_height)
        force |= phys < float(args.min_physical_height)
        force |= high & (ratio < args.high_phys_ratio) & (under > 15.0)
        force |= low & (ratio > args.low_phys_ratio) & (over > 10.0)
        force |= pixels < args.min_pixels
        for local, global_idx in enumerate(indices):
            scores[global_idx] = float(bin_scores[local])
            force_bad[global_idx] = bool(force[local])

    disp_order_bad_indices = set()
    if not args.no_force_bad_disp_order:
        disp_order_bad_indices = {
            i for i, row in enumerate(rows)
            if bool(row.get("disp_order_bad", False))
        }
    ignore_indices = set(disp_order_bad_indices)
    ignore_indices.update(i for i, bad in enumerate(force_bad) if bool(bad))
    num_bins = len(args.bin_edges) + 1
    ratios_by_bin = None
    if args.filter_ratios_by_bin:
        if len(args.filter_ratios_by_bin) != num_bins:
            raise ValueError(f"--filter_ratios_by_bin must contain {num_bins} values.")
        ratios_by_bin = [max(0.0, min(0.95, float(x))) for x in args.filter_ratios_by_bin]

    def ratio_for_bin(bin_idx):
        if ratios_by_bin is not None:
            return ratios_by_bin[bin_idx]
        if bin_idx <= 1:
            return max(0.0, min(0.95, args.filter_ratio_low))
        if bin_idx <= 3:
            return max(0.0, min(0.95, args.filter_ratio_mid))
        return max(0.0, min(0.95, args.filter_ratio_high))

    for (city, bin_idx), indices in by_city_bin.items():
        indices = np.asarray(indices, dtype=np.int64)
        available = [int(i) for i in indices.tolist() if int(i) not in ignore_indices]
        n = len(available)
        quota = min(
            max(0, n - args.min_keep_per_city_bin),
            int(math.floor(len(indices) * ratio_for_bin(bin_idx))),
        )
        if quota <= 0:
            continue
        order = sorted(available, key=lambda i: (force_bad[i], scores[i], rows[i]["abs_gap"]), reverse=True)
        for idx in order[:quota]:
            ignore_indices.add(int(idx))

    ignore_instances = {}
    kept_rows = []
    ignored_rows = []
    for i, row in enumerate(rows):
        rec = dict(row)
        rec["score"] = float(scores[i])
        rec["force_bad"] = bool(force_bad[i])
        rec["ignored"] = i in ignore_indices
        if rec["ignored"]:
            ignore_instances.setdefault(rec["key"], []).append(rec["instance_id"])
            ignored_rows.append(rec)
        else:
            kept_rows.append(rec)

    for ids in ignore_instances.values():
        ids.sort()

    valid_tiles = sorted({row["key"] for row in kept_rows})

    manifest = {
        "data_root": args.data_root,
        "cities": args.cities,
        "cache_root": args.cache_root,
        "delta_subdir": args.delta_subdir,
        "seed": args.seed,
        "total_instances": total,
        "kept_instances": len(kept_rows),
        "ignored_instances": len(ignored_rows),
        "disp_order_bad_instances": len(disp_order_bad_indices),
        "ignored_disp_order_bad_instances": sum(1 for i in disp_order_bad_indices if i in ignore_indices),
        "disp_order_rule": "drop instance when median_building_disparity > local_ground_disparity + margin",
        "bin_edges": args.bin_edges,
        "filter_ratio_low": args.filter_ratio_low,
        "filter_ratio_mid": args.filter_ratio_mid,
        "filter_ratio_high": args.filter_ratio_high,
        "filter_ratios_by_bin": ratios_by_bin,
        "min_keep_per_city_bin": args.min_keep_per_city_bin,
        "min_label_height": args.min_label_height,
        "min_physical_height": args.min_physical_height,
        "disp_order_margin": args.disp_order_margin,
        "force_bad_disp_order": not args.no_force_bad_disp_order,
        "ignore_instances": ignore_instances,
        "valid_tiles": valid_tiles,
    }

    manifest_path = osp.join(args.out_dir, "instance_filter_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    detail_path = osp.join(args.out_dir, "instance_filter_details.csv")
    with open(detail_path, "w", newline="") as f:
        fieldnames = [
            "city",
            "core",
            "instance_id",
            "pixels",
            "gt_mean",
            "phys_mean",
            "delta_mean",
            "disp_mean",
            "disp_p50",
            "ground_disp",
            "disp_order_gap",
            "disp_order_bad",
            "abs_gap",
            "under_gap",
            "over_gap",
            "bin",
            "score",
            "force_bad",
            "ignored_by_disp_order",
            "ignored",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {k: row[k] for k in fieldnames if k in row}
            out["score"] = float(scores[idx])
            out["force_bad"] = bool(force_bad[idx])
            out["ignored_by_disp_order"] = idx in disp_order_bad_indices
            out["ignored"] = idx in ignore_indices
            writer.writerow(out)

    summary_path = osp.join(args.out_dir, "instance_filter_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "city",
                "bin",
                "total",
                "disp_order_bad",
                "ignored_disp_order_bad",
                "ignored",
                "kept",
                "ignored_ratio",
            ],
        )
        writer.writeheader()
        for (city, bin_idx), indices in sorted(by_city_bin.items()):
            ignored = sum(1 for i in indices if i in ignore_indices)
            disp_order_bad = sum(1 for i in indices if i in disp_order_bad_indices)
            ignored_disp_order_bad = sum(1 for i in indices if i in disp_order_bad_indices and i in ignore_indices)
            writer.writerow(
                {
                    "city": city,
                    "bin": rows[indices[0]]["bin"],
                    "total": len(indices),
                    "disp_order_bad": disp_order_bad,
                    "ignored_disp_order_bad": ignored_disp_order_bad,
                    "ignored": ignored,
                    "kept": len(indices) - ignored,
                    "ignored_ratio": ignored / max(1, len(indices)),
                }
            )

    print(f"Manifest saved: {manifest_path}")
    print(f"Details saved:  {detail_path}")
    print(f"Summary saved:  {summary_path}")
    print(
        f"Total={total}, kept={len(kept_rows)}, ignored={len(ignored_rows)}, "
        f"disp_order_bad={len(disp_order_bad_indices)}"
    )


if __name__ == "__main__":
    main()
