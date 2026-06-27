#!/usr/bin/env python3
"""Build instance-level height statistics for UnifiedBH.

This script is intentionally separate from training. The labels are noisy
height products, so we need a cached, inspectable summary before choosing
sampling weights and reporting stratified metrics.
"""
import argparse
import csv
import os
from glob import glob
from pathlib import Path

import numpy as np
import skimage.io
from tqdm import tqdm


def read_single_channel(path):
    arr = skimage.io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_key(filename):
    base = Path(filename).stem
    return base.replace("_left", "").replace("_right", "").replace("_label", "").replace("_mask", "")


def build_file_maps(root, city):
    maps = {}
    for kind in ("left", "right", "label", "mask"):
        files = []
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png"):
            files.extend(glob(str(root / kind / city / ext)))
        maps[kind] = {extract_key(path): Path(path) for path in files}
    keys = sorted(set(maps["left"]) & set(maps["right"]) & set(maps["label"]) & set(maps["mask"]))
    return maps, keys


def instance_means(height, mask):
    mask_flat = mask.astype(np.int64).ravel()
    height_flat = height.ravel()
    valid = (mask_flat > 0) & np.isfinite(height_flat) & (height_flat > 0)
    if not valid.any():
        return np.empty((0,), dtype=np.float32)

    ids = mask_flat[valid]
    values = height_flat[valid]
    max_id = int(ids.max())
    sums = np.bincount(ids, weights=values, minlength=max_id + 1)
    counts = np.bincount(ids, minlength=max_id + 1)
    used_ids = np.nonzero(counts)[0]
    used_ids = used_ids[used_ids > 0]
    return (sums[used_ids] / np.maximum(counts[used_ids], 1)).astype(np.float32)


def summarize_bins(values, bins):
    rows = []
    for idx, lower in enumerate(bins):
        upper = bins[idx + 1] if idx + 1 < len(bins) else np.inf
        if np.isinf(upper):
            keep = values > lower
            name = f">{lower:g}m"
        else:
            keep = (values > lower) & (values <= upper)
            name = f"{lower:g}-{upper:g}m"
        rows.append((name, int(keep.sum()), float(keep.mean() * 100.0) if values.size else 0.0))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15")
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--output_csv", default="height_dataset_stats.csv")
    parser.add_argument("--bins", nargs="+", type=float, default=[0, 12, 24, 30, 50, 100, 150, 300])
    args = parser.parse_args()

    root = Path(args.data_path)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    city_rows = []
    all_instances = []

    for city in args.cities:
        maps, keys = build_file_maps(root, city)
        city_instances = []
        image_with_30m = 0
        sampler_weights = []

        for key in tqdm(keys, desc=f"Scanning {city}", dynamic_ncols=True):
            height = read_single_channel(maps["label"][key])
            mask = read_single_channel(maps["mask"][key])
            inst = instance_means(height, mask)
            if inst.size:
                city_instances.append(inst)
                image_with_30m += int((inst > 30.0).any())

            valid_height = height[np.isfinite(height) & (height > 0)]
            if valid_height.size:
                top_count = max(1, int(valid_height.size * 0.05))
                top_mean = np.partition(valid_height, -top_count)[-top_count:].mean()
                sampler_weights.append(max(1.0, float(top_mean) / 10.0))

        city_instances = np.concatenate(city_instances) if city_instances else np.empty((0,), dtype=np.float32)
        all_instances.append(city_instances)

        row = {
            "city": city,
            "images": len(keys),
            "images_with_30m": image_with_30m,
            "instances": int(city_instances.size),
            "mean": float(city_instances.mean()) if city_instances.size else 0.0,
            "median": float(np.median(city_instances)) if city_instances.size else 0.0,
            "max": float(city_instances.max()) if city_instances.size else 0.0,
            "gt_30m": int((city_instances > 30.0).sum()),
            "gt_50m": int((city_instances > 50.0).sum()),
            "gt_100m": int((city_instances > 100.0).sum()),
            "sampler_weight_mean": float(np.mean(sampler_weights)) if sampler_weights else 0.0,
            "sampler_weight_max": float(np.max(sampler_weights)) if sampler_weights else 0.0,
        }
        city_rows.append(row)

    all_instances = np.concatenate(all_instances) if all_instances else np.empty((0,), dtype=np.float32)

    with open(output_csv, "w", newline="") as f:
        fieldnames = list(city_rows[0].keys()) if city_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(city_rows)

    print(f"Saved city stats to {output_csv}")
    print("Overall:")
    print(f"  instances={all_instances.size}")
    if all_instances.size:
        print(f"  mean={all_instances.mean():.3f} median={np.median(all_instances):.3f} max={all_instances.max():.3f}")
        for name, count, pct in summarize_bins(all_instances, args.bins):
            print(f"  {name:<12} {count:>8d} {pct:>6.2f}%")


if __name__ == "__main__":
    main()
