#!/usr/bin/env python3
"""Build a noisy-instance filter manifest directly from instance_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def _to_float(row, *names, default=0.0):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return float(default)


def _to_int(row, *names, default=0):
    return int(round(_to_float(row, *names, default=default)))


def bin_name_for_height(height, edges):
    lower = 0.0
    for edge in edges:
        if height <= edge:
            return f"{lower:g}-{edge:g}m"
        lower = edge
    return f">{lower:g}m"


def bin_index_for_height(height, edges):
    return int(np.searchsorted(np.asarray(edges, dtype=np.float32), height, side="left"))


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_csv",
        default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622/instance_metrics.csv",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--bin_edges", nargs="+", type=float, default=[12, 24, 30, 50, 100, 150, 300])
    parser.add_argument("--filter_ratios_by_bin", nargs="*", type=float, default=None)
    parser.add_argument("--filter_ratio_low", type=float, default=0.12)
    parser.add_argument("--filter_ratio_mid", type=float, default=0.10)
    parser.add_argument("--filter_ratio_high", type=float, default=0.06)
    parser.add_argument("--min_keep_per_city_bin", type=int, default=30)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--min_label_height", type=float, default=3.0)
    parser.add_argument("--min_physical_height", type=float, default=0.0)
    parser.add_argument("--high_phys_ratio", type=float, default=0.35)
    parser.add_argument("--low_phys_ratio", type=float, default=3.5)
    parser.add_argument("--disp_order_margin", type=float, default=0.0)
    parser.add_argument("--no_force_bad_disp_order", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    city_set = set(args.cities)

    rows = []
    with Path(args.metrics_csv).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            city = raw.get("city", "")
            if city not in city_set:
                continue
            gt_mean = _to_float(raw, "gt_mean")
            if gt_mean < args.min_label_height:
                continue
            core = raw.get("core_name", raw.get("core", ""))
            inst_id = _to_int(raw, "inst_id", "instance_id")
            if not core or inst_id <= 0:
                continue
            phys_mean = _to_float(raw, "height_physical_mean", "phys_mean")
            delta_mean = _to_float(raw, "delta_abs_mean", "delta_mean")
            disp_mean = _to_float(raw, "disp_mean")
            disp_p50 = _to_float(raw, "disp_p50", default=disp_mean)
            ground_disp = _to_float(raw, "ground_disp")
            pixels = _to_int(raw, "pixel_count", "pixels")
            disp_order_gap = _to_float(raw, "disp_order_gap", default=disp_p50 - ground_disp)
            disp_order_bad = str(raw.get("disp_order_bad", "")).lower() in {"true", "1", "yes"} or (
                disp_order_gap > float(args.disp_order_margin)
            )
            rows.append(
                {
                    "city": city,
                    "core": core,
                    "key": f"{city}/{core}",
                    "instance_id": inst_id,
                    "pixels": pixels,
                    "gt_mean": gt_mean,
                    "phys_mean": phys_mean,
                    "delta_mean": delta_mean,
                    "disp_mean": disp_mean,
                    "disp_p50": disp_p50,
                    "ground_disp": ground_disp,
                    "disp_order_gap": disp_order_gap,
                    "disp_order_bad": bool(disp_order_bad),
                    "abs_gap": abs(gt_mean - phys_mean),
                    "under_gap": max(0.0, gt_mean - phys_mean),
                    "over_gap": max(0.0, phys_mean - gt_mean),
                    "bin_idx": bin_index_for_height(gt_mean, args.bin_edges),
                    "bin": bin_name_for_height(gt_mean, args.bin_edges),
                }
            )

    if not rows:
        raise RuntimeError("No rows found in metrics CSV.")

    by_city_bin = defaultdict(list)
    for idx, row in enumerate(rows):
        by_city_bin[(row["city"], row["bin_idx"])].append(idx)

    total = len(rows)
    scores = np.zeros(total, dtype=np.float32)
    force_bad = np.zeros(total, dtype=bool)
    for (_city, _bin_idx), indices in by_city_bin.items():
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
        for local_idx, global_idx in enumerate(indices):
            scores[global_idx] = float(bin_scores[local_idx])
            force_bad[global_idx] = bool(force[local_idx])

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

    for (_city, bin_idx), indices in by_city_bin.items():
        available = [int(i) for i in indices if int(i) not in ignore_indices]
        n = len(available)
        quota = min(
            max(0, n - args.min_keep_per_city_bin),
            int(math.floor(len(indices) * ratio_for_bin(bin_idx))),
        )
        if quota <= 0:
            continue
        order = sorted(available, key=lambda i: (force_bad[i], scores[i], rows[i]["abs_gap"]), reverse=True)
        ignore_indices.update(int(i) for i in order[:quota])

    ignore_instances = defaultdict(list)
    kept_rows = []
    ignored_rows = []
    for idx, row in enumerate(rows):
        rec = dict(row)
        rec["score"] = float(scores[idx])
        rec["force_bad"] = bool(force_bad[idx])
        rec["ignored"] = idx in ignore_indices
        if rec["ignored"]:
            ignore_instances[rec["key"]].append(rec["instance_id"])
            ignored_rows.append(rec)
        else:
            kept_rows.append(rec)
    ignore_instances = {key: sorted(ids) for key, ids in sorted(ignore_instances.items()) if ids}
    valid_tiles = sorted({row["key"] for row in kept_rows})

    manifest = {
        "source_metrics_csv": str(Path(args.metrics_csv)),
        "cities": args.cities,
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
    with (out_dir / "instance_filter_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    with (out_dir / "instance_filter_summary.csv").open("w", newline="") as f:
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
        for (city, _bin_idx), indices in sorted(by_city_bin.items()):
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

    print(f"Manifest saved: {out_dir / 'instance_filter_manifest.json'}")
    print(f"Summary saved:  {out_dir / 'instance_filter_summary.csv'}")
    print(
        f"Total={total}, kept={len(kept_rows)}, ignored={len(ignored_rows)}, "
        f"disp_order_bad={len(disp_order_bad_indices)}"
    )


if __name__ == "__main__":
    main()
