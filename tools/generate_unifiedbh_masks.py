#!/usr/bin/env python3
"""Generate per-tile building instance masks from UnifiedBH height labels."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import rasterio
from skimage.measure import label as connected_components


TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass
class MaskResult:
    source: Path
    destination: Path
    city: str
    instances: int
    building_pixels: int
    status: str
    error: Optional[str] = None


def mask_filename(label_path: Path) -> str:
    """Replace the label token while preserving the original TIFF suffix."""
    stem = label_path.stem
    replaced = re.sub(r"_label_", "_mask_", stem, count=1, flags=re.IGNORECASE)
    if replaced == stem:
        replaced = re.sub(r"_label$", "_mask", stem, count=1, flags=re.IGNORECASE)
    if replaced == stem:
        replaced = f"{stem}_mask"
    return replaced + label_path.suffix


def discover_labels(label_root: Path, cities: Optional[List[str]]) -> List[Path]:
    if cities:
        city_dirs = [label_root / city for city in cities]
    else:
        city_dirs = sorted(path for path in label_root.iterdir() if path.is_dir())

    missing = [str(path) for path in city_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("City label directories not found: " + ", ".join(missing))

    labels: List[Path] = []
    for city_dir in city_dirs:
        labels.extend(
            sorted(
                path
                for path in city_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
            )
        )
    return labels


def remove_and_remap_small_instances(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or mask.max() == 0:
        return mask
    counts = np.bincount(mask.reshape(-1))
    keep_ids = np.flatnonzero(counts >= min_pixels)
    keep_ids = keep_ids[keep_ids > 0]
    lookup = np.zeros(mask.max() + 1, dtype=np.int64)
    lookup[keep_ids] = np.arange(1, keep_ids.size + 1, dtype=np.int64)
    return lookup[mask]


def build_instance_mask(
    height: np.ndarray,
    threshold: float,
    connectivity: int,
    min_pixels: int,
) -> np.ndarray:
    building = np.isfinite(height) & (height > threshold)
    mask = connected_components(building.astype(np.uint8), connectivity=connectivity).astype(
        np.int64,
        copy=False,
    )
    return remove_and_remap_small_instances(mask, min_pixels)


def output_dtype(instance_count: int) -> str:
    return "uint16" if instance_count <= np.iinfo(np.uint16).max else "uint32"


def process_label(
    source: Path,
    label_root: Path,
    mask_root: Path,
    threshold: float,
    connectivity: int,
    min_pixels: int,
    overwrite: bool,
) -> MaskResult:
    relative = source.relative_to(label_root)
    destination = mask_root / relative.parent / mask_filename(source)
    city = relative.parts[0] if relative.parts else "UNKNOWN"
    if destination.exists() and not overwrite:
        return MaskResult(source, destination, city, 0, 0, "skipped")

    try:
        with rasterio.open(source) as dataset:
            height = dataset.read(1)
            profile = dataset.profile.copy()
            dataset_tags = dataset.tags()
            band_tags = dataset.tags(1)

        mask = build_instance_mask(height, threshold, connectivity, min_pixels)
        instance_count = int(mask.max())
        dtype = output_dtype(instance_count)
        mask = mask.astype(dtype, copy=False)

        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=0,
            compress="lzw",
        )
        profile.pop("predictor", None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
        with rasterio.open(temporary, "w", **profile) as output:
            output.write(mask, 1)
            if dataset_tags:
                output.update_tags(**dataset_tags)
            if band_tags:
                output.update_tags(1, **band_tags)
        os.replace(temporary, destination)

        return MaskResult(
            source=source,
            destination=destination,
            city=city,
            instances=instance_count,
            building_pixels=int(np.count_nonzero(mask)),
            status="written",
        )
    except Exception as exc:
        return MaskResult(source, destination, city, 0, 0, "failed", str(exc))


def print_summary(results: Iterable[MaskResult]) -> None:
    summary: Dict[str, Dict[str, int]] = {}
    for result in results:
        city = summary.setdefault(
            result.city,
            {"written": 0, "skipped": 0, "failed": 0, "instances": 0},
        )
        city[result.status] += 1
        city["instances"] += result.instances

    print("\nCity             Written  Skipped  Failed  Instances")
    print("-" * 57)
    for city_name, values in sorted(summary.items()):
        print(
            f"{city_name:<16} {values['written']:>7}  {values['skipped']:>7}  "
            f"{values['failed']:>6}  {values['instances']:>9}"
        )
    totals = {
        key: sum(values[key] for values in summary.values())
        for key in ("written", "skipped", "failed", "instances")
    }
    print("-" * 57)
    print(
        f"{'Total':<16} {totals['written']:>7}  {totals['skipped']:>7}  "
        f"{totals['failed']:>6}  {totals['instances']:>9}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate UnifiedBH instance masks from weak height labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="UnifiedBH overlap15 directory containing label/<city>.",
    )
    parser.add_argument("--cities", nargs="*", default=None, help="Optional city subset.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Pixels above this label height are treated as buildings (default: 0).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2),
        default=2,
        help="Connected-component neighborhood: 1=4-neighbor, 2=8-neighbor.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=1,
        help="Minimum instance area in pixels; default 1 preserves all labels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Parallel worker count.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing masks.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    label_root = data_root / "label"
    mask_root = data_root / "mask"
    if not label_root.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    labels = discover_labels(label_root, args.cities)
    if not labels:
        raise RuntimeError(f"No TIFF labels found under: {label_root}")

    print(f"Data root : {data_root}")
    print(f"Labels    : {len(labels)}")
    print(f"Mask root : {mask_root}")
    print(
        f"Rule      : height > {args.threshold:g}, connectivity={args.connectivity}, "
        f"min_pixels={args.min_pixels}"
    )

    results: List[MaskResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                process_label,
                source,
                label_root,
                mask_root,
                args.threshold,
                args.connectivity,
                max(1, args.min_pixels),
                args.overwrite,
            )
            for source in labels
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.status == "failed":
                print(f"[FAILED] {result.source}: {result.error}")
            elif index % 100 == 0 or index == len(futures):
                print(f"Processed {index}/{len(futures)}")

    print_summary(results)
    failures = [result for result in results if result.status == "failed"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

