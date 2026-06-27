#!/usr/bin/env python3
"""Synchronize UnifiedBH disparity caches with the current dataset tiles.

The script keeps only cache files whose core tile name is present in the
current left/right/label intersection for each city. It is intentionally
conservative: dry-run is the default, and files are only removed with --apply.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.height_datasets import output_core_name, resolve_city_dir


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

CACHE_PATTERNS = {
    "disp_npy": ("Disp_", ".npy"),
    "delta_relu_npy": ("DeltaDisp_", ".npy"),
    "delta_signed_npy": ("DeltaDispSigned_", ".npy"),
    "delta_reverse_relu_npy": ("DeltaDispReverse_", ".npy"),
    "delta_abs_npy": ("DeltaDispAbs_", ".npy"),
    "local_ground_npy": ("LocalGroundDisp_", ".npy"),
    "disp_png": ("Disp_", ".png"),
    "delta_relu_png": ("DeltaDisp_", ".png"),
    "height_physical_png": ("HeightPhysical_", ".png"),
    "compare_png": ("Compare_", ".png"),
}


def _iter_raster_files(folder: Path) -> Iterable[Path]:
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png"):
        yield from folder.glob(pattern)


def _collect_keys(data_root: Path, city: str, kind: str) -> Set[str]:
    folder = data_root / kind / resolve_city_dir(city, kind)
    if not folder.exists():
        return set()
    return {output_core_name(str(path)) for path in _iter_raster_files(folder)}


def _current_valid_keys(data_root: Path, city: str) -> Tuple[Set[str], Dict[str, int]]:
    left = _collect_keys(data_root, city, "left")
    right = _collect_keys(data_root, city, "right")
    label = _collect_keys(data_root, city, "label")
    valid = left & right & label
    return valid, {"left": len(left), "right": len(right), "label": len(label), "valid": len(valid)}


def _cache_core_name(path: Path, prefix: str, suffix: str) -> str | None:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def _cleanup_cache_dir(cache_root: Path, subdir: str, city: str, valid_keys: Set[str], apply: bool) -> Tuple[int, int]:
    prefix, suffix = CACHE_PATTERNS[subdir]
    folder = cache_root / subdir / city
    if not folder.exists():
        return 0, 0

    kept = 0
    removed = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        core = _cache_core_name(path, prefix, suffix)
        if core is None:
            continue
        if core in valid_keys:
            kept += 1
            continue
        removed += 1
        if apply:
            path.unlink()
    return kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15")
    parser.add_argument(
        "--cache_root",
        default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_full9_abs_groundfix_whusside724_20260622",
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--subdirs", nargs="+", default=list(CACHE_PATTERNS))
    parser.add_argument("--apply", action="store_true", help="Actually remove stale cache files. Default is dry-run.")
    parser.add_argument(
        "--strict_counts",
        action="store_true",
        help="Fail if left/right/label counts differ for any selected city.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    unknown_subdirs = [subdir for subdir in args.subdirs if subdir not in CACHE_PATTERNS]
    if unknown_subdirs:
        raise ValueError(f"Unknown cache subdirs: {unknown_subdirs}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[*] Sync mode: {mode}")
    print(f"[*] Data root:  {data_root}")
    print(f"[*] Cache root: {cache_root}")

    city_key_info = []
    count_errors = []
    for city in args.cities:
        valid_keys, counts = _current_valid_keys(data_root, city)
        if len({counts["left"], counts["right"], counts["label"]}) != 1:
            count_errors.append((city, counts))
        city_key_info.append((city, valid_keys, counts))

    if count_errors and args.strict_counts:
        print("\n[!] Count mismatches detected before cache cleanup:")
        for city, counts in count_errors:
            print(f"    {city}: left={counts['left']} right={counts['right']} label={counts['label']} valid={counts['valid']}")
        raise SystemExit("left/right/label counts do not match; rerun after upload finishes.")

    total_removed = 0
    summary = []
    for city, valid_keys, counts in city_key_info:
        city_kept = defaultdict(int)
        city_removed = defaultdict(int)
        for subdir in args.subdirs:
            kept, removed = _cleanup_cache_dir(cache_root, subdir, city, valid_keys, args.apply)
            city_kept[subdir] = kept
            city_removed[subdir] = removed
            total_removed += removed

        summary.append((city, counts, dict(city_kept), dict(city_removed)))

    print("\nCity       left  right  label  valid | stale cache files by subdir")
    print("-" * 86)
    for city, counts, _kept, removed in summary:
        removed_text = ", ".join(f"{k}:{v}" for k, v in removed.items() if v)
        if not removed_text:
            removed_text = "none"
        print(
            f"{city:<10} {counts['left']:>5d} {counts['right']:>6d} "
            f"{counts['label']:>6d} {counts['valid']:>6d} | {removed_text}"
        )

    if count_errors:
        print("\n[!] Count mismatches detected:")
        for city, counts in count_errors:
            print(f"    {city}: left={counts['left']} right={counts['right']} label={counts['label']} valid={counts['valid']}")
        if args.strict_counts:
            raise SystemExit("left/right/label counts do not match; rerun after upload finishes.")

    action = "removed" if args.apply else "would remove"
    print(f"\n[*] Total stale cache files {action}: {total_removed}")
    if not args.apply:
        print("[*] Re-run with --apply after upload is complete to delete stale cache files.")


if __name__ == "__main__":
    main()
