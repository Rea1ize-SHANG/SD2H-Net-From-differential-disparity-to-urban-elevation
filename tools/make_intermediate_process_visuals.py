#!/usr/bin/env python3
"""Create intermediate-process visualizations for SD2H height estimation.

The script uses the cached disparity products and the adopted intra-city 50/50
height-calibration checkpoint. It renders the main interpretable stages:
left image, label contour overlay, disparity, local ground disparity,
differential disparity, instance mask, predicted height, and reference height.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from matplotlib.colors import Normalize
from skimage.measure import label as connected_components
from skimage.segmentation import find_boundaries

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.height_datasets import (
    GF7_CITY_K_MAPPING,
    output_core_name,
    read_image_as_float32,
    read_npy_as_float32,
    read_single_channel_as_float32,
    remove_small_instances,
)
from core.sd2h_height_finetuner import SD2H_Height_Finetuner


DEFAULT_DATA_ROOT = "/home/shangying/workspace/dataset/UnifiedBH/overlap15"
DEFAULT_CACHE_ROOT = (
    "/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/"
    "sd2h_full9_abs_groundfix_whusside724_20260622"
)
DEFAULT_CONFIG = "/home/shangying/workspace/myrepos/SD2H/config/train_city_stratified_50_50.yaml"
DEFAULT_CKPT = "/home/shangying/workspace/myrepos/SD2H/checkpoints/city_stratified_50_50/finetune_best_rmse.pth"
DEFAULT_OUT = "/home/shangying/workspace/output/sd2h_intermediate_process_50_50"


REQUESTED_SAMPLES = [
    "HZ_049_left_contour_2_5.png",
    "HZ_049_left_contour_2_16.png",
    "HZ_049_left_contour_3_11.png",
    "HZ_049_left_contour_33_2.png",
    "HZ_050_left_contour_0_23.png",
    "HZ_050_left_contour_2_22.png",
    "HZ_050_left_contour_3_25.png",
    "HZ_050_left_contour_22_18.png",
    "HZ_057_left_contour_2_5.png",
    "HZ_057_left_contour_2_8.png",
    "HZ_057_left_contour_15_19.png",
    "HZ_057_left_contour_17_5.png",
    "HZ_DG_058_left_contour_9_2.png",
    "HZ_DG_058_left_contour_6_3.png",
    "HZ_DG_058_left_contour_4_2.png",
    "HZ_DG_058_left_contour_5_3.png",
    "PRE_039_left_contour_2_25.png",
    "PRE_039_left_contour_3_26.png",
    "PRE_039_left_contour_26_26.png",
    "PRE_039_left_contour_29_23.png",
    "HK_062_left_contour_35_25.png",
    "HK_062_left_contour_35_27.png",
    "HK_062_left_contour_35_29.png",
    "HK_062_left_contour_35_33.png",
    "SZ_DG_038_left_contour_0_16.png",
    "SZ_DG_038_left_contour_0_24.png",
    "SZ_DG_038_left_contour_3_24png",
    "SZ_DG_038_left_contour_22_8.png",
    "SZ_060_left_contour_0_2.png",
    "SZ_060_left_contour_0_13.png",
    "SZ_060_left_contour_1_16.png",
    "SZ_060_left_contour_2_10.pn",
]


def _cfg_without_defaults(cfg):
    data = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(data, dict):
        data.pop("defaults", None)
    return OmegaConf.create(data)


def _load_config_with_defaults(config_yaml, visited=None):
    cfg_path = Path(config_yaml).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg_path = cfg_path.resolve()

    if visited is None:
        visited = set()
    if cfg_path in visited:
        return OmegaConf.create()
    visited.add(cfg_path)

    cfg = OmegaConf.load(cfg_path)
    defaults = cfg.get("defaults", [])
    merged = OmegaConf.create()
    self_merged = False

    def merge_self():
        nonlocal merged, self_merged
        merged = OmegaConf.merge(merged, _cfg_without_defaults(cfg))
        self_merged = True

    for item in defaults:
        if isinstance(item, str):
            if item == "_self_":
                merge_self()
                continue
            child_path = cfg_path.parent / f"{item}.yaml"
            if child_path.exists():
                merged = OmegaConf.merge(merged, _load_config_with_defaults(child_path, visited))
            continue
        if isinstance(item, dict):
            for group, name in item.items():
                if group == "_self_" or name == "_self_":
                    merge_self()
                    continue
                child_path = cfg_path.parent / str(group) / f"{name}.yaml"
                if child_path.exists():
                    merged = OmegaConf.merge(merged, _load_config_with_defaults(child_path, visited))

    if not self_merged:
        merge_self()
    return merged


def _find_file(directory: Path, stem: str, exts: Sequence[str] = (".tif", ".tiff", ".png")) -> Optional[Path]:
    for ext in exts:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
        path = directory / f"{stem}{ext.upper()}"
        if path.exists():
            return path
    return None


def _parse_sample_name(name: str) -> Optional[Tuple[str, int, int]]:
    clean = name.replace(".pn", ".png")
    clean = re.sub(r"(?<=\d)(png|tif|tiff)$", r".\1", clean)
    match = re.match(r"(?P<city>.+?)_left(?:_contour)?_(?P<row>\d+)_(?P<col>\d+)(?:\.\w+)?$", clean)
    if not match:
        return None
    return match.group("city"), int(match.group("row")), int(match.group("col"))


def _load_manifest_ignored_ids(path: str) -> Dict[str, List[int]]:
    if not path or not osp.exists(path):
        return {}
    with open(path, "r") as f:
        payload = json.load(f)
    raw = payload.get("ignore_instances", payload)
    out = {}
    for key, ids in raw.items():
        clean = sorted({int(x) for x in ids if int(x) > 0})
        if clean:
            out[str(key)] = clean
    return out


def _apply_manifest_filter(mask: np.ndarray, city: str, core: str, ignored: Dict[str, List[int]]) -> np.ndarray:
    ids = ignored.get(f"{city}/{core}") or ignored.get(core)
    if not ids:
        return mask
    bad = np.isin(mask.astype(np.int64, copy=False), np.asarray(ids, dtype=np.int64))
    if not bad.any():
        return mask
    filtered = mask.copy()
    filtered[bad] = 0
    return filtered


def _label_to_instance_mask(
    height: np.ndarray,
    threshold: float,
    min_pixels: int,
    city: str,
    core: str,
    ignored: Dict[str, List[int]],
) -> np.ndarray:
    building = np.isfinite(height) & (height > threshold)
    if building.any():
        mask = connected_components(building.astype(np.uint8), connectivity=2).astype(np.float32)
    else:
        mask = np.zeros_like(height, dtype=np.float32)
    mask = remove_small_instances(mask, min_pixels)
    mask = _apply_manifest_filter(mask, city, core, ignored)
    return mask.astype(np.float32, copy=False)


def _to_rgb_uint8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(3):
        channel = arr[..., ch].astype(np.float32)
        finite = np.isfinite(channel)
        if not finite.any():
            continue
        lo, hi = np.percentile(channel[finite], [p_low, p_high])
        if hi <= lo:
            hi = lo + 1.0
        out[..., ch] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _normalize_map(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: Tuple[float, float] = (1.0, 99.0),
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if valid_mask is not None:
        finite &= valid_mask
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32), 0.0, 1.0
    if vmin is None or vmax is None:
        lo, hi = np.percentile(a[finite], robust)
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    norm[~np.isfinite(norm)] = 0.0
    return norm.astype(np.float32), float(vmin), float(vmax)


def _colorize(
    arr: np.ndarray,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: Tuple[float, float] = (1.0, 99.0),
    valid_mask: Optional[np.ndarray] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    norm, _, _ = _normalize_map(arr, vmin=vmin, vmax=vmax, robust=robust, valid_mask=valid_mask)
    rgb = (plt.get_cmap(cmap)(norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    if valid_mask is not None:
        rgb[~valid_mask] = np.asarray(background, dtype=np.uint8)
    return rgb


def _instance_color(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask[mask > 0].astype(np.int64, copy=False))
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for inst_id in ids:
        rng = np.random.default_rng(inst_id * 7919 + 17)
        color = rng.integers(50, 245, size=3, dtype=np.uint8)
        out[mask.astype(np.int64, copy=False) == inst_id] = color
    return out


def _overlay_blue_contours(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base_rgb.copy()
    boundaries = find_boundaries(mask > 0, mode="outer")
    # Thicken a little without adding OpenCV as a dependency.
    thick = boundaries.copy()
    thick[:-1, :] |= boundaries[1:, :]
    thick[1:, :] |= boundaries[:-1, :]
    thick[:, :-1] |= boundaries[:, 1:]
    thick[:, 1:] |= boundaries[:, :-1]
    out[thick] = np.asarray([0, 120, 255], dtype=np.uint8)
    return out


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), image.astype(np.uint8), check_contrast=False)


def _load_model(args, device: torch.device) -> SD2H_Height_Finetuner:
    model = SD2H_Height_Finetuner(args)
    checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
    ckpt_dict = checkpoint.get("state_dict", checkpoint)
    ckpt_dict = {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
    missing, unexpected = model.load_state_dict(ckpt_dict, strict=False)
    print(f"[*] Loaded checkpoint: {args.restore_ckpt}")
    print(f"[*] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _sample_paths(data_root: Path, cache_root: Path, city: str, row: int, col: int) -> Optional[dict]:
    core = f"{city}_{row}_{col}"
    left = _find_file(data_root / "left" / city, f"{city}_left_{row}_{col}")
    right = _find_file(data_root / "right" / city, f"{city}_right_{row}_{col}")
    label = _find_file(data_root / "label" / city, f"{city}_label_{row}_{col}")
    disp = cache_root / "disp_npy" / city / f"Disp_{core}.npy"
    ground = cache_root / "local_ground_npy" / city / f"LocalGroundDisp_{core}.npy"
    delta = cache_root / "delta_abs_npy" / city / f"DeltaDispAbs_{core}.npy"
    required = [left, right, label, disp, ground, delta]
    if any(p is None or not Path(p).exists() for p in required):
        return None
    return {
        "city": city,
        "row": row,
        "col": col,
        "core": core,
        "left": Path(left),
        "right": Path(right),
        "label": Path(label),
        "disp": disp,
        "ground": ground,
        "delta": delta,
    }


def _score_sample(paths: dict) -> Tuple[float, int, float, float]:
    height = read_single_channel_as_float32(str(paths["label"]))
    valid = np.isfinite(height) & (height > 3.0)
    if not valid.any():
        return (0.0, 0, 0.0, 0.0)
    inst = connected_components(valid.astype(np.uint8), connectivity=2)
    count = int(np.unique(inst[inst > 0]).size)
    max_h = float(np.max(height[valid]))
    mean_h = float(np.mean(height[valid]))
    area = float(valid.mean())
    # Prefer samples with enough buildings and some height variation.
    score = max_h + 0.04 * count + 30.0 * area + 0.2 * mean_h
    return (score, count, max_h, mean_h)


@torch.no_grad()
def _predict_height(model, args, device: torch.device, paths: dict, mask: np.ndarray) -> np.ndarray:
    left = read_image_as_float32(str(paths["left"]))
    right = read_image_as_float32(str(paths["right"]))
    disp = read_npy_as_float32(str(paths["disp"]))
    delta = read_npy_as_float32(str(paths["delta"]))
    ground = read_npy_as_float32(str(paths["ground"]))

    left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float().to(device)
    right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float().to(device)
    disp_t = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).float().to(device)
    delta_t = torch.from_numpy(delta).unsqueeze(0).unsqueeze(0).float().to(device)
    ground_t = torch.from_numpy(ground).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    k = torch.tensor([[GF7_CITY_K_MAPPING[paths["city"]]]], dtype=torch.float32, device=device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        out = model(
            left_t,
            right_t,
            mask=mask_t,
            k_tensor=k,
            iters=0,
            test_mode=True,
            precomputed_disp=disp_t,
            precomputed_delta=delta_t,
            precomputed_ground=ground_t,
        )
    pred = out["height"] if isinstance(out, dict) else out
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=320.0, neginf=0.0).clamp(min=0.0, max=320.0)
    return pred.squeeze().detach().cpu().numpy().astype(np.float32)


def _save_sample_visuals(model, args, device: torch.device, paths: dict, ignored: Dict[str, List[int]], out_dir: Path) -> dict:
    sample_dir = out_dir / paths["core"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    left_raw = read_image_as_float32(str(paths["left"]))
    label_h = read_single_channel_as_float32(str(paths["label"]))
    disp = read_npy_as_float32(str(paths["disp"]))
    ground = read_npy_as_float32(str(paths["ground"]))
    delta = read_npy_as_float32(str(paths["delta"]))

    core_for_manifest = output_core_name(str(paths["left"]))
    mask = _label_to_instance_mask(
        label_h,
        threshold=float(args.mask_label_threshold),
        min_pixels=int(args.mask_min_instance_pixels),
        city=paths["city"],
        core=core_for_manifest,
        ignored=ignored,
    )
    valid_building = mask > 0
    pred_h = _predict_height(model, args, device, paths, mask)

    left_rgb = _to_rgb_uint8(left_raw)
    contour_rgb = _overlay_blue_contours(left_rgb, mask)
    instance_rgb = _instance_color(mask)

    # Disparity-like products use per-image robust scaling for stronger contrast.
    disp_rgb = _colorize(disp, "turbo", robust=(1.0, 99.0))
    ground_rgb = _colorize(ground, "turbo", robust=(1.0, 99.0))
    map_background = (238, 238, 238)
    delta_rgb = _colorize(delta, "inferno", robust=(1.0, 99.5), valid_mask=valid_building, background=map_background)

    # Height maps use a common 0--120 m scale to avoid exaggerating small errors.
    pred_rgb = _colorize(pred_h, "viridis", vmin=0.0, vmax=120.0, valid_mask=valid_building, background=map_background)
    label_rgb = _colorize(label_h, "viridis", vmin=0.0, vmax=120.0, valid_mask=valid_building, background=map_background)

    outputs = {
        "left": left_rgb,
        "label_contour_overlay": contour_rgb,
        "disparity": disp_rgb,
        "local_ground_disparity": ground_rgb,
        "differential_disparity": delta_rgb,
        "instance_mask": instance_rgb,
        "predicted_height": pred_rgb,
        "reference_height": label_rgb,
    }
    for name, image in outputs.items():
        _write_png(sample_dir / f"{paths['core']}_{name}.png", image)

    valid_h = label_h[valid_building]
    pred_valid = pred_h[valid_building]
    stats = {
        "core": paths["core"],
        "city": paths["city"],
        "num_instances": int(np.unique(mask[valid_building]).size) if valid_building.any() else 0,
        "max_reference_height": float(np.max(valid_h)) if valid_h.size else 0.0,
        "mean_reference_height": float(np.mean(valid_h)) if valid_h.size else 0.0,
        "max_predicted_height": float(np.max(pred_valid)) if pred_valid.size else 0.0,
        "mean_predicted_height": float(np.mean(pred_valid)) if pred_valid.size else 0.0,
        "sample_dir": str(sample_dir),
    }
    with open(sample_dir / f"{paths['core']}_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    stats["panel_images"] = outputs
    return stats


def _make_panel_grid(rows: List[dict], out_dir: Path, max_rows: int) -> None:
    if not rows:
        return
    rows = rows[:max_rows]
    columns = [
        ("left", "Left image"),
        ("label_contour_overlay", "Label outline"),
        ("disparity", "Disparity"),
        ("local_ground_disparity", "Ground disp."),
        ("differential_disparity", "Diff. disp."),
        ("instance_mask", "Instance mask"),
        ("predicted_height", "Pred. height"),
        ("reference_height", "Ref. height"),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.dpi": 220,
        }
    ):
        fig, axes = plt.subplots(
            len(rows),
            len(columns),
            figsize=(2.05 * len(columns), 2.05 * len(rows)),
            squeeze=False,
        )
        for r, row in enumerate(rows):
            for c, (key, title) in enumerate(columns):
                ax = axes[r, c]
                ax.imshow(row["panel_images"][key])
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(title, pad=5)
                if c == 0:
                    ax.text(
                        0.02,
                        0.98,
                        row["core"],
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                    )
        fig.subplots_adjust(left=0.01, right=0.995, top=0.965, bottom=0.01, wspace=0.025, hspace=0.06)
        fig.savefig(out_dir / "sd2h_intermediate_process_grid.png", dpi=300)
        fig.savefig(out_dir / "sd2h_intermediate_process_grid.pdf")
        plt.close(fig)


def _discover_requested_samples(data_root: Path, cache_root: Path, names: Iterable[str]) -> Tuple[List[dict], List[str]]:
    found: List[dict] = []
    missing: List[str] = []
    seen = set()
    for name in names:
        parsed = _parse_sample_name(name)
        if parsed is None:
            missing.append(f"{name} | parse_failed")
            continue
        city, row, col = parsed
        key = (city, row, col)
        if key in seen:
            continue
        seen.add(key)
        paths = _sample_paths(data_root, cache_root, city, row, col)
        if paths is None:
            missing.append(name)
            continue
        score, count, max_h, mean_h = _score_sample(paths)
        paths.update({"score": score, "num_instances_hint": count, "max_h_hint": max_h, "mean_h_hint": mean_h})
        found.append(paths)
    return found, missing


def _select_samples(found: List[dict], max_samples: int) -> List[dict]:
    if len(found) <= max_samples:
        return sorted(found, key=lambda x: (x["city"], x["row"], x["col"]))

    by_city: Dict[str, List[dict]] = {}
    for item in found:
        by_city.setdefault(item["city"], []).append(item)
    selected: List[dict] = []
    for city in sorted(by_city):
        if len(selected) >= max_samples:
            break
        selected.append(sorted(by_city[city], key=lambda x: x["score"], reverse=True)[0])

    remaining = [x for x in found if x not in selected]
    for item in sorted(remaining, key=lambda x: x["score"], reverse=True):
        if len(selected) >= max_samples:
            break
        selected.append(item)
    return sorted(selected, key=lambda x: (x["city"], x["row"], x["col"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--config_yaml", default=DEFAULT_CONFIG)
    parser.add_argument("--restore_ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=12)
    parser.add_argument("--grid_rows", type=int, default=8)
    return parser.parse_args()


def main():
    cli = parse_args()
    data_root = Path(cli.data_root)
    cache_root = Path(cli.cache_root)
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_defaults(cli.config_yaml)
    cfg.restore_ckpt = cli.restore_ckpt
    cfg.mask_label_threshold = float(getattr(cfg, "mask_label_threshold", 3.0))
    cfg.mask_min_instance_pixels = int(getattr(cfg, "mask_min_instance_pixels", 16))
    cfg.height_eval_output_keys = ["height"]
    cfg.mixed_precision = "bf16"
    cfg.valid_iters = 0

    device = torch.device(cli.device if torch.cuda.is_available() and cli.device.startswith("cuda") else "cpu")
    found, missing = _discover_requested_samples(data_root, cache_root, REQUESTED_SAMPLES)
    if not found:
        raise RuntimeError("No requested samples were found with matching left/right/label/cache files.")
    selected = _select_samples(found, max_samples=max(1, cli.max_samples))

    ignored = _load_manifest_ignored_ids(str(getattr(cfg, "instance_filter_manifest", "")))
    model = _load_model(cfg, device)

    print(f"[*] Found requested samples: {len(found)} | selected: {len(selected)} | missing: {len(missing)}")
    if missing:
        print("[*] Missing requested samples:")
        for item in missing:
            print(f"    - {item}")

    rendered_rows: List[dict] = []
    summary_rows: List[dict] = []
    for paths in selected:
        print(f"[*] Rendering {paths['core']} ...")
        stats = _save_sample_visuals(model, cfg, device, paths, ignored, out_dir)
        panel_images = stats.pop("panel_images")
        if int(stats.get("num_instances", 0)) <= 0:
            print(f"    [skip] {paths['core']} has no valid filtered building instances.")
            continue
        summary_rows.append(dict(stats))
        stats["panel_images"] = panel_images
        rendered_rows.append(stats)

    with open(out_dir / "selected_samples_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(out_dir / "selected_samples.txt", "w") as f:
        f.write("Selected samples\n")
        for row in summary_rows:
            f.write(
                f"{row['core']}\tinstances={row['num_instances']}\t"
                f"ref_max={row['max_reference_height']:.3f}\t"
                f"pred_max={row['max_predicted_height']:.3f}\n"
            )
        if missing:
            f.write("\nMissing requested samples\n")
            for item in missing:
                f.write(f"{item}\n")

    _make_panel_grid(rendered_rows, out_dir, max_rows=max(1, cli.grid_rows))
    print(f"[*] Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
