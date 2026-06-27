import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import skimage.io
import torch
from tqdm import tqdm

from core.sd2h import SD2H
from core.height_datasets import UnifiedBHDataset, GF7_CITY_K_MAPPING, resolve_city_dir
from core.utils.utils import InputPadder


def _load_sd2h(args):
    ckpt_path = Path(args.restore_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[*] Loading disparity checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

    saved_cfg = checkpoint.get("args", checkpoint.get("cfg"))
    if saved_cfg is not None:
        for key in ["encoder", "corr_levels", "hidden_dims", "n_gru_layers", "shared_backbone"]:
            if isinstance(saved_cfg, dict) and key in saved_cfg:
                setattr(args, key, saved_cfg[key])
            elif hasattr(saved_cfg, key):
                setattr(args, key, getattr(saved_cfg, key))

    for key, value in state_dict.items():
        if "patch_embed.proj.weight" in key and len(value.shape) == 4:
            inferred_encoder = {
                384: "vits",
                768: "vitb",
                1024: "vitl",
                1536: "vitg",
            }.get(value.shape[0])
            if inferred_encoder is not None:
                args.encoder = inferred_encoder
            break

    if "724" in str(ckpt_path) or "pretrain_whusside" in str(ckpt_path):
        args.shared_backbone = True

    print(f"[*] Building SD2H disparity model: encoder={args.encoder}, shared_backbone={args.shared_backbone}")
    model = SD2H(args)
    model_state = model.state_dict()
    matched_state = {
        k: v for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = model.load_state_dict(matched_state, strict=False)
    print(f"[*] Matched tensors: {len(matched_state)} | missing: {len(missing)} | unexpected: {len(unexpected)}")
    model.to(args.device)
    model.eval()
    return model


def _pick_info_path(info, item_index, field_index):
    if isinstance(info, (list, tuple)):
        if len(info) > field_index and isinstance(info[field_index], (list, tuple)):
            return info[field_index][item_index]
        if len(info) > field_index and isinstance(info[field_index], str):
            return info[field_index]
        if len(info) > item_index and isinstance(info[item_index], (list, tuple)):
            row = info[item_index]
            if len(row) > field_index:
                return row[field_index]
    raise ValueError(f"Cannot parse sample info: {info}")


def _output_core_name(left_path):
    base = Path(left_path).stem
    original = base
    for prefix in ["ImgLGF7_", "ImgRGF7_", "DispGF7_"]:
        if base.startswith(prefix):
            return base[len(prefix):]
    for token in ["_left_", "_right_", "_label_", "_mask_", "_disparity_", "_disp_"]:
        base = base.replace(token, "_")
    for suffix in ["_left", "_right", "_label", "_mask", "_disparity", "_disp"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    for prefix in ["left_", "right_", "label_", "mask_", "disparity_", "disp_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base or original


def _city_from_path(path, data_path):
    try:
        rel = Path(path).resolve().relative_to(Path(data_path).resolve())
        if len(rel.parts) >= 2:
            return rel.parts[1]
    except Exception:
        pass
    return Path(path).parent.name


def _cached_disp_path(root, city, core_name):
    if not root:
        return None
    return Path(root) / "disp_npy" / city / f"Disp_{core_name}.npy"


def _cached_ground_path(root, city, core_name):
    if not root:
        return None
    return Path(root) / "local_ground_npy" / city / f"LocalGroundDisp_{core_name}.npy"


def _select_dataset_indices(dataset, args):
    selected = []
    per_city_seen = {city: 0 for city in args.cities}
    per_city_kept = {city: 0 for city in args.cities}

    for idx, pair in enumerate(dataset.image_list):
        left_path = pair[0]
        path_city = _city_from_path(left_path, args.data_path)
        logical_city = None
        for requested in args.cities:
            if requested == path_city or resolve_city_dir(requested, "left") == path_city:
                logical_city = requested
                break
        if logical_city is None:
            continue

        seen = per_city_seen[logical_city]
        per_city_seen[logical_city] += 1
        sample_offset = int(getattr(args, "sample_offset", 0))
        if args.sample_stride > 1 and (seen % args.sample_stride) != (sample_offset % args.sample_stride):
            continue
        if args.max_samples_per_city > 0 and per_city_kept[logical_city] >= args.max_samples_per_city:
            continue

        selected.append(idx)
        per_city_kept[logical_city] += 1

    print(f"[*] Selected sample counts: {per_city_kept}")
    return selected


def _make_ring(instance_binary, image_shape, inner_radius, outer_radius):
    outer_radius = int(max(outer_radius, 1))
    inner_radius = int(max(inner_radius, 0))
    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_radius * 2 + 1, outer_radius * 2 + 1)
    )
    kernel_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inner_radius * 2 + 1, inner_radius * 2 + 1)
    )
    inst_uint = instance_binary.astype(np.uint8)
    outer = cv2.dilate(inst_uint, kernel_outer, iterations=1).astype(bool)
    inner = cv2.dilate(inst_uint, kernel_inner, iterations=1).astype(bool)
    ring = outer & ~inner
    if ring.shape != image_shape:
        ring = ring[:image_shape[0], :image_shape[1]]
    return ring


def _ring_values_local(disp, instance_binary, building_mask, inner_radius, outer_radius, exclude_all_buildings=True):
    ys, xs = np.nonzero(instance_binary)
    if ys.size == 0:
        return np.empty((0,), dtype=np.float32)

    margin = int(max(outer_radius, 1)) + 2
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, disp.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, disp.shape[1])

    local_inst = instance_binary[y0:y1, x0:x1]
    local_ring = _make_ring(local_inst, local_inst.shape, inner_radius, outer_radius)
    if exclude_all_buildings:
        local_ring &= ~building_mask[y0:y1, x0:x1]
    else:
        local_ring &= ~local_inst

    return _safe_values(disp[y0:y1, x0:x1][local_ring])


def _safe_values(values):
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values)]


def _build_delta_and_instances(disp, mask, height, k_value, args):
    mask = np.asarray(mask).astype(np.int64)
    height = np.asarray(height, dtype=np.float32)
    disp = np.asarray(disp, dtype=np.float32)

    delta = np.full_like(disp, np.nan, dtype=np.float32)
    delta_signed = np.full_like(disp, np.nan, dtype=np.float32)
    delta_reverse = np.full_like(disp, np.nan, dtype=np.float32)
    delta_abs = np.full_like(disp, np.nan, dtype=np.float32)
    ground = np.full_like(disp, np.nan, dtype=np.float32)

    building_mask = mask > 0
    inst_ids = np.unique(mask[building_mask])
    instance_records = []

    for inst_id_raw in inst_ids:
        inst_id = int(inst_id_raw)
        inst = mask == inst_id
        pixel_count = int(inst.sum())
        if pixel_count < args.min_instance_pixels:
            continue

        label_vals = _safe_values(height[inst])
        label_vals = label_vals[label_vals > args.label_min_height]
        if label_vals.size == 0:
            continue

        ring_vals = _ring_values_local(
            disp,
            inst,
            building_mask,
            args.ring_inner_radius,
            args.ring_outer_radius,
            exclude_all_buildings=True,
        )

        fallback_used = "local_ring"
        if ring_vals.size < args.min_ring_pixels:
            ring_vals = _ring_values_local(
                disp,
                inst,
                building_mask,
                0,
                max(args.ring_outer_radius, args.fallback_ring_radius),
                exclude_all_buildings=True,
            )
            fallback_used = "wide_ring"

        if ring_vals.size < args.min_ring_pixels:
            ring_vals = _safe_values(disp[~building_mask])
            fallback_used = "image_nonbuilding"

        if ring_vals.size < args.min_ring_pixels:
            ring_vals = _safe_values(disp)
            fallback_used = "image_all"

        if ring_vals.size < args.min_ring_pixels:
            continue

        ground_disp = float(np.percentile(ring_vals, args.ground_percentile))
        disp_vals = _safe_values(disp[inst])
        if disp_vals.size == 0:
            continue
        disp_p50 = float(np.percentile(disp_vals, 50))
        disp_order_gap = disp_p50 - ground_disp

        instance_records.append({
            "inst_id": inst_id,
            "inst_mask": inst,
            "pixel_count": pixel_count,
            "gt_mean": float(label_vals.mean()),
            "gt_p50": float(np.percentile(label_vals, 50)),
            "gt_p95": float(np.percentile(label_vals, 95)),
            "gt_max": float(label_vals.max()),
            "disp_mean": float(disp_vals.mean()),
            "disp_p50": disp_p50,
            "ground_disp_raw": ground_disp,
            "disp_order_gap_raw": disp_order_gap,
            "disp_order_bad_raw": bool(disp_order_gap > 0.0),
            "ground_source": fallback_used,
        })

    raw_grounds = np.array([r["ground_disp_raw"] for r in instance_records], dtype=np.float32)
    corrected_ground_default = np.nan
    ground_mad = np.nan
    ground_tol = float(args.ground_outlier_abs_tol)
    if raw_grounds.size:
        initial_median = float(np.nanmedian(raw_grounds))
        abs_dev = np.abs(raw_grounds - initial_median)
        ground_mad = float(np.nanmedian(abs_dev))
        if not np.isfinite(ground_mad) or ground_mad < 1.0e-6:
            ground_mad = float(np.nanstd(raw_grounds))
        if not np.isfinite(ground_mad):
            ground_mad = 0.0
        ground_tol = max(
            float(args.ground_outlier_abs_tol),
            float(args.ground_outlier_mad_scale) * max(ground_mad, 1.0e-6),
        )
        inlier = np.isfinite(raw_grounds) & (np.abs(raw_grounds - initial_median) <= ground_tol)
        if inlier.any():
            corrected_ground_default = float(np.mean(raw_grounds[inlier]))
        else:
            corrected_ground_default = initial_median

    instance_rows = []
    for record in instance_records:
        inst = record.pop("inst_mask")
        raw_ground = float(record["ground_disp_raw"])
        ground_corrected = False
        ground_disp = raw_ground
        if (
            bool(args.correct_ground_outliers)
            and raw_grounds.size >= int(args.ground_outlier_min_instances)
            and np.isfinite(corrected_ground_default)
            and np.isfinite(raw_ground)
            and abs(raw_ground - corrected_ground_default) > ground_tol
        ):
            ground_disp = corrected_ground_default
            ground_corrected = True

        inst_disp = disp[inst]
        signed = ground_disp - inst_disp
        reverse_signed = inst_disp - ground_disp
        relu = np.maximum(signed, 0.0)
        reverse_relu = np.maximum(reverse_signed, 0.0)
        abs_delta = np.abs(signed)

        delta_signed[inst] = signed
        delta_reverse[inst] = reverse_relu
        delta[inst] = relu
        delta_abs[inst] = abs_delta
        ground[inst] = ground_disp

        disp_vals = _safe_values(inst_disp)
        signed_vals = _safe_values(signed)
        relu_vals = _safe_values(relu)
        reverse_vals = _safe_values(reverse_relu)
        abs_vals = _safe_values(abs_delta)
        if disp_vals.size == 0 or abs_vals.size == 0:
            continue
        disp_p50 = float(np.percentile(disp_vals, 50))
        disp_order_gap = disp_p50 - ground_disp

        instance_rows.append({
            **record,
            "ground_disp": ground_disp,
            "ground_disp_image_median": corrected_ground_default,
            "ground_disp_image_mad": ground_mad,
            "ground_outlier_tol": ground_tol,
            "ground_corrected": bool(ground_corrected),
            "disp_order_gap": disp_order_gap,
            "disp_order_bad": bool(disp_order_gap > 0.0),
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

    return delta, delta_signed, delta_reverse, delta_abs, ground, instance_rows


def _build_instance_rows_from_cached(disp, delta, delta_signed, delta_reverse, delta_abs, ground, mask, height, k_value, args):
    mask = np.asarray(mask).astype(np.int64)
    height = np.asarray(height, dtype=np.float32)
    disp = np.asarray(disp, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    delta_signed = np.asarray(delta_signed, dtype=np.float32)
    delta_reverse = np.asarray(delta_reverse, dtype=np.float32)
    delta_abs = np.asarray(delta_abs, dtype=np.float32)
    ground = np.asarray(ground, dtype=np.float32)

    building_mask = mask > 0
    inst_ids = np.unique(mask[building_mask])
    instance_rows = []

    for inst_id_raw in inst_ids:
        inst_id = int(inst_id_raw)
        inst = mask == inst_id
        pixel_count = int(inst.sum())
        if pixel_count < args.min_instance_pixels:
            continue

        label_vals = _safe_values(height[inst])
        label_vals = label_vals[label_vals > args.label_min_height]
        if label_vals.size == 0:
            continue

        disp_vals = _safe_values(disp[inst])
        relu_vals = _safe_values(delta[inst])
        if disp_vals.size == 0 or relu_vals.size == 0:
            continue

        signed_vals = _safe_values(delta_signed[inst])
        reverse_vals = _safe_values(delta_reverse[inst])
        abs_vals = _safe_values(delta_abs[inst])
        ground_vals = _safe_values(ground[inst])
        ground_disp = float(np.percentile(ground_vals, 50)) if ground_vals.size else 0.0
        disp_p50 = float(np.percentile(disp_vals, 50))
        disp_order_gap = disp_p50 - ground_disp

        instance_rows.append({
            "inst_id": inst_id,
            "pixel_count": pixel_count,
            "gt_mean": float(label_vals.mean()),
            "gt_p50": float(np.percentile(label_vals, 50)),
            "gt_p95": float(np.percentile(label_vals, 95)),
            "gt_max": float(label_vals.max()),
            "disp_mean": float(disp_vals.mean()),
            "disp_p50": disp_p50,
            "ground_disp": ground_disp,
            "disp_order_gap": disp_order_gap,
            "disp_order_bad": bool(disp_order_gap > 0.0),
            "ground_source": "cached",
            "ground_disp_raw": ground_disp,
            "ground_disp_image_median": 0.0,
            "ground_disp_image_mad": 0.0,
            "ground_outlier_tol": 0.0,
            "ground_corrected": False,
            "delta_signed_mean": float(signed_vals.mean()) if signed_vals.size else 0.0,
            "delta_signed_p50": float(np.percentile(signed_vals, 50)) if signed_vals.size else 0.0,
            "delta_relu_mean": float(relu_vals.mean()),
            "delta_relu_p50": float(np.percentile(relu_vals, 50)),
            "delta_abs_mean": float(abs_vals.mean()) if abs_vals.size else 0.0,
            "delta_abs_p50": float(np.percentile(abs_vals, 50)) if abs_vals.size else 0.0,
            "delta_reverse_relu_mean": float(reverse_vals.mean()) if reverse_vals.size else 0.0,
            "height_physical_mean": float(k_value * abs_vals.mean()) if abs_vals.size else 0.0,
            "height_physical_p50": float(k_value * np.percentile(abs_vals, 50)) if abs_vals.size else 0.0,
            "height_forward_relu_mean": float(k_value * relu_vals.mean()),
            "height_reverse_mean": float(k_value * reverse_vals.mean()) if reverse_vals.size else 0.0,
        })

    return instance_rows


def _build_delta_from_cached_ground(disp, cached_ground, mask, height, k_value, args):
    mask = np.asarray(mask).astype(np.int64)
    height = np.asarray(height, dtype=np.float32)
    disp = np.asarray(disp, dtype=np.float32)
    cached_ground = np.asarray(cached_ground, dtype=np.float32)

    delta = np.full_like(disp, np.nan, dtype=np.float32)
    delta_signed = np.full_like(disp, np.nan, dtype=np.float32)
    delta_reverse = np.full_like(disp, np.nan, dtype=np.float32)
    delta_abs = np.full_like(disp, np.nan, dtype=np.float32)
    ground = np.full_like(disp, np.nan, dtype=np.float32)

    building_mask = mask > 0
    inst_ids = np.unique(mask[building_mask])
    records = []
    for inst_id_raw in inst_ids:
        inst_id = int(inst_id_raw)
        inst = mask == inst_id
        pixel_count = int(inst.sum())
        if pixel_count < args.min_instance_pixels:
            continue
        label_vals = _safe_values(height[inst])
        label_vals = label_vals[label_vals > args.label_min_height]
        if label_vals.size == 0:
            continue
        disp_vals = _safe_values(disp[inst])
        ground_vals = _safe_values(cached_ground[inst])
        if disp_vals.size == 0 or ground_vals.size == 0:
            continue
        ground_disp = float(np.percentile(ground_vals, 50))
        disp_p50 = float(np.percentile(disp_vals, 50))
        records.append({
            "inst_id": inst_id,
            "inst_mask": inst,
            "pixel_count": pixel_count,
            "gt_mean": float(label_vals.mean()),
            "gt_p50": float(np.percentile(label_vals, 50)),
            "gt_p95": float(np.percentile(label_vals, 95)),
            "gt_max": float(label_vals.max()),
            "disp_mean": float(disp_vals.mean()),
            "disp_p50": disp_p50,
            "ground_disp_raw": ground_disp,
            "disp_order_gap_raw": disp_p50 - ground_disp,
            "disp_order_bad_raw": bool((disp_p50 - ground_disp) > 0.0),
            "ground_source": "cached_ground",
        })

    raw_grounds = np.array([r["ground_disp_raw"] for r in records], dtype=np.float32)
    corrected_ground_default = np.nan
    ground_mad = 0.0
    ground_tol = float(args.ground_outlier_abs_tol)
    if raw_grounds.size:
        initial_median = float(np.nanmedian(raw_grounds))
        abs_dev = np.abs(raw_grounds - initial_median)
        ground_mad = float(np.nanmedian(abs_dev))
        if not np.isfinite(ground_mad) or ground_mad < 1.0e-6:
            ground_mad = float(np.nanstd(raw_grounds))
        if not np.isfinite(ground_mad):
            ground_mad = 0.0
        ground_tol = max(
            float(args.ground_outlier_abs_tol),
            float(args.ground_outlier_mad_scale) * max(ground_mad, 1.0e-6),
        )
        inlier = np.isfinite(raw_grounds) & (np.abs(raw_grounds - initial_median) <= ground_tol)
        corrected_ground_default = float(np.mean(raw_grounds[inlier])) if inlier.any() else initial_median

    instance_rows = []
    for record in records:
        inst = record.pop("inst_mask")
        raw_ground = float(record["ground_disp_raw"])
        ground_disp = raw_ground
        ground_corrected = False
        if (
            bool(args.correct_ground_outliers)
            and raw_grounds.size >= int(args.ground_outlier_min_instances)
            and np.isfinite(corrected_ground_default)
            and abs(raw_ground - corrected_ground_default) > ground_tol
        ):
            ground_disp = corrected_ground_default
            ground_corrected = True

        inst_disp = disp[inst]
        signed = ground_disp - inst_disp
        reverse_signed = inst_disp - ground_disp
        relu = np.maximum(signed, 0.0)
        reverse_relu = np.maximum(reverse_signed, 0.0)
        abs_delta = np.abs(signed)

        delta_signed[inst] = signed
        delta_reverse[inst] = reverse_relu
        delta[inst] = relu
        delta_abs[inst] = abs_delta
        ground[inst] = ground_disp

        disp_vals = _safe_values(inst_disp)
        signed_vals = _safe_values(signed)
        relu_vals = _safe_values(relu)
        reverse_vals = _safe_values(reverse_relu)
        abs_vals = _safe_values(abs_delta)
        if disp_vals.size == 0 or abs_vals.size == 0:
            continue
        disp_p50 = float(np.percentile(disp_vals, 50))
        disp_order_gap = disp_p50 - ground_disp
        instance_rows.append({
            **record,
            "ground_disp": ground_disp,
            "ground_disp_image_median": corrected_ground_default,
            "ground_disp_image_mad": ground_mad,
            "ground_outlier_tol": ground_tol,
            "ground_corrected": bool(ground_corrected),
            "disp_order_gap": disp_order_gap,
            "disp_order_bad": bool(disp_order_gap > 0.0),
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
    return delta, delta_signed, delta_reverse, delta_abs, ground, instance_rows


def _to_uint8_rgb(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[..., :3]

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        channel = img[..., c].astype(np.float32)
        valid = np.isfinite(channel)
        if valid.any():
            lo, hi = np.percentile(channel[valid], [1.0, 99.0])
            if hi <= lo:
                hi = lo + 1.0
            out[..., c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _colorize_scalar(img, valid_mask=None, cmap="turbo", vmin=0.0, vmax=80.0, invalid_color=(255, 255, 255)):
    img = np.asarray(img, dtype=np.float32).squeeze()
    invalid = ~np.isfinite(img)
    if valid_mask is not None:
        invalid |= ~np.asarray(valid_mask).astype(bool).squeeze()
    if vmax <= vmin:
        vmax = vmin + 1.0
    img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    cmap_m = matplotlib.colormaps[cmap]
    color = (cmap_m(img_norm)[..., :3] * 255.0).astype(np.uint8)
    color[invalid] = np.array(invalid_color, dtype=np.uint8)
    return color


def _save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), image, check_contrast=False)


def _regression_stats(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0, "r2": 0.0}
    diff = y_pred - y_true
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if y_true.size < 2 else 1.0 - ss_res / (ss_tot + 1e-8)
    return {
        "count": int(y_true.size),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "r2": float(r2),
    }


def _height_bin_rows(y_true, y_pred, edges):
    rows = []
    edges = [float(x) for x in edges]
    for idx, lower in enumerate(edges):
        upper = edges[idx + 1] if idx + 1 < len(edges) else math.inf
        if math.isinf(upper):
            keep = y_true > lower
            bin_name = f">{lower:g}m"
        else:
            keep = (y_true > lower) & (y_true <= upper)
            bin_name = f"{lower:g}-{upper:g}m"
        rows.append({"bin": bin_name, **_regression_stats(y_true[keep], y_pred[keep])})
    return rows


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_affine(x, y, sample_weight=None, fit_intercept=True):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)[valid]
    else:
        w = np.ones_like(x)
    if x.size == 0:
        return 1.0, 0.0
    if fit_intercept:
        design = np.stack([x, np.ones_like(x)], axis=1)
    else:
        design = x[:, None]
    sw = np.sqrt(np.maximum(w, 1e-8))[:, None]
    coef, *_ = np.linalg.lstsq(design * sw, y * sw[:, 0], rcond=None)
    if fit_intercept:
        return float(coef[0]), float(coef[1])
    return float(coef[0]), 0.0


def _balanced_weights_by_gt_bins(y, edges):
    y = np.asarray(y, dtype=np.float64)
    bins = np.digitize(y, np.asarray(edges[1:], dtype=np.float64), right=True)
    weights = np.ones_like(y, dtype=np.float64)
    for b in np.unique(bins):
        keep = bins == b
        if keep.any():
            weights[keep] = 1.0 / float(keep.sum())
    weights *= len(weights) / (weights.sum() + 1e-8)
    return weights


def _piecewise_pred_bin_calibrator(x_train, y_train, x_test, edges):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    bins = np.digitize(x_train, edges[1:], right=True)
    test_bins = np.digitize(x_test, edges[1:], right=True)

    global_a, global_b = _fit_affine(x_train, y_train, fit_intercept=True)
    pred = global_a * x_test + global_b
    params = []
    for b in np.unique(test_bins):
        train_keep = bins == b
        if train_keep.sum() < 8:
            continue
        a, b0 = _fit_affine(x_train[train_keep], y_train[train_keep], fit_intercept=True)
        pred[test_bins == b] = a * x_test[test_bins == b] + b0
        params.append({"pred_bin": int(b), "a": float(a), "b": float(b0), "train_count": int(train_keep.sum())})
    return np.maximum(pred, 0.0), params


def _calibration_report(instance_rows, args):
    if not instance_rows:
        return [], []

    train_cities = set(args.train_cities)
    test_cities = set(args.test_cities)
    train = [r for r in instance_rows if r["city"] in train_cities]
    test = [r for r in instance_rows if r["city"] in test_cities]
    if not train or not test:
        return [], []

    edges = [0.0] + [float(x) for x in args.height_bins]
    x_train = np.array([r["height_physical_mean"] for r in train], dtype=np.float64)
    y_train = np.array([r["gt_mean"] for r in train], dtype=np.float64)
    x_test = np.array([r["height_physical_mean"] for r in test], dtype=np.float64)
    y_test = np.array([r["gt_mean"] for r in test], dtype=np.float64)
    rev_test = np.array([r["height_reverse_mean"] for r in test], dtype=np.float64)

    reports = []
    bin_reports = []

    candidates = [("raw_physical", x_test)]
    a0, b0 = _fit_affine(x_train, y_train, fit_intercept=False)
    candidates.append(("linear_origin", np.maximum(a0 * x_test + b0, 0.0)))
    a1, b1 = _fit_affine(x_train, y_train, fit_intercept=True)
    candidates.append(("affine", np.maximum(a1 * x_test + b1, 0.0)))
    weights = _balanced_weights_by_gt_bins(y_train, edges)
    aw, bw = _fit_affine(x_train, y_train, sample_weight=weights, fit_intercept=True)
    candidates.append(("balanced_affine", np.maximum(aw * x_test + bw, 0.0)))
    piecewise_pred, piecewise_params = _piecewise_pred_bin_calibrator(x_train, y_train, x_test, edges)
    candidates.append(("piecewise_predbin_affine", piecewise_pred))
    candidates.append(("reverse_sign_physical", rev_test))

    param_map = {
        "raw_physical": {},
        "linear_origin": {"a": a0, "b": b0},
        "affine": {"a": a1, "b": b1},
        "balanced_affine": {"a": aw, "b": bw},
        "piecewise_predbin_affine": {"piecewise_params": piecewise_params},
        "reverse_sign_physical": {},
    }

    for name, pred in candidates:
        stats = _regression_stats(y_test, pred)
        row = {"model": name, "train_cities": "|".join(args.train_cities), "test_cities": "|".join(args.test_cities), **stats}
        params = param_map.get(name, {})
        if "a" in params:
            row["a"] = params["a"]
            row["b"] = params["b"]
        reports.append(row)
        for bin_row in _height_bin_rows(y_test, pred, edges):
            bin_reports.append({"model": name, **bin_row})

    return reports, bin_reports


@torch.no_grad()
def run_precompute(model, loader, args):
    out_dir = Path(args.output_directory)
    raw_dirs = {
        "disp": out_dir / "disp_npy",
        "delta": out_dir / "delta_relu_npy",
        "delta_signed": out_dir / "delta_signed_npy",
        "delta_reverse": out_dir / "delta_reverse_relu_npy",
        "delta_abs": out_dir / "delta_abs_npy",
        "ground": out_dir / "local_ground_npy",
    }
    vis_dirs = {
        "disp": out_dir / "disp_png",
        "delta": out_dir / "delta_relu_png",
        "height_physical": out_dir / "height_physical_png",
        "compare": out_dir / "compare_png",
    }
    for path in raw_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.save_visuals:
        for path in vis_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    city_seen = {}
    manifest_rows = []
    instance_rows = []
    processed = 0

    pbar = tqdm(loader, dynamic_ncols=True, desc="Recomputing disparity/delta")
    for data in pbar:
        info, left, right = data[0], data[1], data[2]
        instance_mask, height_gt, k_tensor = data[5], data[6], data[7]
        if instance_mask.dim() == 4:
            instance_mask_cpu = instance_mask[:, 0].numpy()
        else:
            instance_mask_cpu = instance_mask.numpy()
        if height_gt.dim() == 4:
            height_cpu = height_gt[:, 0].numpy()
        else:
            height_cpu = height_gt.numpy()

        batch_size = left.shape[0]
        sample_meta = []
        run_indices = []
        for item_index in range(batch_size):
            left_path = _pick_info_path(info, item_index, 0)
            right_path = _pick_info_path(info, item_index, 1)
            label_path = _pick_info_path(info, item_index, 2)
            city_name = _city_from_path(left_path, args.data_path)
            logical_city = city_name
            for requested in args.cities:
                if resolve_city_dir(requested, "left") == city_name:
                    logical_city = requested
                    break

            city_seen[logical_city] = city_seen.get(logical_city, 0) + 1

            core_name = _output_core_name(left_path)
            disp_path = raw_dirs["disp"] / logical_city / f"Disp_{core_name}.npy"
            delta_path = raw_dirs["delta"] / logical_city / f"DeltaDisp_{core_name}.npy"
            signed_path = raw_dirs["delta_signed"] / logical_city / f"DeltaDispSigned_{core_name}.npy"
            reverse_path = raw_dirs["delta_reverse"] / logical_city / f"DeltaDispReverse_{core_name}.npy"
            abs_path = raw_dirs["delta_abs"] / logical_city / f"DeltaDispAbs_{core_name}.npy"
            ground_path = raw_dirs["ground"] / logical_city / f"LocalGroundDisp_{core_name}.npy"
            reuse_disp_path = _cached_disp_path(args.reuse_disp_root, logical_city, core_name)
            reuse_ground_path = _cached_ground_path(args.reuse_disp_root, logical_city, core_name)
            has_reuse_disp = bool(reuse_disp_path is not None and reuse_disp_path.exists())
            has_reuse_ground = bool(reuse_ground_path is not None and reuse_ground_path.exists())

            already_done = (
                args.skip_existing
                and disp_path.exists()
                and delta_path.exists()
                and signed_path.exists()
                and reverse_path.exists()
                and abs_path.exists()
                and ground_path.exists()
            )
            sample_meta.append({
                "item_index": item_index,
                "left_path": left_path,
                "right_path": right_path,
                "label_path": label_path,
                "logical_city": logical_city,
                "core_name": core_name,
                "disp_path": disp_path,
                "delta_path": delta_path,
                "signed_path": signed_path,
                "reverse_path": reverse_path,
                "abs_path": abs_path,
                "ground_path": ground_path,
                "reuse_disp_path": reuse_disp_path,
                "reuse_ground_path": reuse_ground_path,
                "has_reuse_disp": has_reuse_disp,
                "has_reuse_ground": has_reuse_ground,
                "already_done": already_done,
            })
            if not already_done and not has_reuse_disp:
                run_indices.append(item_index)

        batch_disp = {}
        if run_indices:
            if model is None:
                missing = [str(sample_meta[i]["reuse_disp_path"]) for i in run_indices[:8]]
                raise FileNotFoundError(f"Missing reusable disparity files, examples: {missing}")
            left_run = left[run_indices].to(args.device)
            right_run = right[run_indices].to(args.device)
            padder = InputPadder(left_run.shape, divis_by=32)
            left_pad, right_pad = padder.pad(left_run, right_run)

            use_amp = args.mixed_precision and str(args.device).startswith("cuda")
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                disp_pred = model(left_pad, right_pad, iters=args.valid_iters, test_mode=True)
            disp_pred = padder.unpad(disp_pred).detach().float().cpu().numpy()
            for out_index, item_index in enumerate(run_indices):
                batch_disp[item_index] = disp_pred[out_index, 0]

        skipped = 0
        for meta in sample_meta:
            item_index = meta["item_index"]
            left_path = meta["left_path"]
            right_path = meta["right_path"]
            label_path = meta["label_path"]
            logical_city = meta["logical_city"]
            core_name = meta["core_name"]
            disp_path = meta["disp_path"]
            delta_path = meta["delta_path"]
            signed_path = meta["signed_path"]
            reverse_path = meta["reverse_path"]
            abs_path = meta["abs_path"]
            ground_path = meta["ground_path"]

            if meta["already_done"] and args.skip_reports:
                skipped += 1
                processed += 1
                pbar.set_postfix({"processed": processed, "skipped": skipped, "instances": len(instance_rows)})
                continue

            if meta["already_done"]:
                disp_np = np.load(disp_path).astype(np.float32)
                delta_np = np.load(delta_path).astype(np.float32)
                signed_np = np.load(signed_path).astype(np.float32)
                reverse_np = np.load(reverse_path).astype(np.float32)
                abs_np = np.load(abs_path).astype(np.float32)
                ground_np = np.load(ground_path).astype(np.float32)
                skipped += 1
            elif meta["has_reuse_disp"]:
                disp_np = np.load(meta["reuse_disp_path"]).astype(np.float32)
            else:
                disp_np = batch_disp[item_index]

            mask_np = instance_mask_cpu[item_index].astype(np.int64)
            height_np = height_cpu[item_index].astype(np.float32)
            k_value = float(k_tensor[item_index].view(-1)[0].item())
            if meta["already_done"]:
                sample_instance_rows = _build_instance_rows_from_cached(
                    disp_np, delta_np, signed_np, reverse_np, abs_np, ground_np, mask_np, height_np, k_value, args
                )
            elif args.reuse_cached_ground:
                if not meta["has_reuse_ground"]:
                    raise FileNotFoundError(f"Missing reusable local ground map: {meta['reuse_ground_path']}")
                cached_ground_np = np.load(meta["reuse_ground_path"]).astype(np.float32)
                delta_np, signed_np, reverse_np, abs_np, ground_np, sample_instance_rows = _build_delta_from_cached_ground(
                    disp_np, cached_ground_np, mask_np, height_np, k_value, args
                )
            else:
                delta_np, signed_np, reverse_np, abs_np, ground_np, sample_instance_rows = _build_delta_and_instances(
                    disp_np, mask_np, height_np, k_value, args
                )

            if not meta["already_done"]:
                for path, arr in [
                    (disp_path, disp_np),
                    (delta_path, delta_np),
                    (signed_path, signed_np),
                    (reverse_path, reverse_np),
                    (abs_path, abs_np),
                    (ground_path, ground_np),
                ]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, arr.astype(np.float32))

            valid_delta = np.isfinite(abs_np)
            height_physical = np.where(valid_delta, k_value * abs_np, np.nan).astype(np.float32)
            save_vis = (
                args.save_visuals
                and not meta["already_done"]
                and (args.save_visual_every <= 1 or processed % args.save_visual_every == 0)
            )
            if save_vis:
                disp_color = _colorize_scalar(
                    disp_np, valid_mask=np.isfinite(disp_np), cmap=args.disp_cmap,
                    vmin=args.min_disp, vmax=args.max_disp
                )
                delta_color = _colorize_scalar(
                    delta_np, valid_mask=valid_delta, cmap=args.delta_cmap,
                    vmin=0.0, vmax=args.delta_vmax
                )
                height_color = _colorize_scalar(
                    height_physical, valid_mask=valid_delta, cmap=args.height_cmap,
                    vmin=0.0, vmax=args.height_vmax
                )
                compare = np.concatenate([_to_uint8_rgb(left[item_index]), disp_color, delta_color, height_color], axis=1)
                _save_png(vis_dirs["disp"] / logical_city / f"Disp_{core_name}.png", disp_color)
                _save_png(vis_dirs["delta"] / logical_city / f"DeltaDisp_{core_name}.png", delta_color)
                _save_png(vis_dirs["height_physical"] / logical_city / f"HeightPhysical_{core_name}.png", height_color)
                _save_png(vis_dirs["compare"] / logical_city / f"Compare_{core_name}.png", compare)

            for row in sample_instance_rows:
                row.update({
                    "city": logical_city,
                    "core_name": core_name,
                    "left_path": left_path,
                    "right_path": right_path,
                    "label_path": label_path,
                    "k": k_value,
                    "disp_npy": str(disp_path),
                    "delta_npy": str(abs_path),
                    "delta_relu_npy": str(delta_path),
                    "delta_abs_npy": str(abs_path),
                    "ground_npy": str(ground_path),
                })
                instance_rows.append(row)

            gt_positive = height_np[np.isfinite(height_np) & (height_np > args.label_min_height)]
            manifest_rows.append({
                "city": logical_city,
                "core_name": core_name,
                "left_path": left_path,
                "right_path": right_path,
                "label_path": label_path,
                "k": k_value,
                "instance_count": len(sample_instance_rows),
                "label_positive_pixels": int(gt_positive.size),
                "label_mean": float(gt_positive.mean()) if gt_positive.size else 0.0,
                "label_max": float(gt_positive.max()) if gt_positive.size else 0.0,
                "disp_npy": str(disp_path),
                "delta_npy": str(abs_path),
                "delta_relu_npy": str(delta_path),
                "delta_signed_npy": str(signed_path),
                "delta_reverse_npy": str(reverse_path),
                "delta_abs_npy": str(abs_path),
                "ground_npy": str(ground_path),
            })
            processed += 1
            pbar.set_postfix({"processed": processed, "skipped": skipped, "instances": len(instance_rows)})

    if args.skip_reports:
        print(f"[*] Processed samples: {processed}")
        print(f"[*] Instance rows: {len(instance_rows)}")
        print(f"[*] Output directory: {out_dir}")
        print("[*] Report writing skipped by --skip_reports")
        return

    _write_csv(out_dir / "manifest.csv", manifest_rows)
    _write_csv(out_dir / "instance_metrics.csv", instance_rows)

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

    _write_csv(out_dir / "physical_metrics_summary.csv", summary_rows)
    _write_csv(out_dir / "physical_metrics_by_bin.csv", bin_rows)

    calibration_rows, calibration_bin_rows = _calibration_report(instance_rows, args)
    if calibration_rows:
        _write_csv(out_dir / "calibration_metrics_summary.csv", calibration_rows)
        _write_csv(out_dir / "calibration_metrics_by_bin.csv", calibration_bin_rows)

    print(f"[*] Processed samples: {processed}")
    print(f"[*] Instance rows: {len(instance_rows)}")
    print(f"[*] Output directory: {out_dir}")
    if summary_rows:
        print("[*] Physical summary:")
        for row in summary_rows:
            if row["city"] == "ALL":
                print(
                    f"    {row['prediction']}: MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} "
                    f"R2={row['r2']:.3f} Bias={row['bias']:+.3f} N={row['count']}"
                )
    if calibration_rows:
        print("[*] Calibration test summary:")
        for row in calibration_rows:
            print(
                f"    {row['model']}: MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} "
                f"R2={row['r2']:.3f} Bias={row['bias']:+.3f} N={row['count']}"
            )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Recompute SD2H disparity and label-guided differential disparity for UnifiedBH."
    )
    parser.add_argument("--restore_ckpt", default="./checkpoints/pretrain_whusside_60ep_724/final.pth")
    parser.add_argument("--data_path", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/")
    parser.add_argument("--cities", nargs="+", default=["HZ_049", "HZ_050", "HZ_057"])
    parser.add_argument("--train_cities", nargs="+", default=["HZ_049", "HZ_050"])
    parser.add_argument("--test_cities", nargs="+", default=["HZ_057"])
    parser.add_argument("--output_directory", default="/home/shangying/workspace/dataset/UnifiedBH/overlap15/disp/sd2h_recomputed_whusside724")
    parser.add_argument("--reuse_disp_root", default="", help="Optional cache root containing disp_npy to reuse while recomputing delta maps.")
    parser.add_argument("--reuse_disp_only", action="store_true", help="Do not instantiate SD2H; require --reuse_disp_root for every selected sample.")
    parser.add_argument("--reuse_cached_ground", action="store_true", help="Reuse local_ground_npy from --reuse_disp_root and only repair outlier ground values.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples_per_city", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument("--min_instance_pixels", type=int, default=16)
    parser.add_argument("--min_ring_pixels", type=int, default=32)
    parser.add_argument("--ring_inner_radius", type=int, default=3)
    parser.add_argument("--ring_outer_radius", type=int, default=35)
    parser.add_argument("--fallback_ring_radius", type=int, default=80)
    parser.add_argument("--ground_percentile", type=float, default=50.0)
    parser.add_argument("--correct_ground_outliers", action="store_true")
    parser.add_argument("--ground_outlier_abs_tol", type=float, default=3.0)
    parser.add_argument("--ground_outlier_mad_scale", type=float, default=3.5)
    parser.add_argument("--ground_outlier_min_instances", type=int, default=3)
    parser.add_argument("--label_min_height", type=float, default=0.0)
    parser.add_argument("--height_bins", nargs="+", type=float, default=[12, 24, 30, 50, 100, 150, 300])

    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--save_visual_every", type=int, default=50)
    parser.add_argument("--skip_reports", action="store_true", help="Only write cache files, not manifest/metrics CSV files.")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--min_disp", type=int, default=-320)
    parser.add_argument("--max_disp", type=int, default=64)
    parser.add_argument("--delta_vmax", type=float, default=80.0)
    parser.add_argument("--height_vmax", type=float, default=120.0)
    parser.add_argument("--disp_cmap", default="turbo_r")
    parser.add_argument("--delta_cmap", default="turbo")
    parser.add_argument("--height_cmap", default="turbo")

    parser.add_argument("--encoder", default="vitb")
    parser.add_argument("--use_mono", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--use_stereo", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=4)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    return parser


def main():
    args = build_argparser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    args.data_path = str(Path(args.data_path))
    missing_k = [c for c in args.cities if c not in GF7_CITY_K_MAPPING]
    if missing_k:
        raise ValueError(f"Cities missing K values: {missing_k}")

    if args.reuse_disp_only:
        if not args.reuse_disp_root:
            raise ValueError("--reuse_disp_only requires --reuse_disp_root")
        model = None
    else:
        model = _load_sd2h(args)
    dataset = UnifiedBHDataset(
        aug_params={},
        data_path=args.data_path,
        cities=args.cities,
        is_train=False,
        mask_from_label=True,
        mask_label_threshold=args.label_min_height,
        mask_connectivity=2,
        strict_mask=False,
    )
    selected_indices = _select_dataset_indices(dataset, args)
    if not selected_indices:
        raise RuntimeError("No samples selected for precompute.")
    if len(selected_indices) < len(dataset):
        dataset = torch.utils.data.Subset(dataset, selected_indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    run_precompute(model, loader, args)


if __name__ == "__main__":
    main()
