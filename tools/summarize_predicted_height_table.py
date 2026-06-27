#!/usr/bin/env python3
"""Summarize predicted building-height distributions for a finetuned SD2H model."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.sd2h_height_finetuner import SD2H_Height_Finetuner
from core.utils.utils import InputPadder
from evaluate_height import (
    _apply_config_yaml,
    _arg_value,
    _collect_instance_means,
    _pad_like_input_padder,
)
from plotting.make_unifiedbh_figures import _FigureCachedDataset, _city_from_path, _raw_path_from_info


@torch.no_grad()
def collect_predicted_heights(model, args, cities):
    eval_manifest = _arg_value(
        args,
        "val_instance_filter_manifest",
        _arg_value(args, "instance_filter_manifest", ""),
    )
    dataset = _FigureCachedDataset(
        aug_params={},
        data_path=args.data_path,
        cities=cities,
        mask_from_label=not args.no_mask_from_label,
        mask_label_threshold=args.mask_label_threshold,
        mask_connectivity=args.mask_connectivity,
        strict_mask=args.strict_mask,
        mask_min_instance_pixels=int(_arg_value(args, "mask_min_instance_pixels", 0)),
        precomputed_disp_delta_root=_arg_value(args, "precomputed_disp_delta_root", ""),
        precomputed_disp_subdir=_arg_value(args, "precomputed_disp_subdir", "disp_npy"),
        precomputed_delta_subdir=_arg_value(args, "precomputed_delta_subdir", "delta_relu_npy"),
        precomputed_ground_subdir=_arg_value(args, "precomputed_ground_subdir", ""),
        return_precomputed_ground=bool(_arg_value(args, "return_precomputed_ground", False)),
        require_precomputed_disp_delta=bool(_arg_value(args, "require_precomputed_disp_delta", False)),
        instance_filter_manifest=eval_manifest if bool(_arg_value(args, "apply_instance_filter_val", True)) else "",
        drop_empty_after_instance_filter=bool(_arg_value(args, "drop_empty_after_instance_filter", False)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    pred_by_city = defaultdict(list)
    patches_by_city = defaultdict(int)
    model.eval()

    for data in tqdm(loader, dynamic_ncols=True, desc="Summarizing predictions"):
        info, left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor = data
        left = left.to(args.device, non_blocking=True)
        right = right.to(args.device, non_blocking=True)
        cached_disp = cached_disp.to(args.device, non_blocking=True)
        cached_delta = cached_delta.to(args.device, non_blocking=True)
        cached_ground = cached_ground.to(args.device, non_blocking=True)
        instance_mask = instance_mask.to(args.device, non_blocking=True)
        height_gt = height_gt.to(args.device, non_blocking=True)
        k_tensor = k_tensor.to(args.device, non_blocking=True)
        if k_tensor.dim() == 1:
            k_tensor = k_tensor.unsqueeze(1)

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        if instance_mask.dim() == 3:
            instance_mask = instance_mask.unsqueeze(1)
        if height_gt.dim() == 3:
            height_gt = height_gt.unsqueeze(1)
        if cached_disp.dim() == 3:
            cached_disp = cached_disp.unsqueeze(1)
        if cached_delta.dim() == 3:
            cached_delta = cached_delta.unsqueeze(1)
        if cached_ground.dim() == 3:
            cached_ground = cached_ground.unsqueeze(1)

        instance_mask_pad = _pad_like_input_padder(instance_mask, padder, mode="constant", value=0)
        cached_disp_pad = _pad_like_input_padder(cached_disp, padder, mode="constant", value=0)
        cached_delta_pad = _pad_like_input_padder(cached_delta, padder, mode="constant", value=0)
        cached_ground_pad = _pad_like_input_padder(cached_ground, padder, mode="constant", value=0)

        with torch.amp.autocast("cuda", enabled=args.mixed_precision, dtype=torch.bfloat16):
            pred_out = model(
                left,
                right,
                mask=instance_mask_pad,
                k_tensor=k_tensor,
                iters=args.valid_iters,
                test_mode=True,
                precomputed_disp=cached_disp_pad,
                precomputed_delta=cached_delta_pad,
                precomputed_ground=cached_ground_pad,
            )
            h_pred = pred_out["height"] if isinstance(pred_out, dict) else pred_out

        h_pred = padder.unpad(h_pred)
        if h_pred.dim() == 3:
            h_pred = h_pred.unsqueeze(1)
        h_pred = F.relu(torch.nan_to_num(h_pred, nan=0.0, posinf=320.0, neginf=0.0))

        for b in range(h_pred.shape[0]):
            _, pred_m = _collect_instance_means(
                instance_mask[b : b + 1],
                height_gt[b : b + 1],
                h_pred[b : b + 1],
            )
            city = _city_from_path(_raw_path_from_info(info, b))
            patches_by_city[city] += 1
            if pred_m is None or pred_m.numel() == 0:
                continue
            pred = pred_m.detach().cpu().numpy().astype(np.float64)
            if args.min_pred_height is not None:
                pred = pred[pred >= float(args.min_pred_height)]
            if pred.size:
                pred_by_city[city].append(pred)

    merged = {}
    for city in cities:
        values = pred_by_city.get(city, [])
        merged[city] = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    return merged, dict(patches_by_city)


def summarize(values):
    if values.size == 0:
        return {
            "buildings": 0,
            "max": 0.0,
            "mean": 0.0,
            "sd": 0.0,
            "low": 0.0,
            "high": 0.0,
            "super": 0.0,
        }
    low = float(np.mean(values < 24.0) * 100.0)
    high = float(np.mean((values >= 24.0) & (values < 100.0)) * 100.0)
    super_high = float(np.mean(values >= 100.0) * 100.0)
    return {
        "buildings": int(values.size),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "low": low,
        "high": high,
        "super": super_high,
    }


def format_latex(rows, total, caption, label):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} lrrrrrrrr @{}}",
        r"\toprule",
        r"\multirow{2}{*}{City / Region} & \multirow{2}{*}{Patches} & \multirow{2}{*}{Buildings} & \multicolumn{3}{c}{Predicted building height (m)} & \multicolumn{3}{c}{Area proportion (\%)} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
        r"& & & Max & Mean & SD & Low-rise & High-rise & Super high-rise \\",
        r"\midrule",
    ]
    for city, patches, stats in rows:
        city_tex = city.replace("_", r"\_")
        lines.append(
            rf"{city_tex:<13} & {patches:,} & {stats['buildings']:,} & "
            rf"{stats['max']:.3f} & {stats['mean']:.3f} & {stats['sd']:.3f} & "
            rf"{stats['low']:.1f} & {stats['high']:.1f} & {stats['super']:.1f} \\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & \textbf{{{total['patches']:,}}} & \textbf{{{total['buildings']:,}}} & "
            rf"\textbf{{{total['max']:.3f}}} & \textbf{{{total['mean']:.3f}}} & \textbf{{{total['sd']:.3f}}} & "
            rf"\textbf{{{total['low']:.1f}}} & \textbf{{{total['high']:.1f}}} & \textbf{{{total['super']:.1f}}} \\",
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_cities", nargs="+", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_iters", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--min_pred_height", type=float, default=None)
    parser.add_argument("--no_mask_from_label", action="store_true")
    parser.add_argument("--mask_label_threshold", type=float, default=0.0)
    parser.add_argument("--mask_connectivity", type=int, default=2)
    parser.add_argument("--mask_min_instance_pixels", type=int, default=None)
    parser.add_argument("--strict_mask", action="store_true")
    args = parser.parse_args()
    args.eval_cities = args.val_cities
    args = _apply_config_yaml(args)

    model = SD2H_Height_Finetuner(args)
    checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)

    pred_by_city, patches_by_city = collect_predicted_heights(model, args, args.val_cities)
    rows = []
    all_values = []
    for city in args.val_cities:
        values = pred_by_city[city]
        all_values.append(values)
        rows.append((city, int(patches_by_city.get(city, 0)), summarize(values)))
    total_values = np.concatenate(all_values) if all_values else np.asarray([], dtype=np.float64)
    total_stats = summarize(total_values)
    total_stats["patches"] = int(sum(patches_by_city.values()))

    payload = {
        "rows": [
            {"city": city, "patches": patches, **stats}
            for city, patches, stats in rows
        ],
        "total": total_stats,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[*] Saved summary JSON: {out_path}")

    print()
    print(format_latex(
        rows,
        total_stats,
        "Statistical overview of the SD2H-Net predicted building heights under the intra-city 50/50 adaptation protocol.",
        "tab:unifiedbh_predicted_height_stats",
    ))


if __name__ == "__main__":
    main()
