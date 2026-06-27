import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import core.height_datasets as datasets
from core.sd2h_height_finetuner import SD2H_Height_Finetuner
from core.utils.utils import InputPadder
from evaluate_height import (
    _apply_config_yaml,
    _arg_value,
    _collect_instance_means,
    _density_cmap,
    _pad_like_input_padder,
    _preferred_serif_font,
    _regression_metrics,
    _robust_fitted_line,
    _save_scatter,
)


def _raw_path_from_info(info, batch_index):
    if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], (list, tuple)):
        return str(info[0][batch_index])
    if isinstance(info, (list, tuple)):
        return str(info[batch_index])
    return str(info)


def _city_from_path(path):
    p = Path(path)
    for token in p.parts:
        if token.startswith(("HZ_", "SZ_", "HK_", "PRE_")):
            return token
    return p.parent.name


class _FigureCachedDataset(torch.utils.data.Dataset):
    """Evaluation-only dataset that keeps left-image features but skips unused right-image I/O."""

    def __init__(self, **kwargs):
        self.base = datasets.UnifiedBHDataset(**kwargs)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        index = index % len(self.base.image_list)
        left_img_path, right_img_path = self.base.image_list[index]

        img1 = datasets.read_image_as_float32(left_img_path)
        height = datasets.read_single_channel_as_float32(self.base.label_list[index])
        mask_path = self.base.mask_list[index]
        if mask_path is not None:
            mask = datasets.read_single_channel_as_float32(mask_path)
        else:
            mask = datasets.label_to_instance_mask(
                height,
                threshold=self.base.mask_label_threshold,
                connectivity=self.base.mask_connectivity,
            )
        mask = datasets.remove_small_instances(mask, self.base.mask_min_instance_pixels)
        mask = self.base._apply_instance_filter(mask, left_img_path)

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.zeros_like(img1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        height = torch.from_numpy(height).unsqueeze(0)

        precomputed_disp = torch.zeros_like(mask)
        precomputed_delta = torch.zeros_like(mask)
        precomputed_ground = torch.zeros_like(mask)
        disp_path, delta_path, ground_path = self.base._get_precomputed_paths(left_img_path)
        if disp_path is not None and delta_path is not None:
            if Path(disp_path).exists() and Path(delta_path).exists():
                precomputed_disp = torch.from_numpy(datasets.read_npy_as_float32(disp_path)).unsqueeze(0)
                precomputed_delta = torch.from_numpy(datasets.read_npy_as_float32(delta_path)).unsqueeze(0)
            elif self.base.require_precomputed_disp_delta:
                missing = [p for p in (disp_path, delta_path) if not Path(p).exists()]
                raise FileNotFoundError(f"Missing precomputed disparity/delta files: {missing}")
            if self.base.return_precomputed_ground and ground_path is not None:
                if Path(ground_path).exists():
                    precomputed_ground = torch.from_numpy(datasets.read_npy_as_float32(ground_path)).unsqueeze(0)
                elif self.base.require_precomputed_disp_delta:
                    raise FileNotFoundError(f"Missing precomputed local-ground disparity file: {ground_path}")

        if self.base.img_pad is not None:
            pad_h, pad_w = self.base.img_pad
            img1 = F.pad(img1, [pad_w] * 2 + [pad_h] * 2)
            img2 = F.pad(img2, [pad_w] * 2 + [pad_h] * 2)
            mask = F.pad(mask, [pad_w] * 2 + [pad_h] * 2)
            height = F.pad(height, [pad_w] * 2 + [pad_h] * 2)
            precomputed_disp = F.pad(precomputed_disp, [pad_w] * 2 + [pad_h] * 2)
            precomputed_delta = F.pad(precomputed_delta, [pad_w] * 2 + [pad_h] * 2)
            precomputed_ground = F.pad(precomputed_ground, [pad_w] * 2 + [pad_h] * 2)

        info = [
            left_img_path,
            right_img_path,
            self.base.label_list[index],
            mask_path if mask_path is not None else "AUTO_MASK_FROM_LABEL",
        ]
        k_tensor = torch.tensor([self.base._get_k_value_from_path(left_img_path)], dtype=torch.float32)
        return info, img1, img2, precomputed_disp, precomputed_delta, precomputed_ground, mask, height, k_tensor


@torch.no_grad()
def collect_predictions(model, args, cities):
    eval_manifest = _arg_value(
        args,
        "val_instance_filter_manifest",
        _arg_value(args, "instance_filter_manifest", ""),
    )
    dataset_kwargs = dict(
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
        instance_filter_manifest=(
            eval_manifest
            if bool(_arg_value(args, "apply_instance_filter_val", True))
            else ""
        ),
        drop_empty_after_instance_filter=bool(_arg_value(args, "drop_empty_after_instance_filter", False)),
    )
    use_fast_cached_dataset = (
        str(_arg_value(args, "height_delta_source", "")).lower() == "precomputed"
        and bool(_arg_value(args, "require_precomputed_disp_delta", False))
    )
    dataset = _FigureCachedDataset(**dataset_kwargs) if use_fast_cached_dataset else datasets.UnifiedBHDataset(**dataset_kwargs)
    if use_fast_cached_dataset:
        print("[*] Figure fast dataset enabled: right-image I/O is skipped; left-image features are preserved.")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    y_true_by_city = defaultdict(list)
    y_pred_by_city = defaultdict(list)
    model.eval()

    for data in tqdm(loader, dynamic_ncols=True, desc=f"Collecting predictions ({','.join(cities)})"):
        if len(data) == 9:
            info, left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor = data
        else:
            info, left, right = data[0], data[1], data[2]
            cached_disp, cached_delta = data[3], data[4]
            cached_ground = None
            instance_mask, height_gt, k_tensor = data[5], data[6], data[7]

        left = left.to(args.device, non_blocking=True)
        right = right.to(args.device, non_blocking=True)
        cached_disp = cached_disp.to(args.device, non_blocking=True)
        cached_delta = cached_delta.to(args.device, non_blocking=True)
        if cached_ground is not None:
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
        if cached_ground is not None and cached_ground.dim() == 3:
            cached_ground = cached_ground.unsqueeze(1)

        instance_mask_pad = _pad_like_input_padder(instance_mask, padder, mode="constant", value=0)
        cached_disp_pad = _pad_like_input_padder(cached_disp, padder, mode="constant", value=0)
        cached_delta_pad = _pad_like_input_padder(cached_delta, padder, mode="constant", value=0)
        cached_ground_pad = (
            _pad_like_input_padder(cached_ground, padder, mode="constant", value=0)
            if cached_ground is not None
            else None
        )

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
            gt_m, pred_m = _collect_instance_means(
                instance_mask[b : b + 1],
                height_gt[b : b + 1],
                h_pred[b : b + 1],
            )
            if gt_m is None:
                continue
            min_pred_height = getattr(args, "min_pred_height", None)
            if min_pred_height is not None:
                keep = pred_m >= float(min_pred_height)
                gt_m = gt_m[keep]
                pred_m = pred_m[keep]
            if gt_m.numel() == 0:
                continue
            city = _city_from_path(_raw_path_from_info(info, b))
            y_true_by_city[city].append(gt_m.detach().cpu().numpy())
            y_pred_by_city[city].append(pred_m.detach().cpu().numpy())

    merged = {}
    for city in sorted(y_true_by_city):
        merged[city] = (
            np.concatenate(y_true_by_city[city]).astype(np.float64),
            np.concatenate(y_pred_by_city[city]).astype(np.float64),
        )
    return merged


def _plot_city_error_boxplot(path, by_city, train_cities, val_cities, y_max=40.0, color_mode="validation"):
    train_set = set(train_cities)
    val_set = set(val_cities)
    city_order = [c for c in val_cities if c in by_city]
    errors = [np.abs(by_city[c][1] - by_city[c][0]) for c in city_order]
    y_max = float(y_max)
    clipped_errors = [np.clip(e, 0, y_max) for e in errors]

    with plt.rc_context({
        "font.family": _preferred_serif_font(),
        "font.size": 13,
        "axes.labelsize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.1,
    }):
        fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=300)
        bp = ax.boxplot(
            clipped_errors,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            medianprops=dict(color="#222222", linewidth=1.4),
            whiskerprops=dict(color="#555555", linewidth=1.0),
            capprops=dict(color="#555555", linewidth=1.0),
        )

        train_color = "#8ecae6"
        val_color = "#ffb703"
        if color_mode == "train":
            box_color = train_color
            legend_label = "Training/validation cities"
            legend_color = train_color
        else:
            box_color = val_color
            legend_label = "Validation cities"
            legend_color = val_color
        for patch, city in zip(bp["boxes"], city_order):
            patch.set_facecolor(box_color)
            patch.set_edgecolor("#4d4d4d")
            patch.set_linewidth(1.0)

        for idx, city in enumerate(city_order, start=1):
            stats = _regression_metrics(by_city[city][0], by_city[city][1])
            ax.scatter(idx, min(stats["mae"], y_max), marker="D", s=38, color="#023047", zorder=4)
            ax.scatter(idx, min(stats["rmse"], y_max), marker="o", s=40, color="#ff0000", zorder=4)

        ax.set_ylabel("Building-level absolute error (m)")
        ax.set_xlabel("")
        ax.set_ylim(0, y_max)
        ax.set_xticks(np.arange(1, len(city_order) + 1))
        ax.set_xticklabels([c.replace("_", "\\_") for c in city_order], rotation=35, ha="right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.scatter([], [], marker="D", s=38, color="#023047", label="MAE")
        ax.scatter([], [], marker="o", s=40, color="#ff0000", label="RMSE")
        ax.bar([], [], color=legend_color, edgecolor="#4d4d4d", label=legend_label)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.55", ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=600, bbox_inches="tight")
        fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _draw_scatter_panel(ax, y_true, y_pred, max_points=80000, title=None, max_axis=None, show_xlabel=True, show_ylabel=True):
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_all = y_true[finite].astype(np.float64)
    y_pred_all = y_pred[finite].astype(np.float64)
    if y_true_all.size == 0:
        ax.set_axis_off()
        return None

    stats = _regression_metrics(y_true_all, y_pred_all)
    slope, intercept = _robust_fitted_line(y_true_all, y_pred_all)

    plot_true = y_true_all
    plot_pred = y_pred_all
    if plot_true.size > max_points:
        rng = np.random.default_rng(2026)
        keep = rng.choice(plot_true.size, size=max_points, replace=False)
        plot_true = plot_true[keep]
        plot_pred = plot_pred[keep]

    if max_axis is None:
        max_axis = float(max(np.nanmax(y_true_all), np.nanmax(y_pred_all), 10.0))
        max_axis = min(max_axis * 1.08, 320.0)
    axis_line = np.array([0.0, max_axis], dtype=np.float64)
    density_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=50.0, clip=True)
    hb = ax.hexbin(
        plot_true,
        plot_pred,
        gridsize=58,
        extent=(0, max_axis, 0, max_axis),
        cmap=_density_cmap(),
        mincnt=1,
        norm=density_norm,
        linewidths=0.0,
    )
    ax.plot(axis_line, axis_line, color="#d62728", linestyle="-", linewidth=1.45)
    ax.plot(axis_line, intercept + slope * axis_line, color="#1f77b4", linestyle="-", linewidth=1.45)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title.replace("_", "\\_"), fontsize=14, pad=5)
    ax.set_xlabel("Reference height (m)" if show_xlabel else "")
    ax.set_ylabel("Estimated height (m)" if show_ylabel else "")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3)
    text = (
        f"$N$={stats['count']:,}\n"
        f"MAE={stats['mae']:.2f} m\n"
        f"RMSE={stats['rmse']:.2f} m\n"
        f"$R^2$={stats['r2']:.2f}"
    )
    ax.text(
        0.045,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.24,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.55", alpha=0.86, linewidth=0.6),
    )
    return hb


def _plot_scatter_grid(path, panels, max_points=80000, ncols=3):
    panels = [(name, y_true, y_pred) for name, y_true, y_pred in panels if y_true.size > 0]
    if not panels:
        return
    ncols = int(ncols)
    nrows = int(np.ceil(len(panels) / ncols))
    max_axis = max(
        float(max(np.nanmax(y_true), np.nanmax(y_pred), 10.0))
        for _, y_true, y_pred in panels
    )
    max_axis = min(max_axis * 1.08, 320.0)

    with plt.rc_context({
        "font.family": _preferred_serif_font(),
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 11.5), dpi=300)
        axes = np.asarray(axes).reshape(-1)
        last_hb = None
        for idx, ax in enumerate(axes):
            if idx >= len(panels):
                ax.set_axis_off()
                continue
            name, y_true, y_pred = panels[idx]
            row = idx // ncols
            col = idx % ncols
            last_hb = _draw_scatter_panel(
                ax,
                y_true,
                y_pred,
                max_points=max_points,
                title=name,
                max_axis=max_axis,
                show_xlabel=row == nrows - 1,
                show_ylabel=col == 0,
            )
        if last_hb is not None:
            cb = fig.colorbar(last_hb, ax=axes.tolist(), fraction=0.026, pad=0.018)
            cb.set_label("Point density", fontsize=13)
            cb.set_ticks([0, 10, 20, 30, 40, 50])
            cb.set_ticklabels(["0", "10", "20", "30", "40", "50+"])
            cb.ax.tick_params(labelsize=10)
        fig.savefig(path, dpi=600, bbox_inches="tight")
        fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _height_range_labels(edges):
    labels = []
    for idx in range(len(edges) - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        if np.isinf(hi):
            labels.append(f"H{idx + 1} ({lo:g}-inf)")
        else:
            labels.append(f"H{idx + 1} ({lo:g}-{hi:g})")
    return labels


def _plot_height_error_ridgeline(path, y_true, y_pred, edges=None, x_min=-120.0, x_max=90.0):
    if edges is None:
        edges = [3.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0, np.inf]
    edges = np.asarray(edges, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite].astype(np.float64)
    y_pred = y_pred[finite].astype(np.float64)
    if y_true.size == 0:
        return

    errors = y_pred - y_true
    x_grid = np.linspace(float(x_min), float(x_max), 900)
    x_edges = np.linspace(float(x_min), float(x_max), x_grid.size + 1)
    labels = _height_range_labels(edges)
    colors = [
        "#3b4cc0",
        "#6687ed",
        "#9bbcff",
        "#c6d6ee",
        "#e8cfc1",
        "#f6a385",
        "#e7644e",
        "#c4002f",
    ]
    if len(colors) != len(labels):
        cmap_tmp = plt.get_cmap("coolwarm")
        colors = [matplotlib.colors.to_hex(cmap_tmp(i / max(len(labels) - 1, 1))) for i in range(len(labels))]

    with plt.rc_context({
        "font.family": _preferred_serif_font(),
        "font.size": 18,
        "axes.labelsize": 27,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.2,
    }):
        fig, ax = plt.subplots(figsize=(12.6, 7.4), dpi=300)
        density_rows = []
        bin_stats = []

        for idx in range(len(edges) - 1):
            lo, hi = edges[idx], edges[idx + 1]
            if np.isinf(hi):
                mask = y_true >= lo
            else:
                mask = (y_true >= lo) & (y_true < hi)
            err = errors[mask]
            err = err[np.isfinite(err)]
            err_in_view = err[(err >= x_min) & (err <= x_max)]
            if err_in_view.size >= 2:
                hist, _ = np.histogram(err_in_view, bins=x_edges, density=False)
                density = gaussian_filter1d(hist.astype(np.float64), sigma=4.2, mode="nearest")
                density_max = float(np.nanmax(density))
                if density_max > 0.0:
                    density = density / density_max
                rmse = float(np.sqrt(np.mean(err ** 2)))
            elif err_in_view.size == 1:
                density = np.zeros_like(x_grid)
                peak_idx = int(np.argmin(np.abs(x_grid - err_in_view[0])))
                density[peak_idx] = 1.0
                density = gaussian_filter1d(density, sigma=8.0, mode="nearest")
                density_max = float(np.nanmax(density))
                if density_max > 0.0:
                    density = density / density_max
                rmse = float(np.sqrt(np.mean(err ** 2)))
            else:
                density = np.zeros_like(x_grid)
                rmse = float(np.sqrt(np.mean(err ** 2))) if err.size > 0 else np.nan
            density_rows.append(density)
            bin_stats.append((int(err.size), rmse))

        vertical_scale = 0.70
        for idx, density in enumerate(density_rows):
            base = idx + 1
            y_curve = base + density * vertical_scale
            ax.hlines(base, x_min, x_max, color="black", linewidth=1.45, zorder=1)
            if bin_stats[idx][0] > 0:
                ax.fill_between(x_grid, base, y_curve, color=colors[idx], alpha=0.82, linewidth=0.0, zorder=2)
                ax.plot(x_grid, y_curve, color="black", linewidth=1.45, zorder=3)
                ax.text(
                    x_max - 2.0,
                    base + 0.18,
                    f"RMSE: {bin_stats[idx][1]:.1f} m",
                    ha="right",
                    va="center",
                    fontsize=23,
                    color="black",
                )
            else:
                ax.text(
                    x_max - 2.0,
                    base + 0.18,
                    "N = 0",
                    ha="right",
                    va="center",
                    fontsize=19,
                    color="0.35",
                )
            ax.text(
                x_min + 2.8,
                base + 0.18,
                f"H{idx + 1}",
                ha="left",
                va="center",
                fontsize=23,
                fontweight="bold",
                color="black",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, len(labels) + 1.5)
        ax.set_yticks([])
        ax.set_xlabel("Estimation Error (m)")
        ax.set_ylabel("Density")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.75, alpha=0.35)
        for y in range(1, len(labels) + 1):
            ax.axhline(y, color="0.72", linestyle="--", linewidth=0.55, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(len(labels) + 1), cmap.N)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.052, pad=0.040, ticks=np.arange(len(labels)) + 0.5)
        cb.ax.set_yticklabels(labels)
        cb.set_label("Height Range (m)", fontsize=25, fontweight="bold", labelpad=18)
        cb.ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(path, dpi=600, bbox_inches="tight")
        fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--restore_ckpt", default="")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--train_cities", nargs="+", required=True)
    parser.add_argument("--val_cities", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--valid_iters", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--max_scatter_points", type=int, default=120000)
    parser.add_argument("--figure_prefix", default="strategy2")
    parser.add_argument("--boxplot_only", action="store_true")
    parser.add_argument("--skip_overall_scatter", action="store_true")
    parser.add_argument("--boxplot_ymax", type=float, default=40.0)
    parser.add_argument("--boxplot_color_mode", choices=["train", "validation"], default="validation")
    parser.add_argument(
        "--scatter_grid_mode",
        choices=["none", "overall_plus_cities", "cities_only"],
        default="none",
        help="Optional 3x3 scatter grid. overall_plus_cities uses overall validation plus city panels.",
    )
    parser.add_argument("--height_error_ridgeline", action="store_true")
    parser.add_argument("--height_error_ridgeline_only", action="store_true")
    parser.add_argument("--ridgeline_xmin", type=float, default=-120.0)
    parser.add_argument("--ridgeline_xmax", type=float, default=90.0)
    parser.add_argument("--prediction_cache_npz", default="")
    parser.add_argument("--save_prediction_cache", action="store_true")
    parser.add_argument("--min_pred_height", type=float, default=None)
    parser.add_argument("--no_mask_from_label", action="store_true")
    parser.add_argument("--mask_label_threshold", type=float, default=0.0)
    parser.add_argument("--mask_connectivity", type=int, default=2)
    parser.add_argument("--mask_min_instance_pixels", type=int, default=None)
    parser.add_argument("--strict_mask", action="store_true")
    parser.add_argument("--instance_filter_manifest", default=None)
    parser.add_argument("--train_instance_filter_manifest", default=None)
    parser.add_argument("--val_instance_filter_manifest", default=None)
    parser.add_argument("--apply_instance_filter_val", action="store_true", default=None)
    parser.add_argument("--no_apply_instance_filter_val", dest="apply_instance_filter_val", action="store_false")
    parser.add_argument("--drop_empty_after_instance_filter", action="store_true", default=None)
    parser.add_argument("--precomputed_disp_delta_root", default=None)
    parser.add_argument("--precomputed_disp_subdir", default=None)
    parser.add_argument("--precomputed_delta_subdir", default=None)
    parser.add_argument("--precomputed_ground_subdir", default=None)
    parser.add_argument("--return_precomputed_ground", action="store_true", default=None)
    parser.add_argument("--require_precomputed_disp_delta", action="store_true", default=None)
    args = parser.parse_args()
    args.eval_cities = args.val_cities
    args = _apply_config_yaml(args)

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.prediction_cache_npz) if args.prediction_cache_npz else None
    if cache_path is not None and cache_path.exists():
        print(f"[*] Loading prediction cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        by_city = {
            str(city): (cached[f"{city}_true"].astype(np.float64), cached[f"{city}_pred"].astype(np.float64))
            for city in cached["cities"]
        }
    else:
        model = SD2H_Height_Finetuner(args)
        if args.restore_ckpt:
            print(f"[*] Loading height checkpoint: {args.restore_ckpt}")
            checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            print("[*] No height checkpoint supplied; using calibration parameters from config.")
        model.to(args.device)

        by_city = collect_predictions(model, args, args.val_cities)
        if cache_path is not None and args.save_prediction_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"cities": np.asarray(list(by_city.keys()), dtype=object)}
            for city, (true_values, pred_values) in by_city.items():
                payload[f"{city}_true"] = true_values
                payload[f"{city}_pred"] = pred_values
            np.savez_compressed(cache_path, **payload)
            print(f"[*] Saved prediction cache: {cache_path}")

    val_true = np.concatenate([by_city[c][0] for c in args.val_cities if c in by_city])
    val_pred = np.concatenate([by_city[c][1] for c in args.val_cities if c in by_city])
    overall = _regression_metrics(val_true, val_pred)

    city_scatter_paths = []
    scatter_path = output_dir / f"{args.figure_prefix}_all_validation_scatter.png"
    if not args.boxplot_only and not args.height_error_ridgeline_only:
        if not args.skip_overall_scatter:
            _save_scatter(scatter_path, val_true, val_pred, max_points=args.max_scatter_points, title="Validation cities")
        for city in args.val_cities:
            if city not in by_city:
                continue
            city_path = output_dir / f"{args.figure_prefix}_scatter_{city}.png"
            _save_scatter(city_path, by_city[city][0], by_city[city][1], max_points=args.max_scatter_points, title=city)
            city_scatter_paths.append(city_path)
        if args.scatter_grid_mode != "none":
            if args.scatter_grid_mode == "overall_plus_cities":
                panels = [("Overall validation", val_true, val_pred)]
                panels.extend((city, by_city[city][0], by_city[city][1]) for city in args.val_cities if city in by_city)
            else:
                panels = [(city, by_city[city][0], by_city[city][1]) for city in args.val_cities if city in by_city]
            grid_path = output_dir / f"{args.figure_prefix}_scatter_grid_3x3.png"
            _plot_scatter_grid(grid_path, panels[:9], max_points=args.max_scatter_points, ncols=3)
            city_scatter_paths.append(grid_path)
    boxplot_path = output_dir / f"{args.figure_prefix}_city_error_boxplot.png"
    if not args.height_error_ridgeline_only:
        _plot_city_error_boxplot(
            boxplot_path,
            by_city,
            args.train_cities,
            args.val_cities,
            y_max=args.boxplot_ymax,
            color_mode=args.boxplot_color_mode,
        )
    ridgeline_path = output_dir / f"{args.figure_prefix}_height_error_ridgeline.png"
    if args.height_error_ridgeline or args.height_error_ridgeline_only:
        _plot_height_error_ridgeline(
            ridgeline_path,
            val_true,
            val_pred,
            x_min=args.ridgeline_xmin,
            x_max=args.ridgeline_xmax,
        )

    print("\n[Validation summary]")
    print(f"N={overall['count']} | MAE={overall['mae']:.4f} | RMSE={overall['rmse']:.4f} | R2={overall['r2']:.4f} | Bias={overall['bias']:+.4f}")
    print("\n[City metrics]")
    for city in args.val_cities:
        if city not in by_city:
            continue
        split = "val"
        stats = _regression_metrics(by_city[city][0], by_city[city][1])
        print(f"{city:<10} {split:<5} N={stats['count']:>7d} MAE={stats['mae']:.4f} RMSE={stats['rmse']:.4f} R2={stats['r2']:.4f}")
    saved_main = [str(boxplot_path)]
    if not args.skip_overall_scatter:
        saved_main.insert(0, str(scatter_path))
    print(f"\n[*] Saved figures: {', '.join(saved_main)}")
    if args.height_error_ridgeline or args.height_error_ridgeline_only:
        print(f"    {ridgeline_path}")
    for path in city_scatter_paths:
        print(f"    {path}")


if __name__ == "__main__":
    main()
