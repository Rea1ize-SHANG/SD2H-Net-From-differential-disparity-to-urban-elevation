import os
import torch
import numpy as np
from tqdm import tqdm
import skimage.io
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    from sklearn.linear_model import HuberRegressor
except Exception:  # pragma: no cover - plotting still works without sklearn
    HuberRegressor = None

from core.sd2h_height_finetuner import SD2H_Height_Finetuner
from core.utils.utils import InputPadder
import core.height_datasets as datasets

# ================== Helper: convert height maps to color maps =================
def gray_2_colormap_np(img, cmap='magma', max_val=240.0):
    img = img.cpu().detach().numpy().squeeze()
    img[img < 0] = 0
    mask_invalid = img < 1e-3
    
    img_norm = np.clip(img / (max_val + 1e-8), 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap_m = matplotlib.colormaps[cmap]
    map_ = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    
    colormap = (map_.to_rgba(img_norm)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


def _pad_like_input_padder(x, padder, mode='constant', value=0):
    if sum(padder._pad) == 0:
        return x
    return F.pad(x, padder._pad, mode=mode, value=value)


def _collect_instance_means(instance_mask, height_gt, height_pred):
    b_size = height_pred.shape[0]
    batch_offsets = torch.arange(b_size, device=instance_mask.device).view(b_size, 1, 1, 1) * 100000
    global_inst_mask = torch.where(instance_mask > 0, instance_mask + batch_offsets, torch.zeros_like(instance_mask))
    inst_flat = global_inst_mask.view(-1).long()
    valid_mask = inst_flat > 0

    if not valid_mask.any():
        return None, None

    v_gt = height_gt.view(-1)[valid_mask]
    v_pred = height_pred.view(-1)[valid_mask]
    v_inst_ids = inst_flat[valid_mask]

    u_inst, inv_idx = torch.unique(v_inst_ids, return_inverse=True)
    num_i = u_inst.numel()

    counts = torch.zeros(num_i, dtype=v_gt.dtype, device=v_gt.device).scatter_add_(0, inv_idx, torch.ones_like(v_gt))
    gt_mean = torch.zeros(num_i, dtype=v_gt.dtype, device=v_gt.device).scatter_add_(0, inv_idx, v_gt) / counts
    pred_mean = torch.zeros(num_i, dtype=v_pred.dtype, device=v_pred.device).scatter_add_(0, inv_idx, v_pred) / counts
    return gt_mean, pred_mean


def _regression_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {'count': 0, 'mae': 0.0, 'rmse': 0.0, 'bias': 0.0, 'r2': 0.0}

    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2
    ss_res = float(np.sum(sq_diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        'count': int(y_true.size),
        'mae': float(np.mean(abs_diff)),
        'rmse': float(np.sqrt(np.mean(sq_diff))),
        'bias': float(np.mean(diff)),
        'r2': float(1.0 - ss_res / (ss_tot + 1e-8)),
    }


def _robust_fitted_line(y_true, y_pred, max_bins=32, min_bin_points=12):
    """Fit a robust trend line using x-bin medians followed by Huber regression."""
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    x = y_true[finite].astype(np.float64)
    y = y_pred[finite].astype(np.float64)
    if x.size < 2 or np.std(x) <= 1e-8:
        return 1.0, 0.0

    def fallback():
        try:
            slope, intercept = np.polyfit(x, y, 1)
            return float(slope), float(intercept)
        except Exception:
            return 1.0, 0.0

    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return fallback()

    edges = np.linspace(lo, hi, int(max_bins) + 1)
    bx, by = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x < right)
        if mask.sum() < int(min_bin_points):
            continue
        bx.append(float(np.median(x[mask])))
        by.append(float(np.median(y[mask])))

    if len(bx) < 3:
        return fallback()

    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    try:
        if HuberRegressor is not None:
            model = HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=1000)
            model.fit(bx.reshape(-1, 1), by)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
        else:
            slope, intercept = np.polyfit(bx, by, 1)
            slope, intercept = float(slope), float(intercept)
    except Exception:
        return fallback()

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return fallback()
    return slope, intercept


def _save_metrics_csv(path, rows):
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['bin', 'count', 'mae', 'rmse', 'bias'], extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_summary_csv(path, stats, include_bias=True):
    import csv

    fieldnames = ['count', 'mae', 'rmse', 'bias', 'r2'] if include_bias else ['count', 'mae', 'rmse', 'r2']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(stats)


def _save_scatter_points_csv(path, y_true, y_pred):
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gt_height', 'pred_height'])
        writer.writeheader()
        for gt, pred in zip(y_true.tolist(), y_pred.tolist()):
            writer.writerow({'gt_height': float(gt), 'pred_height': float(pred)})


def _cfg_without_defaults(cfg):
    data = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(data, dict):
        data.pop('defaults', None)
    return OmegaConf.create(data)


def _load_config_with_defaults(config_yaml, visited=None):
    """Resolve simple Hydra defaults used by the finetune configs."""
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
    defaults = cfg.get('defaults', [])
    merged = OmegaConf.create()
    self_merged = False

    def merge_self():
        nonlocal merged, self_merged
        merged = OmegaConf.merge(merged, _cfg_without_defaults(cfg))
        self_merged = True

    for item in defaults:
        if isinstance(item, str):
            if item == '_self_':
                merge_self()
                continue
            child_path = cfg_path.parent / f"{item}.yaml"
            if child_path.exists():
                merged = OmegaConf.merge(merged, _load_config_with_defaults(child_path, visited))
            continue

        if isinstance(item, dict):
            for group, name in item.items():
                if group == '_self_' or name == '_self_':
                    merge_self()
                    continue
                child_path = cfg_path.parent / str(group) / f"{name}.yaml"
                if child_path.exists():
                    merged = OmegaConf.merge(merged, _load_config_with_defaults(child_path, visited))

    if not self_merged:
        merge_self()
    return merged


def _preferred_serif_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ('Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'):
        if candidate in available:
            return candidate
    return 'serif'


def _density_cmap():
    return LinearSegmentedColormap.from_list(
        'sd2h_density_bright',
        [
            (0.00, '#08306b'),
            (0.25, '#2171b5'),
            (0.50, '#41b6c4'),
            (0.70, '#ffffbf'),
            (0.86, '#fdae61'),
            (1.00, '#ff0000'),
        ],
    )


def _apply_config_yaml(args):
    """Load the resolved training config while preserving runtime eval options."""
    config_yaml = getattr(args, 'config_yaml', '')
    if not config_yaml:
        return args

    runtime_keys = {
        'config_yaml',
        'device',
        'restore_ckpt',
        'data_path',
        'eval_cities',
        'output_directory',
        'batch_size',
        'valid_iters',
        'mixed_precision',
        'save_vis',
        'no_scatter',
        'no_bin_metrics',
        'save_scatter_csv',
        'max_scatter_points',
        'max_height_vis',
        'min_pred_height',
    }
    optional_override_keys = {
        'instance_filter_manifest',
        'apply_instance_filter_val',
        'drop_empty_after_instance_filter',
        'mask_min_instance_pixels',
        'precomputed_disp_delta_root',
        'precomputed_disp_subdir',
        'precomputed_delta_subdir',
        'precomputed_ground_subdir',
        'return_precomputed_ground',
        'require_precomputed_disp_delta',
    }
    runtime_values = {key: getattr(args, key, None) for key in runtime_keys if hasattr(args, key)}
    for key in optional_override_keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value is None:
            continue
        if isinstance(value, str) and value == '':
            continue
        runtime_values[key] = value
    cfg = _load_config_with_defaults(config_yaml)
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        setattr(args, key, value)
    for key, value in runtime_values.items():
        setattr(args, key, value)
    return args


def _arg_value(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


def _save_scatter(path, y_true, y_pred, max_points=80000, title=None):
    if y_true.size == 0:
        return

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_all = y_true[finite].astype(np.float64)
    y_pred_all = y_pred[finite].astype(np.float64)
    if y_true_all.size == 0:
        return

    stats = _regression_metrics(y_true_all, y_pred_all)
    if y_true_all.size >= 2 and np.std(y_true_all) > 1e-8:
        slope, intercept = np.polyfit(y_true_all, y_pred_all, 1)
    else:
        slope, intercept = 1.0, 0.0

    plot_true = y_true_all
    plot_pred = y_pred_all
    if plot_true.size > max_points:
        rng = np.random.default_rng(2026)
        keep = rng.choice(plot_true.size, size=max_points, replace=False)
        plot_true = plot_true[keep]
        plot_pred = plot_pred[keep]

    max_axis = float(max(np.nanmax(y_true_all), np.nanmax(y_pred_all), 10.0))
    max_axis = min(max_axis * 1.08, 320.0)
    axis_line = np.array([0.0, max_axis], dtype=np.float64)

    with plt.rc_context({
        'font.family': _preferred_serif_font(),
        'font.size': 17,
        'axes.labelsize': 20,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 16,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.2,
    }):
        fig, ax = plt.subplots(figsize=(7.6, 6.8), dpi=300)
        density_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=50.0, clip=True)
        hb = ax.hexbin(
            plot_true,
            plot_pred,
            gridsize=80,
            extent=(0, max_axis, 0, max_axis),
            cmap=_density_cmap(),
            mincnt=1,
            norm=density_norm,
            linewidths=0.0,
        )

        ax.plot(axis_line, axis_line, color='#d62728', linestyle='-', linewidth=2.1,
                label='Ideal regression line')
        ax.plot(axis_line, intercept + slope * axis_line, color='#1f77b4', linestyle='-', linewidth=2.1,
                label='Fitted regression line')

        metrics_text = (
            f"$N$ = {stats['count']:,}\n"
            f"MAE = {stats['mae']:.3f} m\n"
            f"RMSE = {stats['rmse']:.3f} m\n"
            f"$R^2$ = {stats['r2']:.3f}"
        )
        box_top = 0.980
        metrics_artist = ax.text(
            0.055,
            box_top,
            metrics_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=17,
            linespacing=1.42,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='0.55', alpha=0.88, linewidth=0.8),
        )

        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Reference height (m)')
        ax.set_ylabel('Estimated height (m)')
        if title:
            ax.set_title(title, fontsize=20, pad=10)
        ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.30)
        legend = ax.legend(
            loc='upper right',
            bbox_to_anchor=(0.980, box_top),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.88,
            edgecolor='0.55',
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.035)
        cb.set_label('Point density', fontsize=18)
        cb.set_ticks([0, 10, 20, 30, 40, 50])
        cb.set_ticklabels(['0', '10', '20', '30', '40', '50+'])
        cb.ax.tick_params(labelsize=15)

        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        metrics_bbox = metrics_artist.get_bbox_patch().get_window_extent(renderer)
        legend_bbox = legend.get_frame().get_window_extent(renderer)
        pixel_shift = legend_bbox.y1 - metrics_bbox.y1
        if abs(pixel_shift) > 0.5:
            x0, y0 = ax.transAxes.transform(metrics_artist.get_position())
            _, aligned_y = ax.transAxes.inverted().transform((x0, y0 + pixel_shift))
            metrics_artist.set_position((metrics_artist.get_position()[0], aligned_y))
            fig.canvas.draw()
        fig.savefig(path, dpi=600, bbox_inches='tight')
        fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)

@torch.no_grad()
def evaluate_height(model, val_loader, args):
    model.eval()
    
    # Metric accumulators.
    total_abs_err = 0.0
    total_sq_err = 0.0
    all_preds_vec = []
    all_gts_vec = []
    total_buildings_count = 0 
    filtered_pred_low_count = 0

    output_dir = Path(args.output_directory)
    if args.save_vis:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Height prediction maps will be saved to: {output_dir.absolute()}")

    pbar_val = tqdm(val_loader, dynamic_ncols=True, desc="Evaluating Height")
    for batch_idx, data in enumerate(pbar_val):
        # UnifiedBHDataset returns an optional cached ground map when
        # return_precomputed_ground=true. Keep this unpacking aligned with
        # train_finetune.py; otherwise the ground map is mistaken for a mask.
        if len(data) == 9:
            info, left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor = data
        else:
            info, left, right = data[0], data[1], data[2]
            cached_disp, cached_delta = data[3], data[4]
            cached_ground = None
            instance_mask, height_gt, k_tensor = data[5], data[6], data[7]
        
        left, right = left.to(args.device), right.to(args.device)
        cached_disp = cached_disp.to(args.device)
        cached_delta = cached_delta.to(args.device)
        if cached_ground is not None:
            cached_ground = cached_ground.to(args.device)
        instance_mask = instance_mask.to(args.device)
        height_gt = height_gt.to(args.device)
        k_tensor = k_tensor.to(args.device)
        if k_tensor.dim() == 1:
            k_tensor = k_tensor.unsqueeze(1)

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        if instance_mask.dim() == 3: instance_mask = instance_mask.unsqueeze(1)
        if height_gt.dim() == 3: height_gt = height_gt.unsqueeze(1)
        if cached_disp.dim() == 3: cached_disp = cached_disp.unsqueeze(1)
        if cached_delta.dim() == 3: cached_delta = cached_delta.unsqueeze(1)
        if cached_ground is not None and cached_ground.dim() == 3:
            cached_ground = cached_ground.unsqueeze(1)
        instance_mask_pad = _pad_like_input_padder(instance_mask, padder, mode='constant', value=0)
        cached_disp_pad = _pad_like_input_padder(cached_disp, padder, mode='constant', value=0)
        cached_delta_pad = _pad_like_input_padder(cached_delta, padder, mode='constant', value=0)
        cached_ground_pad = (
            _pad_like_input_padder(cached_ground, padder, mode='constant', value=0)
            if cached_ground is not None else None
        )

        # ====== Model forward pass ======
        with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=torch.bfloat16):
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
            h_pred = pred_out['height'] if isinstance(pred_out, dict) else pred_out
            
        h_pred = padder.unpad(h_pred)

        # Normalize tensor dimensions defensively.
        if h_pred.dim() == 3: h_pred = h_pred.unsqueeze(1)
        h_pred = F.relu(torch.nan_to_num(h_pred, nan=0.0, posinf=320.0, neginf=0.0))
        b_size = h_pred.shape[0]

        # ====== Instance aggregation aligned with training logic ======
        g_m, p_m = _collect_instance_means(instance_mask, height_gt, h_pred)
        if g_m is not None:
            min_pred_height = getattr(args, 'min_pred_height', None)
            if min_pred_height is not None:
                keep_pred = p_m >= float(min_pred_height)
                filtered_pred_low_count += int((~keep_pred).sum().item())
                g_m = g_m[keep_pred]
                p_m = p_m[keep_pred]
            num_i = g_m.numel()
            if num_i == 0:
                continue
            
            # Accumulate instance-level metrics.
            diff = torch.abs(p_m - g_m)
            total_abs_err += diff.sum().item()
            total_sq_err += (diff ** 2).sum().item()
            total_buildings_count += num_i
            
            # Store vectors for global R2 computation.
            all_preds_vec.append(p_m.cpu().numpy())
            all_gts_vec.append(g_m.cpu().numpy())
        # ==============================================================

        # Save visualizations.
        if args.save_vis:
            h_pred_vis = h_pred.squeeze(1)
            for b in range(b_size):
                # Handle nested paths returned by PyTorch DataLoader.
                try:
                    if isinstance(info, (list, tuple)) and isinstance(info[0], (list, tuple)):
                        raw_path = info[0][b] 
                    elif isinstance(info, (list, tuple)):
                        raw_path = info[b]
                    else:
                        raw_path = str(info)
                    filename = os.path.basename(raw_path)
                except Exception:
                    filename = f"batch{batch_idx:04d}_{b}.png"
                
                base_name = os.path.splitext(filename)[0]
                
                # Save outputs.
                tiff_path = output_dir / f"{base_name}_height.tiff"
                skimage.io.imsave(str(tiff_path), h_pred_vis[b].cpu().numpy().astype(np.float32), check_contrast=False)
                
                color_png = gray_2_colormap_np(h_pred_vis[b], max_val=args.max_height_vis)
                skimage.io.imsave(str(output_dir / f"{base_name}_height_color.png"), color_png, check_contrast=False)

    # ================= Final instance-level statistics =================
    mae = total_abs_err / total_buildings_count if total_buildings_count > 0 else 0
    rmse = (total_sq_err / total_buildings_count) ** 0.5 if total_buildings_count > 0 else 0
    
    # Compute global R2 with NumPy.
    if len(all_preds_vec) > 0:
        y_true = np.concatenate(all_gts_vec)
        y_pred = np.concatenate(all_preds_vec)
        overall_stats = _regression_metrics(y_true, y_pred)
        r2 = overall_stats['r2']
    else:
        y_true = np.array([], dtype=np.float32)
        y_pred = np.array([], dtype=np.float32)
        overall_stats = {'count': 0, 'mae': 0.0, 'rmse': 0.0, 'bias': 0.0, 'r2': 0.0}
        r2 = 0

    print("\n" + "===================== Building Height Evaluation =====================")
    print(f"N    : {total_buildings_count}")
    print(f"MAE  : {mae:.4f} m")
    print(f"RMSE : {rmse:.4f} m")
    print(f"R2   : {r2:.4f}")
    if filtered_pred_low_count > 0:
        print(f"[*] Filtered pred<{float(args.min_pred_height):.2f}m instances: {filtered_pred_low_count}")
    print("=======================================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "height_metrics_summary.csv"
    _save_summary_csv(summary_csv, overall_stats, include_bias=not getattr(args, 'no_bin_metrics', False))
    print(f"[*] Summary metrics saved to: {summary_csv}")

    metric_rows = []
    if y_true.size > 0:
        bins = list(args.height_bins)
        if bins[0] != 0:
            bins = [0.0] + bins
        scatter_png = output_dir / "height_scatter.png"
        if not getattr(args, 'no_bin_metrics', False):
            print("===================== Stratified Height Metrics =====================")
            print(f"{'Bin':<16} {'N':>8} {'MAE':>10} {'RMSE':>10} {'Bias':>10}")
            print("-" * 58)
            for idx in range(len(bins)):
                lower = bins[idx]
                upper = bins[idx + 1] if idx + 1 < len(bins) else np.inf
                if np.isinf(upper):
                    mask = y_true > lower
                    bin_name = f">{lower:g}m"
                else:
                    mask = (y_true > lower) & (y_true <= upper)
                    bin_name = f"{lower:g}-{upper:g}m"
                stats = _regression_metrics(y_true[mask], y_pred[mask])
                row = {'bin': bin_name, **stats}
                metric_rows.append(row)
                print(f"{bin_name:<16} {stats['count']:>8d} {stats['mae']:>10.3f} {stats['rmse']:>10.3f} {stats['bias']:>+10.3f}")
            print("=======================================================\n")

            metrics_csv = output_dir / "height_metrics_by_bin.csv"
            _save_metrics_csv(metrics_csv, metric_rows)
        if not args.no_scatter:
            _save_scatter(scatter_png, y_true, y_pred, max_points=args.max_scatter_points)
            print(f"[*] Scatter saved to: {scatter_png}")
        if getattr(args, 'save_scatter_csv', False):
            scatter_csv = output_dir / "height_scatter_points.csv"
            _save_scatter_points_csv(scatter_csv, y_true, y_pred)
            print(f"[*] Scatter points saved to: {scatter_csv}")
        if not getattr(args, 'no_bin_metrics', False):
            print(f"[*] Stratified metrics saved to: {metrics_csv}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'N': total_buildings_count}

def main(args):
    args = _apply_config_yaml(args)
    model = SD2H_Height_Finetuner(args)
    restore_ckpt = getattr(args, 'restore_ckpt', None)
    if restore_ckpt:
        print(f"[*] Loading height finetuning checkpoint: {restore_ckpt}")
        checkpoint = torch.load(restore_ckpt, map_location='cpu')
        ckpt_dict = checkpoint.get('state_dict', checkpoint)
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
        model.load_state_dict(ckpt_dict, strict=False)
    else:
        print("[*] No height finetuning checkpoint provided; evaluating with configured calibration parameters.")
    model.to(args.device)
    model.eval()

    print(f"[*] Loading evaluation data: {args.data_path}")
    val_dataset = datasets.UnifiedBHDataset(
        aug_params={},
        data_path=args.data_path,
        cities=args.eval_cities,
        mask_from_label=not args.no_mask_from_label,
        mask_label_threshold=args.mask_label_threshold,
        mask_connectivity=args.mask_connectivity,
        strict_mask=args.strict_mask,
        mask_min_instance_pixels=int(_arg_value(args, 'mask_min_instance_pixels', 0)),
        precomputed_disp_delta_root=_arg_value(args, 'precomputed_disp_delta_root', ''),
        precomputed_disp_subdir=_arg_value(args, 'precomputed_disp_subdir', 'disp_npy'),
        precomputed_delta_subdir=_arg_value(args, 'precomputed_delta_subdir', 'delta_relu_npy'),
        precomputed_ground_subdir=_arg_value(args, 'precomputed_ground_subdir', ''),
        return_precomputed_ground=bool(_arg_value(args, 'return_precomputed_ground', False)),
        require_precomputed_disp_delta=bool(_arg_value(args, 'require_precomputed_disp_delta', False)),
        instance_filter_manifest=(
            _arg_value(args, 'instance_filter_manifest', '')
            if bool(_arg_value(args, 'apply_instance_filter_val', True)) else ''
        ),
        drop_empty_after_instance_filter=bool(_arg_value(args, 'drop_empty_after_instance_filter', False)),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluate_height(model, val_loader, args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SD2H Height Evaluation Script")
    parser.add_argument('--config_yaml', default='', help="Resolved Hydra config yaml from the finetune run.")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--restore_ckpt', default='', help="Finetuned .pth checkpoint. Empty uses configured calibration parameters only.")
    parser.add_argument('--data_path', required=True, help="Dataset root directory.")
    parser.add_argument('--eval_cities', nargs='+', default=['ALL'], help="List of evaluation cities.")
    parser.add_argument('--output_directory', default="./height_results")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--valid_iters', type=int, default=12)
    parser.add_argument('--max_height_vis', type=float, default=100.0, help="Height clipping value for rendered maps.")
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--height_bins', nargs='+', type=float, default=[12, 24, 30, 50, 100, 150, 300])
    parser.add_argument('--height_bin_edges', nargs='+', type=float, default=[12, 24, 30, 50, 100, 150, 300])
    parser.add_argument('--height_output_mode', default='physical')
    parser.add_argument('--height_fusion_alpha', type=float, default=0.0)
    parser.add_argument('--height_residual_clip', type=float, default=120.0)
    parser.add_argument('--height_correction_clip', type=float, default=80.0)
    parser.add_argument('--height_correction_scale', type=float, default=1.0)
    parser.add_argument('--no_scatter', action='store_true', help="Disable scatter plot generation.")
    parser.add_argument('--no_bin_metrics', action='store_true', help="Disable stratified metric printing and CSV output.")
    parser.add_argument('--save_scatter_csv', action='store_true', help="Save instance-level GT/pred points used by the scatter plot.")
    parser.add_argument('--max_scatter_points', type=int, default=80000)
    parser.add_argument('--min_pred_height', type=float, default=None, help="Drop instances whose predicted mean height is below this threshold.")
    parser.add_argument('--no_mask_from_label', action='store_true', help="Disable automatic connected-component masks from height labels.")
    parser.add_argument('--mask_label_threshold', type=float, default=0.0)
    parser.add_argument('--mask_connectivity', type=int, default=2)
    parser.add_argument('--mask_min_instance_pixels', type=int, default=None)
    parser.add_argument('--strict_mask', action='store_true', help="Require mask files instead of using labels to generate masks.")
    parser.add_argument('--instance_filter_manifest', default=None, help="Override instance filter manifest used for evaluation.")
    parser.add_argument('--apply_instance_filter_val', action='store_true', default=None)
    parser.add_argument('--no_apply_instance_filter_val', dest='apply_instance_filter_val', action='store_false')
    parser.add_argument('--drop_empty_after_instance_filter', action='store_true', default=None)
    parser.add_argument('--precomputed_disp_delta_root', default=None)
    parser.add_argument('--precomputed_disp_subdir', default=None)
    parser.add_argument('--precomputed_delta_subdir', default=None)
    parser.add_argument('--precomputed_ground_subdir', default=None)
    parser.add_argument('--return_precomputed_ground', action='store_true', default=None)
    parser.add_argument('--require_precomputed_disp_delta', action='store_true', default=None)
    
    parser.add_argument('--encoder', type=str, default='vitb')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=4)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', default="reg")
    
    parser.add_argument('--min_disp', type=int, default=-320, help="min disp of geometry encoding volume")
    parser.add_argument('--max_disp', type=int, default=64, help="max disp of geometry encoding volume")
    
    parser.add_argument('--flight_height', type=float, default=505900.0)
    parser.add_argument('--baseline_P', type=float, default=311600.0)
    parser.add_argument('--buffer_method', default='fast_pool')
    parser.add_argument('--buffer_size', type=int, default=12)

    args = parser.parse_args()
    main(args)
