import os
import gc
import hydra
import torch
import math
from tqdm import tqdm
import torch.optim as optim
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, pad_across_processes
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
import torch.nn as nn
import numpy as np
from pathlib import Path

from core.sd2h_height_finetuner import SD2H_Height_Finetuner
import core.height_datasets as datasets


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _cfg_list(cfg, name, default):
    value = _cfg_value(cfg, name, default)
    return list(value) if value is not None else list(default)


def _bin_centers_from_edges(edges, num_bins, device, dtype):
    centers = []
    prev = 0.0
    for edge in edges:
        centers.append((prev + float(edge)) * 0.5)
        prev = float(edge)
    centers.append(prev + max(20.0, prev * 0.25))
    while len(centers) < num_bins:
        centers.append(centers[-1] + 50.0)
    return torch.tensor(centers[:num_bins], device=device, dtype=dtype)


def _height_eval_edges(cfg):
    bins = _cfg_list(cfg, 'height_eval_bins', _cfg_list(cfg, 'height_bin_edges', [12, 24, 30, 50, 100, 150, 300]))
    bins = sorted({float(x) for x in bins if float(x) > 0})
    if not bins or bins[0] != 0.0:
        bins = [0.0] + bins
    return bins


def _regression_stats_torch(y_true, y_pred):
    count = int(y_true.numel())
    if count == 0:
        return {'count': 0, 'mae': 0.0, 'rmse': 0.0, 'bias': 0.0, 'r2': 0.0}

    diff = y_pred - y_true
    sq = diff ** 2
    ss_res = float(sq.sum().item())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum().item())
    r2 = 0.0 if count < 2 else 1.0 - ss_res / (ss_tot + 1e-8)
    return {
        'count': count,
        'mae': float(diff.abs().mean().item()),
        'rmse': float(torch.sqrt(sq.mean()).item()),
        'bias': float(diff.mean().item()),
        'r2': float(r2),
    }


def _height_bin_rows_torch(y_true, y_pred, edges):
    rows = []
    for idx, lower in enumerate(edges):
        upper = edges[idx + 1] if idx + 1 < len(edges) else math.inf
        if math.isinf(upper):
            mask = y_true > lower
            bin_name = f">{lower:g}m"
        else:
            mask = (y_true > lower) & (y_true <= upper)
            bin_name = f"{lower:g}-{upper:g}m"
        rows.append({'bin': bin_name, **_regression_stats_torch(y_true[mask], y_pred[mask])})
    return rows


def _save_height_bin_csv(path, rows):
    import csv

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['bin', 'count', 'mae', 'rmse', 'bias'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ['bin', 'count', 'mae', 'rmse', 'bias']})


def _print_height_bin_rows(rows, title):
    print(f"\n[{title}]")
    print(f"{'Height Bin':<12}  {'Count':>8}  {'MAE(m)':>8}  {'RMSE(m)':>8}  {'Bias(m)':>9}")
    print("-" * 55)
    for row in rows:
        print(
            f"{row['bin']:<12}  {row['count']:>8d}  "
            f"{row['mae']:>8.3f}  {row['rmse']:>8.3f}  "
            f"{row['bias']:>+9.3f}"
        )


def instance_height_loss(preds_list, height_gt, instance_mask, epoch, total_epochs, lambda_decay=0.9, cfg=None):
    """
    Instance-level robust height loss with global and stratified terms.
    """
    progress = epoch / total_epochs 
    physics_weight = max(0.0, 2.0 * (1.0 - progress)) 
    l1_weight = 1.0 + progress
    huber_beta = float(_cfg_value(cfg, 'height_huber_beta', 4.0))
    high_threshold = float(_cfg_value(cfg, 'high_height_threshold', 30.0))
    high_under_weight = float(_cfg_value(cfg, 'high_under_weight', 2.5))
    high_bias_weight = float(_cfg_value(cfg, 'high_bias_weight', 0.25))
    low_height_threshold = float(_cfg_value(cfg, 'low_height_threshold', 12.0))
    low_over_weight = float(_cfg_value(cfg, 'low_over_weight', 0.0))
    low_under_weight = float(_cfg_value(cfg, 'low_under_weight', 0.0))
    low_bias_weight = float(_cfg_value(cfg, 'low_bias_weight', 0.0))
    global_loss_weight = float(_cfg_value(cfg, 'global_loss_weight', 0.45))
    stratified_loss_weight = float(_cfg_value(cfg, 'stratified_loss_weight', 0.55))
    aux_ce_weight = float(_cfg_value(cfg, 'aux_ce_weight', 0.20))
    aux_res_weight = float(_cfg_value(cfg, 'aux_res_weight', 0.50))
    aux_height_weight = float(_cfg_value(cfg, 'aux_height_weight', 0.0))
    aux_height_low_over_weight = float(_cfg_value(cfg, 'aux_height_low_over_weight', 0.0))
    aux_height_low_under_weight = float(_cfg_value(cfg, 'aux_height_low_under_weight', 0.0))
    aux_height_nonlow_under_weight = float(_cfg_value(cfg, 'aux_height_nonlow_under_weight', 0.0))
    aux_height_nonlow_threshold = float(_cfg_value(cfg, 'aux_height_nonlow_threshold', 12.0))
    aux_height_high_under_weight = float(_cfg_value(cfg, 'aux_height_high_under_weight', 0.0))
    aux_physical_weight = float(_cfg_value(cfg, 'aux_physical_weight', 0.0))
    aux_direct_weight = float(_cfg_value(cfg, 'aux_direct_weight', 0.0))
    correction_target_weight = float(_cfg_value(cfg, 'correction_target_weight', 0.0))
    correction_target_high_weight = float(_cfg_value(cfg, 'correction_target_high_weight', 0.0))
    correction_target_base_key = str(_cfg_value(cfg, 'correction_target_base_key', 'height_base'))
    correction_l1_weight = float(_cfg_value(cfg, 'correction_l1_weight', 0.0))
    correction_high_under_weight = float(_cfg_value(cfg, 'correction_high_under_weight', 0.0))
    bin_floor_weight = float(_cfg_value(cfg, 'height_bin_floor_weight', 0.0))
    bin_floor_start_bin = int(_cfg_value(cfg, 'height_bin_floor_start_bin', 2))
    bin_floor_margin = float(_cfg_value(cfg, 'height_bin_floor_margin', 0.0))
    high_floor_weight = float(_cfg_value(cfg, 'height_high_floor_weight', 0.0))
    high_floor_margin = float(_cfg_value(cfg, 'height_high_floor_margin', high_threshold))
    low_high_boost_weight = float(_cfg_value(cfg, 'height_low_high_boost_weight', 0.0))
    low_high_gate_weight = float(_cfg_value(cfg, 'height_low_high_gate_weight', 0.0))
    high_gate_target_weight = float(_cfg_value(cfg, 'height_high_gate_target_weight', 0.0))
    high_boost_target_weight = float(_cfg_value(cfg, 'height_high_boost_target_weight', 0.0))
    high_boost_target_scope = str(_cfg_value(cfg, 'height_high_boost_target_scope', 'high')).lower()
    high_boost_target_min_height = float(_cfg_value(cfg, 'height_high_boost_target_min_height', high_threshold))
    gate_residual_target_weight = float(_cfg_value(cfg, 'height_high_gate_residual_target_weight', 0.0))
    gate_residual_target_scope = str(_cfg_value(cfg, 'height_high_gate_residual_target_scope', high_boost_target_scope)).lower()
    gate_residual_target_min_height = float(_cfg_value(cfg, 'height_high_gate_residual_target_min_height', high_boost_target_min_height))
    gate_residual_target_beta = float(_cfg_value(cfg, 'height_high_gate_residual_target_beta', 0.10))
    signed_correction_target_weight = float(_cfg_value(cfg, 'height_signed_correction_target_weight', 0.0))
    low_signed_positive_weight = float(_cfg_value(cfg, 'height_low_signed_positive_weight', 0.0))
    base_anchor_weight = float(_cfg_value(cfg, 'height_base_anchor_weight', 0.0))
    low_base_anchor_weight = float(_cfg_value(cfg, 'height_low_base_anchor_weight', 0.0))
    physical_prior_under_weight = float(_cfg_value(cfg, 'height_physical_prior_under_weight', 0.0))
    physical_prior_min_height = float(_cfg_value(cfg, 'height_physical_prior_min_height', high_threshold))
    physical_prior_evidence_min = float(_cfg_value(cfg, 'height_physical_prior_evidence_min', 0.0))
    physical_prior_target_clip = float(_cfg_value(cfg, 'height_physical_prior_target_clip', 0.0))
    physical_prior_beta = float(_cfg_value(cfg, 'height_physical_prior_beta', huber_beta))
    loss_clip = float(_cfg_value(cfg, 'instance_loss_clip', 80.0))
    weak_soft_enabled = bool(_cfg_value(cfg, 'weak_label_soft_loss_enabled', False))
    weak_soft_abs_margin = float(_cfg_value(cfg, 'weak_label_soft_abs_margin', 0.0))
    weak_soft_rel_margin = float(_cfg_value(cfg, 'weak_label_soft_rel_margin', 0.0))
    weak_soft_margin_max = float(_cfg_value(cfg, 'weak_label_soft_margin_max', 0.0))
    weak_ordinal_weight = float(_cfg_value(cfg, 'weak_label_ordinal_weight', 0.0))
    weak_ordinal_min_gap = float(_cfg_value(cfg, 'weak_label_ordinal_min_gap', 6.0))
    weak_ordinal_margin = float(_cfg_value(cfg, 'weak_label_ordinal_margin', 0.5))
    weak_ordinal_max_pairs = int(_cfg_value(cfg, 'weak_label_ordinal_max_pairs', 4096))
    weak_ordinal_physical_weight = bool(_cfg_value(cfg, 'weak_label_ordinal_physical_weight', True))
    weak_ordinal_min_weight = float(_cfg_value(cfg, 'weak_label_ordinal_physical_min_weight', 0.25))

    is_scheme4 = isinstance(preds_list[0], dict)
    
    if is_scheme4:
        height_preds = [torch.nan_to_num(p['height'], nan=0.0, posinf=100.0, neginf=0.0) for p in preds_list]
    else:
        height_preds = [torch.nan_to_num(h, nan=0.0, posinf=100.0, neginf=0.0) for h in preds_list]
        
    n_predictions = len(height_preds)
    total_loss = 0.0
    B = height_gt.shape[0]
    
    batch_offsets = torch.arange(B, device=instance_mask.device, dtype=instance_mask.dtype).view(B, 1, 1, 1) * 100000
    global_instance_mask = torch.where(instance_mask > 0, instance_mask + batch_offsets, torch.zeros_like(instance_mask))
    inst_flat = global_instance_mask.view(-1).long()
    valid_mask = inst_flat > 0

    if not valid_mask.any():
        fake_loss = None
        for p_item in preds_list:
            if isinstance(p_item, dict):
                for key in (
                    'height',
                    'height_physical',
                    'height_base',
                    'height_calibrated',
                    'height_aux',
                    'height_direct',
                    'height_correction',
                    'height_signed_correction',
                    'height_high_boost',
                    'height_high_boost_raw',
                    'height_high_gate',
                    'height_aux_raise',
                    'height_pre_post_base_boost',
                    'height_post_base_gate',
                    'height_post_base_evidence_gate',
                    'height_post_base_high_boost',
                    'height_post_base_aux_gap',
                    'height_post_base_aux_gap_gate',
                    'height_pre_aux_floor',
                    'height_aux_floor_gate',
                    'height_aux_floor_evidence_gate',
                    'height_aux_floor_raise',
                    'height_aux_floor_gap',
                    'height_pre_physical_ratio_floor',
                    'height_physical_ratio_floor_target',
                    'height_physical_ratio_floor_gate',
                    'height_physical_ratio_floor_evidence_gate',
                    'height_physical_ratio_floor_raise',
                    'height_physical_ratio_floor_aux_gap',
                    'height_pre_learned_physical_floor',
                    'height_learned_physical_floor_raw',
                    'height_learned_physical_floor_fraction',
                    'height_learned_physical_floor_target',
                    'height_learned_physical_floor_gate',
                    'height_learned_physical_floor_evidence_gate',
                    'height_learned_physical_floor_raise',
                    'height_pre_high_residual',
                    'height_high_residual_raw',
                    'height_high_residual_gate_logit',
                    'height_high_residual_learned_gate',
                    'height_high_residual_gate',
                    'height_high_residual_evidence_gate',
                    'height_high_residual',
                    'height_gated_residual',
                    'height_residual_gate',
                    'disp_est',
                    'logits',
                    'residuals',
                ):
                    if key in p_item:
                        term = torch.nan_to_num(p_item[key], nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
                        fake_loss = term if fake_loss is None else fake_loss + term
            else:
                term = torch.nan_to_num(p_item, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
                fake_loss = term if fake_loss is None else fake_loss + term
        if fake_loss is None:
            fake_loss = torch.zeros((), dtype=height_gt.dtype, device=height_gt.device, requires_grad=True)
            
        return fake_loss, {
            'train/mae_sum': torch.tensor(0.0, device=height_gt.device),
            'train/mse_sum': torch.tensor(0.0, device=height_gt.device),
            'train/sum_y': torch.tensor(0.0, device=height_gt.device),
            'train/sum_y_sq': torch.tensor(0.0, device=height_gt.device),
            'train/inst_num': torch.tensor(0.0, device=height_gt.device)
        }

    height_gt_float = torch.nan_to_num(height_gt.float(), nan=0.0, posinf=320.0, neginf=0.0)
    gt_flat = height_gt_float.view(-1)[valid_mask]
    v_inst = inst_flat[valid_mask]
    unique_insts, inverse_indices = torch.unique(v_inst, return_inverse=True)
    num_instances = unique_insts.numel()

    counts = torch.zeros(num_instances, dtype=gt_flat.dtype, device=gt_flat.device).scatter_add_(0, inverse_indices, torch.ones_like(gt_flat))
    gt_inst_means = torch.zeros(num_instances, dtype=gt_flat.dtype, device=gt_flat.device).scatter_add_(0, inverse_indices, gt_flat) / counts

    # ==============================================================================
    # Build height-bin boundaries.
    # ==============================================================================
    bin_edges = _cfg_list(cfg, 'height_bin_edges', [12, 24, 30, 50, 100, 150, 300])
    boundaries = torch.tensor(bin_edges, device=height_gt.device, dtype=torch.float32)
    gt_bins = torch.bucketize(gt_inst_means, boundaries).long()

    def _scope_mask(scope, min_height=None, positive_target=None):
        if scope in ('all', 'all_positive', 'positive'):
            if positive_target is not None:
                return positive_target > 0
            return torch.ones_like(gt_inst_means, dtype=torch.bool)
        if scope in ('non_low', 'mid_high', 'not_low'):
            return gt_inst_means > low_height_threshold
        if scope in ('from_min', 'min_height', 'threshold'):
            threshold = high_threshold if min_height is None else float(min_height)
            return gt_inst_means >= threshold
        return high_mask
    
    # Smooth logarithmic adaptive weight.
    log_weight = torch.clamp(1.0 + 0.45 * torch.log1p(gt_inst_means), min=1.0, max=3.2)
    bin_weight_values = _cfg_list(cfg, 'height_bin_weights', [0.8, 1.0, 1.4, 2.4, 3.5, 4.5, 5.0, 5.0])
    while len(bin_weight_values) <= int(gt_bins.max().item()):
        bin_weight_values.append(bin_weight_values[-1])
    bin_weights = torch.tensor(bin_weight_values, device=height_gt.device, dtype=torch.float32)
    adaptive_weight = log_weight * bin_weights[gt_bins]

    for i, p_item in enumerate(preds_list):
        weight_decay = lambda_decay ** (n_predictions - 1 - i)
        
        h_pred = p_item['height'] if is_scheme4 else p_item
        h_pred = torch.nan_to_num(h_pred.float(), nan=0.0, posinf=320.0, neginf=0.0)
        pred_flat = h_pred.view(-1)[valid_mask]
        pred_inst_means = torch.zeros(num_instances, dtype=pred_flat.dtype, device=pred_flat.device).scatter_add_(0, inverse_indices, pred_flat) / counts

        # 1. Weak-label height loss. Noisy height products are treated as
        # interval supervision: errors inside the tolerance band carry no
        # gradient, while larger errors still follow a robust Huber penalty.
        if weak_soft_enabled:
            tolerance = weak_soft_abs_margin + weak_soft_rel_margin * gt_inst_means.abs()
            if weak_soft_margin_max > 0:
                tolerance = torch.clamp(tolerance, max=weak_soft_margin_max)
            signed_delta_soft = pred_inst_means - gt_inst_means
            soft_target = torch.sign(signed_delta_soft) * F.relu(signed_delta_soft.abs() - tolerance)
            smooth_l1_phy = F.smooth_l1_loss(
                soft_target,
                torch.zeros_like(soft_target),
                beta=huber_beta,
                reduction='none',
            )
        else:
            smooth_l1_phy = F.smooth_l1_loss(pred_inst_means, gt_inst_means, beta=huber_beta, reduction='none')
        
        # 2. Physical asymmetric penalty to discourage negative heights.
        physics_flat = F.relu(-pred_flat)
        physics_penalty = (torch.zeros(num_instances, dtype=physics_flat.dtype, device=physics_flat.device).scatter_add_(0, inverse_indices, physics_flat) / counts) * physics_weight

        iter_loss = (l1_weight * smooth_l1_phy) + physics_penalty
        signed_err = pred_inst_means - gt_inst_means
        high_mask = (gt_inst_means >= high_threshold)
        if high_mask.any() and high_under_weight > 0:
            under_err = F.relu(-signed_err)
            under_penalty = F.smooth_l1_loss(under_err, torch.zeros_like(under_err), beta=huber_beta, reduction='none')
            iter_loss = iter_loss + high_under_weight * under_penalty * high_mask.float()
        low_mask = gt_inst_means <= low_height_threshold
        if low_mask.any() and low_over_weight > 0:
            over_err = F.relu(signed_err)
            over_penalty = F.smooth_l1_loss(over_err, torch.zeros_like(over_err), beta=huber_beta, reduction='none')
            iter_loss = iter_loss + low_over_weight * over_penalty * low_mask.float()
        if low_mask.any() and low_under_weight > 0:
            low_under_err = F.relu(-signed_err)
            low_under_penalty = F.smooth_l1_loss(low_under_err, torch.zeros_like(low_under_err), beta=huber_beta, reduction='none')
            iter_loss = iter_loss + low_under_weight * low_under_penalty * low_mask.float()
        if bin_floor_weight > 0:
            lower_edges = torch.cat([
                torch.zeros(1, device=height_gt.device, dtype=torch.float32),
                boundaries,
            ])
            if lower_edges.numel() <= int(gt_bins.max().item()):
                pad_count = int(gt_bins.max().item()) - lower_edges.numel() + 1
                lower_edges = torch.cat([lower_edges, lower_edges[-1:].repeat(pad_count)])
            floor_targets = torch.clamp(lower_edges[gt_bins] - bin_floor_margin, min=0.0)
            floor_mask = gt_bins >= max(0, bin_floor_start_bin)
            if floor_mask.any():
                floor_under = F.relu(floor_targets - pred_inst_means)
                floor_penalty = F.smooth_l1_loss(floor_under, torch.zeros_like(floor_under), beta=huber_beta, reduction='none')
                iter_loss = iter_loss + bin_floor_weight * floor_penalty * floor_mask.float()
        if high_mask.any() and high_floor_weight > 0:
            high_floor_under = F.relu(high_floor_margin - pred_inst_means)
            high_floor_penalty = F.smooth_l1_loss(
                high_floor_under,
                torch.zeros_like(high_floor_under),
                beta=huber_beta,
                reduction='none',
            )
            iter_loss = iter_loss + high_floor_weight * high_floor_penalty * high_mask.float()

        consistency_weight = torch.ones_like(gt_inst_means)
        if bool(_cfg_value(cfg, 'physical_consistency_weighting', False)) and is_scheme4 and 'height_physical' in p_item:
            height_physical = torch.nan_to_num(p_item['height_physical'], nan=0.0, posinf=320.0, neginf=0.0).float()
            physical_flat = height_physical.view(-1)[valid_mask]
            physical_inst = torch.zeros(
                num_instances,
                dtype=physical_flat.dtype,
                device=physical_flat.device,
            ).scatter_add_(0, inverse_indices, physical_flat) / counts

            min_noise_weight = float(_cfg_value(cfg, 'physical_consistency_min_weight', 0.25))
            min_noise_weight = max(0.0, min(1.0, min_noise_weight))
            high_ratio_min = float(_cfg_value(cfg, 'physical_consistency_high_ratio_min', 0.35))
            high_ratio_width = max(float(_cfg_value(cfg, 'physical_consistency_high_ratio_width', 0.08)), 1.0e-6)
            high_gap = float(_cfg_value(cfg, 'physical_consistency_high_gap', 12.0))
            low_ratio_max = float(_cfg_value(cfg, 'physical_consistency_low_ratio_max', 3.5))
            low_ratio_width = max(float(_cfg_value(cfg, 'physical_consistency_low_ratio_width', 0.8)), 1.0e-6)
            low_gap = float(_cfg_value(cfg, 'physical_consistency_low_gap', 10.0))

            physical_ratio = physical_inst / torch.clamp(gt_inst_means, min=1.0e-3)
            high_inconsistent = high_mask & ((gt_inst_means - physical_inst) > high_gap)
            if high_inconsistent.any():
                high_conf = torch.sigmoid((physical_ratio - high_ratio_min) / high_ratio_width)
                high_conf = min_noise_weight + (1.0 - min_noise_weight) * high_conf
                consistency_weight = torch.where(high_inconsistent, high_conf, consistency_weight)

            low_inconsistent = low_mask & ((physical_inst - gt_inst_means) > low_gap)
            if low_inconsistent.any():
                low_conf = torch.sigmoid((low_ratio_max - physical_ratio) / low_ratio_width)
                low_conf = min_noise_weight + (1.0 - min_noise_weight) * low_conf
                consistency_weight = torch.where(low_inconsistent, torch.minimum(consistency_weight, low_conf), consistency_weight)

        # Decoupled height-bin supervision.
        if is_scheme4:
            logits = p_item.get('logits', None)
            if logits is not None:
                num_head_bins = logits.shape[1]
            else:
                num_head_bins = len(bin_edges) + 1
            gt_bins_aux = torch.clamp(gt_bins, max=num_head_bins - 1)

            if aux_ce_weight > 0 and logits is not None:
                logits_inst = torch.zeros(num_instances, num_head_bins, device=height_gt.device)
                for c in range(num_head_bins):
                    c_log = logits[:, c, :, :].contiguous().view(-1)[valid_mask]
                    logits_inst[:, c] = torch.zeros(num_instances, device=height_gt.device).scatter_add_(0, inverse_indices, c_log) / counts

                # 3. Classification loss for hard bin supervision.
                aux_ce_weights = _cfg_list(cfg, 'aux_ce_loss_weights', [])
                ce_weight = None
                if len(aux_ce_weights) > 0:
                    while len(aux_ce_weights) < num_head_bins:
                        aux_ce_weights.append(aux_ce_weights[-1])
                    ce_weight = torch.tensor(
                        aux_ce_weights[:num_head_bins],
                        device=height_gt.device,
                        dtype=logits_inst.dtype,
                    )
                ce_loss = F.cross_entropy(logits_inst, gt_bins_aux, weight=ce_weight, reduction='none')
                iter_loss = iter_loss + aux_ce_weight * ce_loss

            if aux_res_weight > 0 and 'residuals' in p_item:
                residuals = p_item['residuals']
                bin_centers = _bin_centers_from_edges(bin_edges, num_head_bins, height_gt.device, height_gt.dtype)
                gt_res = gt_inst_means - bin_centers[gt_bins_aux]
                res_inst = torch.zeros(num_instances, num_head_bins, device=height_gt.device)
                for c in range(num_head_bins):
                    c_res = residuals[:, c, :, :].contiguous().view(-1)[valid_mask]
                    res_inst[:, c] = torch.zeros(num_instances, device=height_gt.device).scatter_add_(0, inverse_indices, c_res) / counts
                pred_res_target = res_inst[torch.arange(num_instances, device=height_gt.device), gt_bins_aux]
                res_l1 = F.smooth_l1_loss(pred_res_target, gt_res, beta=huber_beta, reduction='none')
                iter_loss = iter_loss + aux_res_weight * res_l1
            if aux_physical_weight > 0 and 'height_physical' in p_item:
                height_physical = torch.nan_to_num(p_item['height_physical'], nan=0.0, posinf=100.0, neginf=0.0)
                pred_physical_flat = height_physical.view(-1)[valid_mask]
                pred_physical_inst = torch.zeros(num_instances, dtype=pred_physical_flat.dtype, device=pred_physical_flat.device).scatter_add_(0, inverse_indices, pred_physical_flat) / counts
                physical_l1 = F.smooth_l1_loss(pred_physical_inst, gt_inst_means, beta=huber_beta, reduction='none')
                iter_loss = iter_loss + aux_physical_weight * physical_l1

            if aux_direct_weight > 0 and 'height_direct' in p_item:
                height_direct = torch.nan_to_num(p_item['height_direct'], nan=0.0, posinf=100.0, neginf=0.0)
                pred_direct_flat = height_direct.view(-1)[valid_mask]
                pred_direct_inst = torch.zeros(num_instances, dtype=pred_direct_flat.dtype, device=pred_direct_flat.device).scatter_add_(0, inverse_indices, pred_direct_flat) / counts
                direct_l1 = F.smooth_l1_loss(pred_direct_inst, gt_inst_means, beta=huber_beta, reduction='none')
                iter_loss = iter_loss + aux_direct_weight * direct_l1

            if (
                aux_height_weight > 0
                or aux_height_low_over_weight > 0
                or aux_height_low_under_weight > 0
                or aux_height_nonlow_under_weight > 0
                or aux_height_high_under_weight > 0
            ) and 'height_aux' in p_item:
                height_aux = torch.nan_to_num(p_item['height_aux'], nan=0.0, posinf=320.0, neginf=0.0)
                pred_aux_flat = height_aux.view(-1)[valid_mask]
                pred_aux_inst = torch.zeros(
                    num_instances,
                    dtype=pred_aux_flat.dtype,
                    device=pred_aux_flat.device,
                ).scatter_add_(0, inverse_indices, pred_aux_flat) / counts
                aux_l1 = F.smooth_l1_loss(pred_aux_inst, gt_inst_means, beta=huber_beta, reduction='none')
                if aux_height_weight > 0:
                    iter_loss = iter_loss + aux_height_weight * aux_l1

                aux_err = pred_aux_inst - gt_inst_means
                if low_mask.any() and aux_height_low_over_weight > 0:
                    aux_low_over = F.relu(aux_err)
                    aux_low_over_loss = F.smooth_l1_loss(
                        aux_low_over,
                        torch.zeros_like(aux_low_over),
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = iter_loss + aux_height_low_over_weight * aux_low_over_loss * low_mask.float()
                if low_mask.any() and aux_height_low_under_weight > 0:
                    aux_low_under = F.relu(-aux_err)
                    aux_low_under_loss = F.smooth_l1_loss(
                        aux_low_under,
                        torch.zeros_like(aux_low_under),
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = iter_loss + aux_height_low_under_weight * aux_low_under_loss * low_mask.float()
                nonlow_aux_mask = gt_inst_means >= aux_height_nonlow_threshold
                if nonlow_aux_mask.any() and aux_height_nonlow_under_weight > 0:
                    aux_nonlow_under = F.relu(-aux_err)
                    aux_nonlow_under_loss = F.smooth_l1_loss(
                        aux_nonlow_under,
                        torch.zeros_like(aux_nonlow_under),
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = iter_loss + aux_height_nonlow_under_weight * aux_nonlow_under_loss * nonlow_aux_mask.float()
                if high_mask.any() and aux_height_high_under_weight > 0:
                    aux_high_under = F.relu(-aux_err)
                    aux_high_under_loss = F.smooth_l1_loss(
                        aux_high_under,
                        torch.zeros_like(aux_high_under),
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = iter_loss + aux_height_high_under_weight * aux_high_under_loss * high_mask.float()

            if (base_anchor_weight > 0 or low_base_anchor_weight > 0) and 'height_base' in p_item:
                height_base = torch.nan_to_num(p_item['height_base'], nan=0.0, posinf=100.0, neginf=0.0)
                pred_base_flat = height_base.view(-1)[valid_mask]
                pred_base_inst = torch.zeros(num_instances, dtype=pred_base_flat.dtype, device=pred_base_flat.device).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                base_anchor = F.smooth_l1_loss(pred_inst_means, pred_base_inst.detach(), beta=huber_beta, reduction='none')
                if base_anchor_weight > 0:
                    iter_loss = iter_loss + base_anchor_weight * base_anchor
                if low_base_anchor_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + low_base_anchor_weight * base_anchor * low_mask.float()

            if 'height_correction' in p_item and 'height_physical' in p_item:
                height_correction = torch.nan_to_num(p_item['height_correction'], nan=0.0, posinf=100.0, neginf=-100.0)
                correction_flat = height_correction.view(-1)[valid_mask]
                correction_inst = torch.zeros(num_instances, dtype=correction_flat.dtype, device=correction_flat.device).scatter_add_(0, inverse_indices, correction_flat) / counts

                if correction_l1_weight > 0:
                    iter_loss = iter_loss + correction_l1_weight * torch.abs(correction_inst)

                if correction_target_weight > 0 or (correction_target_high_weight > 0 and high_mask.any()):
                    base_key = correction_target_base_key
                    if base_key not in p_item:
                        base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst = torch.zeros(num_instances, dtype=pred_base_flat.dtype, device=pred_base_flat.device).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    refine_mode = str(_cfg_value(cfg, 'height_instance_refine_mode', 'additive')).lower()
                    pred_base_detached = pred_base_inst.detach()
                    if refine_mode in ('log_ratio', 'ratio', 'multiplicative', 'log_multiplicative'):
                        base_floor = float(_cfg_value(cfg, 'height_instance_ratio_base_floor', 1.0))
                        target_floor = float(_cfg_value(cfg, 'height_instance_ratio_target_floor', 0.25))
                        correction_target = torch.log(
                            torch.clamp(gt_inst_means, min=target_floor)
                            / torch.clamp(pred_base_detached, min=max(base_floor, 1.0e-4))
                        )
                        target_clip = float(_cfg_value(cfg, 'height_instance_log_ratio_target_clip', 0.0))
                        if target_clip > 0:
                            correction_target = torch.clamp(correction_target, min=-target_clip, max=target_clip)
                    elif refine_mode in ('relative', 'fractional', 'residual_ratio'):
                        base_floor = float(_cfg_value(cfg, 'height_instance_ratio_base_floor', 1.0))
                        correction_target = (
                            gt_inst_means / torch.clamp(pred_base_detached, min=max(base_floor, 1.0e-4))
                        ) - 1.0
                        target_clip = float(_cfg_value(cfg, 'height_instance_relative_target_clip', 0.0))
                        if target_clip > 0:
                            correction_target = torch.clamp(correction_target, min=-target_clip, max=target_clip)
                    else:
                        correction_target = gt_inst_means - pred_base_detached
                    correction_target_loss = F.smooth_l1_loss(correction_inst, correction_target, beta=huber_beta, reduction='none')
                    if correction_target_weight > 0:
                        iter_loss = iter_loss + correction_target_weight * correction_target_loss
                    if correction_target_high_weight > 0 and high_mask.any():
                        iter_loss = iter_loss + correction_target_high_weight * correction_target_loss * high_mask.float()

                if correction_high_under_weight > 0 and high_mask.any():
                    iter_loss = iter_loss + correction_high_under_weight * F.relu(-correction_inst) * high_mask.float()

            boost_target = None
            boost_target_mask = None
            pred_base_inst_for_boost = None
            if 'height_high_boost' in p_item and (
                low_high_boost_weight > 0
                or high_boost_target_weight > 0
                or gate_residual_target_weight > 0
            ):
                high_boost = torch.nan_to_num(p_item['height_high_boost'], nan=0.0, posinf=100.0, neginf=0.0)
                high_boost_flat = high_boost.view(-1)[valid_mask]
                high_boost_inst = torch.zeros(
                    num_instances,
                    dtype=high_boost_flat.dtype,
                    device=high_boost_flat.device,
                ).scatter_add_(0, inverse_indices, high_boost_flat) / counts
                if low_high_boost_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + low_high_boost_weight * high_boost_inst.abs() * low_mask.float()
                if high_boost_target_weight > 0 or gate_residual_target_weight > 0:
                    base_key = str(_cfg_value(cfg, 'height_high_boost_target_base_key', 'height_base'))
                    if base_key not in p_item:
                        base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst_for_boost = torch.zeros(
                        num_instances,
                        dtype=pred_base_flat.dtype,
                        device=pred_base_flat.device,
                    ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    boost_target = F.relu(gt_inst_means - pred_base_inst_for_boost.detach())
                    boost_target_mask = _scope_mask(
                        high_boost_target_scope,
                        high_boost_target_min_height,
                        positive_target=boost_target,
                    )
                    if high_boost_target_weight > 0 and boost_target_mask.any():
                        boost_target_loss = F.smooth_l1_loss(
                            high_boost_inst,
                            boost_target,
                            beta=huber_beta,
                            reduction='none',
                        )
                        iter_loss = iter_loss + high_boost_target_weight * boost_target_loss * boost_target_mask.float()

            raw_boost_target_weight = float(_cfg_value(cfg, 'height_high_boost_raw_target_weight', 0.0))
            if raw_boost_target_weight > 0 and 'height_high_boost_raw' in p_item:
                raw_boost = torch.nan_to_num(p_item['height_high_boost_raw'], nan=0.0, posinf=100.0, neginf=0.0)
                raw_boost_flat = raw_boost.view(-1)[valid_mask]
                raw_boost_inst = torch.zeros(
                    num_instances,
                    dtype=raw_boost_flat.dtype,
                    device=raw_boost_flat.device,
                ).scatter_add_(0, inverse_indices, raw_boost_flat) / counts
                base_key = str(_cfg_value(cfg, 'height_high_boost_target_base_key', 'height_base'))
                if base_key not in p_item:
                    base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                pred_base_flat = height_base.view(-1)[valid_mask]
                pred_base_inst = torch.zeros(
                    num_instances,
                    dtype=pred_base_flat.dtype,
                    device=pred_base_flat.device,
                ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                raw_boost_target = F.relu(gt_inst_means - pred_base_inst.detach())
                raw_boost_target_mask = _scope_mask(
                    high_boost_target_scope,
                    high_boost_target_min_height,
                    positive_target=raw_boost_target,
                )
                raw_boost_target_loss = F.smooth_l1_loss(
                    raw_boost_inst,
                    raw_boost_target,
                    beta=huber_beta,
                    reduction='none',
                )
                if raw_boost_target_mask.any():
                    iter_loss = iter_loss + raw_boost_target_weight * raw_boost_target_loss * raw_boost_target_mask.float()

            if 'height_high_gate' in p_item and (
                (low_high_gate_weight > 0 and low_mask.any())
                or (high_gate_target_weight > 0 and high_mask.any())
                or gate_residual_target_weight > 0
            ):
                high_gate = torch.nan_to_num(p_item['height_high_gate'], nan=0.0, posinf=1.0, neginf=0.0)
                high_gate_flat = high_gate.view(-1)[valid_mask]
                high_gate_inst = torch.zeros(
                    num_instances,
                    dtype=high_gate_flat.dtype,
                    device=high_gate_flat.device,
                ).scatter_add_(0, inverse_indices, high_gate_flat) / counts
                if low_high_gate_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + low_high_gate_weight * high_gate_inst * low_mask.float()
                if high_gate_target_weight > 0 and high_mask.any():
                    high_gate_target = torch.ones_like(high_gate_inst)
                    high_gate_loss = F.smooth_l1_loss(high_gate_inst, high_gate_target, beta=0.2, reduction='none')
                    iter_loss = iter_loss + high_gate_target_weight * high_gate_loss * high_mask.float()
                if gate_residual_target_weight > 0 and 'height_high_boost_raw' in p_item:
                    raw_boost = torch.nan_to_num(p_item['height_high_boost_raw'], nan=0.0, posinf=100.0, neginf=0.0)
                    raw_boost_flat = raw_boost.view(-1)[valid_mask]
                    raw_boost_inst = torch.zeros(
                        num_instances,
                        dtype=raw_boost_flat.dtype,
                        device=raw_boost_flat.device,
                    ).scatter_add_(0, inverse_indices, raw_boost_flat) / counts
                    if boost_target is None:
                        base_key = str(_cfg_value(cfg, 'height_high_boost_target_base_key', 'height_base'))
                        if base_key not in p_item:
                            base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                        height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                        pred_base_flat = height_base.view(-1)[valid_mask]
                        pred_base_inst_for_boost = torch.zeros(
                            num_instances,
                            dtype=pred_base_flat.dtype,
                            device=pred_base_flat.device,
                        ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                        boost_target = F.relu(gt_inst_means - pred_base_inst_for_boost.detach())
                    gate_target = torch.clamp(boost_target / raw_boost_inst.detach().clamp_min(1.0), 0.0, 1.0)
                    gate_target_mask = _scope_mask(
                        gate_residual_target_scope,
                        gate_residual_target_min_height,
                        positive_target=boost_target,
                    )
                    if gate_target_mask.any():
                        gate_target_loss = F.smooth_l1_loss(
                            high_gate_inst,
                            gate_target,
                            beta=max(gate_residual_target_beta, 1.0e-3),
                            reduction='none',
                        )
                        iter_loss = iter_loss + gate_residual_target_weight * gate_target_loss * gate_target_mask.float()

            post_boost_target_weight = float(_cfg_value(cfg, 'height_post_base_boost_target_weight', 0.0))
            post_boost_low_weight = float(_cfg_value(cfg, 'height_post_base_low_boost_weight', 0.0))
            post_boost_non_high_weight = float(_cfg_value(cfg, 'height_post_base_non_high_boost_weight', 0.0))
            if 'height_post_base_high_boost' in p_item and (
                post_boost_target_weight > 0
                or post_boost_low_weight > 0
                or post_boost_non_high_weight > 0
            ):
                post_boost = torch.nan_to_num(
                    p_item['height_post_base_high_boost'],
                    nan=0.0,
                    posinf=100.0,
                    neginf=0.0,
                )
                post_boost_flat = post_boost.view(-1)[valid_mask]
                post_boost_inst = torch.zeros(
                    num_instances,
                    dtype=post_boost_flat.dtype,
                    device=post_boost_flat.device,
                ).scatter_add_(0, inverse_indices, post_boost_flat) / counts
                if post_boost_low_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + post_boost_low_weight * post_boost_inst.abs() * low_mask.float()
                if post_boost_non_high_weight > 0:
                    non_high_mask = gt_inst_means < high_threshold
                    if non_high_mask.any():
                        iter_loss = iter_loss + post_boost_non_high_weight * post_boost_inst.abs() * non_high_mask.float()
                if post_boost_target_weight > 0:
                    base_key = str(_cfg_value(cfg, 'height_post_base_boost_target_base_key', 'height_pre_post_base_boost'))
                    if base_key not in p_item:
                        base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst = torch.zeros(
                        num_instances,
                        dtype=pred_base_flat.dtype,
                        device=pred_base_flat.device,
                    ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    post_boost_target = F.relu(gt_inst_means - pred_base_inst.detach())
                    post_boost_scope = str(_cfg_value(cfg, 'height_post_base_boost_target_scope', high_boost_target_scope)).lower()
                    post_boost_min_height = float(
                        _cfg_value(cfg, 'height_post_base_boost_target_min_height', high_boost_target_min_height)
                    )
                    post_boost_target_mask = _scope_mask(
                        post_boost_scope,
                        post_boost_min_height,
                        positive_target=post_boost_target,
                    )
                    if post_boost_target_mask.any():
                        post_boost_target_loss = F.smooth_l1_loss(
                            post_boost_inst,
                            post_boost_target,
                            beta=huber_beta,
                            reduction='none',
                        )
                        iter_loss = (
                            iter_loss
                            + post_boost_target_weight
                            * post_boost_target_loss
                            * post_boost_target_mask.float()
                        )

            physical_floor_target_weight = float(_cfg_value(cfg, 'height_physical_ratio_floor_target_weight', 0.0))
            physical_floor_low_weight = float(_cfg_value(cfg, 'height_physical_ratio_floor_low_raise_weight', 0.0))
            physical_floor_non_high_weight = float(_cfg_value(cfg, 'height_physical_ratio_floor_non_high_raise_weight', 0.0))
            if 'height_physical_ratio_floor_raise' in p_item and (
                physical_floor_target_weight > 0
                or physical_floor_low_weight > 0
                or physical_floor_non_high_weight > 0
            ):
                floor_raise = torch.nan_to_num(
                    p_item['height_physical_ratio_floor_raise'],
                    nan=0.0,
                    posinf=100.0,
                    neginf=0.0,
                )
                floor_raise_flat = floor_raise.view(-1)[valid_mask]
                floor_raise_inst = torch.zeros(
                    num_instances,
                    dtype=floor_raise_flat.dtype,
                    device=floor_raise_flat.device,
                ).scatter_add_(0, inverse_indices, floor_raise_flat) / counts
                if physical_floor_low_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + physical_floor_low_weight * floor_raise_inst.abs() * low_mask.float()
                if physical_floor_non_high_weight > 0:
                    non_high_mask = gt_inst_means < high_threshold
                    if non_high_mask.any():
                        iter_loss = (
                            iter_loss
                            + physical_floor_non_high_weight
                            * floor_raise_inst.abs()
                            * non_high_mask.float()
                        )
                if physical_floor_target_weight > 0:
                    base_key = str(_cfg_value(cfg, 'height_physical_ratio_floor_target_base_key', 'height_pre_physical_ratio_floor'))
                    if base_key not in p_item:
                        base_key = 'height_pre_post_base_boost' if 'height_pre_post_base_boost' in p_item else 'height_base'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst = torch.zeros(
                        num_instances,
                        dtype=pred_base_flat.dtype,
                        device=pred_base_flat.device,
                    ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    floor_target = F.relu(gt_inst_means - pred_base_inst.detach())
                    floor_target_clip = float(_cfg_value(cfg, 'height_physical_ratio_floor_raise_target_clip', 0.0))
                    if floor_target_clip > 0:
                        floor_target = torch.clamp(floor_target, max=floor_target_clip)
                    floor_scope = str(_cfg_value(cfg, 'height_physical_ratio_floor_target_scope', 'from_min')).lower()
                    floor_min_height = float(_cfg_value(cfg, 'height_physical_ratio_floor_target_min_height', high_threshold))
                    floor_target_mask = _scope_mask(
                        floor_scope,
                        floor_min_height,
                        positive_target=floor_target,
                    )
                    if floor_target_mask.any():
                        floor_target_loss = F.smooth_l1_loss(
                            floor_raise_inst,
                            floor_target,
                            beta=huber_beta,
                            reduction='none',
                        )
                        iter_loss = (
                            iter_loss
                            + physical_floor_target_weight
                            * floor_target_loss
                            * floor_target_mask.float()
                        )

            learned_floor_target_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_target_weight', 0.0))
            learned_floor_low_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_low_raise_weight', 0.0))
            learned_floor_non_high_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_non_high_raise_weight', 0.0))
            if 'height_learned_physical_floor_raise' in p_item and (
                learned_floor_target_weight > 0
                or learned_floor_low_weight > 0
                or learned_floor_non_high_weight > 0
            ):
                floor_raise = torch.nan_to_num(
                    p_item['height_learned_physical_floor_raise'],
                    nan=0.0,
                    posinf=100.0,
                    neginf=0.0,
                )
                floor_raise_flat = floor_raise.view(-1)[valid_mask]
                floor_raise_inst = torch.zeros(
                    num_instances,
                    dtype=floor_raise_flat.dtype,
                    device=floor_raise_flat.device,
                ).scatter_add_(0, inverse_indices, floor_raise_flat) / counts
                if learned_floor_low_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + learned_floor_low_weight * floor_raise_inst.abs() * low_mask.float()
                if learned_floor_non_high_weight > 0:
                    non_high_scope = str(_cfg_value(cfg, 'height_learned_physical_floor_non_high_raise_scope', 'below_high')).lower()
                    if non_high_scope in ('low', 'low_only', 'below_low'):
                        non_high_mask = gt_inst_means < low_height_threshold
                    elif non_high_scope in ('mid_low', 'below_mid', 'below_24'):
                        non_high_mask = gt_inst_means < 24.0
                    elif non_high_scope in ('none', 'disabled', 'off'):
                        non_high_mask = torch.zeros_like(gt_inst_means, dtype=torch.bool)
                    else:
                        non_high_mask = gt_inst_means < high_threshold
                    if non_high_mask.any():
                        iter_loss = (
                            iter_loss
                            + learned_floor_non_high_weight
                            * floor_raise_inst.abs()
                            * non_high_mask.float()
                        )
                if learned_floor_target_weight > 0:
                    base_key = str(_cfg_value(cfg, 'height_learned_physical_floor_target_base_key', 'height_pre_learned_physical_floor'))
                    if base_key not in p_item:
                        base_key = 'height_base' if 'height_base' in p_item else 'height'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst = torch.zeros(
                        num_instances,
                        dtype=pred_base_flat.dtype,
                        device=pred_base_flat.device,
                    ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    floor_target = F.relu(gt_inst_means - pred_base_inst.detach())
                    floor_target_clip = float(_cfg_value(cfg, 'height_learned_physical_floor_raise_target_clip', 0.0))
                    if floor_target_clip > 0:
                        floor_target = torch.clamp(floor_target, max=floor_target_clip)
                    floor_scope = str(_cfg_value(cfg, 'height_learned_physical_floor_target_scope', 'from_min')).lower()
                    floor_min_height = float(_cfg_value(cfg, 'height_learned_physical_floor_target_min_height', high_threshold))
                    floor_target_mask = _scope_mask(
                        floor_scope,
                        floor_min_height,
                        positive_target=floor_target,
                    )
                    if floor_target_mask.any():
                        floor_target_loss = F.smooth_l1_loss(
                            floor_raise_inst,
                            floor_target,
                            beta=huber_beta,
                            reduction='none',
                        )
                        iter_loss = (
                            iter_loss
                            + learned_floor_target_weight
                            * floor_target_loss
                            * floor_target_mask.float()
                        )

            learned_fraction_target_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_target_weight', 0.0))
            if learned_fraction_target_weight > 0 and 'height_learned_physical_floor_fraction' in p_item:
                fraction_map = torch.nan_to_num(
                    p_item['height_learned_physical_floor_fraction'],
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                fraction_flat = fraction_map.view(-1)[valid_mask]
                fraction_inst = torch.zeros(
                    num_instances,
                    dtype=fraction_flat.dtype,
                    device=fraction_flat.device,
                ).scatter_add_(0, inverse_indices, fraction_flat) / counts
                frac_min = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_min', 0.0))
                frac_max = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_max', 1.0))
                denom = max(abs(frac_max - frac_min), 1.0e-6)
                fraction_prob = torch.clamp((fraction_inst - frac_min) / denom, 1.0e-4, 1.0 - 1.0e-4)

                pos_height = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_pos_height', high_threshold))
                neg_height = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_neg_height', 24.0))
                pos_target = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_pos_target', 1.0))
                neg_target = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_neg_target', 0.0))
                pos_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_pos_weight', 6.0))
                neg_weight = float(_cfg_value(cfg, 'height_learned_physical_floor_fraction_neg_weight', 1.0))
                pos_mask = gt_inst_means >= pos_height
                neg_mask = gt_inst_means < neg_height
                fraction_target_mask = pos_mask | neg_mask
                if fraction_target_mask.any():
                    target = torch.where(
                        pos_mask,
                        torch.full_like(fraction_prob, pos_target),
                        torch.full_like(fraction_prob, neg_target),
                    )
                    target = torch.clamp(target, 0.0, 1.0)
                    sample_weight = torch.where(
                        pos_mask,
                        torch.full_like(fraction_prob, pos_weight),
                        torch.full_like(fraction_prob, neg_weight),
                    )
                    loss_type = str(_cfg_value(cfg, 'height_learned_physical_floor_fraction_target_loss', 'bce')).lower()
                    if loss_type in ('l1', 'smooth_l1', 'huber'):
                        fraction_target_loss = F.smooth_l1_loss(
                            fraction_prob,
                            target,
                            beta=0.2,
                            reduction='none',
                        )
                    else:
                        fraction_target_loss = F.binary_cross_entropy(
                            fraction_prob,
                            target,
                            reduction='none',
                        )
                    iter_loss = (
                        iter_loss
                        + learned_fraction_target_weight
                        * fraction_target_loss
                        * sample_weight
                        * fraction_target_mask.float()
                    )

            high_res_target_weight = float(_cfg_value(cfg, 'height_high_residual_target_weight', 0.0))
            high_res_raw_target_weight = float(_cfg_value(cfg, 'height_high_residual_raw_target_weight', 0.0))
            high_res_low_weight = float(_cfg_value(cfg, 'height_high_residual_low_weight', 0.0))
            high_res_non_high_weight = float(_cfg_value(cfg, 'height_high_residual_non_high_weight', 0.0))
            high_res_gate_target_weight = float(_cfg_value(cfg, 'height_high_residual_gate_target_weight', 0.0))
            high_res_low_gate_weight = float(_cfg_value(cfg, 'height_high_residual_low_gate_weight', 0.0))
            high_res_non_high_gate_weight = float(_cfg_value(cfg, 'height_high_residual_non_high_gate_weight', 0.0))
            high_res_mid1224_weight = float(_cfg_value(cfg, 'height_high_residual_mid1224_weight', 0.0))
            high_res_mid2430_over_weight = float(_cfg_value(cfg, 'height_high_residual_mid2430_over_weight', 0.0))
            high_res_mid1224_gate_weight = float(_cfg_value(cfg, 'height_high_residual_mid1224_gate_weight', 0.0))
            high_res_mid2430_gate_weight = float(_cfg_value(cfg, 'height_high_residual_mid2430_gate_weight', 0.0))
            high_res_mid2430_res_cap = float(_cfg_value(cfg, 'height_high_residual_mid2430_res_cap', 0.0))
            high_res_mid2430_gate_cap = float(_cfg_value(cfg, 'height_high_residual_mid2430_gate_cap', 0.0))
            high_res_learned_gate_ce_weight = float(_cfg_value(cfg, 'height_high_residual_learned_gate_ce_weight', 0.0))
            if 'height_high_residual' in p_item and (
                high_res_target_weight > 0
                or high_res_raw_target_weight > 0
                or high_res_low_weight > 0
                or high_res_non_high_weight > 0
                or high_res_gate_target_weight > 0
                or high_res_low_gate_weight > 0
                or high_res_non_high_gate_weight > 0
                or high_res_mid1224_weight > 0
                or high_res_mid2430_over_weight > 0
                or high_res_mid1224_gate_weight > 0
                or high_res_mid2430_gate_weight > 0
                or high_res_learned_gate_ce_weight > 0
            ):
                high_residual = torch.nan_to_num(
                    p_item['height_high_residual'],
                    nan=0.0,
                    posinf=100.0,
                    neginf=0.0,
                )
                high_res_flat = high_residual.view(-1)[valid_mask]
                high_res_inst = torch.zeros(
                    num_instances,
                    dtype=high_res_flat.dtype,
                    device=high_res_flat.device,
                ).scatter_add_(0, inverse_indices, high_res_flat) / counts
                if high_res_low_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + high_res_low_weight * high_res_inst.abs() * low_mask.float()
                if high_res_non_high_weight > 0:
                    non_high_mask = gt_inst_means < high_threshold
                    if non_high_mask.any():
                        iter_loss = (
                            iter_loss
                            + high_res_non_high_weight
                            * high_res_inst.abs()
                            * non_high_mask.float()
                        )
                mid1224_mask = (gt_inst_means > low_height_threshold) & (gt_inst_means < 24.0)
                mid2430_mask = (gt_inst_means >= 24.0) & (gt_inst_means < high_threshold)
                if high_res_mid1224_weight > 0 and mid1224_mask.any():
                    iter_loss = (
                        iter_loss
                        + high_res_mid1224_weight
                        * high_res_inst.abs()
                        * mid1224_mask.float()
                    )
                if high_res_mid2430_over_weight > 0 and mid2430_mask.any():
                    if high_res_mid2430_res_cap > 0:
                        mid2430_over = F.relu(high_res_inst - high_res_mid2430_res_cap)
                    else:
                        mid2430_over = high_res_inst.abs()
                    mid2430_over_loss = F.smooth_l1_loss(
                        mid2430_over,
                        torch.zeros_like(mid2430_over),
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = (
                        iter_loss
                        + high_res_mid2430_over_weight
                        * mid2430_over_loss
                        * mid2430_mask.float()
                    )

                base_key = str(_cfg_value(cfg, 'height_high_residual_target_base_key', 'height_pre_high_residual'))
                if base_key not in p_item:
                    base_key = 'height_base' if 'height_base' in p_item else 'height'
                height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                pred_base_flat = height_base.view(-1)[valid_mask]
                pred_base_inst = torch.zeros(
                    num_instances,
                    dtype=pred_base_flat.dtype,
                    device=pred_base_flat.device,
                ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                high_res_target = F.relu(gt_inst_means - pred_base_inst.detach())
                target_clip = float(_cfg_value(cfg, 'height_high_residual_target_clip', 0.0))
                if target_clip > 0:
                    high_res_target = torch.clamp(high_res_target, max=target_clip)
                target_scope = str(_cfg_value(cfg, 'height_high_residual_target_scope', 'from_min')).lower()
                target_min_height = float(_cfg_value(cfg, 'height_high_residual_target_min_height', high_threshold))
                high_res_target_mask = _scope_mask(
                    target_scope,
                    target_min_height,
                    positive_target=high_res_target,
                )
                high_res_evidence_inst = None
                if 'height_high_residual_evidence_gate' in p_item:
                    high_res_evidence = torch.nan_to_num(
                        p_item['height_high_residual_evidence_gate'],
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    evidence_flat = high_res_evidence.view(-1)[valid_mask]
                    high_res_evidence_inst = torch.zeros(
                        num_instances,
                        dtype=evidence_flat.dtype,
                        device=evidence_flat.device,
                    ).scatter_add_(0, inverse_indices, evidence_flat) / counts
                    min_target_evidence = float(_cfg_value(cfg, 'height_high_residual_target_min_evidence', 0.0))
                    if min_target_evidence > 0.0:
                        high_res_target_mask = high_res_target_mask & (
                            high_res_evidence_inst >= min_target_evidence
                        )
                high_res_target_weight_inst = torch.ones_like(high_res_target)
                if (
                    bool(_cfg_value(cfg, 'height_high_residual_target_weight_by_evidence', False))
                    and high_res_evidence_inst is not None
                ):
                    ev_power = max(
                        float(_cfg_value(cfg, 'height_high_residual_target_evidence_power', 1.0)),
                        1.0e-3,
                    )
                    ev_floor = max(0.0, float(_cfg_value(cfg, 'height_high_residual_target_evidence_floor', 0.0)))
                    high_res_target_weight_inst = torch.clamp(high_res_evidence_inst, min=ev_floor, max=1.0)
                    if abs(ev_power - 1.0) > 1.0e-6:
                        high_res_target_weight_inst = torch.pow(high_res_target_weight_inst, ev_power)
                if high_res_target_weight > 0 and high_res_target_mask.any():
                    high_res_loss = F.smooth_l1_loss(
                        high_res_inst,
                        high_res_target,
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = (
                        iter_loss
                        + high_res_target_weight
                        * high_res_loss
                        * high_res_target_weight_inst
                        * high_res_target_mask.float()
                    )

                if (
                    high_res_raw_target_weight > 0
                    and high_res_target_mask.any()
                    and 'height_high_residual_raw' in p_item
                ):
                    high_res_raw = torch.nan_to_num(
                        p_item['height_high_residual_raw'],
                        nan=0.0,
                        posinf=160.0,
                        neginf=0.0,
                    )
                    high_res_raw_flat = high_res_raw.view(-1)[valid_mask]
                    high_res_raw_inst = torch.zeros(
                        num_instances,
                        dtype=high_res_raw_flat.dtype,
                        device=high_res_raw_flat.device,
                    ).scatter_add_(0, inverse_indices, high_res_raw_flat) / counts

                    raw_target = high_res_target.detach()
                    raw_scale = float(_cfg_value(cfg, 'height_high_residual_raw_target_scale', 1.0))
                    if raw_scale != 1.0:
                        raw_target = raw_target * raw_scale
                    high_residual_mode = str(_cfg_value(cfg, 'height_high_residual_mode', 'additive')).lower()
                    if (
                        high_residual_mode
                        in (
                            'physical_ratio',
                            'phys_ratio',
                            'physical_scaled',
                            'delta_physical_ratio',
                            'differential_disparity_ratio',
                        )
                        and 'height_high_residual_basis' in p_item
                    ):
                        basis_map = torch.nan_to_num(
                            p_item['height_high_residual_basis'],
                            nan=0.0,
                            posinf=160.0,
                            neginf=0.0,
                        )
                        basis_flat = basis_map.view(-1)[valid_mask]
                        basis_inst = torch.zeros(
                            num_instances,
                            dtype=basis_flat.dtype,
                            device=basis_flat.device,
                        ).scatter_add_(0, inverse_indices, basis_flat) / counts
                        raw_gate = torch.ones_like(basis_inst)
                        if high_res_evidence_inst is not None:
                            gate_floor = max(
                                float(_cfg_value(cfg, 'height_high_residual_raw_target_gate_floor', 0.20)),
                                1.0e-3,
                            )
                            raw_gate = torch.clamp(high_res_evidence_inst.detach(), min=gate_floor)
                        raw_basis_floor = max(
                            float(_cfg_value(cfg, 'height_high_residual_raw_target_basis_floor', 3.0)),
                            1.0e-3,
                        )
                        raw_target = raw_target / torch.clamp(
                            basis_inst.detach() * raw_gate,
                            min=raw_basis_floor,
                        )
                        raw_ratio_clip = float(
                            _cfg_value(cfg, 'height_high_residual_raw_target_ratio_clip', 4.0)
                        )
                        if raw_ratio_clip > 0:
                            raw_target = torch.clamp(raw_target, max=raw_ratio_clip)
                    elif (
                        bool(_cfg_value(cfg, 'height_high_residual_raw_target_compensate_gate', False))
                        and high_res_evidence_inst is not None
                    ):
                        gate_floor = max(
                            float(_cfg_value(cfg, 'height_high_residual_raw_target_gate_floor', 0.20)),
                            1.0e-3,
                        )
                        raw_target = raw_target / torch.clamp(high_res_evidence_inst.detach(), min=gate_floor)
                    raw_target_clip = float(
                        _cfg_value(cfg, 'height_high_residual_raw_target_clip', target_clip)
                    )
                    if raw_target_clip > 0:
                        raw_target = torch.clamp(raw_target, max=raw_target_clip)
                    raw_loss = F.smooth_l1_loss(
                        high_res_raw_inst,
                        raw_target,
                        beta=huber_beta,
                        reduction='none',
                    )
                    iter_loss = (
                        iter_loss
                        + high_res_raw_target_weight
                        * raw_loss
                        * high_res_target_weight_inst
                        * high_res_target_mask.float()
                    )

                if (
                    (
                        high_res_gate_target_weight > 0
                        or high_res_low_gate_weight > 0
                        or high_res_non_high_gate_weight > 0
                        or high_res_mid1224_gate_weight > 0
                        or high_res_mid2430_gate_weight > 0
                    )
                    and 'height_high_residual_gate' in p_item
                ):
                    high_res_gate = torch.nan_to_num(
                        p_item['height_high_residual_gate'],
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    high_res_gate_flat = high_res_gate.view(-1)[valid_mask]
                    high_res_gate_inst = torch.zeros(
                        num_instances,
                        dtype=high_res_gate_flat.dtype,
                        device=high_res_gate_flat.device,
                    ).scatter_add_(0, inverse_indices, high_res_gate_flat) / counts
                    gate_target = (high_res_target > 0).float()
                    gate_target_mask = _scope_mask(
                        target_scope,
                        target_min_height,
                        positive_target=high_res_target,
                    )
                    min_target_evidence = float(_cfg_value(cfg, 'height_high_residual_target_min_evidence', 0.0))
                    if (
                        min_target_evidence > 0.0
                        and high_res_evidence_inst is not None
                    ):
                        gate_target_mask = gate_target_mask & (
                            high_res_evidence_inst >= min_target_evidence
                        )
                    if gate_target_mask.any():
                        gate_loss = F.smooth_l1_loss(
                            high_res_gate_inst,
                            gate_target,
                            beta=0.2,
                            reduction='none',
                        )
                        iter_loss = (
                            iter_loss
                            + high_res_gate_target_weight
                            * gate_loss
                            * gate_target_mask.float()
                        )
                    gate_zero_loss = F.smooth_l1_loss(
                        high_res_gate_inst,
                        torch.zeros_like(high_res_gate_inst),
                        beta=0.08,
                        reduction='none',
                    )
                    if high_res_low_gate_weight > 0 and low_mask.any():
                        iter_loss = (
                            iter_loss
                            + high_res_low_gate_weight
                            * gate_zero_loss
                            * low_mask.float()
                        )
                    if high_res_non_high_gate_weight > 0:
                        non_high_mask = gt_inst_means < high_threshold
                        if non_high_mask.any():
                            iter_loss = (
                                iter_loss
                                + high_res_non_high_gate_weight
                                * gate_zero_loss
                                * non_high_mask.float()
                            )
                    if high_res_mid1224_gate_weight > 0 and mid1224_mask.any():
                        iter_loss = (
                            iter_loss
                            + high_res_mid1224_gate_weight
                            * gate_zero_loss
                            * mid1224_mask.float()
                        )
                    if high_res_mid2430_gate_weight > 0 and mid2430_mask.any():
                        if high_res_mid2430_gate_cap > 0:
                            mid2430_gate_over = F.relu(high_res_gate_inst - high_res_mid2430_gate_cap)
                            mid2430_gate_loss = F.smooth_l1_loss(
                                mid2430_gate_over,
                                torch.zeros_like(mid2430_gate_over),
                                beta=0.08,
                                reduction='none',
                            )
                        else:
                            mid2430_gate_loss = gate_zero_loss
                        iter_loss = (
                            iter_loss
                            + high_res_mid2430_gate_weight
                            * mid2430_gate_loss
                            * mid2430_mask.float()
                        )

                if (
                    high_res_learned_gate_ce_weight > 0
                    and 'height_high_residual_learned_gate' in p_item
                ):
                    learned_gate = torch.nan_to_num(
                        p_item['height_high_residual_learned_gate'],
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    learned_gate_flat = learned_gate.view(-1)[valid_mask]
                    learned_gate_inst = torch.zeros(
                        num_instances,
                        dtype=learned_gate_flat.dtype,
                        device=learned_gate_flat.device,
                    ).scatter_add_(0, inverse_indices, learned_gate_flat) / counts
                    learned_gate_prob = torch.clamp(learned_gate_inst, 1.0e-4, 1.0 - 1.0e-4)
                    pos_height = float(_cfg_value(cfg, 'height_high_residual_learned_gate_pos_height', high_threshold))
                    neg_height = float(_cfg_value(cfg, 'height_high_residual_learned_gate_neg_height', 24.0))
                    pos_mask = gt_inst_means >= pos_height
                    neg_mask = gt_inst_means < neg_height
                    learned_gate_mask = pos_mask | neg_mask
                    if learned_gate_mask.any():
                        gate_target = torch.where(
                            pos_mask,
                            torch.ones_like(learned_gate_prob),
                            torch.zeros_like(learned_gate_prob),
                        )
                        gate_sample_weight = torch.where(
                            pos_mask,
                            torch.full_like(
                                learned_gate_prob,
                                float(_cfg_value(cfg, 'height_high_residual_learned_gate_pos_weight', 8.0)),
                            ),
                            torch.full_like(
                                learned_gate_prob,
                                float(_cfg_value(cfg, 'height_high_residual_learned_gate_neg_weight', 1.0)),
                            ),
                        )
                        gate_ce = F.binary_cross_entropy(
                            learned_gate_prob,
                            gate_target,
                            reduction='none',
                        )
                        iter_loss = (
                            iter_loss
                            + high_res_learned_gate_ce_weight
                            * gate_ce
                            * gate_sample_weight
                            * learned_gate_mask.float()
                        )

            if (
                physical_prior_under_weight > 0
                and 'height_physical_ratio_floor_target' in p_item
                and 'height_physical_ratio_floor_evidence_gate' in p_item
            ):
                floor_target_map = torch.nan_to_num(
                    p_item['height_physical_ratio_floor_target'],
                    nan=0.0,
                    posinf=320.0,
                    neginf=0.0,
                )
                floor_evidence_map = torch.nan_to_num(
                    p_item['height_physical_ratio_floor_evidence_gate'],
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                floor_target_flat = floor_target_map.view(-1)[valid_mask]
                floor_evidence_flat = floor_evidence_map.view(-1)[valid_mask]
                floor_target_inst = torch.zeros(
                    num_instances,
                    dtype=floor_target_flat.dtype,
                    device=floor_target_flat.device,
                ).scatter_add_(0, inverse_indices, floor_target_flat) / counts
                floor_evidence_inst = torch.zeros(
                    num_instances,
                    dtype=floor_evidence_flat.dtype,
                    device=floor_evidence_flat.device,
                ).scatter_add_(0, inverse_indices, floor_evidence_flat) / counts
                if physical_prior_target_clip > 0:
                    floor_target_inst = torch.clamp(floor_target_inst, max=physical_prior_target_clip)
                prior_mask = (
                    (gt_inst_means >= physical_prior_min_height)
                    & (floor_evidence_inst >= physical_prior_evidence_min)
                    & (floor_target_inst > pred_inst_means.detach())
                )
                if prior_mask.any():
                    # Use the stereo-derived physical prior only as a lower-bound hint.
                    # GT remains the ceiling so noisy physical overshoots cannot dominate.
                    prior_target = torch.minimum(floor_target_inst, gt_inst_means)
                    prior_under = F.relu(prior_target.detach() - pred_inst_means)
                    prior_under_loss = F.smooth_l1_loss(
                        prior_under,
                        torch.zeros_like(prior_under),
                        beta=max(physical_prior_beta, 1.0e-3),
                        reduction='none',
                    )
                    iter_loss = (
                        iter_loss
                        + physical_prior_under_weight
                        * prior_under_loss
                        * prior_mask.float()
                        * floor_evidence_inst.detach().clamp(0.0, 1.0)
                    )

            if 'height_signed_correction' in p_item and (
                (low_signed_positive_weight > 0 and low_mask.any()) or signed_correction_target_weight > 0
            ):
                signed_corr = torch.nan_to_num(p_item['height_signed_correction'], nan=0.0, posinf=100.0, neginf=-100.0)
                signed_corr_flat = signed_corr.view(-1)[valid_mask]
                signed_corr_inst = torch.zeros(
                    num_instances,
                    dtype=signed_corr_flat.dtype,
                    device=signed_corr_flat.device,
                ).scatter_add_(0, inverse_indices, signed_corr_flat) / counts
                if low_signed_positive_weight > 0 and low_mask.any():
                    iter_loss = iter_loss + low_signed_positive_weight * F.relu(signed_corr_inst) * low_mask.float()
                if signed_correction_target_weight > 0:
                    base_key = str(_cfg_value(cfg, 'height_signed_correction_target_base_key', 'height_base'))
                    if base_key not in p_item:
                        base_key = 'height_base' if 'height_base' in p_item else 'height_physical'
                    height_base = torch.nan_to_num(p_item[base_key], nan=0.0, posinf=100.0, neginf=0.0)
                    pred_base_flat = height_base.view(-1)[valid_mask]
                    pred_base_inst = torch.zeros(
                        num_instances,
                        dtype=pred_base_flat.dtype,
                        device=pred_base_flat.device,
                    ).scatter_add_(0, inverse_indices, pred_base_flat) / counts
                    signed_target = gt_inst_means - pred_base_inst.detach()
                    signed_target_loss = F.smooth_l1_loss(
                        signed_corr_inst,
                        signed_target,
                        beta=huber_beta,
                        reduction='none',
                    )
                    target_mask = low_mask.float() if str(_cfg_value(cfg, 'height_signed_correction_target_scope', 'low')).lower() == 'low' else torch.ones_like(gt_inst_means)
                    iter_loss = iter_loss + signed_correction_target_weight * signed_target_loss * target_mask

        # ==============================================================================
        # Stratified fusion loss and high-building underestimation penalty.
        # ==============================================================================
        if loss_clip > 0:
            if bool(_cfg_value(cfg, 'instance_loss_soft_clip', False)):
                iter_loss = loss_clip * torch.log1p(torch.clamp(iter_loss, min=0.0) / loss_clip)
            else:
                iter_loss = torch.clamp(iter_loss, max=loss_clip)
        weighted_inst_loss = iter_loss * adaptive_weight * consistency_weight

        ordinal_loss = pred_inst_means.new_tensor(0.0)
        if weak_ordinal_weight > 0 and num_instances > 1:
            pair_source = torch.arange(num_instances, device=gt_inst_means.device)
            max_instances_for_pairs = int(max(2, math.sqrt(max(weak_ordinal_max_pairs, 1))))
            if num_instances > max_instances_for_pairs:
                stride = max(1, int(math.ceil(num_instances / max_instances_for_pairs)))
                pair_source = pair_source[::stride][:max_instances_for_pairs]
            gt_pair = gt_inst_means[pair_source]
            pred_pair = pred_inst_means[pair_source]
            gt_diff = gt_pair[:, None] - gt_pair[None, :]
            pair_mask = gt_diff.abs() >= weak_ordinal_min_gap
            upper_mask = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
            pair_mask = pair_mask & upper_mask
            if pair_mask.any():
                sign = torch.sign(gt_diff[pair_mask])
                pred_diff = pred_pair[:, None] - pred_pair[None, :]
                rank_violation = F.relu(weak_ordinal_margin - sign * pred_diff[pair_mask])
                pair_weights = torch.ones_like(rank_violation)
                if weak_ordinal_physical_weight and is_scheme4 and 'height_physical' in p_item:
                    height_physical = torch.nan_to_num(p_item['height_physical'], nan=0.0, posinf=320.0, neginf=0.0).float()
                    physical_flat = height_physical.view(-1)[valid_mask]
                    physical_inst = torch.zeros(
                        num_instances,
                        dtype=physical_flat.dtype,
                        device=physical_flat.device,
                    ).scatter_add_(0, inverse_indices, physical_flat) / counts
                    phys_pair = physical_inst[pair_source]
                    phys_diff = phys_pair[:, None] - phys_pair[None, :]
                    phys_agree = torch.sign(phys_diff[pair_mask]) == sign
                    phys_gap = torch.sigmoid((phys_diff[pair_mask].abs() - weak_ordinal_min_gap) / max(weak_ordinal_min_gap, 1.0e-3))
                    pair_weights = torch.where(
                        phys_agree,
                        phys_gap,
                        torch.full_like(phys_gap, weak_ordinal_min_weight),
                    )
                ordinal_loss = (rank_violation * pair_weights).mean()
        
        # 1. Global weighted loss for the dominant low-rise population.
        global_loss = weighted_inst_loss.mean()
        
        # 2. Stratified mean loss to give sparse high-rise samples sufficient gradients.
        unique_present_bins = torch.unique(gt_bins)
        stratified_loss = 0.0
        high_bias_loss = 0.0
        low_bias_loss = 0.0
        high_bin_count = 0
        low_bin_count = 0
        for b in unique_present_bins:
            b_mask = (gt_bins == b)
            stratified_loss += weighted_inst_loss[b_mask].mean()
            if gt_inst_means[b_mask].mean() >= high_threshold:
                high_bias_loss = high_bias_loss + F.relu(-(pred_inst_means[b_mask] - gt_inst_means[b_mask]).mean())
                high_bin_count += 1
            if gt_inst_means[b_mask].mean() <= low_height_threshold:
                low_bias_loss = low_bias_loss + F.relu((pred_inst_means[b_mask] - gt_inst_means[b_mask]).mean())
                low_bin_count += 1
        stratified_loss = stratified_loss / unique_present_bins.numel()
        if high_bin_count > 0:
            high_bias_loss = high_bias_loss / high_bin_count
        if low_bin_count > 0:
            low_bias_loss = low_bias_loss / low_bin_count
        
        blended_loss = (
            global_loss_weight * global_loss
            + stratified_loss_weight * stratified_loss
            + high_bias_weight * high_bias_loss
            + low_bias_weight * low_bias_loss
            + weak_ordinal_weight * ordinal_loss
        )

        total_loss += blended_loss * weight_decay

    final_diff = torch.abs(pred_inst_means - gt_inst_means)
    mae_sum = final_diff.sum()
    mse_sum = (final_diff ** 2).sum()
    sum_y = gt_inst_means.sum()
    sum_y_sq = (gt_inst_means ** 2).sum()

    metrics = {
        'train/mae_sum': mae_sum.detach(),
        'train/mse_sum': mse_sum.detach(),
        'train/sum_y': sum_y.detach(),
        'train/sum_y_sq': sum_y_sq.detach(),
        'train/inst_num': torch.tensor([num_instances], dtype=mae_sum.dtype, device=mae_sum.device)
    }
    return total_loss, metrics


def fetch_optimizer(args, model, steps_per_epoch):
    dtm_params = []
    head_params = []
    calib_params = []
    delta_source = str(getattr(args, 'height_delta_source', 'learned_base_plane')).lower()
    optimize_dtm = delta_source not in ('precomputed', 'cached', 'cache', 'label_ring')
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'base_plane_estimator' in name and optimize_dtm:
            dtm_params.append(p)
        elif (
                'height_head' in name
                or 'instance_correction_head' in name
                or 'high_residual_head' in name
                or 'high_residual_gate_head' in name
                or 'learned_physical_floor_head' in name
            ):
            head_params.append(p)
        elif (
                name.startswith('height_physical_')
                or name.startswith('height_instance_feature_')
            ):
            calib_params.append(p)
            
    dtm_lr = getattr(args, 'dtm_lr', args.lr * 1.0) 
    head_lr = getattr(args, 'head_lr', args.lr * 1.0) 
    calib_lr = getattr(args, 'calibration_lr', head_lr)
    
    print(
        f"[*] Optimizing specific modules -> DTM Estimator: {len(dtm_params)} tensors | "
        f"Hybrid Head: {len(head_params)} tensors | Calibration: {len(calib_params)} tensors"
    )

    param_groups = []
    max_lrs = []
    if dtm_params:
        param_groups.append({'params': dtm_params, 'lr': dtm_lr, 'name': 'dtm'})
        max_lrs.append(dtm_lr)
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr, 'name': 'head'})
        max_lrs.append(head_lr)
    if calib_params:
        param_groups.append({'params': calib_params, 'lr': calib_lr, 'name': 'calibration', 'weight_decay': 0.0})
        max_lrs.append(calib_lr)
    if not param_groups:
        raise RuntimeError("No trainable parameters found for fine-tuning.")

    optimizer = optim.AdamW(param_groups, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler


def apply_epoch_lr_guards(optimizer, cfg, epoch):
    dtm_freeze_epochs = int(getattr(cfg, 'dtm_freeze_epochs', 0))
    if dtm_freeze_epochs <= 0 or epoch > dtm_freeze_epochs:
        return
    for group in optimizer.param_groups:
        if group.get('name') == 'dtm':
            group['lr'] = 0.0


def gradients_are_finite(model):
    device = next(model.parameters()).device
    finite = torch.ones((), device=device)
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            finite = torch.zeros((), device=device)
            break
    return finite


def _load_matching_state_dict(model, checkpoint, accelerator=None, skip_prefixes=None):
    source = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    source = {k.replace('module.', ''): v for k, v in source.items()}
    skip_prefixes = [str(p) for p in (skip_prefixes or []) if str(p)]
    target = model.state_dict()
    matched = {}
    skipped = []
    skipped_by_prefix = []
    for key, value in source.items():
        if skip_prefixes and any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
        elif key in target and tuple(target[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    if accelerator is None or accelerator.is_main_process:
        print(
            f"[*] Loaded matching checkpoint tensors: {len(matched)} | "
            f"skipped_shape_or_missing={len(skipped)} | skipped_by_prefix={len(skipped_by_prefix)} | "
            f"missing_after_load={len(missing)} | unexpected={len(unexpected)}"
        )
        if skipped_by_prefix:
            preview = ', '.join(skipped_by_prefix[:8])
            suffix = ' ...' if len(skipped_by_prefix) > 8 else ''
            print(f"[*] Skipped checkpoint tensors by prefix: {preview}{suffix}")
        if skipped:
            preview = ', '.join(skipped[:8])
            suffix = ' ...' if len(skipped) > 8 else ''
            print(f"[*] Skipped checkpoint tensors: {preview}{suffix}")
    return missing, unexpected, skipped


@hydra.main(version_base=None, config_path='config', config_name='train_finetune') 
def main(cfg):
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    set_seed(cfg.seed)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    
    ddp_find_unused = bool(getattr(cfg, 'ddp_find_unused_parameters', True))
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
    acc_steps = getattr(cfg, 'gradient_accumulation_steps', 1)
    
    mixed_precision = str(getattr(cfg, 'mixed_precision', 'fp16')).lower()
    if mixed_precision in ('none', 'false', '0'):
        mixed_precision = 'no'

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=acc_steps,  
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), 
        kwargs_handlers=[kwargs]
    )

    train_dataset = datasets.fetch_dataloader(cfg)
    require_non_empty = bool(getattr(cfg, 'require_non_empty_datasets', True))
    if require_non_empty and len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty. train_data_path={getattr(cfg, 'train_data_path', '')}, "
            f"train_cities={list(getattr(cfg, 'train_cities', []))}"
        )
    
    train_num_workers = int(getattr(cfg, 'train_num_workers', 8))
    val_num_workers = int(getattr(cfg, 'val_num_workers', 2))
    train_loader_timeout = int(getattr(cfg, 'train_loader_timeout', 0))
    val_loader_timeout = int(getattr(cfg, 'val_loader_timeout', 0))
    pin_memory = bool(getattr(cfg, 'dataloader_pin_memory', True))

    if hasattr(train_dataset, 'sample_weights') and train_dataset.sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        epoch_samples = int(getattr(cfg, 'train_epoch_samples', len(train_dataset)))
        epoch_samples = max(cfg.batch_size, min(epoch_samples, len(train_dataset)))
        sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=epoch_samples, replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size // getattr(cfg, 'num_gpu', 1), 
            pin_memory=pin_memory, sampler=sampler, shuffle=False,
            num_workers=train_num_workers, drop_last=True,
            timeout=train_loader_timeout if train_num_workers > 0 else 0
        )
        if accelerator.is_main_process:
            print(f"[*] Loaded WeightedRandomSampler for height-balanced sampling. epoch_samples={epoch_samples}")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size // getattr(cfg, 'num_gpu', 1), 
            pin_memory=pin_memory, shuffle=True,
            num_workers=train_num_workers, drop_last=True,
            timeout=train_loader_timeout if train_num_workers > 0 else 0
        )

    val_dataset = datasets.fetch_val_dataloader(cfg)
    if require_non_empty and len(val_dataset) == 0:
        raise RuntimeError(
            f"Validation dataset is empty. val_data_path={getattr(cfg, 'val_data_path', '')}, "
            f"val_cities={list(getattr(cfg, 'val_cities', []))}"
        )
    val_batch_size = max(1, int(getattr(cfg, 'val_batch_size', 1)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, pin_memory=pin_memory,
        shuffle=False, num_workers=val_num_workers,
        timeout=val_loader_timeout if val_num_workers > 0 else 0
    )

    if accelerator.is_main_process:
        print(
            f"Start finetuning | Samples: {len(train_dataset)} | Acc steps: {acc_steps} "
            f"| workers train/val={train_num_workers}/{val_num_workers} | val_batch={val_batch_size}"
        )

    model = SD2H_Height_Finetuner(cfg)
    start_epoch = 1
    best_val_rmse = float('inf')
    best_high_rmse = float('inf')
    best_score = float('inf')
    early_stop_best = float('inf')
    early_stop_bad_epochs = 0

    if getattr(cfg, 'restore_ckpt', None) is not None and getattr(cfg, 'resume_ckpt', None) is None:
        if accelerator.is_main_process: print(f"Loading weights: {cfg.restore_ckpt}")
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu', weights_only=True)
        missing, _, skipped = _load_matching_state_dict(
            model,
            checkpoint,
            accelerator,
            skip_prefixes=getattr(cfg, 'restore_skip_prefixes', []),
        )
        required_prefixes = [str(p) for p in getattr(cfg, 'restore_require_loaded_prefixes', []) if str(p)]
        if required_prefixes:
            bad_missing = [
                key for key in missing
                if any(key.startswith(prefix) for prefix in required_prefixes)
            ]
            bad_skipped = [
                key for key in skipped
                if any(key.startswith(prefix) for prefix in required_prefixes)
            ]
            if bad_missing or bad_skipped:
                preview = ', '.join((bad_missing + bad_skipped)[:12])
                raise RuntimeError(
                    "Required checkpoint prefixes were not fully restored: "
                    f"{required_prefixes}. Offending tensors: {preview}"
                )

    optimizer, lr_scheduler = fetch_optimizer(cfg, model, len(train_loader))

    if getattr(cfg, 'resume_ckpt', None) is not None:
        if accelerator.is_main_process: print(f"Resuming from: {cfg.resume_ckpt}")
        checkpoint = torch.load(cfg.resume_ckpt, map_location='cpu', weights_only=False)
        _load_matching_state_dict(model, checkpoint, accelerator)
        
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_rmse = checkpoint.get('best_val_rmse', best_val_rmse)
            best_high_rmse = checkpoint.get('best_high_rmse', best_high_rmse)
            best_score = checkpoint.get('best_score', best_score)
            if accelerator.is_main_process: print(f"Resume training from Epoch {start_epoch}")

    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler, val_loader
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        should_stop_epoch = False
        model.train()
        model_module = model.module if hasattr(model, 'module') else model
        if hasattr(model_module, 'set_finetune_epoch'):
            model_module.set_finetune_epoch(epoch)
        if getattr(model_module, 'disp_net', None) is not None:
            model_module.disp_net.eval() 
        apply_epoch_lr_guards(optimizer, cfg, epoch)

        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_mae_sum = 0.0
        epoch_mse_sum = 0.0
        epoch_sum_y = 0.0
        epoch_sum_y_sq = 0.0
        epoch_inst_count = 0.0
        collect_train_bins = bool(getattr(cfg, 'log_train_height_bins', False))
        train_gt_list = []
        train_pred_list = []

        max_train_batches = int(getattr(cfg, 'max_train_batches_per_epoch', 0) or 0)
        pbar = tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
        for train_step, data in enumerate(pbar, start=1):
            if len(data) == 9:
                _, left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor = data
            else:
                _, left, right, cached_disp, cached_delta, instance_mask, height_gt, k_tensor = data
                cached_ground = None
            if instance_mask.dim() == 3: instance_mask = instance_mask.unsqueeze(1)
            if height_gt.dim() == 3: height_gt = height_gt.unsqueeze(1)
            if k_tensor.dim() == 1: k_tensor = k_tensor.unsqueeze(1) 
            
            with accelerator.accumulate(model):  
                with accelerator.autocast():  
                    preds_list = model(
                        left,
                        right,
                        mask=instance_mask,
                        k_tensor=k_tensor,
                        iters=cfg.train_iters,
                        test_mode=False,
                        precomputed_disp=cached_disp,
                        precomputed_delta=cached_delta,
                        precomputed_ground=cached_ground,
                    )
                
                loss, metrics = instance_height_loss(preds_list, height_gt, instance_mask, epoch=epoch, total_epochs=cfg.epochs, cfg=cfg)
                finite_flag = torch.tensor(
                    [1.0 if torch.isfinite(loss).item() else 0.0],
                    device=accelerator.device,
                )
                finite_count = accelerator.reduce(finite_flag, reduction='sum')
                if finite_count.item() < accelerator.num_processes:
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.is_main_process:
                        print(f"[!] Non-finite loss at epoch {epoch}; skipping batch on all ranks.")
                    continue
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_finite = accelerator.reduce(gradients_are_finite(model), reduction='sum')
                    if grad_finite.item() < accelerator.num_processes:
                        optimizer.zero_grad(set_to_none=True)
                        if accelerator.is_main_process:
                            print(f"[!] Non-finite gradient at epoch {epoch}; skipping optimizer step.")
                        continue
                    max_grad_norm = float(getattr(cfg, 'max_grad_norm', 1.0))
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                apply_epoch_lr_guards(optimizer, cfg, epoch)
                optimizer.zero_grad(set_to_none=True)

            loss_red = accelerator.reduce(loss.detach(), reduction='mean')
            mae_sum_red = accelerator.reduce(metrics['train/mae_sum'], reduction='sum')
            mse_sum_red = accelerator.reduce(metrics['train/mse_sum'], reduction='sum')
            sum_y_red = accelerator.reduce(metrics['train/sum_y'], reduction='sum')
            sum_y_sq_red = accelerator.reduce(metrics['train/sum_y_sq'], reduction='sum')
            inst_num_red = accelerator.reduce(metrics['train/inst_num'], reduction='sum')

            loss_val = loss_red.item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                epoch_loss_sum += loss_val
                epoch_steps += 1
                epoch_mae_sum += mae_sum_red.item()
                epoch_mse_sum += mse_sum_red.item()
                epoch_sum_y += sum_y_red.item()
                epoch_sum_y_sq += sum_y_sq_red.item()
                epoch_inst_count += inst_num_red.item()

            if collect_train_bins and accelerator.is_main_process:
                last_out_train = preds_list[-1] if isinstance(preds_list, list) else preds_list
                h_train_pred = last_out_train['height'] if isinstance(last_out_train, dict) else last_out_train
                with torch.no_grad():
                    b_size = h_train_pred.shape[0]
                    batch_offsets = torch.arange(b_size, device=instance_mask.device).view(b_size, 1, 1, 1) * 100000
                    global_inst_mask = torch.where(instance_mask > 0, instance_mask + batch_offsets, torch.zeros_like(instance_mask))
                    inst_flat_train = global_inst_mask.view(-1).long()
                    valid_train = inst_flat_train > 0
                    if valid_train.any():
                        v_gt_train = torch.nan_to_num(height_gt.float(), nan=0.0, posinf=320.0, neginf=0.0).view(-1)[valid_train]
                        v_pred_train = torch.clamp(
                            torch.nan_to_num(h_train_pred.detach().float(), nan=0.0, posinf=320.0, neginf=0.0),
                            min=0.0,
                            max=320.0,
                        ).view(-1)[valid_train]
                        v_inst_train = inst_flat_train[valid_train]
                        u_train, inv_train = torch.unique(v_inst_train, return_inverse=True)
                        c_train = torch.zeros(u_train.numel(), dtype=v_gt_train.dtype, device=v_gt_train.device).scatter_add_(
                            0, inv_train, torch.ones_like(v_gt_train)
                        )
                        g_train = torch.zeros(u_train.numel(), dtype=v_gt_train.dtype, device=v_gt_train.device).scatter_add_(
                            0, inv_train, v_gt_train
                        ) / c_train
                        p_train = torch.zeros(u_train.numel(), dtype=v_pred_train.dtype, device=v_pred_train.device).scatter_add_(
                            0, inv_train, v_pred_train
                        ) / c_train
                        train_gt_list.append(g_train.float().cpu())
                        train_pred_list.append(p_train.float().cpu())
                        del v_gt_train, v_pred_train, v_inst_train
                        del u_train, inv_train, c_train, g_train, p_train
                    del batch_offsets, global_inst_mask, inst_flat_train, valid_train
                del last_out_train, h_train_pred
            
            if accelerator.is_main_process and epoch_steps > 0:
                pbar.set_description(f"Epoch [{epoch}/{cfg.epochs}]")
                batch_mae_disp = mae_sum_red.item() / (inst_num_red.item() + 1e-6)
                pbar.set_postfix({'Loss': f"{loss_val:.4f}", 'MAE': f"{batch_mae_disp:.2f}"})

            del preds_list, loss, metrics
            del loss_red, mae_sum_red, mse_sum_red, sum_y_red, sum_y_sq_red, inst_num_red
            del finite_flag, finite_count
            if 'grad_finite' in locals():
                del grad_finite
            del left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor
            empty_cache_steps = int(getattr(cfg, 'train_empty_cache_steps', 0))
            if empty_cache_steps > 0 and epoch_steps > 0 and epoch_steps % empty_cache_steps == 0:
                accelerator.wait_for_everyone()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if max_train_batches > 0 and train_step >= max_train_batches:
                break

        if accelerator.is_main_process and epoch_inst_count > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            avg_epoch_mae = epoch_mae_sum / epoch_inst_count
            avg_epoch_rmse = math.sqrt(epoch_mse_sum / epoch_inst_count)
            global_mean_y = epoch_sum_y / epoch_inst_count
            global_ss_tot = epoch_sum_y_sq - epoch_inst_count * (global_mean_y ** 2)
            avg_epoch_r2 = 1.0 - (epoch_mse_sum / (global_ss_tot + 1e-8))
            print(f"[Epoch {epoch} Train] Bldgs: {int(epoch_inst_count)} | Loss: {avg_epoch_loss:.4f} | MAE: {avg_epoch_mae:.3f}m | RMSE: {avg_epoch_rmse:.3f}m | R²: {avg_epoch_r2:.3f}")
            if collect_train_bins and train_gt_list:
                train_gt_tensor = torch.cat(train_gt_list)
                train_pred_tensor = torch.cat(train_pred_list)
                train_bin_rows = _height_bin_rows_torch(train_gt_tensor.float(), train_pred_tensor.float(), _height_eval_edges(cfg))
                _print_height_bin_rows(train_bin_rows, "Train Height Stratified Analysis")
                train_bin_csv = Path(cfg.save_path) / f"train_height_metrics_by_bin_epoch_{epoch}.csv"
                _save_height_bin_csv(train_bin_csv, train_bin_rows)
                print(f"[*] Stratified train metrics saved to: {train_bin_csv}\n")

        if epoch % cfg.val_epochs == 0:
            model.eval()
            if torch.cuda.is_available():
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
            model_module = model.module if hasattr(model, 'module') else model
            if hasattr(model_module, 'set_finetune_epoch'):
                model_module.set_finetune_epoch(epoch)
            val_elem_num, val_mae_sum, val_mse_sum, val_sum_y, val_sum_y_sq = 0.0, 0.0, 0.0, 0.0, 0.0
            
            all_gt_list = []
            all_pred_list = []
            log_high_diag = bool(getattr(cfg, 'height_log_high_residual_diag', False))
            high_diag_keys = [
                'height_pre_high_residual',
                'height_high_residual_basis',
                'height_high_residual',
                'height_high_residual_raw',
                'height_high_residual_gate',
                'height_high_residual_evidence_gate',
                'height_high_residual_learned_gate',
            ]
            high_diag_lists = {key: [] for key in high_diag_keys}

            max_val_batches = int(getattr(cfg, 'max_val_batches_per_epoch', 0) or 0)
            pbar_val = tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
            val_empty_cache_steps = int(getattr(cfg, 'val_empty_cache_steps', 0))
            val_gc_steps = int(getattr(cfg, 'val_gc_steps', 0))
            val_forward_batch_size = max(1, int(getattr(cfg, 'val_forward_batch_size', 1)))
            
            for val_step, data in enumerate(pbar_val, start=1):
                if len(data) == 9:
                    _, left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor = data
                else:
                    _, left, right, cached_disp, cached_delta, instance_mask, height_gt, k_tensor = data
                    cached_ground = None
                if instance_mask.dim() == 3: instance_mask = instance_mask.unsqueeze(1)
                if height_gt.dim() == 3: height_gt = height_gt.unsqueeze(1)
                if k_tensor.dim() == 1: k_tensor = k_tensor.unsqueeze(1)

                h_pred_chunks = []
                high_diag_map_chunks = {key: [] for key in high_diag_keys} if log_high_diag else {}
                batch_size_for_forward = int(left.shape[0])
                with torch.inference_mode(), accelerator.autocast():
                    for start_idx in range(0, batch_size_for_forward, val_forward_batch_size):
                        end_idx = min(start_idx + val_forward_batch_size, batch_size_for_forward)
                        preds_out = model(
                            left[start_idx:end_idx],
                            right[start_idx:end_idx],
                            mask=instance_mask[start_idx:end_idx],
                            k_tensor=k_tensor[start_idx:end_idx],
                            iters=cfg.valid_iters,
                            test_mode=True,
                            precomputed_disp=cached_disp[start_idx:end_idx],
                            precomputed_delta=cached_delta[start_idx:end_idx],
                            precomputed_ground=(
                                cached_ground[start_idx:end_idx]
                                if cached_ground is not None
                                else None
                            ),
                        )
                        last_out = preds_out[-1] if isinstance(preds_out, list) else preds_out

                        if isinstance(last_out, dict):
                            h_pred_part = last_out['height']
                        else:
                            h_pred_part = last_out

                        h_pred_part = torch.clamp(
                            torch.nan_to_num(h_pred_part.float(), nan=0.0, posinf=320.0, neginf=0.0),
                            min=0.0,
                            max=320.0,
                        )
                        h_pred_chunks.append(h_pred_part.detach())
                        if log_high_diag and isinstance(last_out, dict):
                            for key in high_diag_keys:
                                if key not in last_out:
                                    continue
                                high_diag_map_chunks[key].append(
                                    torch.nan_to_num(
                                        last_out[key].float(),
                                        nan=0.0,
                                        posinf=320.0,
                                        neginf=-320.0,
                                    ).detach()
                                )
                        del preds_out, last_out, h_pred_part
                    h_pred = torch.cat(h_pred_chunks, dim=0)
                    high_diag_maps = {
                        key: torch.cat(chunks, dim=0)
                        for key, chunks in high_diag_map_chunks.items()
                        if chunks
                    }
                del h_pred_chunks
                
                b_size = h_pred.shape[0]
                batch_offsets = torch.arange(b_size, device=instance_mask.device).view(b_size, 1, 1, 1) * 100000
                global_inst_mask = torch.where(instance_mask > 0, instance_mask + batch_offsets, torch.zeros_like(instance_mask))
                inst_flat_val = global_inst_mask.view(-1).long()
                valid_mask = inst_flat_val > 0
                
                # ==============================================================================
                # Keep DDP collective communication outside conditional branches.
                # ==============================================================================
                if not valid_mask.any():
                    batch_metrics = {
                        'mae_sum': torch.tensor([0.0], device=accelerator.device), 'mse_sum': torch.tensor([0.0], device=accelerator.device),
                        'sum_y': torch.tensor([0.0], device=accelerator.device), 'sum_y_sq': torch.tensor([0.0], device=accelerator.device),
                        'inst_num': torch.tensor([0.0], device=accelerator.device)
                    }
                    # Use -1 placeholders for empty batches to keep DDP gather synchronized.
                    g_m = torch.tensor([-1.0], device=accelerator.device)
                    p_m = torch.tensor([-1.0], device=accelerator.device)
                    diag_inst_maps = {
                        key: torch.tensor([-1.0], device=accelerator.device)
                        for key in high_diag_maps.keys()
                    }
                    num_i = 0
                else:
                    v_gt = torch.nan_to_num(height_gt.float(), nan=0.0, posinf=320.0, neginf=0.0).view(-1)[valid_mask]
                    v_pred = h_pred.view(-1)[valid_mask]
                    v_inst_ids = inst_flat_val[valid_mask]
                    u_inst, inv_idx = torch.unique(v_inst_ids, return_inverse=True)
                    num_i = u_inst.numel()
                    
                    c = torch.zeros(num_i, dtype=v_gt.dtype, device=v_gt.device).scatter_add_(0, inv_idx, torch.ones_like(v_gt))
                    g_m = torch.zeros(num_i, dtype=v_gt.dtype, device=v_gt.device).scatter_add_(0, inv_idx, v_gt) / c
                    p_m = torch.zeros(num_i, dtype=v_gt.dtype, device=v_gt.device).scatter_add_(0, inv_idx, v_pred) / c
                    
                    diff = torch.abs(p_m - g_m)
                    batch_metrics = {
                        'mae_sum': diff.sum().unsqueeze(0), 'mse_sum': (diff ** 2).sum().unsqueeze(0),
                        'sum_y': g_m.sum().unsqueeze(0), 'sum_y_sq': (g_m ** 2).sum().unsqueeze(0),
                        'inst_num': torch.tensor([num_i], dtype=diff.dtype, device=accelerator.device)
                    }
                    diag_inst_maps = {}
                    if log_high_diag:
                        for key in high_diag_keys:
                            if key not in high_diag_maps:
                                continue
                            diag_map = high_diag_maps[key]
                            diag_flat = diag_map.view(-1)[valid_mask]
                            diag_inst = torch.zeros(
                                num_i,
                                dtype=diag_flat.dtype,
                                device=diag_flat.device,
                            ).scatter_add_(0, inv_idx, diag_flat) / c
                            diag_inst_maps[key] = diag_inst
                            del diag_flat, diag_inst

                # All ranks pad and gather unconditionally.
                g_m_pad = pad_across_processes(g_m.unsqueeze(0), dim=1, pad_index=-1.0).squeeze(0)
                p_m_pad = pad_across_processes(p_m.unsqueeze(0), dim=1, pad_index=-1.0).squeeze(0)
                g_m_gathered = accelerator.gather(g_m_pad)
                p_m_gathered = accelerator.gather(p_m_pad)
                valid_gathered = g_m_gathered != -1.0
                
                if accelerator.is_main_process:
                    # Remove -1 placeholders and recover valid instances.
                    gt_cpu = g_m_gathered[valid_gathered].detach().cpu()
                    pred_cpu = p_m_gathered[valid_gathered].detach().cpu()
                    all_gt_list.append(gt_cpu)
                    all_pred_list.append(pred_cpu)
                    del gt_cpu, pred_cpu

                if log_high_diag:
                    for key in high_diag_keys:
                        if key not in diag_inst_maps:
                            continue
                        diag_pad = pad_across_processes(
                            diag_inst_maps[key].unsqueeze(0),
                            dim=1,
                            pad_index=-1.0,
                        ).squeeze(0)
                        diag_gathered = accelerator.gather(diag_pad)
                        if accelerator.is_main_process:
                            diag_cpu = diag_gathered[valid_gathered].detach().cpu()
                            high_diag_lists[key].append(diag_cpu)
                            del diag_cpu
                        del diag_pad, diag_gathered

                gathered_metrics = accelerator.gather_for_metrics(batch_metrics)
                val_elem_num += gathered_metrics['inst_num'].sum().item()
                val_mae_sum += gathered_metrics['mae_sum'].sum().item()
                val_mse_sum += gathered_metrics['mse_sum'].sum().item()
                val_sum_y += gathered_metrics['sum_y'].sum().item()
                val_sum_y_sq += gathered_metrics['sum_y_sq'].sum().item()

                del h_pred, high_diag_map_chunks, high_diag_maps, batch_metrics, gathered_metrics
                del left, right, cached_disp, cached_delta, cached_ground, instance_mask, height_gt, k_tensor
                del batch_offsets, global_inst_mask, inst_flat_val, valid_mask
                del g_m, p_m, g_m_pad, p_m_pad, g_m_gathered, p_m_gathered, valid_gathered, diag_inst_maps
                del data
                if 'v_gt' in locals():
                    del v_gt, v_pred, v_inst_ids, u_inst, inv_idx, c, diff
                if val_gc_steps > 0 and val_step % val_gc_steps == 0:
                    gc.collect()
                if torch.cuda.is_available() and val_empty_cache_steps > 0 and val_step % val_empty_cache_steps == 0:
                    torch.cuda.empty_cache()
                if max_val_batches > 0 and val_step >= max_val_batches:
                    break
            
            if accelerator.is_main_process and val_elem_num > 0:
                final_val_mae = val_mae_sum / val_elem_num
                final_val_rmse = math.sqrt(val_mse_sum / val_elem_num)
                val_mean_y = val_sum_y / val_elem_num
                val_ss_tot = val_sum_y_sq - val_elem_num * (val_mean_y ** 2)
                final_val_r2 = 1.0 - (val_mse_sum / (val_ss_tot + 1e-8))
                print(f"[Epoch {epoch} Val] Bldgs: {int(val_elem_num)} | MAE: {final_val_mae:.3f}m | RMSE: {final_val_rmse:.3f}m | R²: {final_val_r2:.3f}")
                
                # =========================================================================
                # Stratified bias analysis.
                # =========================================================================
                all_gt_tensor = None
                all_pred_tensor = None
                bin_metric_rows = []
                if len(all_gt_list) > 0:
                    all_gt_tensor = torch.cat(all_gt_list)
                    all_pred_tensor = torch.cat(all_pred_list)
                    eval_edges = _height_eval_edges(cfg)
                    bin_metric_rows = _height_bin_rows_torch(all_gt_tensor.float(), all_pred_tensor.float(), eval_edges)

                    _print_height_bin_rows(bin_metric_rows, "Val Height Stratified Analysis")
                    bin_csv = Path(cfg.save_path) / f"val_height_metrics_by_bin_epoch_{epoch}.csv"
                    _save_height_bin_csv(bin_csv, bin_metric_rows)
                    print(f"[*] Stratified validation metrics saved to: {bin_csv}")
                    print("\n")

                high_threshold = float(getattr(cfg, 'best_high_threshold', 30.0))
                high_score_weight = float(getattr(cfg, 'best_high_score_weight', 1.0))
                under_bias_weight = float(getattr(cfg, 'best_under_bias_weight', 0.5))
                high_rmse = final_val_rmse
                high_bias = 0.0
                high_count = 0
                if all_gt_tensor is not None and all_pred_tensor is not None:
                    high_mask = all_gt_tensor >= high_threshold
                    high_count = int(high_mask.sum().item())
                    if high_count > 0:
                        high_diff = all_pred_tensor[high_mask] - all_gt_tensor[high_mask]
                        high_rmse = torch.sqrt((high_diff ** 2).mean()).item()
                        high_bias = high_diff.mean().item()

                high_under_bias = max(0.0, -high_bias)
                mae_score_weight = float(getattr(cfg, 'best_mae_score_weight', 0.0))
                r2_score_weight = float(getattr(cfg, 'best_r2_score_weight', 0.0))
                r2_target = float(getattr(cfg, 'best_r2_target', 0.0))
                r2_penalty = max(0.0, r2_target - final_val_r2)
                score = (
                    final_val_rmse
                    + mae_score_weight * final_val_mae
                    + high_score_weight * high_rmse
                    + under_bias_weight * high_under_bias
                    + r2_score_weight * r2_penalty
                )
                print(f"[Epoch {epoch} BestMonitor] High>={high_threshold:.1f}m: N={high_count} | RMSE={high_rmse:.3f}m | Bias={high_bias:+.3f}m | Score={score:.3f}")

                if log_high_diag and all_gt_tensor is not None and all_pred_tensor is not None:
                    diag_tensors = {
                        key: torch.cat(values).float()
                        for key, values in high_diag_lists.items()
                        if values
                    }
                    if diag_tensors:
                        diag_groups = [
                            ('low<12m', all_gt_tensor < 12.0),
                            ('mid12-30m', (all_gt_tensor >= 12.0) & (all_gt_tensor < 30.0)),
                            (f'high>={high_threshold:g}m', all_gt_tensor >= high_threshold),
                        ]
                        print("[HighResidualDiag] group      N   pred   pre   res  basis   raw   gate  evidence learned")
                        for group_name, group_mask in diag_groups:
                            count = int(group_mask.sum().item())
                            if count == 0:
                                continue
                            pred_mean = all_pred_tensor[group_mask].float().mean().item()
                            def _diag_mean(key):
                                if key not in diag_tensors:
                                    return float('nan')
                                return diag_tensors[key][group_mask].mean().item()
                            print(
                                f"[HighResidualDiag] {group_name:<9} {count:5d} "
                                f"{pred_mean:6.2f} "
                                f"{_diag_mean('height_pre_high_residual'):6.2f} "
                                f"{_diag_mean('height_high_residual'):6.2f} "
                                f"{_diag_mean('height_high_residual_basis'):6.2f} "
                                f"{_diag_mean('height_high_residual_raw'):6.2f} "
                                f"{_diag_mean('height_high_residual_gate'):6.3f} "
                                f"{_diag_mean('height_high_residual_evidence_gate'):8.3f} "
                                f"{_diag_mean('height_high_residual_learned_gate'):7.3f}"
                            )

                metric_snapshot = {
                    'val_mae': final_val_mae,
                    'val_rmse': final_val_rmse,
                    'val_r2': final_val_r2,
                    'high_threshold': high_threshold,
                    'high_count': high_count,
                    'high_rmse': high_rmse,
                    'high_bias': high_bias,
                    'score_mae_weight': mae_score_weight,
                    'score_r2_weight': r2_score_weight,
                    'score_r2_target': r2_target,
                    'score_r2_penalty': r2_penalty,
                    'score': score,
                    'height_bin_metrics': bin_metric_rows,
                }

                def save_best_checkpoint(filename, metrics):
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'best_val_rmse': best_val_rmse,
                        'best_high_rmse': best_high_rmse,
                        'best_score': best_score,
                        'metrics': metrics,
                    }
                    torch.save(save_dict, Path(cfg.save_path) / filename)

                if final_val_rmse < best_val_rmse:
                    best_val_rmse = final_val_rmse
                    save_best_checkpoint('finetune_best_rmse.pth', metric_snapshot)
                    print(f"[*] Saved best overall RMSE checkpoint: {best_val_rmse:.3f}m")

                if high_count > 0 and high_rmse < best_high_rmse:
                    best_high_rmse = high_rmse
                    save_best_checkpoint(f'finetune_best_high{int(high_threshold)}_rmse.pth', metric_snapshot)
                    print(f"[*] Saved best high-building RMSE checkpoint: {best_high_rmse:.3f}m")

                if score < best_score:
                    best_score = score
                    save_best_checkpoint('finetune_best_score.pth', metric_snapshot)
                    print(f"[*] Saved best combined-score checkpoint: {best_score:.3f}")

                early_stop_patience = int(getattr(cfg, 'early_stop_patience', 0))
                if early_stop_patience > 0:
                    early_metric_name = str(getattr(cfg, 'early_stop_metric', 'score')).lower()
                    if early_metric_name == 'rmse':
                        early_metric = final_val_rmse
                    elif early_metric_name == 'mae':
                        early_metric = final_val_mae
                    elif early_metric_name == 'r2':
                        early_metric = -final_val_r2
                    else:
                        early_metric = score

                    min_delta = float(getattr(cfg, 'early_stop_min_delta', 0.0))
                    min_epoch = int(getattr(cfg, 'early_stop_min_epoch', 1))
                    if early_metric < early_stop_best - min_delta:
                        early_stop_best = early_metric
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                    print(
                        f"[EarlyStop] metric={early_metric_name} value={early_metric:.4f} "
                        f"best={early_stop_best:.4f} bad_epochs={early_stop_bad_epochs}/{early_stop_patience}"
                    )
                    if epoch >= min_epoch and early_stop_bad_epochs >= early_stop_patience:
                        should_stop_epoch = True
                        print(f"[*] Early stopping triggered at epoch {epoch}.")

                del all_gt_tensor, all_pred_tensor, bin_metric_rows, metric_snapshot
                if 'diag_tensors' in locals():
                    del diag_tensors

            if torch.cuda.is_available():
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
            del all_gt_list, all_pred_list, high_diag_lists
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (
            epoch % cfg.save_epochs == 0
            and accelerator.is_main_process
            and bool(getattr(cfg, 'save_epoch_checkpoints', True))
        ):
            save_dict = {
                'epoch': epoch,
                'state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_val_rmse': best_val_rmse,
                'best_high_rmse': best_high_rmse,
                'best_score': best_score
            }
            torch.save(save_dict, Path(cfg.save_path) / f'finetune_epoch_{epoch}.pth')

        stop_tensor = torch.tensor(
            [1.0 if should_stop_epoch else 0.0],
            device=accelerator.device,
        )
        stop_tensor = accelerator.reduce(stop_tensor, reduction='max')
        if stop_tensor.item() > 0.0:
            break

    if accelerator.is_main_process and bool(getattr(cfg, 'save_final_checkpoint', True)):
        save_dict = {
            'epoch': cfg.epochs,
            'state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'best_high_rmse': best_high_rmse,
            'best_score': best_score
        }
        torch.save(save_dict, Path(cfg.save_path) / 'finetune_final.pth')
        
    accelerator.end_training()

if __name__ == '__main__':
    main()
