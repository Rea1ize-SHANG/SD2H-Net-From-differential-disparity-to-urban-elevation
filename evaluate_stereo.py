import os
import torch
import numpy as np
import warnings
from tqdm import tqdm
import skimage.io
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import contextlib
import io

# Suppress low-contrast warnings from skimage.
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

from core.sd2h import SD2H
from core.utils.utils import InputPadder
import core.stereo_datasets as datasets

# ================== Helper: globally consistent color mapping =================
def gray_2_colormap_np(img, valid_mask=None, cmap='turbo_r', vmin=-320, vmax=64, gamma=1.0):
    """
    Global fixed color mapping for long-tailed disparity maps.
    """
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().detach().numpy()

    img = img.squeeze()
    
    # 1. Invalid mask: use valid_mask for labels, otherwise filter extreme values only.
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        mask_invalid = (valid_mask < 0.5) | np.isnan(img) | (img < -500)
    else:
        mask_invalid = np.isnan(img) | (img < -500)
    
    # 2. Fixed global clipping keeps the color scale consistent across all models.
    img_clipped = np.clip(img, vmin, vmax)
    img_norm = (img_clipped - vmin) / (vmax - vmin + 1e-8)
    
    # 3. Gamma correction.
    if gamma != 1.0:
        img_norm = np.power(img_norm, gamma)

    # 4. Color mapping.
    cmap_m = matplotlib.colormaps[cmap]
    colormap = (cmap_m(img_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # 5. Mark background or invalid areas as white.
    colormap[mask_invalid] = [255, 255, 255] 
    
    return colormap


def _pick_left_path_from_info(info, batch_index, batch_size):
    """Extract the left image path from DataLoader-collated sample info."""
    if isinstance(info, dict):
        for key in ['left_filename', 'left_name', 'left', 'image1', 'img1', 'filename', 'name', 'path']:
            if key not in info:
                continue
            value = info[key]
            path = _pick_left_path_from_info(value, batch_index, batch_size)
            if path is not None:
                return path
        return None

    if isinstance(info, (list, tuple)):
        if len(info) == 0:
            return None

        # Default PyTorch collation turns [left, right, disp] into
        # [[left_0, left_1, ...], [right_0, ...], [disp_0, ...]].
        if isinstance(info[0], (list, tuple)):
            if len(info[0]) > batch_index and isinstance(info[0][batch_index], str):
                return info[0][batch_index]

            # Also handle the untransposed shape: [[left, right, disp], ...].
            if len(info) > batch_index and isinstance(info[batch_index], (list, tuple)):
                row = info[batch_index]
                if len(row) > 0 and isinstance(row[0], str):
                    return row[0]

        if isinstance(info[0], str):
            # A batch of plain left paths.
            if len(info) == batch_size and batch_index < len(info):
                return info[batch_index]

            # A single sample info list, usually [left, right, disp].
            for candidate in info:
                base = os.path.basename(candidate).lower()
                if base.startswith('imgl') or 'left' in base:
                    return candidate
            return info[0]

    if isinstance(info, str):
        return info

    return None


def _output_core_name_from_left_path(left_path):
    base_name = os.path.splitext(os.path.basename(left_path))[0]
    original_base_name = base_name

    for prefix in ['ImgLGF7_', 'ImgRGF7_', 'DispGF7_']:
        if base_name.startswith(prefix):
            return base_name[len(prefix):]

    for token in ['_left_', '_right_', '_disparity_', '_disp_']:
        base_name = base_name.replace(token, '_')

    for suffix in ['_left', '_right', '_disparity', '_disp']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    for prefix in ['left_', 'right_', 'disparity_', 'disp_']:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]

    if base_name.startswith('ImgL') or base_name.startswith('ImgR') or base_name.startswith('Disp'):
        base_name = base_name[4:].lstrip('_')

    return base_name or original_base_name


@torch.no_grad()
def evaluate_whustereo(model, val_loader, args):
    """ Evaluation for the Stereo dataset """
    model.eval()
    
    total_epe, total_sse, total_out_3px, total_out_5px, total_pixels = 0, 0, 0, 0, 0
    total_time = 0.0
    total_samples_for_time = 0  # Number of samples used for per-pair timing.

    output_dir = Path(args.output_directory)
    if args.save_vis:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Prediction results will be saved to: {output_dir.absolute()}")
    else:
        print("[*] Fast evaluation mode: metrics only, no images will be saved.")

    pbar_val = tqdm(val_loader, dynamic_ncols=True)
    for batch_idx, data in enumerate(pbar_val):
        info, left, right = data[0], data[1], data[2]
        disp_gt, valid = data[3], data[4]
        
        left, right = left.to(args.device), right.to(args.device)
        disp_gt = disp_gt.to(args.device)  
        valid = valid.to(args.device)  

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        # ====== Start timing ======
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        
        with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=torch.bfloat16):
            disp_pred = model(left, right, iters=args.valid_iters, test_mode=True)
            
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000.0 

        # Ignore the first warm-up batch.
        if batch_idx > 0: 
            total_time += curr_time
            total_samples_for_time += left.shape[0]  # Accumulate sample pairs in the current batch.

        disp_pred = padder.unpad(disp_pred)

        if disp_pred.ndim == 4:
            disp_pred = disp_pred.squeeze(1)  
        if disp_gt.ndim == 4:
            disp_gt = disp_gt.squeeze(1)
        if valid.ndim == 4:
            valid = valid.squeeze(1)

        assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
        
        epe = torch.abs(disp_pred - disp_gt)
        valid_mask = (valid >= 0.5) & (disp_gt >= args.min_disp) & (disp_gt < args.max_disp)
        
        valid_epe = epe[valid_mask]
        
        if valid_epe.numel() > 0:
            valid_out_3px = (valid_epe > 3.0).float()
            valid_out_5px = (valid_epe > 5.0).float()

            pixels_in_batch = valid_epe.numel()
            total_pixels += pixels_in_batch
            total_sse += (valid_epe ** 2).sum().item()
            total_epe += valid_epe.sum().item()
            total_out_3px += valid_out_3px.sum().item()
            total_out_5px += valid_out_5px.sum().item()

            postfix_val = {
                'EPE': f"{total_epe / total_pixels:.2f}", 
                '3px_Err': f"{100 * total_out_3px / total_pixels:.2f}%"
            }
            pbar_val.set_description("Testing")
            pbar_val.set_postfix(postfix_val)

        # ================== Save visualizations with aligned filenames ==================
        if args.save_vis:
            disp_preds_np = disp_pred.cpu().numpy()
            disp_gt_np = disp_gt.cpu().numpy() 
            valid_np = valid.cpu().numpy()     
            batch_size_current = disp_pred.shape[0]
            
            vis_vmin = args.min_disp
            vis_vmax = args.max_disp
            
            for b in range(batch_size_current):
                left_path = _pick_left_path_from_info(info, b, batch_size_current)
                if left_path is None:
                    left_path = f"batch{batch_idx:04d}_{b}.tif"

                core_name = _output_core_name_from_left_path(left_path)
                
                color_pred = gray_2_colormap_np(
                    disp_preds_np[b], 
                    valid_mask=None, 
                    cmap='turbo_r', 
                    vmin=vis_vmin, vmax=vis_vmax
                )
                pred_path = output_dir / f"Disp_{core_name}_color.png"
                skimage.io.imsave(str(pred_path), color_pred, check_contrast=False)
                
                color_pred_masked = gray_2_colormap_np(
                    disp_preds_np[b], 
                    valid_mask=valid_np[b], 
                    cmap='turbo_r', 
                    vmin=vis_vmin, vmax=vis_vmax
                )
                pred_masked_path = output_dir / f"Disp_Masked_{core_name}_color.png"
                skimage.io.imsave(str(pred_masked_path), color_pred_masked, check_contrast=False)

                # --- Ground truth ---
                color_gt = gray_2_colormap_np(
                    disp_gt_np[b], 
                    valid_mask=valid_np[b], 
                    cmap='turbo_r', 
                    vmin=vis_vmin, vmax=vis_vmax
                )
                gt_path = output_dir / f"GT_{core_name}_color.png"
                skimage.io.imsave(str(gt_path), color_gt, check_contrast=False)

    # ================= Summary statistics =================
    avg_epe = total_epe / total_pixels if total_pixels > 0 else 0
    err_3px = (total_out_3px / total_pixels) * 100 if total_pixels > 0 else 0
    err_5px = (total_out_5px / total_pixels) * 100 if total_pixels > 0 else 0
    rmse = (total_sse / total_pixels) ** 0.5 if total_pixels > 0 else 0
    
    # Recompute accurate per-pair inference time (ms/pair).
    avg_time_ms = (total_time / total_samples_for_time * 1000) if total_samples_for_time > 0 else 0.0

    print("\n" + "="*50)
    print(f"[*] Evaluation finished. Iterations: {args.valid_iters} | mixed precision: {args.mixed_precision}")
    print("-" * 50)
    print(f"Average EPE      : {avg_epe:.4f} px") 
    print(f"3-pixel Error    : {err_3px:.2f} %")  
    print(f"5-pixel Error    : {err_5px:.2f} %")
    print(f"RMSE             : {rmse:.4f} px") 
    print("-" * 50)
    # Compact model-complexity summary.
    print(f"Model Params (Total)       : {args.params_total:.3f} M")
    print(f"Frozen DepthAnythingV2     : {args.params_mono:.3f} M")
    print(f"Trainable Stereo/Fusion    : {args.params_stereo:.3f} M")
    print(f"Inference Time             : {avg_time_ms:.1f} ms / pair")
    if hasattr(args, 'flops_g') and args.flops_g > 0:
        print(f"Model FLOPs                : {args.flops_g:.3f} G")
    print("="*50 + "\n")

    return {'epe': avg_epe, 'rmse': rmse, '3px_Error': f"{err_3px:.2f}%"}


def main(args):
    assert os.path.exists(args.restore_ckpt), f"Checkpoint path {args.restore_ckpt} doesn't exist"
    
    print(f"[*] Reading checkpoint: {args.restore_ckpt}")
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False) 
    
    ckpt_dict = dict()
    state_dict_raw = checkpoint.get('state_dict', checkpoint) 
    for key in state_dict_raw:
        ckpt_dict[key.replace('module.', '')] = state_dict_raw[key]

    if 'args' in checkpoint or 'cfg' in checkpoint:
        saved_cfg = checkpoint.get('args', checkpoint.get('cfg'))
        key_params = ['encoder', 'corr_levels', 'hidden_dims', 'n_gru_layers']
        for k in key_params:
            if hasattr(saved_cfg, k):
                setattr(args, k, getattr(saved_cfg, k))
            elif isinstance(saved_cfg, dict) and k in saved_cfg:
                setattr(args, k, saved_cfg[k])

    inferred_encoder = None
    for k, v in ckpt_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) == 4:
            embed_dim = v.shape[0]
            if embed_dim == 384: inferred_encoder = 'vits'
            elif embed_dim == 768: inferred_encoder = 'vitb'
            elif embed_dim == 1024: inferred_encoder = 'vitl'
            elif embed_dim == 1536: inferred_encoder = 'vitg'
            break
            
    if inferred_encoder and args.encoder != inferred_encoder:
        args.encoder = inferred_encoder

    # Force the dimension-alignment setting for WHU-SSIDE checkpoints.
    if "724" in args.restore_ckpt or "pretrain_whusside" in args.restore_ckpt:
        args.shared_backbone = True

    print(f"[*] Building model (Encoder: {args.encoder}, Mono: {args.use_mono}, Stereo: {args.use_stereo}) ...")
    
    model = SD2H(args)
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    print("[*] Matched and loaded checkpoint weights.")

    model.to(args.device)
    model.eval()

    # ================= Model parameters and FLOPs =================
    args.params_total = sum(p.numel() for p in model.parameters()) / 1e6
    mono_prefixes = ('mono_encoder', 'mono_decoder', 'feat_decoder')
    args.params_mono = sum(p.numel() for n, p in model.named_parameters() if n.startswith(mono_prefixes)) / 1e6
    args.params_stereo = args.params_total - args.params_mono
    # Keep these values for downstream code that may still rely on them.
    args.params_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    args.params_trainable = args.params_total - args.params_frozen

    args.flops_g = 0.0
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_left = torch.randn(1, 3, 1024, 1024).to(args.device)
        dummy_right = torch.randn(1, 3, 1024, 1024).to(args.device)
        with contextlib.redirect_stdout(io.StringIO()):
            flops_analyser = FlopCountAnalysis(model, (dummy_left, dummy_right, args.valid_iters, True))
            args.flops_g = flops_analyser.total() / 1e9
    except Exception:
        print("[!] FLOPs computation skipped for the current network structure.")

    # ================= Load evaluation data =================
    print(f"[*] Loading evaluation data: {args.data_path} ({args.dataset})")
    try:
        val_dataset = datasets.get_dataset_by_name(args.dataset, args.data_path, aug_params={})
    except Exception as e:
        raise ValueError(f"[Error] Failed to load dataset: {args.dataset}. Details: {e}")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    results = evaluate_whustereo(model, val_loader, args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SD2H Test Evaluation Script")
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Default output path.
    parser.add_argument('--output_directory', default="/home/shangying/workspace/output/disp/sd2h/whusside/valtest")
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--restore_ckpt', required=True)
    parser.add_argument('--dataset', type=str, default='whustereo')
    parser.add_argument('--data_path', required=True) 
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--encoder', type=str, default='vitb')
    parser.add_argument('--use_mono', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_stereo', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--valid_iters', type=int, default=16)
    parser.add_argument('--min_disp', type=int, default=-320) 
    parser.add_argument('--max_disp', type=int, default=64)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=4)
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)

    args = parser.parse_args()
    main(args)
