import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.sd2h import SD2H 
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def check_nan(layer, input, output):
    if isinstance(output, tuple):  
        output = output[1][-1]
    if torch.isnan(output).any():
        print(f"NaN detected in {layer.__class__.__name__}")

def check_nan_grad(layer, grad_input, grad_output):
    if isinstance(grad_input, tuple):  
        grad_input = grad_input[0]
    if torch.isnan(grad_input).any():
        print(f"NaN detected in gradient of {layer.__class__.__name__}")

def gray_2_colormap_np(img, cmap='rainbow', max_val=None, min_val=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    
    _min = min_val if min_val is not None else img.min()
    _max = max_val if max_val is not None else img.max()
    
    # Linearly map the disparity range to [0, 1].
    img_norm = (img - _min) / (_max - _min + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap_m = matplotlib.colormaps[cmap]
    map_ = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map_.to_rgba(img_norm)[:,:,:3]*255).astype(np.uint8)
    
    mask_invalid = img < (_min - 10)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, min_disp=-128, max_disp=64):
    """ Loss function defined over sequence of flow predictions """
    
    # =========================================================================
    # ULTIMATE DIMENSION DEFENSE (Fixing the 3D vs 4D Broadcast Bug)
    # =========================================================================
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    elif valid.dim() == 4 and valid.shape[1] > 1:
        valid = valid[:, 0:1, :, :]
    
    if disp_gt.dim() == 3:
        disp_gt = disp_gt.unsqueeze(1)
    elif disp_gt.dim() == 4 and disp_gt.shape[1] > 1:
        disp_gt = disp_gt[:, 0:1, :, :]
        
    if disp_init_pred.dim() == 3:
        disp_init_pred = disp_init_pred.unsqueeze(1)
    elif disp_init_pred.dim() == 4 and disp_init_pred.shape[1] > 1:
        disp_init_pred = disp_init_pred[:, 0:1, :, :]

    clean_disp_preds = []
    for d in disp_preds:
        if d.dim() == 3:
            clean_disp_preds.append(d.unsqueeze(1))
        elif d.dim() == 4 and d.shape[1] > 1:
            clean_disp_preds.append(d[:, 0:1, :, :])
        else:
            clean_disp_preds.append(d)
    disp_preds = clean_disp_preds
    # =========================================================================

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    valid = ((valid >= 0.5) & (disp_gt >= min_disp) & (disp_gt < max_disp))

    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum(torch.abs(disp_preds[-1] - disp_gt), dim=1)
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()
    
    total_loss = disp_loss
    
    metrics = {
        'train/epe': epe.mean(),
        'train/3px': (epe > 3).float().mean()*100,
        'train/5px': (epe > 5).float().mean()*100,
    }
    return total_loss, metrics

def fetch_optimizer(cfg, model, steps_per_epoch):
    """ Create the optimizer and learning rate scheduler based on Epochs """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': cfg.lr/2.0}, 
                   {'params': rest_params, 'lr': cfg.lr}, ]
                   
    # Use AdamW as specified in the paper
    optimizer = optim.AdamW(params_dict, lr=cfg.lr, weight_decay=cfg.wdecay, eps=1e-8)

    # Use StepLR based on Epochs
    if getattr(cfg, 'lr_scheduler', 'step') == 'step':
        # Scheduler steps per epoch (not per iteration)
        step_size_epochs = getattr(cfg, 'lr_step_size', 1) 
        gamma = getattr(cfg, 'lr_gamma', 0.96)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_epochs, gamma=gamma)
    else:
        # Fallback to OneCycleLR (needs total steps)
        total_steps = cfg.epochs * steps_per_epoch

        pct_start_val = getattr(cfg, 'pct_start', 0.05)          
        anneal_strategy_val = getattr(cfg, 'anneal_strategy', 'cos') 
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[cfg.lr/2.0, cfg.lr], 
            total_steps=total_steps+100,
            pct_start=pct_start_val,            
            cycle_momentum=False, 
            anneal_strategy=anneal_strategy_val  
        )

    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_pretrain')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # VRAM and precision optimization config
    accelerator = Accelerator(
        gradient_accumulation_steps=2,  
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), 
        log_with='wandb', 
        kwargs_handlers=[kwargs], 
        step_scheduler_with_optimizer=False
    )
    
    wandb_kwargs = getattr(cfg, 'wandb', {})
    if not isinstance(wandb_kwargs, dict):
        wandb_kwargs = OmegaConf.to_container(wandb_kwargs, resolve=True)
    
    wandb_kwargs['dir'] = cfg.save_path 
    accelerator.init_trackers(
        project_name=cfg.project_name, 
        config=OmegaConf.to_container(cfg, resolve=True), 
        init_kwargs={'wandb': wandb_kwargs}
    )
    
    # ========================== Train Loader ==========================
    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size//cfg.num_gpu, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=int(8), 
        drop_last=True
    )

    steps_per_epoch = len(train_loader)

    # ========================== Val Loader ==========================
    val_dataset = datasets.fetch_val_dataloader(cfg)
    
    if accelerator.is_main_process:
        print(f"\n[*] Train set loaded: {cfg.train_data_path} ({steps_per_epoch} steps/epoch)")
        print(f"[*] Val set loaded: {cfg.val_data_path} ({len(val_dataset)} samples)\n")
        
    if len(val_dataset) < accelerator.num_processes:
        raise ValueError(f"\n[FATAL ERROR] Val set size ({len(val_dataset)}) is smaller than GPU count ({accelerator.num_processes})!\n"
                         f"This causes GPU starvation. Please check val_data_path in YAML.")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(8), drop_last=False)
    # ================================================================

    # ====================== Resume initialization ======================
    model = SD2H(cfg)
    optimizer, lr_scheduler = fetch_optimizer(cfg, model, steps_per_epoch)
    
    start_epoch = 1
    total_step = 0

    if getattr(cfg, 'restore_ckpt', None) is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        if accelerator.is_main_process:
            print(f"[*] Loading checkpoint: {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        
        # 1. Parse checkpoint dictionaries from different save formats.
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        clean_ckpt = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_ckpt, strict=True)
        
        # 2. Resume training state when requested.
        if getattr(cfg, 'resume', False):
            if 'optimizer_state' in checkpoint and 'scheduler_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                total_step = checkpoint.get('total_step', 0)
                if accelerator.is_main_process:
                    print(f"[*] Resume enabled. Continuing from epoch {start_epoch}, step {total_step}.")
            else:
                if accelerator.is_main_process:
                    print("[!] resume=True was set, but optimizer state was not found. Starting as finetuning.")
        else:
            if accelerator.is_main_process:
                print(f"[*] Loaded weights from {cfg.restore_ckpt}; using them as finetuning initialization.")
                
        del clean_ckpt, checkpoint
    else:
        if accelerator.is_main_process:
            print("[*] No checkpoint provided. Training from scratch.")

    # ======================================================================

    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler, val_loader
    )
    model.to(accelerator.device)

    # ======================= Epoch-based Training Loop =======================
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        if hasattr(model, 'module'):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
        
        pbar = tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
        
        # --- Running training averages ---
        epoch_loss_sum = 0.0
        epoch_metrics_sum = {'epe': 0.0, '3px': 0.0, '5px': 0.0}
        steps_in_epoch = 0

        for data in pbar:
            _, left, right, disp_gt, valid = [x for x in data]
            
            with accelerator.accumulate(model):  
                with accelerator.autocast():  
                    disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
                
                loss, metrics = sequence_loss(
                    disp_preds, disp_init_pred, disp_gt, valid, 
                    min_disp=cfg.min_disp, 
                    max_disp=cfg.max_disp
                )
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if getattr(cfg, 'lr_scheduler', 'step') != 'step':
                lr_scheduler.step()

            total_step += 1
            loss_reduced = accelerator.reduce(loss.detach(), reduction='mean')
            metrics_reduced = accelerator.reduce(metrics, reduction='mean')
            
            # Accumulate metrics.
            steps_in_epoch += 1
            epoch_loss_sum += loss_reduced.item()
            epoch_metrics_sum['epe'] += metrics_reduced['train/epe'].item()
            epoch_metrics_sum['3px'] += metrics_reduced['train/3px'].item()
            epoch_metrics_sum['5px'] += metrics_reduced['train/5px'].item()
            
            accelerator.log({'train/loss': loss_reduced, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics_reduced, total_step)
            
            if accelerator.is_main_process:
                # Compute and display running averages.
                avg_loss = epoch_loss_sum / steps_in_epoch
                avg_epe = epoch_metrics_sum['epe'] / steps_in_epoch
                avg_3px = epoch_metrics_sum['3px'] / steps_in_epoch
                avg_5px = epoch_metrics_sum['5px'] / steps_in_epoch

                postfix_metrics = {
                    'Epoch': f"{epoch}/{cfg.epochs}", 
                    'AvgLoss': f"{avg_loss:.4f}",
                    'AvgEPE': f"{avg_epe:.3f}",
                    '3px': f"{avg_3px:.2f}%",
                    '5px': f"{avg_5px:.2f}%"
                }
                pbar.set_description(f"Pre-training [Epoch {epoch}]")
                pbar.set_postfix(postfix_metrics)

            #### Visualize
            if total_step % 100 == 0 and accelerator.is_main_process:
                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze(), min_val=cfg.min_disp, max_val=cfg.max_disp)
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze(), min_val=cfg.min_disp, max_val=cfg.max_disp)
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze(), min_val=cfg.min_disp, max_val=cfg.max_disp)
                
                accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption=f"Epoch {epoch}, Step {total_step}")}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption=f"Epoch {epoch}, Step {total_step}")}, total_step)
                accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption=f"Epoch {epoch}, Step {total_step}")}, total_step)

        if accelerator.is_main_process:
            final_avg_loss = epoch_loss_sum / steps_in_epoch
            final_avg_epe = epoch_metrics_sum['epe'] / steps_in_epoch
            final_avg_3px = epoch_metrics_sum['3px'] / steps_in_epoch
            print(f"[*] [Train Summary - Epoch {epoch}] Avg Loss: {final_avg_loss:.4f} | Avg EPE: {final_avg_epe:.3f} | Avg D1 (>3px): {final_avg_3px:.2f}%")

        if getattr(cfg, 'lr_scheduler', 'step') == 'step':
            lr_scheduler.step()

        # ======================= Checkpoint state saving =======================
        if epoch % getattr(cfg, 'save_epochs', 5) == 0 or epoch == cfg.epochs:
            if accelerator.is_main_process:
                save_path = Path(cfg.save_path + f'/epoch_{epoch}.pth')
                model_save = accelerator.unwrap_model(model)
                
                # Save all context required for resume training.
                save_dict = {
                    'epoch': epoch,
                    'total_step': total_step,
                    'model_state': model_save.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': lr_scheduler.state_dict(),
                    'cfg': OmegaConf.to_container(cfg, resolve=True)
                }
                torch.save(save_dict, save_path)
                del model_save

        # ======================= Epoch-based Validation =======================
        if epoch % getattr(cfg, 'val_epochs', 5) == 0:
            torch.cuda.empty_cache()
            model.eval()
            
            # --- Global validation accumulators ---
            total_val_epe = 0.0
            total_val_out = 0.0
            total_valid_pixels = 0
            
            pbar_val = tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
            for data in pbar_val:
                _, left, right, disp_gt, valid = [x for x in data]
                with torch.no_grad():
                    with accelerator.autocast():  
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    
                # -------------------------------------------------------------
                if disp_pred.dim() == 4 and disp_pred.shape[1] > 1:
                    disp_pred = disp_pred[:, 0:1, :, :]
                # -------------------------------------------------------------
                
                assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                
                valid_mask = (valid >= 0.5) & (disp_gt >= cfg.min_disp) & (disp_gt < cfg.max_disp)  
                
                # Extract errors on valid pixels.
                valid_epe = torch.abs(disp_pred - disp_gt)[valid_mask]
                valid_out = (valid_epe > 3.0).float()  

                # DDP safety: avoid gathering variable-length tensors across GPUs.
                if valid_epe.numel() == 0:
                    local_metrics = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device)
                else:
                    local_metrics = torch.tensor([
                        valid_epe.sum().item(), 
                        valid_out.sum().item(), 
                        float(valid_epe.numel())
                    ], device=accelerator.device)

                # Aggregate scalar sums across GPUs.
                gathered_metrics = accelerator.reduce(local_metrics, reduction='sum')

                # Global accumulation.
                total_val_epe += gathered_metrics[0].item()
                total_val_out += gathered_metrics[1].item()
                total_valid_pixels += gathered_metrics[2].item()
                
                if accelerator.is_main_process and total_valid_pixels > 0:
                    current_epe = total_val_epe / total_valid_pixels
                    current_d1 = 100.0 * total_val_out / total_valid_pixels
                    
                    postfix_val = {'Epoch': f"{epoch}", 'Global EPE': f"{current_epe:.3f}", 'Global D1': f"{current_d1:.2f}%"}
                    pbar_val.set_description(f"Validation [Epoch {epoch}]")
                    pbar_val.set_postfix(postfix_val)

            torch.cuda.empty_cache()

            # --- Validation summary and WandB logging ---
            if accelerator.is_main_process and total_valid_pixels > 0:
                final_val_epe = total_val_epe / total_valid_pixels
                final_val_d1 = 100.0 * total_val_out / total_valid_pixels
                
                print(f"[*] [Val Summary - Epoch {epoch}] Val EPE: {final_val_epe:.3f} | Val D1 (>3px): {final_val_d1:.2f}%")
                accelerator.log({'val/EPE': final_val_epe, 'val/D1': final_val_d1}, total_step)

    # Save final checkpoint after training.
    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        save_dict = {
            'epoch': cfg.epochs,
            'total_step': total_step,
            'model_state': model_save.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': lr_scheduler.state_dict(),
            'cfg': OmegaConf.to_container(cfg, resolve=True)
        }
        torch.save(save_dict, save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    main()
