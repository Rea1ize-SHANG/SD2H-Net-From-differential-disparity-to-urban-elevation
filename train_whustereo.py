import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
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


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path

def check_nan(layer, input, output):
    if isinstance(output, tuple):  # 检查是否为元组
        output = output[1][-1]
    if torch.isnan(output).any():
        print(f"NaN detected in {layer.__class__.__name__}")

def check_nan_grad(layer, grad_input, grad_output):
    if isinstance(grad_input, tuple):  # 检查是否为元组
        grad_input = grad_input[0]
    if torch.isnan(grad_input).any():
        print(f"NaN detected in gradient of {layer.__class__.__name__}")



def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps[cmap]
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, min_disp=-128, max_disp=64):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    #mag = torch.sum(disp_gt**2, dim=1).sqrt()       # mag.shape == [B, H, W]
    #valid = ((valid >= 0.5) & (mag >= min_disp) & (mag < max_disp)).unsqueeze(1)                    # 3.shape == [B, 1, H, W]
    valid = ((valid >= 0.5) & (disp_gt >= min_disp) & (disp_gt < max_disp))

    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum(torch.abs(disp_preds[-1] - disp_gt), dim=1)
    #epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/3px': (epe < 3).float().mean()*100,
        'train/5px': (epe < 5).float().mean()*100,
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_whustereo')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

    aug_params = {}
    val_dataset = datasets.WHUStereo(aug_params, split='val')          #, split='val'


    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)
    model = SD2H(cfg)
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]
        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    else:
        print("No checkpoint to load, training from scratch.")
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        active_train_loader = train_loader

        model.train()
        #model.module.freeze_bn()
        if hasattr(model, 'module'):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
        
        # 在训练循环里，先定义pbar
        pbar = tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
        for data in pbar:
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)
            
            if accelerator.is_main_process:
                postfix_metrics = {'Step': f"{total_step}", 'Loss': f"{loss.item():.4f}",}

                for k, v in metrics.items():
                    key_clean = k.replace('train/', '')  # 去掉train/前缀
                    postfix_metrics[key_clean] = f"{v.item():.2f}" if hasattr(v, 'item') else f"{v:.2f}%"

                pbar.set_description("")
                pbar.set_postfix(postfix_metrics)


            ####visualize the depth_mono and disp_preds
            if total_step % 20 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))


                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                
                accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save


            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                torch.cuda.empty_cache()
                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                pbar_val = tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)
                for data in pbar_val:
                    _, left, right, disp_gt, valid = [x for x in data]
                    #padder = InputPadder(left.shape, divis_by=32)
                    #left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    #disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    epe = torch.abs(disp_pred - disp_gt)
                    out = (epe < 3.0).float()        #3.0
                    epe = torch.squeeze(epe, dim=1)
                    out = torch.squeeze(out, dim=1)
                    # 计算 EPE 和 out 的部分
                    valid_mask = (valid >= 0.5) & (disp_gt.abs() >= cfg.min_disp) & (disp_gt.abs() < cfg.max_disp)  # 创建有效像素掩码
                    valid_epe = torch.abs(disp_pred - disp_gt)  # 计算每个像素的误差

                    # 展平并应用有效像素掩码
                    valid_epe = valid_epe[valid_mask]
                    valid_out = (valid_epe < 3.0).float()  # 计算低于阈值的像素

                    # 聚合结果
                    epe, out = accelerator.gather_for_metrics((valid_epe.mean(), valid_out.mean()))

                    # 累加结果
                    elem_num += valid_epe.shape[0]
                    for i in range(valid_epe.shape[0]):
                        total_epe += valid_epe[i]
                        total_out += valid_out[i]
                    
                    if accelerator.is_main_process and elem_num > 0:
                        postfix_val = {'Step': f"{total_step}", 'EPE': f"{(total_epe / elem_num):.2f}", 'D1': f"{100 * total_out / elem_num:.2f}%",}
                        
                        pbar_val.set_description("Validation")
                        pbar_val.set_postfix(postfix_val)

                model.train()
                if hasattr(model, 'module'):
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    main()