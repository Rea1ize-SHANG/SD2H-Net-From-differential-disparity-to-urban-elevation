import os
import sys
import os.path as osp
from glob import glob
import logging

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.getcwd())
from core.utils.augmentor import FlowAugmentor

def read_image_as_float32(path):
    img = Image.open(path)
    img = np.array(img)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    elif img.shape[2] > 3:
        img = img[..., :3]
    return img.astype(np.float32)

def read_disp_as_float32(path):
    img = Image.open(path)
    disp = np.array(img).astype(np.float32)
    disp = np.nan_to_num(disp, nan=0.0) 
    return disp

class BaseStereoDataset(data.Dataset):
    def __init__(self, aug_params=None):
        self.augmentor = None
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)
            
        self.image_list = []
        self.disparity_list = []
        self.init_seed = False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        disp = read_disp_as_float32(self.disparity_list[index])
        img1 = read_image_as_float32(self.image_list[index][0])
        img2 = read_image_as_float32(self.image_list[index][1])

        # Flow is represented as [disp, 0].
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)

        # Build a valid-pixel mask.
        valid = (flow[0] >= -500) & (flow[0] <= 500)

        # Apply padding.
        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        # Keep the single-channel disparity.
        flow = flow[:1]
        
        # Disparity pretraining always returns five elements.
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

class WHUStereoDataset(BaseStereoDataset):
    def __init__(self, aug_params=None, data_path=''):
        super().__init__(aug_params)
        self.data_path = data_path

        left_dir = osp.join(self.data_path, 'left')
        right_dir = osp.join(self.data_path, 'right')
        disp_dir = osp.join(self.data_path, 'disp')

        left_files = sorted(glob(osp.join(left_dir, '*.*')))
        right_files = sorted(glob(osp.join(right_dir, '*.*')))
        disp_files = sorted(glob(osp.join(disp_dir, '*.*')))

        def extract_key(filename):
            base = osp.basename(filename)
            base, _ = osp.splitext(base)
            base = base.replace('_left_', '_').replace('_right_', '_').replace('_disparity_', '_').replace('_disp_', '_')
            if base.endswith('_left'): base = base[:-5]
            if base.endswith('_right'): base = base[:-6]
            if base.endswith('_disparity'): base = base[:-10]
            if base.endswith('_disp'): base = base[:-5]
            return base

        left_map = {extract_key(f): f for f in left_files if f.endswith(('.tif', '.tiff', '.png'))}
        right_map = {extract_key(f): f for f in right_files if f.endswith(('.tif', '.tiff', '.png'))}
        disp_map = {extract_key(f): f for f in disp_files if f.endswith(('.tif', '.tiff', '.png'))}

        keys = set(left_map.keys()) & set(right_map.keys()) & set(disp_map.keys())
        for k in sorted(keys):
            self.image_list.append([left_map[k], right_map[k]])
            self.disparity_list.append(disp_map[k])

class WHUSSIDEDataset(BaseStereoDataset):
    def __init__(self, aug_params=None, data_path=''):
        super().__init__(aug_params)
        self.data_path = data_path

        left_dir = osp.join(self.data_path, 'ImageL')
        right_dir = osp.join(self.data_path, 'ImageR')
        disp_dir = osp.join(self.data_path, 'Disp')

        left_files = sorted(glob(osp.join(left_dir, '*.*')))
        right_files = sorted(glob(osp.join(right_dir, '*.*')))
        disp_files = sorted(glob(osp.join(disp_dir, '*.*')))

        def extract_key(filename):
            base = osp.basename(filename)
            base, _ = osp.splitext(base)
            if base.startswith('ImgL'): base = base[4:]
            elif base.startswith('ImgR'): base = base[4:]
            elif base.startswith('Disp'): base = base[4:]
            return base

        left_map = {extract_key(f): f for f in left_files if f.endswith(('.tif', '.tiff', '.png'))}
        right_map = {extract_key(f): f for f in right_files if f.endswith(('.tif', '.tiff', '.png'))}
        disp_map = {extract_key(f): f for f in disp_files if f.endswith(('.tif', '.tiff', '.png'))}

        keys = set(left_map.keys()) & set(right_map.keys()) & set(disp_map.keys())
        for k in sorted(keys):
            self.image_list.append([left_map[k], right_map[k]])
            self.disparity_list.append(disp_map[k])

class WHUMixDispDataset(BaseStereoDataset):
    def __init__(self, aug_params=None, data_path=''):
        super().__init__(aug_params)
        self.data_path = data_path

        left_dir = osp.join(self.data_path, 'left') if os.path.exists(osp.join(self.data_path, 'left')) else osp.join(self.data_path, 'ImageL')
        right_dir = osp.join(self.data_path, 'right') if os.path.exists(osp.join(self.data_path, 'right')) else osp.join(self.data_path, 'ImageR')
        disp_dir = osp.join(self.data_path, 'disp') if os.path.exists(osp.join(self.data_path, 'disp')) else osp.join(self.data_path, 'Disp')

        left_files = sorted(glob(osp.join(left_dir, '*.*')))
        right_files = sorted(glob(osp.join(right_dir, '*.*')))
        disp_files = sorted(glob(osp.join(disp_dir, '*.*')))

        def extract_key(filename):
            base = osp.basename(filename)
            base, _ = osp.splitext(base)
            
            base = base.replace('_left_', '_').replace('_right_', '_').replace('_disparity_', '_').replace('_disp_', '_')
            if base.endswith('_left'): base = base[:-5]
            if base.endswith('_right'): base = base[:-6]
            if base.endswith('_disparity'): base = base[:-10]
            if base.endswith('_disp'): base = base[:-5]
            
            if base.startswith('ImgL'): base = base[4:]
            elif base.startswith('ImgR'): base = base[4:]
            elif base.startswith('Disp'): base = base[4:]
            
            return base

        left_map = {extract_key(f): f for f in left_files if f.endswith(('.tif', '.tiff', '.png'))}
        right_map = {extract_key(f): f for f in right_files if f.endswith(('.tif', '.tiff', '.png'))}
        disp_map = {extract_key(f): f for f in disp_files if f.endswith(('.tif', '.tiff', '.png'))}

        keys = set(left_map.keys()) & set(right_map.keys()) & set(disp_map.keys())
        for k in sorted(keys):
            self.image_list.append([left_map[k], right_map[k]])
            self.disparity_list.append(disp_map[k])

# ================= Dataset dispatch and dataloader helpers =================

def get_dataset_by_name(dataset_name, data_path, aug_params):
    name_clean = dataset_name.lower().strip()
    
    if name_clean == 'whustereo':
        return WHUStereoDataset(aug_params=aug_params, data_path=data_path)
    elif name_clean == 'whusside':
        return WHUSSIDEDataset(aug_params=aug_params, data_path=data_path)
    elif name_clean == 'whumixdisp':
        return WHUMixDispDataset(aug_params=aug_params, data_path=data_path)
    else:
        raise ValueError(f"[Stereo Dataset] Unsupported dataset name: {dataset_name}")

def fetch_dataloader(args):
    """Build the dataloader dataset for disparity pretraining."""
    aug_params = {}                
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range)
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    dataset_name = getattr(args, 'train_dataset_name', 'whustereo')
    train_data_path = getattr(args, 'train_data_path', '')

    train_dataset = get_dataset_by_name(dataset_name, train_data_path, aug_params)
    return train_dataset

def fetch_val_dataloader(args):
    """Build the validation dataset for disparity pretraining."""
    dataset_name = getattr(args, 'val_dataset_name', 'whustereo')
    val_data_path = getattr(args, 'val_data_path', '')
    
    val_dataset = get_dataset_by_name(dataset_name, val_data_path, aug_params={})
    return val_dataset
