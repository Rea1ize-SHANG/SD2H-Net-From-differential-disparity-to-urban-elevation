import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
from glob import glob
import os.path as osp
from pathlib import Path
import copy
import re

import sys
sys.path.append(os.getcwd())

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from PIL import Image


def read_image_as_float32(path):
    """读取tiff,单通道复制三通道,多通道取前三通道,转float32"""
    img = Image.open(path)
    img = np.array(img)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    elif img.shape[2] > 3:
        img = img[..., :3]
    img = img.astype(np.float32)
    return img


def read_disp_as_float32(path):
    """读取视差图,float32"""
    img = Image.open(path)
    img = np.array(img).astype(np.float32)
    return img


def read_label_as_uint8(path):
    """读取标签,uint8"""
    img = Image.open(path)
    img = np.array(img).astype(np.uint8)
    return img


class WHUStereo(data.Dataset):
    def __init__(self, aug_params=None, root='/home/shangying/workspace/dataset/WHUdataset/withGroundTruth', split='train'):
        self.root = root
        self.split = split
        self.augmentor = None
        self.sparse = False
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        split_path = osp.join(self.root, split)
        left_dir = osp.join(split_path, 'left')
        right_dir = osp.join(split_path, 'right')
        disp_dir = osp.join(split_path, 'disp')

        left_files = sorted(glob(osp.join(left_dir, '*.tiff')) + glob(osp.join(left_dir, '*.tif')))
        right_files = sorted(glob(osp.join(right_dir, '*.tiff')) + glob(osp.join(right_dir, '*.tif')))
        disp_files = sorted(glob(osp.join(disp_dir, '*.tiff')) + glob(osp.join(disp_dir, '*.tif')))

        def extract_key(filename):
            base = osp.basename(filename)
            base = base.replace('left', '').replace('right', '').replace('disparity', '')
            base = re.sub(r'[^a-zA-Z0-9]', '', base)
            return base

        left_map = {extract_key(f): f for f in left_files}
        right_map = {extract_key(f): f for f in right_files}
        disp_map = {extract_key(f): f for f in disp_files}

        keys = set(left_map.keys()) & set(right_map.keys()) & set(disp_map.keys())

        self.image_list = []
        self.disparity_list = []

        for k in sorted(keys):
            self.image_list.append([left_map[k], right_map[k]])
            self.disparity_list.append(disp_map[k])

        self.is_test = False
        self.init_seed = False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.is_test:
            img1 = read_image_as_float32(self.image_list[index][0])
            img2 = read_image_as_float32(self.image_list[index][1])
            img1 = torch.from_numpy(img1).permute(2, 0, 1)
            img2 = torch.from_numpy(img2).permute(2, 0, 1)
            return img1, img2, None

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        disp = read_disp_as_float32(self.disparity_list[index])
        valid = disp < 192                    #512

        img1 = read_image_as_float32(self.image_list[index][0])
        img2 = read_image_as_float32(self.image_list[index][1])

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)

        valid = (flow[0].abs() < 192) & (flow[1].abs() < 192)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


class US3D(data.Dataset):
    def __init__(self, aug_params=None, root='/home/fzq/workplacesy/dataset/us3d/JAX', split='train'):
        self.root = root
        self.split = split
        self.use_label = True
        self.augmentor = None
        self.sparse = False
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        split_path = osp.join(self.root, split)
        left_dir = osp.join(split_path, 'left')
        right_dir = osp.join(split_path, 'right')
        disp_dir = osp.join(split_path, 'disp')
        label_dir = osp.join(split_path, 'label')

        left_files = sorted(glob(osp.join(left_dir, '*.tiff')) + glob(osp.join(left_dir, '*.tif')))
        right_files = sorted(glob(osp.join(right_dir, '*.tiff')) + glob(osp.join(right_dir, '*.tif')))
        disp_files = sorted(glob(osp.join(disp_dir, '*.tiff')) + glob(osp.join(disp_dir, '*.tif')))
        label_files = sorted(glob(osp.join(label_dir, '*.tiff')) + glob(osp.join(label_dir, '*.tif')))

        def extract_key(filename):
            base = osp.basename(filename)
            base = base.replace('left', '').replace('right', '').replace('disparity', '').replace('label', '')
            base = re.sub(r'[^a-zA-Z0-9]', '', base)
            return base

        left_map = {extract_key(f): f for f in left_files}
        right_map = {extract_key(f): f for f in right_files}
        disp_map = {extract_key(f): f for f in disp_files}
        label_map = {extract_key(f): f for f in label_files}

        keys = set(left_map.keys()) & set(right_map.keys()) & set(disp_map.keys()) & set(label_map.keys())

        self.image_list = []
        self.disparity_list = []
        self.label_list = []

        for k in sorted(keys):
            self.image_list.append([left_map[k], right_map[k]])
            self.disparity_list.append(disp_map[k])
            self.label_list.append(label_map[k])

        self.is_test = False
        self.init_seed = False

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.is_test:
            img1 = read_image_as_float32(self.image_list[index][0])
            img2 = read_image_as_float32(self.image_list[index][1])
            img1 = torch.from_numpy(img1).permute(2, 0, 1)
            img2 = torch.from_numpy(img2).permute(2, 0, 1)
            label = read_label_as_uint8(self.label_list[index])
            label = torch.from_numpy(label).long()
            return img1, img2, label

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        disp = read_disp_as_float32(self.disparity_list[index])
        valid = disp < 512

        label = read_label_as_uint8(self.label_list[index])

        img1 = read_image_as_float32(self.image_list[index][0])
        img2 = read_image_as_float32(self.image_list[index][1])

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        label = torch.from_numpy(label).long()

        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
            label = F.pad(label.unsqueeze(0).float(), [padW]*2 + [padH]*2).long().squeeze(0)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index], self.label_list[index]], img1, img2, flow, label, valid.float()


def fetch_dataloader(args):
    aug_params = {}                 #'crop_size': list(args.image_size),'min_scale': args.spatial_scale[0],'max_scale': args.spatial_scale[1],'do_flip': False,'yjitter': not args.noyjitter
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range)
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name == 'whustereo':
            new_dataset = WHUStereo(aug_params=aug_params, split='train')
            logging.info(f"Added {len(new_dataset)} samples from WHUStereo")
        elif dataset_name == 'us3d':
            new_dataset = US3D(aug_params=aug_params, split='train')
            logging.info(f"Added {len(new_dataset)} samples from US3D")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    return train_dataset
