import os
import glob
import time
import argparse
import numpy as np
import torch
import skimage.io
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from core.sd2h import SD2H
from core.utils.utils import InputPadder

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return torch.from_numpy(img).permute(2,0,1).float()[None].to(DEVICE)

def collect_pairs(test_dir):
    left_dir  = Path(test_dir) / 'left'
    right_dir = Path(test_dir) / 'right'
    pairs = []
    for left_fp in sorted(left_dir.glob('*.tif')):
        stem = left_fp.stem  # e.g. "left_10_1"
        if stem.startswith('left_'):
            right_stem = 'right_' + stem[len('left_'):]
        else:
            right_stem = stem
        right_fp = right_dir / f"{right_stem}.tif"
        if not right_fp.exists():
            raise FileNotFoundError(f"Missing right image for {left_fp.name}: expected {right_fp.name}")
        pairs.append((str(left_fp), str(right_fp)))
    return pairs

def demo(args):
    model = torch.nn.DataParallel(SD2H(args), device_ids=[0])
    assert os.path.exists(args.restore_ckpt), f"Checkpoint not found: {args.restore_ckpt}"
    ckpt = torch.load(args.restore_ckpt, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    model.load_state_dict({f"module.{k}": v for k, v in sd.items()}, strict=True)
    model = model.module.to(DEVICE)
    model.eval()

    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.test_data_dir)
    print(f"Found {len(pairs)} pairs, saving to {out_dir}")

    with torch.no_grad():
        pbar = tqdm(pairs, desc='Inference', unit='pair', ncols=100)
        for left_fp, right_fp in pbar:
            im1, im2 = load_image(left_fp), load_image(right_fp)
            padder = InputPadder(im1.shape, divis_by=32)
            im1, im2 = padder.pad(im1, im2)

            t0 = time.time()
            disp = model(im1, im2, iters=args.valid_iters, test_mode=True)
            pbar.set_postfix({'time': f"{time.time()-t0:.3f}s"})

            disp = padder.unpad(disp)[0,0]
            disp_np = disp.cpu().numpy().astype(np.float32)

            stem = Path(left_fp).stem
            out_tif = out_dir / f"{stem}_disp.tiff"
            skimage.io.imsave(str(out_tif), disp_np)

            if args.save_numpy:
                out_npy = out_dir / f"{stem}.npy"
                np.save(str(out_npy), disp_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default="/home/shangying/workspace/project/SD2H-Net/checkpoints/whustereo/42500.pth")
    parser.add_argument('--test_data_dir', default="/home/shangying/workspace/dataset/HZSZDG")
    parser.add_argument('--output_directory', default="/home/shangying/workspace/project/SD2H-Net/outputs/save_images")
    parser.add_argument('--save_numpy', action='store_true')
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--encoder', choices=['vits','vitb','vitl','vitg'], default='vitb')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128,128,128])
    parser.add_argument('--corr_implementation', choices=["reg","alt","reg_cuda","alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=64)
    parser.add_argument('--min_disp', type=int, default=-128)
    args = parser.parse_args()
    demo(args)