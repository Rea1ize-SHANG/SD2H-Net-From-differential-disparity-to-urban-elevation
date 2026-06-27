import sys
import os
import glob
import time
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('core')
from core.sd2h import SD2H
from core.utils.utils import InputPadder

DEVICE = 'cuda'

# ================= Color mapping with negative-disparity support =================
def gray_2_colormap_np(img, cmap='rainbow', max_val=None, min_val=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    
    # Treat invalid extreme values as background.
    mask_invalid = img < (min_val if min_val is not None else -500)
    
    _max = max_val if max_val is not None else img.max()
    _min = min_val if min_val is not None else img.min()
    
    # Normalize to [0, 1].
    img_norm = (img - _min) / (_max - _min + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap_m = matplotlib.colormaps[cmap]
    map_ = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    
    # Convert RGB color map to OpenCV BGR for video writing.
    colormap = (map_.to_rgba(img_norm)[:, :, :3] * 255).astype(np.uint8)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR) 
    colormap[mask_invalid] = 0

    return colormap


def load_image(imfile):
    """Robust image loading for single-channel, three-channel, or multi-channel images."""
    img = np.array(Image.open(imfile))
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    elif img.shape[2] > 3:
        img = img[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    # 1. Load model.
    model = SD2H(args)
    assert os.path.exists(args.restore_ckpt), f"[Error] Checkpoint not found: {args.restore_ckpt}"
    
    # Use weights_only=True to avoid PyTorch 2.x safety warnings.
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    # Remove an optional 'module.' prefix from DDP checkpoints.
    clean_ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    
    model.to(DEVICE)
    model.eval()

    # 2. Prepare output folder.
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    output_video_path = str(output_directory / 'satellite_disparity_demo.mp4')

    # 3. Collect images.
    left_images = sorted(glob.glob(os.path.join(args.left_imgs, '*.*')))
    right_images = sorted(glob.glob(os.path.join(args.right_imgs, '*.*')))
    
    valid_ext = ('.png', '.tif', '.tiff', '.jpg')
    left_images = [f for f in left_images if f.lower().endswith(valid_ext)][:args.max_frames]
    right_images = [f for f in right_images if f.lower().endswith(valid_ext)][:args.max_frames]

    print(f"[*] Found {len(left_images)} image pairs. Generating demo video...")
    
    if len(left_images) == 0:
        print("[!] No images found. Please check left_imgs and right_imgs.")
        return

    # 4. Determine video resolution. Vertical stacking doubles the height.
    sample_img = np.array(Image.open(left_images[0]))
    h, w = sample_img.shape[:2]
    videoWrite = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h * 2))

    # 5. Inference and video generation loop.
    with torch.no_grad():
        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            # Pad images so dimensions are divisible by 32.
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            
            start_time = time.time()
            
            # Use the current amp.autocast API.
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
                
            inference_time = time.time() - start_time
            
            disp = padder.unpad(disp)
            
            # Convert disparity to a color map.
            disp_color = gray_2_colormap_np(disp, max_val=args.max_disp, min_val=args.min_disp)
            
            # Read the original image as BGR background.
            img1_bgr = cv2.imread(imfile1)
            if img1_bgr is None:
                 img1_bgr = cv2.cvtColor(np.array(Image.open(imfile1))[..., :3], cv2.COLOR_RGB2BGR)

            # Stack original image on top and disparity color map below.
            out_img = np.concatenate((img1_bgr, disp_color), axis=0)
            
            # Draw inference time in the upper-left corner.
            cv2.putText(out_img, f"Inference: {inference_time*1000:.1f} ms", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            videoWrite.write(out_img)

    videoWrite.release()
    print(f"\n[*] Demo video saved to: {output_video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # ================= Core paths =================
    parser.add_argument('--restore_ckpt', required=True, help="Model checkpoint (.pth) path.")
    parser.add_argument('-l', '--left_imgs', required=True, help="Left-image folder.")
    parser.add_argument('-r', '--right_imgs', required=True, help="Right-image folder.")
    parser.add_argument('--output_directory', default="./satellite_demo_output", help="Output folder for the video.")
    
    # ================= Video parameters =================
    parser.add_argument('--fps', type=int, default=3, help="Video frame rate.")
    parser.add_argument('--max_frames', type=int, default=300, help="Maximum number of frames.")
    
    # ================= Model parameters =================
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed-precision inference.')
    parser.add_argument('--valid_iters', type=int, default=12, help='Number of inference iterations.')
    parser.add_argument('--min_disp', type=int, default=-320, help="Minimum disparity.")
    parser.add_argument('--max_disp', type=int, default=64, help="Maximum disparity.")

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=4)
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)
    
    args = parser.parse_args()
    demo(args)
