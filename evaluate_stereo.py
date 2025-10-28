import os
import torch
import numpy as np
from tqdm import tqdm
import skimage.io
from pathlib import Path
from core.sd2h import SD2H
from core.utils.utils import InputPadder
import core.stereo_datasets as datasets

# Test evaluation function
# Test evaluation function
# Test evaluation function
# Test evaluation function
@torch.no_grad()
def evaluate_whustereo(model, val_loader, args):
    """ Evaluation for the WHUStereo dataset """
    model.eval()
    total_epe, total_out, total_images = 0, 0, 0

    pbar_val = tqdm(val_loader, dynamic_ncols=True)
    for data in pbar_val:
        _, left, right, disp_gt, valid = [x for x in data]
        
        # Move input data to the same device as the model
        left, right = left.to(args.device), right.to(args.device)
        disp_gt = disp_gt.to(args.device)  # Ensure disp_gt is on the same device
        valid = valid.to(args.device)  # Ensure valid is also on the same device

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        # Perform inference
        disp_pred = model(left, right, iters=args.valid_iters, test_mode=True)
        disp_pred = padder.unpad(disp_pred)

        # Ensure the shape matches
        assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
        
        # Calculate EPE (End-Point Error)
        epe = torch.abs(disp_pred - disp_gt)
        valid_mask = (valid >= 0.5) & (disp_gt.abs() >= args.min_disp) & (disp_gt.abs() < args.max_disp)
        
        valid_epe = epe[valid_mask]
        valid_out = (valid_epe < 3.0).float()

        # Gather results
        epe, out = valid_epe.mean(), valid_out.mean()
        total_images += valid_epe.shape[0]
        total_epe += epe.item() * valid_epe.shape[0]
        total_out += out.item() * valid_epe.shape[0]

        # Update the progress bar
        postfix_val = {'EPE': f"{total_epe / total_images:.2f}", 'D1': f"{100 * total_out / total_images:.2f}%"}
        pbar_val.set_description("Validation")
        pbar_val.set_postfix(postfix_val)

    avg_epe = total_epe / total_images
    d1_accuracy = (total_out / total_images) * 100

    print(f"Validation completed: Average EPE = {avg_epe:.2f}, D1 accuracy = {d1_accuracy:.2f}%")

    return {'epe': avg_epe, 'd1': f"{d1_accuracy:.2f}%"}

# Test evaluation main function
def main(args):
    # Load the model
    model = SD2H(args)
    assert os.path.exists(args.restore_ckpt), f"Checkpoint path {args.restore_ckpt} doesn't exist"
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        ckpt[key.replace('module.', '')] = checkpoint[key]
    model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint from {args.restore_ckpt}")

    # Prepare the model and data for testing
    model.to(args.device)
    model.eval()

    # Prepare dataset
    val_dataset = datasets.WHUStereo(split='test')  # Assuming you have a val split
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Perform the evaluation
    results = evaluate_whustereo(model, val_loader, args)
    print(f"Final evaluation results: {results}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument('--restore_ckpt', default='/home/fzq/workplacesy/code/SDH-Net/checkpoints/whustereo/45000.pth', help="Checkpoint file to restore")
    parser.add_argument('--output_directory', help="Directory to save the output", default="./outputs")
    parser.add_argument('--valid_iters', type=int, default=8, help='Number of iterations for disparity prediction')
    parser.add_argument('--min_disp', type=int, default=-128, help="Minimum disparity")
    parser.add_argument('--max_disp', type=int, default=64, help="Maximum disparity")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for validation")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    
    # Setting GPU for evaluation
    parser.add_argument('--device', type=str, default='cuda:7', help="Device to run the model on (e.g., 'cuda:0')")

    args = parser.parse_args()
    main(args)