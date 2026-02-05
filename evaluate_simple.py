#!/usr/bin/env python3
"""
Simple evaluation script for iColoriT
Computes PSNR only (no LPIPS due to network/CUDA issues)
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

# Add parent directory to path
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, currentdir)
from utils import psnr


def get_args():
    parser = argparse.ArgumentParser('Simple Evaluation')
    parser.add_argument('--pred_dir', type=str, default='results/', help='prediction images directory')
    parser.add_argument('--gt_dir', type=str, default='validation/imgs/', help='ground truth images directory')
    parser.add_argument('--hint_size', type=int, default=2)
    parser.add_argument('--num_hint', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    args.pred_dir = osp.join(args.pred_dir, f'h{args.hint_size}-n{args.num_hint}')
    assert osp.isdir(args.pred_dir), f'{args.pred_dir} does not exist'
    
    args.save_path = osp.join(args.save_dir, f'h{args.hint_size}-n{args.num_hint}.txt')
    os.makedirs(osp.dirname(args.save_path), exist_ok=True)
    
    return args


class GtPredImageDataset(Dataset):
    """Dataset for ground truth and predicted images"""
    
    def __init__(self, gt_dir, pred_dir):
        super().__init__()
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        
        gt_files = sorted([
            f for f in os.listdir(self.gt_dir)
            if osp.splitext(f)[1].lower() in valid_extensions
        ])
        pred_files = sorted([
            f for f in os.listdir(self.pred_dir)
            if osp.splitext(f)[1].lower() in valid_extensions
        ])
        
        # Check that we have the same number of files
        assert len(gt_files) == len(pred_files), \
            f'Mismatch: {len(gt_files)} GT files vs {len(pred_files)} pred files'
        
        # Check that file names match (without extension)
        for gt_file, pred_file in zip(gt_files, pred_files):
            gt_name = osp.splitext(gt_file)[0]
            pred_name = osp.splitext(pred_file)[0]
            assert gt_name == pred_name, f'Name mismatch: {gt_file} vs {pred_file}'
        
        self.gt_files = gt_files
        self.pred_files = pred_files
        
        self.tf = Compose([
            Resize((224, 224)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.gt_files)
    
    def __getitem__(self, idx):
        gt_path = osp.join(self.gt_dir, self.gt_files[idx])
        pred_path = osp.join(self.pred_dir, self.pred_files[idx])
        
        gt = Image.open(gt_path).convert('RGB')
        pred = Image.open(pred_path).convert('RGB')
        
        gt = self.tf(gt)
        pred = self.tf(pred)
        
        name = osp.splitext(self.gt_files[idx])[0]
        return (gt, pred), name


def main(args):
    print("=" * 70)
    print("Simple Evaluation (PSNR only)")
    print("=" * 70)
    print(f"\nGT Directory:   {args.gt_dir}")
    print(f"Pred Directory: {args.pred_dir}")
    print(f"Hint Count:     {args.num_hint}")
    
    # Create dataset
    dataset = GtPredImageDataset(args.gt_dir, args.pred_dir)
    print(f"Found {len(dataset)} image pairs")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        drop_last=False,
        shuffle=False
    )
    
    # Evaluate
    total_psnr = 0.0
    total_images = 0
    
    print("\nEvaluating...")
    pbar = tqdm(desc='Evaluation', total=len(dataloader))
    
    with torch.no_grad():
        for batch in dataloader:
            (gts, preds), names = batch
            
            # Compute PSNR
            batch_psnr = psnr(preds, gts).item()
            batch_size = gts.size(0)
            
            total_psnr += batch_psnr * batch_size
            total_images += batch_size
            
            pbar.update()
    
    pbar.close()
    
    avg_psnr = total_psnr / total_images
    
    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Total Images: {total_images}")
    print("=" * 70)
    
    # Save results
    with open(args.save_path, 'w') as f:
        f.write(f'Evaluation Results\n')
        f.write(f'==================\n')
        f.write(f'Hint Count: {args.num_hint}\n')
        f.write(f'Total Images: {total_images}\n')
        f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
    
    print(f"\nResults saved to: {args.save_path}")


if __name__ == '__main__':
    args = get_args()
    main(args)
