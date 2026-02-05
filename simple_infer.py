#!/usr/bin/env python3
"""
Simple inference script for iColoriT - works with any images
Generates random hints for colorization
"""

import argparse
import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torchvision
from einops import rearrange
from timm.models import create_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

import modeling
from utils import lab2rgb, psnr, rgb2lab, seed_worker
from torchvision.transforms import Compose, Resize, ToTensor


class SimpleImageDataset(Dataset):
    """Simple dataset for any images"""
    
    def __init__(self, img_dir, input_size=224):
        self.img_dir = img_dir
        self.input_size = input_size
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        self.img_list = sorted([
            f for f in os.listdir(img_dir)
            if osp.splitext(f)[1].lower() in valid_extensions
        ])
        
        self.transform = Compose([
            Resize((input_size, input_size)),
            ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, self.img_list[idx]


def get_args():
    parser = argparse.ArgumentParser('Simple iColoriT Inference')
    parser.add_argument('--model_path', type=str, required=True, help='checkpoint path')
    parser.add_argument('--val_data_path', type=str, required=True, help='validation dataset path')
    parser.add_argument('--pred_dir', type=str, default='results/', help='output directory')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--num_hints', default=10, type=int, help='number of hints')
    parser.add_argument('--hint_size', default=2, type=int, help='hint region size')
    
    args = parser.parse_args()
    return args


def get_model(args):
    print(f"Creating model: icolorit_base_4ch_patch16_224")
    model = create_model(
        'icolorit_base_4ch_patch16_224',
        pretrained=False,
        drop_path_rate=0.0,
        drop_block_rate=None,
        use_rpb=True,
        avg_hint=True,
        head_mode='cnn',
        mask_cent=False,
    )
    return model


def generate_random_hints(num_patches, num_hints, hint_size=2):
    """Generate random hint mask"""
    num_hint_locations = num_patches
    hint = np.hstack([
        np.ones(num_hint_locations - num_hints),
        np.zeros(num_hints),
    ])
    np.random.shuffle(hint)
    return hint


def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # Load model
    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print(f"Patch size = {patch_size}")
    
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create dataset
    dataset = SimpleImageDataset(args.val_data_path, args.input_size)
    print(f"Found {len(dataset)} images")
    
    if len(dataset) == 0:
        print(f"❌ No images found in {args.val_data_path}")
        return
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        shuffle=False,
    )
    
    # Create output directories
    output_dir = osp.join(args.pred_dir, f'h{args.hint_size}-n{args.num_hints}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Inference
    psnr_sum = 0.0
    total_shown = 0
    
    print("\n" + "=" * 70)
    print(f"🎨 Running inference with {args.num_hints} hints...")
    print("=" * 70 + "\n")
    
    with torch.no_grad():
        pbar = tqdm(desc='Colorizing', ncols=100, total=len(data_loader))
        for batch_idx, (images, names) in enumerate(data_loader):
            B, _, H, W = images.shape
            h, w = H // patch_size[0], W // patch_size[1]
            
            # Prepare batch
            images = images.to(device, non_blocking=True)
            images_lab = rgb2lab(images)
            images_patch = rearrange(
                images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_size[0], p2=patch_size[1]
            )
            labels = rearrange(
                images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c',
                p1=patch_size[0], p2=patch_size[1]
            )
            
            # Generate random hints
            num_patches = h * w
            bool_hints = []
            for _ in range(B):
                hint = generate_random_hints(num_patches, args.num_hints, args.hint_size)
                bool_hints.append(hint)
            bool_hint = torch.from_numpy(np.array(bool_hints)).to(device).to(torch.bool)
            
            # Inference
            with torch.cuda.amp.autocast():
                outputs = model(images_lab.clone(), bool_hint.clone())
                outputs = rearrange(
                    outputs, 'b n (p1 p2 c) -> b n (p1 p2) c',
                    p1=patch_size[0], p2=patch_size[1]
                )
            
            # Reconstruct image
            pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
            pred_imgs_lab = rearrange(
                pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                h=h, w=w, p1=patch_size[0], p2=patch_size[1]
            )
            pred_imgs = lab2rgb(pred_imgs_lab)
            
            # Compute PSNR
            psnr_val = psnr(images, pred_imgs).item() * B
            psnr_sum += psnr_val
            total_shown += B
            
            # Save results
            for name, pred_img in zip(names, pred_imgs):
                output_path = osp.join(output_dir, osp.splitext(name)[0] + '.png')
                torchvision.utils.save_image(pred_img.unsqueeze(0), output_path)
            
            pbar.update()
            pbar.set_postfix({'PSNR': f'{psnr_sum / total_shown:.2f}'})
        
        pbar.close()
    
    avg_psnr = psnr_sum / total_shown
    print("\n" + "=" * 70)
    print(f"✅ Inference completed!")
    print(f"📊 Average PSNR: {avg_psnr:.2f} dB")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📈 Processed {total_shown} images")
    print("=" * 70)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    main(args)
