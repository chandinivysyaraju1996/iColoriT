#!/usr/bin/env python3
"""
Pre-compute random hints for images
Usage: python3 precompute_hints.py --img_dir /Users/chandinivysyaraju/Documents/Thesis/iColoriT-main/Test/grayscale/imgs --output_dir hints_grayscale/
"""

import argparse
import os
import os.path as osp
import numpy as np
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser('Pre-compute Hints')
    parser.add_argument('--img_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for hints')
    parser.add_argument('--hint_size', type=int, default=2, help='Hint region size')
    parser.add_argument('--num_hints', type=int, nargs='+', default=[0, 1, 2, 5, 10, 20, 50, 100, 200],
                        help='Number of hints to generate')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


def get_image_files(img_dir):
    """Get all image files from directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if osp.splitext(f)[1].lower() in valid_extensions
    ])
    return img_files


def generate_hints(num_patches, num_hint, seed=None):
    """
    Generate random hint mask

    Args:
        num_patches: Total number of patches (h*w where h,w = input_size/patch_size)
        num_hint: Number of hints to generate
        seed: Random seed for reproducibility

    Returns:
        hint_mask: Binary array [num_patches] where 0=hint, 1=no hint
        hint_coords: List of (x, y) coordinates for hints
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random hint mask
    hint_mask = np.hstack([
        np.ones(num_patches - num_hint),
        np.zeros(num_hint),
    ])
    np.random.shuffle(hint_mask)

    # Convert mask to coordinates
    hint_coords = np.where(hint_mask == 0)[0]

    return hint_mask, hint_coords


def mask_to_coordinates(hint_mask, hint_size, input_size):
    """
    Convert hint mask to pixel coordinates

    Args:
        hint_mask: Binary mask at patch level [num_patches]
        hint_size: Size of each patch (e.g., 2 for 2x2)
        input_size: Input image size (e.g., 224)

    Returns:
        coords: List of (x, y) pixel coordinates
    """
    num_patches_h = input_size // hint_size
    num_patches_w = input_size // hint_size

    coords = []
    hint_idx = 0
    for h in range(num_patches_h):
        for w in range(num_patches_w):
            if hint_mask[hint_idx] == 0:  # 0 means hint location
                # Convert patch index to pixel coordinates
                x = h * hint_size + hint_size // 2
                y = w * hint_size + hint_size // 2
                coords.append((x, y))
            hint_idx += 1

    return coords


def save_hints(output_file, coords):
    """Save hint coordinates to file"""
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for x, y in coords:
            f.write(f'{x} {y}\n')


def main(args):
    print("=" * 70)
    print("Pre-Computing Hints for iColoriT")
    print("=" * 70)

    # Get image files
    img_files = get_image_files(args.img_dir)
    print(f"\nFound {len(img_files)} images in {args.img_dir}")

    if len(img_files) == 0:
        print("❌ No images found!")
        return

    # Calculate number of patches
    num_patches = (args.input_size // args.hint_size) ** 2
    print(f"Input size: {args.input_size}×{args.input_size}")
    print(f"Hint size: {args.hint_size}×{args.hint_size}")
    print(f"Number of patch locations: {num_patches}")

    # Pre-compute hints for each num_hint value
    for num_hint in sorted(args.num_hints):
        print(f"\n{'─' * 70}")
        print(f"Pre-computing hints with {num_hint} hints per image...")
        print(f"{'─' * 70}")

        output_subdir = osp.join(args.output_dir, f'h{args.hint_size}-n{num_hint}')
        os.makedirs(output_subdir, exist_ok=True)

        # Generate hints for each image
        for img_idx, img_file in enumerate(img_files):
            img_name = osp.splitext(img_file)[0]
            output_file = osp.join(output_subdir, f'{img_name}.txt')

            # Generate random hints (use image index as seed for reproducibility)
            hint_mask, hint_indices = generate_hints(num_patches, num_hint, seed=args.seed + img_idx)

            # Convert to pixel coordinates
            coords = mask_to_coordinates(hint_mask, args.hint_size, args.input_size)

            # Save hints
            save_hints(output_file, coords)

            if (img_idx + 1) % max(1, len(img_files) // 10) == 0 or img_idx == 0:
                print(f"  [{img_idx + 1}/{len(img_files)}] {img_name}: {num_hint} hints")

        print(f"✅ Saved {len(img_files)} hint files to {output_subdir}")

    print("\n" + "=" * 70)
    print("✅ Hint pre-computation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Print summary
    print("\nGenerated hint directories:")
    for num_hint in sorted(args.num_hints):
        hint_dir = osp.join(args.output_dir, f'h{args.hint_size}-n{num_hint}')
        if osp.exists(hint_dir):
            num_files = len(os.listdir(hint_dir))
            print(f"  ✅ {hint_dir}: {num_files} files")


if __name__ == '__main__':
    args = get_args()
    main(args)
