#!/usr/bin/env python3
"""
Convert color images to grayscale using LAB colorspace.

This script converts images to grayscale using the same LAB-based approach
that iColoriT uses internally. The L (lightness) channel in LAB colorspace
represents the grayscale information.

Usage:
    python scripts/convert_to_grayscale.py --input_dir path/to/images --output_dir path/to/output
    python scripts/convert_to_grayscale.py --input_file path/to/image.jpg --output_dir path/to/output

Examples:
    # Convert all images in a directory
    python scripts/convert_to_grayscale.py --input_dir Test/imgs --output_dir Test/grayscale

    # Convert a single image
    python scripts/convert_to_grayscale.py --input_file Test/imgs/egypt_01.jpg --output_dir Test/grayscale
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_xyz(rgb):
    """Convert RGB to XYZ colorspace."""
    # sRGB to linear RGB
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # RGB to XYZ matrix (sRGB D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    xyz = np.dot(rgb_linear, matrix.T)
    return xyz


def xyz_to_lab(xyz):
    """Convert XYZ to LAB colorspace."""
    # D65 white point
    ref_white = np.array([0.95047, 1.0, 1.08883])

    xyz_norm = xyz / ref_white

    # f(t) function
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz_norm > epsilon
    f_xyz = np.where(mask, np.cbrt(xyz_norm), (kappa * xyz_norm + 16) / 116)

    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab):
    """Convert LAB to XYZ colorspace."""
    # D65 white point
    ref_white = np.array([0.95047, 1.0, 1.08883])

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    # Inverse f(t)
    x = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)

    xyz = np.stack([x, y, z], axis=-1) * ref_white
    return xyz


def xyz_to_rgb(xyz):
    """Convert XYZ to RGB colorspace."""
    # XYZ to RGB matrix (sRGB D65)
    matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    rgb_linear = np.dot(xyz, matrix.T)

    # Clip negative values
    rgb_linear = np.clip(rgb_linear, 0, None)

    # Linear RGB to sRGB
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1 / 2.4)) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb, 0, 1)


def rgb_to_grayscale_lab(img_rgb):
    """
    Convert RGB image to grayscale using LAB colorspace.

    This extracts the L (lightness) channel from LAB colorspace,
    which is how iColoriT internally represents grayscale.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3), values in [0, 255]

    Returns:
        Grayscale image as numpy array (H, W, 3), RGB format, uint8
    """
    # Normalize to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0

    # Convert RGB to LAB
    xyz = rgb_to_xyz(img_float)
    lab = xyz_to_lab(xyz)

    # Create grayscale LAB image (L channel only, ab = 0)
    lab_gray = np.zeros_like(lab)
    lab_gray[..., 0] = lab[..., 0]
    # a and b channels stay at 0 (neutral gray)

    # Convert back to RGB
    xyz_gray = lab_to_xyz(lab_gray)
    rgb_gray = xyz_to_rgb(xyz_gray)

    # Convert to uint8
    img_gray_rgb = (rgb_gray * 255).astype(np.uint8)

    return img_gray_rgb


def rgb_to_grayscale_simple(img_rgb):
    """
    Convert RGB image to grayscale using standard luminance weights.

    Args:
        img_rgb: RGB image as numpy array (H, W, 3)

    Returns:
        Grayscale image as numpy array (H, W, 3), RGB format
    """
    # Standard luminance weights (ITU-R BT.601)
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(img_rgb[..., :3].astype(np.float32), weights)

    # Stack to create 3-channel grayscale
    img_gray = np.stack([gray, gray, gray], axis=-1)

    return img_gray.astype(np.uint8)


def convert_image(input_path, output_path, method='lab'):
    """
    Convert a single image to grayscale.

    Args:
        input_path: Path to input image
        output_path: Path to save output image
        method: 'lab' for LAB-based conversion, 'simple' for luminance-based
    """
    # Load image
    img = Image.open(input_path).convert('RGB')
    img_rgb = np.array(img)

    # Convert to grayscale
    if method == 'lab':
        img_gray = rgb_to_grayscale_lab(img_rgb)
    else:
        img_gray = rgb_to_grayscale_simple(img_rgb)

    # Save
    Image.fromarray(img_gray).save(output_path)


def is_image_file(filename):
    """Check if file is an image based on extension."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in extensions


def main():
    parser = argparse.ArgumentParser(
        description='Convert images to grayscale using LAB colorspace (iColoriT compatible)')

    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing input images')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Single input image file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save grayscale images')
    parser.add_argument('--method', type=str, default='lab', choices=['lab', 'simple'],
                        help='Conversion method: "lab" (LAB colorspace, default) or "simple" (luminance)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to add to output filenames (e.g., "_gray")')
    parser.add_argument('--format', type=str, default=None,
                        help='Output format (e.g., "png", "jpg"). If None, keeps original format')

    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        parser.error('Either --input_dir or --input_file must be specified')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of images to process
    if args.input_file:
        image_files = [Path(args.input_file)]
    else:
        input_dir = Path(args.input_dir)
        image_files = [f for f in input_dir.iterdir() if is_image_file(f.name)]

    print(f'Found {len(image_files)} images to convert')
    print(f'Using {args.method.upper()} conversion method')

    for img_path in sorted(image_files):
        # Determine output filename
        stem = img_path.stem + args.suffix
        ext = f'.{args.format}' if args.format else img_path.suffix
        output_path = Path(args.output_dir) / f'{stem}{ext}'

        try:
            convert_image(str(img_path), str(output_path), method=args.method)
            print(f'Converted: {img_path.name} -> {output_path.name}')
        except Exception as e:
            print(f'Error processing {img_path.name}: {e}')

    print(f'\nDone! Grayscale images saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
