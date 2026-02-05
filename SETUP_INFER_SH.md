# How to Setup and Run `infer.sh`

This guide explains how to pre-compute hints and configure `scripts/infer.sh` to work with your images.

---

## Overview

To use `bash scripts/infer.sh`, you need:

1. ✅ **Pre-computed hints** (`.txt` files with coordinates)
2. ✅ **Image names matching hint file names**
3. ✅ **Correct directory structure**

---

## Step 1: Pre-compute Hints

### Quick Start

```bash
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/ \
    --num_hints 0 1 2 5 10 20 50 100 200
```

This generates hint files for all your images.

### What It Does

Creates directories with hint coordinates:

```
my_hints/
├── h2-n0/           # 0 hints per image
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
├── h2-n1/           # 1 hint per image
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
├── h2-n2/           # 2 hints per image
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
└── ... (more directories for different hint counts)
```

### Hint File Format

Each `.txt` file contains pixel coordinates of hints:

```
# 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
112 56
168 224
45 189
```

Each line is: `x y` (pixel coordinates)

### Parameters

```bash
python3 precompute_hints.py \
    --img_dir validation/imgs/              # Input images directory
    --output_dir my_hints/                  # Output hints directory
    --hint_size 2                           # Patch size (default: 2)
    --num_hints 0 1 2 5 10 20 50 100 200   # Hint counts to generate
    --input_size 224                        # Input image size (default: 224)
    --seed 42                               # Random seed (for reproducibility)
```

---

## Step 2: Organize Your Images

### Directory Structure

Your images should be in a directory:

```
validation/
└── imgs/
    ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.webp
    └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.webp
```

**Important**: Image names must match hint file names (without extension)

### Supported Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- WebP (`.webp`)
- BMP (`.bmp`)
- TIFF (`.tiff`)

---

## Step 3: Update `scripts/infer.sh`

Edit the paths in `scripts/infer.sh`:

```bash
# path to model and validation dataset
MODEL_PATH='checkpoints/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='validation/'                    # Directory containing imgs/ subdirectory
VAL_HINT_DIR='my_hints/'                       # Directory with pre-computed hints
PRED_DIR='results/'                            # Output directory for colorized images
```

**Current state** (already updated):
```bash
MODEL_PATH='checkpoints/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='validation/'
VAL_HINT_DIR='ctest10k_hint/'
PRED_DIR='results/'
```

---

## Step 4: Run Inference

### Run with Default Hints

```bash
bash scripts/infer.sh
```

This will:
1. Load images from `validation/imgs/`
2. Load hints from `my_hints/h2-n10/` (or whatever is in `VAL_HINT_DIR`)
3. Run colorization
4. Save results to `results/h2-n{num_hints}/`

### Run with Specific Hints

```bash
# Test with 10 hints
bash scripts/infer.sh 0 "--val_hint_list 10"

# Test with multiple hint counts
bash scripts/infer.sh 0 "--val_hint_list 0 5 10 20 50"

# Use GPU
bash scripts/infer.sh 0 "--device cuda"
```

### Output

Results will be saved to:

```
results/
├── h2-n0/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
├── h2-n1/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
└── ... (more directories for different hint counts)
```

---

## Step 5: Evaluate Results

### Run Evaluation

```bash
bash scripts/eval.sh
```

This will:
1. Compare predictions with ground truth
2. Compute PSNR and LPIPS metrics
3. Save results to `evaluation_results/`

### Evaluate Specific Hint Count

```bash
bash scripts/eval.sh 0 10
```

### Output

Results saved to `evaluation_results/h2-n10.txt`:

```
Image1: PSNR=26.5, LPIPS=0.15
Image2: PSNR=27.2, LPIPS=0.14
Average: PSNR=26.85, LPIPS=0.145
```

---

## Complete Workflow

### 1. Pre-compute Hints

```bash
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/ \
    --num_hints 0 1 2 5 10 20 50 100 200
```

### 2. Update `scripts/infer.sh`

```bash
# Edit scripts/infer.sh
VAL_HINT_DIR='my_hints/'
```

### 3. Run Inference

```bash
bash scripts/infer.sh
```

### 4. Evaluate

```bash
bash scripts/eval.sh
```

---

## Understanding Hints

### What is a Hint?

A hint is a pixel location where the user provides color information.

### Hint Coordinates

Hints are stored as pixel coordinates (x, y):

```
112 56    # Hint at pixel (112, 56)
168 224   # Hint at pixel (168, 224)
45 189    # Hint at pixel (45, 189)
```

### How Many Patches?

For 224×224 image with 16×16 patches:
- Number of patches: (224/16)² = 14² = 196 patches

For 224×224 image with 2×2 hint regions:
- Number of hint locations: (224/2)² = 112² = 12,544 locations

### Hint Counts

- **0 hints**: Pure colorization (no user input)
- **1-5 hints**: Minimal user effort
- **10 hints**: Balanced (default)
- **20-50 hints**: More control
- **100+ hints**: Fine-tuned results

---

## Troubleshooting

### Issue: "No such file or directory"

**Cause**: Hint files not found

**Solution**:
```bash
# Check hint directory exists
ls -la my_hints/h2-n10/

# Check image names match hint names
ls validation/imgs/
ls my_hints/h2-n10/
```

### Issue: "Assertion error: len(img_list) != len(hint_list)"

**Cause**: Number of images doesn't match number of hints

**Solution**:
1. Check all images have corresponding hint files
2. Verify image names match hint file names (without extension)

```bash
# Should show same files
ls validation/imgs/ | sed 's/\.[^.]*$//'
ls my_hints/h2-n10/ | sed 's/\.[^.]*$//'
```

### Issue: "Image names don't match"

**Cause**: Image and hint file names are different

**Solution**: Rename images or regenerate hints with correct names

```bash
# Regenerate hints (will use current image names)
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/
```

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Use CPU
bash scripts/infer.sh 0 "--device cpu"

# Or reduce batch size
bash scripts/infer.sh 0 "--batch_size 1"
```

---

## Quick Reference

### Pre-compute Hints
```bash
python3 precompute_hints.py --img_dir validation/imgs/ --output_dir my_hints/
```

### Run Inference
```bash
bash scripts/infer.sh
```

### Evaluate Results
```bash
bash scripts/eval.sh
```

### Check Results
```bash
open results/h2-n10/
```

---

## Key Differences: `infer.sh` vs `simple_infer.py`

| Feature | `infer.sh` | `simple_infer.py` |
|---------|-----------|-------------------|
| **Hints** | Pre-computed files | Generated on-the-fly |
| **Setup** | Requires pre-computation | No setup needed |
| **Flexibility** | Strict naming | Any image names |
| **Reproducibility** | Exact same hints | Random each time |
| **Use Case** | Paper reproduction | Quick testing |

---

## Summary

To use `bash scripts/infer.sh`:

1. ✅ Pre-compute hints: `python3 precompute_hints.py --img_dir validation/imgs/ --output_dir my_hints/`
2. ✅ Update `scripts/infer.sh` with paths
3. ✅ Run: `bash scripts/infer.sh`
4. ✅ Evaluate: `bash scripts/eval.sh`

For quick testing without pre-computation, use:
```bash
python3 simple_infer.py --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth --val_data_path=validation/imgs/
```

---

*Last Updated: November 20, 2025*
