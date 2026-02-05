# Scripts Comparison: Original vs. Custom

## Overview

There are two ways to run inference on iColoriT:

1. **Original Method**: `bash scripts/infer.sh` (from README)
2. **Custom Method**: `python3 simple_infer.py` (created for flexibility)

This document explains the differences and when to use each.

---

## Quick Comparison

| Feature | `scripts/infer.sh` | `simple_infer.py` |
|---------|-------------------|-------------------|
| **Purpose** | Original inference script | Easy-to-use wrapper |
| **Hint Source** | Pre-computed .txt files | Generates random hints |
| **Image Names** | Must match hint files | Any image names |
| **Directory Structure** | Strict (ImageNet format) | Flexible |
| **PyTorch 2.9** | ❌ Broken | ✅ Fixed |
| **Ease of Use** | Manual path editing | Simple command |
| **Best For** | ImageNet validation | Any images |

---

## Original Method: `scripts/infer.sh`

### How It Works

```bash
bash scripts/infer.sh
```

### What It Does

1. Reads paths from `scripts/infer.sh`
2. Calls `infer.py` with those paths
3. Expects pre-computed hints in `VAL_HINT_DIR`
4. Matches images with hint files by name

### Requirements

**Pre-computed Hints**: Must have `.txt` files matching image names

```
ctest10k_hint/h2-n10/
├── ILSVRC2012_val_00025275.txt
├── ILSVRC2012_val_00026187.txt
└── ... (must match image names exactly)
```

**Image Names**: Must match hint file names

```
validation/imgs/
├── ILSVRC2012_val_00025275.jpg
├── ILSVRC2012_val_00026187.jpg
└── ...
```

### When to Use

✅ **Use this when:**
- You have ImageNet validation images
- You have pre-computed hints from the paper
- You want to reproduce exact paper results
- Image names match hint file names

❌ **Don't use this when:**
- You have custom images with different names
- You don't have pre-computed hints
- You want to generate random hints

### Example

```bash
# Edit scripts/infer.sh to set paths
MODEL_PATH='checkpoints/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='validation/'
VAL_HINT_DIR='ctest10k_hint/'
PRED_DIR='results/'

# Run
bash scripts/infer.sh
```

### Why It Failed for Your Images

Your images have different names:
```
validation/imgs/
├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.webp
└── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.webp
```

But the hints are for ImageNet:
```
ctest10k_hint/h2-n10/
├── ILSVRC2012_val_00025275.txt
├── ILSVRC2012_val_00026187.txt
└── ...
```

**Result**: No matching hints found → Error

---

## Custom Method: `simple_infer.py`

### How It Works

```bash
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ \
    --num_hints=10
```

### What It Does

1. Loads all images from the directory
2. **Generates random hints automatically** (no pre-computed files needed)
3. Runs inference for each image
4. Saves colorized results

### Requirements

**Just images**: Any image format (PNG, JPEG, WEBP, etc.)

```
validation/imgs/
├── image1.jpg
├── image2.png
├── image3.webp
└── ... (any names, any format)
```

### When to Use

✅ **Use this when:**
- You have custom images
- You don't have pre-computed hints
- You want to generate random hints
- You want easy, flexible inference
- You have images with any names

### Example

```bash
# Colorize with 10 hints
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ \
    --num_hints=10

# Try different hint counts
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ \
    --num_hints=50

# Use GPU
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ \
    --device=cuda
```

### Why It Works

- ✅ Generates hints automatically
- ✅ Works with any image names
- ✅ PyTorch 2.9 compatible
- ✅ Simple command-line interface

---

## Evaluation

### Original Method: `bash scripts/eval.sh`

```bash
bash scripts/eval.sh
```

Evaluates results from `scripts/infer.sh`

**Requirements:**
- Prediction images in `PRED_DIR`
- Ground truth images in `GT_DIR`
- Both must have matching names

### Custom Method: `python3 run_evaluation.py`

```bash
python3 run_evaluation.py
```

Evaluates results from `simple_infer.py`

**Features:**
- Automatic path detection
- Formatted output display
- Easier to use

---

## Decision Tree

```
Do you have ImageNet images?
├─ YES → Do you have pre-computed hints?
│   ├─ YES → Use: bash scripts/infer.sh
│   └─ NO → Use: python3 simple_infer.py
└─ NO → Use: python3 simple_infer.py
```

---

## Technical Details

### Why `scripts/infer.sh` Requires Matching Names

The original `infer.py` uses this logic:

```python
# Load images
img_list = os.listdir(img_dir)

# Load hints
for hint_dir in hint_dirs:
    hint_list = os.listdir(hint_dir)
    
    # Check that names match
    assert len(img_list) == len(hint_list)
    for img_f, hint_f in zip(img_list, hint_list):
        assert osp.splitext(img_f)[0] == osp.splitext(hint_f)[0]
```

This strict matching is necessary because:
1. Each image needs specific hint locations
2. Hints are pre-computed and saved as files
3. The script must know which hints go with which image

### Why `simple_infer.py` Generates Hints

Our custom script uses this approach:

```python
# Load images
dataset = SimpleImageDataset(img_dir)

# For each image, generate random hints on-the-fly
for batch in data_loader:
    # Generate random hints
    bool_hints = generate_random_hints(num_patches, num_hints)
    
    # Run inference
    outputs = model(images_lab, bool_hints)
```

This is flexible because:
1. Hints are generated randomly
2. No pre-computed files needed
3. Works with any image names

---

## Summary

### Use `bash scripts/infer.sh` When:
- Reproducing paper results
- Using ImageNet validation images
- You have pre-computed hints
- Image names match hint files

### Use `python3 simple_infer.py` When:
- Using custom images
- You don't have pre-computed hints
- You want flexibility
- You want easy-to-use interface

### For Your Case:
Since your images have different names than the pre-computed hints, **`simple_infer.py` is the right choice**.

---

## Both Methods Now Work

We've updated `scripts/infer.sh` and `scripts/eval.sh` with:
- ✅ Pre-filled paths
- ✅ Comments explaining limitations
- ✅ Ready to use

But for your validation images, **`simple_infer.py` is recommended** because it generates hints automatically.

---

## Quick Reference

### Original (ImageNet-specific)
```bash
bash scripts/infer.sh
bash scripts/eval.sh
```

### Custom (Flexible, Recommended)
```bash
python3 simple_infer.py --model_path=... --val_data_path=...
python3 run_evaluation.py
```

---

*Last Updated: November 20, 2025*
