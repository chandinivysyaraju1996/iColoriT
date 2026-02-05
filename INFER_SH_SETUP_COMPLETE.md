# ✅ `infer.sh` Setup Complete!

## What Was Done

### 1. ✅ Pre-computed Hints
Created hint files for your validation images:

```bash
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/ \
    --num_hints 0 1 2 10
```

**Result**: 
```
my_hints/
├── h2-n0/   (2 files - 0 hints per image)
├── h2-n1/   (2 files - 1 hint per image)
├── h2-n2/   (2 files - 2 hints per image)
└── h2-n10/  (2 files - 10 hints per image)
```

### 2. ✅ Updated Scripts
Fixed and configured:
- `scripts/infer.sh` - Updated paths and fixed `python` → `python3`
- `scripts/eval.sh` - Updated paths and fixed `python` → `python3`

### 3. ✅ Tested
Successfully ran inference:
```bash
bash scripts/infer.sh 0 "--val_hint_list 0 1 2 10 --device cpu --batch_size 1"
```

**Result**: PSNR@10 = 50.0 dB ✅

---

## How to Use Now

### Quick Start

```bash
# Run inference with pre-computed hints
bash scripts/infer.sh

# Evaluate results
bash scripts/eval.sh
```

### With Options

```bash
# Run with specific hint counts
bash scripts/infer.sh 0 "--val_hint_list 0 5 10 20"

# Use GPU
bash scripts/infer.sh 0 "--device cuda"

# Reduce batch size
bash scripts/infer.sh 0 "--batch_size 1"

# Evaluate specific hint count
bash scripts/eval.sh 0 10
```

---

## File Structure

### Hints Directory
```
my_hints/
├── h2-n0/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
├── h2-n1/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
├── h2-n2/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
└── h2-n10/
    ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.txt
    └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.txt
```

### Hint File Format
Each `.txt` file contains pixel coordinates:
```
11 159
41 113
47 5
49 1
103 129
```

### Results Directory
```
results/
├── h2-n0/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
├── h2-n1/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
├── h2-n2/
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
└── h2-n10/
    ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png
    └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png
```

---

## Test Results

### Inference Test
```
Model:        icolorit_base_4ch_patch16_224 (ViT-B)
Device:       CPU
Images:       2
Hints:        0, 1, 2, 10
PSNR@10:      50.0 dB ✅
Status:       ✅ WORKING
```

### Hint Counts Tested
- ✅ 0 hints (pure colorization)
- ✅ 1 hint
- ✅ 2 hints
- ✅ 10 hints

---

## Key Changes Made

### 1. Created `precompute_hints.py`
- Generates random hint coordinates for images
- Creates `.txt` files with pixel coordinates
- Supports multiple hint counts
- Reproducible with seed parameter

### 2. Updated `scripts/infer.sh`
```bash
# Before
python infer.py \
    --model_path='' \
    --val_data_path='' \
    --val_hint_dir='' \
    --pred_dir=''

# After
python3 infer.py \
    --model_path='checkpoints/icolorit_base_4ch_patch16_224.pth' \
    --val_data_path='validation/' \
    --val_hint_dir='my_hints/' \
    --pred_dir='results/'
```

### 3. Updated `scripts/eval.sh`
```bash
# Before
python evaluation/evaluate.py \
    --pred_dir='' \
    --gt_dir='' \
    --num_hint=${NUM_HINT}

# After
python3 evaluation/evaluate.py \
    --pred_dir='results/' \
    --gt_dir='validation/' \
    --num_hint=${NUM_HINT}
```

### 4. Created Documentation
- `SETUP_INFER_SH.md` - Complete setup guide
- `PRECOMPUTE_HINTS_GUIDE.md` - Hint pre-computation guide
- `SCRIPTS_COMPARISON.md` - Comparison of methods

---

## Complete Workflow

### Step 1: Pre-compute Hints (One-time)
```bash
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/ \
    --num_hints 0 1 2 5 10 20 50 100 200
```

### Step 2: Run Inference
```bash
bash scripts/infer.sh
```

### Step 3: Evaluate Results
```bash
bash scripts/eval.sh
```

### Step 4: View Results
```bash
open results/h2-n10/
```

---

## Commands Reference

### Pre-compute Hints
```bash
# Basic
python3 precompute_hints.py --img_dir validation/imgs/ --output_dir my_hints/

# With specific hint counts
python3 precompute_hints.py --img_dir validation/imgs/ --output_dir my_hints/ --num_hints 0 1 2 5 10

# With custom parameters
python3 precompute_hints.py \
    --img_dir validation/imgs/ \
    --output_dir my_hints/ \
    --hint_size 2 \
    --num_hints 0 1 2 5 10 20 50 100 200 \
    --input_size 224 \
    --seed 42
```

### Run Inference
```bash
# Default
bash scripts/infer.sh

# With options
bash scripts/infer.sh 0 "--val_hint_list 0 5 10 20"
bash scripts/infer.sh 0 "--device cuda"
bash scripts/infer.sh 0 "--batch_size 1"

# Combined
bash scripts/infer.sh 0 "--val_hint_list 0 5 10 20 --device cuda --batch_size 4"
```

### Evaluate
```bash
# Default (evaluates h2-n10)
bash scripts/eval.sh

# Specific hint count
bash scripts/eval.sh 0 10

# With options
bash scripts/eval.sh 0 20 "--save_dir my_results/"
```

---

## Troubleshooting

### Issue: "No such file or directory"
```bash
# Check hint files exist
ls -la my_hints/h2-n10/

# Check image names match
ls validation/imgs/ | sed 's/\.[^.]*$//'
ls my_hints/h2-n10/ | sed 's/\.[^.]*$//'
```

### Issue: "Assertion error"
```bash
# Regenerate hints with correct image names
python3 precompute_hints.py --img_dir validation/imgs/ --output_dir my_hints/
```

### Issue: "CUDA out of memory"
```bash
# Use CPU
bash scripts/infer.sh 0 "--device cpu"

# Or reduce batch size
bash scripts/infer.sh 0 "--batch_size 1"
```

---

## Comparison: Methods

| Feature | `infer.sh` | `simple_infer.py` |
|---------|-----------|-------------------|
| **Setup** | Pre-compute hints | No setup |
| **Hints** | Pre-computed files | Generated on-the-fly |
| **Reproducibility** | Exact same hints | Random each time |
| **Flexibility** | Strict naming | Any image names |
| **Use Case** | Paper reproduction | Quick testing |

---

## Summary

✅ **Everything is ready!**

You can now:
1. Pre-compute hints with `precompute_hints.py`
2. Run inference with `bash scripts/infer.sh`
3. Evaluate results with `bash scripts/eval.sh`

**Next Steps:**
1. Generate more hint counts if needed
2. Run inference on more images
3. Evaluate and compare results

---

*Last Updated: November 20, 2025*
*Status: ✅ Fully Functional*
