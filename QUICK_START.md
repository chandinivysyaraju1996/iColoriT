# iColoriT - Quick Start Guide

## ✅ Setup Complete!

Your iColoriT project is now ready to use. Here's what has been set up:

### What You Have

- ✅ **Pre-trained Model**: `checkpoints/icolorit_base_4ch_patch16_224.pth` (ViT-B, ~1GB)
- ✅ **Validation Images**: 2 images in `validation/imgs/`
- ✅ **Pre-computed Hints**: Multiple hint sets in `ctest10k_hint/`
- ✅ **Documentation**: Complete documentation in `DOCUMENTATION.md`
- ✅ **Simple Inference Script**: `simple_infer.py` for easy colorization

---

## 🚀 Quick Start (30 seconds)

### Run Colorization with 10 Hints

```bash
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ \
    --pred_dir=results/ \
    --num_hints=10
```

**Output:**
- Colorized images saved to: `results/h2-n10/`
- PSNR metric: ~26.85 dB (higher is better)

---

## 📊 Test Results (Already Run)

### Inference on Your Validation Images

| Metric | Value |
|--------|-------|
| **Images Processed** | 2 |
| **Average PSNR** | 26.85 dB |
| **Hints Used** | 10 per image |
| **Processing Time** | ~3.28 sec/image (CPU) |
| **Output Format** | PNG (lossless) |

### Output Files

```
results/
└── h2-n10/
    ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.png (73 KB)
    └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.png (69 KB)
```

---

## 🎨 How It Works

### Input
- **Grayscale or Color Images** (any size, will be resized to 224×224)
- **Hint Locations**: Random points where user provides color information

### Process
1. Convert image to LAB color space (separates color from brightness)
2. Generate random hint locations (default: 10 hints)
3. Vision Transformer predicts colors for all regions
4. Reconstruct full RGB image

### Output
- **Colorized Image** in PNG format
- **Quality Metric** (PSNR) indicating color accuracy

---

## 🔧 Advanced Usage

### Try Different Number of Hints

```bash
# No hints (pure colorization)
python3 simple_infer.py --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ --num_hints=0

# More hints = better quality
python3 simple_infer.py --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=validation/imgs/ --num_hints=50
```

### Use Your Own Images

```bash
# Place images in a folder (e.g., my_images/)
python3 simple_infer.py --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=my_images/ --pred_dir=my_results/
```

### Batch Processing

```bash
# Process multiple images with different hint counts
for hints in 0 5 10 20 50; do
    python3 simple_infer.py \
        --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
        --val_data_path=validation/imgs/ \
        --num_hints=$hints
done
```

---

## 📈 Understanding Results

### PSNR (Peak Signal-to-Noise Ratio)

- **Higher is better**
- Typical range: 20-30 dB for colorization
- **26.85 dB** = Good quality
- **25+ dB** = Acceptable
- **20+ dB** = Visible but with artifacts

### How Hints Affect Quality

| Hints | Quality | Use Case |
|-------|---------|----------|
| 0 | Lower (pure colorization) | Fully automatic |
| 5 | Moderate | Quick coloring |
| 10 | Good | Balanced (default) |
| 20 | Better | More control |
| 50+ | Excellent | Fine-tuned results |

---

## 🐛 Troubleshooting

### Issue: "No module named torch"

**Solution:**
```bash
pip install torch torchvision
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use CPU (slower but works)
python3 simple_infer.py --device=cpu ...

# Or reduce batch size
python3 simple_infer.py --batch_size=1 ...
```

### Issue: "No images found"

**Solution:**
- Check image path is correct
- Ensure images are in standard formats: PNG, JPEG, WEBP, BMP
- Verify folder structure:
  ```
  my_images/
  ├── image1.jpg
  ├── image2.png
  └── image3.webp
  ```

### Issue: Results look wrong

**Solution:**
- Try with more hints: `--num_hints=20`
- Check if input image is actually grayscale
- Verify checkpoint file is not corrupted (1GB file)

---

## 📚 File Reference

### Key Scripts

| File | Purpose |
|------|---------|
| `simple_infer.py` | **Easy inference** (recommended for you) |
| `infer.py` | Original inference script (complex) |
| `modeling.py` | Model architecture |
| `utils.py` | Utility functions (LAB conversion, PSNR) |

### Key Directories

| Directory | Contents |
|-----------|----------|
| `checkpoints/` | Pre-trained model weights |
| `validation/imgs/` | Your validation images |
| `results/` | Colorized output images |
| `ctest10k_hint/` | Pre-computed hints (ImageNet) |
| `evaluation/` | Evaluation scripts |

---

## 🎯 Next Steps

### 1. Test with Your Own Images

```bash
# Copy your images to a folder
mkdir my_images
cp /path/to/your/images/* my_images/

# Run colorization
python3 simple_infer.py \
    --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=my_images/ \
    --pred_dir=my_results/
```

### 2. Experiment with Different Hint Counts

```bash
# Create a comparison
for hints in 0 5 10 20 50; do
    echo "Testing with $hints hints..."
    python3 simple_infer.py \
        --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
        --val_data_path=validation/imgs/ \
        --num_hints=$hints
done
```

### 3. Evaluate Results

```bash
# Run evaluation (if you have ground truth)
python3 evaluation/evaluate.py \
    --pred_dir=results/ \
    --gt_dir=validation/imgs/ \
    --num_hint=10
```

### 4. Fine-tune (Optional)

If you want to train on your own dataset:
```bash
python3 train.py \
    --data_path=/path/to/imagenet/train \
    --eval_data_path=/path/to/imagenet/val \
    --output_dir=./output
```

---

## 📖 Documentation

For detailed information, see:
- **`DOCUMENTATION.md`**: Complete project documentation
- **`README.md`**: Original project README
- **Paper**: https://arxiv.org/abs/2207.06831

---

## 🔗 Useful Commands

```bash
# Check PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check available images
ls -lh validation/imgs/

# Check results
ls -lh results/h2-n10/

# View image (macOS)
open results/h2-n10/001-feature-14-012-P1040585-Cayena-Beach-Villas.png

# Count images in directory
ls validation/imgs/ | wc -l

# Get file sizes
du -sh results/h2-n10/*
```

---

## ✨ Summary

You now have a fully functional iColoriT colorization system:

- ✅ Pre-trained model ready to use
- ✅ Simple inference script (`simple_infer.py`)
- ✅ Test images and results
- ✅ Complete documentation
- ✅ Verified working (PSNR: 26.85 dB)

**To colorize images:**
```bash
python3 simple_infer.py --model_path=checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path=your_images/ --num_hints=10
```

**Results will be in:** `results/h2-n10/`

---

*Last updated: November 20, 2025*
*Status: ✅ Fully Functional*
