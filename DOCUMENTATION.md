# iColoriT - Complete Documentation

## Project Overview

**iColoriT** is an official PyTorch implementation of the WACV 2023 paper: *"iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer"*

### What is iColoriT?

iColoriT is a point-interactive image colorization system that uses Vision Transformers (ViT) to colorize grayscale images based on user-provided color hints. The key innovation is leveraging the global receptive field of Transformers to intelligently propagate user hints to relevant regions in the image.

**Key Features:**
- Real-time colorization using pixel shuffling for efficient upsampling
- Global receptive field via Transformer self-attention for better hint propagation
- Minimal user effort required (few local hints needed)
- Multiple model variants: ViT-B (Base), ViT-S (Small), ViT-Ti (Tiny)

---

## Project Structure

```
iColoriT-main/
├── README.md                      # Original project README
├── DOCUMENTATION.md               # This file
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
│
├── checkpoints/                   # Pre-trained model weights
│   └── icolorit_base_4ch_patch16_224.pth
│
├── validation/                    # Validation images for testing
│   ├── 001-feature-14-012-P1040585-Cayena-Beach-Villas.webp
│   └── african-elephant-common-zebras-nature-wildlife-photography-james-warwick-bw.webp
│
├── ctest10k_hint/                 # Pre-computed hints for testing (53600 items)
│
├── scripts/                       # Execution scripts
│   ├── infer.sh                   # Inference script
│   ├── eval.sh                    # Evaluation script
│   └── train.sh                   # Training script
│
├── evaluation/                    # Evaluation utilities
│   ├── evaluate.py                # Main evaluation script
│   ├── hpr.py                     # HPR metric
│   ├── rollout.py                 # Attention rollout visualization
│   └── vis_rollout.py             # Visualization utilities
│
├── iColoriT_demo/                 # Interactive demo application
│
├── modeling.py                    # Model architecture (Vision Transformer)
├── infer.py                       # Inference script
├── train.py                       # Training script
├── engine.py                      # Training engine
│
├── datasets.py                    # Dataset classes and transformations
├── dataset_folder.py              # Custom dataset folder implementation
├── hint_generator.py              # Hint generation utilities
├── losses.py                      # Loss functions
├── utils.py                       # Utility functions (LAB conversion, PSNR, etc.)
├── optim_factory.py               # Optimizer factory
│
├── docs/                          # Documentation assets
│   └── iColoriT_demo.gif          # Demo GIF
│
└── preparation/                   # Data preparation utilities
```

---

## Important Files Explained

### Core Model Files

**`modeling.py`** (578 lines)
- Defines the iColoriT Vision Transformer architecture
- Key components:
  - `Attention`: Multi-head self-attention with optional relative positional bias (RPB)
  - `TransformerBlock`: Transformer encoder block with attention and MLP
  - `iColoriT`: Main model class with patch embedding and transformer layers
  - Supports different backbone sizes (Base, Small, Tiny)
  - Includes local stabilizing layer for pixel shuffling artifacts

**`infer.py`** (198 lines)
- Inference/testing script for colorization
- Loads pre-trained checkpoint
- Processes validation images with hints
- Computes PSNR metrics
- Saves colorized results

**`train.py`** (13398 bytes)
- Training script for the model
- Handles distributed training
- Implements learning rate scheduling
- Saves checkpoints periodically

### Dataset & Hint Generation

**`datasets.py`** (198 lines)
- `DataAugmentationForIColoriT`: Training data augmentation
- `DataTransformationForIColoriT`: Validation data transformation
- `DataTransformationFixedHint`: Fixed hint transformation for evaluation

**`hint_generator.py`** (87 lines)
- `RandomHintGenerator`: Generates random hint locations during training
- `InteractiveHintGenerator`: For user-interactive colorization
- Hints are represented as binary masks at patch level

**`dataset_folder.py`** (13481 bytes)
- Custom ImageFolder implementations
- `ImageWithFixedHint`: Loads images with pre-computed hints
- `ImageWithFixedHintAndCoord`: Includes coordinate information

### Utilities

**`utils.py`** (23394 bytes)
- `rgb2lab()`: Convert RGB images to LAB color space
- `lab2rgb()`: Convert LAB back to RGB
- `psnr()`: Peak Signal-to-Noise Ratio calculation
- Various other utility functions

**`losses.py`** (999 bytes)
- Loss function definitions for training

**`engine.py`** (7544 bytes)
- Training engine with epoch loop
- Validation logic

---

## Dependencies

### Required Packages (from requirements.txt)

```
torch                    # PyTorch framework
torchvision             # Vision utilities
einops==0.4.1           # Tensor rearrangement
lpips==0.1.4            # LPIPS metric for evaluation
opencv_python==4.6.0.66 # Image processing
Pillow                  # Image library
tensorboardX==2.5.1     # TensorBoard logging
timm==0.4.12            # PyTorch Image Models (for ViT)
tqdm==4.64.0            # Progress bars
```

### Installation

```bash
# Clone the repository (already done)
git clone https://github.com/pmh9960/iColoriT.git

# Install dependencies
pip install -r requirements.txt
```

---

## Pre-trained Checkpoints

Three pre-trained models are available:

| Model | Backbone | Size | Link |
|-------|----------|------|------|
| iColoriT | ViT-B | ~1GB | [Google Drive](https://drive.google.com/file/d/16i9ulB4VRbFLbLlAa7UjIQR6J334BeKW/view?usp=sharing) |
| iColoriT-S | ViT-S | Smaller | [Google Drive](https://drive.google.com/file/d/1yKwFTQGDBvr9B7NIyXhxQH0K-BNlCs4L/view?usp=sharing) |
| iColoriT-T | ViT-Ti | Smallest | [Google Drive](https://drive.google.com/file/d/1GMmjfxAoM95cABwlZD8555WxI7nmIZrR/view?usp=sharing) |

**Status:** You already have `icolorit_base_4ch_patch16_224.pth` (ViT-B) in the `checkpoints/` folder.

---

## Implementation Steps

### Step 1: Environment Setup

```bash
# Install Python 3.8+ (recommended: 3.8-3.10)
python --version

# Install PyTorch (choose appropriate version for your system)
# For CPU:
pip install torch torchvision

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

**For Inference:**
- Place grayscale/color images in a directory (e.g., `validation/`)
- Pre-computed hints should be in format: `h{hint_size}-n{num_hints}/`
  - Example: `h2-n10/` for hint size 2 and 10 hints
  - Hints are binary masks stored as numpy arrays or images

**For Training:**
- Prepare ImageNet dataset with structure:
  ```
  train/
   └ id1/
     └ image1.JPEG
     └ image2.JPEG
   └ id2/
     └ image1.JPEG
  ```

### Step 3: Generate Hints (if needed)

```bash
# Use hint_generator.py to create random hints
python hint_generator.py --input_dir validation/ --output_dir ctest10k_hint/ --num_hints 10
```

### Step 4: Run Inference

See the "How to Run" section below.

### Step 5: Evaluate Results

See the "Testing & Evaluation" section below.

---

## How to Run

### Quick Start: Inference with Your Checkpoint

#### Option 1: Using Python Directly

```bash
cd /Users/chandinivysyaraju/Documents/Thesis/iColoriT-main

python infer.py \
    --model_path checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path validation/ \
    --val_hint_dir ctest10k_hint/ \
    --pred_dir results/ \
    --batch_size 2 \
    --num_workers 0 \
    --device cuda
```

**Key Parameters:**
- `--model_path`: Path to checkpoint (you have this)
- `--val_data_path`: Directory containing validation images
- `--val_hint_dir`: Directory with pre-computed hints
- `--pred_dir`: Output directory for colorized images
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--num_workers`: Number of data loading workers
- `--device`: 'cuda' for GPU, 'cpu' for CPU
- `--val_hint_list`: List of hint counts to test (default: [0, 1, 2, 5, 10, 20, 50, 100, 200])

#### Option 2: Using the Shell Script (Recommended)

Edit `scripts/infer.sh`:

```bash
# Set these paths:
MODEL_PATH='checkpoints/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='validation/'
VAL_HINT_DIR='ctest10k_hint/'
PRED_DIR='results/'

# Run inference
bash scripts/infer.sh
```

### Testing & Evaluation

#### Step 1: Run Inference (generates colorized images)

```bash
python infer.py \
    --model_path checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path validation/ \
    --val_hint_dir ctest10k_hint/ \
    --pred_dir results/ \
    --batch_size 2
```

Output structure:
```
results/
├── h2-n0/      # Results with 0 hints
├── h2-n1/      # Results with 1 hint
├── h2-n2/      # Results with 2 hints
├── h2-n5/      # Results with 5 hints
├── h2-n10/     # Results with 10 hints
└── ...
```

#### Step 2: Evaluate Results

```bash
python evaluation/evaluate.py \
    --pred_dir results/ \
    --gt_dir validation/ \
    --num_hint 10 \
    --hint_size 2 \
    --save_dir evaluation_results/
```

**Metrics Computed:**
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

Output: `evaluation_results/h2-n10.txt` with metrics

---

## Model Architecture Overview

### Input/Output

**Input:**
- Grayscale image (L channel in LAB color space): `[B, 1, H, W]`
- Hint mask (binary): `[B, num_patches]` where 0 = hint location, 1 = no hint

**Output:**
- Predicted AB channels: `[B, num_patches, patch_size²×2]`

### Key Components

1. **Patch Embedding**: Converts image to patches (default: 16×16)
2. **Vision Transformer Blocks**: Multi-head self-attention with:
   - Relative Positional Bias (RPB) for better spatial awareness
   - MLP feed-forward networks
   - Drop path regularization
3. **Local Stabilizing Layer**: Reduces artifacts from pixel shuffling
4. **Pixel Shuffling**: Efficient upsampling instead of transposed convolution

### Model Variants

- **iColoriT (Base)**: ViT-B, 768 hidden dims, 12 layers, 12 heads
- **iColoriT-S (Small)**: ViT-S, 384 hidden dims, 12 layers, 6 heads
- **iColoriT-T (Tiny)**: ViT-Ti, 192 hidden dims, 12 layers, 3 heads

---

## Training (Optional)

### Prepare Training Data

```bash
# ImageNet structure required:
train/
 └ n01440764/
   └ n01440764_10045.JPEG
   └ n01440764_10046.JPEG
 └ n01443537/
   └ n01443537_10007.JPEG
```

### Run Training

Edit `scripts/train.sh`:

```bash
TRAIN_DIR='path/to/train/'
VAL_DIR='path/to/val/'

bash scripts/train.sh
```

Or directly:

```bash
python train.py \
    --model icolorit_base_4ch_patch16_224 \
    --data_path path/to/imagenet/train \
    --eval_data_path path/to/imagenet/val \
    --output_dir ./output \
    --batch_size 128 \
    --epochs 100 \
    --lr 1.5e-4 \
    --warmup_epochs 20
```

---

## Color Space Explanation

The model works in **LAB color space**:
- **L**: Lightness (0-100)
- **A**: Green-Red axis (-128 to 127)
- **B**: Blue-Yellow axis (-128 to 127)

**Why LAB?**
- Separates luminance from chrominance
- Model only predicts AB channels (color)
- L channel (grayscale) is preserved from input

**Conversion Functions** (in `utils.py`):
- `rgb2lab()`: RGB → LAB
- `lab2rgb()`: LAB → RGB

---

## Hint System

### How Hints Work

1. **Hint Mask**: Binary array at patch resolution
   - 0 = hint location (user provides color)
   - 1 = no hint (model predicts color)

2. **Hint Size**: Controls patch granularity
   - Default: 2×2 pixels per hint location
   - For 224×224 image: 112×112 hint locations

3. **Number of Hints**: How many locations get user input
   - 0 hints: Pure colorization (no user input)
   - 10 hints: 10 random locations colored by user
   - 100+ hints: More user guidance

### Hint Generation

**Random Hints** (for training/testing):
```python
from hint_generator import RandomHintGenerator

gen = RandomHintGenerator(input_size=224, hint_size=2, num_hint_range=[10, 10])
hint_mask = gen()  # Returns binary array
```

**Interactive Hints** (for user input):
```python
from hint_generator import InteractiveHintGenerator

gen = InteractiveHintGenerator(input_size=224, hint_size=2)
hint_mask, coords = gen()  # User inputs coordinates
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python infer.py --batch_size 1 ...

# Or use CPU
python infer.py --device cpu ...
```

### Issue: Missing Hint Files

**Solution:**
```bash
# Generate hints if not present
python hint_generator.py \
    --input_dir validation/ \
    --output_dir ctest10k_hint/ \
    --hint_size 2 \
    --num_hints 10
```

### Issue: Image Format Not Supported

**Solution:**
- Ensure images are in standard formats: PNG, JPEG, WEBP
- Convert if needed:
  ```bash
  # Using PIL
  from PIL import Image
  img = Image.open('image.webp').convert('RGB')
  img.save('image.png')
  ```

### Issue: Model Checkpoint Not Found

**Solution:**
- Verify checkpoint path is correct
- Download from Google Drive if missing
- Check file size: ~1GB for ViT-B model

---

## Expected Results

### Metrics (on standard datasets)

With 10 hints (h2-n10):
- **PSNR**: ~25-28 dB (higher is better)
- **LPIPS**: ~0.15-0.20 (lower is better)

Results improve with more hints:
- 0 hints: Lower quality (pure colorization)
- 10 hints: Good quality
- 50+ hints: Excellent quality

### Visual Quality

- Natural color propagation to similar regions
- Minimal artifacts at hint boundaries
- Realistic color choices for ambiguous regions

---

## References

### Paper
- **Title**: iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer
- **Authors**: Jooyeol Yun, Sanghyeon Lee, Minho Park, Jaegul Choo (KAIST)
- **Conference**: WACV 2023
- **Paper Link**: https://arxiv.org/abs/2207.06831
- **Project Page**: https://pmh9960.github.io/research/iColoriT/

### Citation

```bibtex
@InProceedings{Yun_2023_WACV,
    author    = {Yun, Jooyeol and Lee, Sanghyeon and Park, Minho and Choo, Jaegul},
    title     = {iColoriT: Towards Propagating Local Hints to the Right Region in Interactive Colorization by Leveraging Vision Transformer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {1787-1796}
}
```

### Related Codebases
- BEiT: https://github.com/microsoft/unilm/tree/master/beit
- timm (PyTorch Image Models): https://github.com/rwightman/pytorch-image-models
- DeiT: https://github.com/facebookresearch/deit
- DINO: https://github.com/facebookresearch/dino

---

## Quick Command Reference

```bash
# Setup
pip install -r requirements.txt

# Inference
python infer.py \
    --model_path checkpoints/icolorit_base_4ch_patch16_224.pth \
    --val_data_path validation/ \
    --val_hint_dir ctest10k_hint/ \
    --pred_dir results/

# Evaluation
python evaluation/evaluate.py \
    --pred_dir results/ \
    --gt_dir validation/ \
    --num_hint 10

# Training (if needed)
python train.py \
    --data_path /path/to/imagenet/train \
    --eval_data_path /path/to/imagenet/val \
    --output_dir ./output
```

---

## Next Steps

1. **Verify Setup**: Run inference with your checkpoint and validation images
2. **Test with Hints**: Try different hint counts (0, 5, 10, 20, 50)
3. **Evaluate**: Compare metrics across hint counts
4. **Visualize**: Check colorized images in `results/` directory
5. **Fine-tune** (optional): Train on custom dataset if needed

---

*Documentation created for iColoriT project*
*Last updated: November 2025*
