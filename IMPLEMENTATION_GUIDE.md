# iColoriT - Implementation Guide

## Overview

This guide explains the key implementation details of iColoriT, including the model architecture, data flow, and how to extend or modify the system.

---

## 1. Architecture Overview

### Model Pipeline

```
Input Image (RGB)
    ↓
Convert to LAB Color Space
    ↓
Patch Embedding (16×16 patches)
    ↓
Vision Transformer Encoder
    ├─ Self-Attention (with Relative Positional Bias)
    ├─ MLP Feed-forward
    └─ Repeat 12 times
    ↓
Local Stabilizing Layer
    ↓
Pixel Shuffling Upsampling
    ↓
Output (AB channels)
    ↓
Reconstruct RGB Image
```

### Key Components

#### 1. **Patch Embedding** (`modeling.py`)
```python
# Converts image to patches
# Input: [B, 3, 224, 224]
# Output: [B, 196, 768]  # 196 = (224/16)²
```

#### 2. **Vision Transformer Blocks** (`modeling.py`)
```python
class TransformerBlock(nn.Module):
    - Multi-head Self-Attention (12 heads)
    - Relative Positional Bias (RPB)
    - MLP Feed-forward (3072 hidden dims)
    - Layer Normalization
    - Residual Connections
```

#### 3. **Hint Mechanism**
- Hints are binary masks at patch level
- 0 = hint location (user provides color)
- 1 = no hint (model predicts)
- Passed to model as additional input

#### 4. **Pixel Shuffling**
- Efficient upsampling technique
- Replaces transposed convolution
- Reduces artifacts with local stabilizing layer

---

## 2. Data Flow

### Training Data Flow

```
ImageNet Image
    ↓
Random Resize & Crop (224×224)
    ↓
Convert to RGB Tensor
    ↓
Convert to LAB Color Space
    ├─ L channel (lightness): [B, 1, 224, 224]
    └─ AB channels (color): [B, 2, 224, 224]
    ↓
Patch Embedding
    ├─ L channel patches: [B, 196, 256]
    └─ AB channel patches: [B, 196, 512]
    ↓
Generate Random Hints
    └─ Hint mask: [B, 196] (binary)
    ↓
Vision Transformer
    ├─ Input: Concatenate L + AB + hint mask
    └─ Output: Predicted AB channels [B, 196, 512]
    ↓
Loss Computation
    └─ L1 Loss between predicted and ground truth AB
    ↓
Backpropagation & Update
```

### Inference Data Flow

```
Input Image (any size)
    ↓
Resize to 224×224
    ↓
Convert to LAB
    ├─ L channel: [1, 1, 224, 224]
    └─ AB channels: [1, 2, 224, 224]
    ↓
Generate Random Hints (or use user hints)
    └─ Hint mask: [1, 196]
    ↓
Vision Transformer (no gradients)
    └─ Output: Predicted AB [1, 196, 512]
    ↓
Reconstruct Image
    ├─ Combine L + predicted AB
    └─ Convert LAB → RGB
    ↓
Save Result (PNG)
```

---

## 3. Color Space: LAB

### Why LAB?

LAB separates **luminance** from **chrominance**:
- **L**: Lightness (0-100) - brightness
- **A**: Green-Red axis (-128 to 127)
- **B**: Blue-Yellow axis (-128 to 127)

### Advantages for Colorization

1. Model only predicts AB (color), preserves L (brightness)
2. More perceptually uniform than RGB
3. Easier to learn color relationships
4. Natural separation of structure and color

### Conversion Functions (`utils.py`)

```python
# RGB → LAB
def rgb2lab(rgb_img):
    # Input: [B, 3, H, W] in range [0, 1]
    # Output: [B, 3, H, W] in LAB space
    
# LAB → RGB
def lab2rgb(lab_img):
    # Input: [B, 3, H, W] in LAB space
    # Output: [B, 3, H, W] in range [0, 1]
```

---

## 4. Hint System

### Hint Representation

```python
# Hint mask at patch level
# For 224×224 image with 16×16 patches:
# - Total patches: 14×14 = 196
# - Hint mask shape: [B, 196]
# - Value 0: hint location (user provides color)
# - Value 1: no hint (model predicts)

# Example with 10 hints:
hint_mask = [1, 1, 1, ..., 0, 1, ..., 0]  # 186 ones, 10 zeros
```

### Hint Generation

#### Random Hints (Training & Testing)
```python
from hint_generator import RandomHintGenerator

gen = RandomHintGenerator(
    input_size=224,
    hint_size=2,
    num_hint_range=[10, 10]  # Always 10 hints
)
hint_mask = gen()  # Returns [1, 1, ..., 0, ..., 0]
```

#### Interactive Hints (User Input)
```python
from hint_generator import InteractiveHintGenerator

gen = InteractiveHintGenerator(input_size=224, hint_size=2)
hint_mask, coords = gen()  # User inputs coordinates
```

### How Hints Are Used

```python
# In model forward pass:
# 1. Hint mask is flattened: [B, 196] → [B, 196]
# 2. Converted to boolean: 0→True (hint), 1→False (no hint)
# 3. Used in attention mechanism to guide predictions
# 4. Hint locations use ground truth color information
```

---

## 5. Model Architecture Details

### Vision Transformer (ViT-B)

```
Input: [B, 196, 768]
    ↓
12 × TransformerBlock
    ├─ LayerNorm
    ├─ MultiHeadAttention (12 heads, 64 dims each)
    │  └─ Relative Positional Bias (14×14 window)
    ├─ MLP (768 → 3072 → 768)
    └─ Residual Connections
    ↓
Output: [B, 196, 768]
```

### Relative Positional Bias (RPB)

```python
# Instead of absolute position embeddings
# Use relative distances between patches

# For 14×14 window:
# - Relative distance range: [-13, 13]
# - RPB table shape: [27×27, 12 heads]
# - Computed for each attention head

# Benefits:
# - Better generalization to different image sizes
# - More efficient than absolute embeddings
# - Captures local spatial relationships
```

### Local Stabilizing Layer

```python
# Reduces artifacts from pixel shuffling
# Applied after upsampling

# Process:
# 1. Extract local patches around each pixel
# 2. Compute local statistics (mean, std)
# 3. Stabilize predictions using local context
# 4. Smooth transitions between patches
```

---

## 6. Loss Function

### Training Loss

```python
# L1 Loss (Mean Absolute Error)
loss = L1Loss(predicted_AB, ground_truth_AB)

# Why L1 instead of L2?
# - More robust to outliers
# - Better for color prediction
# - Encourages sparse predictions
```

### Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
```python
def psnr(img1, img2):
    # Higher is better (typical: 20-30 dB)
    # Measures pixel-level accuracy
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

#### LPIPS (Learned Perceptual Image Patch Similarity)
```python
# Lower is better
# Measures perceptual similarity using pretrained network
# More aligned with human perception than PSNR
```

---

## 7. Key Implementation Files

### `modeling.py` (578 lines)

**Main Classes:**
- `Attention`: Multi-head attention with RPB
- `TransformerBlock`: Transformer encoder block
- `iColoriT`: Main model class

**Key Methods:**
```python
class iColoriT(nn.Module):
    def __init__(self, ...):
        # Initialize patch embedding, transformer blocks
        
    def forward(self, images_lab, bool_hint):
        # images_lab: [B, 3, H, W] in LAB space
        # bool_hint: [B, num_patches] binary mask
        # Returns: [B, num_patches, patch_size²×2]
```

### `infer.py` (198 lines)

**Main Function:**
```python
def main(args):
    # 1. Load model and checkpoint
    # 2. Create validation dataset
    # 3. Run inference on batches
    # 4. Compute PSNR metrics
    # 5. Save colorized images
```

### `datasets.py` (198 lines)

**Main Classes:**
- `DataAugmentationForIColoriT`: Training augmentation
- `DataTransformationForIColoriT`: Validation transformation
- `DataTransformationFixedHint`: Fixed hint transformation

### `utils.py` (671 lines)

**Key Functions:**
- `rgb2lab()`: RGB → LAB conversion
- `lab2rgb()`: LAB → RGB conversion
- `psnr()`: PSNR computation
- `seed_worker()`: Random seed for data loading

### `hint_generator.py` (87 lines)

**Main Classes:**
- `RandomHintGenerator`: Random hint generation
- `InteractiveHintGenerator`: User-interactive hints

---

## 8. Extending the System

### Add Custom Loss Function

```python
# In losses.py
def custom_loss(predicted, ground_truth):
    # Your loss computation
    return loss

# In train.py
loss = custom_loss(outputs, labels)
```

### Add New Model Variant

```python
# In modeling.py
@register_model
def icolorit_custom(pretrained=False, **kwargs):
    model = iColoriT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=512,  # Custom dimension
        depth=12,
        num_heads=8,
        mlp_ratio=4.,
        **kwargs
    )
    return model
```

### Use Different Backbone

```python
# In infer.py
model = create_model(
    'icolorit_small_4ch_patch16_224',  # Different variant
    pretrained=False,
    **model_args
)
```

### Custom Dataset

```python
# In datasets.py
class CustomDataset(Dataset):
    def __init__(self, img_dir, hint_dir):
        self.img_list = os.listdir(img_dir)
        self.hint_list = os.listdir(hint_dir)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        hint = load_hint(self.hint_list[idx])
        return img, hint
```

---

## 9. Performance Optimization

### Inference Speed

```python
# Use smaller model
model = create_model('icolorit_tiny_4ch_patch16_224')

# Reduce batch size
--batch_size=1

# Use GPU
--device=cuda

# Use mixed precision
with torch.cuda.amp.autocast():
    outputs = model(images_lab, bool_hint)
```

### Memory Optimization

```python
# Gradient checkpointing (training)
model.gradient_checkpointing_enable()

# Reduced precision
model.half()  # Use float16

# Smaller input size
--input_size=192
```

---

## 10. Debugging Tips

### Check Model Output Shape

```python
import torch
from timm.models import create_model

model = create_model('icolorit_base_4ch_patch16_224')
x = torch.randn(1, 3, 224, 224)
hint = torch.ones(1, 196, dtype=torch.bool)

output = model(x, hint)
print(output.shape)  # Should be [1, 196, 512]
```

### Verify Data Pipeline

```python
from datasets import build_fixed_validation_dataset

dataset = build_fixed_validation_dataset(args)
batch = dataset[0]
print(f"Image shape: {batch[0][0].shape}")
print(f"Hint shape: {batch[0][1].shape}")
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir=./output

# Check loss
tail -f output/log.txt
```

---

## 11. Common Issues & Solutions

### Issue: Model outputs NaN

**Cause:** Numerical instability in attention
**Solution:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: Poor colorization quality

**Cause:** Insufficient hints or wrong hint locations
**Solution:**
```python
# Increase hints
--num_hints=50

# Use better hint generation
# Ensure hints are at important regions
```

### Issue: Slow inference

**Cause:** CPU processing or large batch size
**Solution:**
```python
# Use GPU
--device=cuda

# Reduce batch size
--batch_size=1

# Use smaller model
'icolorit_tiny_4ch_patch16_224'
```

---

## 12. References

### Paper
- **Title**: iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer
- **Authors**: Jooyeol Yun, Sanghyeon Lee, Minho Park, Jaegul Choo
- **Conference**: WACV 2023
- **Link**: https://arxiv.org/abs/2207.06831

### Related Work
- **ViT**: Vision Transformer (Dosovitskiy et al., 2020)
- **DeiT**: Data-efficient Image Transformers (Touvron et al., 2021)
- **BEiT**: BERT Pre-Training of Image Transformers (Bao et al., 2022)

### Libraries
- **timm**: PyTorch Image Models (https://github.com/rwightman/pytorch-image-models)
- **einops**: Tensor operations (https://github.com/arogozhnikov/einops)
- **LPIPS**: Perceptual similarity (https://github.com/richzhang/PerceptualSimilarity)

---

*Last updated: November 20, 2025*
*For questions, refer to DOCUMENTATION.md or QUICK_START.md*
