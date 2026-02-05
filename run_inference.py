#!/usr/bin/env python
"""
Quick inference script for iColoriT colorization
Usage: python run_inference.py
"""

import os
import sys
import subprocess

# Configuration
CHECKPOINT_PATH = 'checkpoints/icolorit_base_4ch_patch16_512.pth'
VAL_DATA_PATH = 'validation/'
VAL_HINT_DIR = 'ctest10k_hint/'
PRED_DIR = 'results/'
BATCH_SIZE = 2
NUM_WORKERS = 0
DEVICE = 'cuda'  # Change to 'cpu' if no GPU available

def main():
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    # Check if validation data exists
    if not os.path.exists(VAL_DATA_PATH):
        print(f"❌ Validation data not found: {VAL_DATA_PATH}")
        sys.exit(1)
    
    # Check if hints exist
    if not os.path.exists(VAL_HINT_DIR):
        print(f"❌ Hint directory not found: {VAL_HINT_DIR}")
        sys.exit(1)
    
    print("=" * 70)
    print("🎨 iColoriT Inference")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation data: {VAL_DATA_PATH}")
    print(f"Hints directory: {VAL_HINT_DIR}")
    print(f"Output directory: {PRED_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(PRED_DIR, exist_ok=True)
    
    # Run inference
    cmd = [
        'python', 'infer.py',
        f'--model_path={CHECKPOINT_PATH}',
        f'--val_data_path={VAL_DATA_PATH}',
        f'--val_hint_dir={VAL_HINT_DIR}',
        f'--pred_dir={PRED_DIR}',
        f'--batch_size={BATCH_SIZE}',
        f'--num_workers={NUM_WORKERS}',
        f'--device={DEVICE}',
    ]
    
    print("\n📝 Running command:")
    print(" ".join(cmd))
    print("\n" + "=" * 70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("✅ Inference completed successfully!")
        print(f"📁 Results saved to: {PRED_DIR}")
        print("=" * 70)
        
        # List output directories
        if os.path.exists(PRED_DIR):
            output_dirs = sorted([d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d))])
            if output_dirs:
                print("\n📊 Output directories created:")
                for d in output_dirs:
                    count = len(os.listdir(os.path.join(PRED_DIR, d)))
                    print(f"   - {d}: {count} images")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print(f"❌ Inference failed with error code: {e.returncode}")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    main()
