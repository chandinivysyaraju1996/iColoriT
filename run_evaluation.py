#!/usr/bin/env python
"""
Quick evaluation script for iColoriT colorization results
Usage: python run_evaluation.py
"""

import os
import sys
import subprocess

# Configuration
PRED_DIR = 'results/'
GT_DIR = 'validation/'
HINT_SIZE = 2
NUM_HINTS = 10
SAVE_DIR = 'evaluation_results/'

def main():
    # Check if prediction directory exists
    if not os.path.exists(PRED_DIR):
        print(f"❌ Prediction directory not found: {PRED_DIR}")
        print("Please run inference first using: python run_inference.py")
        sys.exit(1)
    
    # Check if ground truth directory exists
    if not os.path.exists(GT_DIR):
        print(f"❌ Ground truth directory not found: {GT_DIR}")
        sys.exit(1)
    
    print("=" * 70)
    print("📊 iColoriT Evaluation")
    print("=" * 70)
    print(f"Prediction directory: {PRED_DIR}")
    print(f"Ground truth directory: {GT_DIR}")
    print(f"Hint size: {HINT_SIZE}")
    print(f"Number of hints: {NUM_HINTS}")
    print(f"Save directory: {SAVE_DIR}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Run evaluation
    cmd = [
        'python', 'evaluation/evaluate.py',
        f'--pred_dir={PRED_DIR}',
        f'--gt_dir={GT_DIR}',
        f'--hint_size={HINT_SIZE}',
        f'--num_hint={NUM_HINTS}',
        f'--save_dir={SAVE_DIR}',
    ]
    
    print("\n📝 Running command:")
    print(" ".join(cmd))
    print("\n" + "=" * 70 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("✅ Evaluation completed successfully!")
        
        # Display results
        result_file = os.path.join(SAVE_DIR, f'h{HINT_SIZE}-n{NUM_HINTS}.txt')
        if os.path.exists(result_file):
            print(f"\n📄 Results saved to: {result_file}")
            print("\n" + "-" * 70)
            with open(result_file, 'r') as f:
                print(f.read())
            print("-" * 70)
        
        print("=" * 70)
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print(f"❌ Evaluation failed with error code: {e.returncode}")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    main()
