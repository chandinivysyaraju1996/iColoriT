#!/bin/bash

export OMP_NUM_THREADS=1

# Directories
PRED_DIR='results/'
GT_DIR='validation/imgs/'
NUM_HINT=${1:-10}

# Run simple evaluation (PSNR only, no CUDA/network requirements)
python3 evaluate_simple.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT}
