export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}

# NOTE: This script requires pre-computed hints in the VAL_HINT_DIR
# For images with different names than ImageNet, use simple_infer.py instead
# which generates random hints automatically

# path to model and validation dataset
MODEL_PATH='checkpoints/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='Test/imgs'
VAL_HINT_DIR=''                               # Pre-computed hints for your images
# Set the path to save checkpoints
PRED_DIR='results_egyptian/'

# other options
opt=${2:-}

# batch_size can be adjusted according to the graphics card
python3 simple_infer.py \
    --model_path=${MODEL_PATH} \
    --val_data_path=${VAL_DATA_PATH} \
    --pred_dir=${PRED_DIR} \
    --device cpu \
    $opt