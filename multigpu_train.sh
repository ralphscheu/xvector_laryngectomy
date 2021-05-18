#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e
echo "$0 $@"  # Print the command line for logging.


CUDA_VISIBLE_DEVICES=4,5,6,7 \
$train_cmd logs/train_xvector_4gpu.log \
    python3 multigpu_train.py exp/torch_xvector_1a/egs

# CUDA_VISIBLE_DEVICES=4,5 \
# $train_cmd logs/train_xvector_2gpu.log \
#     python3 multigpu_train.py exp/torch_xvector_1a/egs

# CUDA_VISIBLE_DEVICES=3 \
# $train_cmd logs/train_xvector_1gpu.log \
#     python3 multigpu_train.py
