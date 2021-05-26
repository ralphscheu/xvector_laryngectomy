#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e
echo "$0 $@"  # Print the command line for logging.

modelType=xvector

. ./utils/parse_options.sh

CUDA_VISIBLE_DEVICES=6,7 \
$train_cmd logs/${modelType}__event_$(date '+%Y%m%dT%H%M%S').log \
    python3 multigpu_train.py \
        --modelType $modelType \
        --batchSize 32 \
        exp/torch_xvector_1a/egs
