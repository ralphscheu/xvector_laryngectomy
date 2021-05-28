#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e
echo "$0 $@"  # Print the command line for logging.

modelType=xvector
numAttnHeads=12

. ./utils/parse_options.sh

CUDA_VISIBLE_DEVICES=6,7 \
$train_cmd logs/${modelType}__event_$(date -u '+%Y%m%dT%H%M%S').log \
    python3 multigpu_train.py \
        --modelType $modelType \
        --numAttnHeads $numAttnHeads \
        --batchSize 32 \
        exp/torch_xvector_1a/egs
