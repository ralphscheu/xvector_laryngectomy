#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

. ./utils/parse_options.sh

$train_cmd logs/pathologic_voices_CTRL_PARE_LARY/plot_xvectors.log \
    python local/pathologic_voices/plot_xvectors.py \
        `pwd`/xvectors/pathologic_voices_CTRL_PARE_LARY/xvector_normalized.scp \
        `pwd`/plots
