#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

. ./utils/parse_options.sh

for i in 1 2 3 4 5; do
    $train_cmd logs/pathologic_voices_CTRL_PARE_LARY/plot_xvectors.log \
        python local/pathologic_voices/plot_xvectors.py \
            `pwd`/xvectors/pathologic_voices_CTRL_PARE_LARY/xvector_normalized.scp \
            `pwd`/plots
done
