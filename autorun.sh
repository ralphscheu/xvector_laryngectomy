#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

start_stage=7
cuda_device_id=0

. ./utils/parse_options.sh


for run_stage in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if [ $start_stage -le $run_stage ]; then
    ./run.sh --stage $run_stage
  fi
done

exit 0;
