#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

stage=7

. ./utils/parse_options.sh


if [ $stage -le 0 ]; then
  ./run.sh --stage 0
fi

if [ $stage -le 1 ]; then
  ./run.sh --stage 1
fi

if [ $stage -le 2 ]; then
  ./run.sh --stage 2
fi

if [ $stage -le 3 ]; then
  ./run.sh --stage 3
fi

if [ $stage -le 4 ]; then
  ./run.sh --stage 4
fi

if [ $stage -le 5 ]; then
  ./run.sh --stage 5
fi

if [ $stage -le 6 ]; then
  ./run.sh --stage 6
fi

if [ $stage -le 7 ]; then
  ./run.sh --stage 7
fi

if [ $stage -le 8 ]; then
  ./run.sh --stage 8
fi

if [ $stage -le 9 ]; then
  ./run.sh --stage 9
fi

if [ $stage -le 10 ]; then
  ./run.sh --stage 10
fi

exit 0;
