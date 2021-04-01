#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#
# Copied from egs/sre16/v1/local/nnet3/xvector/tuning/run_xvector_1a.sh (commit e082c17d4a8f8a791428ae4d9f7ceb776aef3f0b).
#
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2018_icassp_xvectors.pdf

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=data/train
nnet_dir=exp/xvector_nnet_1a/
egs_dir=exp/xvector_nnet_1a/egs

trainFeatDir=data/train_combined_no_sil
trainXvecDir=xvectors/torch_xvector_1a/train
testFeatDir=data/voxceleb1_test
testXvecDir=xvectors/torch_xvector_1a/test

cuda_device_id=0


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -le 6 ]; then
  echo "$0: Getting neural network training egs";
  # Dump egs
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$data" $egs_dir

  # Train the model
  CUDA_VISIBLE_DEVICES=$cuda_device_id python -m torch.distributed.launch --nproc_per_node=1 \
    local/torch_xvector/train.py \
      --stage $stage \
      --nnet-dir $nnet_dir \
      --egs-dir $egs_dir \
      $egs_dir

  modelDir=models/`ls -t | head -n1`
fi

# STAGE 7: Extract X-Vectors
if [ $stage -le 7 ]; then
  modelDir=models/`ls models/ -t | head -n1`

  # echo python local/torch_xvector/extract.py $modelDir $trainFeatDir $trainXvecDir
  # CUDA_VISIBLE_DEVICES=$cuda_device_id python local/torch_xvector/extract.py $modelDir $trainFeatDir $trainXvecDir

  echo python local/torch_xvector/extract.py $modelDir $testFeatDir $testXvecDir
  python local/torch_xvector/extract.py $modelDir $testFeatDir $testXvecDir
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123

# # STAGE 8: 
# if [ $stage -le 8 ]; then
#   # Reproducing voxceleb results
#   # Compute the mean vector for centering the evaluation xvectors.
#   $train_cmd $trainXvecDir/log/compute_mean.log \
#     ivector-mean scp:$trainXvecDir/xvector.scp \
#     $trainXvecDir/mean.vec

#   # This script uses LDA to decrease the dimensionality prior to PLDA.
#   lda_dim=200
#   $train_cmd $trainXvecDir/log/lda.log \
#     ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#     "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- |" \
#     ark:$trainFeatDir/utt2spk $trainXvecDir/transform.mat

#   # Train the PLDA model.
#   $train_cmd $trainXvecDir/log/plda.log \
#     ivector-compute-plda ark:$trainFeatDir/spk2utt \
#     "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
#     $trainXvecDir/plda

# fi

# if [ $stage -le 9 ]; then

#   $train_cmd $testXvecDir/log/voxceleb1_test_scoring.log \
#     ivector-plda-scoring --normalize-length=true \
#     "ivector-copy-plda --smoothing=0.0 $trainXvecDir/plda - |" \
#     "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $testXvecDir/scores_voxceleb1_test

#   eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $testXvecDir/scores_voxceleb1_test) 2> /dev/null`
#   mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $testXvecDir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#   mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $testXvecDir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#   echo "EER: $eer%"
#   echo "minDCF(p-target=0.01): $mindcf1"
#   echo "minDCF(p-target=0.001): $mindcf2"

# fi


exit 0;
