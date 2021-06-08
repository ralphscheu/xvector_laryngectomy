#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ./README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

echo "$0 $@"  # Print the command line for logging.


# point to source data
voxceleb1_root=/mnt/md0/data/VoxCeleb1
voxceleb2_root=/mnt/md0/data/VoxCeleb2
musan_root=/mnt/md0/data/musan

# location of mfcc and vad files
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# configure training and test dataset
trainFeatDir=data/train_combined_no_sil
testFeatDir=data/voxceleb1_test_no_sil
voxceleb1_trials=data/voxceleb1_test/trials  # The trials file is downloaded by local/make_voxceleb1_v2.pl.

# configure the model to run
nnet_dir=exp/torch_xvector_1a
nnet_name=xvector

# which CUDA device to use (defaults to single gpu)
nproc=1
cuda_device_id=0

stage=0
modelType=xvector
numAttnHeads=5

. ./utils/parse_options.sh

modelDir=models/$nnet_name

# X-vector directories
trainXvecDir=xvectors/$nnet_name/train
testXvecDir=xvectors/$nnet_name/test

echo "trainXvecDir: $trainXvecDir"
echo "testXvecDir:  $testXvecDir"
echo "modelDir:     $modelDir"

if [ $stage -eq 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,323 speakers and 1,276,888 utterances.
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
fi


if [ $stage -eq 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi


# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -eq 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip -O /tmp/rirs_noises.zip
    unzip /tmp/rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi


if [ $stage -eq 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi


if [ $stage -eq 4 ]; then
  echo "Stage $stage: Prepare features to generate examples for xvector training"

  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.

  local/torch_xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_combined $trainFeatDir exp/train_combined_no_sil
  utils/fix_data_dir.sh $trainFeatDir


  # Preparing the test features as well. This will be used only during testing
  local/torch_xvector/prepare_feats_for_egs.sh --nj 10 --cmd "$train_cmd" \
    data/voxceleb1_test $testFeatDir exp/voxceleb1_test_no_sil
  utils/fix_data_dir.sh $testFeatDir
fi


if [ $stage -eq 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 4s (400 frames) per utterance.
  min_len=400
  mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
  mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
  mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
  mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_no_sil
fi


if [ $stage -eq 6 ]; then

  echo "Stage $stage: Getting neural network training egs";

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

  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$trainFeatDir" $nnet_dir/egs

fi


if [ $stage -eq 7 ]; then
  echo "Stage $stage: Train the model"

  CUDA_VISIBLE_DEVICES=$cuda_device_id \
    $train_cmd logs/${modelType}__$(date -u '+%Y%m%dT%H%M%S')_train.log \
      python -m torch.distributed.launch --nproc_per_node=$nproc \
        local/torch_xvector/train.py \
          --modelType $modelType \
          --numAttnHeads $numAttnHeads \
          --batchSize 32 \
          exp/torch_xvector_1a/egs
fi


if [ $stage -eq 8 ]; then
  echo "Stage $stage: Extract X-Vectors (+ visualize)"

  CUDA_VISIBLE_DEVICES=$cuda_device_id python local/torch_xvector/extract.py \
    --numSplits 400 \
    --modelType $modelType \
    --numAttnHeads $numAttnHeads \
    $modelDir $trainFeatDir $trainXvecDir
  # concat separate scp files into one
  cat $trainXvecDir/split400/xvector_split400_*.scp > $trainXvecDir/xvector.scp

  CUDA_VISIBLE_DEVICES=$cuda_device_id python local/torch_xvector/extract.py \
    --modelType $modelType \
    --numAttnHeads $numAttnHeads \
    $modelDir $testFeatDir $testXvecDir
fi



# STAGE 9: COMPUTE MEAN VECTORS, TRAIN PLDA MODEL, COMPUTE SCORES
if [ $stage -eq 9 ]; then
  # Reproducing voxceleb results
  echo "Compute the mean vector for centering the evaluation xvectors..."
  $train_cmd $trainXvecDir/log/compute_mean.log \
    ivector-mean scp:$trainXvecDir/xvector.scp \
    $trainXvecDir/mean.vec

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  echo "Decrease dimensionality using LDA..."
  $train_cmd $trainXvecDir/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- |" \
    ark:$trainFeatDir/utt2spk $trainXvecDir/transform.mat

  echo "Train PLDA..."
  $train_cmd $trainXvecDir/log/plda.log \
    ivector-compute-plda ark:$trainFeatDir/spk2utt \
    "ark:ivector-subtract-global-mean scp:$trainXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $trainXvecDir/plda

  
  
  $train_cmd $testXvecDir/log/save_xvec_for_plot.log \
    ivector-normalize-length \
      "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- |" \
      ark,scp:$testXvecDir/xvec_for_plot.ark,$testXvecDir/xvec_for_plot.scp

  python local/plot_xvec.py $testXvecDir/xvec_for_plot.scp voxceleb1_test ./plots --dim-reduction-method=tsne

fi

if [ $stage -eq 10 ]; then

  scores_dir=$testXvecDir/scores_voxceleb1_test_cosine
  cat $voxceleb1_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      $scores_dir

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $scores_dir) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 $scores_dir $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $scores_dir $voxceleb1_trials 2> /dev/null`
  printf "\nCosine scoring:\n"
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"


  scores_dir=$testXvecDir/scores_plda
  $train_cmd $testXvecDir/log/plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $trainXvecDir/plda - |" \
    "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $scores_dir

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $scores_dir) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $scores_dir $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $scores_dir $voxceleb1_trials 2> /dev/null`
  printf "\nPLDA scoring:\n"
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

exit 0;
