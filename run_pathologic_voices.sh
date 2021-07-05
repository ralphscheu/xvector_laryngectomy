#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

nnet_name=xvector  # default to baseline model
model_type=xvector_legacy
cuda_device_id=0  # default to first GPU in system

. ./utils/parse_options.sh

modelDir=`pwd`/models/$nnet_name
trainXvecDir=`pwd`/xvectors/$nnet_name/train
testXvecDir=`pwd`/xvectors/$nnet_name/test
pathovoicesXvecDir=`pwd`/xvectors/$nnet_name/pathologic_voices

dataset_name=pathologic_voices_CTRL_PARE_LARY
data_dir=`pwd`/data/$dataset_name

# create datasets
$train_cmd logs/altersstimme110_cut/make_dataset.log \
    local/pathologic_voices/make_dataset.sh \
        --wav-root /mnt/speechdata/pathologic_voices/altersstimme110_cut/WAV \
        --dataset-name altersstimme110_cut --dataset-prefix CTRL

$train_cmd logs/teilres85/make_dataset.log \
    local/pathologic_voices/make_dataset.sh \
        --wav-root /mnt/speechdata/pathologic_voices/teilres85/WAV \
        --dataset-name teilres85 --dataset-prefix PARE

$train_cmd logs/laryng41/make_dataset.log \
    local/pathologic_voices/make_dataset.sh \
        --wav-root /mnt/speechdata/pathologic_voices/laryng41/WAV \
        --dataset-name laryng41 --dataset-prefix LARY


# combine datasets
utils/combine_data.sh data/pathologic_voices_CTRL_LARY      data/altersstimme110_cut data/laryng41
utils/combine_data.sh data/pathologic_voices_CTRL_PARE_LARY data/altersstimme110_cut data/laryng41 data/teilres85

mfcc_dir=`pwd`/mfcc/$dataset_name
vad_dir=`pwd`/mfcc/$dataset_name

# Make MFCCs
$train_cmd logs/$dataset_name/make_mfcc.log \
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 10 --cmd "$train_cmd" \
        $data_dir exp/${dataset_name}_make_mfcc $mfcc_dir
utils/fix_data_dir.sh $data_dir

# Compute the energy-based VAD for each dataset
$train_cmd logs/$dataset_name/compute_vad_decision.log \
    sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
        $data_dir exp/${dataset_name}_make_vad $vad_dir
utils/fix_data_dir.sh $data_dir

# This script applies CMVN and removes nonspeech frames.
$train_cmd logs/$dataset_name/prepare_feats_for_egs.log \
    local/torch_xvector/prepare_feats_for_egs.sh --nj 10 --cmd "$train_cmd" \
        $data_dir ${data_dir}_no_sil exp/${dataset_name}_no_sil
utils/fix_data_dir.sh ${data_dir}_no_sil

echo "Extract embeddings..."
$train_cmd logs/$nnet_name/pathologic_voices/extract_xvectors.log \
    CUDA_VISIBLE_DEVICES=$cuda_device_id \
        python local/torch_xvector/extract.py $modelDir ${data_dir}_no_sil $pathovoicesXvecDir \
            --modelType $model_type

echo "Plot embeddings..."
$train_cmd logs/$nnet_name/pathologic_voices/plot_xvectors.log \
    python local/pathologic_voices/plot_xvectors.py \
        "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$pathovoicesXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        --output-dir `pwd`/xvectors/$nnet_name/_plots
