#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

modelDir=models/`ls models/ -t | head -n1`  # load latest model by default

# temporary override: use modelType_xvecTDNN_event_20210409T145152 by default
# TODO: remove temporary model override
modelDir=models/modelType_xvecTDNN_event_20210409T145152

cuda_device_id=0
trainXvecDir=xvectors/torch_xvector_1a/train  # xvectors of training set

. ./utils/parse_options.sh


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


for dataset_name in pathologic_voices_CTRL_LARY pathologic_voices_CTRL_PARE_LARY; do

    data_dir=`pwd`/data/$dataset_name
    mfcc_dir=`pwd`/mfcc/$dataset_name
    vad_dir=`pwd`/mfcc/$dataset_name
    xvecDir=`pwd`/xvectors/$dataset_name

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

    # Extract X-Vectors
    $train_cmd logs/$dataset_name/extract_xvectors.log \
        CUDA_VISIBLE_DEVICES=$cuda_device_id \
            python local/torch_xvector/extract.py $modelDir ${data_dir}_no_sil $xvecDir \
                --modelType xvecTDNN_Legacy
    
    # Normalize X-Vectors
    $train_cmd logs/$dataset_name/normalize_xvectors.log \
        ivector-normalize-length \
            "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$xvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- |" \
            ark,scp:$xvecDir/xvector_normalized.ark,$xvecDir/xvector_normalized.scp
    
done


$train_cmd logs/pathologic_voices_CTRL_PARE_LARY/plot_xvectors.log \
    python local/pathologic_voices/plot_xvectors.py \
        `pwd`/xvectors/pathologic_voices_CTRL_PARE_LARY/xvector_normalized.scp \
        `pwd`/plots