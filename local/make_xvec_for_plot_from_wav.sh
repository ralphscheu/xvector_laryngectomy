#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

trainXvecDir=xvectors/torch_xvector_1a/train
wav_root=/mnt/speechdata/pathologic_voices/altersstimme110_cut/WAV
dataset_name=altersstimme110_cut
dataset_prefix=CTRL

. ./utils/parse_options.sh

data_dir=./data/$dataset_name
mfcc_dir=`pwd`/mfcc/$dataset_name
vad_dir=`pwd`/mfcc/$dataset_name

# create data directory
mkdir -p $data_dir

# create wav.scp
if [ -f $data_dir/wav.scp ]; then
    mkdir -p $data_dir/.backup
    mv $data_dir/wav.scp $data_dir/.backup/wav.scp
fi

for wav_file in $wav_root/*; do
    echo ${dataset_prefix}_$(basename $wav_file .wav) $wav_file >> $data_dir/wav.scp
done


# create utt2spk
if [ -f $data_dir/utt2spk ]; then
    mkdir -p $data_dir/.backup
    mv $data_dir/utt2spk $data_dir/.backup/utt2spk
fi

for wav_file in $wav_root/*; do
    echo ${dataset_prefix}_$(basename $wav_file .wav) ${dataset_prefix}_$(basename $wav_file .wav) >> $data_dir/utt2spk
done


# create spk2utt
if [ -f $data_dir/spk2utt ]; then
    mkdir -p $data_dir/.backup
    mv $data_dir/spk2utt $data_dir/.backup/spk2utt
fi

for wav_file in $wav_root/*; do
    echo ${dataset_prefix}_$(basename $wav_file .wav) ${dataset_prefix}_$(basename $wav_file .wav) >> $data_dir/spk2utt
done


# Make MFCCs and compute the energy-based VAD for each dataset
steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 10 --cmd "$train_cmd" \
    $data_dir exp/${dataset_name}_make_mfcc $mfcc_dir
utils/fix_data_dir.sh $data_dir
sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
    $data_dir exp/${dataset_name}_make_vad $vad_dir
utils/fix_data_dir.sh $data_dir


# This script applies CMVN and removes nonspeech frames.
local/torch_xvector/prepare_feats_for_egs.sh --nj 10 --cmd "$train_cmd" \
    $data_dir ${data_dir}_no_sil exp/${dataset_name}_no_sil
  utils/fix_data_dir.sh ${data_dir}_no_sil


# Extract X-Vectors
modelDir=models/`ls models/ -t | head -n1`
testXvecDir=xvectors/${dataset_name}_no_sil
python local/torch_xvector/extract.py $modelDir ${data_dir}_no_sil $testXvecDir
