#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e

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

