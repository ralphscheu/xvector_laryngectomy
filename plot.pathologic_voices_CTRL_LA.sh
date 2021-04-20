#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh
set -e


$train_cmd logs/altersstimme110_cut/make_xvec_for_plot_from_wav.log \
    local/make_xvec_for_plot_from_wav.sh \
        --wav-root /mnt/speechdata/pathologic_voices/altersstimme110_cut/WAV \
        --dataset-name altersstimme110_cut \
        --dataset-prefix CTRL

$train_cmd logs/laryng41/make_xvec_for_plot_from_wav.log \
    local/make_xvec_for_plot_from_wav.sh \
        --wav-root /mnt/speechdata/pathologic_voices/laryng41/WAV \
        --dataset-name laryng41 \
        --dataset-prefix LA


testXvecDir=xvectors/pathologic_voices_CTRL_LA

# combine xvector sets
mkdir -p xvectors/pathologic_voices_CTRL_LA
rm $testXvecDir/xvector.scp
cat xvectors/altersstimme110_cut_no_sil/xvector.scp xvectors/laryng41_no_sil/xvector.scp >> $testXvecDir/xvector.scp


# Plot X-Vectors
trainXvecDir=xvectors/torch_xvector_1a/train
$train_cmd $testXvecDir/log/xvec_for_plot_$dataset_name.log \
    ivector-normalize-length \
        "ark:ivector-subtract-global-mean $trainXvecDir/mean.vec scp:$testXvecDir/xvector.scp ark:- | transform-vec $trainXvecDir/transform.mat ark:- ark:- |" \
        ark,scp:$testXvecDir/xvec_for_plot.ark,$testXvecDir/xvec_for_plot.scp


$train_cmd logs/pathologic_voices_CTRL_LA/plot_xvec_pathologic_voice.log \
    python local/plot_xvec_pathologic_voice.py \
        $testXvecDir/xvec_for_plot.scp \
        pathologic_voices_CTRL_LA \
        ./plots \
        --dim-reduction-method=tsne
