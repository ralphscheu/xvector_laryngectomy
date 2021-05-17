#!/usr/bin/env python3
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from kaldi_python_io import ScriptReader
from plot_xvectors import get_mean_scores, parse_ages


laryng_df = pd.merge(
    get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores"),
    parse_ages("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.ages"),
    on='utt')
# print(list(laryng_df.columns))
ctrl_df = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")


laryng_df.drop(columns=['age']).plot.box()
plt.savefig('laryng_allscores_span.png')
pd.plotting.scatter_matrix(laryng_df[['utt', 'age', 'effort', 'intell', 'overall']], figsize=(10,10))
plt.savefig('laryng_keyscores_matrix.png')
pd.plotting.scatter_matrix(laryng_df, figsize=(10,10))
plt.savefig('laryng_allscores_matrix.png')


ctrl_df.plot.box()
plt.savefig('ctrl_allscores_span.png')
pd.plotting.scatter_matrix(ctrl_df[['utt', 'effort', 'intell', 'overall']], figsize=(10,10))
plt.savefig('ctrl_keyscores_matrix.png')
pd.plotting.scatter_matrix(ctrl_df, figsize=(10,10))
plt.savefig('ctrl_allscores_matrix.png')


print("created all plots.")
