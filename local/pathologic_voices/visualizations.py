#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from viz_utils import get_mean_scores, parse_ages, plot_scores_histograms, save_plot


mean_scores_laryng = get_mean_scores(
    "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", 
    "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")

mean_scores_partres = get_mean_scores(
    "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", 
    "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")

mean_scores_ctrl = get_mean_scores(
    "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits",
    "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")


mean_scores_laryng = pd.merge(
    get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores"),
    parse_ages("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.ages"),
    on='utt')


print("laryng")
print("age: span={}-{} | mean={}".format(mean_scores_laryng.age.min(), mean_scores_laryng.age.max(), mean_scores_laryng.age.mean()))
print(mean_scores_laryng.describe())

mean_scores_laryng.drop(columns=['age']).plot.box()
plt.savefig('laryng_allscores_span.png')
pd.plotting.scatter_matrix(mean_scores_laryng[['utt', 'age', 'effort', 'intell', 'overall']], figsize=(10,10))
plt.savefig('laryng_keyscores_matrix.png')
pd.plotting.scatter_matrix(mean_scores_laryng, figsize=(10,10))
plt.savefig('laryng_allscores_matrix.png')

print("partres")
print(mean_scores_partres.describe())
plt.savefig('partres_allscores_span.png')
pd.plotting.scatter_matrix(mean_scores_partres[['utt', 'effort', 'intell', 'overall']], figsize=(10,10))
plt.savefig('partres_keyscores_matrix.png')
pd.plotting.scatter_matrix(mean_scores_partres, figsize=(10,10))
plt.savefig('partres_allscores_matrix.png')

print("ctrl")
print(mean_scores_ctrl.describe())
mean_scores_ctrl.plot.box()
plt.savefig('ctrl_allscores_span.png')
pd.plotting.scatter_matrix(mean_scores_ctrl[['utt', 'effort', 'intell', 'overall']], figsize=(10,10))
plt.savefig('ctrl_keyscores_matrix.png')
pd.plotting.scatter_matrix(mean_scores_ctrl, figsize=(10,10))
plt.savefig('ctrl_allscores_matrix.png')

# visualize scores stats
plot_scores_histograms(mean_scores_laryng, mean_scores_partres, mean_scores_ctrl)


print("created all plots.")
