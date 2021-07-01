#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from viz_utils import get_mean_scores, parse_ages, plot_scores_histograms, save_plot

ages = parse_ages(
    "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.ages",
    "/mnt/speechdata/pathologic_voices/teilres85/labels/teilresDVD1+2.ages",
    "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.measures"
    )

mean_scores_laryng = pd.merge(
    get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores"), 
    ages['laryng'], on='utt')
    
mean_scores_partres = get_mean_scores("/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")

mean_scores_ctrl = pd.merge(
    get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores"),
    ages['ctrl'], on='utt')


print(ages['partres'].shape)

print("laryng:  span={:.2f}-{:.2f} | mean={:.2f} | median={:.2f}".format(mean_scores_laryng.age.min(), mean_scores_laryng.age.max(), mean_scores_laryng.age.mean(), mean_scores_laryng.age.median()))
print("partres: span={:.2f}-{:.2f} | mean={:.2f} | median={:.2f}".format(ages['partres'].age.min(), ages['partres'].age.max(), ages['partres'].age.mean(), ages['partres'].age.median()))
print("ctrl: span={:.2f}-{:.2f} | mean={:.2f} | median={:.2f}".format(mean_scores_ctrl.age.min(), mean_scores_ctrl.age.max(), mean_scores_ctrl.age.mean(), mean_scores_ctrl.age.median()))

# mean_scores_laryng.drop(columns=['age']).plot.box()
# plt.savefig('laryng_allscores_span.png')
# pd.plotting.scatter_matrix(mean_scores_laryng[['utt', 'age', 'effort', 'intell', 'overall']], figsize=(10,10))
# plt.savefig('laryng_keyscores_matrix.png')
# pd.plotting.scatter_matrix(mean_scores_laryng, figsize=(10,10))
# plt.savefig('laryng_allscores_matrix.png')

# print("partres")
# print(mean_scores_partres.describe())
# plt.savefig('partres_allscores_span.png')
# pd.plotting.scatter_matrix(mean_scores_partres[['utt', 'effort', 'intell', 'overall']], figsize=(10,10))
# plt.savefig('partres_keyscores_matrix.png')
# pd.plotting.scatter_matrix(mean_scores_partres, figsize=(10,10))
# plt.savefig('partres_allscores_matrix.png')

# print("ctrl")
# print(mean_scores_ctrl.describe())
# mean_scores_ctrl.plot.box()
# plt.savefig('ctrl_allscores_span.png')
# pd.plotting.scatter_matrix(mean_scores_ctrl[['utt', 'effort', 'intell', 'overall']], figsize=(10,10))
# plt.savefig('ctrl_keyscores_matrix.png')
# pd.plotting.scatter_matrix(mean_scores_ctrl, figsize=(10,10))
# plt.savefig('ctrl_allscores_matrix.png')

# visualize scores stats
# plot_scores_histograms(mean_scores_laryng, mean_scores_partres)


# print("created all plots.")
