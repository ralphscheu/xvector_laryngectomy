#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


def save_plot(output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    save_filepath = '{}/{}__{}'.format(output_dir, timestamp, filename)
    plt.savefig(save_filepath + ".png", bbox_inches='tight')
    plt.savefig(save_filepath + ".eps", format='eps', bbox_inches='tight')
    print("Saved plot to {}".format( save_filepath ))


def parse_ages(filename):
    """ parse ages labels file into dictionary """
    return pd.read_table(filename, names=['utt', 'age'])


def get_mean_scores(f_crits, f_scores):
    criteria = pd.read_table(f_crits, header=None, encoding="iso-8859-1")[0].values
    mean_scores = pd.read_table(f_scores, ",", names=criteria)
    mean_scores.rename(columns={'Untersucher': 'rater', 'PatientNr': 'utt', 'file': 'utt', 'Anstr': 'effort', 'Verst': 'intell', 'Gesamt': 'overall'}, inplace=True)
    mean_scores = mean_scores.groupby(["utt"]).mean().reset_index()  # compute mean score across all raters
    return mean_scores


def plot_scores_histograms(scores_laryng, scores_partres, bar_color='white', edgecolor='black', hatch='xxxx'):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 8))
    plt.subplots_adjust(wspace=.32, hspace=.32)

    axes[0,0].set_title("laryng intell")
    scores_laryng['intell'].plot.hist(alpha=0.5, ax=axes[0,0], color=bar_color, edgecolor=edgecolor, hatch=hatch)
    axes[1,0].set_title("laryng overall")
    scores_laryng['overall'].plot.hist(alpha=0.5, ax=axes[1,0], color=bar_color, edgecolor=edgecolor, hatch=hatch)
    axes[2,0].set_title("laryng effort")
    scores_laryng['effort'].plot.hist(alpha=0.5, ax=axes[2,0], color=bar_color, edgecolor=edgecolor, hatch=hatch)

    axes[0,1].set_title("partres intell")
    scores_partres['intell'].plot.hist(alpha=0.5, ax=axes[0,1], color=bar_color, edgecolor=edgecolor, hatch=hatch)
    axes[1,1].set_title("partres overall")
    scores_partres['overall'].plot.hist(alpha=0.5, ax=axes[1,1], color=bar_color, edgecolor=edgecolor, hatch=hatch)
    axes[2,1].set_title("partres effort")
    scores_partres['effort'].plot.hist(alpha=0.5, ax=axes[2,1], color=bar_color, edgecolor=edgecolor, hatch=hatch)

    save_plot("plots", "scores_hist")