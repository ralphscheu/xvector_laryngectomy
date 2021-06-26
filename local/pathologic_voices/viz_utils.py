#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


def save_plot(output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    save_filepath = '{}/{}__{}.png'.format(output_dir, timestamp, filename)
    plt.savefig(save_filepath, bbox_inches='tight')
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


def plot_scores_histograms(scores_laryng, scores_partres):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    axes[0, 0].set_title("laryng effort")
    scores_laryng['effort'].plot.hist(alpha=0.5, ax=axes[0,0])
    axes[0, 1].set_title("laryng intell")
    scores_laryng['intell'].plot.hist(alpha=0.5, ax=axes[0,1])
    axes[0, 2].set_title("laryng overall")
    scores_laryng['overall'].plot.hist(alpha=0.5, ax=axes[0,2])

    axes[1, 0].set_title("partres effort")
    scores_partres['effort'].plot.hist(alpha=0.5, ax=axes[1,0])
    axes[1, 1].set_title("partres intell")
    scores_partres['intell'].plot.hist(alpha=0.5, ax=axes[1,1])
    axes[1, 2].set_title("partres overall")
    scores_partres['overall'].plot.hist(alpha=0.5, ax=axes[1,2])

    save_plot("plots", "scores_hist")