import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import argparse
import sys
import subprocess
from kaldi_python_io import ScriptReader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime
import matplotlib.cm as cm
import pandas as pd


def save_plot(output_dir, filename):
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    save_filepath ='{}/{}__{}.png'.format(output_dir, filename, timestamp)
    plt.savefig(save_filepath, bbox_inches='tight')
    print("Saved plot to {}".format( save_filepath ))


def plot_speakergroups(xvectors, title, output_dir, include_PARE=False):
    """ create scatter plot with colors representing speaker groups """
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title(title)
    pointsize = 160
    
    x_LARY = xvectors.loc[xvectors.speaker_group == 'LARY']
    embeddings_LARY = numpy.vstack(x_LARY.embedding.values)
    ax.scatter(embeddings_LARY[:,0], embeddings_LARY[:,1],
        c='tab:red', label='LARY', s=pointsize)
    
    if include_PARE:
        x_PARE = xvectors.loc[xvectors.speaker_group == 'PARE']
        embeddings_PARE = numpy.vstack(x_PARE.embedding.values)
        if x_PARE.size == 0:
            sys.exit("No x-vectors found for PARE!")
        ax.scatter(embeddings_PARE[:,0], embeddings_PARE[:,1],
                c='tab:olive', label='PARE', s=pointsize)
    
    x_CTRL = xvectors.loc[xvectors.speaker_group == 'CTRL']
    embeddings_CTRL = numpy.vstack(x_CTRL.embedding.values)
    ax.scatter(embeddings_CTRL[:,0], embeddings_CTRL[:,1],
        c='tab:blue', label='CTRL', s=pointsize)
   
    plt.axis('off')
    ax.legend(fontsize=32)
    save_plot(output_dir, title)


def plot_scores(xvectors, score_col, title, output_dir, annotate=False):
    """ create scatter plot with colors visualizing scores """
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    pointsize = 160
    colormap = 'RdYlGn'
    colormap_score_factor = -1
    
    xvectors_LARY = xvectors.loc[xvectors.speaker_group == 'LARY']
    embeddings_LARY = numpy.vstack(xvectors_LARY.embedding.values)
    ax.scatter(embeddings_LARY[:,0], embeddings_LARY[:,1],
        c=xvectors_LARY[score_col] * colormap_score_factor, 
        marker="^", label='LARY', s=pointsize, cmap=colormap)
    if annotate:
        for row in xvectors_LARY.iterrows():
            row = row[1]  # remove index
            ax.annotate(round(row[score_col], 2), row.embedding)
    
    xvectors_CTRL = xvectors.loc[xvectors.speaker_group == 'CTRL']
    embeddings_CTRL = numpy.vstack(xvectors_CTRL.embedding.values)
    ax.scatter(embeddings_CTRL[:,0], embeddings_CTRL[:,1],
        c=xvectors_CTRL[score_col] * colormap_score_factor,
         marker="o", label='CTRL', s=pointsize, cmap=colormap)
    if annotate:
        for row in xvectors_CTRL.iterrows():
            row = row[1]  # remove index
            ax.annotate(round(row[score_col], 2), row.embedding)
   
    plt.axis('off')
    ax.legend(fontsize=32)
    save_plot(output_dir, title)


def plot_scores_histograms(scores_LARY, scores_CTRL):
    fig, axes = plt.subplots(nrows=2, ncols=3)
    plt.subplots_adjust(wspace=.6, hspace=.5)
    axes[0, 0].set_title("LARY effort")
    scores_LARY['effort'].plot.hist(alpha=0.5, ax=axes[0,0])
    axes[0, 1].set_title("LARY intell")
    scores_LARY['intell'].plot.hist(alpha=0.5, ax=axes[0,1])
    axes[0, 2].set_title("LARY overall")
    scores_LARY['overall'].plot.hist(alpha=0.5, ax=axes[0,2])
    axes[1, 0].set_title("CTRL effort")
    scores_CTRL['effort'].plot.hist(alpha=0.5, ax=axes[1,0])
    axes[1, 1].set_title("CTRL intell")
    scores_CTRL['intell'].plot.hist(alpha=0.5, ax=axes[1,1])
    axes[1, 2].set_title("CTRL overall")
    scores_CTRL['overall'].plot.hist(alpha=0.5, ax=axes[1,2])
    save_plot("plots", "mean_scores_hist")


def parse_ages(filename):
    """ parse ages labels file into dictionary """
    with open(filename):
        pass


def get_mean_scores(f_crits, f_scores):
    criteria = pd.read_table(f_crits, header=None, encoding="iso-8859-1")[0].values
    mean_scores = pd.read_table(f_scores, ",", names=criteria)
    mean_scores.rename(columns={'Untersucher': 'rater', 'PatientNr': 'utt', 'file': 'utt', 'Anstr': 'effort', 'Verst': 'intell', 'Gesamt': 'overall'}, inplace=True)
    mean_scores = mean_scores.groupby(["utt"]).mean().reset_index()  # compute mean score across all raters
    return mean_scores[['utt', 'effort', 'intell', 'overall']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp file for xvectors to plot')
    parser.add_argument('output_dir', help='directory to save plot into')
    args = parser.parse_args()
    PLOT_DIM = 2


    xvectors = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ScriptReader(args.scp_input)
    for key, mat in reader:
        xvectors = xvectors.append({
            'utt': key.split('_', maxsplit=1)[1], 
            'speaker_group': key.split('_', maxsplit=1)[0], 
            'embedding': mat}, ignore_index=True)
    print("Reducing dimensions using t-SNE...")
    xvectors.embedding = list(TSNE(n_components=PLOT_DIM).fit_transform( numpy.vstack(xvectors.embedding.values) ))


    # plot colored speakergroups with and w/o partial resections
    plot_speakergroups(xvectors, "pathologic_voices_CTRL_LARY", 
        args.output_dir, include_PARE=False)
    plot_speakergroups(xvectors, "pathologic_voices_CTRL_PARE", 
        args.output_dir, include_PARE=True)


    # get effort scores
    mean_scores_LARY = get_mean_scores(
        "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", 
        "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")
    # print("=== LARY scores ===")
    # print(mean_scores_LARY.describe())

    mean_scores_CTRL = get_mean_scores(
        "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits",
        "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")
    # print("=== CTRL scores ===")
    # print(mean_scores_CTRL.describe())

    mean_scores = pd.concat([mean_scores_LARY, mean_scores_CTRL])  # concat the dictionaries containing scores

    plot_scores_histograms(mean_scores_LARY, mean_scores_CTRL)
    
    for crit in ['effort', 'intell', 'overall']:
        plot_scores(xvectors.merge(mean_scores, on="utt"), crit, "pathologic_voices_CTRL_LARY - {}".format(crit), args.output_dir)


if __name__ == "__main__":
    main()
