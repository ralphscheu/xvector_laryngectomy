import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
from kaldi_python_io import ScriptReader
from sklearn.manifold import TSNE
from datetime import datetime


def save_plot(output_dir, filename):
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    save_filepath ='{}/{}__{}.png'.format(output_dir, filename, timestamp)
    plt.savefig(save_filepath, bbox_inches='tight')
    print("Saved plot to {}".format( save_filepath ))


def get_outliers(utterances, n=10):
    print("---")
    embs = np.vstack(utterances.embedding.values)
    print("embs:", embs.shape)
    centroid = np.mean(embs, axis=0)
    print("centroid:", centroid)
    df = utterances.copy()
    df["distance"] = df.embedding.apply(lambda x: np.linalg.norm(x - centroid))
    print("distance:", df.distance.min(), df.distance.max())
    return df.sort_values("distance", ascending=False).head(df.shape[0] // 6)


def plot_speakergroups(pathovoices_emb, title, output_dir, annotate_outliers=False, include_VOXCELEB=True, include_PARE=False):
    """ create scatter plot with colors representing speaker groups """
    pointsize = 160
    figsize = (16, 16)
    if annotate_outliers:
        figsize = (22, 22)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1) 
    ax.set_title(title)

    x_LARY = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'LARY']
    embeddings_LARY = np.vstack(x_LARY.embedding.values)
    ax.scatter(embeddings_LARY[:,0], embeddings_LARY[:,1], c='tab:red', label='laryng', s=pointsize)
    if annotate_outliers:
        for _, row in get_outliers(x_LARY).iterrows():
            ax.annotate(row.utt, row.embedding)

    if include_PARE:
        x_PARE = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'PARE']
        embeddings_PARE = np.vstack(x_PARE.embedding.values)
        if x_PARE.size == 0:
            sys.exit("No x-vectors found for PARE!")
        ax.scatter(embeddings_PARE[:,0], embeddings_PARE[:,1], c='tab:olive', label='partres', s=pointsize)
        if annotate_outliers:
            for _, row in get_outliers(x_PARE).iterrows():
                ax.annotate(row.utt, row.embedding)

    x_CTRL = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'CTRL']
    embeddings_CTRL = np.vstack(x_CTRL.embedding.values)
    ax.scatter(embeddings_CTRL[:,0], embeddings_CTRL[:,1], c='tab:blue', label='ctrl', s=pointsize)
    if annotate_outliers:
        for _, row in get_outliers(x_CTRL).iterrows():
            ax.annotate(row.utt, row.embedding)

        
    if include_VOXCELEB:
        x_VOXCELEB = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'VOXCELEB']
        embeddings_VOXCELEB = np.vstack(x_VOXCELEB.embedding.values)
        if embeddings_VOXCELEB.shape[0] == 0:
            sys.exit("No x-vectors found for VOXCELEB!")
        ax.scatter(embeddings_VOXCELEB[:,0], embeddings_VOXCELEB[:,1], c='tab:orange', label='voxceleb', s=pointsize)
   
    plt.axis('off')
    ax.legend(fontsize=28)
    save_plot(output_dir, title)


def plot_scores(pathovoices_emb, score_col, title, output_dir, annotate=False):
    """ create scatter plot with colors visualizing scores """
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    pointsize = 160
    colormap = 'RdYlGn'
    colormap_score_factor = -1
    
    xvectors_LARY = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'LARY']
    embeddings_LARY = np.vstack(xvectors_LARY.embedding.values)
    ax.scatter(embeddings_LARY[:,0], embeddings_LARY[:,1], c=xvectors_LARY[score_col] * colormap_score_factor, marker="^", label='LARY', s=pointsize, cmap=colormap)
    if annotate:
        for row in xvectors_LARY.iterrows():
            row = row[1]  # remove index
            ax.annotate(round(row[score_col], 2), row.embedding)
    
    xvectors_CTRL = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'CTRL']
    embeddings_CTRL = np.vstack(xvectors_CTRL.embedding.values)
    ax.scatter(embeddings_CTRL[:,0], embeddings_CTRL[:,1], c=xvectors_CTRL[score_col] * colormap_score_factor, marker="o", label='CTRL', s=pointsize, cmap=colormap)
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
    return pd.read_table(filename, names=['utt', 'age'])


def get_mean_scores(f_crits, f_scores):
    criteria = pd.read_table(f_crits, header=None, encoding="iso-8859-1")[0].values
    mean_scores = pd.read_table(f_scores, ",", names=criteria)
    mean_scores.rename(columns={'Untersucher': 'rater', 'PatientNr': 'utt', 'file': 'utt', 'Anstr': 'effort', 'Verst': 'intell', 'Gesamt': 'overall'}, inplace=True)
    mean_scores = mean_scores.groupby(["utt"]).mean().reset_index()  # compute mean score across all raters
    return mean_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nnet_name', help='nnet name')
    parser.add_argument('output_dir', help='directory to save plot into')
    args = parser.parse_args()


    # get extracted pathovoices_emb
    pathovoices_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ScriptReader("./xvectors/{}/pathologic_voices/xvector_normalized.scp".format(args.nnet_name))
    for key, mat in reader:
        pathovoices_emb = pathovoices_emb.append({'utt': key.split('_', maxsplit=1)[1], 'speaker_group': key.split('_', maxsplit=1)[0], 'embedding': mat}, ignore_index=True)


    # plot colored speakergroups with and w/o partial resections
    pathovoices_emb.embedding = list(TSNE(n_components=2).fit_transform(np.vstack(pathovoices_emb.embedding.values)))
    plot_speakergroups(pathovoices_emb, None, "pathovoices_CTRL_LARY",                      args.output_dir, include_VOXCELEB=False, include_PARE=False)
    plot_speakergroups(pathovoices_emb, None, "pathovoices_CTRL_PARE_LARY",                 args.output_dir, include_VOXCELEB=False, include_PARE=True)


    # include random embeddings from voxceleb1-test
    voxceleb_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ScriptReader("./xvectors/{}/test/xvec_for_plot.scp".format(args.nnet_name))
    for key, mat in reader:
        voxceleb_emb = voxceleb_emb.append({'utt': key, 'speaker_group': 'VOXCELEB', 'embedding': mat}, ignore_index=True)
    voxceleb_emb = voxceleb_emb.sample(30, random_state=0)
    pathovoices_voxceleb = pd.concat([pathovoices_emb, voxceleb_emb], axis=0)

    pathovoices_voxceleb.embedding = list( TSNE(n_components=2).fit_transform( np.vstack(pathovoices_voxceleb.embedding.values) ) )
        
    plot_speakergroups(pathovoices_voxceleb, f"pathovoices_ctrl_laryng_voxceleb",         args.output_dir,   include_VOXCELEB=True,  include_PARE=False)
    plot_speakergroups(pathovoices_voxceleb, f"pathovoices_ctrl_partres_laryng_voxceleb", args.output_dir,   include_VOXCELEB=True,  include_PARE=True)


    # get effort scores
    mean_scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'effort', 'intell', 'overall']]
    mean_scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'effort', 'intell', 'overall']]
    mean_scores = pd.concat([mean_scores_LARY, mean_scores_CTRL])  # concat the dictionaries containing scores

    plot_scores_histograms(mean_scores_LARY, mean_scores_CTRL)

    for crit in ['effort', 'intell', 'overall']:
        plot_scores(pathovoices_emb.merge(mean_scores, on="utt"), crit, "pathologic_voices_CTRL_LARY - {}".format(crit), args.output_dir)

