import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys, os
from kaldi_python_io import ScriptReader
from sklearn.manifold import TSNE
from viz_utils import get_mean_scores, save_plot


def get_outliers(utterances, n=10):
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
    embeddings_laryng = np.vstack(x_LARY.embedding.values)
    ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c='tab:red', label='laryng', s=pointsize)
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
    embeddings_laryng = np.vstack(x_CTRL.embedding.values)
    ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c='tab:blue', label='ctrl', s=pointsize)
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
    colormap_score_factor = -1  # causes lower values to appear green and higher values (bad scores) red
    
    xvectors_laryng = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'LARY']
    if len(xvectors_laryng.index) > 0:
        embeddings_laryng = np.vstack(xvectors_laryng.embedding.values)
        ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c=xvectors_laryng[score_col] * colormap_score_factor, marker="^", label='laryng', s=pointsize, cmap=colormap)
        if annotate:
            for row in xvectors_laryng.iterrows():
                row = row[1]  # remove index
                ax.annotate(round(row[score_col], 2), row.embedding)

    xvectors_partres = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'PARE']
    if len(xvectors_partres.index) > 0:
        embeddings_partres = np.vstack(xvectors_partres.embedding.values)
        ax.scatter(embeddings_partres[:,0], embeddings_partres[:,1], c=xvectors_partres[score_col] * colormap_score_factor, marker="D", label='partres', s=pointsize, cmap=colormap)
        if annotate:
            for row in xvectors_partres.iterrows():
                row = row[1]  # remove index
                ax.annotate(round(row[score_col], 2), row.embedding)
    
    xvectors_ctrl = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'CTRL']
    if len(xvectors_ctrl.index) > 0:
        embeddings_laryng = np.vstack(xvectors_ctrl.embedding.values)
        ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c=xvectors_ctrl[score_col] * colormap_score_factor, marker="o", label='ctrl', s=pointsize, cmap=colormap)
        if annotate:
            for row in xvectors_ctrl.iterrows():
                row = row[1]  # remove index
                ax.annotate(round(row[score_col], 2), row.embedding)
   
    plt.axis('off')
    ax.legend(fontsize=32)
    save_plot(output_dir, title)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nnet_name', help='nnet name')
    parser.add_argument('--output_dir', default='./plots', help='directory to save plot into')
    args = parser.parse_args()


    # get extracted embeddings
    pathovoices_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ScriptReader("./xvectors/{}/pathologic_voices/xvector_normalized.scp".format(args.nnet_name))
    for key, mat in reader:
        pathovoices_emb = pathovoices_emb.append({'utt': key.split('_', maxsplit=1)[1], 'speaker_group': key.split('_', maxsplit=1)[0], 'embedding': mat}, ignore_index=True)
    pathovoices_emb_original = pathovoices_emb.copy()

    # get expert scores
    mean_scores_laryng = get_mean_scores(
        "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", 
        "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'effort', 'intell', 'overall']]
    mean_scores_partres = get_mean_scores(
        "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", 
        "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")[['utt', 'effort', 'intell', 'overall']]
    mean_scores_ctrl = get_mean_scores(
        "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits",
        "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'effort', 'intell', 'overall']]
    mean_scores = pd.concat([mean_scores_laryng, mean_scores_partres, mean_scores_ctrl])  # concat the dictionaries containing scores


    # plot colored speakergroups w/ and w/o partial resections
    pathovoices_emb.embedding = list(TSNE(n_components=2).fit_transform(np.vstack(pathovoices_emb.embedding.values)))
    plot_speakergroups(pathovoices_emb, "pathovoices__laryng_ctrl",         args.output_dir, include_VOXCELEB=False, include_PARE=False)
    plot_speakergroups(pathovoices_emb, "pathovoices__laryng_partres_ctrl", args.output_dir, include_VOXCELEB=False, include_PARE=True)


    # include random embeddings from voxceleb1-test
    voxceleb_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ScriptReader("./xvectors/{}/test/xvec_for_plot.scp".format(args.nnet_name))
    for key, mat in reader:
        voxceleb_emb = voxceleb_emb.append({'utt': key, 'speaker_group': 'VOXCELEB', 'embedding': mat}, ignore_index=True)
    voxceleb_emb = voxceleb_emb.sample(30, random_state=0)
    pathovoices_voxceleb = pd.concat([pathovoices_emb_original, voxceleb_emb], axis=0)
    pathovoices_voxceleb.embedding = list( TSNE(n_components=2).fit_transform( np.vstack(pathovoices_voxceleb.embedding.values) ) )
    plot_speakergroups(pathovoices_voxceleb, f"pathovoices__laryng_partres_ctrl_voxceleb", args.output_dir, include_VOXCELEB=True, include_PARE=True)


    # color embeddings based on scores
    for crit in ['effort', 'intell', 'overall']:
        
        # laryng + ctrl
        plot_scores(pathovoices_emb.loc[pathovoices_emb.speaker_group != 'PARE'].merge(mean_scores, on="utt"), crit, "pathovoices__laryng_ctrl__{}".format(crit), args.output_dir)

        # laryng + partres
        plot_scores(pathovoices_emb.loc[pathovoices_emb.speaker_group != 'CTRL'].merge(mean_scores, on="utt"), crit, "pathovoices__laryng_partres__{}".format(crit), args.output_dir)

        # laryng + partres + ctrl
        plot_scores(pathovoices_emb.merge(mean_scores, on="utt"), crit, "pathovoices__laryng_partres_ctrl__{}".format(crit), args.output_dir)
