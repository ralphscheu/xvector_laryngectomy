import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
from kaldiio import ReadHelper
from sklearn.manifold import TSNE
from viz_utils import save_plot


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

    x_LARY = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'LARY']
    embeddings_laryng = np.vstack(x_LARY.embedding.values)
    ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c='tab:red', label='TL', s=pointsize)
    if annotate_outliers:
        for _, row in get_outliers(x_LARY).iterrows():
            ax.annotate(row.utt, row.embedding)

    if include_PARE:
        x_PARE = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'PARE']
        embeddings_PARE = np.vstack(x_PARE.embedding.values)
        if x_PARE.size == 0:
            sys.exit("No x-vectors found for PARE!")
        ax.scatter(embeddings_PARE[:,0], embeddings_PARE[:,1], c='tab:olive', label='PL', s=pointsize)
        if annotate_outliers:
            for _, row in get_outliers(x_PARE).iterrows():
                ax.annotate(row.utt, row.embedding)

    x_CTRL = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'CTRL']
    embeddings_laryng = np.vstack(x_CTRL.embedding.values)
    ax.scatter(embeddings_laryng[:,0], embeddings_laryng[:,1], c='tab:blue', label='CTRL', s=pointsize)
    if annotate_outliers:
        for _, row in get_outliers(x_CTRL).iterrows():
            ax.annotate(row.utt, row.embedding)

        
    if include_VOXCELEB:
        x_VOXCELEB = pathovoices_emb.loc[pathovoices_emb.speaker_group == 'VOXCELEB']
        embeddings_VOXCELEB = np.vstack(x_VOXCELEB.embedding.values)
        if embeddings_VOXCELEB.shape[0] == 0:
            sys.exit("No x-vectors found for VOXCELEB!")
        ax.scatter(embeddings_VOXCELEB[:,0], embeddings_VOXCELEB[:,1], c='tab:orange', label='VOXCELEB', s=pointsize)
   
    plt.axis('off')
    ax.legend(fontsize=28)
    save_plot(output_dir, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input_pathovoices', help='scp string to xvectors of pathological speech')
    parser.add_argument('scp_input_voxceleb', help='scp string to xvectors of voxceleb1-test')
    parser.add_argument('--output-dir', default='./plots', help='directory to save plot into')
    args = parser.parse_args()

    # load xvectors of pathological speech
    pathovoices_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ReadHelper(args.scp_input_pathovoices)
    for key, mat in reader:
        pathovoices_emb = pathovoices_emb.append({'utt': key.split('_', maxsplit=1)[1], 'speaker_group': key.split('_', maxsplit=1)[0], 'embedding': mat}, ignore_index=True)
    pathovoices_emb_original = pathovoices_emb.copy()

    # load xvectors of voxceleb1-test
    voxceleb_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ReadHelper(args.scp_input_voxceleb)
    for key, mat in reader:
        voxceleb_emb = voxceleb_emb.append({'utt': key, 'speaker_group': 'VOXCELEB', 'embedding': mat}, ignore_index=True)
    voxceleb_emb = voxceleb_emb.sample(30, random_state=0)


    # plot colored speakergroups w/ and w/o partial resections
    pathovoices_emb.embedding = list(TSNE(n_components=2).fit_transform(np.vstack(pathovoices_emb.embedding.values)))
    plot_speakergroups(pathovoices_emb, "pathovoices__laryng_ctrl",         args.output_dir, include_VOXCELEB=False, include_PARE=False)
    plot_speakergroups(pathovoices_emb, "pathovoices__laryng_partres_ctrl", args.output_dir, include_VOXCELEB=False, include_PARE=True)


    # include random embeddings from voxceleb1-test
    voxceleb_emb = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    reader = ReadHelper(args.scp_input_voxceleb)
    for key, mat in reader:
        voxceleb_emb = voxceleb_emb.append({'utt': key, 'speaker_group': 'VOXCELEB', 'embedding': mat}, ignore_index=True)
    voxceleb_emb = voxceleb_emb.sample(30, random_state=0)
    pathovoices_voxceleb = pd.concat([pathovoices_emb_original, voxceleb_emb], axis=0)
    pathovoices_voxceleb.embedding = list( TSNE(n_components=2).fit_transform( np.vstack(pathovoices_voxceleb.embedding.values) ) )
    plot_speakergroups(pathovoices_voxceleb, "pathovoices__laryng_partres_ctrl_voxceleb", args.output_dir, include_VOXCELEB=True, include_PARE=True)
