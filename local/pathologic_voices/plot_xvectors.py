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


def save_plot(output_dir, dataset_name, dim_reduction_method):
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    save_filename ='{}/{}__{}__{}.png'.format(output_dir, dataset_name, dim_reduction_method, timestamp)
    plt.savefig(save_filename, bbox_inches='tight')
    print("Saved plot to {}".format( save_filename ))

def create_plot(x, speaker_groups, dataset_name, dim_reduction_method, output_dir, plot_PARE=False):
    """create scatter plot"""
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1) 
    # ax.set_title('{} X-Vectors ({})'.format(dataset_name, dim_reduction_method), fontsize = 16)
    pointsize = 160
    
    x_LARY = x[numpy.where(speaker_groups=='LARY')]
    ax.scatter(x_LARY[:,0], x_LARY[:,1],
        c='tab:red', label='LARY', s=pointsize)
    
    if plot_PARE:
        x_PARE = x[numpy.where(speaker_groups=='PARE')]
        if x_PARE.size == 0:
            sys.exit("No x-vectors found for PARE!")
        ax.scatter(x_PARE[:,0], x_PARE[:,1],
                c='tab:olive', label='PARE', s=pointsize)
    
    x_CTRL = x[numpy.where(speaker_groups=='CTRL')]
    ax.scatter(x_CTRL[:,0], x_CTRL[:,1],
        c='tab:blue', label='CTRL', s=pointsize)
   
    plt.axis('off')
    ax.legend(fontsize=32)
    save_plot(output_dir, dataset_name, dim_reduction_method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp file for xvectors to plot')
    parser.add_argument('output_dir', help='directory to save plot into')
    parser.add_argument('--dim-reduction-method', default='tsne', dest='dim_reduction_method', help='Method to use for dimensionality reduction (tsne, pca) (default=tsne)')
    args = parser.parse_args()

    reader = ScriptReader(args.scp_input)
    
    speaker_groups = []  # this will hold the group labels of each utterance
    x = []
    for key, mat in reader:
        speaker_groups.append(key.split('_')[0])
        x.append(mat)

    x = numpy.stack( x, axis=0 )
    speaker_groups = numpy.stack( speaker_groups, axis=0 )

    input_dim = x.shape[1]
    plot_dim = 2
    num_utterances = x.shape[0]
    num_speakers = num_utterances

    if args.dim_reduction_method == 'tsne':
        print("Applying T-SNE for plotting...")
        x_reduced = TSNE(n_components=plot_dim).fit_transform(x)
    elif args.dim_reduction_method == 'pca':
        print("Applying PCA for plotting...")
        x_reduced = PCA(n_components=plot_dim).fit_transform(x)
    else:
        sys.exit("Please select a valid method for dimensionality reduction (tsne, pca)")

    create_plot(x_reduced, speaker_groups, "pathologic_voices_CTRL_LARY", 
        args.dim_reduction_method, args.output_dir, plot_PARE=False)

    create_plot(x_reduced, speaker_groups, "pathologic_voices_CTRL_PARE_LARY", 
        args.dim_reduction_method, args.output_dir, plot_PARE=True)
    

if __name__ == "__main__":
    main()
