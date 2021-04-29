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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp file for xvectors to plot')
    parser.add_argument('dataset_title', help='Name of dataset to plot')
    parser.add_argument('output_dir', help='directory to save plot into')
    parser.add_argument('--dim-reduction-method', default='tsne', dest='dim_reduction_method', help='Method to use for dimensionality reduction (tsne, pca) (default=tsne)')
    args = parser.parse_args()

    reader = ScriptReader(args.scp_input)
    
    # speaker_ids = []
    speaker_groups = []
    x = []
    for key, mat in reader:
        # speaker_ids.append(int(key[0:12]))
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

    # create scatter plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title('{} X-Vectors ({})'.format(args.dataset_title, args.dim_reduction_method), 
        fontsize = 16)
    
    # map speaker_groups string labels to corresponding integers

    import matplotlib.cm as cm

    colors = cm.rainbow(numpy.linspace(0, 1, len(numpy.unique(speaker_groups))))
    
    x_LA = x_reduced[numpy.where(speaker_groups=='LA')]
    ax.scatter(x_LA[:,0], x_LA[:,1],
        c='red', label='LA', s=20)
    
    x_CTRL = x_reduced[numpy.where(speaker_groups=='CTRL')]
    ax.scatter(x_CTRL[:,0], x_CTRL[:,1],
        c='blue', label='CTRL', s=20)
   
    ax.legend()

    # save to png
    timestamp = datetime.now().strftime('%Y%m-%d%H-%M%S')
    save_filename ='{}/{}__{}__{}.png'.format(args.output_dir, args.dataset_title, args.dim_reduction_method, timestamp)
    plt.savefig(save_filename)
    print("Saved plot to {}".format( save_filename ))
    

if __name__ == "__main__":
    main()
