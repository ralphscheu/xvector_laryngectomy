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
    
    targets = []
    x = []
    for key, mat in reader:
        targets.append(int(key[2:7]))
        x.append(mat)

    x = numpy.stack( x, axis=0 )
    targets = numpy.stack( targets, axis=0 )

    input_dim = x.shape[1]
    plot_dim = 2

    if args.dim_reduction_method == 'tsne':
        x_reduced = TSNE(n_components=plot_dim).fit_transform(x)
    elif args.dim_reduction_method == 'pca':
        pca = PCA(n_components=plot_dim)
        x_reduced = pca.fit_transform(x)
    else:
        sys.exit("Please select a valid method for dimensionality reduction (tsne, pca)")

    # create scatter plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title('{} X-Vectors ({})'.format(args.dataset_title, args.dim_reduction_method), 
        fontsize = 16)
    ax.scatter(x_reduced[:,0], x_reduced[:,1], c=targets, s=20)
    
    timestamp = datetime.now().strftime('%Y%m-%d%H-%M%S')
    
    plt.savefig('{}/{}__{}__{}.png'.format(
        args.output_dir, args.dataset_title, args.dim_reduction_method, timestamp)
        )
    

if __name__ == "__main__":
    main()
