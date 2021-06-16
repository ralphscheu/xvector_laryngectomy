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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp file for xvectors to plot')
    parser.add_argument('dataset_title', help='Name of dataset to plot')
    parser.add_argument('output_dir', help='directory to save plot into')
    args = parser.parse_args()

    reader = ScriptReader(args.scp_input)
    targets = []
    x = []
    for key, mat in reader:
        targets.append(int(key[2:7]))
        x.append(mat)

    x = numpy.stack( x, axis=0 )
    targets = numpy.stack( targets, axis=0 )

    # apply t-SNE
    x_reduced = TSNE(n_components=2).fit_transform(x)

    # create scatter plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title('{}'.format(args.dataset_title,), fontsize = 16)
    ax.scatter(x_reduced[:,0], x_reduced[:,1], c=targets, s=20)
    plt.savefig('{}/{}__{}.png'.format(args.output_dir, args.dataset_title, datetime.now().strftime('%Y%m%dT%H%M%S')))
