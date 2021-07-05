import matplotlib.pyplot as plt
import numpy
import argparse
from kaldiio import ReadHelper
from sklearn.manifold import TSNE
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp string to xvectors')
    parser.add_argument('dataset_title', help='Name of dataset to plot')
    parser.add_argument('--output-dir', default='./plots', help='directory to save plot into')
    args = parser.parse_args()

    reader = ReadHelper(args.scp_input)
    targets = []
    x = []
    for key, mat in reader:
        targets.append(int(key[2:7]))
        x.append(mat)

    x = numpy.stack( x, axis=0 )
    targets = numpy.stack( targets, axis=0 )

    print("creating X-Vector plots...")

    # apply t-SNE
    x_reduced = TSNE(n_components=2).fit_transform(x)

    # create scatter plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.scatter(x_reduced[:,0], x_reduced[:,1], c=targets, s=20)
    filename = '{}/{}__{}'.format(args.output_dir, args.dataset_title, datetime.now().strftime('%Y%m%dT%H%M%S'))
    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.savefig(f"{filename}.eps", format="eps", bbox_inches='tight')

    print("Done. Saved plot to " + '{}/{}__{}.png'.format(args.output_dir, args.dataset_title, datetime.now().strftime('%Y%m%dT%H%M%S')))
