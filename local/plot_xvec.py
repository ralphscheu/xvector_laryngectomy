import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import argparse
import os
import sys
import subprocess
from kaldi_python_io import ScriptReader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_input', help='scp file for xvectors to plot')
    args = parser.parse_args()

    reader = ScriptReader(args.scp_input)
    
    targets = []
    x = []
    for key, mat in reader:
        targets.append(int(key[2:7]))
        x.append(mat)

    # only use ~50% of the vectors (also speakers)
    targets, x = targets[0:2400], x[0:2400]

    x = numpy.stack( x, axis=0 )
    targets = numpy.stack( targets, axis=0 )

    input_dim = x.shape[1]
    plot_dim = 2

    pca = PCA(n_components=plot_dim)
    principalComponents = pca.fit_transform(x)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title('xvectors voxceleb1_test ({}dim LDA -> {}-dim PCA)'.format(input_dim, plot_dim), 
        fontsize = 16)
    ax.scatter(principalComponents[:,0], principalComponents[:,1], c=targets, s=20)
    plt.savefig('plots/voxceleb1_test_xvectors_{}dimLDA_{}dimPCA.png'.format(
        input_dim, plot_dim)
        )
    

if __name__ == "__main__":
    main()
