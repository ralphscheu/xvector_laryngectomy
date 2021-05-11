#!/usr/bin/env python3
import numpy as np
from kaldiio import ReadHelper
import argparse
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument('xvecDir', default=None)
args = parser.parse_args()
# print(args)

reader = ReadHelper('scp:{}/xvector_normalized.scp'.format(args.xvecDir))
keys, xvecs = [], []
for key, xvec in reader:
    keys.append(key)
    xvecs.append(xvec)

speaker_groups = np.array([ k[:4] for k in keys ])
keys = np.array(keys)
xvecs = np.array(xvecs)
ctrl_keys   = keys  [   np.where(speaker_groups == 'CTRL')  ]
ctrl_xvecs  = xvecs [   np.where(speaker_groups == 'CTRL')  ]
lary_keys   = keys  [   np.where(speaker_groups == 'LARY')  ]
lary_xvecs  = xvecs [   np.where(speaker_groups == 'LARY')  ]
# print(ctrl_keys.shape,ctrl_xvecs.shape,lary_keys.shape,lary_xvecs.shape)

neighbors_xvecs = list( range(lary_xvecs.shape[0]) )
neighbors_keys = list( range(lary_keys.shape[0]) )
for i in range(lary_xvecs.shape[0]):
    lary_id = lary_keys[i]
    lary_xvec = lary_xvecs[i]
    
    nn_distance = np.inf
    nn_j = -1
    nn_xvec = None
    for j in range(ctrl_xvecs.shape[0]):
        ctrl_id = ctrl_keys[j]
        ctrl_xvec = ctrl_xvecs[j]

        distance = np.dot( lary_xvec, ctrl_xvec )
        if distance < nn_distance:
            nn_distance = distance
            nn_j = j
            nn_xvec = ctrl_xvec

    assert (np.dot( lary_xvecs[i], ctrl_xvecs[nn_j] ) == nn_distance)
    
    nn_id = ctrl_keys[nn_j]
    print(f"{lary_id} {nn_id} {nn_distance}")

    neighbors_keys[i] = nn_id
    neighbors_xvecs[i] = nn_xvec

# print(len(neighbors))
# print(lary_xvecs[0].shape, neighbors_xvecs[0].shape)

plot=False
if plot:
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1) 

    ax.set_title('pathologic_voices_CTRL_LARY nearest neighbors', fontsize = 28)
    pointsize = 160

    pca = PCA(n_components=37).fit_transform
    tsne = TSNE(n_components=2).fit_transform

    xvecs_reduced = tsne(pca(xvecs))

    print("reducing lary_xvecs...")
    lary_xvecs = np.stack(lary_xvecs, 0)
    lary_xvecs = tsne(pca(lary_xvecs))
    print("reducing neighbors_xvecs...")
    neighbors_xvecs = np.stack(neighbors_xvecs, 0)
    neighbors_xvecs = tsne(pca(neighbors_xvecs))
    print("reducing remaining ctrl_vecs...")
    ctrl_xvecs = np.stack(ctrl_xvecs, 0)
    ctrl_xvecs = tsne(pca(ctrl_xvecs))

    print(lary_xvecs.shape, neighbors_xvecs.shape)

    ax.scatter(lary_xvecs[:,0], lary_xvecs[:,1],
        c='tab:red', label='LARY', s=pointsize)

    ax.scatter(ctrl_xvecs[:,0], ctrl_xvecs[:,1],
        c='tab:blue', label='CTRL', s=pointsize*2)

    ax.scatter(neighbors_xvecs[:,0], neighbors_xvecs[:,1],
        c='tab:purple', label='CTRL', s=pointsize)

    plt.axis('off')
    ax.legend(fontsize=32)
    plt.savefig("plots/lary_ctrl_nearestneighbors.png", bbox_inches='tight')
