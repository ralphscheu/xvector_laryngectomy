#!/usr/bin/env python3
"""
MIT License
Copyright (c) 2020 Manoj Kumar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
    Date Created: Feb 26 2018
    This script extracts trained embeddings given the model directory, and saves them in kaldi format
"""
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from models import xvector
import kaldi_python_io
import socket
from train_utils import *
from collections import OrderedDict
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from kaldiio import ReadHelper


def getSplitNum(text):
    return int(text.split('/')[-1].lstrip('split'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Extract params
    parser.add_argument('--numSplits', default=None, help='number of feature splits')
    parser.add_argument('--modelType', default='xvector', help='Model type to use')
    parser.add_argument('--numSpkrs', default=7323, type=int, help='Number of output labels for model')
    parser.add_argument('--layerName', default='fc1', help="DNN layer for embeddings")
    parser.add_argument('modelDirectory', help='Directory containing the model checkpoints')
    parser.add_argument('featDir', help='Directory containing features ready for extraction')
    parser.add_argument('embeddingDir', help='Output directory')
    args = parser.parse_args()
    print(args)

    # Checking for input features
    if not os.path.isfile('%s/feats.scp' %(args.featDir,)):
        print('Cannot find input features')
        sys.exit(1)

    # Check for trained model
    try:
        modelFile = max(glob.glob(args.modelDirectory+'/*.tar'), key=os.path.getctime)
    except ValueError:
        print("[ERROR] No trained model has been found in {}.".format(args.modelDirectory) )
        sys.exit(1)

    # Load model definition
    if args.modelType == 'xvector' or args.modelType == 'xvector-ams':
        net = xvector(args.numSpkrs, p_dropout=0)

    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v

    # Load trained weights
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()


    if not os.path.isdir(args.embeddingDir):
        os.makedirs(args.embeddingDir)
    # Extract x-vectors
    if args.numSplits == None:
        print('Writing xvectors to %s/xvector.ark' %(args.embeddingDir))
        par_core_extractXvectors(
            inFeatsScp='%s/feats.scp' %(args.featDir),
            outXvecArk='%s/xvector.ark' %(args.embeddingDir),
            outXvecScp='%s/xvector.scp' %(args.embeddingDir),
            net=net,
            layerName=args.layerName
        )
    else:
        if not os.path.isdir('%s/split400' %(args.embeddingDir)):
            os.makedirs('%s/split400' %(args.embeddingDir))
        for split_i in range(1, 401):
            print('Writing xvectors to %s/split400/xvector_split400_%s.ark' %(args.embeddingDir, split_i))
            par_core_extractXvectors(
                inFeatsScp='%s/split400/%s/feats.scp' %(args.featDir, split_i),
                outXvecArk='%s/split400/xvector_split400_%s.ark' %(args.embeddingDir, split_i),
                outXvecScp='%s/split400/xvector_split400_%s.scp' %(args.embeddingDir, split_i),
                net=net,
                layerName=args.layerName
            )
    
    print('Finished.')
