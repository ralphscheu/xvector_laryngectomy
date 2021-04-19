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
from models import *
import kaldi_python_io
import socket
from train_utils import *
from collections import OrderedDict
from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)

def getSplitNum(text):
    return int(text.split('/')[-1].lstrip('split'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelType', default='xvecTDNN', help='Refer train_utils.py ')
    parser.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels for model')
    parser.add_argument('-layerName', default='fc1', help="DNN layer for embeddings")
    parser.add_argument('modelDirectory', help='Directory containing the model checkpoints')
    parser.add_argument('featDir', help='Directory containing features ready for extraction')
    parser.add_argument('embeddingDir', help='Output directory')
    args = parser.parse_args()

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
    net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))

    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    if 'relation' in args.modelType:
        checkpoint_dict = checkpoint['encoder_state_dict']
    else:
        checkpoint_dict = checkpoint['model_state_dict']
    for k, v in checkpoint_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v

    # load trained weights
    net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()

    if not os.path.isdir(args.embeddingDir):
        os.makedirs(args.embeddingDir)

    print('Extracting xvectors... ')
    par_core_extractXvectors(
        inFeatsScp='%s/feats.scp' %(args.featDir),
        outXvecArk='%s/xvector.ark' %(args.embeddingDir),
        outXvecScp='%s/xvector.scp' %(args.embeddingDir),
        net=net,
        layerName=args.layerName
        )

    print('Writing xvectors to {}'.format(args.embeddingDir))
    print('Finished.')


if __name__ == "__main__":
    main()