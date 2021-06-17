#!/bin/python3.6

"""
    Date Created: Feb 11 2020
    This file will contain the training utils

"""

import os
import sys
import glob
import h5py
import torch
import random
import argparse
from datetime import datetime
import numpy as np
from models import *
import kaldi_python_io
from kaldiio import ReadHelper
from torch.utils.data import Dataset, IterableDataset
from collections import OrderedDict
from models import xvector
import torch.nn as nn


class nnet3EgsDL(IterableDataset):
    """ Data loader class to read directly from egs files, no HDF5
    """

    def __init__(self, arkFile):
        self.fid = kaldi_python_io.Nnet3EgsReader(arkFile)

    def __iter__(self):
        return iter(self.fid)


class nnet3EgsDLNonIterable(Dataset):
    """ Data loader class to read directly from egs files, no HDF5
    Loads the entire file at once to enable using DistributedSampler (needs len())
    """

    def __init__(self, arkFile):
        self.reader = kaldi_python_io.Nnet3EgsReader(arkFile)
        self.items = []
        for item in self.reader:
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def prepareModel(args):
    device = torch.device("cuda:" + str(args.local_rank) if torch.cuda.is_available() else "cpu")
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.local_rank)
    torch.backends.cudnn.benchmark = True

    if args.trainingMode == 'init':
        if args.is_master:
            print(f'Initializing models...')
        step = 0
        net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
        optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)


        if torch.cuda.device_count() > 1 and args.is_master:
                print("Using ", torch.cuda.device_count(), "GPUs!")

        net.to(device)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        
        eventID = datetime.now().strftime(r'%Y%m%d-%H%M%S')
        saveDir = './models/modelType_{}_rank_{}_event_{}' .format(args.modelType, args.local_rank, eventID)
        os.makedirs(saveDir)

    return net, optimizer, step, saveDir


def getParams():
    parser = argparse.ArgumentParser()

    # PyTorch distributed run
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    # General Parameters
    parser.add_argument('--modelType', default='xvector', help='Model class. Check models.py')
    parser.add_argument('--featDim', default=30, type=int, help='Frame-level feature dimension')
    parser.add_argument('--trainingMode', default='init', help='(init) Train from scratch')
    parser.add_argument('--resumeModelDir', default=None, help='Path containing training checkpoints')
    parser.add_argument('featDir', default=None, help='Directory with training archives')

    # Training Parameters - no more trainFullXvector = 0
    trainingArgs = parser.add_argument_group('General Training Parameters')
    trainingArgs.add_argument('--numArchives', default=84, type=int, help='Number of egs.*.ark files')
    trainingArgs.add_argument('--numSpkrs', default=7323, type=int, help='Number of output labels')
    trainingArgs.add_argument('--logStepSize', default=200, type=int, help='Iterations per log')
    trainingArgs.add_argument('--batchSize', default=32, type=int, help='Batch size')
    trainingArgs.add_argument('--numEgsPerArk', default=366150, type=int, help='Number of training examples per egs file')

    # Optimization Params
    optArgs = parser.add_argument_group('Optimization Parameters')
    optArgs.add_argument('--preFetchRatio', default=30, type=int, help='xbatchSize to fetch from dataloader')
    optArgs.add_argument('--optimMomentum', default=0.5, type=float, help='Optimizer momentum')
    optArgs.add_argument('--baseLR', default=1e-3, type=float, help='Initial LR')
    optArgs.add_argument('--maxLR', default=4e-3, type=float, help='Maximum LR')
    optArgs.add_argument('--numEpochs', default=2, type=int, help='Number of training epochs')
    optArgs.add_argument('--noiseEps', default=1e-5, type=float, help='Noise strength before pooling')
    optArgs.add_argument('--pDropMax', default=0.2, type=float, help='Maximum dropout probability')
    optArgs.add_argument('--stepFrac', default=0.5, type=float,
        help='Training iteration when dropout = pDropMax')

    return parser


def checkParams(args):
    if args.featDir is None:
        print('Features directory cannot be empty!')
        sys.exit()

    if args.protoMinClasses > args.protoMaxClasses:
        print('Max Classes must be greater than or equal to min classes')
        sys.exit(1)

    if args.trainingMode not in [ 'init', 'resume', 'sanity_check', 'initMeta', 'resumeMeta' ]:
        print('Invalid training mode')
        sys.exit(1)

    if 'Meta' in args.trainingMode and args.preTrainedModelDir is None:
        print('Missing pretrained model directory')
        sys.exit(1)

    if 'resume' in args.trainingMode and args.resumeModelDir is None:
        print('Provide model directory to resume training from')
        sys.exit(1)


def computeValidAccuracy(args, modelDir):
    """ Computes frame-level validation accuracy """
    modelFile = max(glob.glob(modelDir+'/*'), key=os.path.getctime)
    # Load the model

    if args.modelType == 'xvector':
        net = xvector(numSpkrs=args.numSpkrs, rank=args.local_rank).to(args.local_rank)
    elif args.modelType == 'xvector-ams':
        net = xvector(numSpkrs=args.numSpkrs, rank=args.local_rank).to(args.local_rank)
    
    checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            new_state_dict[k] = v
            
    # load params
    net.load_state_dict(new_state_dict)
    net.eval()

    correct, incorrect = 0, 0
    for validArk in glob.glob(args.featDir+'/valid_egs.*.ark'):
        x = kaldi_python_io.Nnet3EgsReader(validArk)
        for key, mat in x:
            out = net(x=torch.Tensor(mat[0]['matrix']).permute(1,0).unsqueeze(0).cuda(),eps=0)
            if mat[1]['matrix'][0][0][0]+1 == torch.argmax(out)+1:
                correct += 1
            else:
                incorrect += 1
    return 100.0*correct/(correct+incorrect)


def par_core_extractXvectors(inFeatsScp, outXvecArk, outXvecScp, net, layerName):
    """ To be called using pytorch multiprocessing
        Note: This function reads all the data from feats.scp into memory
        before inference. Hence, make sure the file is not too big (Hint: use
        split_data_dir.sh)
    """

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    eval('net.%s.register_forward_hook(get_activation(layerName))' %layerName)

    with kaldi_python_io.ArchiveWriter(outXvecArk, outXvecScp) as writer:
        with ReadHelper('scp:%s'%inFeatsScp) as reader:
            for key, mat in reader:
                try:
                    out = net(x=torch.Tensor(mat).permute(1,0).unsqueeze(0).cuda(), eps=0)
                    writer.write(key, np.squeeze(activation[layerName].cpu().numpy()))
                except:
                    print(key, mat.shape[0], "excluded: too big for GPU memory")
                    continue


class AMSoftmax(nn.Module):
    """ https://github.com/CoinCheung/pytorch-loss/blob/e8e997c85fe69d6de4028d96702b9b3253e68546/pytorch_loss/amsoftmax.py """

    def __init__(self, in_feats, n_classes, rank, m=0.3, s=15):
        super().__init__()
        self.rank = rank
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss

