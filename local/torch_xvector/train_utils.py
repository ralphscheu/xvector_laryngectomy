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
import configparser
import argparse
from datetime import datetime
import numpy as np
from models import *
import kaldi_python_io
from kaldiio import ReadHelper
from torch.utils.data import Dataset, IterableDataset
from collections import OrderedDict
from models import xvector, xvector_mha


def readHdf5File_full(fileName):
    """ Read at-once from the hdf5 file. Rarely used
        Outputs:
        feats: (N,1,chunkLen,30)
        labels: (N,1)
    """
    with h5py.File(fileName,'r') as x:
        feats, labels = np.array(x.get('feats')), np.array(x.get('labels'))
    chunkLen = feats.shape[1]
    feats = torch.from_numpy(feats).unsqueeze(1) # make in (N,1,chunkLen,30)
    labels = torch.from_numpy(labels)
    return feats, labels

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


class myH5DL(Dataset):
    """ Data loader class customized to reading from hdf5 files
    """

    def __init__(self, hdf5File):
        x = h5py.File(hdf5File,'r')
        self.feats = x.get('feats')
        self.labels = x.get('labels')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return samples from idx:idx+batch_size
        """
        X = self.feats[idx,:,:]
        Y = self.labels[idx]
        return X, Y

class myH5DL_sampler(Dataset):
    """ Data loader class customized to reading from hdf5 files
        Based on https://github.com/cyvius96/prototypical-network-pytorch/blob/master/samplers.py
    """

    def __init__(self, hdf5File, minClasses, maxClasses, samplesPerClass, numEpisodes=100):
        self.samplesPerClass = samplesPerClass
        self.minClasses = minClasses
        self.maxClasses = maxClasses
        self.numEpisodes = numEpisodes
        x = h5py.File(hdf5File,'r')
        self.feats = x.get('feats')
        self.labels = x.get('labels')
        npLabels = self.labels[()].reshape(-1)
        self.uniqLabels = np.ndarray.tolist(np.unique(npLabels))
        try:
            assert self.maxClasses <= len(self.uniqLabels)
        except:
            print('Requesting more classes (%d) than available (%d)' %(self.maxClasses, len(self.uniqLabels)))
            sys.exit(1)

        self.labelIndices = {}
        for lab in self.uniqLabels:
            ind = np.argwhere(npLabels==lab).reshape(-1)
            # self.labelIndices[lab] = torch.from_numpy(ind)
            self.labelIndices[lab] = np.ndarray.tolist(ind)
        self.minSamplesPerClass = min([len(v) for v in self.labelIndices.values()])
        try:
            assert self.samplesPerClass <= self.minSamplesPerClass
        except:
            print('Requesting more samples (%d) than available (%d)' %(self.samplesPerClass, self.minSamplesPerClass))
            sys.exit(1)
        self.nClasses = random.randint(self.minClasses, self.maxClasses+1)


    def __iter__(self):
        for _ in range(self.numEpisodes):
            classes = random.sample(self.uniqLabels, self.nClasses)
            batchInd = np.empty((self.samplesPerClass, self.nClasses))
            for i,c in enumerate(classes):
                selectSampleInd = np.random.choice(self.labelIndices[c], self.samplesPerClass)
                batchInd[:,i] = selectSampleInd
            yield batchInd.ravel()


def prepareModel(args):

    device = torch.device("cuda:" + str(args.local_rank) if torch.cuda.is_available() else "cpu")
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.local_rank)
    torch.backends.cudnn.benchmark = True

    if args.trainingMode == 'resume':
        # select the latest model from modelDir
        modelFile = max(glob.glob(args.resumeModelDir+'/*'), key=os.path.getctime)
        net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.baseLR)
        net.to(device)

        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
            else:
                new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)

        step = checkpoint['step']
        totalSteps = args.numEpochs * args.numArchives
        print('Resuming training from step %d' %step)

        # set the dropout
        if 1.0*step < args.stepFrac*totalSteps:
            p_drop = args.pDropMax*step*args.stepFrac/totalSteps
        else:
            p_drop = max(0,args.pDropMax*(totalSteps + args.stepFrac - 2*step)/(totalSteps - totalSteps*args.stepFrac))
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop
        saveDir = args.resumeModelDir

    elif args.trainingMode == 'sanity_check':

        # select the latest model from modelDir
        modelFile = max(glob.glob(args.resumeModelDir+'/*'), key=os.path.getctime)
        net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.baseLR)
        net.to(device)

        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        checkpoint = torch.load(modelFile,map_location=torch.device('cuda'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
            else:
                new_state_dict[k] = v

        net.tdnn1.weight = torch.nn.Parameter(new_state_dict['tdnn1.weight'])
        net.tdnn1.bias = torch.nn.Parameter(new_state_dict['tdnn1.bias'])
        net.tdnn2.weight = torch.nn.Parameter(new_state_dict['tdnn2.weight'])
        net.tdnn2.bias = torch.nn.Parameter(new_state_dict['tdnn2.bias'])
        net.tdnn3.weight = torch.nn.Parameter(new_state_dict['tdnn3.weight'])
        net.tdnn3.bias = torch.nn.Parameter(new_state_dict['tdnn3.bias'])
        net.tdnn4.weight = torch.nn.Parameter(new_state_dict['tdnn4.weight'])
        net.tdnn4.bias = torch.nn.Parameter(new_state_dict['tdnn4.bias'])
        net.tdnn5.weight = torch.nn.Parameter(new_state_dict['tdnn5.weight'])
        net.tdnn5.bias = torch.nn.Parameter(new_state_dict['tdnn5.bias'])

        step = checkpoint['step']
        totalSteps = args.numEpochs * args.numArchives
        print('Resuming training from step %d' %step)

        # set the dropout
        if 1.0*step < args.stepFrac*totalSteps:
            p_drop = args.pDropMax*step*args.stepFrac/totalSteps
        else:
            p_drop = max(0,args.pDropMax*(totalSteps + args.stepFrac - 2*step)/(totalSteps - totalSteps*args.stepFrac))
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop
        saveDir = args.resumeModelDir
        step += 1

    elif args.trainingMode == 'init':
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
    parser.add_argument('--trainingMode', default='init',
        help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
    parser.add_argument('--resumeModelDir', default=None, help='Path containing training checkpoints')
    parser.add_argument('featDir', default=None, help='Directory with training archives')

    # Training Parameters - no more trainFullXvector = 0
    trainingArgs = parser.add_argument_group('General Training Parameters')
    trainingArgs.add_argument('--numArchives', default=84, type=int, help='Number of egs.*.ark files')
    trainingArgs.add_argument('--numSpkrs', default=7323, type=int, help='Number of output labels')
    trainingArgs.add_argument('--logStepSize', default=200, type=int, help='Iterations per log')
    trainingArgs.add_argument('--batchSize', default=32, type=int, help='Batch size')
    trainingArgs.add_argument('--numEgsPerArk', default=366150, type=int,
        help='Number of training examples per egs file')
    trainingArgs.add_argument('--numAttnHeads', default=12, type=int)

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

    # Metalearning params
    protoArgs = parser.add_argument_group('Protonet Parameters')
    protoArgs.add_argument('--preTrainedModelDir', default=None, help='Embedding model to initialize training')
    protoArgs.add_argument('--protoMinClasses', default=5, type=int, help='Minimum N-way')
    protoArgs.add_argument('--protoMaxClasses', default=35, type=int, help='Maximum N-way')
    protoArgs.add_argument('--protoEpisodesPerArk', default=25, type=int, help='Episodes per ark file')
    protoArgs.add_argument('--totalEpisodes', default=100, type=int, help='Number of training episodes')
    protoArgs.add_argument('--supportFrac', default=0.7, type=float, help='Fraction of samples as supports')

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
    elif args.modelType == 'xvector-mha':
        net = xvector_mha(numSpkrs=args.numSpkrs, num_attn_heads=args.numAttnHeads, rank=args.local_rank).to(args.local_rank)
    
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

