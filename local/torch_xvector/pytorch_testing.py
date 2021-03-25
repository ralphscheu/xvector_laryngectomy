#!/usr/bin/env python3
import os
import sys
import glob
import time
import socket
import argparse
import models
from kaldi_python_io import ScriptReader
import train_utils
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',
        dest="stage", type=int, default="0",
        help="stage to start from")
    parser.add_argument('--egs-dir',
        dest="egs_dir",
        help="egs directory")
    parser.add_argument('--nnet-dir',
        dest="nnet_dir",
        help="nnet directory")
    parser.add_argument('--data',
        dest="data",
        help="data directory")

    # General Parameters
    parser.add_argument('-modelType', default='xvecTDNN', help='Model class. Check models.py')
    parser.add_argument('-featDim', default=30, type=int, help='Frame-level feature dimension')
    parser.add_argument('-trainingMode', default='init',
        help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
    parser.add_argument('-resumeModelDir', default=None, help='Path containing training checkpoints')
    # parser.add_argument('featDir', default=None, help='Directory with training archives')

    # Training Parameters
    trainingArgs = parser.add_argument_group('General Training Parameters')
    trainingArgs.add_argument('-numArchives', default=84, type=int, help='Number of egs.*.ark files')
    trainingArgs.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels')
    trainingArgs.add_argument('-logStepSize', default=200, type=int, help='Iterations per log')
    trainingArgs.add_argument('-batchSize', default=32, type=int, help='Batch size')
    trainingArgs.add_argument('-numEgsPerArk', default=366150, type=int,
        help='Number of training examples per egs file')

    # Optimization Params
    optArgs = parser.add_argument_group('Optimization Parameters')
    optArgs.add_argument('-preFetchRatio', default=30, type=int, help='xbatchSize to fetch from dataloader')
    optArgs.add_argument('-optimMomentum', default=0.5, type=float, help='Optimizer momentum')
    optArgs.add_argument('-baseLR', default=1e-3, type=float, help='Initial LR')
    optArgs.add_argument('-maxLR', default=2e-3, type=float, help='Maximum LR')
    optArgs.add_argument('-numEpochs', default=2, type=int, help='Number of training epochs')
    optArgs.add_argument('-noiseEps', default=1e-5, type=float, help='Noise strength before pooling')
    optArgs.add_argument('-pDropMax', default=0.2, type=float, help='Maximum dropout probability')
    optArgs.add_argument('-stepFrac', default=0.5, type=float,
        help='Training iteration when dropout = pDropMax')

    args = parser.parse_args()
    print(args)

    #####
    # SEEDS
    torch.manual_seed(0)
    np.random.seed(0)
    #####

    #####
    # PREPARE MODEL
    totalSteps = args.numEpochs * args.numArchives
    net, optimizer, step, saveDir = train_utils.prepareModel(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)
    #####
    print(totalSteps)
    print(numBatchesPerArk)
    # LR SCHEDULERS
    cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                            max_lr=args.maxLR,
                            cycle_momentum=False,
                            div_factor=5,
                            final_div_factor=1e+3,
                            total_steps=totalSteps*numBatchesPerArk,
                            pct_start=0.15)
    criterion = nn.CrossEntropyLoss()
    eps = args.noiseEps


    # scp_reader = ScriptReader(f"{args.data}/feats.scp")
    # for key, mat in scp_reader:
    #     print(f"{key}: {mat.shape}")


    # TRAINING

    ark_file = f"{args.egs_dir}/egs.1.ark"
    data_loader = train_utils.nnet3EgsDL(ark_file)
    par_data_loader = DataLoader(data_loader,
                                 batch_size=64, #preFetchRatio*args.batchSize,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False,
                                 pin_memory=True)
    
    print(data_loader)
    print(par_data_loader)



if __name__ == "__main__":
    main()
