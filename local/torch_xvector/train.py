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
import torch.cuda
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--egs-dir',
        dest="egs_dir",
        help="egs directory")

    # General Parameters
    parser.add_argument('-modelType', default='xvecTDNN', help='Model class. Check models.py')
    parser.add_argument('-featDim', default=30, type=int, help='Frame-level feature dimension')
    parser.add_argument('-trainingMode', default='init',
        help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
    parser.add_argument('-resumeModelDir', default=None, help='Path containing training checkpoints')

    # PyTorch distributed run
    parser.add_argument("--local_rank", type=int, default=0)

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

    # TRAINING
    while step < totalSteps:

        archiveI = step%args.numArchives + 1
        archive_start_time = time.time()
        ark_file = '{}/egs.{}.ark'.format(args.egs_dir,archiveI)
        print('Reading from archive %d' %archiveI)

        preFetchRatio = args.preFetchRatio
        # Read with data data_loader
        data_loader = train_utils.nnet3EgsDL(ark_file)
        par_data_loader = DataLoader(data_loader,
                                    batch_size=preFetchRatio*args.batchSize,
                                    shuffle=False,
                                    num_workers=0,
                                    drop_last=False,
                                    pin_memory=True)

        batchI, loggedBatch = 0, 0
        loggingLoss =  0.0
        start_time = time.time()
        for _,(X, Y) in par_data_loader:
            Y = Y['matrix'][0][0][0].to(device)
            X = X['matrix'].to(device)
            try:
                assert max(Y) < args.numSpkrs and min(Y) >= 0
            except:
                print('Read an out of range value at iter %d' %iter)
                continue
            if torch.isnan(X).any():
                print('Read a nan value at iter %d' %iter)
                continue

            accumulateStepSize = 4
            preFetchBatchI = 0  # this counter within the prefetched batches only
            while preFetchBatchI < int(len(Y)/args.batchSize) - accumulateStepSize:

                # Accumulated gradients used
                optimizer.zero_grad()
                for _ in range(accumulateStepSize):
                    batchI += 1
                    preFetchBatchI += 1
                    # fwd + bckwd + optim
                    output = net(X[preFetchBatchI*args.batchSize:(preFetchBatchI+1)*args.batchSize,:,:].permute(0,2,1), eps)
                    loss = criterion(output, Y[preFetchBatchI*args.batchSize:(preFetchBatchI+1)*args.batchSize].squeeze())
                    if np.isnan(loss.item()):
                        print('Nan encountered at iter %d. Exiting..' %iter)
                        sys.exit(1)
                    loss.backward()
                    loggingLoss += loss.item()

                optimizer.step()    # Does the update
                cyclic_lr_scheduler.step()

                # Log
                if batchI-loggedBatch >= args.logStepSize:
                    logStepTime = time.time() - start_time
                    print('Batch: (%d/%d)     Avg Time/batch: %1.3f      Avg Loss/batch: %1.3f' %(
                        batchI,
                        numBatchesPerArk,
                        logStepTime/(batchI-loggedBatch),
                        loggingLoss/(batchI-loggedBatch)))
                    loggingLoss = 0.0
                    start_time = time.time()
                    loggedBatch = batchI

        print('Archive processing time: %1.3f' %(time.time()-archive_start_time))
        # Update dropout
        if 1.0*step < args.stepFrac*totalSteps:
            p_drop = args.pDropMax*step/(args.stepFrac*totalSteps)
        else:
            p_drop = max(0,args.pDropMax*(2*step - totalSteps*(args.stepFrac+1))/(totalSteps*(args.stepFrac-1))) # fast decay
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop
        print('Dropout updated to %f' %p_drop)

        # Save checkpoint
        torch.save({
            'step': step,
            'archiveI':archiveI,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'args': args,
            }, '{}/checkpoint_step{}.tar'.format(saveDir, step))

        # Compute validation loss, update LR if using plateau rule
        valAcc = train_utils.computeValidAccuracy(args, saveDir)
        print('Validation accuracy is %1.2f precent' %(valAcc))

        # Cleanup. We always retain the last 10 models
        if step > 10:
            if os.path.exists('%s/checkpoint_step%d.tar' %(saveDir,step-10)):
                os.remove('%s/checkpoint_step%d.tar' %(saveDir,step-10))
        step += 1


if __name__ == "__main__":
    main()
