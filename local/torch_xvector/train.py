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
from torch.utils.data.distributed import DistributedSampler
import math
from models import *
from torch.distributed import init_process_group
from datetime import datetime


def train(args):
    init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=args.world_size, 
        rank=args.local_rank
    )

    device = torch.device("cuda:" + str(args.local_rank))

    # args.modelType = (xvecTDNN | xvecTDNN_MHAttn)
    net = eval('{}({}, p_dropout=0)'.format(args.modelType, args.numSpkrs))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)

    net.to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    if torch.cuda.device_count() > 1 and args.is_master:
            print("Using ", torch.cuda.device_count(), "GPUs!")

    # determine model dir
    eventID = datetime.now().strftime('%Y%m%dT%H%M%S')
    saveDir = './models/modelType_{}_event_{}_rank_{}' .format(args.modelType, eventID, args.local_rank)
    os.makedirs(saveDir)

    with torch.cuda.device(device):

        numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)

        # TRAINING LOOP
        step = 0
        totalSteps = args.numEpochs * args.numArchives

        cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                max_lr=args.maxLR,
                                cycle_momentum=False,
                                div_factor=5,
                                final_div_factor=1e+3,
                                total_steps=totalSteps*numBatchesPerArk,
                                pct_start=0.15)

        while step < totalSteps:
            

            archiveI = step % args.numArchives + 1
            archive_start_time = time.time()

            # Read training data
            ark_file = '{}/egs.{}.ark'.format(args.featDir,archiveI)
            if args.is_master:
                print(f"Reading {ark_file}")

            train_dataset = train_utils.nnet3EgsDLNonIterable(ark_file)
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank
            )

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.preFetchRatio*args.batchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler
            )

            # Loop through batches
            batchI, loggedBatch = 0, 0
            loggingLoss =  0.0
            start_time = time.time()
            for _,(X, Y) in train_loader:
                Y = Y['matrix'][0][0][0].to(device)
                X = X['matrix'].to(device)
                # try:
                #     assert max(Y) < args.numSpkrs and min(Y) >= 0
                # except:
                #     print('Read an out of range value at iter %d' %iter)
                #     continue
                # if torch.isnan(X).any():
                #     print('Read a nan value at iter %d' %iter)
                #     continue

                accumulateStepSize = 4
                preFetchBatchI = 0  # this counter within the prefetched batches only
                while preFetchBatchI < int(len(Y)/args.batchSize) - accumulateStepSize:

                    # Accumulated gradients used
                    optimizer.zero_grad()
                    for _ in range(accumulateStepSize):
                        batchI += 1
                        preFetchBatchI += 1
                        
                        # fwd + bckwd + optim
                        output = net(X[preFetchBatchI*args.batchSize:(preFetchBatchI+1)*args.batchSize,:,:].permute(0,2,1), args.noiseEps)
                        loss = criterion(output, Y[preFetchBatchI*args.batchSize:(preFetchBatchI+1)*args.batchSize].squeeze())
                        if np.isnan(loss.item()):
                            print('Nan encountered at iter %d. Exiting..' %iter)
                            sys.exit(1)
                        loss.backward()
                        loggingLoss += loss.item()

                    optimizer.step()    # Does the update
                    cyclic_lr_scheduler.step()    # Update Learning Rate

                    # Log batches
                    if batchI-loggedBatch >= args.logStepSize:
                        logStepTime = time.time() - start_time
                        
                        if args.is_master:
                            print('Epoch: (%d/%d)    Batch: (%d/%d)    Avg Time/batch: %1.3f    Avg Loss/batch: %1.3f' %
                                (
                                    int( max(1, math.ceil(step / args.numArchives) ) ),
                                    int( math.ceil(totalSteps / args.numArchives) ),
                                    batchI,
                                    numBatchesPerArk,
                                    logStepTime / (batchI-loggedBatch),
                                    loggingLoss / (batchI-loggedBatch)
                                )
                            )
                        loggingLoss = 0.0
                        start_time = time.time()
                        loggedBatch = batchI

            # Finished archive file
            
            if args.is_master:
                print('Archive processing time: %1.3f' %(time.time()-archive_start_time))
            
            # Update dropout
            if 1.0*step < args.stepFrac*totalSteps:
                p_drop = args.pDropMax*step/(args.stepFrac*totalSteps)
            else:
                p_drop = max(0,args.pDropMax*(2*step - totalSteps*(args.stepFrac+1))/(totalSteps*(args.stepFrac-1))) # fast decay
            for x in net.modules():
                if isinstance(x, torch.nn.Dropout):
                    x.p = p_drop

            if args.is_master:
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
            if args.is_master:
                valAcc = train_utils.computeValidAccuracy(args, saveDir)
                print('Validation accuracy is %1.2f precent' %(valAcc))

            # Cleanup. We always retain the last 10 models
            if step > 10:
                if os.path.exists('%s/checkpoint_step%d.tar' %(saveDir,step-10)):
                    os.remove('%s/checkpoint_step%d.tar' %(saveDir,step-10))
            step += 1


def main():
    parser = train_utils.getParams()
    args = parser.parse_args()
    args.is_master = args.local_rank == 0
    args.world_size = torch.cuda.device_count() * args.num_nodes
    if args.is_master:
        print(args)

    # SEEDS
    torch.manual_seed(0)
    np.random.seed(0)

    train(args)


if __name__ == "__main__":
    main()
