import os, sys, logging, math, time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import datetime
from models import xvector, xvector_mha
from train_utils import nnet3EgsDLNonIterable, getParams, computeValidAccuracy
import numpy as np


def setup(rank, args):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=args.world_size)
    torch.backends.cudnn.benchmark = True

    print("Initializing model...")
    if args.modelType == 'xvector':
        model = xvector(numSpkrs=args.numSpkrs, rank=rank).to(rank)
    elif args.modelType == 'xvector-mha':
        model = xvector_mha(numSpkrs=args.numSpkrs, num_attn_heads=args.numAttnHeads, rank=rank).to(rank)

    print(model)

    model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.baseLR)
    numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)
    total_steps = args.numEpochs * args.numArchives
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                            max_lr=args.maxLR,
                            cycle_momentum=False,
                            div_factor=5,
                            final_div_factor=1e+3,
                            total_steps=total_steps*numBatchesPerArk,
                            pct_start=0.15)
    return model, loss_fn, optimizer, lr_scheduler, numBatchesPerArk

def save_checkpoint(step, archive_id, model, optimizer, loss, checkpoint_dir, args):
    torch.save({
        'step': step,
        'archiveI': archive_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': args,
        }, '{}/checkpoint_step{}.tar'.format(checkpoint_dir, step))

    # Cleanup. We always retain the last 10 models
    if step > 10:
        if os.path.exists('%s/checkpoint_step%d.tar' %(checkpoint_dir, step-10)):
            os.remove('%s/checkpoint_step%d.tar' %(checkpoint_dir, step-10))

def update_dropout(model, step, total_steps, args):
    if 1.0*step < args.stepFrac * total_steps:
        p_drop = args.pDropMax * step/(args.stepFrac * total_steps)
    else:
        p_drop = max(0,args.pDropMax * (2 * step - total_steps * (args.stepFrac+1)) / (total_steps * (args.stepFrac-1))) # fast decay
    for x in model.modules():
        if isinstance(x, torch.nn.Dropout):
            x.p = p_drop
    return p_drop

def get_dist_dataloader(egs_filepath, args, rank):
    train_dataset = nnet3EgsDLNonIterable(egs_filepath)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    return train_loader

def cleanup():
    dist.destroy_process_group()

def train(rank, args):
    if rank == 0:
        checkpoint_dir = './models/{}__{}'.format(args.modelType, datetime.utcnow().strftime('%Y%m%dT%H%M%S'))
        os.makedirs(checkpoint_dir)
    
    model, loss_fn, optimizer, lr_scheduler, numBatchesPerArk = setup(rank, args)
    
    total_steps = args.numEpochs * args.numArchives
    batchI, step = 0, 0
    logging_loss = 0.0
    #####
    total_steps = 50
    #####
    while step < total_steps:
        starttime_archive = datetime.utcnow()
        archive_id = step % args.numArchives + 1
        if rank == 0:
            print("\nProcessing archive {} (epoch {})".format( archive_id, int(max(1,math.ceil(step/args.numArchives)))))

        dataloader = get_dist_dataloader("{}/egs.{}.ark".format(args.featDir, archive_id), args, rank)
        for _, (X,y) in dataloader:
            X = X['matrix'].to(rank)
            y = y['matrix'][0][0][0].to(rank)

            try:
                assert max(y) < args.numSpkrs and min(y) >= 0
            except:
                print('Read an out of range value at iter %d' %iter)
                continue
            if torch.isnan(X).any():
                print('Read a nan value at iter %d' %iter)
                continue

            loss = None
            accumulateStepSize = 4
            preFetchBatchI = 0  # this counter within the prefetched batches only

            print("preFetchBatchI", preFetchBatchI)
            print("int(len(y) / args.batchSize)", int(len(y) / args.batchSize))
            print("accumulateStepSize", accumulateStepSize)
            print("int(len(y) / args.batchSize) - accumulateStepSize", int(len(y) / args.batchSize) - accumulateStepSize)
            while preFetchBatchI < int(len(y) / args.batchSize) - accumulateStepSize:

                # Accumulated gradients used
                optimizer.zero_grad()
                for _ in range(accumulateStepSize):
                    batchI += 1
                    preFetchBatchI += 1

                    # forward step
                    outputs = model(X[preFetchBatchI * args.batchSize : (preFetchBatchI+1) * args.batchSize,:,:].permute(0,2,1))
                
                    # calculate loss
                    loss = loss_fn(outputs, y[preFetchBatchI*args.batchSize:(preFetchBatchI+1)*args.batchSize].squeeze())

                    if np.isnan(loss.item()):
                        print('Nan encountered at iter %d. Exiting..' %iter)
                        sys.exit(1)

                    # backpropagation
                    loss.backward()
                    logging_loss += loss.item()

                optimizer.step()
                lr_scheduler.step()  # Update Learning Rate

        # Log and save checkpoint after finishing archive file
        if rank == 0:
            print('Archive processing time: {}'.format(datetime.utcnow() - starttime_archive))
            logging_loss = 0.0

            # Compute validation loss, update LR if using plateau rule
            # valAcc = computeValidAccuracy(args, checkpoint_dir)
            # logger.info('Validation accuracy is %1.2f precent' %(valAcc))

            if loss:
                save_checkpoint(step, archive_id, model, optimizer, loss, checkpoint_dir, args)
            else:
                sys.exit("ERROR: loss = None")

        p_drop = update_dropout(model, step, total_steps, args)
        if rank == 0:
            print('Dropout updated to {}'.format(p_drop))

        step += 1

    cleanup()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', '12355'
    parser = getParams()
    args = parser.parse_args()
    args.world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    torch.manual_seed(0)
    np.random.seed(0)
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
