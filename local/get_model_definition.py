#!/usr/bin/env python3
import glob
import os
import sys
import torch
from collections import OrderedDict
import argparse
from torch_xvector.models import *
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--modelType', default='xvector', help='Refer train_utils.py ')
parser.add_argument('--numSpkrs', default=7323, type=int, help='Number of output labels for model')
parser.add_argument('checkpointDir', help='Directory containing the model checkpoints')
args = parser.parse_args()



# Check for trained model
try:
    checkpointFile = max(glob.glob(args.checkpointDir+'/*.tar'), key=os.path.getctime)
except ValueError:
    print("[ERROR] No trained model has been found in {}.".format(args.checkpointDir) )
    sys.exit(1)

checkpoint_state_dict = torch.load(checkpointFile)['model_state_dict']
new_state_dict = OrderedDict()
for k, v in checkpoint_state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
    else:
        new_state_dict[k] = v

pprint(list(new_state_dict.keys()))
