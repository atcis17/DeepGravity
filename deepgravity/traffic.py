from __future__ import print_function

import argparse

import torch.optim as optim
import torch.utils.data.distributed

import pandas as pd
import numpy as np

import random

import os

import time

from importlib.machinery import SourceFileLoader

# Training settings
parser = argparse.ArgumentParser(description='DeepGravity')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--mode', default='train', help='Can be train or test')
# Model arguments
parser.add_argument('--tessellation-area', default='United Kingdom',
                    help='The area to tessel if a tessellation is not provided')
parser.add_argument('--tessellation-size', type=int, default=25000,
                    help='The tessellation size (meters) if a tessellation is not provided')
parser.add_argument('--dataset', default='new_york', help='The dataset to use')

# Dataset arguments
parser.add_argument('--tile-id-column', default='tile_ID',
                    help='Column name of tile\'s identifier')
parser.add_argument('--tile-geometry', default='geometry',
                    help='Column name of tile\'s geometry')

parser.add_argument('--oa-id-column', default='oa_ID',
                    help='Column name of oa\'s identifier')
parser.add_argument('--oa-geometry', default='geometry',
                    help='Column name of oa\'s geometry')

parser.add_argument('--flow-origin-column', default='origin',
                    help='Column name of flows\' origin')
parser.add_argument('--flow-destination-column', default='destination',
                    help='Column name of flows\' destination')
parser.add_argument('--flow-flows-column', default='flow',
                    help='Column name of flows\' actual value')

args = parser.parse_args()


# global settings
model_type = 'DG'
data_name = args.dataset

# random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# loading DataLoader and utilities
path = 'deepgravity/data_loader.py'
dgd = SourceFileLoader('dg_data', path).load_module()
path = 'deepgravity/utils.py'
utils = SourceFileLoader('utils', path).load_module()

# set the device
args.cuda = args.device.find("gpu") != -1

if args.device.find("gpu") != -1:
    torch.cuda.manual_seed(args.seed)
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

# check if raw data exists and otherwise stop the execution
if not os.path.isdir('deepgravity/data/' + data_name):
    raise ValueError('There is no dataset named ' + data_name + ' in ./data/')

db_dir = 'deepgravity/data/' + data_name
