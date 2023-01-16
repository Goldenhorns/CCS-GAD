import torch as torch
import os
import random
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from ourmodel import AE
from data_process import *
from traintest  import *
parser = argparse.ArgumentParser(description='version 1.0')
parser.add_argument('--dataset', type=str, default='cora') 
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--negsamp_ratio', type=int, default=1)
args = parser.parse_args()


#载入数据
nmu=Solver_graphRCA(
        args.dataset,
        hidden_dim=128,  # number of hidden neurons in RCA  
        seed=0,  # random seed
        learning_rate=1e-3,  # learning rate
        batch_size=128,  #  batchsize
        max_epochs=100,  #  training epochs
        coteaching=1.0,  #  whether selects sample based on loss value
        oe=0.0,  # how much we overestimate the ground-truth anomaly ratio
        #missing_ratio=0.0,  # missing ratio in the data
    )
