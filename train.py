import torch as torch
import os
import random
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from ourmodel import AE
from data_process import *

parser = argparse.ArgumentParser(description='version 1.0')
parser.add_argument('--dataset', type=str, default='cora') 
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--negsamp_ratio', type=int, default=1)
args = parser.parse_args()


#载入数据
data=RealDataset(args.dataset)
print(
            "{} |Data Dimension: {}| Data Noise Ratio:{}".format(
            args.dataset.upper(), data.ft_size, '%.4f'%(data.truth.sum()/data.nb_nodes)
            )
        )
#模型
data.docuda()
#with tqdm(total=args.num_epoch) as pbar:
ba,bf,lbl=data.get_babf(4,[0,1,2,3],args.negsamp_ratio)
ba1,bf1,lbl=data.get_babf(4,[0,6,2,1],args.negsamp_ratio)
