import torch as torch
import os
import random
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from model import AE
from data_process import *

#载入数据
data=RealDataset("cora")

#模型
data.docuda()
data.node_sub(4)