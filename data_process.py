import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import neighbors
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.io as sio
from utils import *
from sklearn.preprocessing import MinMaxScaler
'''
读取数据，处理成->(节点-子图对）
dataset=["cora","ACM","Flickr","pubmed","citeseer","BlogCatalog"]
**Flickr换成minmax试试**
'''
class RealDataset():
    def __init__(self,dataset):
        feat, truth, adj=load_data(dataset)
        self.adj=adj
        self.feat=feat
        self.truth=truth #标签
        self.norm_adj=preprocess_adj(adj)
        self.norm_feat=preprocess_features(feat)
        #数据属性
        self.nb_nodes = self.norm_feat.shape[0]
        self.ft_size = self.norm_feat.shape[1]
        #转换成tensorfloat
        self.norm_feat = torch.FloatTensor(self.norm_feat[np.newaxis])
        self.norm_adj = torch.FloatTensor(self.norm_adj[np.newaxis])
        self.truth= torch.FloatTensor(self.truth[np.newaxis])
    

    def docuda(self):
        if torch.cuda.is_available():
            split_line()
            print('Using CUDA')
            self.norm_feat = self.norm_feat.cuda()
            self.norm_adj = self.norm_adj.cuda()
            self.truth = self.truth.cuda()
        else:
            split_line()
            print("Using CPU")
    
    def node_sub(self,subgraph_size):
        added_adj_zero_row = torch.zeros((self.nb_nodes, 1, subgraph_size))
        added_adj_zero_col = torch.zeros((self.nb_nodes, subgraph_size + 1, 1))
        added_adj_zero_col[:,-1,:] = 1.
        added_feat_zero_row = torch.zeros((self.nb_nodes, 1, self.ft_size))
        if torch.cuda.is_available():
            added_adj_zero_row = added_adj_zero_row.cuda()
            added_adj_zero_col = added_adj_zero_col.cuda()
            added_feat_zero_row = added_feat_zero_row.cuda()
        split_line()
        print("Subgraph has produced")
