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
    
    def get_babf(self,subgraph_size,idx,negsamp_ratio):
        #所需数据准备
        ba = []
        bf = []
        cur_batch_size = len(idx)
        dgl_graph = adj_to_dgl_graph(self.adj)       
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

        lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * negsamp_ratio))), 1)
        
        for i in idx:
            cur_adj = self.norm_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = self.norm_feat[:, subgraphs[i], :]
            ba.append(cur_adj)
            bf.append(cur_feat)

        ba = torch.cat(ba)
        ba =  ba.cuda()
        bf = torch.cat(bf)
        bf =  bf.cuda()

        return ba,bf,lbl

    def get_babf_raw(self,subgraph_size,idx,negsamp_ratio):
        #所需数据准备
        ba = []
        bf = []
        cur_batch_size = len(idx)
        dgl_graph = adj_to_dgl_graph(self.adj)       
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

        added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
        added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((cur_batch_size, 1, self.ft_size))
        lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * negsamp_ratio))), 1)
        if torch.cuda.is_available():
            lbl = lbl.cuda()

            added_adj_zero_row = added_adj_zero_row.cuda()
            added_adj_zero_col = added_adj_zero_col.cuda()
            added_feat_zero_row = added_feat_zero_row.cuda()
        for i in idx:
            cur_adj = self.norm_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = self.norm_feat[:, subgraphs[i], :]
            ba.append(cur_adj)
            bf.append(cur_feat)

        ba = torch.cat(ba)
        ba =  ba.cuda()
        ba = torch.cat((ba, added_adj_zero_row), dim=1)
        ba = torch.cat((ba, added_adj_zero_col), dim=2)
        bf = torch.cat(bf)
        bf =  bf.cuda()
        bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)
        return ba,bf,lbl