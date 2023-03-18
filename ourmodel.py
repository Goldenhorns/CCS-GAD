import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from numpy import random
class GraphConvolution(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class Encoder(nn.Module):#一层编码器
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x

class Structure_Decoder(nn.Module):#结构解码
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        y=x.transpose(1,2)
        x = x @ y  
        return x

class AvgReadout(nn.Module):#取平均值
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class Discriminator(nn.Module):#求距离
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

class SM(nn.Module):
    def __init__(self, feat_size, hidden_size,negsamp_round, dropout):
        super(SM, self).__init__()
        self.encoder = Encoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
        self.read = AvgReadout()
        self.disc = Discriminator(hidden_size, negsamp_round)
    def forward(self, x, adj):
        x = self.encoder(x, adj)
        x_raw=x[:,:-1,:]
        x_raw[:,-1,:]=x[:,-1,:]
        adj_raw=adj[:,:-1,:-1]
        A_hat = self.struct_decoder(x_raw, adj_raw)

        #disc
        c = self.read(x[:,: -1,:])
        h_mv = x[:,-1,:]
        ret = self.disc(c, h_mv)
        
        return A_hat,ret

class AE(nn.Module):
    def __init__(self, feat_size, hidden_size,negsamp_round, dropout):
        super(AE, self).__init__()
        self.ende1=SM(feat_size, hidden_size,negsamp_round, dropout)
        self.ende2=SM(feat_size, hidden_size,negsamp_round, dropout)
    def forward(self,  x1, adj1, x2, adj2):
        A_hat1,ret1=self.ende1(x1, adj1)
        A_hat2,ret2=self.ende2(x2, adj2)
        return A_hat1,ret1,A_hat2,ret2