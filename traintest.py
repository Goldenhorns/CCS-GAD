import torch as torch
import os
import sys
import random
import torch.utils.data as data
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse
from model import AE, Global
from data_process import RealDataset
from utils import *
from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )

class Solver_graphRCA:
    def __init__(
        self,
        data_name,
        hidden_dim,
        seed,
        learning_rate,
        batch_size,  
        max_epochs,  
        coteaching=1.0,  #  whether selects sample based on loss value
        oe=0.0, 
        subgraph_size=4,
        negsamp_round=1,
        dropout=0.3,
        alpha_G=0.3,
        alpha_L=0.8,
        testround=100,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.learning_rate = learning_rate
        self.seed = seed
        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.beta = 0.0  # initially, select all data
        self.negsamp_round=negsamp_round
        self.subgraph_size=subgraph_size
        self.dropout=dropout
        self.batch_size=batch_size
        self.hidden_dim = hidden_dim
        self.testround=testround
        self.dataset = RealDataset(data_name)
        self.nb_nodes=self.dataset.nb_nodes
        self.data_anomaly_ratio = self.dataset.truth.sum()/self.dataset.nb_nodes + oe
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        self.batch_num = self.nb_nodes // self.batch_size + 1
        self.input_dim = self.dataset.ft_size

        self.dropout_G=0.3
        self.hidden_dim_G=45
        self.lr_G=5e-3
        self.alpha_G=alpha_G
        self.score_G=0
        self.b=alpha_L

        print(
            "{}| Data dimension: {}| Data noise ratio:{}".format(
                self.data_name.upper(), self.input_dim, '%0.4f'%self.data_anomaly_ratio
            )
        )

        self.decay_ratio = abs(self.beta - (1 - self.data_anomaly_ratio)) / (
                    self.max_epochs / 2
                )

        self.ae = None
        self.build_model()

        if torch.cuda.is_available():
            self.b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.negsamp_round]).cuda())
        else:
            self.b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.negsamp_round]))

    def build_model(self):
        self.ae = AE(
            feat_size=self.input_dim, hidden_size=self.hidden_dim,
            negsamp_round=self.negsamp_round, dropout=self.dropout
        )
        self.ae = self.ae.to(self.device)

    def get_idx(self, all_idx, batch_idx):
        is_final_batch = (batch_idx == (self.batch_num - 1))
        if not is_final_batch:
            idx = all_idx[batch_idx * self.batch_size: (batch_idx + 1) * self. batch_size]                            
        else:
            idx = all_idx[batch_idx * self.batch_size:]                            
        cur_batch_size = len(idx)
        return cur_batch_size,idx
    
    def get_idx_test(self, all_idx, batch_idx,test_batch,batch_num):
        is_final_batch = (batch_idx == (batch_num - 1))
        if not is_final_batch:
            idx = all_idx[batch_idx * test_batch: (batch_idx + 1) * test_batch]                            
        else:
            idx = all_idx[batch_idx * test_batch:]                            
        cur_batch_size = len(idx)
        return cur_batch_size,idx

    def train(self):
        auc_G=0
        adj, attrs, label, adj_label = load_anomaly_detection_dataset(self.data_name)
        adj = torch.FloatTensor(adj)
        adj_label = torch.FloatTensor(adj_label)
        attrs = torch.FloatTensor(attrs)
        model_G = Global(feat_size = self.input_dim, hidden_size = self.hidden_dim_G, dropout = self.dropout_G)

        adj = adj.to(self.device)
        adj_label = adj_label.to(self.device)
        attrs = attrs.to(self.device)
        model_G = model_G.cuda()

        optimizer_G = torch.optim.Adam(model_G.parameters(), lr = self.lr_G)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)

        self.ae.eval()
        with tqdm(total=self.max_epochs,ncols=100,colour='blue') as pbar:
            pbar.set_description('Training')
            for epoch in range(self.max_epochs):  
                loss_full_batch = torch.zeros((self.nb_nodes,1))
                if torch.cuda.is_available():
                    loss_full_batch = loss_full_batch.cuda()
                all_idx = list(range(self.dataset.nb_nodes))
                random.shuffle(all_idx)

                model_G.train()
                optimizer.zero_grad()
                A_hat, X_hat, x_g= model_G(attrs, adj)
                loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, self.alpha_G)
                l = torch.mean(loss)
                l.backward(retain_graph=True)
                optimizer_G.step()
                for batch_idx in tqdm(range(self.batch_num),leave=False,ncols=100,colour='green'):
                    cur_batch_size, idx = self.get_idx(all_idx, batch_idx)
                    x_g_b=x_g[idx]

                    ba, bf, lbl= self.dataset.get_babf_raw(self.subgraph_size,idx,self.negsamp_round)
                    
                    n = bf.shape[0]
                    n_selected = int(n * (1-self.beta))

                    if self.coteaching == 0.0:
                        n_selected = n
                    if batch_idx == 0:
                        current_ratio = "{}/{}".format(n_selected, n)

                    optimizer.zero_grad()
                    with torch.no_grad():
                        self.ae.eval()
                        ret1, ret2= self.ae(bf, ba, bf, ba, x_g_b)
                        loss_all_1 = self.b_xent(ret1, lbl)
                        loss_all_2 = self.b_xent(ret2, lbl)

                        loss1=-(loss_all_1[:cur_batch_size]-loss_all_1[cur_batch_size:])
                        loss2=-(loss_all_2[:cur_batch_size]-loss_all_2[cur_batch_size:])

                        loss1=torch.squeeze(loss1,1)
                        loss2=torch.squeeze(loss2,1)
                        
                        _, index1 = torch.sort(loss1) 
                        _, index2 = torch.sort(loss2)

                        index1 = index1[:n_selected]
                        index2 = index2[:n_selected]

                        ba1=ba[index2,:,:]
                        bf1=bf[index2,:,:]
                        
                        ba2=ba[index1,:,:]
                        bf2=bf[index1,:,:]
                        lbl = torch.unsqueeze(torch.cat((torch.ones(n_selected), torch.zeros(n_selected * self.negsamp_round))), 1)
                        lbl=lbl.cuda()
                    self.ae.train()

                    ret1, ret2 = self.ae(bf1,ba1,bf2,ba2, x_g_b)

                    loss =self.b_xent(ret1, lbl) + self.b_xent(ret2, lbl)
                    
                    loss = loss.sum()
                
                    mean_loss=loss/cur_batch_size
                    mean_loss =mean_loss.detach().cpu().numpy()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                
                if epoch%10 == 0 or epoch == self.max_epochs - 1:
                    model_G.eval()
                    A_hat, X_hat, x_g= model_G(attrs, adj)
                    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, self.alpha_G)
                    score = loss.detach().cpu().numpy()
                    auc_G_t=roc_auc_score(label, score)
                    if auc_G_t > auc_G:
                        auc_G=auc_G_t
                        self.score_G=score
                        self.x_g=x_g
                pbar.update(1)
                if self.beta < self.data_anomaly_ratio:
                    self.beta = min(
                        self.data_anomaly_ratio, self.beta + self.decay_ratio
                    )

            
    def get_value(self):
        return self.alpha_G, self.b
    
    def test(self,test_batch=3000):
        self.ae.train()
        multi_round_ano_score = np.zeros((self.testround, self.nb_nodes))
        with tqdm(total=self.testround,ncols=100,colour='blue') as pbar:
            pbar.set_description('Testing ')
            for round in range(self.testround): 
                with torch.no_grad():
                    all_idx = list(range(self.nb_nodes))
                    random.shuffle(all_idx)
                    test_batch_num=self.nb_nodes //test_batch + 1
                    for batch_idx in tqdm(range(test_batch_num),leave=False,ncols=100,colour='green'):
                        
                        cur_batch_size, idx = self.get_idx_test(all_idx, batch_idx,test_batch,test_batch_num)

                        ba,bf,lbl=self.dataset.get_babf_raw(self.subgraph_size,idx,self.negsamp_round)

                        ret1, ret2 = self.ae(bf,ba,bf,ba,self.x_g)

                        loss1= self.b_xent(ret1, lbl)
                        loss2= self.b_xent(ret2, lbl)

                        loss1=-(loss1[:cur_batch_size]-loss1[cur_batch_size:])
                        loss2=-(loss2[:cur_batch_size]-loss2[cur_batch_size:])

                        loss=loss1+loss2
                        loss = loss.mean(dim=1)
                        loss = loss.data.cpu().numpy()
                        multi_round_ano_score[round, idx]=loss
                    pbar.update(1)

        y=self.dataset.truth
        ano_score_final=0
        ano_score_final_t = -self.b*np.mean(multi_round_ano_score, axis=0)+(1-self.b)*self.score_G
        auc=roc_auc_score(y, ano_score_final_t) 
        return auc, ano_score_final




