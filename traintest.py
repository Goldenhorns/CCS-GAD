import torch as torch
import os
import random
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse
from ourmodel import AE
from data_process import RealDataset
'''
come true train and test
'''
class Solver_graphRCA:
    def __init__(
        self,
        data_name,
        hidden_dim,  # number of hidden neurons in RCA
        seed,  # random seed
        learning_rate,  # learning rate
        batch_size,  #  batchsize
        max_epochs,  #  training epochs
        coteaching=1.0,  #  whether selects sample based on loss value
        oe=0.0,  # how much we overestimate the ground-truth anomaly ratio
        subgraph_size=4,
        negsamp_round=1,
        dropout=0.3,
        testround=100,
        balance=0.5
        #missing_ratio=0.0,  # missing ratio in the data
    ):
        # Data loader
        # read data here
        print("=========================Init=========================")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()

        self.data_name = data_name
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #self.missing_ratio = missing_ratio
        self.result_path = "./results/{}/".format(data_name)

        self.learning_rate = learning_rate
        self.seed = seed
        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.beta = 0.0  # initially, select all data
        self.alpha = 0.5 #?
        self.balance=balance
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

        n_sample = self.nb_nodes
        self.n_train = n_sample 
        self.n_test = n_sample

        print(
            "{}| Data dimension: {}| Data noise ratio:{}".format(
                self.data_name.upper(), self.input_dim, '%0.4f'%self.data_anomaly_ratio
            )
        )

        self.decay_ratio = abs(self.beta - (1 - self.data_anomaly_ratio)) / (
            self.max_epochs / 2
        )

        self.ae = None
        self.discriminator = None
        self.build_model()
        self.print_network()

    def build_model(self):
        self.ae = AE(
            feat_size=self.input_dim, hidden_size=self.hidden_dim,
            negsamp_round=self.negsamp_round, dropout=self.dropout
        )
        self.ae = self.ae.to(self.device)

    def print_network(self):
        num_params = 0
        for p in self.ae.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def loss_func(self,  adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 2))
        attribute_cost = torch.mean(attribute_reconstruction_errors)
        # structure reconstruction loss
        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 2))
        structure_cost = torch.mean(structure_reconstruction_errors)
        cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors
        return cost

    def getidx(self, all_idx, batch_idx):
        is_final_batch = (batch_idx == (self.batch_num - 1))
        if not is_final_batch:
            idx = all_idx[batch_idx * self.batch_size: (batch_idx + 1) * self. batch_size]                            
        else:
            idx = all_idx[batch_idx * self.batch_size:]                            
        cur_batch_size = len(idx)
        return cur_batch_size,idx

    def train(self):
        print("======================Train MODE======================")
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)
        self.ae.eval()
#region
        #loss_mse = torch.nn.MSELoss(reduction='none')
        #if self.data_name == 'optdigits':
            #loss_mse = torch.nn.BCELoss(reduction='none')
#endregion
        with tqdm(total=self.max_epochs,ncols=100,colour='blue') as pbar:
                pbar.set_description('Training')
                for epoch in range(self.max_epochs):  
                    loss_full_batch = torch.zeros((self.nb_nodes,1))
                    if torch.cuda.is_available():
                        loss_full_batch = loss_full_batch.cuda()
                    
                    all_idx = list(range(self.dataset.nb_nodes))
                    random.shuffle(all_idx)
                    total_loss = 0.

                    for batch_idx in tqdm(range(self.batch_num),leave=False,ncols=100,colour='green'):
                        _, idx = self.getidx(all_idx, batch_idx)
                        ba, bf, _= self.dataset.get_babf(self.subgraph_size,idx,self.negsamp_round)
                        
                        n = bf.shape[0]
                        n_selected = int(n * (1-self.beta))

                        if self.coteaching == 0.0:
                            n_selected = n
                        if batch_idx == 0:
                            current_ratio = "{}/{}".format(n_selected, n)

                        optimizer.zero_grad()
                        with torch.no_grad():
                            self.ae.eval()
                            A_hat1, x_hat1,re1, \
                            A_hat2, x_hat2,ret2 = self.ae(bf,ba,bf,ba)##后续添加ret

                            error1= self.loss_func(ba, A_hat1, bf, x_hat1, self.balance)
                            error2= self.loss_func(ba, A_hat2, bf, x_hat2, self.balance)
                            error1 = error1.sum(dim=1)
                            error2 = error2.sum(dim=1)

                            _, index1 = torch.sort(error1) #index为原来的数据下标
                            _, index2 = torch.sort(error2)

                            index1 = index1[:n_selected]
                            index2 = index2[:n_selected]
    
                            ba1=ba[index2,:,:]
                            bf1=bf[index2,:,:]
                            
                            ba2=ba[index1,:,:]
                            bf2=bf[index1,:,:]

                        

                        self.ae.train()
                        A_hat1, x_hat1,re1, \
                            A_hat2, x_hat2,ret2 = self.ae(bf1,ba1,bf2,ba2)

                        loss = self.loss_func(ba1, A_hat1, bf1, x_hat1, self.balance) \
                        + self.loss_func(ba2, A_hat2, bf2, x_hat2, self.balance)

                        loss = loss.sum()
                        mean_loss=loss/self.nb_nodes
                        mean_loss =mean_loss.detach().cpu().numpy()
                        loss.backward()
                        optimizer.step()

                        
                        pbar.set_postfix(loss=mean_loss)
                    pbar.update(1)
                    if self.beta < self.data_anomaly_ratio:
                        self.beta = min(
                            self.data_anomaly_ratio, self.beta + self.decay_ratio
                        )

    def test(self):
        print("======================TEST MODE======================")
        self.ae.train()

        #mse_loss = torch.nn.MSELoss(reduction='none')
        #if self.data_name == 'optdigits':
            #mse_loss = torch.nn.BCELoss(reduction='none')
        
        multi_round_ano_score = np.zeros((self.testround, self.nb_nodes))
        with tqdm(total=self.testround,ncols=100,colour='blue') as pbar:
            pbar.set_description('Testing')
            for round in range(self.testround):  # ensemble score over 100 stochastic feedforward
                with torch.no_grad():
                    all_idx = list(range(self.nb_nodes))
                    random.shuffle(all_idx)
                    for batch_idx in tqdm(range(self.batch_num),leave=False,ncols=100,colour='green'):# testing data loader has n_test batchsize, if it is image data, need change this part
                        
                        _, idx = self.getidx(all_idx, batch_idx)

                        ba,bf,lbl=self.dataset.get_babf(self.subgraph_size,idx,self.negsamp_round)

                        A_hat1, x_hat1,re1, \
                        A_hat2, x_hat2,ret2 = self.ae(bf,ba,bf,ba)

                        error1= self.loss_func(ba, A_hat1, bf, x_hat1, 0.5)
                        error2= self.loss_func(ba, A_hat2, bf, x_hat2, 0.5)
                        error=error1+error2
                        error = error.mean(dim=1)
                        error = error.data.cpu().numpy()
                        multi_round_ano_score[round, idx]=error
                    pbar.update(1)
                
        y=self.dataset.truth
        ano_score_final = np.mean(multi_round_ano_score, axis=0)
        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )

        thresh = np.percentile(ano_score_final, self.data_anomaly_ratio * 100)
        print("Threshold :", thresh)
        
        pred = (ano_score_final > thresh).astype(int)
    
        auc = roc_auc_score(y, ano_score_final)
        accuracy = accuracy_score(y, pred)
        precision, recall, f_score, support = prf(y, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
                accuracy, precision, recall, f_score, auc
            )
        )

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
                "auc": auc,
            },
        )
        print("result save to {}".format(self.result_path))
        return accuracy, precision, recall, f_score, auc
