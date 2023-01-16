import torch as torch
import os

import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from ourmodel import AE
from data_process import RealDataset


class Solver_graphRCA:
    def __init__(
        self,
        data_name,
        hidden_dim=128,  # number of hidden neurons in RCA
        seed=0,  # random seed
        learning_rate=1e-3,  # learning rate
        batch_size=128,  #  batchsize
        max_epochs=100,  #  training epochs
        coteaching=1.0,  #  whether selects sample based on loss value
        oe=0.0,  # how much we overestimate the ground-truth anomaly ratio
        #missing_ratio=0.0,  # missing ratio in the data
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()

        self.data_name = data_name
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #self.missing_ratio = missing_ratio
        self.result_path = "./results/GraphRCA/{}/".format(
                data_name)

        self.learning_rate = learning_rate
        self.seed = seed
        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.beta = 0.0  # initially, select all data
        self.alpha = 0.5
        self.negsamp_round=1
        self.dropout=0.01

        self.dataset = RealDataset(data_name)
        self.data_anomaly_ratio = self.dataset.truth.sum()/self.dataset.nb_nodes + oe
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        self.input_dim = self.dataset.ft_size
        self.hidden_dim = hidden_dim

        n_sample = self.dataset.nb_nodes
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

    def train(self):
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)
        self.ae.eval()
        loss_mse = torch.nn.MSELoss(reduction='none')
        if self.data_name == 'optdigits':
            loss_mse = torch.nn.BCELoss(reduction='none')

        for epoch in tqdm(range(self.max_epochs)):  # train 3 time classifier
            for i, (x, y) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                n = x.shape[0]
                n_selected = int(n * (1-self.beta))

                if config.coteaching == 0.0:
                    n_selected = n
                if i == 0:
                    current_ratio = "{}/{}".format(n_selected, n)

                optimizer.zero_grad()

                with torch.no_grad():
                    self.ae.eval()
                    z1, z2, xhat1, xhat2 = self.ae(x.float(), x.float())

                    error1 = loss_mse(xhat1, x)
                    error1 = error1
                    error2 = loss_mse(xhat2, x)
                    error2 = error2

                    error1 = error1.sum(dim=1)
                    error2 = error2.sum(dim=1)
                    _, index1 = torch.sort(error1)
                    _, index2 = torch.sort(error2)

                    index1 = index1[:n_selected]
                    index2 = index2[:n_selected]

                    x1 = x[index2, :]
                    x2 = x[index1, :]


                self.ae.train()
                z1, z2, xhat1, xhat2 = self.ae(x1.float(), x2.float())
                loss = loss_mse(xhat1, x1) + loss_mse(xhat2, x2)
                loss = loss.sum()
                loss.backward()
                optimizer.step()

            if self.beta < self.data_anomaly_ratio:
                self.beta = min(
                    self.data_anomaly_ratio, self.beta + self.decay_ratio
                )
'''
    def test(self):
        print("======================TEST MODE======================")
        self.ae.train()
        mse_loss = torch.nn.MSELoss(reduction='none')
        if self.data_name == 'optdigits':
            mse_loss = torch.nn.BCELoss(reduction='none')

        error_list = []
        for _ in range(1000):  # ensemble score over 100 stochastic feedforward
            with torch.no_grad():
                for _, (x, y) in enumerate(self.testing_loader):  # testing data loader has n_test batchsize, if it is image data, need change this part
                    y = y.data.cpu().numpy()
                    x = x.to(self.device).float()
                    _, _, xhat1, xhat2 = self.ae(x.float(), x.float())
                    error = mse_loss(xhat1, x) + mse_loss(xhat2, x)
                    error = error.mean(dim=1)
                error = error.data.cpu().numpy()
                error_list.append(error)
        error_list = np.array(error_list)
        error = error_list.mean(axis=0)
        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )
        gt = y.astype(int)

        thresh = np.percentile(error, self.dataset.__anomalyratio__() * 100)
        print("Threshold :", thresh)

        pred = (error > thresh).astype(int)
        gt = y.astype(int)
        auc = roc_auc_score(gt, error)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

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
'''
