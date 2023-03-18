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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RCA")
    parser.add_argument("--seed", type=int, default=1, required=False)
    parser.add_argument("--data", type=str, default="cora", required=False)
    parser.add_argument("--max_epochs", type=int, default=100, required=False)
    parser.add_argument("--hidden_dim", type=int, default=120, required=False)
    parser.add_argument("--batch_size", type=int, default=300, required=False)
    parser.add_argument("--oe", type=float, default=0.0, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.6, required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3, required=False)
    parser.add_argument("--coteaching", type=float, default=1.0, required=False)

    parser.add_argument("--negsamp_round", type=int, default=1, required=False)
    parser.add_argument("--subgraphsize", type=int, default=4, required=False)
    parser.add_argument("--dropout", type=float, default=0, required=False)
    parser.add_argument("--testround", type=int, default=100, required=False)
    parser.add_argument("--balance", type=float, default=0.5, required=False)
    args = parser.parse_args()


    """
    read data
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    Solver = Solver_graphRCA(
        data_name=args.data,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        oe=args.oe,
        subgraph_size=args.subgraphsize,
        negsamp_round=args.negsamp_round,
        dropout=args.dropout,
        testround=args.testround,
        balance=args.balance
    )

    Solver.train()
    Solver.test()
    print("Data {} finished".format(args.data))
