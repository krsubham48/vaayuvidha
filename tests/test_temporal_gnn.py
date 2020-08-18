# implementation of tempopral GNN on static graphs
# the sample data is like the weather data that it is a very long 
# sequence that si broken up into sub sequences for sample

import numpy as np
import torch
import networkx as nx
import torch.nn as nn
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import trange

import torch_geometric as tgx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# set seeds for reproducibility
np.random.seed(4)
torch.manual_seed(4)

args = ArgumentParser(description = "train a temporal graph neural network on weather like data")
args.add_argument("--num_nodes", type = int, default = 30, help = "number of nodes in the sample graphs")
args.add_argument("--node_args", type = int, default = 4, help = "number of meta fields to add")
args.add_argument("--hdim", type=int, default=32, help="hidden dimension")
args.add_argument("--p", type = float, default = 0.3, help = "probability of connection nodes in sample graphs")
args.add_argument("--num", type = int, default = 100, help = "number of steps")
args.add_argument("--maxlen", type = int, default = 10, help = "maximm training length")
args.add_argument("--batch_size", type = int, default = 1, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 50, help = "number of training epochs")
args = args.parse_args()

assert args.num > args.batch_size, "Number of steps has to be > batch size"

data = []
node_fields = [f"i{i}" for i in range(args.node_args)] + [f"t{i}" for i in range(args.node_args)]
g = nx.binomial_graph(args.num_nodes, args.p)
while not nx.is_connected(g):
    g = nx.binomial_graph(args.num_nodes, args.p)
while len(data) < args.num:
    g2 = nx.Graph()
    for n in list(g.nodes):
        g2.add_node(n, **{f: [np.random.random()] for f in node_fields})
    for e in list(g.edges):
        g2.add_edge(*e)
    data.append(g2)

data_tgx = [tgx.utils.from_networkx(g) for g in data]

# need to build wrapper over tgx.data.DataLoader
class DataLoader(object):
    def __init__(self, data: list, maxlen:int, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

    def __iter__(self):
        data = []
        for i in range(len(self.data) - self.maxlen):
            data.append(self.data[i:i+self.maxlen])
        return iter(tgx.data.DataLoader(data, self.batch_size))
        
dl = DataLoader(data_tgx, maxlen = args.maxlen, batch_size=args.batch_size)

for idx, item in enumerate(dl):
    if idx:
        break
    print(item)

