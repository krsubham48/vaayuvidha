# implementation of tempopral GNN on static graphs
# the sample data is like the weather data that it is a very long 
# sequence that is broken up into sub sequences for sample

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
args.add_argument("--batch_size", type = int, default = 32, help = "batch size for training")
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
        # print(f"%%%%%%%, {self.data}")

    def __iter__(self):
        # return [time_step * batch_size, ... ] seequence and caller at runtime would convert to blocks
        data = [] # [total_samples, maxlen]
        for i in range(len(self.data) - self.maxlen):
            data.extend(self.data[i:i+self.maxlen])
        
        data = []
        for i in range(len(self.data) - self.maxlen + 1):
            data.append(self.data[i:i+self.maxlen])
        idx = np.arange(len(data)); np.random.shuffle(idx)
        
        data_time_batched = []; max_idx = -1
        for i in range(0, (len(data) // self.batch_size) + int(len(data) % self.batch_size > 0)):
            samples = [data[idx[i*self.batch_size + j]] for j in range(min(self.batch_size, len(data) - max_idx))]
            max_idx = (i+1) * self.batch_size
            data_time_batched.append(samples)
        return iter(data_time_batched)

# define the custom dataloader
dl = DataLoader(data_tgx, maxlen = args.maxlen, batch_size=args.batch_size)
for idx, flat_seq in enumerate(dl):
    if idx:
        break
    output = []
    print(f"flat_seq: {len(flat_seq)}; flat_seq[0]: {len(flat_seq[0])}; maxlen: {args.maxlen}, batch_size: {args.batch_size}")
    for t in range(args.maxlen):
        this_sample = None
        batch = tgx.data.Batch.from_data_list([seq[t] for seq in flat_seq])
        print(f"T{t} => {batch}")

        # hidden_state = None
        # input_this = tgx.utils.Batch([tgx.utils.from_networkx(seq[i]) for seq in flat_seq])
        # out_graphs, hidden_state = model(torch.from_numpy(input_this), hidden_state)
        # output.append(out_graphs)
    

# making that bitch of a model
class Mdel 