# simple GNN auto-encoder like model
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-gcn-layer


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

args = ArgumentParser(description = "train a graph neural network")
args.add_argument("--num_nodes", type = int, default = 30, help = "number of nodes in the sample graphs")
args.add_argument("--node_args", type = int, default = 4, help = "number of meta fields to add")
args.add_argument("--hdim", type=int, default=32, help="hidden dimension")
args.add_argument("--p", type = float, default = 0.3, help = "probability of connection nodes in sample graphs")
args.add_argument("--num", type = int, default = 1000, help = "number of samples of dummy language")
args.add_argument("--batch_size", type = int, default = 8, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 50, help = "number of training epochs")
args = args.parse_args()

data = []
node_fields = [f"i{i}" for i in range(args.node_args)] + [f"t{i}" for i in range(args.node_args)]
while len(data) < args.num:
    g = nx.binomial_graph(args.num_nodes, args.p)
    if nx.is_connected(g):
        g2 = nx.Graph()
        for n in list(g.nodes):
            g2.add_node(n, **{f: [np.random.random()] for f in node_fields})
        for e in list(g.edges):
            g2.add_edge(*e)
        data.append(tgx.utils.from_networkx(g2))

dataloader = tgx.data.DataLoader(data, args.batch_size)
# for idx, b in enumerate(dataloader):
#     if idx:
#         break
#     print(b)

# --- Making the model --- #
class GCNConv(MessagePassing):
    def __init__(self, config, node_fields, aggr = None):
        # "Add" aggregation (Step 5).
        super().__init__(aggr = aggr)
        self.config = config
        self.node_fields = node_fields
        self.lin = torch.nn.Linear(config.node_args, config.hdim)
        self.lin2 = nn.Linear(config.hdim, config.node_args)

    def forward(self, batched_data):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # take only the inputs
        x = torch.cat([
            getattr(batched_data, k) for k in self.node_fields if "i" in k
        ], dim = -1)
        edge_index = batched_data.edge_index

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        x = self.propagate(edge_index, x=x, norm=norm)
        return self.lin2(x)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


gnn = GCNConv(args, node_fields, "max")
loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(gnn.parameters(), lr=0.1)
print(gnn)

# run one epoch and let's see what happens
print("----- BEFORE -----")
with torch.no_grad():
    dataloader = tgx.data.DataLoader(data, 3)
    for i,b in enumerate(dataloader):
        if i:
            break
        print("Batch:", b)
        out = gnn(b)
        targ = torch.cat([getattr(b, k) for k in node_fields if "t" in k], dim=-1)
        # loss = loss_function(out, targ)
        print("out_shape:", out.shape)
        print("targ_shape:", targ.shape)
        print(targ[0], out[0])
        # print("loss:", loss.item())

# train for sometime
print("----- LEARNING -----")
pbar = trange(args.epochs, ncols = 100)
prev_loss = -1 
for e in pbar:
    this_ep_loss = []
    pbar.set_description(f"Epoch: {e+1}/{args.epochs}; loss: {round(prev_loss, 3)}")
    for i, b in enumerate(dataloader):
        gnn.zero_grad()
        outputs = gnn(b)
        target = torch.cat([getattr(b, k) for k in node_fields if "t" in k], dim=-1)
        # print("TR:", target.shape)
        loss = loss_function(outputs, target)
        # print("loss:", loss)
        this_ep_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    prev_loss = sum(this_ep_loss)/len(this_ep_loss)

# see what it has learned
print("----- AFTER -----")
with torch.no_grad():
    dataloader = tgx.data.DataLoader(data, 3)
    for i, b in enumerate(dataloader):
        if i:
            break
        print("Batch:", b)
        out = gnn(b)
        targ = torch.cat([getattr(b, k) for k in node_fields if "t" in k], dim=-1)
        print("out_shape:", out.shape)
        print("targ_shape:", targ.shape)
        print(targ[0], out[0])
