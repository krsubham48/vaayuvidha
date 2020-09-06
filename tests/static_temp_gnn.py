# implementation of temporal GNN on static graphs
# the sample data is like the weather data that it is a very long 
# sequence that is broken up into sub sequences for sample

from typing import List
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

from torch_scatter.scatter import scatter_max
from torch_scatter import scatter_mean

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
args.add_argument("--drop_p", type = float, default = 0.1, help = "dropout probability")
args.add_argument("--epochs", type = int, default = 50, help = "number of training epochs")
args.add_argument("--use_attention", type = bool, default = True, help = "To use attention in each LSTM layer")
args.add_argument("--num_layers", type = int, default = 3, help = "number of layers")
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
        g2.add_edge(*e, dist=[np.random.random()])
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
# dl = DataLoader(data_tgx, maxlen = args.maxlen, batch_size=args.batch_size)
# for idx, flat_seq in enumerate(dl):
#     if idx:
#         break
#     output = []
#     print(f"flat_seq: {len(flat_seq)}; flat_seq[0]: {len(flat_seq[0])}; maxlen: {args.maxlen}, batch_size: {args.batch_size}")
#     for t in range(args.maxlen):
#         this_sample = None
#         batch = tgx.data.Batch.from_data_list([seq[t] for seq in flat_seq])
#         print(f"T{t} => {batch}")

#         # hidden_state = None
#         # input_this = tgx.utils.Batch([tgx.utils.from_networkx(seq[i]) for seq in flat_seq])
#         # out_graphs, hidden_state = model(torch.from_numpy(input_this), hidden_state)
#         # output.append(out_graphs)
    

# making that bitch of a model
"""Model is super simple and can be considered as follows:

Predicted state at time step "t+1" aggregated using GraphDecoder() Network

(attention at each layer)
      |    [DEC]
   _[att.]_  |\  (Residual Connection to maintain the)
  /        \ | \ (graph structure over the hidden state)
[t-1]------>[t]-|---->[t+1]
             | /
             |/
           [ENC]

Graph state at time step "t" aggregated using GraphEncoder() Network

For graph we use the CGConv from ["Crystal Graph Convolutional Neural Networks for an
Accurate and Interpretable Prediction of Material Properties”](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv

"""

class GraphBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for node_edge <- node + edge_attr
        self.edge_lin = nn.Sequential(
            nn.Linear(int(config.hdim * 2), config.hdim),
            nn.ReLU()
        )
        self.edge_drop = nn.Dropout(config.drop_p)

        # for node_2 <- node + global + node_edge
        self.node_lin = nn.Sequential(
            nn.Linear(int(config.hdim * 3), int(config.hdim * 4)),
            nn.ReLU(),
            nn.Linear(int(config.hdim * 4), config.hdim)
        )
        self.node_drop = nn.Dropout(config.drop_p)

        # for global <- node_2 + global
        self.glob_lin = nn.Sequential(
            nn.Linear(int(config.hdim * 2), config.hdim),
            nn.ReLU()
        )
        self.glob_drop = nn.Dropout(config.drop_p)

    def forward(self, x, edge_index, e, u, batch):
        row, col = edge_index

        # print(row.shape, col.shape, x.shape, x[row].shape, e.shape)

        out = self.edge_lin(torch.cat((x[row], e), dim = -1))
        out, argmax = scatter_max(out, col, dim=0, dim_size=x.size(0))
        out = self.edge_drop(out)
        out = self.node_drop(self.node_lin(torch.cat((out, x, u[batch]), dim = -1)))
        x = x + out # residual

        x_u, argmax = scatter_max(x, batch, dim=0)
        out = self.glob_lin(torch.cat((x_u, u), dim = 1))
        u = u + out # residual
        return x, u


class GraphTemporal(nn.Module):
    def __init__(self, config, inputs: List[str]) -> None:
        super().__init__()

        self.inputs = inputs
        self.num_imput_types = len(inputs)
        self.wte = nn.Embedding(self.num_imput_types, config.hdim)
        self.edge_emb = nn.Linear(1, config.hdim)

        # # using attention is inspired from here
        # # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        # self.attn = nn.Linear(config.hdim * 2, con)
        # self.attn_memory = Variable(torch.zeros())
        
        self.input_layer_names = []
        self.output_layer_names = []
        for inp in inputs:
            # define the input embedding layer
            inp_layer = f"_auto_lin_inp_{inp}"
            out_layer = f"_auto_lin_out_{inp}"
            setattr(self, inp_layer, nn.Linear(1, config.hdim))
            setattr(self, out_layer, nn.Linear(config.hdim, 1))
            self.input_layer_names.append(inp_layer)
            self.output_layer_names.append(out_layer)

        # graph blocks + LSTM
        self.temporal_blocks = nn.ModuleList([nn.LSTMCell(config.hdim, config.hdim),]*config.num_layers)
        self.graph_blocks = nn.ModuleList([GraphBlock(config),] * config.num_layers)

    # def _block_pass(self, x, u, e, gf, lf, edge_index, batch, h):
    #     x, u = gf(x = x, e=e, u=u, edge_index=edge_index, batch=batch)
    #     u, h = lf(u, h)


    def forward(self, batched_data, batch_size, hidden_states = None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # take only the inputs

        if hidden_states is not None:
            assert len(hidden_states) == len(self.temporal_blocks)

        # join different datapoints to one embedding
        x = 0
        for inp, inp_l in zip(self.inputs, self.input_layer_names):
            inp_batch = getattr(batched_data, inp)
            inp_type = self.wte(torch.ones((inp_batch.size(0),)).long() * int(inp[-1]))
            out = inp_type + getattr(self, inp_l)(inp_batch)
            x += out # [N, edim]

        # convert the simple edge attr to edim
        e = self.edge_emb(batched_data.dist)

        # define a few things for first block
        u = torch.zeros((batch_size, x.size(-1)))
        if hidden_states is None:
            hidden_states = [None,]*len(self.temporal_blocks)

        # now pass through multiple layer blocks
        out_hidden_states = []
        for i,(g,l,h) in enumerate(zip(self.graph_blocks, self.temporal_blocks, hidden_states)):
            x, u = g(x=x, e=e, u=u, edge_index=batched_data.edge_index, batch=batched_data.batch)
            u, h = l(u, h)
            out_hidden_states.append(h)

        # break one embedding to different data points
        outs = []
        for out_l in self.output_layer_names:
            outs.append(getattr(self, out_l)(x))
        
        # return the outputs and hidden states
        return outs, out_hidden_states


model = GraphTemporal(args, [f"i{i}" for i in range(args.node_args)])
# print(model)

# dry run model
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
        o, h = model(batched_data = batch, batch_size = args.batch_size)
        print([x.size() for x in o])
        print([x.size() for x in h])

# class GraphEncoderBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.node_lin = nn.Linear(2 * config.hdim, config.hdim)
#         self.graph_lin = nn.Linear(config.hdim + config.graph_dim, config.hdim)
#         self.node_bn = nn.BatchNorm1d(config.hdim)
#         self.graph_bn = nn.BatchNorm1d(config.graph_dim)

#         self.graph_layer = tgx.nn.CGConv()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.bn.reset_parameters()

#     def forward(self, x, edge_index, edge_attr, batch, graph_emb):
#         # x has shape [num_nodes, in_channels], edge_index has shape [2, num_edges]
#         edge_index, _ = add_self_loops(edge_index, num_nodes = x.shape(0))
#         row, col = edge_index

#         # first we perform edge handling
#         node_out = torch.cat([x[row], edge_attr], dim=-1)  # [1, 2*edim]
#         node_out = self.lin(node_out)# do one FF for the node emebdding + residual connection
#         max_pooled_graph = tgx.nn.global_max_pool(node_out, batch) # [bs, ]
#         v = torch.cat([graph_emb, max_pooled_graph], dim = -1)
#         v = self.graph_lin(v)

#         # addition and batch_norm
#         x = self.node_bn(node_out + x)
#         graph_emb = self.graph_bn(v + graph_emb)

#         return x, edge_index, edge_attr, batch, graph_emb


# class GraphDecoder(MessagePassing):
#     def __init__(self, **mp_kwargs):
#         super().__init__(
#             aggr=mp_kwargs.pop("aggr", "add"),
#             flow="target_to_source",
#             node_dim=mp_kwargs.pop("node_dim", -2)
#         )
