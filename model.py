# making R-GNN Network, please refer to tests/ folder where I have done several 
# experiments with the smaller networks and how different parts of it can be
# combined to create the final model
# 26.08.2020 - @yashbonde
# Trying to code Karpathy style

import torch
from torch import nn
from torch_scatter.scatter import scatter_max

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for node_edge <- node + edge_attr
        self.edge_lin = nn.Sequential(
            nn.Linear(int(config.edim * 2), config.edim),
            nn.ReLU()
        )
        self.edge_drop = nn.Dropout(config.drop_p)

        # for node_2 <- node + global + node_edge
        self.node_lin = nn.Sequential(
            nn.Linear(int(config.edim * 3), int(config.edim * 4)),
            nn.ReLU(),
            nn.Linear(int(config.edim * 4), config.edim)
        )
        self.node_drop = nn.Dropout(config.drop_p)

        # for global <- node_2 + global
        self.glob_lin = nn.Sequential(
            nn.Linear(int(config.edim * 2), config.edim),
            nn.ReLU()
        )
        self.glob_drop = nn.Dropout(config.drop_p)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = self.edge_lin(torch.cat((x[row], edge_attr), dim = -1))
        out = self.edge_drop(scatter_max(out, col, dim = 0, dim_size = x.size(0)))
        out = self.node_drop(self.node_lin(torch.cat((out, x, u), dim = -1)).sigmoid())
        x = x + out

        x_u = scatter_max(x, batch, dim=0, dim_size=x.size(0))
        out = self.glob_lin(torch.cat((x_u, u), dim = -1))
        u = u + out

        return x, edge_index, edge_attr, u, batch

class GraphTemporalNetwork(nn.Module):
    """
    This is a very weird network that uses both the graph neural network and
    LSTM with residual connections
    """
    def __init__(self, config) -> None:
        super().__init__()
        
    def forward(self, graphs, hidden_state):
        """
        first perform the embeddings and get three things:
        1. node embeddings (x_i)
        2. edge_embeddings (v_ij)
        3. glob_embeddings (u)

        :param graph: Namespace(
            
            edge_index: []
        )
        """
        
        
    
