# making R-GNN Network, please refer to tests/ folder where I have done several 
# experiments with the smaller networks and how different parts of it can be
# combined to create the final model

import torch
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphTemporalNetwork(nn.Module):
    def __init__(self, config):
        pass

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters())
