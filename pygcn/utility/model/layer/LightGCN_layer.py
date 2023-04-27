import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphLightGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphLightGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, adj, input):
        output = torch.mm(adj, input)
        
        return output
    
class GraphNewLightGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphNewLightGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        
    def forward(self, adj, input, New_adj, New_input):
        output = torch.mm(adj, input)

        New_output = torch.mm(New_adj, New_input)

        return output, New_output