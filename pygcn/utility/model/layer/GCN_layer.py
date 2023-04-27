import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        initializer = nn.init.xavier_uniform_
        
        self.weight = nn.Parameter(initializer(torch.empty(in_features, out_features)))


    def forward(self, adj, input):
        side_embeddings = torch.mm(adj, input)
        output = torch.matmul(side_embeddings, self.weight)
        
        return output
    
class GraphNewGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphNewGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        initializer = nn.init.xavier_uniform_
        
        self.weight = nn.Parameter(initializer(torch.empty(in_features, out_features)))
        

    def forward(self, adj, input, New_adj, New_input):
        side_embeddings = torch.mm(adj, input)
        output = torch.matmul(side_embeddings, self.weight)

        New_side_embeddings = torch.mm(New_adj, New_input)
        New_output = torch.matmul(New_side_embeddings, self.weight)

        return output, New_output