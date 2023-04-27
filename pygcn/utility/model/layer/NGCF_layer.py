import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphNGCF(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphNGCF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        initializer = nn.init.xavier_uniform_
        
        self.weight_gc_W = nn.Parameter(initializer(torch.empty(in_features, out_features)))
        self.weight_gc_b = Parameter(initializer(torch.empty(1, out_features)))

        self.weight_bi_W = Parameter(initializer(torch.empty(in_features, out_features)))
        self.weight_bi_b = Parameter(initializer(torch.empty(1, out_features)))



    def forward(self, adj, input):
        side_embeddings = torch.mm(adj, input)
        sum_embeddings = torch.matmul(side_embeddings, self.weight_gc_W) + self.weight_gc_b

        bi_embeddings = torch.mm(adj, side_embeddings)
        bi_embeddings = torch.matmul(bi_embeddings, self.weight_bi_W) + self.weight_bi_b

        output = sum_embeddings + bi_embeddings
        
        return output
    
class GraphNewNGCF(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphNewNGCF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        initializer = nn.init.xavier_uniform_
        
        self.weight_gc_W = nn.Parameter(initializer(torch.empty(in_features, out_features)))
        self.weight_gc_b = Parameter(initializer(torch.empty(1, out_features)))

        self.weight_bi_W = Parameter(initializer(torch.empty(in_features, out_features)))
        self.weight_bi_b = Parameter(initializer(torch.empty(1, out_features)))


    def forward(self, adj, input, New_adj, New_input):
        side_embeddings = torch.mm(adj, input)
        sum_embeddings = torch.matmul(side_embeddings, self.weight_gc_W) + self.weight_gc_b

        bi_embeddings = torch.mm(adj, side_embeddings)
        bi_embeddings = torch.matmul(bi_embeddings, self.weight_bi_W) + self.weight_bi_b

        output = sum_embeddings + bi_embeddings

        New_side_embeddings = torch.mm(New_adj, New_input)
        New_sum_embeddings = torch.matmul(New_side_embeddings, self.weight_gc_W) + self.weight_gc_b

        New_bi_embeddings = torch.mm(New_adj, New_side_embeddings)
        New_bi_embeddings = torch.matmul(New_bi_embeddings, self.weight_bi_W) + self.weight_bi_b

        New_output = New_sum_embeddings + New_bi_embeddings
        


        return output, New_output