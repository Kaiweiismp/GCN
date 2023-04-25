import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphNGCF(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphNGCF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_gc_W = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_gc_b = Parameter(torch.FloatTensor(1, out_features))

        self.weight_bi_W = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_bi_b = Parameter(torch.FloatTensor(1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_gc_W.size(1))
        self.weight_gc_W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_gc_b.size(1))
        self.weight_gc_b.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_bi_W.size(1))
        self.weight_bi_W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_bi_b.size(1))
        self.weight_bi_b.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        side_embeddings = torch.mm(adj, input)
        sum_embeddings = torch.matmul(side_embeddings, self.weight_gc_W) + self.weight_gc_b

        bi_embeddings = torch.mm(adj, side_embeddings)
        bi_embeddings = torch.matmul(bi_embeddings, self.weight_bi_W) + self.weight_bi_b

        output = sum_embeddings + bi_embeddings
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphLightGCN(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphLightGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        output = torch.mm(adj, input)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphGAT(Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, bias=False):
        super(GraphGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(2*out_features, 1))
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):

        h = torch.mm(input, self.weight)   # [N, out_features]
        N = h.size()[0]
        print("N ==== ", N)

        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'