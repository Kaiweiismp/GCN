import torch.nn as nn
import torch.nn.functional as F
import torch 

from layers import GraphConvolution
from layers import GraphLight


class GCN(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        print(nfeat)
        print(nhid)
        print(nclass)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    

class LightGCN(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout):
        super(LightGCN, self).__init__()

        self.embedding = nn.Embedding(nuser, 1024)
        
        self.gc1 = GraphLight(1024, nhid)
        self.gchid = GraphLight(nhid, nhid)
        self.gc2 = GraphLight(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):

        embedding = self.embedding.weight

        x = self.gc1(embedding, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gchid(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)


        return F.log_softmax(x, dim=1)
