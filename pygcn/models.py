import torch.nn as nn
import torch.nn.functional as F
import torch 

from layers import *



class GCN(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
class NGCF(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout):
        super(NGCF, self).__init__()

        self.embedding = nn.Embedding(nuser, 1024)

        self.gc1 = GraphNGCF(1024, nhid)
        self.gchid = GraphNGCF(nhid, nhid)
        self.gc2 = GraphNGCF(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):

        embedding = self.embedding.weight

        x = F.relu(self.gc1(embedding, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gchid(embedding, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return F.log_softmax(x, dim=1)




class LightGCN(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout):
        super(LightGCN, self).__init__()

        self.embedding = nn.Embedding(nuser, 1024)
        
        self.gc1 = GraphLightGCN(1024, nhid)
        self.gchid = GraphLightGCN(nhid, nhid)
        self.gc2 = GraphLightGCN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):

        embedding = self.embedding.weight

        x = self.gc1(embedding, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gchid(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)


        return F.log_softmax(x, dim=1)



class GAT(nn.Module):
    def __init__(self, nuser, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()

        self.embedding = nn.Embedding(nuser, 512)

        self.dropout = dropout

        self.attentions = [GraphGAT(512, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphGAT(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):

        embedding = self.embedding.weight
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(embedding, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)