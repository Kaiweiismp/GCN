from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utility.batch_test import *

from utils import load_data, load_data_recommendation, accuracy

#from models import GCN
import models


args.cuda = not args.no_cuda and torch.cuda.is_available()
print('arge.cuda: ',args.cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if(args.dataset == 'cora'):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.path, args.dataset)
elif(args.dataset == 'personality2018'):
    #adj, features, labels, idx_train, idx_val, idx_test = load_data_recommendation(args.path, args.dataset)
    #adj_mat, exist_users, n_train, n_test, n_users, n_items, train_items, test_set = load_data_recommendation(args.path, args.dataset)
    plain_adj, norm_adj, mean_adj, plain_adj_personality, norm_adj_personality, mean_adj_personality = data_generator.get_adj_mat()
    



# Model and optimizer
model = models
if(args.model_type == 'GCN'):
    model = models.GCN(nuser=features.shape[0],
                            nfeat=features.shape[1],
                            nhid=args.hidden,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout)
elif(args.model_type == 'NGCF'):
    model = models.NGCF(nuser=features.shape[0],
                            nfeat=features.shape[1],
                            nhid=args.hidden,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout)
elif(args.model_type == 'LightGCN'):
    model = models.LightGCN(nuser=features.shape[0],
                            nfeat=features.shape[1],
                            nhid=args.hidden,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout)
elif(args.model_type == 'GAT'):
    model = models.GAT(nuser=features.shape[0],
                            nfeat=features.shape[1], 
                            nhid=args.hidden, 
                            nclass=int(labels.max()) + 1, 
                            dropout=args.dropout, 
                            nheads=args.nb_heads, 
                            alpha=args.alpha)

cur_best_pre_0, stopping_step = 0, 0
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    print('Use GPU')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

print("==============================================")
print(model)

def train(epoch):


    t = time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time() - t_total))

# Testing
test()
