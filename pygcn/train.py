from __future__ import division
from __future__ import print_function

#import time
from time import *
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utility.batch_test import *
from utility.helper import *

#from models import GCN
#import models


#from utility.model.NGCF import NGCF as Mymodel
#from utility.model.NewNGCF import NewNGCF as Mymodel
from utility.model.GCN import GCN as Mymodel
#from utility.model.NewGCN import NewGCN as Mymodel
#from utility.model.LightGCN import LightGCN as Mymodel
#from utility.model.NewLightGCN import NewLightGCN as Mymodel


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os.path import join

import warnings
warnings.filterwarnings('ignore')

from tensorboardX import SummaryWriter

BOARD_PATH = "/home/ismp/sda1/kaiwei/pygcn/pygcn/board"
RESULT_PATH = "/home/ismp/sda1/kaiwei/pygcn/pygcn/Record/GCN.txt"
args.device = torch.device('cuda:' + str(args.gpu_id))
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('arge.cuda: ',args.cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


plain_adj, norm_adj, mean_adj, plain_adj_personality, norm_adj_personality, mean_adj_personality = data_generator.get_adj_mat()
    
args.node_dropout = eval(args.node_dropout)
args.mess_dropout = eval(args.mess_dropout)

model = Mymodel(data_generator.n_users,
             data_generator.n_items,
             norm_adj,
             norm_adj_personality,
             args).to(args.device)

# init tensorboard
if args.tensorboard:
    w : SummaryWriter = SummaryWriter(join(BOARD_PATH, strftime("%m-%d-%Hh%Mm%Ss-")))
    print("enable tensorboard")
else:
    w = None
    print("not enable tensorboard")



optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

print("==============================================")
print(model)

def train():
    cur_best_pre_0, stopping_step = 0, 0


    for epoch in range(args.epochs):
        num = epoch+1
        print("===============epoch : %d==========================" % num)
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                        pos_items,
                                                                        neg_items,
                                                                        drop_flag=args.node_dropout_flag)
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                            pos_i_g_embeddings,
                                                                            neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
        if args.tensorboard:
            w.add_scalar(f'BPRLoss/loss', loss, epoch+1)
            w.add_scalar(f'BPRLoss/mf_loss', mf_loss, epoch+1)
            w.add_scalar(f'BPRLoss/emb_loss', emb_loss, epoch+1)
        # *********************************************************
        # 如果 epoch 不是 10 的倍數，則下面都不做 
        if (epoch + 1) % args.round_verbose!= 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch+1, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)

                f = open(RESULT_PATH, 'a')
                f.writelines(perf_str + '\n')
                f.close()
            continue
        # *********************************************************
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret, result = test(model, users_to_test, w, epoch, drop_flag=False)
        t3 = time()
        #f = open(join(RESULT_PATH, str(epoch + 1) + ".txt") , 'w')
        #for re in result:
        #    f.writelines(str(re) + '\n')
        #f.close()        
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs = %.1fs + %.1fs]: train == [%.5f = %.5f + %.5f], recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch+1, t3 - t1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

            f = open(RESULT_PATH, 'a')
            f.writelines(perf_str + '\n')
            f.close()
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break


def final():
        users_to_test = list(data_generator.test_set.keys())
        ret, result = test(model, users_to_test, None, 0, drop_flag=False) 

        perf_str = 'recall=[%.5f, %.5f, %.5f, %.5f, %.5f],\nprecision=[%.5f, %.5f, %.5f, %.5f, %.5f],\nhit=[%.5f, %.5f, %.5f, %.5f, %.5f],\nndcg=[%.5f, %.5f, %.5f, %.5f, %.5f]' % \
                            (ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['precision'][4],
                            ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3], ret['hit_ratio'][4],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4])
        f = open(RESULT_PATH, 'a')
        f.writelines('====================result======================\n')
        f.writelines(perf_str + '\n')
        f.writelines('====================result======================\n')
        f.close()  



# Train model
t_total = time()
train()
final()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time() - t_total))
