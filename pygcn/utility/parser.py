'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--model_type', nargs='?', default='GAT',
                            help='Specify the name of model.')
    parser.add_argument('--path', nargs='?', default='../data/',
                            help='Specify the path of dataset.')                        
    parser.add_argument('--dataset', nargs='?', default='personality2018',
                            help='Specify the dataset.') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, 
                        help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')


    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='?@K')    

    parser.add_argument('--KNN', type=int, default=3,
                        help='Number of neighbor that decide')

    return parser.parse_args()
