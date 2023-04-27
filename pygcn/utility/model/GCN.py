
import torch
import torch.nn as nn
import torch.nn.functional as F


from utility.model.layer.GCN_layer import GraphGCN as layers

class GCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, norm_adj_personality, args):
        super(GCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size

        #self.in_features = args.in_features
        #self.nhid = args.nhid
        #self.out_features = args.out_features

        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.norm_adj = norm_adj
#        self.norm_adj_personality = norm_adj_personality

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        

        """
        *********************************************************
        Init the weight of user-item.
        """
        #self.embedding_dict, self.weight_dict = self.init_weight()
        self.embedding_dict = self.init_weight()
        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
#        self.sparse_norm_adj_personality = self._convert_sp_mat_to_sp_tensor(self.norm_adj_personality).to(self.device)

        self.gc1 = layers(self.emb_size, self.emb_size)
        self.gchid = layers(self.emb_size, self.emb_size)
        self.gc2 = layers(self.emb_size, self.emb_size)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
#            'New_user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
#                                                 self.emb_size))),
#            'New_item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
#                                                 self.emb_size)))                                                                     
            })

        
        # 每層 layers 的 output size(包含初始 embeddings)
        layers = [self.emb_size] + self.layers

        return embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]


#        A_hat_personality = self.sparse_dropout(self.sparse_norm_adj_personality,
#                                    self.node_dropout,
#                                    self.sparse_norm_adj_personality._nnz()) if drop_flag else self.sparse_norm_adj_personality
#
#        ego_personality_embeddings = torch.cat([self.embedding_dict['New_user_emb'],
#                                    self.embedding_dict['New_item_emb']], 0)
#        
#        all_personality_embeddings = [ego_personality_embeddings]


        x = self.gc1(A_hat, ego_embeddings)
        x = F.relu(x)
#        New_x = F.relu(New_x)
        all_embeddings += [x]
#        all_personality_embeddings += [New_x]

        for k in range(1, len(self.layers)-1):
            x = self.gchid(A_hat, x)
            x = F.relu(x)
#            New_x = F.relu(New_x)
            all_embeddings += [x]
#            all_personality_embeddings += [New_x]

        x = self.gc2(A_hat, x)
        all_embeddings += [x]
#        all_personality_embeddings += [New_x]

        all_embeddings = torch.cat(all_embeddings, 1)
#        all_personality_embeddings = torch.cat(all_personality_embeddings, 1)

        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

#        u_g_personality_embeddings = all_personality_embeddings[:self.n_user, :]
#        i_g_personality_embeddings = all_personality_embeddings[self.n_user:, :]


#        u_embeddings = u_g_embeddings + u_g_personality_embeddings
#        i_embeddings = i_g_embeddings + i_g_personality_embeddings

        u_embeddings = u_g_embeddings
        i_embeddings = i_g_embeddings

        #print("u_embeddings : ", u_embeddings.size())
        #print("u_g_embeddings : ", u_g_embeddings.size())
        #print("u_g_personality_embeddings : ", u_g_personality_embeddings.size())

        """
        *********************************************************
        look up.
        """
        u_embeddings = u_embeddings[users, :]
        pos_i_embeddings = i_embeddings[pos_items, :]
        neg_i_embeddings = i_embeddings[neg_items, :]

        return u_embeddings, pos_i_embeddings, neg_i_embeddings