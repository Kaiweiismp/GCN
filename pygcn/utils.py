import numpy as np
import scipy.sparse as sp
import torch
import random as rd


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path, dataset):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}/content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}/cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_recommendation(path, dataset):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    train_file = path + dataset + '/train.txt'
    test_file = path + dataset + '/test.txt'
    personality_file = path + dataset + '/personality.txt'


    #get number of users and items
    n_users, n_items = 0, 0
    n_train, n_test = 0, 0

    exist_users = []

    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                exist_users.append(uid)
                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
                n_train += len(items)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                n_items = max(n_items, max(items))
                n_test += len(items)
    n_items += 1
    n_users += 1    

    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)

    train_items, test_set = {}, {}
    with open(train_file) as f_train:
        with open(test_file) as f_test:
            for l in f_train.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_item = items[0], items[1:]
                for i in train_item:
                    R[uid, i] = 1.
                    # self.R[uid][i] = 1
                train_items[uid] = train_item
            for l in f_test.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                except Exception:
                    continue
                uid, test_item = items[0], items[1:]
                test_set[uid] = test_item



    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    adj_mat = normalize(adj_mat + sp.eye(adj_mat.shape[0]))
    adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)

    print("================")
    print("=====success====")
    print("================")
    return adj_mat, exist_users, n_train, n_test, n_users, n_items, train_items, test_set

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sample(batch_size, exist_users, n_users, n_items, train_items):
        if batch_size <= n_users:
            users = rd.sample(exist_users, batch_size)
        else:
            users = [rd.choice(exist_users) for _ in range(batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=n_items,size=1)[0]
                if neg_id not in train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(neg_pools[u]) - set(train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
