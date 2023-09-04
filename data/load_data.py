
import torch
from torch_geometric.data import Data
import numpy as np
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os.path
from pathlib import Path
from random import randint
import random
import math
from torch_geometric.utils import to_undirected



def load_data_by_args(root,args):  # read

    # check data
    for id, dataset in enumerate(args.train_dataset):
        file = Path(root+'original/{}'.format(dataset))
        if not file.exists():file.mkdir()
        file = Path(root + 'process/{}'.format(dataset))
        if not file.exists(): file.mkdir()
        for i in args.train_graph_ids[id]:
            if not Path(root+'process/{}/split{}.npz'.format(dataset, i)).exists():
                mat = Path(root + 'original/{}/graph{}.npz'.format(dataset, i))
                feats = Path(root + 'original/{}/feats{}.npy'.format(dataset, i))
                if not mat.exists() or not feats.exists():
                    mat_addr = './dataset/{}/adj.npz'.format(dataset)
                    feats_addr = './dataset/{}/feats.npy'.format(dataset)
                    obtain_dataset(root, dataset, mat_addr, feats_addr, i)
                #creat_split(root, dataset, i, args.meta_train_edge_ratio, args.meta_val_edge_ratio)
                creat_split(root,dataset,i,0.2,0.2)
    train_data = []

    for id,dataset in enumerate( args.train_dataset):
        for i in args.train_graph_ids[id]:
            feats = np.load(root+ 'original/{}/feats{}.npy'.format(dataset,i) )
            graph = sp.load_npz(root+ 'original/{}/graph{}.npz'.format(dataset,i))

            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

            edge_index = torch.from_numpy(
                np.vstack((graph.row, graph.col)).astype(np.long))
            x = torch.from_numpy(feats).to(torch.float)


            data = Data(edge_index=edge_index, x=x)
            data.name = dataset
            data.id = i

            split = np.load(root + 'process/{}/split{}.npz'.format(dataset, i))
            data.train_pos_edge_index = torch.LongTensor(split['train'])
            data.val_pos_edge_index = torch.LongTensor(split['val'])
            data.test_pos_edge_index = torch.LongTensor(split['test'])
            data.test_neg_edge_index = torch.LongTensor(split['test_neg'])

            train_data.append(data)


    val_data = []

    for id,dataset in enumerate( args.test_dataset):
        for i in args.val_graph_ids[id]:
            feats = np.load(root + '{}/feats{}.npy'.format(dataset, i))
            graph = sp.load_npz(root + '{}/graph{}.npz'.format(dataset, i))

            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

            edge_index = torch.from_numpy(
                np.vstack((graph.row, graph.col)).astype(np.long))
            x = torch.from_numpy(feats).to(torch.float)

            print('val', x.shape[0],x.shape[1], edge_index.shape[1])

            data = Data(edge_index=edge_index, x=x)
            data.name = dataset
            data.id = i

            split = np.load(root + 'process/{}/split{}.npz'.format(dataset, i))
            data.train_pos_edge_index = torch.LongTensor(split['train'])
            data.val_pos_edge_index = torch.LongTensor(split['val'])
            data.test_pos_edge_index = torch.LongTensor(split['test'])
            data.test_neg_edge_index = torch.LongTensor(split['test_neg'])

            val_data.append(data)




    print('test_data')
    test_data = []

    for id,dataset in enumerate( args.test_dataset):
        for i in args.test_graph_ids[id]:
            feats = np.load(root + 'original/{}/feats{}.npy'.format(dataset, i))
            graph = sp.load_npz(root + 'original/{}/graph{}.npz'.format(dataset, i))

            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

            edge_index = torch.from_numpy(
                np.vstack((graph.row, graph.col)).astype(np.long))
            x = torch.from_numpy(feats).to(torch.float)

            print('val', x.shape[0],x.shape[1], edge_index.shape[1])

            data = Data(edge_index=edge_index, x=x)
            data.name = dataset
            data.id = i

            split = np.load(root + 'process/{}/split{}.npz'.format(dataset, i))
            data.train_pos_edge_index = torch.LongTensor(split['train'])
            data.val_pos_edge_index = torch.LongTensor(split['val'])
            data.test_pos_edge_index = torch.LongTensor(split['test'])
            data.test_neg_edge_index = torch.LongTensor(split['test_neg'])

            test_data.append(data)



    return train_data,val_data,test_data








def sampling(mat,  n_samples=2000):
    g_vertices = list(range(mat.shape[0]))

    sample = set()
    n_iter = 10 * n_samples

    num_vertices = len(g_vertices)

    current = g_vertices[randint(0, num_vertices)]
    sample.add(current)
    count = 0

    while len(sample)<n_samples:
        count+=1
        if count>n_iter:return 0
        neighbors = mat[current,:].nonzero()[1]
        if len(neighbors) == 0:
                continue
        current = random.choice(neighbors)
        sample.add(current)

    sample = sorted(sample)
    adj = mat[sample,:][:,sample]
    for i in range(len(sample)):
        adj[i,i]=0
    return sample,adj.tocoo()


def obtain_dataset(root, name,mat_addr,feats_addr,index):

    assert(Path(mat_addr).exists())
    assert(Path(feats_addr).exists())
    mat = sp.load_npz(mat_addr)
    feats = np.load(feats_addr,mmap_mode = 'r')

    while True:
        submat = sampling(mat, n_samples=2000)
        if submat == 0:
            continue
        else:
            g = submat[1]
            f = feats[submat[0]]
            np.save(root+'original/{}/feats{}.npy'.format(name, index), f)
            sp.save_npz(root+'original/{}/graph{}.npz'.format(name, index), g)
            index += 1
            print(index)



def split_edges(data, val_ratio=0.2, test_ratio=0.6):


    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)
    train_pos_edge_index = to_undirected(train_pos_edge_index)

    train_pos_edge_index = train_pos_edge_index.detach().numpy()
    val_pos_edge_index = val_pos_edge_index.detach().numpy()
    test_pos_edge_index = test_pos_edge_index.detach().numpy()

    num_nodes = data.x.shape[0]
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.tensor(random.sample(range(neg_row.size(0)), n_t))
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

    return train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index




def creat_split(root,dataset,index,meta_train_edge_ratio,meta_val_edge_ratio):
    feats = np.load(root + 'original/{}/feats{}.npy'.format(dataset, index))
    graph = sp.load_npz(root + 'original/{}/graph{}.npz'.format(dataset, index))

    edge_index = torch.from_numpy(
        np.vstack((graph.row, graph.col)).astype(np.long))
    x = torch.from_numpy(feats).to(torch.float)

    print('train', x.shape[0], edge_index.shape[1])

    data = Data(edge_index=edge_index, x=x)
    data.name = dataset
    data.id = index
    data = T.NormalizeFeatures()(data)
    train, val, test, test_neg = split_edges(data, meta_train_edge_ratio, 1-meta_val_edge_ratio-meta_train_edge_ratio)

    np.savez(root+'process/{}/split{}.npz'.format(dataset, index), train=train, val=val, test=test,
             test_neg=test_neg)





