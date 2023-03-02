
from copy import deepcopy
import numpy as np
from sklearn.utils import shuffle
import torch
import itertools
seed = 0
torch.manual_seed(seed)
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader
import torch_geometric
import time
import argparse

from pool import *
from models.models import *
from data.load_data import load_data_by_args
from models.autoencoder import MyGAE, MyVGAE
from train import *


parser = argparse.ArgumentParser()


parser.add_argument('--train_dataset', type=list, default=['yelp','academic','reddit' ])
parser.add_argument('--train_graph_ids', type=list, default=[[i for i in range(5)]]*3)
parser.add_argument('--test_dataset', type=list, default=['product'])
parser.add_argument('--val_graph_ids', type=list, default=[[i for i in range(10,15)]])
parser.add_argument('--test_graph_ids', type=list, default=[[i for i in range(0,5)]])
parser.add_argument('--share_dims', type=int, default=256)
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--meta_lr', type=float, default=0.005)
parser.add_argument('--wd_lr', type=float, default=0.005)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--weight', type=float, default=1)
parser.add_argument('--edge_num', type=int, default=4000)
parser.add_argument('--node_num', type=int, default=2000)
parser.add_argument('--model', type=str, default='VGAE')
parser.add_argument('--gating', type=str, default='signature_cond', choices=[None, 'signature', 'weights', 'signature_cond', 'weights_cond'])
parser.add_argument('--num_gated_layers', default=4, type=int)
parser.add_argument('--output_dims', default=16, type=int)
parser.add_argument('--train_batch', default=15, type=int)
parser.add_argument('--use_gcn_sig', default=True, type=bool)
parser.add_argument('--meta_val_edge_ratio', type=float, default=0.2)
parser.add_argument('--meta_train_edge_ratio', type=float, default=0.2)
parser.add_argument('--inner_train_steps', default=10, type=int)
parser.add_argument('--layer_norm', default=False, action='store_true',help='use layer norm')
parser.add_argument('--cuda', type = int,default=-1)










def test_md_gram(args, device):
    dataset = args.test_dataset[0]
    root = './meta_graph/'
    train_data, val_data, test_data = load_data_by_args(root, args)

    file = open('./model/test_for_{}_all_pre.csv'.format(dataset), 'w')
    print('mdgram_all_with_preprocess')
    for graph in test_data:
        name = graph.name + str(graph.id)
        model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
        model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
        mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
        mlp = space_align(mean_space, args.node_num, graph, args, device)
        mlp.cpu()
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.001, weight_decay=5e-4)
        file.write(name+'\n')
        train_graph(mlp,model, graph,optimizer,device,file)
        file.write('\n\n\n')
        file.flush()






def test_md_gram_all(args, device):
    dataset = args.test_dataset[0]
    root = './meta_graph/'
    train_data, val_data, test_data = load_data_by_args(root, args)

    file = open('./model/test_for_{}_all_pre.csv'.format(dataset), 'w')
    print('mdgram_all_with_preprocess')
    for graph in test_data:
        name = graph.name + str(graph.id)
        model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
        model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
        mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
        mlp = space_align(mean_space, args.node_num, graph, args, device)
        mlp.cpu()
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.001, weight_decay=5e-4)
        file.write(name+'\n')
        train_graph(mlp,model, graph,optimizer,device,file)
        file.write('\n\n\n')
        file.flush()

    file = open('./model/test_for_{}_mlp_pre.csv'.format(dataset), 'w')
    print('mdgram_mlp_with_preprocess')
    for graph in test_data:
        name = graph.name + str(graph.id)
        model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
        model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
        mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
        mlp = space_align(mean_space, args.node_num, graph, args, device)
        mlp.cpu()
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.00, weight_decay=5e-4)
        file.write(name+'\n')
        train_graph(mlp,model, graph,optimizer,device,file)
        file.write('\n\n\n')
        file.flush()

    file = open('./model/test_for_{}_all.csv'.format(dataset), 'w')
    print('mdgram_all_without_preprocess')
    for graph in test_data:
        name = graph.name + str(graph.id)
        model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
        model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
        mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
        mlp = MLP(graph.x.shape[1],args.hidden_dims,args.share_dims)
        mlp.reset_parameters()
        mlp.cpu()
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.001, weight_decay=5e-4)
        file.write(name+'\n')
        train_graph(mlp,model, graph,optimizer,device,file)
        file.write('\n\n\n')
        file.flush()

    file = open('./model/test_for_{}_mlp.csv'.format(dataset), 'w')
    print('mdgram_mlp_without_preprocess')
    for graph in test_data:
        name = graph.name + str(graph.id)
        model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
        model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
        mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
        mlp = MLP(graph.x.shape[1],args.hidden_dims,args.share_dims)
        mlp.reset_parameters()
        mlp.cpu()
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.00, weight_decay=5e-4)
        file.write(name+'\n')
        train_graph(mlp,model, graph,optimizer,device,file)
        file.write('\n\n\n')
        file.flush()





def md_gram(args, device):

    dataset = args.test_dataset[0]


    root = './meta_graph/'

    train_data, val_data, test_data = load_data_by_args(root, args)
    train_loader = DataListLoader(train_data, batch_size=args.train_batch, shuffle=True)

    MLP_pools = MLP_pool(args.hidden_dims, args.share_dims)
    MLP_pools.add_all_MLP(train_data,device)


    wdiscriminator = WDiscriminator(2*args.share_dims)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)


    wdiscriminator.to(device)

    params = []
    for value in MLP_pools.pool.values():
        params.extend(value.parameters())
    meta_model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims))
    meta_model.to(device)

    optimizer_all = torch.optim.Adam([{'params':params,'lr':0.0001},{'params':meta_model.parameters()}], lr=args.meta_lr,
                                     weight_decay=5e-4)
    epoch = 0
    loss_value = 100
    cnt = 0
    optimal_model = meta_model
    optimal_space = 0
    last_model = meta_model

    MLP_pools.update_mean_space()
    while True:
        
        for data in train_loader:
            print(epoch,'epoch')
            MLP_pools.update_embeddings(device)
            MLP_pools.update_mean_space()
            meta_model,loss = md_gram_gradient_step(args, meta_model, wdiscriminator, MLP_pools, data, optimizer_wd,
                                                    optimizer_all, device, epoch)
        if loss<loss_value:
            optimal_model = last_model
            optimal_space = deepcopy(MLP_pools.mean_space)
            loss_value = loss
            cnt=0
        else:cnt+=1

        if cnt == 50 :
            torch.save(optimal_model.state_dict(),'./model/meta_model_for_{}.pth'.format(dataset))
            np.save('./model/meta_space_for_{}.npy'.format(dataset),optimal_space)
            break
            
        last_model = deepcopy(meta_model)

        epoch+=1







def test_graph(args, device,index):
    dataset = args.test_dataset[0]
    root = './meta_graph/'
    train_data, val_data, test_data = load_data_by_args(root, args)
    graph = test_data[index]

    model = MyVGAE(MetaGatedSignatureEncoder(args, args.share_dims, args.output_dims)) 

    model.load_state_dict(torch.load('./model/meta_model_for_{}.pth'.format(dataset)), strict=True)
    mean_space = np.load('./model/meta_space_for_{}.npy'.format(dataset))
    mlp = space_align(mean_space, args.node_num, graph, args, device)
    mlp.cpu()
    optimizer = torch.optim.Adam( [{'params':mlp.parameters(),'lr':0.001},{'params':model.parameters()}],lr=0.001, weight_decay=5e-4)
    train_graph(mlp,model, graph,optimizer,device)




args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")





datasets = ['product', 'academic', 'reddit', 'yelp']
for data in ['product','academic','reddit','yelp' ]:
    args.test_dataset[0] = data
    news = deepcopy(datasets)
    news.remove(data)
    args.train_dataset = news
    md_gram(args, device) # train
    test_md_gram(args, device) # test




#pip3 install https://data.pyg.org/whl/torch-1.10.0%2Bcu102/torch_spline_conv-1.2.1-cp36-cp36m-linux_x86_64.whl




