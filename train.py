import torch
import torch.nn.functional as F
import sklearn.neighbors
import copy
from collections import OrderedDict
import numpy as np
from utils import test,test2
from models.models import MLP,WDiscriminator





def train_wdiscriminator_node(embedding_s, embedding_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):

    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True
    wdiscriminator.train()

    for j in range(batch_d_per_iter):

        optimizer_d.zero_grad()
        w0 = wdiscriminator(embedding_s)
        w1 = wdiscriminator(embedding_t)

        l1 = -torch.mean(w1) + torch.mean(w0)
        loss = l1

        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    loss = -torch.mean(w1) + torch.mean(w0)
    print(loss.item())

    return wdiscriminator



def train_wdiscriminator(embedding_s, embedding_t, wdiscriminator, source_node_num,
                         target_node_num, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True
    wdiscriminator.train()

    for j in range(batch_d_per_iter):

        optimizer_d.zero_grad()
        w0 = wdiscriminator(embedding_s[:source_node_num])
        w1 = wdiscriminator(embedding_t[:target_node_num])

        w2 = wdiscriminator(embedding_s[source_node_num:])
        w3 = wdiscriminator(embedding_t[target_node_num:])


        l1 = -torch.mean(w1) + torch.mean(w0)
        l2 = -torch.mean(w3) + torch.mean(w2)
        
        loss = l1+l2

        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)



    loss = -torch.mean(w1) + torch.mean(w0) - torch.mean(w3) + torch.mean(w2)
    print(loss.item())

    return wdiscriminator





def cal_gradient_penalty(net, device, real, fake):
    """
    WGAN-GP:gradient penalty
    """

    num = min(real.shape[0],fake.shape[0])
    alpha = torch.rand(num, 1).to(device)
    interpolates = alpha * real[:num] + (1 - alpha) * fake[:num]
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = net(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]
    gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty



def train_wdiscriminator_GP(embedding_s, embedding_t, wdiscriminator, source_node_num,
                            target_node_num, optimizer_d, device = 'cpu', batch_d_per_iter=10):
    wdiscriminator.train()
    for j in range(batch_d_per_iter):
        optimizer_d.zero_grad()
        w0 = wdiscriminator(embedding_s[:source_node_num])
        w1 = wdiscriminator(embedding_t[:target_node_num])

        w2 = wdiscriminator(embedding_s[source_node_num:])
        w3 = wdiscriminator(embedding_t[target_node_num:])

        gradient_penalty = cal_gradient_penalty(wdiscriminator, device, embedding_t, embedding_s)

        temp = -torch.mean(w1) + torch.mean(w0) -torch.mean(w3) + torch.mean(w2)  + 0.1*gradient_penalty
        if temp.item()<0:
            loss = -torch.mean(w1) + torch.mean(w0) -torch.mean(w3) + torch.mean(w2)  + 0.1*gradient_penalty
        else: loss = -(-torch.mean(w1) + torch.mean(w0) -torch.mean(w3) + torch.mean(w2)  + 0.1*gradient_penalty)

        loss.backward()
        optimizer_d.step()

    print(loss.item())


    return wdiscriminator




def construct_space(model,edges_num,train_pos_edge_index,graph_x):

    z = model(graph_x)
    node_embs = torch.hstack((z, z))
    if train_pos_edge_index.shape[1] <= edges_num:
        select_edges = train_pos_edge_index
    else:
        perm = np.random.choice(train_pos_edge_index.shape[1], edges_num, replace=False)
        select_edges = train_pos_edge_index[:, perm]
    edge_embs = torch.hstack((z[select_edges[0]], z[select_edges[1]]))
    source_space = torch.vstack((node_embs, edge_embs))
    return source_space



def construct_space2(model,nodes_num,edges_num,train_pos_edge_index,graph_x):

    z = model(graph_x)
    idx = np.random.choice(z.shape[0], nodes_num, replace=False)

    node_embs = torch.hstack((z[idx], z[idx]))


    if train_pos_edge_index.shape[1] <= edges_num:
        select_edges = train_pos_edge_index
    else:
        perm = np.random.choice(train_pos_edge_index.shape[1], edges_num, replace=False)
        select_edges = train_pos_edge_index[:, perm]

    edge_embs = torch.hstack((z[select_edges[0]], z[select_edges[1]]))
    source_space = torch.vstack((node_embs, edge_embs))
    return source_space



    
    





def space_align_node(mean_space,graph,args,device):

    target_space = torch.FloatTensor(mean_space).to(device)
    model = MLP(graph.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index,graph_x = graph_to_device(graph,device)

    wdiscriminator = WDiscriminator(args.share_dims,args.share_dims*2).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)


    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)

    while True:
        source_space = model(graph_x).detach()
        wdiscriminator = train_wdiscriminator_node(source_space, target_space, wdiscriminator, optimizer_wd, batch_d_per_iter=80)

        for p in wdiscriminator.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        source_space = model(graph_x)

        model.train()
        optimizer.zero_grad()
        w0 = wdiscriminator(source_space)
        w1 = wdiscriminator(target_space)

        loss = torch.mean(w1) - torch.mean(w0) 
        print((torch.mean(w1) - torch.mean(w0) ).item())
        value = loss.item()
        print(value)

        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1

        print(cnt)
        loss.backward()
        optimizer.step()



def space_align(mean_space, embs_num, graph, args, device):
    edges_num = embs_num*2
    target_space = torch.FloatTensor(mean_space).to(device)
    model = MLP(graph.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index,graph_x = graph_to_device(graph,device)


    wdiscriminator = WDiscriminator(args.share_dims*2,512).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        source_space = construct_space(model,edges_num,train_pos_edge_index,graph_x).detach()

        wdiscriminator = train_wdiscriminator(source_space, target_space, wdiscriminator, graph_x.shape[0],
                                              embs_num, optimizer_wd, batch_d_per_iter=80)

        for p in wdiscriminator.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        source_space = construct_space(model, edges_num, train_pos_edge_index,graph_x)
        model.train()
        optimizer.zero_grad()
        w0 = wdiscriminator(source_space[:graph_x.shape[0]])
        w1 = wdiscriminator(target_space[:embs_num])

        w2 = wdiscriminator(source_space[graph_x.shape[0]:])
        w3 = wdiscriminator(target_space[embs_num:])

        loss = torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2)
        print((torch.mean(w1) - torch.mean(w0) ).item(),(torch.mean(w3) - torch.mean(w2)).item())
        value = loss.item()
        print(value)

        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)
        loss.backward()
        optimizer.step()









def space_align2(mean_space, embs_num, edges_num, graph, args, device):

    target_space = torch.FloatTensor(mean_space).to(device)
    model = MLP(graph.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index,graph_x = graph_to_device(graph,device)


    wdiscriminator = WDiscriminator(args.share_dims*2,512).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        source_space = construct_space2(model,embs_num,edges_num,train_pos_edge_index,graph_x).detach()

        wdiscriminator = train_wdiscriminator(source_space, target_space, wdiscriminator, embs_num,
                                              embs_num, optimizer_wd, batch_d_per_iter=80)

        for p in wdiscriminator.parameters(): p.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        source_space = construct_space2(model,embs_num, edges_num, train_pos_edge_index,graph_x)
        model.train()
        optimizer.zero_grad()
        w0 = wdiscriminator(source_space[:embs_num])
        w1 = wdiscriminator(target_space[:embs_num])

        w2 = wdiscriminator(source_space[embs_num:])
        w3 = wdiscriminator(target_space[embs_num:])

        loss = torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2)
        print((torch.mean(w1) - torch.mean(w0) ).item(),(torch.mean(w3) - torch.mean(w2)).item())
        value = loss.item()
        print(value)

        
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)
        if cnt >0:return best_model
        loss.backward()
        optimizer.step()



















def train_graph(mlp,model, graph,optimizer,device,file=None):
    graph.train_mask = graph.val_mask = graph.test_mask = graph.y = None
    graph.batch = None
    num_nodes = graph.num_nodes

    train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index,graph_x = graph_to_device(graph,device)
    mlp.to(device)
    model.to(device)
    model.train()
    value = 0
    count = 0

    for epoch in range(1,30000):
        if count >= 400: break
        if graph.name =='reddit' or graph.name =='academic':
            if epoch ==50:
                for p in mlp.parameters():
                    p.data.clamp_(-0.05, 0.05)
                mlp.Dropout.p = 0.8
        if graph.name =='product' or graph.name =='yelp':
            if epoch ==50:
                mlp.Dropout.p = 0.8
        model.train()
        mlp.train()
        optimizer.zero_grad()
        weights = OrderedDict(model.named_parameters())
        z = model.encode(mlp(graph_x), train_pos_edge_index, weights, inner_loop=True)
        loss = model.recon_loss(z, train_pos_edge_index)
        kl_loss = model.kl_loss() * (1 / num_nodes)
        loss = loss + kl_loss

        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)
        model.eval()
        mlp.eval()
        model.zero_grad()
        mlp.zero_grad()
        weights = OrderedDict(model.named_parameters())
        auc, ap = test(model, mlp(graph_x).detach(), train_pos_edge_index,
                       test_pos_edge_index, test_neg_edge_index, weights)
        if epoch<=50:print(epoch,auc,ap,loss)
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,auc,ap))
        if value<auc:
            value = auc
            count = 0
        else:count+=1
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return model












def graph_to_device(data_graph,device):

    return  data_graph.train_pos_edge_index.to(device),data_graph.val_pos_edge_index.to(device), data_graph.test_pos_edge_index.to(device), data_graph.test_neg_edge_index.to(device), data_graph.x.to(device) 

   




def md_gram_gradient_step(args, meta_model, wdiscriminator, pools, data_batch, optimizer_wd, optimizer_all, device):
    task_losses = []
    batch_names = []
    torch.autograd.set_detect_anomaly(True)

    for idx, data_graph in enumerate(data_batch):

        name = data_graph.name + str(data_graph.id)
        batch_names.append(name)
        train_pos_edge_index,val_pos_edge_index,test_pos_edge_index,test_neg_edge_index,graph_x = graph_to_device(data_graph,device)
        pool_model = pools.pool[name].to(device)


        num_nodes = data_graph.num_nodes

        mean_space = torch.FloatTensor(pools.mean_space).to(device)
        embeddings = construct_space(pool_model,pools.edges_num,train_pos_edge_index,graph_x).detach()



        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(embeddings, mean_space, wdiscriminator, graph_x.shape[0],
                                                                 pools.embs_num, optimizer_wd, batch_d_per_iter=60))
        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        fast_weights = OrderedDict(meta_model.named_parameters())


        x = torch.FloatTensor(pools.embeddings[name]).to(device)

        for inner_batch in range(args.inner_train_steps):
            # Perform update of model weights

            z = meta_model.encode(x, train_pos_edge_index, fast_weights, inner_loop=True)
            loss = meta_model.recon_loss(z, train_pos_edge_index)


            if args.model in ['VGAE']:
                kl_loss = meta_model.kl_loss() * (1 / num_nodes)


                loss = loss + kl_loss

                gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                                allow_unused=True, create_graph=True)
                gradients = [0 if grad is None else grad for grad in gradients]

            # Update weights manually
            fast_weights = OrderedDict(
                (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )



        z= pool_model(graph_x)


        z_val = meta_model.encode(z, val_pos_edge_index, fast_weights, inner_loop=False)
        loss_val = meta_model.recon_loss(z_val, val_pos_edge_index)
        if args.model in ['VGAE']:
            kl_loss = meta_model.kl_loss() * (1 / num_nodes)
            loss_val = (loss_val + kl_loss)


        source_space = construct_space(pool_model,pools.edges_num,train_pos_edge_index,graph_x)

        w0 = wdiscriminator_copy(source_space[:graph_x.shape[0]])
        w1 = wdiscriminator_copy(mean_space[:pools.embs_num])

        w2 = wdiscriminator_copy(source_space[graph_x.shape[0]:])
        w3 = wdiscriminator_copy(mean_space[pools.embs_num:])

        Dis_loss = torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2)
        print("valid Dis_loss: %f" % (Dis_loss.item()))
        loss_all =loss_val+args.weight * Dis_loss


        auc, ap = test(meta_model, pool_model(graph_x).detach(), train_pos_edge_index,
                test_pos_edge_index, test_neg_edge_index,fast_weights)
        print(auc,ap,'train result')

        task_losses.append(loss_all)




    if len(task_losses) != 0:
        meta_model.train()
        optimizer_all.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_loss.backward()
        optimizer_all.step()

    for p in meta_model.parameters():
        p.data.clamp_(-0.1, 0.1)
    '''
    for batch in data_batch:
        name = batch.name + str(batch.id)
        print( pools.pool[name].fc1.weight.grad.device)
        pools.pool[name].cpu()
    '''

    return meta_model,meta_batch_loss.item()









