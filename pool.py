from os import device_encoding
from models.models import MLP
import torch
import numpy as np
from bc_calculator.barycenters import *


class VAE_pool():
    def __init__(self,hidden_dims,share_dims):
        self.share_dims = share_dims
        self.hidden_dims = hidden_dims
        self.embs_num = 0
        self.pool = {}
        self.mean_space = None
        self.embeddings = {}
        self.inputs = {}


    def add_VAE(self,VAE,inputs,name):
        self.pool[name] = VAE
        self.embeddings[name] = VAE(inputs)[0].detach().cpu().numpy()


    def update_embeddings(self):
        for name in self.pool.keys():
            new_in = torch.FloatTensor(self.inputs[name])
            self.embeddings[name] = self.pool[name](new_in)[0].detach().cpu().numpy()


    def update_mean_space(self):

        self.embs_num = int(np.mean([value.shape[0] for value in self.embeddings.values()]))*2




        VAE_num = len(self.pool)
        temp = np.zeros((self.embs_num,self.share_dims))
        for i in range(self.embs_num):
            for value in self.embeddings.values():
                num = np.random.choice(value.shape[0])
                temp[i]+=value[num]
            temp[i]/=VAE_num
        self.mean_space = temp




    def add_all_VAE(self,datasets):
        nums= len(datasets)
        for i in range(nums):
            name = datasets[i].name
            id = datasets[i].id
            key= name+str(id)
            self.inputs[key] = datasets[i].x.cpu().numpy()
            self.pool[key] = VAE(datasets[i].x.shape[1],self.hidden_dims,self.share_dims)
            self.embeddings[key] = self.pool[key](datasets[i].x )[0].detach().cpu().numpy()







class MLP_pool():
    def __init__(self,hidden_dims,share_dims):
        self.share_dims = share_dims
        self.hidden_dims = hidden_dims
        self.embs_num = 0
        self.edges_num = 0
        self.pool = {}
        self.mean_space = None
        self.mean_edge_space = None
        self.embeddings = {}
        self.inputs = {}
        self.edges = {}





    def update_embeddings(self,device):
        for name in self.pool.keys():            
            new_in = torch.FloatTensor(self.inputs[name]).to(device)

            self.embeddings[name] = self.pool[name](new_in).detach().cpu().numpy()
            

    def update_mean_space(self):
        self.embs_num = int(np.mean([value.shape[0] for value in self.embeddings.values()]))
        self.edges_num = self.embs_num * 2
        print(self.embs_num,self.edges_num)
        temp = np.zeros((self.embs_num+self.edges_num,self.share_dims*2))

        for value in self.embeddings.values():
            temp[:self.embs_num,:self.share_dims]+=value
        temp[:self.embs_num,:self.share_dims]/=len(self.embeddings)
        temp[:self.embs_num,self.share_dims:] = temp[:self.embs_num,:self.share_dims] 


        for i in range(self.edges_num):
            for key in self.embeddings.keys():
                num = i
                if num>=self.edges[key].shape[1]:num = self.edges[key].shape[1]-1
                temp[i+self.embs_num,:self.share_dims]+=self.embeddings[key][self.edges[key][0,num]]
                temp[i+self.embs_num,self.share_dims:]+=self.embeddings[key][self.edges[key][1,num]]
            temp[i+self.embs_num]/=len(self.embeddings)
        self.mean_space = temp



    
    def update_mean_space3(self,embs_num=0,edges_num=0):

        if embs_num!=0:
            self.embs_num = embs_num
            self.edges_num = edges_num

        print(self.embs_num,self.edges_num)
        temp = np.zeros((self.embs_num+self.edges_num,self.share_dims*2))

        for value in self.embeddings.values():
            total = value.shape[0]
            idx =  np.arange(self.embs_num)*(total//self.embs_num)  
            temp[:self.embs_num,:self.share_dims]+=value[idx]
        temp[:self.embs_num,:self.share_dims]/=len(self.embeddings)
        temp[:self.embs_num,self.share_dims:] = temp[:self.embs_num,:self.share_dims] 

        '''
        for key in self.embeddings.keys():
            total = self.edges[key].shape[1]
            if self.edges_num <= total:
                idx =  np.arange(self.edges_num)#*(total//self.edges_num)  
            else:
                block = self.edges_num // total
                rest = self.edges_num % total
                idx =  np.arange(rest)#*(total//rest) 
                idx = np.concatenate((np.tile(np.arange(total),block),idx))

            temp[self.embs_num:,:self.share_dims]+=self.embeddings[key][self.edges[key][0,idx]]
            temp[self.embs_num:,self.share_dims:]+=self.embeddings[key][self.edges[key][1,idx]]
        temp[self.embs_num:,:]/=len(self.embeddings)
        '''

        for i in range(self.edges_num):
            for key in self.embeddings.keys():
                num = i
                if num>=self.edges[key].shape[1]:num = self.edges[key].shape[1]-1
                temp[i+self.embs_num,:self.share_dims]+=self.embeddings[key][self.edges[key][0,num]]
                temp[i+self.embs_num,self.share_dims:]+=self.embeddings[key][self.edges[key][1,num]]
            temp[i+self.embs_num]/=len(self.embeddings)



        self.mean_space = temp



                



    def update_mean_space2(self):
        self.embs_num = int(np.mean([value.shape[0] for value in self.embeddings.values()]))
        self.edges_num = self.embs_num * 2
        print(self.embs_num,self.edges_num)
        temp = np.zeros((self.embs_num+self.edges_num,self.share_dims*2))

        for i in range(self.embs_num):
            for value in self.embeddings.values():
                num = np.random.choice(value.shape[0])
                temp[i,:self.share_dims]+=value[num]
            temp[i]/=len(self.embeddings)
        temp[:self.embs_num,self.share_dims:] = temp[:self.embs_num,:self.share_dims] 

        
        for i in range(self.edges_num):
            for key in self.embeddings.keys():
                num = np.random.choice(self.edges[key].shape[1])
                temp[i+self.embs_num,:self.share_dims]+=self.embeddings[key][self.edges[key][0,num]]
                temp[i+self.embs_num,self.share_dims:]+=self.embeddings[key][self.edges[key][1,num]]
            temp[i+self.embs_num]/=len(self.embeddings)


        self.mean_space = temp

        
        

        
            

    def draw_graph(self):
        Xs = []
        for key in ['reddit1','yelp1','arxiv1']:
            node_embs = np.hstack(( self.embeddings[key], self.embeddings[key]))
            if self.edges[key].shape[1] <= self.edges_num:
                select_edges = self.edges[key]
            else:
                perm = np.random.choice(self.edges[key].shape[1],self.edges_num,replace=False)
                select_edges = self.edges[key][:,perm]


            edge_embs = np.hstack((self.embeddings[key][select_edges[0]],self.embeddings[key][select_edges[1]]))
            space_vector = np.vstack((node_embs,edge_embs))
            Xs.append(space_vector)
        Xs.append(self.mean_space)
        draw(Xs)



        
    def update_share_space2(self,device):

        self.embs_num = int(np.mean([value.shape[0] for value in self.embeddings.values()]))
        self.edges_num = self.embs_num * 2
        

        print(self.embs_num,self.edges_num)


        Xs = []

        for key in self.edges.keys():
            Xs.append(self.embeddings[key])
        Xbar = np.random.randn(self.embs_num,int(Xs[0].shape[1]))
        #Xbar = self.mean_space[:self.embs_num,0:int(Xs[0].shape[1])]
        mu_s = [unif(X.shape[0]) for X in Xs]
        node_Xbar = sinkhorn_barycenter(mu_s, Xs, Xbar,  reg=1e-3, b=None, weights=None,
                                                    norm="max", metric="sqeuclidean", numItermax=100,
                                                    numInnerItermax=5000, stopThr=1e-4, verbose=False,
                                                    innerVerbose=False,log=False, limit_max=np.infty, callbacks=None,
                                                    implementation='torch',device=device)
        Xs = []
        for key in self.edges.keys():


            if self.edges[key].shape[1] <= self.edges_num:
                select_edges = self.edges[key]
            else:
                select_edges = self.edges[key][:,0:self.edges_num]

            edge_embs = np.hstack((self.embeddings[key][select_edges[0]],self.embeddings[key][select_edges[1]]))

            Xs.append(edge_embs)
        #Xbar = self.mean_space[self.embs_num:,0:int(Xs[0].shape[1])]
        Xbar = np.random.randn(self.edges_num,int(Xs[0].shape[1]))
        mu_s = [unif(X.shape[0]) for X in Xs]
        edge_Xbar = sinkhorn_barycenter(mu_s, Xs, Xbar,  reg=1e-3, b=None, weights=None,
                                                    norm="max", metric="sqeuclidean", numItermax=100,
                                                    numInnerItermax=5000, stopThr=1e-4, verbose=False,
                                                    innerVerbose=False,log=False, limit_max=np.infty, callbacks=None,
                                                    implementation='torch',device=device)


        node_embs = np.hstack((node_Xbar,node_Xbar))
        space_vector = np.vstack((node_embs,edge_Xbar))
                                                       
        self.mean_space = space_vector



    def update_share_space(self,device):

        self.embs_num = int(np.mean([value.shape[0] for value in self.embeddings.values()]))
        self.edges_num = self.embs_num * 2

        print(self.embs_num,self.edges_num)


        Xs = []
        Ys = []

        for key in self.edges.keys():

            node_embs = np.hstack(( self.embeddings[key], self.embeddings[key]))
            if self.edges[key].shape[1] <= self.edges_num:
                select_edges = self.edges[key]
            else:
                select_edges = self.edges[key][:,0:self.edges_num]


            edge_embs = np.hstack((self.embeddings[key][select_edges[0]],self.embeddings[key][select_edges[1]]))
            space_vector = np.vstack((node_embs,edge_embs))

            labels = np.array([0]*node_embs.shape[0] + [1] *edge_embs.shape[0])



            Xs.append(space_vector)
            Ys.append(labels)

        if self.mean_space == None:
            self.update_mean_space()
            
        Xbar = self.mean_space


        mu_s = [unif(X.shape[0]) for X in Xs]
        ybar = np.array([0]*self.embs_num +[1]*self.edges_num)



        Xbar = sinkhorn_barycenter(mu_s, Xs, Xbar, ys=Ys, ybar=ybar, reg=1e-3, b=None, weights=None,
                                                    norm="max", metric="sqeuclidean", numItermax=100,
                                                    numInnerItermax=3000, stopThr=1e-3, verbose=False,
                                                    innerVerbose=False,log=False, limit_max=np.infty, callbacks=None,
                                                    implementation='torch',device=device)


        self.mean_space = Xbar
        









    def add_all_MLP(self,datasets,device):
        nums= len(datasets)
        for i in range(nums):
            name = datasets[i].name
            id = datasets[i].id
            key= name+str(id)
            self.inputs[key] = datasets[i].x.cpu().numpy()
            self.pool[key] = MLP(datasets[i].x.shape[1],self.hidden_dims,self.share_dims)
            self.embeddings[key] = self.pool[key](datasets[i].x ).detach().cpu().numpy()
            self.edges[key] = datasets[i].train_pos_edge_index.detach().cpu().numpy()
            self.pool[key].to(device)











