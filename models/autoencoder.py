import math
import random

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.models import MLP

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)



EPS = 1e-15
LOG_VAR_MAX = 10
LOG_VAR_MIN = EPS

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def negative_sampling(pos_edge_index, num_nodes):


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) #这里是吧整个矩阵flatten
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   #????




    return torch.stack([row, col], dim=0).to(pos_edge_index.device)










class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """


        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class MyGAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

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
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple = False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()



        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))




        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

          

        #print(pos_loss.item(),neg_loss.item(),'neg and pos loss')
        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    



class MyVGAE(MyGAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(MyVGAE, self).__init__(encoder, decoder=decoder)

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logvar__ = self.encoder(*args, **kwargs)
        self.__logvar__ = torch.clamp(self.__logvar__,min=LOG_VAR_MIN,max=LOG_VAR_MAX)
        # self.__logvar__ = torch.clamp(self.__logvar__,max=LOG_VAR_MAX)
        z = self.reparametrize(self.__mu__, self.__logvar__)
        return z

    def kl_loss(self, mu=None, logvar=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar
        # print("KL Variance is %f" %(torch.mean(logvar.exp()).item()))
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))


    






def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')
    
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)

    n_classes = len(classes)

    n_query = n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])



    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]   
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    #_, y_hat = log_p_y.max(2)
    #acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    #print('acc_val****',acc_val)

    return loss_val, prototypes

def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    return loss_val

def proto_qry_result(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    _, y_hat = log_p_y.max(2)
    #acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    #print('acc_val****',acc_val)


    return y_hat







