import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import math
import random 
def pad_tensor(adj_nodes_list):
    """Function pads the neighbourhood nodes before passing through the
    aggregator.
    Args:
        adj_nodes_list (list): the list of node neighbours
    Returns:
        tuple: one of two tensors containing the padded tensor and mask
    """
    # mean_len = int(np.mean([len(adj_nodes) for adj_nodes in adj_nodes_list]))
    mean_len = 4

    padded_nodes = []
    _mask = []
    for adj_nodes in adj_nodes_list:
        if len(adj_nodes) > mean_len:
            x = random.sample(adj_nodes, mean_len)
        else:
            x = list(adj_nodes[:mean_len])
        _mask.append([1] * len(x) + [0] * (mean_len-len(x)))
        x += [0] * (mean_len - len(x))
        padded_nodes.append(x)

    # returning the mask as well
    return torch.tensor(padded_nodes, requires_grad=False).long(), torch.tensor(_mask, requires_grad=False).long()


def SequenceMask(X, X_len,value=-1e6):
    maxlen = X.size(1)
    mask = torch.arange((maxlen),dtype=torch.float, device=X.device)[None, :] >= X_len[:, None]   
    X[mask]=value
    return X
    

def masked_softmax_(X, mask):
    shape = X.shape
    softmax = nn.Softmax(dim=-1)
    # mask = mask.to(X.device)
    # mask = mask.repeat(1, mask.shape[-1]).view(shape).to(X.device)

    replace_ = torch.FloatTensor([-1e6]).repeat(X.shape[0], X.shape[2]).to(X.device)
    X = torch.where(mask==1, X.squeeze(), replace_)
    return softmax(X).reshape(shape)


# class GCN_Dense_Aux(nn.Module):

#     # edges_set按距离（跳数）划分成多个list
#     def __init__(self, n, edges, aux, in_channels, out_channels, hidden_layers):
#         super().__init__()

#         self.n = n
#         # aux_edges = torch.tensor(aux['aux_edges']) # hier_edges, parallel_edges, self_edges
#         # edges = torch.tensor(edges)
#         # edges_r = torch.tensor([edges[:, 1], edges[:, 0]]).permute(1, 0)
#         # aux_edges_r = torch.tensor([aux_edges[:,1], aux_edges[:,0]]).permuete(1,0)
        
#         self.adjs = [[] for _ in range(self.n)]
#         self.adjs_r = [[] for _ in range(self.n)]

#         self.d = 2
        
#         for a, b in edges:
#             self.adjs[a].append(b)
#         #     self.adjs[b].append(a)

#         # for idx, a in enumerate(self.adjs):
#         #     self.adjs[idx] = list(set(a))

#         # self.a_adj_set = []
#         # self.r_adj_set = []

#         # self.a_adj_set.append(edges)
#         # self.a_adj_set.append(aux_edges)

#         # self.r_adj_set.append(r_edges) 
#         # self.r_adj_set.append(r_aux_edges) 


#         # 可学权重值，为三种固定权重值赋予动态调整
#         # self.a_att = nn.Parameter(torch.ones(self.d))
#         # self.r_att = nn.Parameter(torch.ones(self.d))

#         i = 0
#         layer1 = GraphAtt(in_channels, out_channels, relu=True, dropout=True)
#         layer2 = GraphAtt(out_channels, out_channels, relu=True, dropout=True)
#         self.layers = []
#         for i in range(1):
#             self.layers.append(eval(f'layer{str(i+1)}')) 
#             self.add_module(f'att-layer{str(i+1)}', self.layers[i])

#         # self.layers = layers

#         # dst_nodes = torch.arange(self.n, requires_grad=False).view(self.n, 1).repeat(1, mask.size(1)).long()

#         # print('self.padded_tensor.shape:', self.padded_tensor.shape)
#         # print('self.padded_tensor_r.shape:', self.padded_tensor_r.shape)

#     def forward(self, x, reverse=True):
#         padded_tensor, mask = pad_tensor(self.adjs)
#         for layer in self.layers:
#             x = layer(x, padded_tensor, mask, self.n)
            
#         return F.normalize(x)

class GraphAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.proj_before = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.ReLU(),
        )
        self.proj_after = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.ReLU(),
        )

        # self.ln = LayerNorm(out_channels, out_channels)
        self.key = nn.Linear(out_channels, out_channels, bias=False)
        self.query = nn.Linear(out_channels, out_channels, bias=False)
        self.aux_attn = nn.Sequential(
            nn.Linear(4, 1, bias=False),
            nn.Sigmoid())

        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.query.weight)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

        for module in [self.proj_before, self.proj_after]:
            for layer in module:
                try: xavier_uniform_(layer.weight)
                except: pass
        

    def forward(self, word_vec, src_idx, neighs_idx, aux, src_mask):  # att-trainable params
        if self.dropout is not None:
            supports = self.dropout(word_vec)
        # supports = self.proj_before(supports)
        # if self.relu is not None:
        #     supports = self.relu(supports)
        # query = self.query(supports[src_idx]).unsqueeze(1)
        # key = self.key(supports[neighs_idx])
        query = supports[src_idx].unsqueeze(1)
        key = supports[neighs_idx]
        # key = self.ln(self.key(supports[neighs_idx]))
        # values = self.value(neighs_feats)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))*5 # [n, 1, 2048] * [n, 2048, 4] -> [n, 1, 4]
        # attention_scores = attention_scores / math.sqrt(src_mask.size(1))
        # attention_scores = torch.rand(src_mask.shape).unsqueeze(1).cuda()
        attention_scores_aux = self.aux_attn(aux) # [n, 1, 2048] * [n, 2048, 4] -> [n, 1, 4]

        # Normalize the attention scores to probabilities.
        attention_probs = masked_softmax_(attention_scores, src_mask)
        attention_probs2 = masked_softmax_(attention_scores_aux.view(attention_scores.shape), src_mask)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.attn_dropout(attention_probs)
        supports = self.proj_before(supports)
        supports[src_idx] = torch.matmul((attention_probs + attention_probs2)/2, supports[neighs_idx]).squeeze()
        # supports[src_idx] = torch.matmul(attention_probs, supports[neighs_idx]).squeeze()
        # supports[src_idx] = self.proj_after(torch.matmul(attention_probs, key)).squeeze()
        # outputs = torch.matmul(attention_probs, neigh_feats).squeeze()
        
        # if self.relu is not None:
        #     outputs = self.relu(outputs)
        
        return supports
        

class GCN_Dense_Aux(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # aux_edges = torch.tensor(aux['aux_edges']) # hier_edges, parallel_edges, self_edges
        # edges = torch.tensor(edges)
        # edges_r = torch.tensor([edges[:, 1], edges[:, 0]]).permute(1, 0)
        # aux_edges_r = torch.tensor([aux_edges[:,1], aux_edges[:,0]]).permuete(1,0)
        
        # self.adjs = [[] for _ in range(self.n)]
        # self.adjs_r = [[] for _ in range(self.n)]

        # self.d = 2
        
        # for a, b in edges:
        #     self.adjs[a].append(b)
        #     self.adjs[b].append(a)

        # for idx, a in enumerate(self.adjs):
        #     self.adjs[idx] = list(set(a))

        # self.a_adj_set = []
        # self.r_adj_set = []

        # self.a_adj_set.append(edges)
        # self.a_adj_set.append(aux_edges)

        # self.r_adj_set.append(r_edges) 
        # self.r_adj_set.append(r_aux_edges) 

        layer1 = GraphAtt(in_channels, out_channels, relu=True, dropout=True)
        layer2 = GraphAtt(out_channels, out_channels, relu=True, dropout=True)
        layer3 = GraphAtt(out_channels, out_channels, relu=True, dropout=True)
        layer4 = GraphAtt(out_channels, out_channels, relu=True, dropout=True)
        
        # layer1 = GraphAtt(in_channels, 1024, relu=True, dropout=True)
        # layer2 = GraphAtt(1024, 2048, relu=True, dropout=False)
        # layer3 = GraphAtt(2048, out_channels, relu=True, dropout=False)

        self.layers = []
        for i in range(2):
            self.layers.append(eval(f'layer{str(i+1)}')) 
            self.add_module(f'att-layer{str(i+1)}', self.layers[i])

        # self.layers = layers

        # dst_nodes = torch.arange(self.n, requires_grad=False).view(self.n, 1).repeat(1, mask.size(1)).long()

        # print('self.padded_tensor.shape:', self.padded_tensor.shape)
        # print('self.padded_tensor_r.shape:', self.padded_tensor_r.shape)


    # src_feats: [n, 300], neighs_feats:[n*k, 300], src_mask:[n, max_len] 
    # def forward(self, word_vectors, src_idx, neighs_idx, src_mask):
    def forward(self, word_vec, src_idx, neighs_idx, aux, src_mask):
        for idx, layer in enumerate(self.layers):
            word_vec = layer(word_vec, src_idx.squeeze().cuda(), neighs_idx[idx%2].squeeze().cuda(), aux[idx%2].squeeze().cuda(), src_mask[idx%2].squeeze().cuda())
            
        return F.normalize(word_vec[src_idx])
