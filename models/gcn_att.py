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


# 直接通过attn_proj的方式效果很差。
# class GraphAtt(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=False, relu=True):
#         super().__init__()
#         if dropout:
#             self.dropout = nn.Dropout(p=0.5)
#         else:
#             self.dropout = None

#         # self.w = nn.Parameter(torch.empty(in_channels, out_channels))
#         # self.b = nn.Parameter(torch.zeros(out_channels))
#         # xavier_uniform_(self.w)
#         self.proj = nn.Linear(in_channels, out_channels)

#         if relu:
#             self.relu = nn.LeakyReLU(negative_slope=0.2)
#         else:
#             self.relu = None

#         self.attn_act = nn.LeakyReLU(negative_slope=0.2)
#         self.attn_src = nn.Linear(out_channels, 1, bias=False)
#         self.attn_dst = nn.Linear(out_channels, 1, bias=False)
#         self.out_channels = out_channels

#     def forward(self, inputs, padded_tensor, mask, n):  # att-trainable params
#         if self.dropout is not None:
#             inputs = self.dropout(inputs)
#         # src_feats = torch.mm(inputs, self.w) + self.b
#         src_feats = F.normalize(self.proj(inputs))

#         # TODO query, key, value matrix
#         # outputs = torch.empty(n, self.out_channels).type_as(src_feats)
#         # neigh_feats = torch.empty(n, mask.size(1), self.out_channels).cuda()  # 3w*190*2048
#         # dst_feats = torch.empty(n, 1, self.out_channels).type_as(src_feats)  # 3w*1*2048
        
#         # for idx, neigh in enumerate(padded_tensor):
#         #     neigh_feats[idx] = src_feats[neigh]
#         neigh_feats = src_feats[padded_tensor]
#         src_feats = src_feats.unsqueeze(1)
#         attention_scores = torch.matmul(src_feats, neigh_feats.transpose(-1, -2)) # [n, 1, 2048] * [n, 2048, 4] -> [n, 1, 4]
#         attention_scores = attention_scores / math.sqrt(mask.size(1))
        
#         # Normalize the attention scores to probabilities.
#         attention_probs = masked_softmax_(attention_scores, mask)
#         # attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # attention_probs = self.attn_dropout(attention_probs)
        
#         # outputs = (torch.matmul(attention_probs, neigh_feats) + src_feats).squeeze()
#         outputs = (torch.matmul(attention_probs, neigh_feats) + src_feats).squeeze()
 
#         # for idx, (neigh, dst) in enumerate(zip(padded_tensor, dst_nodes)):
#         #     neigh_feats[idx] = src_feats[neigh]
#         #     dst_feats[idx] = src_feats[neigh]

#         #     # attention
#         #     dst_attn = self.attn_act(self.attn_dst(dst_feats))
#         #     neigh_attn = self.attn_act(self.attn_src(neigh_feats))

#         #     # [optional]: edge_attention
#         #     edge_attn = dst_attn + neigh_attn
#         #     attn = masked_softmax_(edge_attn, _mask.unsqueeze(-1)).view(1, -1)

#         #     outputs[i] = torch.matmul(attn, neigh_feats).squeeze()

#         if self.relu is not None:
#             outputs = self.relu(outputs)
#         return outputs
        

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


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.proj = nn.Linear(in_features=in_channels, out_features=out_channels)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, word_vectors, src_idx, neighs_idx, src_mask):  # att-trainable params
        supports = self.proj(word_vectors)
        if self.dropout is not None:
            supports = self.dropout(supports)

        src_feats = supports[src_idx]
        neigh_feats = supports[neighs_idx]
        
        outputs = torch.matmul(src_mask.unsqueeze(1).float(), neigh_feats).squeeze() + src_feats
        if self.relu is not None:
            outputs = self.relu(outputs)
        supports[src_idx] += outputs

        return supports


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-2, keepdim=True)
        s = (x - u).pow(2).mean(-2, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GraphAtt(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.proj_before = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.proj_after = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # self.ln = LayerNorm(out_channels, out_channels)
        self.key = nn.Linear(out_channels, out_channels, bias=False)
        self.query = nn.Linear(out_channels, out_channels, bias=False)
        # self.smp = nn.Parameter(torch.ones(1))
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
        

    def forward(self, word_vec, src_idx, neighs_idx, src_mask):  # att-trainable params
        if self.dropout is not None:
            supports = self.dropout(word_vec)
        # supports = self.proj_before(supports)
        # if self.relu is not None:
        #     supports = self.relu(supports)x
        # query = self.query(supports[src_idx]).unsqueeze(1)
        # key = self.key(supports[neighs_idx])
        query = supports[src_idx].unsqueeze(1)
        key = supports[neighs_idx]
        # key = self.ln(self.key(supports[neighs_idx]))
        # values = self.value(neighs_feats)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))*5 # [n, 1, 2048] * [n, 2048, 4] -> [n, 1, 4]
        # attention_scores = attention_scores / math.sqrt(src_mask.size(1))
        # attention_scores = torch.rand(src_mask.shape).unsqueeze(1).cuda()
        
        # Normalize the attention scores to probabilities.
        attention_probs = masked_softmax_(attention_scores, src_mask)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.attn_dropout(attention_probs)
        supports = self.proj_before(supports)
        supports[src_idx] = torch.matmul(attention_probs, supports[neighs_idx]).squeeze()
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


        # 可学权重值，为三种固定权重值赋予动态调整
        # self.a_att = nn.Parameter(torch.ones(self.d))
        # self.r_att = nn.Parameter(torch.ones(self.d))

        layer1 = GraphAtt(in_channels, 512, relu=True, dropout=True)
        layer2 = GraphAtt(512, 1024, relu=True, dropout=True)
        layer3 = GraphAtt(1024, out_channels, relu=True, dropout=True)
        # layer4 = GraphAtt(out_channels, out_channels, relu=True, dropout=True)

        # layer1 = GraphAtt(in_channels, 1024, relu=True, dropout=True)
        # layer2 = GraphAtt(1024, 2048, relu=True, dropout=False)
        # layer3 = GraphAtt(2048, out_channels, relu=True, dropout=False)

        self.layers = []
        for i in range(3):
            self.layers.append(eval(f'layer{str(i+1)}')) 
            self.add_module(f'att-layer{str(i+1)}', self.layers[i])

        # self.layers = layers

        # dst_nodes = torch.arange(self.n, requires_grad=False).view(self.n, 1).repeat(1, mask.size(1)).long()

        # print('self.padded_tensor.shape:', self.padded_tensor.shape)
        # print('self.padded_tensor_r.shape:', self.padded_tensor_r.shape)


    # src_feats: [n, 300], neighs_feats:[n*k, 300], src_mask:[n, max_len] 
    # def forward(self, word_vectors, src_idx, neighs_idx, src_mask):
    def forward(self, word_vec, src_idx, neighs_idx, src_mask):
        # uni_src = src_idx.flatten()
        # uni_neighs = torch.unique(neighs_idx)
        # neighs_num = torch.nonzero(src_mask.flatten()).shape[0]
        # cnt1 = []
        # cnt2 = []
        # # 统计邻居结点中，有多少邻居会被训练过程影响而修改。
        # for s, n, mask in zip(src_idx, neighs_idx, src_mask):
        #     total = torch.nonzero(mask).shape[0]
        #     if total == 0:
        #         continue
        #     cnt = 0
        #     for nn in n:
        #         if nn in uni_src and nn!=s and nn!=0:
        #             cnt+=1
        #     if s < 1000:
        #         cnt1.append(cnt/total)
        #     else:

        #         cnt2.append(cnt/total)
        # print('%i src, %i neighs, %i unique neighs, train modified : %.2f %%, unseen modified : %.2f %%.' % (uni_src.shape[0], neighs_num, uni_neighs.shape[0], np.mean(cnt1), np.mean(cnt2)))
        for layer in self.layers:
            word_vec = layer(word_vec, src_idx, neighs_idx, src_mask)
            
        # return word_vec[src_idx]
        return F.normalize(word_vec[src_idx])
