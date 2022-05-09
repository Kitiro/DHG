import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set, att_set):  # att-trainable params
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        support = torch.mm(inputs, self.w) + self.b
        outputs = None
        for i, (adjs, att) in enumerate(zip(adj_set, att_set)):
            if len(adjs) == 0:
                continue
            y = torch.mm(adjs, support)*att  # 第i跳的权重
            if outputs is None:
                outputs = y
            else:
                outputs = outputs + y  # 用input过tranform后，再分别与1~n跳的边得到加权和。全部相加。

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense_Aux(nn.Module):

    # edges_set按距离（跳数）划分成多个list
    def __init__(self, n, edges_set, in_channels, out_channels, hidden_layers):
        super().__init__()

        self.n = n
        
        self.d = len(edges_set)  # 2  direction weight
        self.h = len(edges_set[0])  # depth weight

        self.a_adj_set = [[] for _ in range(self.h)]
        self.r_adj_set = [[] for _ in range(self.h)]
        
        # two
        for idx, edges in enumerate(edges_set):
            for layer, edge in enumerate(edges):
                if len(edge) == 0:
                    continue
                edge = np.array(edge)
                adj = sp.coo_matrix((np.ones(len(edge)), (edge[:, 0], edge[:, 1])),
                                    shape=(n, n), dtype='float32')
                adj = spm_to_tensor(normt_spm(adj, method='in')).cuda()
                # r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in')).cuda()
                if idx == 0:
                    self.a_adj_set[layer] = adj
                else:
                    self.r_adj_set[layer] = adj

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        # 可学权重值，为不同距离的边赋予权重

        self.att_set = nn.Parameter(torch.ones(self.d, self.h))  # 每个edges_set的权重

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    
    def forward(self, x):
        graph_side = True
        # shape = self.att_set.shape
        # att_set = F.softmax(self.att_set.view(1, -1)).view(*shape)
        for conv in self.layers:
            if graph_side:
                adj_set = self.a_adj_set
                att_set = self.att_set[0]
            else:
                adj_set = self.r_adj_set
                att_set = self.att_set[1]
            att_set = F.softmax(att_set, dim=0)
            x = conv(x, adj_set, att_set)
            graph_side = not graph_side

        return F.normalize(x)

