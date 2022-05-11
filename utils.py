#!/usr/bin/env python3
# coding=utf-8
"""
Author: Kitiro
Date: 2021-12-06 00:04:29
LastEditTime: 2022-02-17 15:18:25
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/utils.py
"""
import os
import os.path as osp
import shutil
from graphviz import view

import numpy as np
import scipy.sparse as sp
import torch

from torch.utils.data import Dataset
import random

def ensure_path(path):
    if osp.exists(path):
        # if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print("using gpu {}".format(gpu))


def pick_vectors(dic, wnids, is_tensor=False):
    o = next(iter(dic.values()))
    dim = len(o)
    ret = []
    for wnid in wnids:
        v = dic.get(wnid)
        if v is None:
            if not is_tensor:
                v = [0] * dim
            else:
                v = torch.zeros(dim)
        ret.append(v)
    if not is_tensor:
        return torch.FloatTensor(ret)
    else:
        return torch.stack(ret)


def l2_loss(a, b):
    return ((a - b) ** 2).sum() / (len(a) * 2)


def normt_spm(mx, method="in"):
    if method == "in":  # 每一行除以该行sum
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == "sym":
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, pred, opt=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pred)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if opt is not None:
                for params in opt.param_groups:             
                    # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9            
                    params['lr'] *= 0.9 
            self.trace_func(
                f"{score:.6f}/{self.best_score:.6f} EarlyStopping counter: {self.counter} out of {self.patience}. Lr: {opt.param_groups[0]['lr']:.6f}"
            )
            

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pred)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, pred):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model on {self.path} ..."
            )

        model_path = osp.join(self.path, "best_val.pth")
        pred_path = osp.join(self.path, "best_val.pred")

        torch.save(model.state_dict(), model_path)
        torch.save(pred, pred_path)

        self.val_loss_min = val_loss


class DataSet(Dataset):
    # 每次随机采样k个test结点
    def __init__(self, word, graph, random_neighs_num=500, padding_num=50, train=False):
        self.word_vectors = word
        # self.adjs = adjs
        self.n = len(word)
        self.k = random_neighs_num
        self.l = padding_num
        self.train = train
        self.prepare_adjs(graph)

        if self.train:
            self.train_list = list(range(1000))
            self.train_neighs, self.train_aux, self.train_mask = self.pad_tensor(adj_nodes_list=self.adjs, aux_list=self.aux, indices=self.train_list, max_len=self.l)
        else:
            self.test_neighs, self.test_aux, self.test_mask = self.pad_tensor(adj_nodes_list=self.adjs, aux_list=self.aux, indices=list(range(self.n)), max_len=self.l)  

    def prepare_adjs(self, graph):
        self.adjs = [[] for _ in range(self.n)]
        self.aux = [[] for i in range(self.n)]
        aux_feats = np.hstack((graph['hops'].reshape(-1, 1), graph['semantic_dis'].reshape(-1, 1), graph['visual_dis'].reshape(-1, 1), graph['hier'].reshape(-1, 1)))

        aux_feats_r = np.hstack((graph['hops'].reshape(-1, 1), graph['semantic_dis'].reshape(-1, 1), graph['visual_dis'].reshape(-1, 1), graph['hier_r'].reshape(-1, 1)))
        
        for idx, (a, b) in enumerate(graph['edges']):
            self.adjs[a].append(b)
            self.aux[a].append(list(aux_feats[idx]))
        for idx, (a, b) in enumerate(graph['edges_r']):
            if a == b :
                continue
            self.adjs[a].append(b)
            self.aux[a].append(list(aux_feats_r[idx]))

        if self.train:
            print('--------- adjs prepare ----------')
            cnt_seen, cnt_seen_del = 0, 0
            cnt_unseen, cnt_unseen_del = 0, 0
            for i in range(self.n):
                num = len(self.adjs[i])
                if i < 1000:
                    cnt_seen += num
                    cnt_seen_del += self.l if num > self.l else num
                else:
                    cnt_unseen += num
                    cnt_unseen_del += self.l if num > self.l else num
            print(cnt_seen+cnt_unseen, 'edges total.')
            print(cnt_seen_del+cnt_unseen_del, 'edges left after being padded.', f'throwing {cnt_seen-cnt_seen_del} seen edges.', f'throwing {cnt_unseen-cnt_unseen_del} unseen edges.')
            cal_len = max([len(adj_nodes) for adj_nodes in self.adjs])
            self.l = self.l if cal_len > self.l else cal_len
            print('cnt max neigh nodes num in adjs:', cal_len, 'padding_num is:', self.l)


    def pad_tensor(self, adj_nodes_list, aux_list, indices, max_len=10):
        """Function pads the neighbourhood nodes before passing through the
        aggregator. Providing more information on edges.
        Args:
            adj_nodes_list (list): the list of all node neighbours
            aux_list (list): aux information on corresponding edge
            indices (list): source nodes
        Returns:
            tuple: one of two tensors containing the padded tensor and mask
        """

        # 选出一个batch的source nodes及其对应neighbours
        batch_nodes_list = [adj_nodes_list[ind] for ind in indices]
        batch_aux_list = [aux_list[ind] for ind in indices]

        padded_nodes = []
        padded_aux = []
        mask = []
        for adj_nodes, nodes_aux in zip(batch_nodes_list, batch_aux_list):
            if len(adj_nodes) > max_len:
                random_idx = random.sample(range(1, len(adj_nodes)), max_len-1)
                x = [adj_nodes[0]] + [adj_nodes[i] for i in random_idx]
                a = [nodes_aux[0]] + [nodes_aux[i] for i in random_idx]
            else:
                x = list(adj_nodes[:max_len])
                a = nodes_aux[:max_len]
            mask.append([1] * len(x) + [0] * (max_len - len(x)))
            x += [0] * (max_len - len(x))
            a += [[0]*4 for _ in range(max_len - len(a))]
            padded_nodes.append(x)
            padded_aux.append(a)

        # returning the mask as well
        return torch.tensor(padded_nodes).long(), torch.tensor(padded_aux).float(), torch.tensor(mask).long()

    # for training
    def __getitem__(self, index):
        if self.train:
            src_idx = self.train_list
            random_idx = random.sample(list(range(len(self.train_list), self.n)), self.k)
            src_idx = torch.cat((torch.tensor(src_idx), torch.tensor(random_idx)))  # [batch, 1]
            random_neighs, random_aux, random_mask = self.pad_tensor(self.adjs, self.aux, random_idx, self.l)

            neighs_idx = torch.cat((self.train_neighs, random_neighs))  # [batch, padding]

            src_mask = torch.cat((self.train_mask, random_mask))

            aux = torch.cat((self.train_aux, random_aux))

            return src_idx, neighs_idx, aux, src_mask
        else:

            return index, self.test_neighs[index], self.test_aux[index], self.test_mask[index]

    def __len__(self):
        return self.n


# 带反向边的dataset
class DataSet_reverse(Dataset):
    # 每次随机采样k个test结点
    def __init__(self, word, graph, random_neighs_num=500, padding_num=50, train=False):
        self.word_vectors = word
        # self.adjs = adjs
        self.n = len(word)
        self.k = random_neighs_num
        self.l = padding_num
        self.l_r = padding_num

        self.train = train
        self.prepare_adjs(graph)

        if self.train:
            self.train_list = list(range(1000))
            self.train_neighs, self.train_aux, self.train_mask = self.pad_tensor(adj_nodes_list=self.adjs, aux_list=self.aux, indices=self.train_list, max_len=self.l)
            self.train_neighs_r, self.train_aux_r, self.train_mask_r = self.pad_tensor(adj_nodes_list=self.adjs_r, aux_list=self.aux_r, indices=self.train_list, max_len=self.l_r)
        else:
            self.test_neighs, self.test_aux, self.test_mask = self.pad_tensor(adj_nodes_list=self.adjs, aux_list=self.aux, indices=list(range(self.n)), max_len=self.l)  
            self.test_neighs_r, self.test_aux_r, self.test_mask_r = self.pad_tensor(adj_nodes_list=self.adjs_r, aux_list=self.aux_r,  indices=list(range(self.n)), max_len=self.l_r) 

    def prepare_adjs(self, graph):

                #     'wnids':wnids,
#     'edges':edges_save,
#     'edges_r':edges_r_save,
#     'hops':hops.squeeze(),
#     'semantic_dis':semantic_dis.squeeze(),
#     'visual_dis':visual_dis.squeeze(),
#     'hier':hier.squeeze(),
#     'hier_r':hier_r.squeeze(),

        self.adjs = [[] for _ in range(self.n)]
        self.adjs_r = [[] for _ in range(self.n)]
        self.aux = [[] for i in range(self.n)]
        self.aux_r = [[] for i in range(self.n)]
        
        aux_feats = np.hstack((graph['hops'].reshape(-1, 1), graph['semantic_dis'].reshape(-1, 1), graph['visual_dis'].reshape(-1, 1), graph['hier'].reshape(-1, 1)))
        aux_feats_r = np.hstack((graph['hops'].reshape(-1, 1), graph['semantic_dis'].reshape(-1, 1), graph['visual_dis'].reshape(-1, 1), graph['hier_r'].reshape(-1, 1)))

        for idx, (a, b) in enumerate(graph['edges']):
            self.adjs[a].append(b)
            self.aux[a].append(list(aux_feats[idx]))
        for idx, (a, b) in enumerate(graph['edges_r']):
            self.adjs_r[a].append(b)
            self.aux_r[a].append(list(aux_feats_r[idx]))

        if self.train:
            print('--------- adjs prepare ----------')
            cnt_seen, cnt_seen_del = 0, 0
            cnt_unseen, cnt_unseen_del = 0, 0
            # cnt = [len(adj_nodes) for adj_nodes in adjs]
            for i in range(self.n):
                num = len(self.adjs[i])
                if i < 1000:
                    cnt_seen += num
                    cnt_seen_del += self.l if num > self.l else num
                else:
                    cnt_unseen += num
                    cnt_unseen_del += self.l if num > self.l else num
            print(cnt_seen+cnt_unseen, 'edges total.')
            print(cnt_seen_del+cnt_unseen_del, 'edges left after being padded.', f'throwing {cnt_seen-cnt_seen_del} seen edges.', f'throwing {cnt_unseen-cnt_unseen_del} unseen edges.')
            cal_len = max([len(adj_nodes) for adj_nodes in self.adjs])
            self.l = self.l if cal_len > self.l else cal_len
            print('cnt max neigh nodes num in adjs:', cal_len, 'padding_num is:', self.l)

            print('--------- adjs_r prepare ----------')
            cnt_seen, cnt_seen_del = 0, 0
            cnt_unseen, cnt_unseen_del = 0, 0
            # cnt = [len(adj_nodes) for adj_nodes in adjs]
            for i in range(self.n):
                num = len(self.adjs_r[i])
                if i < 1000:
                    cnt_seen += num
                    cnt_seen_del += self.l if num > self.l else num
                else:
                    cnt_unseen += num
                    cnt_unseen_del += self.l if num > self.l else num
            print(cnt_seen+cnt_unseen, 'edges total.')
            print(cnt_seen_del+cnt_unseen_del, 'edges left after being padded.', f'throwing {cnt_seen-cnt_seen_del} seen edges.', f'throwing {cnt_unseen-cnt_unseen_del} unseen edges.')
            cal_len = max([len(adj_nodes) for adj_nodes in self.adjs_r])
            self.l_r = self.l_r if cal_len > self.l_r else cal_len
            print('cnt max neigh nodes num in adjs_r:', cal_len, 'padding_num is:', self.l_r)

        # return [adjs, adjs_r]


    def pad_tensor(self, adj_nodes_list, aux_list, indices, max_len=10):
        """Function pads the neighbourhood nodes before passing through the
        aggregator. Providing more information on edges.
        Args:
            adj_nodes_list (list): the list of all node neighbours
            aux_list (list): aux information on corresponding edge
            indices (list): source nodes
        Returns:
            tuple: one of two tensors containing the padded tensor and mask
        """

        # 选出一个batch的source nodes及其对应neighbours
        batch_nodes_list = [adj_nodes_list[ind] for ind in indices]
        batch_aux_list = [aux_list[ind] for ind in indices]

        padded_nodes = []
        padded_aux = []
        mask = []
        for adj_nodes, nodes_aux in zip(batch_nodes_list, batch_aux_list):
            if len(adj_nodes) > max_len:
                random_idx = random.sample(range(1, len(adj_nodes)), max_len-1)
                x = [adj_nodes[0]] + [adj_nodes[i] for i in random_idx]
                a = [nodes_aux[0]] + [nodes_aux[i] for i in random_idx]
            else:
                x = list(adj_nodes[:max_len])
                a = nodes_aux[:max_len]
            mask.append([1] * len(x) + [0] * (max_len - len(x)))
            x += [0] * (max_len - len(x))
            a += [[0]*4 for _ in range(max_len - len(a))]
            padded_nodes.append(x)
            padded_aux.append(a)

        # returning the mask as well
        return torch.tensor(padded_nodes).long(), torch.tensor(padded_aux).float(), torch.tensor(mask).long()

    # for training
    def __getitem__(self, index):
        if self.train:
            src_idx = self.train_list
            random_idx = random.sample(list(range(len(self.train_list), self.n)), self.k)
            src_idx = torch.cat((torch.tensor(src_idx), torch.tensor(random_idx)))  # [batch, 1]
            random_neighs, random_aux, random_mask = self.pad_tensor(self.adjs, self.aux, random_idx, self.l)
            random_neighs_r, random_aux_r, random_mask_r = self.pad_tensor(self.adjs_r, self.aux_r, random_idx, self.l_r)

            # src_feats = self.word_vectors[src_idx]
            neighs_idx = torch.cat((self.train_neighs, random_neighs))  # [batch, padding]
            # neighs_feats = self.word_vectors[neighs_idx]

            neighs_idx_r = torch.cat((self.train_neighs_r, random_neighs_r))  # [batch, padding]
            # neighs_feats_r = self.word_vectors[neighs_idx]

            src_mask = torch.cat((self.train_mask, random_mask))
            src_mask_r = torch.cat((self.train_mask_r, random_mask_r))

            aux = torch.cat((self.train_aux, random_aux))
            aux_r = torch.cat((self.train_aux_r, random_aux_r))

            return src_idx, [neighs_idx, neighs_idx_r], [aux, aux_r], [src_mask, src_mask_r]
            # return src_feats, [neighs_feats, neighs_feats_r], [src_mask, src_mask_r]
        else:
            return index, [self.test_neighs[index], self.test_neighs_r[index]], [self.test_aux[index], self.test_aux_r[index]], [self.test_mask[index], self.test_mask_r[index]]
            # return self.word_vectors[index], [self.word_vectors[self.test_neighs[index]], self.word_vectors[self.test_neighs_r[index]]], [self.test_mask[index],self.test_mask_r[index]]

    def __len__(self):
        return self.n