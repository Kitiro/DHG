#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-12-06 00:04:29
LastEditTime: 2022-02-09 20:21:42
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/materials/make_dense_grouped_graph.py
'''
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='imagenet-induced-graph.json')
parser.add_argument('--output', default='imagenet-dense-grouped-graph.json')
args = parser.parse_args()

js = json.load(open(args.input, 'r'))
wnids = js['wnids']
vectors = js['vectors']
edges = js['edges']

n = len(wnids)
adjs = {}
for i in range(n):
    adjs[i] = []
for u, v in edges:
    adjs[u].append(v)

new_edges = [[] for i in range(99)]
# 根据跳数将边进行分组
for u, wnid in enumerate(wnids):
    q = [u]
    l = 0
    d = {}
    d[u] = 0
    while l < len(q):
        x = q[l]
        l += 1
        for y in adjs[x]:  # 遍历当前class的邻接点
            if d.get(y) is None:  # d记录邻接点到当前点的距离（跳数）
                d[y] = d[x] + 1
                q.append(y)
    for x, dis in d.items():
        new_edges[dis].append((u, x))

while new_edges[-1] == []:
    new_edges.pop()

json.dump({'wnids': wnids, 'vectors': vectors, 'edges_set': new_edges},
          open(args.output, 'w'))

