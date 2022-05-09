#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-12-06 00:04:29
LastEditTime: 2022-02-14 15:22:05
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/materials/make_dense_graph.py
'''
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='imagenet-induced-graph.json')
parser.add_argument('--output', default='imagenet-dense-graph.json')
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

new_edges = []

# 将连通起来的边进行直接相连。
for u, wnid in enumerate(wnids):
    q = [u]
    l = 0
    d = {}   # 存放到当前节点距离
    d[u] = 0
    while l < len(q):
        x = q[l]
        l += 1
        for y in adjs[x]:
            if d.get(y) is None:
                d[y] = d[x] + 1
                q.append(y)
    for x, dis in d.items():
        new_edges.append((u, x))

json.dump({'wnids': wnids, 'vectors': vectors, 'edges': new_edges},
          open(args.output, 'w'))

