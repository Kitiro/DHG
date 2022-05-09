#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-12-06 00:04:29
LastEditTime: 2022-02-09 20:23:02
LastEditors: Kitiro
Description: 利用wordnet（通过nltk）来构建点间关系。
FilePath: /19CVPR_GCN_imagenet_DGP/materials/make_induced_graph.py
'''
import argparse
import json

from nltk.corpus import wordnet as wn
import torch

from glove import GloVe


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:
            continue
        for p in u.hypernyms():  # hypernyms-父级  hyponyms-下位词
            if p not in vis:  # 将所有词的上位词都加入词表
                vis.add(p)
                q.append(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='imagenet-split.json')
    parser.add_argument('--output', default='imagenet-induced-graph.json')
    args = parser.parse_args()

    print('making graph ...')

    xml_wnids = json.load(open('imagenet-xml-wnids.json', 'r'))
    xml_nodes = list(map(getnode, xml_wnids))  # e.g.:  Synset('toilet_tissue.n.01')
    xml_set = set(xml_nodes)  # 32295

    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']

    key_wnids = train_wnids + test_wnids  # 21842

    s = list(map(getnode, key_wnids))
    induce_parents(s, xml_set)  # 29 of s not in xms_set 

    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)

    wnids = list(map(getwnid, s))
    edges = getedges(s)

    print('making glove embedding ...')

    glove = GloVe('glove.6B.300d.txt')
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()  
    obj['edges'] = edges
    json.dump(obj, open(args.output, 'w'))

