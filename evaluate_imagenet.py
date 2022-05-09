import argparse
import json
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet import make_resnet101_base, make_resnet50_base
from datasets.imagenet import ImageNet
from utils import set_gpu, pick_vectors
import os
import numpy as np


# 读取图片后过cnn得到global features
def test_on_subset(dataset, cnn, n, pred_vectors, all_label,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch # image_tensor, wnid
        data = data.cuda()

        feat = cnn(data) # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1) # 32,2048 -> 32,2049

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18  # 将train class部分置0

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t() # 该batch内，target label的score
        rks = (table >= gth_score).sum(dim=1)  # 若其他类预测分数大于target类，则说明分类错误；若rk==1，则说明分类正确。

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot

# 预提取保存到本地
npy_path = '/home/zzc/datasets/imagenet_feats/test'
def test_on_subset_local(wnid, n, pred_vectors, all_label,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0
    
    feat = torch.FloatTensor(np.load(os.path.join(npy_path, f'{wnid}.npy'))).cuda()
    feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1)
    fcs = pred_vectors.t()

    table = torch.matmul(feat, fcs)
    if not consider_trains:
        table[:, :n] = -1e18  # 将train class部分置0

    gth_score = table[:, all_label].repeat(table.shape[1], 1).t() # 该batch内，target label的score
    rks = (table >= gth_score).sum(dim=1)  # 若其他类预测分数大于target类，则说明分类错误；若rk==1，则说明分类正确。

    assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

    for i, k in enumerate(top):
        hits[i] += (rks <= k).sum().item()
    tot += len(feat)

    return hits, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn')
    parser.add_argument('--pred')

    parser.add_argument('--test-set')

    parser.add_argument('--output', default='save/result.json')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--test-train', action='store_true')

    args = parser.parse_args()

    set_gpu(args.gpu)

    test_sets = json.load(open('materials/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets[args.test_set]

    print('test set: {}, {} classes, ratio={}'
          .format(args.test_set, len(test_wnids), args.keep_ratio))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda()  # 得到train和test的预测的分类器

    n = len(train_wnids)
    m = len(test_wnids)

    cnn = eval(f'make_resnet{args.cnn}_base()')
    cnn.load_state_dict(torch.load(f'materials/resnet{args.cnn}-base.pth'))
    cnn = cnn.cuda()
    cnn.eval()

    TEST_TRAIN = args.test_train

    imagenet_path = 'materials/datasets/imagenet'
    dataset = ImageNet(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda() # top 1 2 5 10 20
    s_tot = 0

    results = {}

    if TEST_TRAIN:
        for idx, wnid in enumerate(train_wnids, 1):
            subset = dataset.get_subset(wnid)
            if not subset.valid:
                continue
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, idx - 1,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(idx, len(train_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))
    else:
        for idx, wnid in enumerate(test_wnids, 1):


            # subset = dataset.get_subset(wnid)
            # if not subset.valid:
            #     continue
            # hits, tot = test_on_subset(subset, cnn, n, pred_vectors, n + idx - 1,
            #                            consider_trains=args.consider_trains)
            # results[wnid] = (hits / tot).tolist()

            # 通过预存的npy文件读取test图像特征
            if not os.path.exists(os.path.join(npy_path, f'{wnid}.npy')):
                continue    
            hits, tot = test_on_subset_local(wnid, n, pred_vectors, n + idx - 1,
                                       consider_trains=args.consider_trains)

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(idx, len(test_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))

    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ')
    print('total {}'.format(s_tot))

    summary = ['{:.3f}%'.format(s_hit / s_tot * 100) for s_hit in s_hits]
    
    res = json.load(open(args.output)) if osp.exists(args.output) else {}
    
    if args.consider_trains:
        key = f'{args.test_set}-gzsl'
    else:
        key = f'{args.test_set}-zsl'
    save_name = args.pred.split('/')[1]
    if res.__contains__(save_name):
        if res[save_name].__contains__(key):
            res[save_name][key].append(summary)
        else:
            res[save_name][key] = [summary]
    else:
        res[save_name] = {}
        res[save_name][key] = [summary]
    print('save key:', key)
    json.dump(res, open(args.output, 'w'))

