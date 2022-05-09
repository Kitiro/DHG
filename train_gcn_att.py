import argparse
import json
import random
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
import numpy as np

from utils import ensure_path, set_gpu, l2_loss, EarlyStopping, DataSet
from models.gcn_att import GCN_Dense_Aux
import scipy.io as sio
import time
from torch.utils.data import DataLoader

from torchviz import make_dot

def save_checkpoint():
    model_path = osp.join(save_path, 'best_val.pth')
    pred_path = osp.join(save_path, 'best_val.pred')
    # if osp.exists(model_path):
    #     os.remove(model_path)
    #     os.remove(pred_path)
    torch.save(gcn.state_dict(), model_path)
    torch.save(pred, pred_path)


def mask_l2_loss(a, b, mask):
    # idx_map = {j:i for i,j in enumerate(tlist)}
    # mask2 = [idx_map[m] for m in mask]
    return l2_loss(a[mask], b[mask])
    # return l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=2000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--save-epoch', type=int, default=500)
    parser.add_argument('--save-path', default='save/gcn-att')
    parser.add_argument('--weight-file', default='materials/imagenet-nell-edges-all.mat', help='imagenet-nell-edges-all.mat')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--hl', default='d2048,d')
    parser.add_argument('--no-pred', action='store_true')

    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    graph = sio.loadmat(args.weight_file)
    edges = graph['edges']
    # edges = np.vstack((graph['edges'], graph['aux_edges']))

    # hops = graph['hops'].reshape(-1,1)
    
    # semantic_dis = graph['semantic_dis'].reshape(-1,1)
    # visual_dis = graph['visual_dis'].reshape(-1,1)
    
    # aux = sio.loadmat('materials/imagenet-graph-nell.mat')
    # edges1 = aux['hier_edges'].squeeze()
    # edges2 = aux['parallel_edges'].squeeze()
    # edges3 = aux['self_edges'].squeeze()
    wnids = list(graph['wnids'])
    n = len(wnids)

    # aux_data = {'hops':hops, 'aux_edges':aux_edges}
    aux_data = None

    js = json.load(open('materials/imagenet-induced-graph.json', 'r'))

    # wnids = js['wnids']
    # n = len(wnids)


    # original word vectors
    # word_vectors = js['vectors'][:n]
    # word_dim = len(word_vectors[0])
    # word_vectors.append(list(np.zeros(word_dim)))
    # word_vectors.append(list(np.zeros(word_dim)))
    
    # word_vectors = torch.tensor(word_vectors)
    # word_vectors = F.normalize(word_vectors)


# extended attributes
    word_vectors = js['vectors'][:n]
    word_dim = len(word_vectors[0])
    
    word_vectors = torch.tensor(word_vectors).float()
    # word_vectors = F.normalize(word_vectors)

    # word_vectors2 = torch.tensor(np.load('materials/class_attribute_map_imagenet.npy')).float()
    # word_vectors2 = F.normalize(word_vectors2)
    
    # word_vectors = torch.cat((word_vectors, word_vectors2), dim=1)
    word_vectors = torch.cat((word_vectors, torch.zeros(2, word_vectors.shape[-1])), dim=0).cuda()

    word_vectors = F.normalize(word_vectors)

    
    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors)
    fc_vectors = F.normalize(fc_vectors).cuda()

    hidden_layers = args.hl # d2048,d
    # gcn = GCN_Dense_Aux(n, edges, aux_data, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()
    gcn = GCN_Dense_Aux(in_channels=word_vectors.shape[1], out_channels=fc_vectors.shape[1]).cuda()
    
    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    early_stopping = EarlyStopping(patience=20, verbose=True, path=args.save_path)  

    min_loss = 1e18

    gcn.train()

    pred = {
        'wnids': wnids,
        'pred': None,
    }
    padding_num = 10 # 随机切割遮盖neighs
    # prepare dataset

    dataset_train = DataSet(
        word_vectors,
        edges,
        random_neighs_num=10000,
        padding_num=padding_num,
        train=True,
    )
    dataset_test = DataSet(
        word_vectors,
        edges,
        padding_num=padding_num,
        train=False
    )

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=1, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=dataset_test, batch_size=1024, shuffle=False, num_workers=0
    )

    for epoch in range(1, args.max_epoch + 1):
        cnt = 0
        gcn.train()

        for src_feats, neighs_feats, src_mask in train_loader:
            if cnt == 10:
                break
            output_vectors = gcn(word_vectors, src_feats.squeeze().cuda(), neighs_feats.squeeze().cuda(), src_mask.squeeze().cuda())
            loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            # print('train finish %i iter' %cnt)
            # torch.cuda.empty_cache()

        with torch.no_grad():
            gcn.eval()
            output_vectors = torch.empty(0, fc_vectors.shape[1]).cuda()
            for src_feats, neighs_feats, src_mask in test_loader:
                output_vector = gcn(word_vectors, src_feats.squeeze().cuda(), neighs_feats.squeeze().cuda(), src_mask.squeeze().cuda())
                output_vectors = torch.cat((output_vectors, output_vector), dim=0)
            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
            if v_val > 0:
                val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
                loss = val_loss
            else:
                val_loss = 0
                loss = train_loss
            print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
                .format(epoch, train_loss, val_loss))
            # print('epoch {}, train_loss={:.4f}'
            #     .format(epoch, loss.item()))
            # trlog['train_loss'].append(train_loss)
            # trlog['val_loss'].append(val_loss)
            # trlog['min_loss'] = min_loss if min_loss < loss.item() else val_loss
            # torch.save(trlog, osp.join(save_path, 'trlog'))
            pred['pred'] = output_vectors
            early_stopping(loss, gcn, pred)
            if early_stopping.early_stop:
                print('trianing early stop on loss %.4f' % loss)
                break
        torch.cuda.empty_cache()
            # if loss.item() <= min_loss:
            #     if args.no_pred:
            #         pred_obj = None
            #     else:
            #         pred_obj = {
            #             'wnids': wnids,
            #             'pred': output_vectors,
            #             'epoch':epoch,
            #             'val_loss':loss.item()
            #         }
                # min_loss = loss.item()
                # save_checkpoint()
            