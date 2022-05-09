import argparse
import json
import random
import os
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np

from utils import ensure_path, set_gpu, l2_loss
from models.gcn_aux_edges import GCN_Dense_Aux
import scipy.io as sio


def save_checkpoint():
    model_path = osp.join(save_path, 'best_val.pth')
    pred_path = osp.join(save_path, 'best_val.pred')
    if osp.exists(model_path):
        os.remove(model_path)
        os.remove(pred_path)
    torch.save(gcn.state_dict(), model_path)
    torch.save(pred_obj, pred_path)


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=500)
    parser.add_argument('--save-path', default='save/gcn-dense-aux')
    parser.add_argument('--weight-file', default='materials/imagenet-nell-edges-all.mat')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--hl', default='d2048,d')
    parser.add_argument('--no-pred', action='store_true')

    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    graph = sio.loadmat(args.weight_file)
    edges = graph['edges']
    #aux_edges = graph['aux_edges']

    # hops = graph['hops'].reshape(-1,1)
    
    # semantic_dis = graph['semantic_dis'].reshape(-1,1)
    # visual_dis = graph['visual_dis'].reshape(-1,1)
    
    # aux = sio.loadmat('materials/imagenet-graph-nell.mat')
    # edges1 = aux['hier_edges'].squeeze()
    # edges2 = aux['parallel_edges'].squeeze()
    # edges3 = aux['self_edges'].squeeze()
    wnids = list(graph['wnids'])
    n = len(wnids)
    
    js = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    # wnids = js['wnids']
    # n = len(wnids)

    word_vectors = torch.tensor(js['vectors'][:n]).float().cuda()
    # word_vectors = F.normalize(word_vectors)

    word_vectors2 = torch.tensor(np.load('materials/class_attribute_map_imagenet.npy')).float().cuda()
    # word_vectors2 = F.normalize(word_vectors2, dim=0)

    # word_mat = sio.loadmat('materials/class_attribute_map_imagenet_bert_prompt.mat')

    # word_vectors = F.normalize(torch.tensor(word_mat['embedding'].squeeze()).float().cuda())

    # word_vectors2 = torch.tensor(word_mat['matrix'].squeeze()).float().cuda()
    # word_vectors2 = F.normalize(word_vectors2, dim=0)

    word_vectors = torch.cat((word_vectors, word_vectors2), dim=1)

    word_vectors = torch.cat((word_vectors, torch.zeros(2, word_vectors.shape[-1]).float().cuda()), dim=0)

    word_vectors = F.normalize(word_vectors)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    hidden_layers = args.hl # d2048,d
    gcn = GCN_Dense_Aux(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()
    
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

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gcn.eval()
        output_vectors = gcn(word_vectors)
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
              .format(epoch, train_loss, val_loss))

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss if min_loss < val_loss else val_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if val_loss <= trlog['min_loss']:
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors,
                    'epoch':epoch,
                    'val_loss':val_loss
                }
            save_checkpoint()
            
        # if (epoch % args.save_epoch == 0):
        #     if args.no_pred:
        #         pred_obj = None
        #     else:
        #         pred_obj = {
        #             'wnids': wnids,
        #             'pred': output_vectors
        #         }

        # if epoch % args.save_epoch == 0:
        #     save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None

