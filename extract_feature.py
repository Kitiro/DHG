#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-10-31 14:29:42
LastEditTime: 2022-02-28 16:43:33
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/extract_feature.py
'''
import argparse
import os
import numpy as np
import json
import scipy.io as sio
from tqdm import tqdm 
import argparse

import scipy.io as sio
from tqdm import tqdm 

import os, torch
import numpy as np
from PIL import Image 
from torchvision import models, transforms
import random 

dir = 'materials'
save_dir = '/home/zzc/datasets/imagenet_feats'
# save_dir = 'materials/web_feats/imagenet'
load_dir = '/home/zzc/datasets/imagenet'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def conver_img(img_files):
    feed = []
    for img_path in img_files:
        try:
            img = transform(Image.open(img_path))
            if img.shape[0] == 1:
                img = torch.Tensor(np.tile(img, (3,1,1)))
            img = normalize(img)
        
            feed.append(img)
        except:
            print('failed!!!', img_path)
    if len(feed) == 0:
        return None
    else:
        return torch.stack(feed)    


def get_model(args):
    if args.model == 'res50':
        model = models.resnet50(pretrained=True)
    elif args.model  == 'res101':
        model = models.resnet101(pretrained=True)
    elif args.model == 'res152':
        model = models.resnet152(pretrained=True)
    del model.fc
    model.fc = lambda x:x
    print('extract feature with pretrained %s' % args.model)
    model.eval()
    return model

    
def extract_web(args, wnids):
    model = get_model(args).cuda()
    # 每次处理一类
    print('start extracting', args.set)
    for wnid in tqdm(wnids):
        save_path = os.path.join(save_dir, wnid+'.npy')
        if os.path.exists(save_path):
            continue
        img_dir = os.path.join(load_dir, wnid)
        img_files = os.listdir(img_dir)
        img_files.sort(key=lambda x:int(x[:-4]))  # 按index排序
        img_files = [os.path.join(img_dir, file) for file in img_files]
        
        imgs_tensor = conver_img(img_files)
        if imgs_tensor is None:
            print(f'%s not exists' % save_path)
            continue 
        else:
            imgs_tensor = imgs_tensor.cuda()
        feats = model(imgs_tensor).detach().cpu().numpy()
        np.save(save_path, feats)
    
    print('save features on %s' % save_dir)
# 'ft_res101_feature_AWA2.mat'

def extract_imagenet(args, wnids):
    model = get_model(args).cuda()
    # 每次处理一类
    print('start extracting', args.set)
    for wnid in tqdm(wnids):
        save_path = os.path.join(save_dir, wnid+'.npy')
        if os.path.exists(save_path):
            continue
        img_dir = os.path.join(load_dir, wnid)
        if not os.path.exists(img_dir):
            print(img_dir, 'not exists')
            continue
        img_files = os.listdir(img_dir)
        random.shuffle(img_files)
        img_files = img_files[:max(1, round(len(img_files) * args.ratio))]
        img_files = [os.path.join(img_dir, file) for file in img_files]
        
        save_feats = torch.empty((0, 2048)).cuda()
        for img_path in img_files:
            try:
                img = transform(Image.open(img_path))
                if img.shape[0] == 1:
                    img = torch.Tensor(np.tile(img, (3,1,1)))
                img = normalize(img).cuda()
                feats = model(img.unsqueeze(0)).detach()
                save_feats = torch.cat((save_feats, feats), dim=0)
            except:
                print('failed!!!', img_path)
            
        np.save(save_path, save_feats.cpu().numpy())
    print('save features on %s' % save_dir)

def parse_arg():
    parser = argparse.ArgumentParser(description='word embeddign type')
    parser.add_argument('--model', type=str, default='res101',
                        help='word embedding type: [inception, res50]')
    parser.add_argument('--set', type=str, default='train', help='2-hops, 3-hops, all...')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--ratio', type=float, default='0.1', help='keep ratio')
        
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args
    
if __name__ == '__main__':
    args = parse_arg()
    if args.set == 'train':
        wnids = json.load(open(os.path.join(dir, 'imagenet-split.json')))['train']
    else:
        wnids = json.load(open(os.path.join(dir, 'imagenet-testsets.json')))[args.set]
    # extract_web(args, wnids)
    extract_imagenet(args, wnids)
    # extract_imagenet(args, ['n04399382'])

