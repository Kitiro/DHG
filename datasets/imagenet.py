#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-12-06 00:04:29
LastEditTime: 2022-02-13 16:27:36
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/datasets/imagenet.py
'''
import json
import os
import os.path as osp
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from torchvision import get_image_backend


class ImageNet():

    def __init__(self, path):
        self.path = path
        self.keep_ratio = 1.0
    
    def get_subset(self, wnid):
        path = osp.join(self.path, wnid)
        return ImageNetSubset(path, wnid, keep_ratio=self.keep_ratio)

    def set_keep_ratio(self, r):
        self.keep_ratio = r


class ImageNetSubset(Dataset):

    def __init__(self, path, wnid, keep_ratio=1.0):
        self.wnid = wnid
        self.valid = True
        def pil_loader(path):
            with open(path, 'rb') as f:
                try:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    return img
                except OSError:
                    return None

        def accimage_loader(path):
            import accimage
            try:
                return accimage.Image(path)
            except IOError:
                return pil_loader(path)

        def default_loader(path):
            if get_image_backend() == 'accimage':
                return accimage_loader(path)
            else:
                return pil_loader(path)

        # get file list
        try:
            all_files = os.listdir(path)
        except:
            print(f'no class {self.wnid}')
            self.valid = False
            return
        files = []
        for f in all_files:
            if f.endswith('.JPEG'):
                files.append(f)
        random.shuffle(files)
        files = files[:max(1, round(len(files) * keep_ratio))]

        # read images
        data = []
        for filename in files:
            image = default_loader(osp.join(path, filename))
            if image is None:
                continue
           
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            img = transform(image)
            
            if img.shape[0] == 1:
                img = torch.Tensor(np.tile(img, (3,1,1)))
            img = normalize(img)
            # 将img第一维unsqueeze，传入模型
            # img = Variable(torch.unsqueeze(img, dim=0).float())
            data.append(img)
            
        if data != []:
            self.data = torch.stack(data) 
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.wnid

