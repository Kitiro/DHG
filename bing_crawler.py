#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-10-28 15:18:43
LastEditTime: 2022-02-28 16:31:44
LastEditors: Kitiro
Description: 
FilePath: /19CVPR_GCN_imagenet_DGP/bing_crawler.py
'''
import requests
from lxml import etree
import re
import time
import os 
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import json 
import shutil

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
}


NUM = 100 # 1000 downloaded images for each class

# 下载原图
class Bing():
    def __init__(self, max_num=1000) -> None:
        self.max = max_num
    def save_img(self, url, save_path, index):
        # img_name = url[-10:]
        # name = re.sub('/', '', img_name)  # img_name中出现/，将其置换成空
        try:
            res = requests.get(url, headers=headers)
        except OSError:
            print('出现错误，错误的url是:', url)
            return 0
        else:
            with open(os.path.join(save_path,str(index)+'.jpg'), 'wb')as f:
                try:
                    f.write(res.content)
                    return 1
                except OSError:
                    print('无法保存，url是：', url)
                    return 0


    # 获取全部图片url:原图
    def parse_source_img(self, img_list, url):
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding
        data = response.content.decode('utf-8', 'ignore')
        html = etree.HTML(data)
        conda_list = html.xpath('//a[@class="iusc"]/@m')
        for i in conda_list:
            img_url = re.search('"murl":"(.*?)"', i).group(1)
            if img_url not in img_list:
                img_list.append(img_url)
        return img_list[:self.max]
    
# 下载检索结果页面缩略图
class HeadlessDriver():
    def __init__(self, max_num=1000):
        self.max = max_num
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def save_img(self, url, save_path, index):
        try:
            res = requests.get(url, headers=headers)
        except OSError:
            print('出现错误，错误的url是:', url)
            return 0
        else:
            with open(os.path.join(save_path,str(index)+'.jpg'), 'wb')as f:
                try:
                    f.write(res.content)
                    return 1
                except OSError:
                    print('无法保存，url是：', url)
                    return 0

    # 获取全部图片url缩略图
    def parse_img(self, url):
        img_list = []
        self.driver.get(url)
        time.sleep(2)
        # 模拟滚到到底部
        height = 1000
        js = 'window.scrollBy(0,%i)' % height
        for _ in range(10):
            self.driver.execute_script(js)
            time.sleep(0.1)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        img_urls = soup.find_all(name='img',attrs={"class":"mimg"})#按照字典的形式给attrs参数赋值
        for u in img_urls:
            uu = re.search(r'https://(.*)\?', str(u))
            if uu==None:
                continue
            img_url = uu.group(0).replace('?','')
            if img_url not in img_list:
                img_list.append(img_url)   # 用https匹配img地址
        return img_list[:self.max]

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

# 主函数
def main():
    save_dir = '/home/zzc/datasets/imagenet_crawl'
    js = json.load(open('materials/imagenet-split.json', 'r'))
    nodes = list(map(getnode, js['test']))  # e.g.:  Synset('toilet_tissue.n.01')
    
    # crawler = Bing(1000)
    crawler = HeadlessDriver(1000)
    pbar = tqdm(nodes)
    for node in pbar:
        name = node.lemma_names()[0]
        wnid = getwnid(node)
        pbar.set_description((f"Processing {name}-{wnid}"))
        save_path = os.path.join(save_dir, wnid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            done = os.listdir(save_path)
            if len(done) > int(NUM*0.8):
                print('skip', wnid)
                continue
            else:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
            
        url = 'https://cn.bing.com/images/search?q={}&form=BESBTB&first=1&tsc=ImageBasicHover&ensearch=1'.format(name)
        img_list = crawler.parse_img(url)
        success = 0
        for idx, img_url in enumerate(img_list[:NUM]):
            res = crawler.save_img(img_url, save_path, idx)
            success += res 
            time.sleep(0.1)
        
if __name__ == '__main__':
    main()
