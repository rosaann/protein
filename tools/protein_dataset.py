#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:05:42 2018

@author: zl
"""

import torch.utils.data as data
import pandas as pd
import cv2
import os

class ProteinDataSet(data.Dataset):
    def __init__(self,preproc=None, base_path='../train/', csv_path='../train.csv'):
        self.df = pd.read_csv(csv_path)
        self.preproc = preproc
        self.base_path = base_path
        self.img_name_tails = [ 'red', 'green', 'blue', 'yellow']
    #    self.img_name_tails = [ 'red', 'green', 'blue']
        self.idx = 0
        
    def __getitem__(self, index):
        img_id = self.df.get_value(index, 'Id')
        target = self.df.get_value(index, 'Target')
        
        imgs = []
        
        for tail in self.img_name_tails:
            img_path = self.base_path + img_id + '_' + tail + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        img_merg = cv2.merge(imgs)
        
        if self.preproc is not None:
            img_merg = self.preproc(img_merg)
       
     #   tar_list = []
     #   print('targets in getitem ', target)
      #  targets = target.split(' ')
      #  for tar in targets:
      #      tar_list.append(tar)
        
    #    print('tar_list in getitem ', tar_list)
        if target.find(self.idx+'') >= 0:
            cv2.imwrite(os.path.join('./data/','{}_.png'.format(self.idx)), img_merg)
            self.idx += 1
        return img_merg, target
        
    def __len__(self):
        return self.df.shape[0]