#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:24:19 2018

@author: zl
"""

import torch.utils.data as data
import pandas as pd
import cv2
import os
import random 
from config import Config

class ProteinTestDataSet(data.Dataset):
    def __init__(self,preproc=None, base_path='../test/', csv_path='../sample_submission.csv'):
        self.df = pd.read_csv(csv_path)
        self.preproc = preproc
        self.base_path = base_path
        self.config = Config()
        self.idx = 0
        
        self.test_image_dir = os.path.join('../', 'test/')
                
        self.test_image_merge_list = self.get_testimg_merge_list(self.test_image_dir)
        
    def get_testimg_merge_list(self,test_image_dir):
        file_list = []
        df=pd.read_csv('../sample_submission.csv')
        for i, row in df.iterrows():
            file_list.append(row['Id'])
            
        return file_list
    def get_merge_image(self, pre_dir):
        img_name_tails = [ 'red', 'green', 'blue', 'yellow']
        imgs = []
        for tail in img_name_tails:
            img_path = pre_dir+ '_' + tail + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        img_merg = cv2.merge(imgs)
        
        if self.preproc is not None:
            img_merg = self.preproc(img_merg) 
        return img_merg
    
    def get_gray_image(self, pre_dir):
        img_path = pre_dir+ '_' + 'green' + '.png'
        print('img_path ', img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        if self.preproc is not None:
            img = self.preproc(img) 
            
    def __getitem__(self, index):  
         img_name = self.test_image_merge_list[index]
         img = self.get_gray_image(self.test_image_dir + img_name)
         return img
         
    def __len__(self):
        return len(self.test_image_merge_list)