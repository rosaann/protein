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
import random 
from config import Config

class ProteinDataSet(data.Dataset):
    def __init__(self,preproc=None,train_class = 0, base_path='../train/', csv_path='../train.csv',group_class_num = 4,phase='train'):
        self.df = pd.read_csv(csv_path)
        self.preproc = preproc
        self.base_path = base_path
        self.train_class = train_class
        self.img_name_tails = [ 'red', 'green', 'blue', 'yellow']
    #    self.img_name_tails = [ 'red', 'green', 'blue']
        self.idx = 0
        self.group_class_num = group_class_num
        self.current_train_group_idx = 0
        self.phase = phase
        self.config = Config()
    #    self.genImgIdListForEveryClass()
    #    self.genTrainImgList()
    
    #    self.genDGroup()
        self.genChecklist()
    def genChecklist(self):
        self.check_id_list = []
        id_to_check = self.config.v('check_id_list')
        for img_id in range(self.df.shape[0]):
            target = self.df.get_value(img_id, 'Target')
            img_tars = [int(tar) for tar in target.split(' ')]
            ifHasTarInCheckList = False
            for thisTar in img_tars:
                if thisTar in id_to_check:
                    ifHasTarInCheckList = True
                    break
            if ifHasTarInCheckList == True:
                    self.check_id_list.append((self.df.get_value(img_id, 'Id'), target))
    def genDGroup(self):
        self.group_list = []
        class_ids = [str(i) for i in range(28)]
        tar_all_list = self.config.v('group_id_list')
        for g_d in range(len(tar_all_list)):
            id_to_check = tar_all_list[g_d]
            group = []
            eva_rate = 0.8
            data_idx_list = range(self.df.shape[0])[:int(self.df.shape[0] * eva_rate)]
            if self.phase == 'eval':
                data_idx_list = range(self.df.shape[0])[int(self.df.shape[0] * eva_rate) :]
         #   print('id_to_check ', id_to_check)
            for img_id in data_idx_list:
                target = self.df.get_value(img_id, 'Target')
                ifFind = False
                for class_id in id_to_check:
                    for tt in target.split(' '):
                        if class_id == int (tt):
                            ifFind = True
                if ifFind == True:
                    group.append((self.df.get_value(img_id, 'Id'), target))
                 #   print('find ', target)
                #else:
                  #  print('not find ',target)
            print('group ',g_d, ' len ', len(group))
            self.group_list.append(group)
    def setTrain_group_idx(self, group_idx):
        self.current_train_group_idx = group_idx
        
    def genTrainImgList(self):
        self.train_imgid_list = []
        por_list = self.class_img_id_list[self.train_class]
        for img_id in por_list:
           self.train_imgid_list.append((img_id, 1)) 
        print('por len ', len(self.train_imgid_list))   
        por_len = len(por_list)
        neg_len_per_class = int(por_len / 28)
        for i , class_img_id_list in enumerate(self.class_img_id_list ):
            if i != self.train_class:
               neg_imgs_id_list = random.sample(class_img_id_list, neg_len_per_class) 
               for img_id in neg_imgs_id_list:
                   self.train_imgid_list.append((img_id, 0)) 
                   
        random.shuffle(self.train_imgid_list)
        print('train len ', len(self.train_imgid_list))   

    def genImgIdListForEveryClass(self):
        class_ids = [str(i) for i in range(28)]
        self.class_img_id_list = []
        for class_id in class_ids:
            img_id_list = []
            for img_id in range(self.df.shape[0]):
                target = self.df.get_value(img_id, 'Target')
                if target.find(class_id):
                    img_id_list.append(self.df.get_value(img_id, 'Id'))
            print('class ', class_id, "len ", len(img_id_list))        
            self.class_img_id_list.append(img_id_list)   
            
    def __getitem__(self, index):   
        img_id, target = self.check_id_list[index]
        
        img_path = self.base_path + img_id + '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        
        if self.preproc is not None:
            img = self.preproc(img)
      #  print('gd ', self.current_train_group_idx, ' tar ', target)
        return img, target
    def __getitem__merge(self, index):
        img_id, target = self.check_id_list[index]
        
        imgs = []
        
        for tail in self.img_name_tails:
            img_path = self.base_path + img_id + '_' + tail + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        img_merg = cv2.merge(imgs)
        
        if self.preproc is not None:
            img_merg = self.preproc(img_merg)
      #  print('gd ', self.current_train_group_idx, ' tar ', target)
        return img_merg, target
        
    def __getitem__old(self, index):
        img_id = self.df.get_value(index, 'Id')
        target = self.df.get_value(index, 'Target')
        
        imgs = []
        
        for tail in self.img_name_tails:
            img_path = self.base_path + img_id + '_' + tail + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        img_merg = cv2.merge(imgs)
    #    if target.find( str(self.idx)) >= 0:
    #        cv2.imwrite(os.path.join('./','{}_.png'.format(self.idx)), img_merg)
    #        self.idx += 1
        
        if self.preproc is not None:
            img_merg = self.preproc(img_merg)
       
     #   tar_list = []
     #   print('targets in getitem ', target)
      #  targets = target.split(' ')
      #  for tar in targets:
      #      tar_list.append(tar)
        
    #    print('tar_list in getitem ', tar_list)
        
        return img_merg, target
        
    def __len__(self):
        return len(self.check_id_list)
      #  return self.df.shape[0]
