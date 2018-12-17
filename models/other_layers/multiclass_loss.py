#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:38:51 2018

@author: zl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import Config
from tkinter import _flatten

class MultiClassLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(MultiClassLoss, self).__init__()
        self.use_gpu = use_gpu
        self.config = Config()
        self.num_classes = self.config.v('num_classes')
        self.background_label =  0
        self.negpos_ratio = 3
        self.threshold = 0.5
        self.unmatched_threshold = 0.5
        self.variance = [0.1, 0.2]
        self.cri = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        conf_data = predictions

        
        conf_t = torch.from_numpy( targets).type(torch.cuda.FloatTensor)
        #conf_t = torch.from_numpy( targets)
        if self.use_gpu:
            conf_t = conf_t.cuda()
            conf_data = conf_data.cuda()

        
        # Compute max conf across batch for hard negative mining
       # batch_conf = conf_data.view(-1, 1)
       # conf_t_v = conf_t.view(-1,1)

     #   print('batch_conf ',batch_conf.shape, ' ', batch_conf)
     #   print('conf_t_v', conf_t_v.shape)
     #   loss_c = F.mse_loss(conf_t_v,batch_conf,  size_average=False)
        loss_c =self.cri(conf_data, conf_t)
     #   print('loss_c ', loss_c)
        return loss_c
        
    def forward_df(self, predictions, targets):
      #  print('predictions ',predictions)
        conf_data = predictions
       # print('loc_data ',loc_data.shape, ' conf_data ', conf_data.shape, 'targets ',len(targets), 'num ', num)
        num_img = len(conf_data)
     #   print('num_img ', num_img)
        conf_t = torch.Tensor(num_img, self.num_classes, 1)
        
        # match priors (default boxes) and ground truth boxes
     #   print('tttt ', targets)
        tr_tar_list = _flatten(self.config.v('group_id_list')) 
      #  print('tr_t ', tr_tar_list)
        for i, img_targets in enumerate( targets):
            tar_list = []
            targets = img_targets.split(' ')
            for tar in targets:
                tar_list.append(int(tar))
            labels = np.zeros((self.num_classes, 1))
        #    print('img_targets ', tar_list)
            for target in tar_list:
              #  print('tar ', target)
                tr_tar_idx = tr_tar_list.index(target)
                labels[tr_tar_idx][0] = 1.0
      #          print('value ', target, ' index ', tr_tar_idx)
             #   labels[int(target)][0] = 1.0
            conf_t[i] = torch.from_numpy( labels).type(torch.cuda.FloatTensor)
            
            
            if self.use_gpu:
                conf_t = conf_t.cuda()
                conf_data = conf_data.cuda()

        
        # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, 1)
            conf_t_v = conf_t.view(-1,1)

    #    print('batch_conf ',batch_conf)
    #    print('conf_t_v', conf_t_v)
        loss_c = F.mse_loss(conf_t_v,batch_conf,  size_average=False)
     #   print('loss_c ', loss_c)
        return loss_c
    
    def forward_pre(self, predictions, targets):
      #  print('predictions ',predictions)
        conf_data = predictions
       # print('loc_data ',loc_data.shape, ' conf_data ', conf_data.shape, 'targets ',len(targets), 'num ', num)
        num_img = len(conf_data)
     #   print('num_img ', num_img)
        # match priors (default boxes) and ground truth boxes
     #   print('tttt ', targets)
     
        tr_tar_list = self.config.v('check_id_list')
        class_num = int(len(tr_tar_list))
        conf_t = torch.Tensor(num_img, class_num, 1)
        for i, img_targets in enumerate( targets):
            tar_list = []
            targets = img_targets.split(' ')
            for tar in targets:
                tar_list.append(int(tar))
            labels = np.zeros((class_num, 1))
        #    print('img_targets ', tar_list)
            for target in tar_list:
             #   print('tar ', target)
                if target in tr_tar_list:
                    ti = tr_tar_list.index(target)
                    labels[ti][0] = 1.0
         #   print('label ', labels)
            conf_t[i] = torch.from_numpy( labels).type(torch.cuda.FloatTensor)
            
            
        if self.use_gpu:
            conf_t = conf_t.cuda()
            conf_data = conf_data.cuda()

        
        # Compute max conf across batch for hard negative mining
            
        batch_conf = conf_data.view(-1, 1)
        conf_t_v = conf_t.view(-1,1)
   #     print('batch_conf ',batch_conf)
   #     print('conf_t_v', conf_t_v)
            
        loss_c = F.mse_loss(conf_t_v,batch_conf,  size_average=False)
     #   print('loss_c ', loss_c)
        return loss_c
