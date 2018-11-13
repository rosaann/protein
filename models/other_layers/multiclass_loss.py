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

    def forward(self, predictions, targets):
        
        conf_data = predictions
       # print('loc_data ',loc_data.shape, ' conf_data ', conf_data.shape, 'targets ',len(targets), 'num ', num)
        num_img = len(conf_data)
        print('num_img ', num_img)
        conf_t = torch.Tensor(num_img, self.num_classes, 1)

        # match priors (default boxes) and ground truth boxes
        for i, img_targets in enumerate( targets):
            labels = np.zeros((self.num_classes, 1))
            print('img_targets ', img_targets)
            for target in img_targets:
              #  print('tar ', target)
                labels[int(target)][0] = 1.0
            conf_t[i] = torch.from_numpy( labels).type(torch.cuda.FloatTensor)
            
            
            if self.use_gpu:
                conf_t = conf_t.cuda()
                conf_data = conf_data.cuda()

        
        # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, 1)
            conf_t_v = conf_t.view(-1,1)

        print('batch_conf ',batch_conf)
        print('conf_t_v', conf_t_v)
        loss_c = F.mse_loss(conf_t_v,batch_conf,  size_average=False)
        print('loss_c ', loss_c)
        return loss_c
