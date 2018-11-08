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
        conf_t = torch.LongTensor(num_img, self.num_classes, 2)

        # match priors (default boxes) and ground truth boxes
        for i, img_targets in enumerate( targets):
            labels = np.zeros((self.num_classes))
            for target in img_targets:
                labels[target] = 1
            conf_t[i] = labels
            
            
            if self.use_gpu:
                conf_t = conf_t.cuda()

        
        # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, 2)
            conf_t_v = conf_t.view(-1,1)


        loss_c = F.cross_entropy(batch_conf, conf_t_v, size_average=False)

        return loss_c
