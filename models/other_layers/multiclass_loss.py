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
from lib.utils.box_utils import match, log_sum_exp
import numpy as np
from config import Config

class MultiClassLoss(nn.Module):
    def __init__(self, cfg, priors, use_gpu=True):
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
       
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        labels = np.zeros((num_classes))
        for target in targets:
            labels[target] = 1
        for i in num_classes:
            conf_t = torch.LongTensor(num, 2)
            conf_t = labels[i]
            
            
            if self.use_gpu:
                conf_t = conf_t.cuda()
        # wrap targets
            conf_t = Variable(conf_t,requires_grad=False)

        # num_pos = pos.sum()
      #  print('conf_t size ', conf_t.shape)
        
        # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, 2)
       # batch_conf = conf_data.view(-1, num_priors) #zl
            conf_t_v = conf_t.view(-1,1)
        #conf_t_v = conf_t.view(-1,num_priors)#zl
        
      #  print('batch_conf size ', batch_conf.shape)
      #  print('conf_t_v size ', conf_t_v.shape)

            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t_v)
        #loss_c = log_sum_exp (batch_conf) - batch_conf.gather (0, conf_t.view (-1, 1))#zz 
        
      #  print('loss_c ', loss_c.shape)
      #  print('pos ', pos.shape)
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        ###
        
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        ###
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
       # pos_idx = pos.expand_as(conf_data) zl
       # neg_idx = neg.expand_as(conf_data) zl
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c
