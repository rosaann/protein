#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:34:21 2018

@author: zl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.other_layers.l2norm import L2Norm
import numpy as np
from models.nets.resnet import resnet_18

class Res_SIM_Z(nn.Module):
    def __init__(self, batch_norm=True):
        super(Res_SIM_Z, self).__init__()
        self.line = nn.Linear(92416 , 28)
        self.sigmoid = nn.Sigmoid()
        layers = resnet_18()
        self.base = nn.ModuleList(layers)
        
    def forward(self, imgs, phase='eval'):
    #    num_img = len(imgs)
        x = imgs

        for k in range(len(self.base)):
           # print('k ', k)
            x = self.base[k](x)
        x = x.view(x.size(0), -1)
        x = self.line(x)
      #  print('x ', x)
        x = self.sigmoid(x)
      #  print('si ', x)
        return x