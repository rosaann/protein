#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:49:30 2018

@author: zl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.other_layers.l2norm import L2Norm
import numpy as np
class VGG_SIM_Z(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG_SIM_Z, self).__init__()
        layers = []
        in_channels = 4
    #    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    #        512, 512, 512]
        
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d]
        #    layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        conv2d = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 128
            
        conv2d = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 128
            
 #       layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
  #      conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
  #      if batch_norm:
  #          layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
  #      else:
  #          layers += [conv2d, nn.ReLU(inplace=True)]
  #      in_channels = 256
            
   #     conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
  #      if batch_norm:
  #          layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
  #      else:
  #          layers += [conv2d, nn.ReLU(inplace=True)]
  #      in_channels = 256
            
  #      conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
  #      if batch_norm:
  #          layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
 #       else:
 #           layers += [conv2d, nn.ReLU(inplace=True)]
 #       in_channels = 256
        
 #       layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
 #       conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
 #       if batch_norm:
 #           layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
 #       else:
 #           layers += [conv2d, nn.ReLU(inplace=True)]
 #       in_channels = 512
        
 #       conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
 #       if batch_norm:
 #           layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
 #       else:
 #           layers += [conv2d, nn.ReLU(inplace=True)]
 #       in_channels = 512
        
 #       conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
 #       if batch_norm:
 #           layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
 #       else:
 #           layers += [conv2d, nn.ReLU(inplace=True)]
 #       in_channels = 512
        
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        self.line = nn.Linear(720000 , 28)
        self.sigmoid = nn.Sigmoid()
        self.base = nn.ModuleList(layers)
       # self.out_layers = nn.ModuleList(out_layers)
        
    def forward(self, imgs, phase='eval'):
    #    num_img = len(imgs)
        x = imgs
        for k in range(len(self.base)):
           # print('k ', k)
            x = self.base[k](x)
        x = x.view(x.size(0), -1)
     #   if phase == 'eval':
     #      self.line = nn.Linear(len(x[0]) , 28).cuda()
        x = self.line(x)
      #  print('x ', x)
        x = self.sigmoid(x)
      #  print('si ', x)
        return x