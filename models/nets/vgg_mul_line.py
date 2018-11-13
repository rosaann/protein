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
class VGG_MUL_LINE(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG_MUL_LINE, self).__init__()
        layers = []
        in_channels = 4
    #    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    #        512, 512, 512]
        
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=5, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
   #     layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=7, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
   #     conv2d = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
   #     if batch_norm:
   #         layers += [conv2d, nn.BatchNorm2d(128)]
   #     else:
   #         layers += [conv2d, nn.ReLU(inplace=True)]
  #      in_channels = 128
            
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

        self.line = nn.Linear(1382976 , 28)
        self.sigmoid = nn.Sigmoid()
        self.base = nn.ModuleList(layers)
        self.addLeafLayers()
       # self.out_layers = nn.ModuleList(out_layers)
    def addLeafLayers(self):
        self.leafLayers = []
        self.leafLayer0 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer1 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer2 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer3 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer4 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer5 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer6 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer7 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer8 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer9 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer10 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer11 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer12 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer13 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer14 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer15 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer16 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer17 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer18 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer19 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer20 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer21 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer22 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer23 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer24 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer25 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer26 = nn.ModuleList(self.addLeaf(64))
        self.leafLayer27 = nn.ModuleList(self.addLeaf(64))
        
        self.leafLayers.append(self.leafLayer0)
        self.leafLayers.append(self.leafLayer1)
        self.leafLayers.append(self.leafLayer2)
        self.leafLayers.append(self.leafLayer3)
        self.leafLayers.append(self.leafLayer4)
        self.leafLayers.append(self.leafLayer5)
        self.leafLayers.append(self.leafLayer6)
        self.leafLayers.append(self.leafLayer7)
        self.leafLayers.append(self.leafLayer8)
        self.leafLayers.append(self.leafLayer9)
        self.leafLayers.append(self.leafLayer10)
        self.leafLayers.append(self.leafLayer11)
        self.leafLayers.append(self.leafLayer12)
        self.leafLayers.append(self.leafLayer13)
        self.leafLayers.append(self.leafLayer14)
        self.leafLayers.append(self.leafLayer15)
        self.leafLayers.append(self.leafLayer16)
        self.leafLayers.append(self.leafLayer17)
        self.leafLayers.append(self.leafLayer18)
        self.leafLayers.append(self.leafLayer19)
        self.leafLayers.append(self.leafLayer20)
        self.leafLayers.append(self.leafLayer21)
        self.leafLayers.append(self.leafLayer22)
        self.leafLayers.append(self.leafLayer23)
        self.leafLayers.append(self.leafLayer24)
        self.leafLayers.append(self.leafLayer25)
        self.leafLayers.append(self.leafLayer26)
        self.leafLayers.append(self.leafLayer27)
            
    def addLeaf(self, in_channels, batch_norm = True):
        layers  = []
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Linear(341056 , 1)]
        layers += [nn.Sigmoid()]
        
        return layers
        
    def forward(self, imgs, phase='eval'):
        num_img = len(imgs)
        x = imgs
        for k in range(len(self.base)):
           # print('k ', k)
            x = self.base[k](x)
        
        output_list = torch.Tensor(28, num_img, 1)  
        for k in range(len(self.leafLayers)):
            out = x
            class_leaf_layers = self.leafLayers[k]
            for h in range(len(class_leaf_layers)):
                out = class_leaf_layers[h](out)
                if h == (len(class_leaf_layers) - 3):
                    out = out.view(out.size(0), -1)
            print('out ', out)
            output_list[k] = out
        output_list = output_list.permute(1, 0, 2)
        print('output_list ', output_list)
        return output_list