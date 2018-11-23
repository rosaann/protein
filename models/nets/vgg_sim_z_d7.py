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
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
            
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(64)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = 64
          
        
    #    conv2d = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(128)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
   #     in_channels = 128
            
    #    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
    #    conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 256
            
    #    conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 256
            
    #    conv2d = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 256
        
    #    layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
    #    conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 512
        
    #    conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
    #    if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 512
        
    #    conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
     #   if batch_norm:
    #        layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    #    else:
    #        layers += [conv2d, nn.ReLU(inplace=True)]
    #    in_channels = 512
        
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        self.line = nn.Linear(360000 , 4)
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
        x = self.line(x)
      #  print('x ', x)
        x = self.sigmoid(x)
      #  print('si ', x)
        return x
    
class VGG_SIM_Z_D7(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG_SIM_Z_D7, self).__init__()
        self.model_num = 7
        self.model_list = []
     #   for i in range(self.model_num):
        self.model_1 = VGG_SIM_Z()
        self.model_list.append(self.model_1)
        self.model_2 = VGG_SIM_Z()
        self.model_list.append(self.model_2)
        self.model_3 = VGG_SIM_Z()
        self.model_list.append(self.model_3)
        self.model_4 = VGG_SIM_Z()
        self.model_list.append(self.model_4)
        self.model_5 = VGG_SIM_Z()
        self.model_list.append(self.model_5)
        self.model_6 = VGG_SIM_Z()
        self.model_list.append(self.model_6)
        self.model_7 = VGG_SIM_Z()
        self.model_list.append(self.model_7)

    def forward(self, imgs, phase='eval', model_idx = 0):
        num_img = len(imgs)
        if phase == 'eval':
            output_list = torch.Tensor(28,num_img)
            for i_m in range(self.model_num):
                model = self.model_list[i_m]
                x = imgs
                for k in range(len(model.base)):
                    # print('k ', k)
                    x = model.base[k](x)
                x = x.view(x.size(0), -1)
                x = model.line(x)
                #  print('x ', x)
                x = model.sigmoid(x)#.unsqueeze(0)
                x = x.permute(1, 0)
            #    print('xi ', x.shape)

                for di, dx in enumerate( x):
            #        print('di ', dx.shape)
                    output_list[i_m * 4 + di]=dx
            output_list = output_list.permute(1, 0)
            return output_list
    
        if phase == 'train':
            torch.set_printoptions(precision=10)
            model = self.model_list[model_idx]
            x = imgs
            for k in range(len(model.base)):
             #   print('k ', model.base[k])
                x = model.base[k](x)
            x = x.view(x.size(0), -1)
            x = model.line(x)
            #  print('x ', x)
            x = model.sigmoid(x)
            print('x ', x.data.numpy())
            return x
            
            
            
            
            
            
                
