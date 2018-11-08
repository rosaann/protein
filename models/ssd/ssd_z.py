#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:56:20 2018

@author: zl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.other_layers.l2norm import L2Norm
import numpy as np
class SSD_Z(nn.Module):
    def __init__(self, base, extras, conflist, feature_layer, num_classes):
        super(SSD_Z, self).__init__()
        self.num_classes = num_classes
        self.num_per_con = 2
        # SSD network
        self.basee = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

        self.conflist = []
        self.conf_0 = nn.ModuleList(conflist[0])      
        self.conf_1 = nn.ModuleList(conflist[1])
        self.conf_2 = nn.ModuleList(conflist[2])
        self.conf_3 = nn.ModuleList(conflist[3])
        self.conf_4 = nn.ModuleList(conflist[4])
        self.conf_5 = nn.ModuleList(conflist[5])
        self.conf_6 = nn.ModuleList(conflist[6])
        self.conf_7 = nn.ModuleList(conflist[7])
        self.conf_8 = nn.ModuleList(conflist[8])
        self.conf_9 = nn.ModuleList(conflist[9])
        self.conf_10 = nn.ModuleList(conflist[10])
        self.conf_11 = nn.ModuleList(conflist[11])
        self.conf_12 = nn.ModuleList(conflist[12])
        self.conf_13 = nn.ModuleList(conflist[13])
        self.conf_14 = nn.ModuleList(conflist[14])
        self.conf_15 = nn.ModuleList(conflist[15])
        self.conf_16 = nn.ModuleList(conflist[16])
        self.conf_17 = nn.ModuleList(conflist[17])
        self.conf_18 = nn.ModuleList(conflist[18])
        self.conf_19 = nn.ModuleList(conflist[19])
        self.conf_20 = nn.ModuleList(conflist[20])
        self.conf_21 = nn.ModuleList(conflist[21])
        self.conf_22 = nn.ModuleList(conflist[22])
        self.conf_23 = nn.ModuleList(conflist[23])
        self.conf_24 = nn.ModuleList(conflist[24])
        self.conf_25 = nn.ModuleList(conflist[25])
        self.conf_26 = nn.ModuleList(conflist[26])
        self.conf_27 = nn.ModuleList(conflist[27])
        self.conflist.append(self.conf_0)
        self.conflist.append(self.conf_1)
        self.conflist.append(self.conf_2)
        self.conflist.append(self.conf_3)
        self.conflist.append(self.conf_4)
        self.conflist.append(self.conf_5)
        self.conflist.append(self.conf_6)
        self.conflist.append(self.conf_7)
        self.conflist.append(self.conf_8)
        self.conflist.append(self.conf_9)
        self.conflist.append(self.conf_10)
        self.conflist.append(self.conf_11)
        self.conflist.append(self.conf_12)
        self.conflist.append(self.conf_13)
        self.conflist.append(self.conf_14)
        self.conflist.append(self.conf_15)
        self.conflist.append(self.conf_16)
        self.conflist.append(self.conf_17)
        self.conflist.append(self.conf_18)
        self.conflist.append(self.conf_19)
        self.conflist.append(self.conf_20)
        self.conflist.append(self.conf_21)
        self.conflist.append(self.conf_22)
        self.conflist.append(self.conf_23)
        self.conflist.append(self.conf_24)
        self.conflist.append(self.conf_25)
        self.conflist.append(self.conf_26)
        self.conflist.append(self.conf_27)
        
     #   for conf in conflist:
     #       self.conflist.append( nn.ModuleList(conf))
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]
        
        
    def forward(self, x, phase='eval'):
        sources = list() 
        for k in range(len(self.basee)):
           # print('k ', k)
            x = self.basee[k](x)
            if k in self.feature_layer:
              #  print('source append herer k ', k)
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            # TODO:maybe donot needs the relu here
            x = F.relu(v(x), inplace=True)
            # TODO:lite is different in here, should be changed
            if k % 2 == 1:
           #     print('source append herer 2-- k ', k)
                sources.append(x)
        
        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        num_img = x.shape[0]
        output_list = torch.Tensor(num_img, self.num_classes, 1)
        
        print('source ',len( sources))
        for i in range(num_img):
          # for every image
          for conf_net in self.conflist:
            #check every class
            conf = list()
            for x, c in zip(sources, conf_net):
                if i == 0:
                   print('x ',x.shape)
                conf.append(c(x[i].unsqueeze(0)).permute(0, 2, 3, 1).contiguous())

            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            if i == 0:
                print('conf ', conf.shape)
           # if phase == 'eval':
            output = self.softmax(conf.view(-1, self.num_per_con))  # conf preds
            if i == 0:
              print('output ', output.shape)
            #print('output ', output.shape)
            output_list[i]= output
          #  else:
          #      output = conf.view(conf.size(0), -1, self.num_per_con),
                
         #       output_list.append(output)
            #print('out put shape', loc.shape)
        return output_list
    
    
def add_extras(base, feature_layer, mbox, num_classes, num_per_con=2):
    extra_layers = []
    conf_layers = []
    in_channels = None
    
    conf_layers_list = []
    for i in range(num_classes):
        conf_layers_list.append([])
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'S':
            extra_layers += [
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
            in_channels = depth
        elif layer == '':
            extra_layers += [
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3)  ]
            in_channels = depth
        else:
            in_channels = depth
        
        for conf_layers in conf_layers_list:
            conf_layers += [nn.Conv2d(in_channels, box * num_per_con, kernel_size=3, padding=1)]
    
    
        
    return base, extra_layers, conf_layers_list

def build_ssd(base, feature_layer, mbox, num_classes):
    """Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    base_, extras_, conflist = add_extras(base(), feature_layer, mbox, num_classes)
    return SSD_Z(base_, extras_, conflist, feature_layer, num_classes)