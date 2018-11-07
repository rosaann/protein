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
class SSD_Z(nn.Module):
    def __init__(self, base, extras, conflist, feature_layer, num_classes):
        super(SSD_Z, self).__init__()
        self.num_classes = num_classes
        self.num_per_con = 2
        # SSD network
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

        for conf in conflist:
            self.conflist.append( nn.ModuleList(conf))
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]
        
        
    def forward(self, x, phase='eval'):
        sources,  conf = [list() for _ in range(2)]
        for k in range(len(self.base)):
           # print('k ', k)
            x = self.base[k](x)
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
        output_list = []
        for conf in self.conflist:
            for (x, c) in zip(sources, conf):
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        
            if phase == 'eval':
                output = (
                    self.softmax(conf.view(-1, self.num_per_con)),  # conf preds
                )
                output_list.append(output)
            else:
                output = (
                    conf.view(conf.size(0), -1, self.num_per_con),
                #conf
                )
                output_list.append(output)
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
    return SSD(base_, extras_, conflist, feature_layer, num_classes)