#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:48:16 2018

@author: zl
"""
from models.nets import vgg
from models.ssd.ssd_z import build_ssd
import torch
from config import Config

def _forward_features_size(model, img_size):
    model.eval()
    x = torch.rand(1, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True).cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3]) for o in feature_maps]

def create_model_on_vgg():

    base = vgg.vgg16
    aspect_ratios = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in aspect_ratios]  
#    print('num_box ',number_box)
    feature_layer = [[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]
    model = build_ssd(base=base, feature_layer=feature_layer, mbox=number_box, num_classes= Config().v('num_classes'))
    #
  #  conf = Config()
  #  feature_maps = _forward_features_size(model, conf.v('image_size'))
  #  print('==>Feature map size:')
 #   print(feature_maps)


    return model