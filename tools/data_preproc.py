#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:23:15 2018

@author: zl
"""
import torch
import cv2
import numpy as np
import random
from torchvision import transforms

class Data_Preproc(object):
    def __init__(self, resize = [300, 300]):
        self.resize = resize
        
    def __call__(self, image):
        image = self.preproc(image)
        return image
       # return torch.from_numpy(image)
        
    def preproc(self, image):
        #print('insize ', insize)
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (self.resize[0], self.resize[1]),interpolation=interp_method)
      #  image = image.astype(np.float32)
    #    image -= (103.94, 116.78, 123.68, 100.5)
        transform = transforms.Compose([
             #   transforms.ToPILImage(),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
              #  transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                ])
        image = transform(image)
        return image
      #  return image.transpose(2, 0, 1)