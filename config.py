#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:37:23 2018

@author: zl
"""

class Config(object):
    def __init__(self):
       self.data = {}
       self.data['batch_size'] = 8
       self.data['batch_size_eval'] = 5
       self.data['num_classes'] = 28
       self.data['image_size'] = [300, 300]
       self.data['learn_rate'] = 0.3
       self.data['momentum'] = 0.9
       self.data['momentum_2'] = 0.99
       self.data['weight_decay'] = 0.0001
       self.data['eps'] = 1e-8
       self.data['lr_gamma'] = 0.98
       self.data['lr_steps'] = [1]
       self.data['epoches'] = 20
       self.data['out_dir'] = 'outs/'
       self.data['save_per'] = 3
       #self.data['group_id_list']=[[27, 15, 10,9], [8,20,17,24],[26,16,13,12],[22,18,6,14],[11,1,19,3],[4,5,7,23],[2,21,25,0]]
      # self.data['check_id_list'] = [24,26,16,13,12,22,18,6,14,11,1,19,3,4,5,7,23,2,21,25,0][8, 9, 10, 15, 16, 17, 27]
       self.data['check_id_list'] = [24,26,16,13,12,22,18,6,14,11,1,19,3,4,5,7,23,2,21,25,0]
       self.data['xgb_len'] = 16
       
    def v(self, key):
        return self.data[key]