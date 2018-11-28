#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:24:37 2018

@author: zl
"""
from tools.protein_dataset import ProteinDataSet
from tools.data_preproc import Data_Preproc
import torch.utils.data as data
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import  metrics

def xgboost_train():
    dataset = ProteinDataSet(None,csv_path='../train.csv', phase='train')
   # config = Config()
    train_loader = data.DataLoader(dataset,int( 31072), num_workers= 8,
                                               shuffle=True, pin_memory=True)
   # batch_iterator = iter(train_loader)
    index = 0
    for images, targets in train_loader:
  #  images, targets = train_loader[0]
        nsamples, nx, ny = images.shape
        images = images.reshape((nsamples,nx*ny))
        print('len ',len(images))
        tr_hot = []
        for img_targets in targets:
            targets = img_targets.split(' ')
            tar_t = np.zeros(28)
            for tar in targets:
                tar_t[int(tar)] = 1
            tr_hot.append(tar_t)  
            
        param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:logistic' }
        num_round = 2
        train_end = int(len(images) * 0.8)
        bst = xgb.train(param, images[:train_end], num_round)
        # make prediction
        preds = bst.predict(images[train_end : ])
        print('i ' , index)
        print ("AUC Score (val): " , metrics.roc_auc_score(tr_hot[train_end:], preds))
        print((tr_hot[train_end:] == preds).mean()) # 打印精度最大的那一个三元组 print(max(results, key=lambda x: x[2]))
        index += 1
        
xgboost_train()