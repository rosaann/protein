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
from sklearn.metrics import f1_score
def xgboost_train():
    dataset = ProteinDataSet(None,csv_path='../train.csv', phase='train')
   # config = Config()
    train_loader = data.DataLoader(dataset,int( 31072/300), num_workers= 8,
                                               shuffle=True, pin_memory=False)
   # batch_iterator = iter(train_loader)
    index = 0
    for images, targets in train_loader:
  #  images, targets = train_loader[0]
        nsamples, nx, ny = images.shape
        images = images.reshape((nsamples,nx*ny))
        print('len ',len(images))
        
        tr_hot = []
        for ti, img_targets in enumerate( targets):
            targets_t = img_targets.split(' ')
            tar_t = np.zeros((28, 1))
            for tar in targets_t:
                tar_t[int(tar)] = 1
          #  print('tar_t ', tar_t.shape)
          #  tr_hot.append(tar_t)
            tr_hot.append(int(targets_t[0]))
            
        param = {'max_depth':20,'num_class':28,  'eta':1, 'silent':1, 'objective':'multi:softprob', 'gpu_id':0, 'max_bin':16, 'seed':10 }
        #param = {'max_depth':20,'num_class':28,  'eta':1, 'silent':1, 'objective':'multi:softprob', 'gpu_id':0, 'max_bin':16,'tree_method': 'gpu_hist', 'seed':10 }

        param['eval_metric'] = ['auc'] 
      #  param['nthread'] = 4
        num_round = 2
        train_end = int(len(images) * 0.8)
        dtrain = xgb.DMatrix(images[:train_end], tr_hot[ : train_end] )
        dtest = xgb.DMatrix(images[train_end : ], tr_hot[train_end : ] )
      #  print('tr_hot ', tr_hot.shape)
        num_round = 10
        evallist  = [(dtest,'eval'), (dtrain,'train')]

        bst = xgb.train(param, dtrain , num_round)
        # make prediction
        preds = bst.predict(dtest)
        print('i ' , index)
        print ("AUC Score (val): " , metrics.roc_auc_score(tr_hot[train_end:], preds))
        print((tr_hot[train_end:] == preds).mean()) # 打印精度最大的那一个三元组 print(max(results, key=lambda x: x[2]))
        index += 1
        
xgboost_train()