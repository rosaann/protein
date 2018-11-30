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
import cv2
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def find_small_num_class_ids():
    type_class = [8, 9, 10, 15, 16,17, 27]
  #  other_class = []
    df = pd.read_csv('../train.csv')
    id_list = []
    for i, row in df.iterrows():
        targets = row['Target'].split(' ')
        targets_t = [int (tthis) for tthis in targets]
      #  tar_in_typelist = []
        for t in targets_t:
            if t in type_class:
                id_list.append((row['Id'], targets))
                break
            
    print('total ', df.shape[0], 'small ', len(id_list))
    return id_list

def xgboost_train():
    id_list = find_small_num_class_ids()
    
    base_path = '../train/'
    data_img_list = []
    data_tar_list = []
    log_idx = 0
    for img_id, targets in id_list:
        img_path = base_path + img_id + '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            
        tar_t = np.zeros((28))
        for tar in targets:
            tar_t[int(tar)] = 1
        data_img_list.append(img)
        data_tar_list.append(targets)
        if log_idx < 10:
            print(tar_t)
            log_idx += 1
        
    data_img_list = np.array(data_img_list)
    data_tar_list = np.array(data_tar_list)

    nsamples, nx, ny = data_img_list.shape
    data_img_list = data_img_list.reshape((nsamples,nx*ny))
    print('img shape', data_img_list.shape)   
    
    Y_enc = MultiLabelBinarizer().fit_transform(data_tar_list)
    for i, y_en in enumerate(Y_enc):
        if i < 10:
            print(y_en)
        else :
            break
    train_end = int(len(data_img_list) * 0.8)
    x = xgb.XGBClassifier(learning_rate=0.05, n_estimators=10,objective='binary:logistic', seed=1)  
    clf = OneVsRestClassifier(x)
    clf.fit(data_img_list[: train_end], Y_enc[:train_end])
    #clf.fit(data_img_list[: train_end][0], data_tar_list[: train_end][1])
    y_p_x = clf.predict_proba(data_img_list[train_end : ])
    
    log_idx = 0
    for y in y_p_x:
        if log_idx < 10:
            print('pre ', y)
            log_idx += 1
        else :
            break
        
    print('f1 ',f1_score(y_p_x, data_tar_list[train_end : ], average = "macro"))
    print('acc ', metrics.accuracy_score(y_p_x, data_tar_list[train_end : ]))
        
def xgboost_train_old():
    dataset = ProteinDataSet(None,csv_path='../train.csv', phase='train')
   # config = Config()
    train_loader = data.DataLoader(dataset,int( 31072/2), num_workers= 8,
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
            
        param = {'max_depth':20,'num_class':28,  'eta':1, 'silent':1, 'objective':'multi:softprob','nthread':8, 'scale_pos_weight':1, 'gpu_id':0, 'max_bin':16, 'seed':10 }
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
        
        print((tr_hot[train_end:] == preds).mean()) # 打印精度最大的那一个三元组 print(max(results, key=lambda x: x[2]))
        print ("Score (val): " , bst.best_score)
        index += 1
        
xgboost_train()