#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:16:32 2018

@author: zl
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from tools.protein_dataset import ProteinDataSet
from tools.data_preproc import Data_Preproc
import torch.utils.data as data
from config import Config

def randomForest():
  preproc = Data_Preproc()
  dataset = ProteinDataSet(preproc,csv_path='../train.csv', phase='train')
   # config = Config()
  train_loader = data.DataLoader(dataset, 31072, num_workers= 8,
                                               shuffle=True, pin_memory=True)
   # batch_iterator = iter(train_loader)
  for images, targets in train_loader:
  #  images, targets = train_loader[0]
    print('len ',len(images))
    tr_hot = []
    for img_targets in targets:
        targets = img_targets.split(' ')
        tar_t = np.zeros(28)
        for tar in targets:
            tar_t[int(tar)] = 1
        tr_hot.append(tar_t)  
            
            
    results = []
    sample_leaf_options = list(range(5, 50, 500))
    n_estimators_options = list(range(1, 1000, 5))
    
    train_end = 31072 * 0.8
    for leaf_size in sample_leaf_options: 
        for n_estimators_size in n_estimators_options: 
            alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50) 
            alg.fit(images[: train_end], tr_hot[:train_end]) 
            predict = alg.predict(images[train_end : ]) # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度 
            results.append((leaf_size, n_estimators_size, (tr_hot[train_end:] == predict).mean())) # 真实结果和预测结果进行比较，计算准确率 
            print((tr_hot[train_end:] == predict).mean()) # 打印精度最大的那一个三元组 print(max(results, key=lambda x: x[2]))
            


randomForest()
