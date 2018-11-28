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
from joblib import dump, load

def randomForest():
  preproc = Data_Preproc()
  dataset = ProteinDataSet(None,csv_path='../train.csv', phase='train')
   # config = Config()
  train_loader = data.DataLoader(dataset,int( 31072 / 1), num_workers= 0,
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
            
            
    results = []
    sample_leaf_options = list(range(5, 50, 500))
    n_estimators_options = list(range(1, 1000, 5))
    
    train_end = int(len(images) * 0.8)
    for leaf_size in sample_leaf_options: 
        for n_estimators_size in n_estimators_options: 
            alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50) 
            alg.fit(images[: train_end], tr_hot[:train_end]) 
            predict = alg.predict(images[train_end : ]) # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度 
            results.append((leaf_size, n_estimators_size, (tr_hot[train_end:] == predict).mean())) # 真实结果和预测结果进行比较，计算准确率 
            print((tr_hot[train_end:] == predict).mean()) # 打印精度最大的那一个三元组 print(max(results, key=lambda x: x[2]))
            dump(alg, './outs/filename_' + str(index)+ '.joblib') 
            index += 1
def get_testimg_imgid_list(df):
        file_list = []
        for i, row in df.iterrows():
            file_list.append(row['Id'])
            
        return file_list
import cv2
import os
def get_test_image_list(pre_dir, df):
        img_id_list = get_testimg_imgid_list(df)
        imgs = np.array()
        for img_id in img_id_list:
            img_path = pre_dir+ '_' + 'green' + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            imgs.append(img)
        

        return imgs
def test():
    alg = load('./outs/filename_10.joblib') 
    test_image_dir = os.path.join('../', 'test/')
    df=pd.read_csv('../sample_submission.csv')

    test_image_list = get_test_image_list(test_image_dir, df)
   # test_image_list = np.array(test_image_list)
    print('shape ', test_image_list.shape)
    nsamples, nx, ny = test_image_list.shape
    test_image_list = test_image_list.reshape((nsamples,nx*ny))
    predicts = alg.predict(test_image_list)
    print('len ', len(predicts))
    for predict in predicts:
        print(predict)
        return

#randomForest()
test()