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
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.externals import joblib
from class_pair import cut_class_pair, minor_type_class, class_pair_list
import os
from config import Config

from sklearn.model_selection import GridSearchCV

def find_small_num_class_ids():
    type_class = minor_type_class 
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
    add_idx = 0
    for i, row in df.iterrows():
        if row['Id'] not in id_list:
            targets = row['Target'].split(' ')
          #  print('add ',i, ' ', row['Id'], ' ', targets)
            id_list.append((row['Id'], targets))
            add_idx += 1
            if add_idx >= 500:
                break
    print('total ', df.shape[0], 'small ', len(id_list))
    return id_list, type_class
def xgboost_train_16seperate_model(ifTrain = True, train_to = 15):
    df = pd.read_csv('../train.csv')
      
    train_data_id_class_list = []
    
    #先从type_class中，选含有其中一种的，剩下的
    
    train_once_num = 80
    for ti, type_class in enumerate( minor_type_class) :
        idinfo_list = get_type_class(type_class, df)
        if len(idinfo_list[0]) < train_once_num:
            class_pair = class_pair_list[ti]
            hav_gotten_id_list = idinfo_list[ 0]
           # print('hav_gotten_id_list ', hav_gotten_id_list)
            idinfo_list = get_rest_id_info(df, hav_gotten_id_list, train_data_id_class_list, class_pair,idinfo_list, train_once_num) 
            print('len ', len(idinfo_list[0]), ' ')
            train_data_id_class_list.append((idinfo_list, class_pair))
        else: 
             train_once_per = int( train_once_num * 0.9)
             full_timie = int(len(idinfo_list[0]) / train_once_per)
             for i in range(full_timie):
                 start = i * train_once_per
                 end = start + train_once_per
                 cut = [idinfo_list[0][start : end], idinfo_list[1][start : end], idinfo_list[2][start : end]]
                 idinfo_list_sub = get_rest_id_info(df, cut[0], train_data_id_class_list, class_pair,cut, train_once_num) 

                 train_data_id_class_list.append((idinfo_list_sub, class_pair))
                 print('cut len ', len(cut[0]))
             rest = [idinfo_list[0][full_timie * train_once_per : ], idinfo_list[1][full_timie * train_once_per : ], idinfo_list[2][full_timie * train_once_per : ]]
             idinfo_list = get_rest_id_info(df, rest[0], train_data_id_class_list, class_pair,rest, train_once_num)
             train_data_id_class_list.append((idinfo_list, class_pair))
             print('with rest len ', len(idinfo_list[0]), ' ')
    min_group_len = len(train_data_id_class_list)
    print('min_group_len ', min_group_len)
    idx_list = []
    id_list = []
    tar_list = []
    for i, row in df.iterrows(): 
          if_in_saved_list = False
          for saved_train_list in train_data_id_class_list:
              if i in saved_train_list[0]:
                  if_in_saved_list = True
                  break
              if if_in_saved_list == True:
                  continue
          targets = row['Target'].split(' ')
          targets_t = [int (tthis) for tthis in targets]  
          idx_list.append(i)
          id_list.append(row['Id'])
          tar_list.append(targets_t)
          if len(idx_list) >= train_once_num:
              train_data_id_class_list.append(([idx_list, id_list, tar_list], class_pair))
             # print('jkj len ', len(idx_list))
              idx_list = []
              id_list = []
              tar_list = []
    train_data_id_class_list.append(([idx_list, id_list, tar_list], class_pair))
    print('last len ', len(idx_list))
    print('train_group ', len(train_data_id_class_list))
    
    if ifTrain == False:
        return train_data_id_class_list;
    
    clr_list = []
    real_class_pair_list = []
    model_base_path = 'outs/'
    start_from = 0
    for train_i, (train_data_id_class, c_pair ) in enumerate( train_data_id_class_list[:train_to]):
        print('part ', train_i , ' of ', len(train_data_id_class_list))
        if train_i < start_from:
            continue
        clr, new_c_pair = train_one_model(train_data_id_class, c_pair)
        model_path = model_base_path + 'xgboost_' + str(train_i) + '.pkl'
      
        joblib.dump(clr, model_path)
        real_class_pair_list.append(new_c_pair)
    
        file=open('outs/class_pair.txt','w')
        file.write(str(real_class_pair_list));
        file.close()
        
    val_model()
def xgboost_train(ifTrain = True, train_to = 16):
    df = pd.read_csv('../train.csv')
      
    train_data_id_class_list = []
    
    #先从type_class中，选含有其中一种的，剩下的
    
    train_once_num = 80
    for ti, type_class in enumerate( minor_type_class) :
        idinfo_list = get_type_class(type_class, df)
        if len(idinfo_list[0]) < train_once_num:
            class_pair = class_pair_list[ti]
            hav_gotten_id_list = idinfo_list[ 0]
           # print('hav_gotten_id_list ', hav_gotten_id_list)
            idinfo_list = get_rest_id_info(df, hav_gotten_id_list, train_data_id_class_list, class_pair,idinfo_list, train_once_num) 
            print('len ', len(idinfo_list[0]), ' ')
            train_data_id_class_list.append((idinfo_list, class_pair))
        else: 
             train_once_per = int( train_once_num * 0.9)
             full_timie = int(len(idinfo_list[0]) / train_once_per)
             for i in range(full_timie):
                 start = i * train_once_per
                 end = start + train_once_per
                 cut = [idinfo_list[0][start : end], idinfo_list[1][start : end], idinfo_list[2][start : end]]
                 idinfo_list_sub = get_rest_id_info(df, cut[0], train_data_id_class_list, class_pair,cut, train_once_num) 

                 train_data_id_class_list.append((idinfo_list_sub, class_pair))
                 print('cut len ', len(cut[0]))
             rest = [idinfo_list[0][full_timie * train_once_per : ], idinfo_list[1][full_timie * train_once_per : ], idinfo_list[2][full_timie * train_once_per : ]]
             idinfo_list = get_rest_id_info(df, rest[0], train_data_id_class_list, class_pair,rest, train_once_num)
             train_data_id_class_list.append((idinfo_list, class_pair))
             print('with rest len ', len(idinfo_list[0]), ' ')
    min_group_len = len(train_data_id_class_list)
    print('min_group_len ', min_group_len)
    idx_list = []
    id_list = []
    tar_list = []
    for i, row in df.iterrows(): 
          if_in_saved_list = False
          for saved_train_list in train_data_id_class_list:
              if i in saved_train_list[0]:
                  if_in_saved_list = True
                  break
              if if_in_saved_list == True:
                  continue
          targets = row['Target'].split(' ')
          targets_t = [int (tthis) for tthis in targets]  
          idx_list.append(i)
          id_list.append(row['Id'])
          tar_list.append(targets_t)
          if len(idx_list) >= train_once_num:
              train_data_id_class_list.append(([idx_list, id_list, tar_list], class_pair))
             # print('jkj len ', len(idx_list))
              idx_list = []
              id_list = []
              tar_list = []
    train_data_id_class_list.append(([idx_list, id_list, tar_list], class_pair))
    print('last len ', len(idx_list))
    print('train_group ', len(train_data_id_class_list))
    
    if ifTrain == False:
        return train_data_id_class_list;
    
    clr_list = []
    real_class_pair_list = []
    model_base_path = 'outs/'
    start_from = 0
    
    for i_c, c in enumerate( minor_type_class):
        param = {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}

        x = xgb.XGBClassifier(**param) 
        
        
        data_img_list = []
        tar_list = []
        base_path = '../train/'
        for train_i, (train_data_id_class, c_pair ) in enumerate( train_data_id_class_list[:train_to]):
            for img_idx, img_id, targets in zip(train_data_id_class[0], train_data_id_class[1],train_data_id_class[2]):
                trans_t = 0
                for t in targets:
                    if t == c:
                        trans_t = 1
                        break
              #  id_list.append(img_id)
                tar_list.append(trans_t)
                
                img_path = base_path + img_id + '_' + 'green' + '.png'
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
                img = cv2.resize(img, (300, 300),interpolation=cv2.INTER_LINEAR)    
                data_img_list.append(img)
                
        train_once_num = 80
        train_time = int(len(tar_list) / train_once_num)
        data_img_list = np.array(data_img_list)
        tar_list = np.array(tar_list)

        nsamples, nx, ny = data_img_list.shape
        data_img_list = data_img_list.reshape((nsamples,nx*ny))
        print('img shape', data_img_list.shape)  
        for train_i in range(train_time):
            start = train_i * train_once_num
            end = start + train_once_num
            if end >= len(tar_list):
                end = len(tar_list) - 1
            print('start fit ', c, ' part ', train_i, 'of ', train_time)
            x.fit(data_img_list[ start : end], tar_list[start : end])
        model_path = model_base_path + 'xgboost_model_per_class' + str(i_c) + '.pkl'        
        x.save_model(model_path)     
        
        
        
 #   val_model()    
def val_model():
    id_list, c_list = find_small_num_class_ids()
    #验证集，都从id_list中取     
    val_img_list, val_tar_list,  c_list, data_tar_list= get_val_data_from_idinfolist(id_list, c_list)
    
    
    pre_list = start_pre(val_img_list, data_tar_list)
  #  pair = [n for n in range(28)]
    y_p_factory = MultiLabelBinarizer()
    y_p_en = y_p_factory.fit_transform(pre_list)
    print('c_p ', y_p_factory.classes_)
   # y_t_en = y_p_factory.fit_transform(val_tar_list)
    print('c_t ', c_list)


    print('---------f1 ',f1_score(y_p_en, val_tar_list, average = "macro"))
    
def start_pre(val_img_list, val_tar_list):
    real_class_pair_list = cut_class_pair
    
    model_base_path = 'outs/'
    result_list = [list() for i in range(len(val_img_list))]
    config = Config()
    
    
    for ci, class_pair in enumerate( minor_type_class):    
        model_path = model_base_path + 'xgboost_model_per_class' + str(ci) + '.pkl'
        print('part ', ci , ' of ', len(real_class_pair_list))

        clr = XGBClassifier()
        clr.load_model(model_path)
        y_p_x = clr.predict_proba(val_img_list)

        
        for i_ys,  ys in enumerate( y_p_x ):
          #  if val_tar_list != None:
            print('ci ', ci, ' i_ys ', i_ys, ' pre ' , ys, ' c ', class_pair, ' t ', val_tar_list[i_ys])
            sub_result = result_list[i_ys]
            for iy, y in enumerate(ys):
                if y >= 0.5:   
                    sub_result.append(class_pair) 
                    
            result_list[i_ys] = sub_result       
      #  print('sub ', ci, ' r:', sub_result)
    
    pre_list = []    
    for this_sub_i, sub_result in enumerate( result_list):
        print('this_sub_i ', this_sub_i, ' sub_result ', sub_result)
        result_i = np.zeros(28)
        for i_s, s in enumerate( sub_result):
            result_i[s] += 1
        
        print('result_i ', result_i)
        result = []
        for i, r_i in enumerate(result_i):
            if r_i == 1 and (i in minor_type_class):
                print('i ', i,  ' r_i ', r_i)
                result.append(i)
        print('pre ', result , ' t ', val_tar_list[this_sub_i])
        pre_list.append(result)
    return pre_list
def start_pre_16seperate_model(val_img_list, val_tar_list):
    real_class_pair_list = cut_class_pair
    
    model_base_path = 'outs/'
    result_list = [list() for i in range(len(val_img_list))]
    config = Config()
    
    class_pair_reference = np.zeros(28)
    for ci, class_pair in enumerate( real_class_pair_list[:config.v('xgb_len')]):
        for ref in  class_pair:
            class_pair_reference[ref] = class_pair_reference[ref] + 1
    
    for ci, class_pair in enumerate( real_class_pair_list[:config.v('xgb_len')]):    
        model_path = model_base_path + 'xgboost_' + str(ci) + '.pkl'
        print('part ', ci , ' of ', len(real_class_pair_list))

        clr =  joblib.load(model_path)
        y_p_x = clr.predict_proba(val_img_list)

        
        for i_ys,  ys in enumerate( y_p_x ):
          #  if val_tar_list != None:
            print('ci ', ci, ' i_ys ', i_ys, ' pre ' , ys, ' c ', class_pair, ' t ', val_tar_list[i_ys])
            sub_result = result_list[i_ys]
            for iy, y in enumerate(ys):
                if y >= 0.5:
                    if class_pair[iy] == 8 or class_pair[iy] == 16:
                        if y >= 0.6:
                            sub_result.append(class_pair[iy]) 
                    else:
                        sub_result.append(class_pair[iy]) 
            result_list[i_ys] = sub_result       
      #  print('sub ', ci, ' r:', sub_result)
    
    pre_list = []    
    print('class_pair_reference ', class_pair_reference)
    for this_sub_i, sub_result in enumerate( result_list):
        print('this_sub_i ', this_sub_i, ' sub_result ', sub_result)
        result_i = np.zeros(28)
        for i_s, s in enumerate( sub_result):
            result_i[s] += 1
        
        print('result_i ', result_i)
        result = []
        for i, r_i in enumerate(result_i):
            t_ref = class_pair_reference[i] / 2
            if r_i > t_ref and (i in minor_type_class):
                print('i ', i, ' t_ref ', t_ref, ' r_i ', r_i)
                result.append(i)
        print('pre ', result , ' t ', val_tar_list[this_sub_i])
        pre_list.append(result)
    return pre_list
def test_xg_model():
    pre_dir = '../test/'
    
    df=pd.read_csv('../sample_submission_1.csv')
   # df = pd.DataFrame(columns = ["Id", "Predicted"])
    file_list = []
    for i, row in df.iterrows():
        file_list.append(row['Id'])
  #  print('len ', len(file_list))   
    img_list = []
    for file_id in file_list:
        img_path = pre_dir + file_id + '_' + 'green' + '.png'
     #   print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        img = cv2.resize(img, (300, 300),interpolation=cv2.INTER_LINEAR)    
        img_list.append(img)
        
    img_list = np.array(img_list)
    
  #  print('img shape qq ', img_list.shape)
    nsamples, nx, ny = img_list.shape
    img_list = img_list.reshape((nsamples,nx*ny))
  #  print('img shape', img_list.shape)
    
    pre_list = start_pre(img_list, None)
    
    return pre_list
    
    for i, row in df.iterrows():
        r = pre_list[i]
        if len(r) == 0:
            continue
        result = r[0]
        if len(r) > 1:
            for r_sub in r[1:]:
                result += ' '
                result += r_sub
        df.set_value(i, 'Predicted', result)
    #    print('idx ', i)

    df.to_csv('pred.csv', index=None)
    df.head(10)    
    print('Evaluating detections')
    return pre_list
    
def get_rest_id_info(df, hav_gotten_id_list, train_data_id_class_list, class_pair,idinfo_list_toadd, train_once_num):           
    for i, row in df.iterrows():
                if i not in hav_gotten_id_list:
                    if_in_saved_list = False
                    for saved_train_list in train_data_id_class_list:
                        if i in saved_train_list[0]:
                            if_in_saved_list = True
                            break
                    if if_in_saved_list == True:
                        continue
                    targets = row['Target'].split(' ')
                    targets_t = [int (tthis) for tthis in targets]
                    if_vali = False
                    for t in targets_t:
                        if t not in class_pair:
                            if_vali = True
                            break
                    if if_vali == False:
                        idinfo_list_toadd[0].append(i)
                        idinfo_list_toadd[1].append(row['Id'])
                        idinfo_list_toadd[2].append(targets_t)
                        if len(idinfo_list_toadd[0]) >= train_once_num:
                            return idinfo_list_toadd        
def get_type_class(type_check, df)  :     
    idx_list =[]
    id_list = []
    tar_list = []
    for i, row in df.iterrows():
        targets = row['Target'].split(' ')
        targets_t = [int (tthis) for tthis in targets]
        for t in targets_t:
            if t == type_check:
               idx_list.append(i)
               id_list.append(row['Id'])
               tar_list.append(targets_t) 
    return [idx_list, id_list, tar_list]
def get_type_class_num_info(type_check, df):
    id_list = []
    
    for i, row in df.iterrows():
        targets = row['Target'].split(' ')
        targets_t = [int (tthis) for tthis in targets]
        for t in targets_t:
            if t == type_check:
               id_list.append((row['Id'], targets_t)) 
    
    class_type_list = []           
    for id_t, targets_i in id_list:
        for t in targets_i : 
            if t not in class_type_list:
                class_type_list.append(t)
    print('type ', type_check,' total ',len(id_list), ' with ', class_type_list)
def train_one_model(idinfo_list, class_pair):
    base_path = '../train/'
    data_img_list = []
    data_tar_list = []
    
  #  print('class ', class_pair)
    
    pre_c = []
    for targets in idinfo_list[2]:
        for t in targets:
            if t not in pre_c:
                pre_c.append(t)
    print('cc ', pre_c)
    for img_idx, img_id, targets in zip(idinfo_list[0], idinfo_list[1],idinfo_list[2]):
        img_path = base_path + img_id + '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        img = cv2.resize(img, (300, 300),interpolation=cv2.INTER_LINEAR)    

        data_img_list.append(img)
        
        data_tar_list.append(targets)
        
    data_img_list = np.array(data_img_list)
    data_tar_list = np.array(data_tar_list)

    nsamples, nx, ny = data_img_list.shape
    data_img_list = data_img_list.reshape((nsamples,nx*ny))
 #   print('img shape', data_img_list.shape)   
    
   
    Y_enc_factory = MultiLabelBinarizer()
    Y_enc = Y_enc_factory.fit_transform(data_tar_list)
    c = Y_enc_factory.classes_
    print('c ', c)
    for i, y_en in enumerate(Y_enc):
        if i < 1:
            print(y_en)
        else :
            break

    param = {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}

    x = xgb.XGBClassifier(**param)  
    clf = OneVsRestClassifier(x)
    clf.fit(data_img_list, Y_enc)
    
    return clf, c
  
  #  parameters = {
  #  "estimator__max_depth": [2,4,8]
  #  }

  #  model_tunning = GridSearchCV(clf, param_grid=parameters,
  #                           scoring='f1')

  #  model_tunning.fit(data_img_list, Y_enc)

  #  print (model_tunning.best_score_)
  #  print (model_tunning.best_params_)
  #  return model_tunning.best_estimator_

def get_val_data_from_idinfolist(id_list,class_pair):
    base_path = '../train/'
    data_img_list = []
    data_tar_list = []
    log_idx = 0
    
   
    for img_id, targets in id_list:
        img_path = base_path + img_id + '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        img = cv2.resize(img, (300, 300),interpolation=cv2.INTER_LINEAR)    

        data_img_list.append(img)
        targets_t = []
        for tthis in targets:
           # if int(tthis) in minor_type_class:
                targets_t.append(int(tthis))
        data_tar_list.append(targets_t)

        
    data_img_list = np.array(data_img_list)
    data_tar_list = np.array(data_tar_list)

    nsamples, nx, ny = data_img_list.shape
    data_img_list = data_img_list.reshape((nsamples,nx*ny))
  #  print('img shape', data_img_list.shape)   
    
    Y_enc_factory = MultiLabelBinarizer()
    Y_enc = Y_enc_factory.fit_transform(data_tar_list)
    c = Y_enc_factory.classes_

    return data_img_list, Y_enc, c, data_tar_list
def xgboost_train_old_again():
    id_list = find_small_num_class_ids()
    
    base_path = '../train/'
    data_img_list = []
    data_tar_list = []
    log_idx = 0
    
   
    for img_id, targets in id_list:
        img_path = base_path + img_id + '_' + 'green' + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        img = cv2.resize(img, (300, 300),interpolation=cv2.INTER_LINEAR)    
     #   tar_t = np.zeros((28))
     #   for tar in targets:
     #       tar_t[int(tar)] = 1
        data_img_list.append(img)
        data_tar_list.append(targets)
     #   if log_idx < 10:
     #       print(tar_t)
     #       log_idx += 1
        
    data_img_list = np.array(data_img_list)
    data_tar_list = np.array(data_tar_list)

    nsamples, nx, ny = data_img_list.shape
    data_img_list = data_img_list.reshape((nsamples,nx*ny))
    print('img shape', data_img_list.shape)   
    
    Y_enc = MultiLabelBinarizer().fit_transform(data_tar_list)
    for i, y_en in enumerate(Y_enc):
        if i < 1:
            print(y_en)
        else :
            break
    train_end = int(len(data_img_list) * 0.8)
    param = {'max_depth':20,'eta':1, 'silent':1,'n_estimators':10
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'max_bin':16, 'seed':10 }

    x = xgb.XGBClassifier(**param)  
    clf = OneVsRestClassifier(x)
    
    steps = 10
    for i in range(steps):
        start = int( i * (len(data_img_list)/steps))
        end = start + int(len(data_img_list)/steps)
        if end > train_end:
            end = train_end
        clf.fit(data_img_list[start: end], Y_enc[start : end])
    #clf.fit(data_img_list[: train_end][0], data_tar_list[: train_end][1])
        y_p_x = clf.predict_proba(data_img_list[train_end : ])
    
        log_idx = 0
        for y in y_p_x:
            if log_idx < 5:
             #   print('pre ', y)
                log_idx += 1
            else :
                break
        y_p_x[y_p_x >= 0.5] = 1
        y_p_x[y_p_x < 0.5] = 0
        
      #  print('acc ', metrics.accuracy_score(y_p_x, Y_enc[train_end : ]))
        log_idx = 0
        for y in y_p_x:
            if log_idx < 5:
             #   print('pre-2 ', y)
                log_idx += 1
            else :
                break
        print('---------f1 ',f1_score(y_p_x, Y_enc[train_end : ], average = "macro"))
        
        
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
        
#xgboost_train()
val_model()
#test_xg_model()