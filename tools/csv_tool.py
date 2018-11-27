#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:34:34 2018

@author: zl
"""
import pandas as pd
import cv2
import os
import numpy as np

def findAlltype():
    df = pd.read_csv('../../train.csv')
    allTargets = []
    for i, row in df.iterrows():
       # print('id ', row['Id'])
       # print(' ', row['Target'])
        targets = row['Target'].split(' ')
        for this_target in targets:
            if_enrolled = False
            for e_tar in allTargets:
                if int(this_target) == int(e_tar):
                    if_enrolled = True
                    break
            if if_enrolled == False:
                allTargets.append(int(this_target))
                
    print('arr ', allTargets)
    print('sor ', sorted( allTargets))
def takeSecond(elem):
    return elem[1]
def findAlltypeNum(df):
    allTargets = [i for i in range(28)]
    out = []
    for tar in allTargets:
        count = 0
        for i, row in df.iterrows():
            targets = row['Target'].split(' ')
            targets = [int (tthis) for tthis in targets]
            if tar in targets:
                count += 1
        print('type ', tar, ' count ', count)
        out.append((tar, count))
    out.sort(key=takeSecond)
    print(out)
    return out

def test():
     df = pd.read_csv('../../train.csv')
     print('d ', df.get_value(1, 'Id'))
     print('len ', df.shape[0])
     
def test_merge_img():
    df = pd.read_csv('../../train.csv')
    img_id = df.get_value(1, 'Id')
    target = df.get_value(1, 'Target')
        
  #  img_name_tails = ['blue', 'green', 'red']
    img_name_tails = [ 'red', 'green', 'blue', 'yellow']
    imgs = []
    for tail in img_name_tails:
        img_path = '../../train/' + img_id + '_' + tail + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
        imgs.append(img)
    img_merg = cv2.merge(imgs)
    cv2.imwrite(os.path.join('./','3.jpg'), img_merg)
def genSingleSampeIdList(df):
    
    allTargets = [i for i in range(28)]
    all_sclass_idlist = []
    for check_id in allTargets:
        this_list = []
        for i, row in df.iterrows():
            targets = row['Target'].split(' ')
            targets = [int (tthis) for tthis in targets]
            if len(targets) == 1 and targets[0] == check_id:
                this_list.append(i)
                
        all_sclass_idlist.append((check_id, this_list))
        print('class ', check_id, ' total ', len (this_list))
    return all_sclass_idlist

def genBalencedData():
    df = pd.read_csv('../../train.csv')
    allClassNumList = findAlltypeNum(df)
    singleClassNumList = genSingleSampeIdList(df)
   # max_class , max_num = allClassNumList[-1]
    max_num = 8000
    num_need_list = []
    for tar_class, tar_num in allClassNumList:
        if tar_num < max_num:
            num_need_list.append((tar_class, max_num - tar_num))
    id_to_check = [24,26,16,13,12,22,18,6,14,11,1,19,3,4,5,7,23,2,21,25,0]
    
    
    for tar_class, tar_num_need in num_need_list:
        if tar_class in id_to_check:
            for s_class , s_list in singleClassNumList:
                if tar_class == s_class:
                    tar_single_list = s_list
                    break
            print('len ', len(tar_single_list), ' num need ', tar_num_need, 'class ', tar_class)
            
            df = genImage(tar_single_list, tar_num_need, df,tar_class)
         #   return #test
    df.to_csv('../../sample_arg.csv', index=None)
            
def rotate(image, angle, center=None, scale=1.0): 
    # 获取图像尺寸 
    (h, w) = image.shape[:2] 
    # 若未指定旋转中心，则将图像中心设为旋转中心 
    if center is None: 
        center = (w / 2, h / 2)
    # 执行旋转 
    M = cv2.getRotationMatrix2D(center, angle, scale) 
    rotated = cv2.warpAffine(image, M, (w, h)) 
    # 返回旋转后的图像 
    return rotated


def genImage(base_list, num_need, df, for_tar)  :  
    img_name_tails = [ 'red', 'green', 'blue', 'yellow']  
    base_num = len(base_list)
    arg_by = int(num_need / base_num)
    img_base = '../../train/'
    img_out_base = '../../train/'
    for idx, img_idx in enumerate( base_list):
        img_id = df.get_value(img_idx, 'Id')
        ang_list = np.random.randint(0, 360, size= int(arg_by / 4))
        #get start idx
        start_idx = df.shape[0]
     #   print('start ', start_idx, ' img_id ', img_id, ' ang_ ', ang_list[:10])
        for ti, tail in enumerate( img_name_tails):
            #一个色一个色地处理，
            img_path = img_base + img_id + '_' + tail + '.png'
            img_0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE )
            img_1 = cv2.flip(img_0, 1)
            img_2 = cv2.flip(img_0, 0)
            img_3 = cv2.flip(img_0, -1)
         #   cv2.imwrite(os.path.join(img_out_base,'{}_0.png'.format(1, tail)), img_0)
         #   cv2.imwrite(os.path.join(img_out_base,'{}_1.png'.format(1, tail)), img_1)
         #   cv2.imwrite(os.path.join(img_out_base,'{}_2.png'.format(1, tail)), img_2)
        #    cv2.imwrite(os.path.join(img_out_base,'{}_3.png'.format(1, tail)), img_3)
            #先得到翻转基础图
            trans_base_list = [img_0, img_1, img_2, img_3]
            for idx_ang, arg in enumerate(ang_list):
            #    print('ang ', arg)
                for idx_trans,  trans_base_img in enumerate( trans_base_list):
                    img = rotate(trans_base_img, arg)
                    this_id = start_idx + (idx_ang * 4 ) + idx_trans
                    sub_img_id = int( (idx_ang * 4 ) + idx_trans)
                    cv2.imwrite(os.path.join(img_out_base,'{}-{}_{}.png'.format(img_id,str(sub_img_id), tail)), img)
                    if ti == 0:
                        df.set_value(this_id, 'Target', for_tar)
                        df.set_value(this_id, 'Id', img_id + '-' +str(sub_img_id))
                #        df.to_csv('sample_arg.csv', index=None) #test
                #    if idx_ang >= 10:
                #        return #test
    return df
import shutil
def vis_every_class():
    #每一类看10张，再看十张组合的
    df = pd.read_csv('../../train.csv')
    allTargets = [i for i in range(28)]
    base_out_path = '../../sample_img/'
    base_src_path = '../../train/'
    
    img_name_tails = [ 'red', 'green', 'blue', 'yellow']  
    
    for tar in allTargets:
        dir_path = base_out_path + str(tar)
        os.makedirs(dir_path)
        count = 0
        for i, row in df.iterrows():
            targets = row['Target'].split(' ')
            targets = [int (tthis) for tthis in targets]
            if tar in targets:
                if len(targets) == 1:
                    file_name = row['Id']
                    imgs_to_merge = []
                    for tail in img_name_tails:
                        srcFile = base_src_path + file_name + '_' + tail + '.png'
                        targetFile = dir_path + '/'+ file_name + '_' + tail + '.png'
                      #  shutil.copyfile(srcFile,targetFile)
                        img = cv2.imread(srcFile, cv2.IMREAD_GRAYSCALE )
                        imgs_to_merge.append(img)
                    img_merg = cv2.merge(imgs_to_merge)
                    cv2.imwrite(dir_path + '/'+ file_name + '_merge'  + '.jpg', img_merg)
                    count += 1
                    if count >= 10:
                        break
                    
def vis_compond_class():
    df = pd.read_csv('../../train.csv')
    allTargets = [i for i in range(28)]
    base_out_path = '../../sample_img/com/'
    base_src_path = '../../train/'
    
    img_name_tails = [ 'red', 'green', 'blue', 'yellow']  
    extrem_thin = [10, 15, 17, 27] 
    for tar in allTargets:            
        dir_path = base_out_path + str(tar)
        os.makedirs(dir_path)
        count = 0
        for i, row in df.iterrows():
            targets = row['Target'].split(' ')
            targets = [int (tthis) for tthis in targets]
            if tar in targets:
                if len(targets) > 1:
                    file_name = row['Id']
                    imgs_to_merge = []
                    for tail in img_name_tails:
                        srcFile = base_src_path + file_name + '_' + tail + '.png'
                        targetFile = dir_path + '/'+ file_name + '_' + row['Target'] + '_' + tail + '.png'
                    #    shutil.copyfile(srcFile,targetFile)
                        img = cv2.imread(srcFile, cv2.IMREAD_GRAYSCALE )
                        imgs_to_merge.append(img)
                    img_merg = cv2.merge(imgs_to_merge)
                    cv2.imwrite(dir_path + '/'+ file_name + '_' + row['Target'] + '_merge'  + '.jpg', img_merg)
                    count += 1
                    if count >= 10:
                        break


vis_compond_class()                
vis_every_class()
#genBalencedData()
#genSingleSampeIdList()
#findAlltype()
#test()
#findAlltypeNum()