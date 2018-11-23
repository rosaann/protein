#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:34:34 2018

@author: zl
"""
import pandas as pd
import cv2
import os

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
def findAlltypeNum():
    df = pd.read_csv('../../train.csv')
    allTargets = [str(i) for i in range(28)]
    out = []
    for tar in allTargets:
        count = 0
        for i, row in df.iterrows():
            targets = row['Target']
            if targets.find(tar) >= 0:
                count += 1
        print('type ', tar, ' count ', count)
        out.append((tar, count))
    out.sort(key=takeSecond)
    print(out)
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
#findAlltype()
#test()
findAlltypeNum()