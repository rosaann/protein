#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:10:23 2018

@author: zl
"""
import random
def radom_sep_train_val(datalist, rate):
    random.seed(90)
    total = len(datalist[0])
    idx_list = range(total)
    idx_list = random.sample(idx_list, total)
    
    train_persent = int(rate * total)
    
    out_train = []
    out_val = []
    for sub_list in datalist:
        train_list = []
        val_list = []
        for i, idx_random in enumerate( idx_list ):
            if i <= train_persent:
                train_list.append(sub_list[idx_random])
            else:
                val_list.append(sub_list[idx_random])
        out_train.append(train_list)
        out_val.append(val_list)
        
    return out_train, out_val