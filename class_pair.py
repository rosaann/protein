import pandas as pd
minor_type_class = [20, 8, 10,9,  15, 17, 27, 24, 26] #[8, 9, 10,13, 15, 16,17,20, 27] #[8, 9, 10, 15, 16, 17, 27]
major_type_class = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 16, 18, 19, 21, 22, 23, 25]
class_pair_list = [[0, 8, 2, 3, 25, 21, 7],
                  [6, 9, 10],
                  [6, 9, 10],
                  [15, 5, 0, 16, 2],
                  [16, 0, 6, 14, 17, 18, 25, 11, 21, 7, 3, 5, 2, 22, 23, 4, 26, 19, 12, 15, 24],
                  [16, 14, 17, 18, 25, 0, 21, 19, 7, 5, 2, 23, 22],
                  [2, 0, 27, 5, 19, 4, 23]]

cut_class_pair = [[ 0,  2,  3,  7,  8, 21, 25],[ 6,  9, 10],[ 6,  9, 10],[ 0,  2,  5, 15, 16],[ 0,  2,  3,  4,  5,  6,  7, 11, 14, 16, 17, 18, 21, 22, 23, 25],[ 0,  2,  4,  5,  6,  7, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26],[ 0,  2,  4,  5,  6,  7, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24,
       25],[ 0,  2,  3,  4,  5,  6,  7, 14, 15, 16, 17, 18, 21, 22, 24, 25, 26],[ 0,  2,  3,  4,  5,  6,  7, 11, 14, 15, 16, 17, 18, 19, 21, 22, 23,
       24, 25],[ 0,  2,  3,  4,  5,  6,  7, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25],[ 0,  2,  4,  5,  7, 11, 12, 14, 16, 17, 18, 19, 21, 23, 25],[ 0,  2,  4,  5,  7, 11, 14, 16, 17, 18, 19, 24, 25],[ 0,  2,  5,  7, 14, 16, 17, 18, 19, 21, 23, 25],[ 0,  2,  5,  7, 14, 16, 17, 18, 19, 21, 22, 23, 25],[ 0,  2,  5,  7, 14, 16, 17, 18, 19, 21, 23, 25]]

param_list = [
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5, 'random_state':10},
               
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.8, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':25
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5, 'reg_alpha':50},#8 , 'reg_alpha':40, 'reg_lambda':0.1
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':4
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
               
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':3
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
               
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.6, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':10
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':15},#15
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.1, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':10
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}]

major_param_list = [ {'max_depth':6,'silent':0,'n_estimators':2
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 

            {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
             
             {'max_depth':6,'silent':0,'n_estimators':3
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
              
              {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
               {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}, 
                
                 {'max_depth':6,'silent':0,'n_estimators':2
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':1
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5}]


def get_train_group():
    df = pd.read_csv('../train.csv')
      
    train_data_id_class_list = []
    
    #先从type_class中，选含有其中一种的，剩下的
    
    train_once_num = 80
    for ti, type_class in enumerate( minor_type_class) :
        idinfo_list = get_type_class(type_class, df, train_data_id_class_list)
        if  len(idinfo_list[0]) < train_once_num :
            print('find c ', type_class, ' len ', len(idinfo_list[0]))
            hav_gotten_id_list = idinfo_list[ 0]
           # print('hav_gotten_id_list ', hav_gotten_id_list)
            idinfo_list = get_rest_id_info(df, hav_gotten_id_list, train_data_id_class_list, idinfo_list, train_once_num) 
            print('len ', len(idinfo_list[0]), ' ')
            train_data_id_class_list.append(idinfo_list)
        else: 
             print('find c- ', type_class, ' len ', len(idinfo_list[0]))
             train_once_per = int( train_once_num * 0.9)
             full_timie = int(len(idinfo_list[0]) / train_once_per)
             for i in range(full_timie):
                 start = i * train_once_per
                 end = start + train_once_per
                 cut = [idinfo_list[0][start : end], idinfo_list[1][start : end], idinfo_list[2][start : end]]
                 idinfo_list_sub = get_rest_id_info(df, cut[0], train_data_id_class_list, cut, train_once_num) 

                 train_data_id_class_list.append((idinfo_list_sub))
                 print('cut len ', len(cut[0]))
             rest = [idinfo_list[0][full_timie * train_once_per : ], idinfo_list[1][full_timie * train_once_per : ], idinfo_list[2][full_timie * train_once_per : ]]
             idinfo_list = get_rest_id_info(df, rest[0], train_data_id_class_list, rest, train_once_num)
             train_data_id_class_list.append(idinfo_list)
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
              train_data_id_class_list.append([idx_list, id_list, tar_list])
             # print('jkj len ', len(idx_list))
              idx_list = []
              id_list = []
              tar_list = []
    train_data_id_class_list.append([idx_list, id_list, tar_list])
    print('last len ', len(idx_list))
    print('train_group ', len(train_data_id_class_list))
    print('original total ', df.shape[0])
    
    
    return train_data_id_class_list;

def get_rest_id_info(df, hav_gotten_id_list, train_data_id_class_list,idinfo_list_toadd, train_once_num):           
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
                    
                    
                    idinfo_list_toadd[0].append(i)
                    idinfo_list_toadd[1].append(row['Id'])
                    idinfo_list_toadd[2].append(targets_t)
                    if len(idinfo_list_toadd[0]) >= train_once_num:
                        return idinfo_list_toadd        
def get_type_class(type_check, df, train_data_id_class_list)  :     
    idx_list =[]
    id_list = []
    tar_list = []
    for i, row in df.iterrows():
        targets = row['Target'].split(' ')
        targets_t = [int (tthis) for tthis in targets]
        for t in targets_t:
            if t == type_check:
               if_in_saved_list = False
               for saved_train_list in train_data_id_class_list:
                   if i in saved_train_list[0]:
                       if_in_saved_list = True
                       break
               if if_in_saved_list == True:
                   continue
               idx_list.append(i)
               id_list.append(row['Id'])
               tar_list.append(targets_t) 
            
    return [idx_list, id_list, tar_list]