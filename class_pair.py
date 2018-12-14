minor_type_class = [20, 8, 9, 10, 15, 17, 27, 24, 26] #[8, 9, 10,13, 15, 16,17,20, 27] #[8, 9, 10, 15, 16, 17, 27]
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
                {'max_depth':8,'silent':0,'n_estimators':8
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':2
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':8, 'random_state':10, 'colsample_bytree' : 0.8, 'colsample_bylevel':0.8},
               
                {'max_depth':6,'silent':0,'n_estimators':5
             ,'learning_rate':0.3, 'objective':'binary:logistic'
             ,'nthread':8, 'scale_pos_weight':4
             ,'tree_method':'gpu_hist', 'predictor':'gpu_predictor'
             ,'seed':10 ,'max_bin':5},
                
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