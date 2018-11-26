#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:24:17 2018

@author: IrisXie
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# model
#from mlxtend.regressor import StackingRegressor,StackingCVRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# self package
from feature import load_data, feature_engineering,log_,resume_
from model import model_mlp

#---------------------------read data------------------------------------
data = pd.read_csv("train.csv")
te_x = pd.read_csv("test.csv")

#----------------------------feature engineering-------------------------
tr_x = data.iloc[:, :-1]
tr_y = data.iloc[:, -1]

train_num = tr_x.shape[0]
data_all = pd.concat([tr_x, te_x])
data_all = feature_engineering(data_all)
data_all = pd.DataFrame(data_all)

tr_x = data_all.iloc[:train_num, :]
te_x = data_all.iloc[train_num:, :]
    
#--------------------------------model mlp--------------------------------
rs = model_mlp(tr_x,tr_y,te_x)
#rs.columns = ['time']
#rs = list(rs['time'])
#predict_result.to_csv("result_%2.2f+%2.2f.csv" % (np.mean(np.array(cv_err)), np.std(np.array(cv_err))), index_label="Id")
#pd.concat([te_x, predict_result], axis=1).to_csv('result_full.csv')


#all_train_set = pd.concat([x_tr, y], axis=1, sort=False)
#
## initial model
#model=['mlp' for i in range(10)] 
#for k in model:
#    add_model(k)
#set_parameters(x,y)
#
## model fusion
#mods = get_models()
#sclf  = StackingRegressor(regressors=mods,use_features_in_secondary =True,meta_regressor=mods[0],verbose=0)
#sclf.fit(x,y)
#result = sclf.predict(test)
#
##-----------------------------save rs-------------------------------------
# use history result to check the new result
#best_rs = pd.read_csv('rs_1.csv')
#best_rs=list(best_rs['time'].values)
#mse_ = mse(np.array(rs),best_rs)
#
#
#
#index = list(range(0,100))
#rs = {'id':index,'time':rs}
#rs = pd.DataFrame(rs)
#rs = pd.DataFrame(rs,columns=['id', 'time'])
#rs.to_csv('rs_'+str(mse_)+'.csv',index=False)
