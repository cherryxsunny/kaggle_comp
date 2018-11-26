# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:24:17 2018

@author: IrisXie
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder
from sklearn.preprocessing import scale,MinMaxScaler,minmax_scale,normalize,StandardScaler
from sklearn.preprocessing import PolynomialFeatures as pl
from math import log10


def load_data(train,test):
    x = pd.read_csv(train)
    y = x.loc[:,'time']
    x_test = pd.read_csv(test)
     
    del x['time']
    del x['id']
    del x_test['id']
    #add feature from self generated data
    real_y_tr = pd.read_csv('data_real/train_data_real.csv')
    real_y_te = pd.read_csv('data_real/test_data_real.csv')
    x['time_real'] = real_y_tr['time']
    x_test['time_real'] = real_y_te['time']
    # n_jobs = -1
    x.loc[x['n_jobs'] == (-1) , 'n_jobs']  = 16
    x_test.loc[x_test['n_jobs'] == (-1) , 'n_jobs']  = 16
    return x,y, x_test

def feature_engineering(all_data): 
    
    # n_job = 16
    all_data.loc[all_data['n_jobs'] == (-1) , 'n_jobs']  = 16
    
    all_data.drop(['id','random_state', 'n_informative', 
                   'n_clusters_per_class', 'flip_y', 'scale',
                   'l1_ratio'], axis=1, inplace=True)

    all_data['feature_1'] = all_data["max_iter"]*all_data["n_samples"]*all_data["n_features"]
    all_data['feature_2'] = all_data['n_classes'] * all_data['n_clusters_per_class']
    all_data.drop(['n_clusters_per_class'], axis=1, inplace=True)
   
    # dummy
    all_data = pd.concat([all_data, pd.get_dummies(all_data['penalty'])], axis=1)
    all_data.drop(['penalty'], axis=1, inplace=True)

    
    for k in ['n_jobs', 'max_iter', 'n_samples', 'n_features', 'feature_1','feature_2','alpha']:
        all_data[k] = all_data[k].apply(lambda x: log10(x))
    
    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(all_data)
    
    print ('feature done!')
    return all_data

def log_(origin_label):
    return pd.Series(origin_label).apply(lambda x: log10(x))


def resume_(encoded_label):
    return pd.Series(encoded_label).apply(lambda x: 10.0 ** x)

#    x_len = x.shape[0]
#    # use 500 data as training set
#    x_all = x.append(x_test,ignore_index = True)
#    x_all['time_real'] = x_all['time_real'].apply(lambda x: np.log1p(x))
#    
#    encoder_cate = ['penalty']
#    for feat in encoder_cate:
#        x_all[feat] = LabelEncoder().fit_transform(x_all[feat])
#        
#    # del & add some new features
#    del x_all['random_state']
#    del x_all['flip_y']
#    del x_all['scale']
#    #del x['penalty']
#    #del x['n_samples']
#    #del x['alpha']
#    del x_all['l1_ratio']
#    del x_all['n_informative']
#    x_all['feature_1'] = x_all['n_classes'] * x_all['n_clusters_per_class']
#    #del x['n_classes']
#    #del x['n_clusters_per_class']
#    x_all['feature_2'] = x_all['max_iter'] * x_all['n_samples'] * x_all['n_features'] * x_all['n_classes'] / x_all['n_jobs']
#
#    # scale
#    for feature_ in x_all.columns:
#        x_all[feature_] = scale(np.array(x_all[feature_]).reshape(-1, 1)) 
#    # special scale
##    for feature_ in x.columns:
##        x[feature_] = (x[feature_] - x_test[feature_].mean())/x_test[feature_].std()
#    
#    # return data to model
#    x = x_all[:x_len]
#    x_test = x_all[x_len:]
#    return x_all

    # poly: used to generate new features
#    t = x.values
#    t = pl(degree=3,interaction_only=True).fit_transform(t)
    # PCA    
#    model = PCA(n_components = 1,svd_solver = 'auto')
#    x = model.fit_transform(x.values)
#    x = pd.DataFrame(x)
#    return x 