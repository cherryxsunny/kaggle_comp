#!/usr/bin/env python3
# -*- coding= utf-8 -*-
"""
Created on Fri Oct 12 20:24:17 2018

@author= IrisXie
"""

from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from copy import deepcopy
import pandas as pd
import numpy as np
from feature import load_data, feature_engineering,log_,resume_


def model_mlp(X,Y,test_data):
    
    # initial regressor
    Y = Y.apply(lambda x: log_(x))
    
    Y = np.array(np.ravel(Y))
    regressor = MLPRegressor(hidden_layer_sizes=(84, 42, 21), alpha=0.005, learning_rate='adaptive',
                             early_stopping=True, n_iter_no_change=50, max_iter=10000)
    
    regressor = AdaBoostRegressor(base_estimator=regressor, n_estimators=100, 
                                  learning_rate=0.05, loss='exponential')

    regressor.fit (X, Y)
    print ('train done!')
    rs = regressor.predict(test_data)

    rs = pd.DataFrame(rs, columns=['time'])
    rs = rs.apply(lambda x: resume_(x))
    
    # output result
    rs.to_csv("rs_1.csv")
    return (rs)


    
def K_Fold(X,Y,regressor):
    
    kf = KFold(n_splits=2, shuffle=True)
    cv_err = []
    regressor_cv = deepcopy(regressor)
    X = np.array(X)
    cnt = 0
    
    # calculate mean mse
    for i in range(1, 2):
        for train_idx, test_idx in kf.split(X, y=Y):
            cnt += 1
            print("iter:", cnt)
            # print("train:", train_idx, "test:", test_idx)
            train_X, test_X, train_Y, test_Y = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
            regressor_cv.fit(train_X, y=train_Y)
            cv_predict = regressor_cv.predict(test_X)
            cv_predict_real = np.vectorize(resume_)(cv_predict)
            test_Y_real = np.vectorize(resume_)(test_Y)
            mse_cv = mean_squared_error(cv_predict_real, test_Y_real)
            cv_err.append(mse_cv)
    print("cv error:", cv_err, "\n mse mean:", np.mean(np.array(cv_err)), "mse std:", np.std(np.array(cv_err)))

#   regressor = MLPRegressor(activation='tanh', alpha= 0.001, 
#                             batch_size= 'auto', beta_1= 0.9, 
#                             beta_2= 0.999, early_stopping= False,
#                             epsilon= 1e-08, 
#                             hidden_layer_sizes= (16,16,16,16,16), 
#                             learning_rate= 'invscaling', 
#                             learning_rate_init= 0.1, max_iter= 1000, 
#                             momentum= 0.9, nesterovs_momentum= True, 
#                             power_t= 0.5, random_state= None, 
#                             shuffle= True, solver= 'lbfgs', 
#                             tol= 0.0001, 
#                             validation_fraction= 0.1, 
#                             verbose= False, warm_start= False) 