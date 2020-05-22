# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:09:24 2020

@author: Md.Abrar Istiak
"""

import numpy as np
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def isolation_forest_score(df):
    df=df.drop(['time'], axis=1)
    df1 = pd.read_csv("normal.csv")
    df1=df1.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)
    df_final=pd.concat([df, df1], ignore_index=True)
    feature= df_final
    minmax = MinMaxScaler(feature_range=(0, 1))
    feature=minmax.fit_transform(feature)
    
    modelanom = pickle.loads(open("./model/mulw_ovelpos.model", "rb").read())
    modelanom.fit(feature)
    scores_pred = (modelanom.decision_function(feature) * -1) + 0.5
    anom_feat=scores_pred
    anom_feat=anom_feat[:len(df)]

    if np.mean(anom_feat)>0.55 or np.mean(anom_feat)<0.46 :
        anom_feat=abs(1.4*anom_feat**3-2.3*anom_feat**2)


    return anom_feat


def univariate_score(df,modelanom):      #if univariate results needed from isolation forest
    df=df.drop(['time'], axis=1)
    #pore
    df1 = pd.read_csv("normal.csv")
    df1=df1.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)
    #df_final=pd.concat([df, df1], ignore_index=True)
    df_final=pd.concat([df, df1], ignore_index=True)
    df= df_final
    #pore
    anom_feat=np.zeros(df.shape)
    fields=list(df.columns)
    for i in range(df.shape[1]):
        feature= df.iloc[:,i:i+1]
        minmax = MinMaxScaler(feature_range=(0, 1))
        feature=minmax.fit_transform(feature)
        #only for XBOS
        feature=pd.DataFrame(feature)
        #modelanom = pickle.loads(open("mulw_ovelpos.model", "rb").read())
        modelanom.fit(feature)
        scores_pred = modelanom.decision_function(feature) * -1 +0.5    #to match the anomaly score range with paper
        anom_feat[:,i]=scores_pred  #for anomaly score
        
    
    return anom_feat,fields