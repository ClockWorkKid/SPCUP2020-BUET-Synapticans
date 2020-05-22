# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:29:57 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:09:24 2020

@author: Md.Abrar Istiak
"""

import numpy as np
import matplotlib.pyplot as plt
#import os
#os.chdir('G:\SP CUP 2020\Isolation forest')
import time
starting=time.time()

from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
def isolation_forest_score(df,modelanom):
    ##Testing
    df=df.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)
    df1 = pd.read_csv("Training.csv")
    df1=df1.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)
    #df1=df1.drop(['time','id'], axis=1)
    df_final=pd.concat([df, df1], ignore_index=True)
    #df=df.drop(['time','id'], axis=1)
    #fields=list(df.columns)
    
    #PCA
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(df)
    #anom_feat=np.zeros(df.shape)
    feature= df_final
    minmax = MinMaxScaler(feature_range=(0, 1))
    feature=minmax.fit_transform(feature)
    
    #modelanom = pickle.loads(open("mulw_ovelpos.model", "rb").read())
    modelanom.fit(feature)
    #scores_pred = clf.decision_function(feature) * 1    #-1 for isolation forest,1 for HBOS 
    scores_pred = (modelanom.decision_function(feature) * -1) + 0.5
    anom_feat=scores_pred
    #del(feature)
    return anom_feat


def univariate_score(df,modelanom):
    df=df.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)
    #pore
    df1 = pd.read_csv("training.csv")
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
        ## predict raw anomaly score##
        #multiply by -1 if isolation forest is used
        scores_pred = modelanom.decision_function(feature) * -1 +0.5    #more the number means more abnormality
        #scores_pred=abs(1.45*scores_pred**3-2.3*scores_pred**2)
        ####Setting threshold is the main thing to set###
        #scores_pred = scores_pred / np.linalg.norm(scores_pred)  #normalizing
        #scores_pred=(scores_pred - np.min(scores_pred))/np.ptp(scores_pred)  #normalizing in range of [0,1]
        anom_feat[:,i]=scores_pred  #for anomaly score
        
    
    return anom_feat,fields