# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:41:38 2020

@author: Md.Abrar Istiak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
np.random.seed(0)

#os.chdir('G:\SP CUP 2020\Isolation forest')
from isofor_test import isolation_forest_score,univariate_score  #importing anomaly detector from isofor.py
from findsensor import find_sensor

modelanom = pickle.loads(open("mulw_ovelpos.model", "rb").read())    #load the desired trained model on IMU data


directory="full_dataset_merged.csv"   #name of the input csv file

df = pd.read_csv(directory)  #input csv file

anomaly_score=isolation_forest_score(df,modelanom) #anomaly score vector
anomaly_score=anomaly_score[:len(df)]


if np.mean(anomaly_score)>0.55 or np.mean(anomaly_score)<0.46 :   #empirical sqashing function to keep the threshold of anomaly score to 0.5
    anomaly_score=abs(1.4*anomaly_score**3-2.3*anomaly_score**2)



plt.plot(anomaly_score)
plt.xlabel('time') 
plt.ylabel('Anomaly Score') 
plt.title('Multivariate anomaly Detection') 
plt.show()



###Univariate
modelanom = pickle.loads(open("anomaly_detectorunivariateIf.model", "rb").read())    #load model trained(univariate) on IMU data
anom_feat,fields=univariate_score(df,modelanom)
anom_feat=anom_feat[:len(df)]
a=abs(1.4*anom_feat**3-2.3*anom_feat**2)
a= pd.DataFrame(a)
sensor=find_sensor(a)

for i in range(len(sensor)):   #as in normal data there is no feature responsible for anomaly so make null at normal data
	if anomaly_score<0.5:
		sensor[i]='0'

pd.DataFrame(sensor).to_csv("sensor_list.csv",header=None, index=None)