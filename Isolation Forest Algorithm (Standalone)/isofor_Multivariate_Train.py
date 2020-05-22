####### THIS IS CODE TO TRAIN ANOMALOUS IMAGE IN ISOLATION FOREST APPROACH #########
#first some dependencies needed to be installed like 
#pip install pandas
#pip install matplotlib
#pip install pickle-mixin
#pip install scikit-learn
#pip install numpy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("normal.csv")  #give the directory name of the normal(training) csv file

df=df.drop(['time','id','pos_x','pos_y','pos_z','vel_lin_x','vel_lin_y','vel_lin_z'], axis=1)


##Multivariate###
feature= df
minmax = MinMaxScaler(feature_range=(0, 1))
feature=minmax.fit_transform(feature)


outliers_fraction = 0.01

modelanom = IsolationForest(n_estimators=200,contamination=outliers_fraction,random_state=0)   #here n_estimator means the number of tree,more trees to reach convergence
modelanom.fit(feature)


#saving model

f = open("isofor_trained.model", "wb")    #trained model
f.write(pickle.dumps(modelanom))
f.close()
