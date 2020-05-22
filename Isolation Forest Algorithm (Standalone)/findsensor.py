# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:00:19 2020

@author: User
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def find_sensor(data):
    #col=
    data.columns = ["ori_x", "ori_y", "ori_z", "ori_w","vel_ang_x","vel_ang_y","vel_ang_z","acc_x","acc_y","acc_z"]
    #data=pd.read_csv('abnormaluni.csv',names=col)
    x=0
    y=0
    #rows, cols = (len(data.index), len(data.columns))
    rows,cols =(data.shape[0],data.shape[1])
    
    ans = [[0 for i in range(cols)] for j in range(rows)]  
    for x in range(len(data)):
        temp = data.loc[x]
        dfs = temp.sort_values(ascending=False)
        dfs=dfs/(dfs[0]-dfs[-1])
        ans[x][0]=[dfs.index[0]]
        k=1
        
        for y in range(len(dfs)-1):
            dist=distance.euclidean(dfs[0], dfs[y+1])
            if dist<0.5:
                ans[x][k]=[dfs.index[y+1]]
                k=k+1
                
    return ans