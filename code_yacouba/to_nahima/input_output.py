# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:08:27 2022

@author: kone
"""

import os
import sys
sys.path.append("E:/")
import pandas as pd
import numpy as np
from SmartStep_optim_version import *
os.chdir("E:/SmartStep")
import time
from progressbar import ProgressBar
import onnxruntime as rt
import warnings
warnings.filterwarnings('ignore')
from mat4py import loadmat 


########################################################
# ------ Intialisation all of constants  
########################################################

########################################################
#-------- Start Demo
# ------ Import Data From Matlab File. It's can be Update for csv file or other type of file 
#------ Import Both models gyro and accelero
########################################################


sess_acc = rt.InferenceSession("Model/acc.onnx")
input_acc = sess_acc.get_inputs()[0].name
                

paths='E:/SmartStep/Data_Test/sig_try.mat'
print(paths)

data = loadmat(paths)

"""import scipy.io
mat = scipy.io.loadmat(filename)
data=pd.DataFrame(mat['signal'],columns=['accx','accy','accz'])"""

data=pd.DataFrame(data['signal'],columns=["Accx","Accy","Accz"])

data=data*9.8

#===============  This part is using for dowsampling ( 200 Hz --> 100Hz ) Keep in your mind 

########################################################
# ------ Compute all features we have need gyro and accelero 
########################################################

step_instant=list()
maxprob=0.5


#acc
feat_acc=['acc_Median_win80','acc_valley_prominences80','acc_kurt_win30','acc_peak_prominences50',
          'acc_skew_win5','acc_peak_prominences80','acc_domfreq1','acc_indMin_win30']


pbar = ProgressBar()

for t in pbar(range(len(data))):
    y_prob=DETECT_STEP().LGBM(imudata=data,iterate=t,model_acc=sess_acc,input_acc=input_acc)
    step_instant.append(int(np.where(y_prob>maxprob,1,0)))

"""
prediction=pd.DataFrame(step_instant)   
prediction.column='step_instant' 
xport=pd.concat([xport.reset_index(drop=True),prediction],axis=1)
xport.column=feat
xport.to_csv('D:/Inmob/_sharing/smartstep_to_okeenea/SmartStep/Result_smartstep/input_oupt_model_onnx.csv',sep=';',header=True,index=False)"""