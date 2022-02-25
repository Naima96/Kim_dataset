# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:10:16 2021

@author: kone
"""


#############################################
#------- Import Library and set dir
############################################
import pandas as pd
from numpy import *
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy import signal,fft
from scipy.signal import find_peaks
from scipy import stats


       
class DETECT_STEP():
     def __init__(self,w_size=128,threshold_proba=0.5):
         self.w_size=w_size
         self.w_size_m1=self.w_size-1
         self.threshold_proba=threshold_proba         
         
     def filter_data(self,acc=[0]):
         A = np.asarray([1, -8.99594505535417, 36.4633075498862, -87.6908102873418, 138.559912556039, -150.302116273606, 113.348650426591, -58.6790854259776, 19.9562604962225, -4.02604302102184, 0.365869040210561]);
         B = np.asarray([5.51456216845228e-12, 5.51456216845228e-11, 2.48155297580353e-10, 6.61747460214274e-10, 1.15805805537498e-09, 1.38966966644997e-09, 1.15805805537498e-09, 6.61747460214274e-10, 2.48155297580353e-10, 5.51456216845228e-11, 5.51456216845228e-12]);
    
         if len(acc)!=1:
             y=acc.copy()
             for col in y:
                 y[col]=signal.filtfilt(B, A, y[col].values) # fix
             acc_filtered=y
    
         return acc_filtered


     def _dominant_frequency(self,signal_x, sampling_rate=100,Ns=10,nfft=1024,nfft2=512,fmin=0.5,fmax=4,cutoff=12.0): #100
    
         signal_x = signal_x-np.mean(signal_x)
         dim = signal_x.shape
        
        #valerie from matlab
         
         freq = (np.fft.fftfreq(nfft) * sampling_rate)[0:nfft2]
        
        #valerie from matlab
          
         lowind=np.where(freq>fmin)[0][0]
         upind=np.max(np.where(freq<fmax))
    
        # fourier transform
        #valerie add hamming
        # nfft is used for padding
         haming= np.hamming(dim[0])
         sp_hat = np.fft.fft(signal_x*haming, nfft)
         furval = sp_hat[0:nfft2] * np.conjugate(sp_hat[0:nfft2])
         furval=furval/sum(furval)
    
        #from the internet
        # cutoff is 12 
         idx_cutoff = np.argwhere(freq <= cutoff)
        #all freq less than cutoff
         freq = freq[idx_cutoff]
        #keep values less than cutoff
         sp = furval[idx_cutoff]
        #normalise
         sp_norm = sp / sum(sp)        
        # from valerie matlab
         
         ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
         domfreq=freq[ind] # it is equal to the maximum frequency ==Max_freq 
         
    
         return sp_norm.max().real,domfreq
        
     def LGBM(self,imudata,iterate,model_acc,input_acc,maxprob=0.5,start_wind80=24,start_wind=64,window_size80= 106,
              start_wind5=62,window_size5=67,start_wind30=49,window_size30=80,start_wind20=44,window_size20=75,start_wind50=39,window_size50= 90,start_wind60=34,window_size60=95,
              start_wind70=29,window_size70= 100,window_slide_step=1,window_freq=128,result=None): #w_size=400,w_size_m1=399
         
         self.imudata=imudata
         self.k=iterate
         self.window_size80= window_size80
         self.window_size5= window_size5
         self.window_size30=window_size30
         self.window_size20= window_size20
         self.window_size50= window_size50
         self.window_size60= window_size60
         self.window_size70= window_size70
         self.window_slide_step=window_slide_step
         self.window_freq=window_freq
         self.start_wind=start_wind
         self.model_acc=model_acc
         self.input_acc=input_acc
         
         """
    
        Parameters
        ----------
        imudata : Data calibre
            Cette fonction permet de calculer l'essemble des features necessaire pour la detection de Zupt puis de predire les instants de Zupt.
    
        INPUT : 
        ------- Données issue de PERSY déja calibrées avec les noom de variables suivants {
        "Accx","Accy","Accz","Gyrx","Gyry","Gyrz","Incx","Incy","Incz"}
        
        OUTPUT : 
        ------- L'essemble des 16 variables {
        'grad_Acc','grad_Gyr','fg_Norm_Acc_Energy_','fg_Norm_Gyr_Energy_','fg_ax-az_r2_','fg_ay-az_r2_',
        'Norm_Acc', 'Norm_Gyr', 'fg_Norm_Acc_', 'fg_Norm_Gyr_','fg_Norm_Acc_mean_', 'fg_Norm_Gyr_mean_',
        'fg_Norm_Acc_mae_','fg_Norm_Gyr_mae_','fg_Norm_Acc_quartile_','fg_Norm_Gyr_quartile_'}
        
        Returns
        ------- Step instant {0,1}
        Cette fonction retourne les instants de pas codes en 0 et 1. {0} : 'No step' et {1} : 'step'
        """
        
         calculate_norm_accandgyro=lambda data,last=127:np.sqrt((data.iloc[last,:]**2).sum(axis=0))
     
         findMiddle=lambda input_list: np.where(float(len(input_list))/2 % 2 != 0,int(float(len(input_list))/2 - .5),int(float(len(input_list))/2))

         calc_index_min=lambda data: list(data).index(max(list(data)))
                 
         if(self.k<self.w_size_m1):
             acc=list(np.zeros(8))

         else:
             
            
             acc_data=self.imudata.loc[self.k-self.w_size_m1:self.k+1,["Accx","Accy","Accz"]].reset_index(drop=True)           
             
             acc_data_filtered=self.filter_data(acc=acc_data)
        
             acc_mag_filtered=calculate_norm_accandgyro(acc=acc_data_filtered)

             acc_mag_unfiltered=calculate_norm_accandgyro(acc=acc_data)
            
            
        #===============================================
           #print("acceleration features")
   
             mag30a=acc_mag_filtered[start_wind30:window_size30]

             mag5a=acc_mag_filtered[start_wind5:window_size5]
   
             mag80a=acc_mag_filtered[start_wind80:window_size80]
   
             mag50a=acc_mag_filtered[start_wind50:window_size50]
      
             mag_freq=acc_mag_unfiltered[0:window_freq+1]
           
             peak_index80a,peak_properties80a= find_peaks(mag80a,prominence=(None,None)) 
   
             peak_index50a,peak_properties50a= find_peaks(mag50a,prominence=(None,None)) 
   
             valley_index80a,valley_properties80a= find_peaks(-mag80a,prominence=(None,None))
               
             _,domfreq=self._dominant_frequency(mag_freq)               
   
             
             if peak_index80a.size>0:
                 peak80a=peak_properties80a["prominences"][findMiddle(peak_index80a)]
             else:
                 peak80a=50                       
                 
             if valley_index80a.size>0:
                 valley80a=valley_properties80a["prominences"][findMiddle(valley_index80a)]
             else:
                 valley80a=50                 
                 
             if peak_index50a.size>0:
                 peak50a=peak_properties50a["prominences"][findMiddle(peak_index50a)]
             else:
                 peak50a=50
         


             acc=[np.median(mag80a),valley80a,stats.kurtosis(mag30a),peak50a,stats.skew(mag5a),peak80a,domfreq,calc_index_min(mag30a)]
            
         return self.model_acc.run(None, {self.input_acc: np.array(acc).reshape(1,-1).astype(np.float32)})[1][0][1]
