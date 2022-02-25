# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:10:16 2021

@author: kone
"""


#############################################
#------- Import Library and set dir
############################################
import numpy as np

from scipy.signal import find_peaks
from scipy import stats

from utility_functions import *
       
class DETECT_STEP():
     def __init__(self,w_size=128,threshold_proba=0.5):
         self.w_size=w_size
         self.w_size_m1=self.w_size-1
         self.threshold_proba=threshold_proba         
         

        
     def LGBM(self,imudata,imudata_filtered,iterate,model_acc,input_acc,maxprob=0.5,start_wind80=24,start_wind=64,window_size80= 106,
              start_wind5=62,window_size5=67,start_wind30=49,window_size30=80,start_wind20=44,window_size20=75,start_wind50=39,window_size50= 90,start_wind60=34,window_size60=95,
              start_wind70=29,window_size70= 100,window_slide_step=1,window_freq=128,result=None,B=[],A=[]): #w_size=400,w_size_m1=399
         
         self.imudata=imudata
         self.imudata_filtered=imudata_filtered
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
         if(self.k<self.w_size_m1):
             acc=list(np.zeros(8))

         else:
             
            
             acc_data=self.imudata[self.k-self.w_size_m1:self.k+1,:]
             
             acc_data_filtered=self.imudata_filtered[self.k-self.w_size_m1:self.k+1,:]
             
             # acc_data_filtered=filter_data(acc=acc_data,B=B,A=A)
        
             acc_mag_filtered=calculate_norm_accandgyro(acc=acc_data_filtered)

             acc_mag_unfiltered=calculate_norm_accandgyro(acc=acc_data)
            
            
        #===============================================
           #print("acceleration features")
   
             mag30a = acc_mag_filtered[start_wind30:window_size30]

             mag5a = acc_mag_filtered[start_wind5:window_size5]
   
             mag80a = acc_mag_filtered[start_wind80:window_size80]
   
             mag50a=acc_mag_filtered[start_wind50:window_size50]
      
             mag_freq=acc_mag_unfiltered[0:window_freq+1]
           
             peak_index80a,peak_properties80a= find_peaks(mag80a,prominence=(None,None)) 
   
             peak_index50a,peak_properties50a= find_peaks(mag50a,prominence=(None,None)) 
   
             valley_index80a,valley_properties80a= find_peaks(-mag80a,prominence=(None,None))
               
             domfreq=dominant_frequency(mag_freq)               
   
             
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
         


             acc=[calc_median(mag80a),valley80a,calc_kurt(mag30a),peak50a,calc_skew(mag5a),peak80a,domfreq,calc_index_min(mag30a)]
            
         return self.model_acc.run(None, {self.input_acc: np.array(acc).reshape(1,-1).astype(np.float32)})[1][0][1] #result#y_probgyro,y_probacc
