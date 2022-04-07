# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:03:29 2022

@author: al-abiad
"""

import numpy as np
import pandas as pd
from utility_functions2 import _find_consecutive_index_ranges
import scipy.io
from utility_functions import calculate_norm_accandgyro
import matplotlib.pyplot as plt
import nolds as nld

def calculate_lyap(filename,step_vector,Numberofstrides,tao=10,dim=5):

    print("preprocessing")
    # data = scipy.io.loadmat(filename)['signal']
    data = scipy.io.loadmat(filename)['acc'][:,1:]

    step_vector=extract_walking_period(step_vector,minimum_seperationwalkingbout=200,longest_walkingbout=60)

    #extract periods of walking.
    Normalize_periods = [None] * len(step_vector)
    
    sampen_periods = [None] * len(step_vector)
    n=0
    total=len(step_vector)

    for episode in step_vector:
        print("finished %d out of %d"%(n,total))
        
        if len(episode)<Numberofstrides+1:
            print("the walking bout is shorter %d.. Number of strides are counted for one foot %d "%(len(episode),Numberofstrides))
            n=n+1
            continue
        
        data_crop=calculate_norm_accandgyro(data[episode[0,0]:episode[Numberofstrides,1],:])
        
        len_data=len(data_crop)
        normalize_data=np.zeros((100*Numberofstrides))
        
        for j in range (0,100*Numberofstrides):
            normalize_data[j]=data_crop[np.round(j*len_data/(100*Numberofstrides)).astype('int')]
            
        print("calculating lyap on magnitude")
        _,debug=nld.lyap_r(normalize_data, emb_dim=dim,lag=tao,
                                min_tsep=40, tau=0.01, trajectory_len=100*10, fit='poly', debug_plot=False,debug_data=True, plot_file=None, fit_offset=0)
        
        stride_duration=100
        # plt.figure()
        # d=debug[1]
        # x=debug[0]/stride_duration
        # plt.plot(x,d)    
            
        k=stride_duration//2
        z=np.polyfit(debug[0][0:k],debug[1][0:k],1)
        # p = np.poly1d(z)
        # plt.plot(debug[0][0:k]/stride_duration,p(debug[0][0:k]))
        lle=z[0]*100
        print("the lyap of acceleration norm is %.3f" %lle)
        
        Normalize_periods[n]=lle
        
        #calculate sample entropy 
        
        # m=5
        # r=0.3
        # samp=nld.sampen(normalize_data, emb_dim=m, tolerance=r,debug_plot=False,debug_data=False, plot_file=None)
        
        # sampen_periods[n]=samp
        # print("the sample entropy of acceleration norm is %.3f" %samp)

        # n=n+1
    return Normalize_periods
        
    # return Normalize_periods,sampen_periods

        
        
        
        
        
        
    
    #calculate norm
    #normalise signal from steps
    #Crop at a number od steps
    #Calculate 
    

def extract_walking_period(steps_vector,minimum_seperationwalkingbout=200,longest_walkingbout=60):
    
    print("extract walking periods")
    # group steps in same walking bouts
    consecutive_index_steps=_find_consecutive_index_ranges(steps_vector, increment = minimum_seperationwalkingbout)
    
    print("found %d walking periods"%(len(consecutive_index_steps)))
    
    # keep walking bouts with more than 60 steps
    consecutive_index_steps=[act for act in consecutive_index_steps if len(act)>=longest_walkingbout]
    print("kept %d more than 60 steps"%(len(consecutive_index_steps)))
    
    start_stop_all=[]
    for steps in consecutive_index_steps: 
        start_stop=[]
        for i in range(0,len(steps)-2,2):
            start_stop.append([steps[i],steps[i+2]])
        start_stop=np.vstack(start_stop)
     
        start_stop_all.append(start_stop)
    

    return start_stop_all
    
    

def calculate_variability_activeperiod(steps_vector,minimum_seperationwalkingbout=100,longest_walkingbout=60):
    

    # group steps in same walking bouts
    consecutive_index_steps=_find_consecutive_index_ranges(steps_vector, increment = minimum_seperationwalkingbout)
    
    print("found %d walking periods"%(len(consecutive_index_steps)))
    
    # keep walking bouts with more than 60 steps
    consecutive_index_steps=[act for act in consecutive_index_steps if len(act)>=longest_walkingbout]
    print("kept %d more than 60 steps"%(len(consecutive_index_steps)))
    
    columns=['time_start','time_end','N_steps','stridetime_std','stridetime_Cov','steptime_std','steptime_Cov']

    variability_data=pd.DataFrame(columns=columns) 
    
    detailed_stridetime= np.empty((0,3))

    for j in range(0,len(consecutive_index_steps)):
        variability=computeVarStride(fs=100,remove_outliers=True,N=3,use_smartstep=True,
                                     manual_peaks=consecutive_index_steps[j],use_peaks=True,pocket=False,remove_step=0)

        time_start=variability['detailed_steptime'][0,0]
        time_end=variability['detailed_steptime'][-1,0]
        
        if len(variability)>1:
            data=[time_start,time_end,len(variability['detailed_steptime']),
                  variability['stridetime_std'],variability['stridetime_Cov'],
                  variability['steptime_std'],variability['steptime_Cov']]
            
            var_data=pd.DataFrame([data],columns=columns)
            variability_data= pd.concat([variability_data,var_data])
            
            detailed_stridetime=np.concatenate((detailed_stridetime,variability['detailed_stridetime']))
            
            #append to dictionary 
            

            
    return variability_data,detailed_stridetime


def computeVarStride(fs=100,remove_outliers=True,N=1,use_smartstep=False,manual_peaks=[],
                     use_peaks=True,pocket=True,remove_step=0,round_data=True):
    """
    compute stride time 
    :param int fs: sampling frequency
    :param bool remove_outlier: whether to remove outliers
    :param int N: Nxstandard deviation 
    :param bool use_peaks: peaks are used as a mark that step happened
    :param bool pocket: whether one stride time is calculated 
    """
    #note: we remove two steps from beginging and end
    cycle_tempparam = {}

    peaks=manual_peaks


    #---if cellphone is in the hand or waist we can detect leading and contralateral foot stride time---
    stride_time_leading=np.diff(peaks[::2])/fs
    stride_time_contralateral=np.diff(peaks[1::2])/fs

    list_stride_time=[]
    list_step_time=[]
    for i in range(0,len(peaks)-2):
        step_time=peaks[i+1]-peaks[i]
        step_time=step_time/fs
        list_step_time.append([peaks[i],peaks[i+1],step_time])

        stride_time=peaks[i+2]-peaks[i]
        stride_time=stride_time/fs
        list_stride_time.append([peaks[i],peaks[i+2],stride_time])
        
    try:
        list_step_time=np.vstack(list_step_time)
        list_stride_time=np.vstack(list_stride_time)
    except Exception:
        print("no steps detected")
        

    if remove_outliers:
        stride_time_leading=np.array([i for i in stride_time_leading if i >= 0.8 and i <= 1.8])
        stride_time_contralateral=np.array([i for i in stride_time_contralateral if i >= 0.8 and i <= 1.8])
        try:
            #---stride time leading foot---
            mean=np.mean(stride_time_leading)
            cut_off=N*np.std(stride_time_leading)
            lower, upper =  mean- cut_off, mean + cut_off
            cycle_tempparam['stride_time_leading'] = np.array([i for i in stride_time_leading if i > lower and i < upper])
        except Exception:
            print("no stride time left")

        #---stride time contralateral foot---
        try:
            mean=np.mean(stride_time_contralateral)
            cut_off=N*np.std(stride_time_contralateral)
            lower, upper =  mean- cut_off, mean + cut_off
            cycle_tempparam['stride_time_contralateral'] = np.array([i for i in stride_time_contralateral if i > lower and i < upper])
        except Exception:
            print("no stride time left")
        #---step time---
        try:
            mean=np.mean(list_step_time[:,2])
            cut_off=N*np.std(list_step_time[:,2])
            lower, upper =  mean- cut_off, mean + cut_off
            cycle_tempparam['steptime'] = np.array([i for i in list_step_time[:,2] if i > lower and i < upper])
        except Exception:
            print("no step time left")

        try:
            list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=0.8 and i[2]<=1.7])
            mean=np.mean(list_stride_time[:,2])
            cut_off=N*np.std(list_stride_time[:,2])
            lower, upper =  mean- cut_off, mean + cut_off
            list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=lower and i[2]<=upper])
        except:
            print("no step time left")
    else:
        cycle_tempparam['stride_time_leading']=stride_time_leading
        cycle_tempparam['stride_time_contralateral']=stride_time_contralateral
        cycle_tempparam['steptime']=step_time

    #---merge left right stride cycle

    rl_stride=list_stride_time[:,2]

    if len(rl_stride)>1:
        cycle_tempparam['stridetime']=rl_stride
        cycle_tempparam["detailed_stridetime"]=list_stride_time
        cycle_tempparam["detailed_steptime"]=list_step_time

        cycle_tempparam['stride_time_leading_std']=np.around(np.std(cycle_tempparam['stride_time_leading']),decimals=3)
        cycle_tempparam['stride_time_leading_Cov']=np.around(np.std(cycle_tempparam['stride_time_leading']*100)/np.mean(cycle_tempparam['stride_time_leading']),decimals=3)

        cycle_tempparam['stride_time_contralateral_std']=np.around(np.std(cycle_tempparam['stride_time_contralateral']),decimals=3)
        cycle_tempparam['stride_time_contralateral_Cov']=np.around(np.std(cycle_tempparam['stride_time_contralateral']*100)/np.mean(cycle_tempparam['stride_time_contralateral']),decimals=3)

        cycle_tempparam['stridetime_std']=np.around(np.std(cycle_tempparam['stridetime']),decimals=3)
        cycle_tempparam['stridetime_Cov']=np.around(np.std(cycle_tempparam['stridetime']*100)/np.mean(cycle_tempparam['stridetime']),decimals=3)

        cycle_tempparam['steptime_std']=np.around(np.std(cycle_tempparam['steptime']),decimals=3)
        cycle_tempparam['steptime_Cov']=np.around(np.std(cycle_tempparam['steptime']*100)/np.mean(cycle_tempparam['steptime']),decimals=3)


    else:
        print("most strides have been filtered because of misdetection ")
    cycle_temp=cycle_tempparam
    return cycle_temp

if __name__=="__main__":
    # subj1_steps=np.load('subj1_steps.npy')
    # var_s1_dff,stride_s1_dff=calculate_variability_activeperiod(subj1_steps,minimum_seperationwalkingbout=200,longest_walkingbout=60)
    
    
    
    subj2_steps=np.load('JBK411_steps.npy')
    var_s2_df,stride_s2_df=calculate_variability_activeperiod(subj2_steps,minimum_seperationwalkingbout=200,longest_walkingbout=60)
    
    # var_s2_df.to_excel('result_JBK411.xlsx')
    # filename="d://Users//al-abiad//Desktop//kim//Naima & Thomas//JBK411.mat"
    # calculate_lyap(filename,subj2_steps,Numberofstrides=100,tao=10,dim=5)
    
    
    