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
import os 

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
    
    

def calculate_variability_activeperiod(steps_vector,minimum_seperationwalkingbout=100,longest_walkingbout=60,remove_step=0):
    

    # group steps in same walking bouts
    consecutive_index_steps=_find_consecutive_index_ranges(steps_vector, increment = minimum_seperationwalkingbout)
    
    # print("found %d walking periods"%(len(consecutive_index_steps)))
    
    # keep walking bouts with more than 60 steps
    consecutive_index_steps=[act for act in consecutive_index_steps if len(act)>=longest_walkingbout]
    # print("kept %d more than 60 steps"%(len(consecutive_index_steps)))
    
    columns=['time_start','time_end','N_steps','stridetime_std','stridetime_Cov','steptime_std','steptime_Cov']

    variability_data=pd.DataFrame(columns=columns) 
    
    detailed_stridetime= np.empty((0,3))

    for j in range(0,len(consecutive_index_steps)):
        variability=computeVarStride(fs=100,remove_outliers=True,N=3,use_smartstep=True,
                                     manual_peaks=consecutive_index_steps[j],use_peaks=True,pocket=False,remove_step=remove_step)
        
        try:

            
            if len(variability)>1:
                
                if variability['detailed_steptime'][0,0]==variability['detailed_steptime'][0,0]:
                
                    time_start=variability['detailed_steptime'][0,0]
                    time_end=variability['detailed_steptime'][-1,0]
                    
                    data=[time_start,time_end,len(variability['detailed_steptime']),
                          variability['stridetime_std'],variability['stridetime_Cov'],
                          variability['steptime_std'],variability['steptime_Cov']]
                    
                    var_data=pd.DataFrame([data],columns=columns)
                    
                    variability_data= pd.concat([variability_data,var_data])
                    
                    detailed_stridetime=np.concatenate((detailed_stridetime,variability['detailed_stridetime']))
                    
                    
                else:
                    
                    data=[np.NAN,np.NAN,np.NAN,
                          np.NAN,np.NAN,
                          np.NAN,np.NAN]
                    
                    var_data=pd.DataFrame([data],columns=columns)
                    
                    variability_data= pd.concat([variability_data,var_data])
                    
                    
                    
                #append to dictionary 
        except:
            print("problem with analysis")

            
    return variability_data,detailed_stridetime


def computeVarStride(fs=100,remove_outliers=True,N=1,use_smartstep=False,manual_peaks=[],
                     use_peaks=True,pocket=True,remove_step=0,round_data=True):

    cycle_tempparam = {}

    peaks=manual_peaks[remove_step:-remove_step]


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
        
        #---step time---
        try:
            list_step_time=np.vstack([i for i in list_step_time if i[2]>=0.3 and i[2]<=0.9])
            mean=np.mean(list_step_time[:,2])
            cut_off=N*np.std(list_step_time[:,2])
            lower, upper =  mean- cut_off, mean + cut_off
            cycle_tempparam['steptime'] = np.array([i for i in list_step_time[:,2] if i > lower and i < upper])
        except Exception:
            print("no step time left")
            cycle_tempparam['steptime'] =np.array([np.NAN])

        try:
            list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=0.8 and i[2]<=1.7])
            mean=np.mean(list_stride_time[:,2])
            cut_off=N*np.std(list_stride_time[:,2])
            lower, upper =  mean- cut_off, mean + cut_off
            list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=lower and i[2]<=upper])
        except:
            print("no step time left")
            cycle_tempparam['stridetime'] =np.array([np.NAN])


    #---merge left right stride cycle


    if len(list_stride_time[:,2])>1:
        cycle_tempparam['stridetime']=list_stride_time[:,2]
        cycle_tempparam["detailed_stridetime"]=list_stride_time
        cycle_tempparam["detailed_steptime"]=list_step_time

        cycle_tempparam['stridetime_std']=np.around(np.std(cycle_tempparam['stridetime']),decimals=3)
        cycle_tempparam['stridetime_Cov']=np.around(np.std(cycle_tempparam['stridetime']*100)/np.mean(cycle_tempparam['stridetime']),decimals=3)

        cycle_tempparam['steptime_std']=np.around(np.std(cycle_tempparam['steptime']),decimals=3)
        cycle_tempparam['steptime_Cov']=np.around(np.std(cycle_tempparam['steptime']*100)/np.mean(cycle_tempparam['steptime']),decimals=3)


    else:
        print("most strides have been filtered because of misdetection ")
        cycle_tempparam['stridetime']=np.array([np.NAN])
        cycle_tempparam["detailed_stridetime"]=np.array([[np.NAN,np.NAN],[np.NAN,np.NAN]])
        cycle_tempparam["detailed_steptime"]=np.array([[np.NAN,np.NAN],[np.NAN,np.NAN]])

        cycle_tempparam['stridetime_std']=np.NAN
        cycle_tempparam['stridetime_Cov']=np.NAN

        cycle_tempparam['steptime_std']=np.NAN
        cycle_tempparam['steptime_Cov']=np.NAN


        
        
                
    cycle_temp=cycle_tempparam
    return cycle_temp


def read_faller_excel(subjects):
    file_name="ST_Falls_unblinded_Naima.xlsx"
    faller_excel=pd.read_excel(file_name,usecols="A,E")
    subjects_names=result.subject.values
    
    s=subjects_names[0]
    faller=[]
    
    
    for s in subjects_names:
        
        f=faller_excel.loc[faller_excel['StudyID']==s.replace(" ", ""),'N_falls_year1'].values
        if len(f)>=1:
            if f>=1:
            
                faller.append([1])
            else:
                faller.append([0])
            
        else:
            faller.append([np.NAN])
            
    faller=np.vstack(faller)
    
    result["prosp_falls"]=faller
    

if __name__=="__main__":
    # subj1_steps=np.load('subj1_steps.npy')
    # var_s1_dff,stride_s1_dff=calculate_variability_activeperiod(subj1_steps,minimum_seperationwalkingbout=200,longest_walkingbout=60)
    
    columns=["subject","No. bouts","No. steps","short percen",
             "med percen","long percen","SD short","SD med","SD long",
             "Cov short","Cov med","Cov long"]
    
    direct_path="results"
    
    files=[f for f in os.listdir(direct_path) if os.path.isfile(os.path.join(direct_path,f))]
    result_list=[]
    
    short=60
    mid=100
    long=200

    for f in files:
        
        if f.endswith(".npy"):
        
            filepath=direct_path +"//" +f
    
            subj2_steps=np.load(filepath)
            var_s2_df,stride_s2_df=calculate_variability_activeperiod(subj2_steps,minimum_seperationwalkingbout=200,longest_walkingbout=short)
        
            subject_file=f[:-10]
            number_walking_bout= len(var_s2_df)
            total_number_steps= np.sum(var_s2_df["N_steps"].values)
            
            percen_short= np.sum(var_s2_df["N_steps"].values<mid)*100/number_walking_bout
            
            percen_med=np.sum((var_s2_df["N_steps"].values>mid) & (var_s2_df["N_steps"].values<long))*100/number_walking_bout
            
            percen_long= np.sum(var_s2_df["N_steps"].values>long)*100/number_walking_bout
            
            var_short_SD=np.median(var_s2_df.loc[var_s2_df["N_steps"]<mid,"stridetime_std"].values)
            
            var_med_SD=np.median(var_s2_df.loc[(var_s2_df["N_steps"]>mid) & 
                                                  (var_s2_df["N_steps"]<long) ,"stridetime_std"].values)
            
            
            var_long_SD=np.median(var_s2_df.loc[var_s2_df["N_steps"]>long,"stridetime_std"].values)
            
            
            var_short_Cov=np.median(var_s2_df.loc[var_s2_df["N_steps"]<mid,"stridetime_Cov"].values)
            
            var_med_Cov=np.median(var_s2_df.loc[(var_s2_df["N_steps"]>mid) & 
                                                  (var_s2_df["N_steps"]<long) ,"stridetime_Cov"].values)
            
            
            var_long_Cov=np.median(var_s2_df.loc[var_s2_df["N_steps"]>long,"stridetime_Cov"].values)
            
            
            result_list.append([subject_file,number_walking_bout,
                                total_number_steps,percen_short,percen_med,percen_long,
                                var_short_SD,var_med_SD,var_long_SD,var_short_Cov,var_med_Cov,var_long_Cov])
            

    result=pd.DataFrame(data=result_list, columns=columns)
    # var_s2_df.to_excel('result_JBK411.xlsx')
    # filename="d://Users//al-abiad//Desktop//kim//Naima & Thomas//JBK411.mat"
    # calculate_lyap(filename,subj2_steps,Numberofstrides=100,tao=10,dim=5)
    
    # result.to_excel("result.xlsx")
    
    subjects_names=result.subject.values
    
    file_name="ST_Falls_unblinded_Naima.xlsx"
    faller_excel=pd.read_excel(file_name,usecols="A,E")
    subjects_names=result.subject.values
    
    s=subjects_names[0]
    faller=[]
    
    
    for s in subjects_names:
        
        f=faller_excel.loc[faller_excel['StudyID']==s.replace(" ", ""),'N_falls_year1'].values
        if len(f)>=1:
            if f>=1:
            
                faller.append([1])
            else:
                faller.append([0])
            
            
        else:
            faller.append([np.NAN])
            
    faller=np.vstack(faller)
    
    result["prosp_falls"]=faller
    
    
    result = result[result['prosp_falls'].notna()]
    
    print("There are %d fallers "%np.sum(result['prosp_falls'].values))
    
    
    print("There are %d nonfallers "%(len(result['prosp_falls'].values)-np.sum(result['prosp_falls'].values)))
    
    
    import seaborn as sns
    for c in result.columns[1:-1]:
        plt.figure()
        sns.boxplot(x="prosp_falls", y=c,
                          data=result)
    