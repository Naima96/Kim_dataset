# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:12:39 2022

@author: al-abiad
"""
from numba import jit, njit
import numpy as np
from SmartStep_optim_version2 import *
from scipy import signal
import onnxruntime as rt
rt.set_default_logger_severity(3)

A = np.asarray([1, -8.99594505535417, 36.4633075498862, -87.6908102873418, 138.559912556039, -150.302116273606, 113.348650426591, -58.6790854259776, 19.9562604962225, -4.02604302102184, 0.365869040210561]);
B = np.asarray([5.51456216845228e-12, 5.51456216845228e-11, 2.48155297580353e-10, 6.61747460214274e-10, 1.15805805537498e-09, 1.38966966644997e-09, 1.15805805537498e-09, 6.61747460214274e-10, 2.48155297580353e-10, 5.51456216845228e-11, 5.51456216845228e-12]);
sess_acc = rt.InferenceSession("acc.onnx")
input_acc = sess_acc.get_inputs()[0].name

def filter_data(acc=[0],B=[],A=[]):

    y = np.copy(acc)

    y = signal.filtfilt(B, A, y,axis=0) # fix
   
    return y

        
@njit
def _find_consecutive_index_ranges(vector, increment = 1):
	return np.split(vector, np.where(np.diff(vector) > increment)[0]+1) 


def _hees_2013_calculate_non_wear_time(data, hz = 100, min_non_wear_time_window = 60, window_overlap = 15, std_mg_threshold = 3, std_min_num_axes = 1,\
										value_range_mg_threshold = 50, value_range_min_num_axes = 1, nwt_encoding = 0, wt_encoding = 1):
 	# number of data samples in 1 minute
 	num_samples_per_min = hz * 60

 	# define the correct number of samples for the window and window overlap
 	min_non_wear_time_window *= num_samples_per_min
 	window_overlap *= num_samples_per_min

 	# convert the standard deviation threshold from mg to g
 	std_mg_threshold /= 1000

     
    #convert the standard deviation from g to m/s^2
# 	std_mg_threshold=std_mg_threshold*9.80665 
    
 	# convert the value range threshold from mg to g
 	value_range_mg_threshold /= 1000

 	non_wear_vector = np.full(shape = data.shape[0], fill_value = wt_encoding, dtype = 'uint8')
 	non_std = np.full(shape = (data.shape[0],3), fill_value = wt_encoding, dtype = 'float64')

 	# loop over the data, start from the beginning with a step size of window overlap
 	for i in range(0, len(data), window_overlap):

		# define the start of the sequence
 	 	start = i
		# define the end of the sequence
 	 	end = i + min_non_wear_time_window

		# slice the data from start to end
 	 	subset_data = data[start:end]
        

		# check if the data sequence has been exhausted, meaning that there are no full windows left in the data sequence (this happens at the end of the sequence)
		# comment out if you want to use all the data
 	 	if len(subset_data) < min_non_wear_time_window:
 	 	 	break

        
 	 	std_value=np.array([np.std(subset_data[:,0]),np.std(subset_data[:,1]),np.std(subset_data[:,2])])
 	 	non_std[start:end,:]= std_value
          
		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
 	 	if np.sum(std_value < std_mg_threshold) >= std_min_num_axes:

 	 	 	non_wear_vector[start:end] = nwt_encoding          
          
            

 	 	value_range=np.array([np.ptp(subset_data[:,0]),np.ptp(subset_data[:,1]),np.ptp(subset_data[:,2])])

		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
 	 	if np.sum(value_range < value_range_mg_threshold) >= value_range_min_num_axes:

 	 	 	non_wear_vector[start:end] = nwt_encoding

 	return non_wear_vector
 
@njit        
def loop_svm(epochs,truncate):
    num_epochs = len(epochs)
    result = np.empty((num_epochs,2))
    
    for epoch_index in range(num_epochs):
        this_epoch = epochs[epoch_index]

        # Epoch start time and sample data
        epoch_time = this_epoch[0,0]
        samples = this_epoch[:,1:]
        
        # Calculate Euclidean norm minus one 
        samples_enmo = np.sqrt(np.sum(samples * samples, axis=1)) - 1
        # samples_enmo = np.linalg.norm(samples,ord=2,axis=1)  - 1

        # This scalar vector magnitude approach takes the absolute value
        if truncate:
            samples_svm = samples_enmo
            samples_svm[samples_svm < 0] = 0
        else:
            samples_svm = np.abs(samples_enmo)

        # Mean of the value
        epoch_value = np.mean(samples_svm)

        # Result
        result[epoch_index,0] = epoch_time
        result[epoch_index,1] = epoch_value
    return result


def cal_feature_predict_step(data,data_filtered,t):
    
    data=data*9.8
    
    data_filtered=data_filtered*9.8

    maxprob=0.5
    
    

    # y_prob=DETECT_STEP().LGBM(imudata=data,iterate=t,model_acc=self.sess_acc,input_acc=self.input_acc,B=B,A=A)
    
    y_prob=DETECT_STEP().LGBM(imudata=data,imudata_filtered=data_filtered,
                              iterate=t,model_acc=sess_acc,input_acc=input_acc,B=B,A=A)
    
    step_instant=int(np.where(y_prob>maxprob,1,0))
    
    return step_instant







def _detect_walk_sixty_sec(sixty_sec_active):

    gait=0
    wind_step=np.full(shape = len(sixty_sec_active), fill_value = 0, dtype = 'uint8')

    first_one_sec=sixty_sec_active[:256,1:]
    first_one_sec_filtered = filter_data(acc=first_one_sec,B=B,A=A)

    second_one_sec=sixty_sec_active[30*100:30*100+256,1:]
    second_one_sec_filtered = filter_data(acc=second_one_sec,B=B,A=A)
    
    third_one_sec=sixty_sec_active[-256:,1:]
    third_one_sec_filtered = filter_data(acc=third_one_sec,B=B,A=A)
    
    print("we are checking if it is a gait period")
    for k in range(127,255,1):

        if cal_feature_predict_step(first_one_sec,first_one_sec_filtered,t=k):
            gait=1 
            print("it is a gait period")
            break

        if cal_feature_predict_step(second_one_sec,second_one_sec_filtered,t=k):
            gait=1
            print("it is a gait period")
            break
            
        if cal_feature_predict_step(third_one_sec,third_one_sec_filtered,t=k):
            gait=1
            print("it is a gait period")
            break
        
    if gait==1:
        print("detecting steps")
        
        # sixty_sec_active=sixty_sec_active.reset_index(drop=True)

        wind_ind=127
        
        data_filtered = filter_data(acc=sixty_sec_active[:,1:],B=B,A=A)
        
        while wind_ind<len(sixty_sec_active)-1:

            
            step_result=cal_feature_predict_step(sixty_sec_active[:,1:],data_filtered,t=wind_ind)
            
            wind_step[wind_ind-64]=step_result
            
            if step_result==1:
                wind_step[wind_ind-63:wind_ind-63+30]=0
                wind_ind=wind_ind+30
            else:
                wind_ind=wind_ind +1
                
            if wind_ind%3000==0:
                print("finished %d out of 6000"%(wind_ind))

    print("we found %d steps"%(np.sum(wind_step)))
        
    return wind_step