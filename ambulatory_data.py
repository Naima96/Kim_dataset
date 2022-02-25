# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:41:18 2022

@author: al-abiad
"""

import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from numba import jit, njit

from SmartStep_optim_version2 import *

# import time
# from progressbar import ProgressBar
import onnxruntime as rt
rt.set_default_logger_severity(3)
import warnings
warnings.filterwarnings('ignore')
# from mat4py import loadmat 
import scipy.io

A = np.asarray([1, -8.99594505535417, 36.4633075498862, -87.6908102873418, 138.559912556039, -150.302116273606, 113.348650426591, -58.6790854259776, 19.9562604962225, -4.02604302102184, 0.365869040210561]);
B = np.asarray([5.51456216845228e-12, 5.51456216845228e-11, 2.48155297580353e-10, 6.61747460214274e-10, 1.15805805537498e-09, 1.38966966644997e-09, 1.15805805537498e-09, 6.61747460214274e-10, 2.48155297580353e-10, 5.51456216845228e-11, 5.51456216845228e-12]);


def lowpass(signal, dt, fc):
    """Low pass filter. Based on constant sampling hypothesis.
    
    :param array signal: signal
    :param float dt: sampling period [s]
    :param float fc: cutoff frequency for the low pass filter [Hz]
    """
    RC = 1/fc
    a = dt / (RC + dt)
    outsig = np.zeros(len(signal))
    outsig[0] = signal[0]
    # for tt=2:size(sig,1)
    # pbsig(tt) = a*sig(tt) + (1-a)*pbsig(tt-1)
    
    for tt in range(len(signal)-1):
        outsig[tt+1] = a*signal[tt+1]+ (1-a)*outsig[tt]
    
    return outsig 


class Ambulatory_IMU(object):
    
    def __init__(self,filename):
        """
        

        Parameters
        ----------
        filename : String
            Path to the file to analyze

        Returns
        -------
        None.

        """
        
        self.sess_acc = rt.InferenceSession("acc.onnx")
        self.input_acc = self.sess_acc.get_inputs()[0].name
        
        # data = loadmat(filename)
        # data=pd.DataFrame(data['signal'],columns=["Accx","Accy","Accz"])

        data = scipy.io.loadmat(filename)['signal']
        data=np.c_[ np.arange(0,len(data)), data  ]  
       
        # data=pd.DataFrame(mat['signal'],columns=['Accx','Accy','Accz'])
        
        activity_time=self.delete_nonactivity_periods(data)
        
        print("we have %.2f percent of active periods"%(np.sum(activity_time)/len(data)))
        
        activity_time=np.squeeze(activity_time)
        
        activity=np.where(activity_time==1)[0]
        consecutive_index_activity=self.find_consecutive_index_ranges(activity, 
                                                                      increment = 1)
        
        print("we found %d number of active periods"%(len(consecutive_index_activity)))
        
        self.active_periods_IMU= [None] * len(consecutive_index_activity)
        
        for i in range(0,len(consecutive_index_activity)):
            self.active_periods_IMU[i]=data[consecutive_index_activity[i],:]


    def detect_walking_period(self):
        
                    
        self.all_gait_segment=[None] * len(self.active_periods_IMU)
            
        l=0
        for active_period in self.active_periods_IMU:
            start=int(active_period[0,0])

            
            active_period_split=np.array_split(active_period, 
                                               len(active_period)//(60*100))
            
            print("we found %d number of 60 sec slices in active period"%(len(active_period_split)))
            
            self.gait_segment = []
            
            j=0
            
            wind_step_active=np.full(shape = len(active_period), fill_value = 0, dtype = 'uint8')
            

            for sixty_sec_active in active_period_split:
                

                print("processing the %d active period"%(j))
                
                gait=0
                
                first_one_sec=sixty_sec_active[:256,1:]
                first_one_sec_filtered = filter_data(acc=first_one_sec,B=B,A=A)
            
                second_one_sec=sixty_sec_active[30*100:30*100+256,1:]
                second_one_sec_filtered = filter_data(acc=second_one_sec,B=B,A=A)
                
                third_one_sec=sixty_sec_active[-256:,1:]
                third_one_sec_filtered = filter_data(acc=third_one_sec,B=B,A=A)
                
                for k in range(127,255,1):

                    if self.cal_feature_predict_step(first_one_sec,first_one_sec_filtered,t=k):
                        gait=1 
                        print("it is a gait period")
                        break
    
                    if self.cal_feature_predict_step(second_one_sec,second_one_sec_filtered,t=k):
                        gait=1
                        print("it is a gait period")
                        break
                        
                    if self.cal_feature_predict_step(third_one_sec,third_one_sec_filtered,t=k):
                        gait=1
                        print("it is a gait period")
                        break
                    
                if gait==1:
                    print("detecting steps")
                    
                    # sixty_sec_active=sixty_sec_active.reset_index(drop=True)
                    wind_step=np.full(shape = len(sixty_sec_active), fill_value = 0, dtype = 'uint8')

                    wind_ind=127
                    
                    data_filtered = filter_data(acc=sixty_sec_active[:,1:],B=B,A=A)
                    
                    while wind_ind<len(sixty_sec_active)-1:

                        
                        step_result=self.cal_feature_predict_step(sixty_sec_active[:,1:],data_filtered,t=wind_ind)
                        
                        wind_step[wind_ind-64]=step_result
                        
                        if step_result==1:
                            wind_step[wind_ind-63:wind_ind-63+30]=0
                            wind_ind=wind_ind+30
                        else:
                            wind_ind=wind_ind +1
                            
                        if wind_ind%3000==0:
                            print("finished %d out of 6000"%(wind_ind))
                

                    
                    wind_step_active[int(sixty_sec_active[0,0])-start:int(sixty_sec_active[-1,0])+1-start]=wind_step
                    print("we found %d steps"%(np.sum(wind_step)))

                    

                j=j+1
                # print(j)
                # if j==2:
                #     break
                
                
            self.all_gait_segment[l]=wind_step_active
            
            l=l+1
            print(l)
            print("finished analysis of %d active period"%(l))
            
            if l==2:
                break



        


    def find_consecutive_index_ranges(self,vector, increment = 1):
    	"""
    	Find ranges of consequetive indexes in numpy array
    	Parameters
    	---------
    	data: numpy vector
    		numpy vector of integer values
    	increment: int (optional)
    		difference between two values (typically 1)
    	Returns
    	-------
    	indexes : list
    		list of ranges, for instance [1,2,3,4],[8,9,10], [44]
    	"""
    
    	return _find_consecutive_index_ranges(vector=vector, increment = increment)
    

    def split_into_epochs(self,sample_values, epoch_time_interval=5, return_indices=False):
        """
        Split the given ndarray data (e.g. [[time,accel_x,accel_y,accel_y,*_]])
        ...based on the timestamps array (will use the first column if not given)
        ...into a list of epochs of the specified time interval.
        """
        # timestamps=sample_values.index/100
        timestamps=sample_values[:,0]/100
        
        epoch_time_offset = timestamps[0]
        
        # Quantize into interval numbers
        epoch_time_index = (timestamps - epoch_time_offset) // epoch_time_interval
        
        # Calculate a mask where the index has changed from the previous one
        epoch_is_different_index = np.concatenate(([False], epoch_time_index[1:] != epoch_time_index[0:-1]))
    
        # Find the index of each change of epoch
        epoch_indices = np.nonzero(epoch_is_different_index)[0]
    
        # Split into epochs
        epochs = np.array_split(sample_values, epoch_indices, axis=0)
    
        del epoch_time_index
        del epoch_is_different_index
    
        if return_indices:
            # Include index of first epoch
            epoch_indices = np.insert(epoch_indices, 0, [0], axis=None)
            return (epochs, epoch_indices)
        else:
            del epoch_indices
            return epochs
        
    def calculate_svm(self,sample_values, epoch_time_interval=5, truncate=False):
        """
        Calculate the mean(abs(SVM-1)) value for the given sample ndarray [[time_seconds, accel_x, accel_y, accel_z]]
        
        :param epoch_time_interval: seconds per epoch (typically 60 seconds)
        :param relative_to_time: None=align epochs to start of data, 0=align epochs to natural time, other=custom alignment
        :param truncate: If true, use max(SVM-1,0) rather than abs(SVM-1)
        :returns: ndarray of [time,svm]
        """
    
        # Split samples into epochs
        epochs = self.split_into_epochs(sample_values, epoch_time_interval)
    
        # Calculate each epoch

        
        result=loop_svm(epochs,truncate=truncate)

        return result
    


        
    
    def detect_inactivity_periods(self,sample_values,result,thresh=7):
        fs=100
        result[:,0]=result[:,0]-result[0,0]
        non_activity_vector1 = np.full(shape = sample_values, fill_value = 1, dtype = 'uint8')
        
        thresh=thresh/1000
        
        for i in range(0, len(result)):
            start=int(result[i,0])
            end=int(result[i,0]+5*fs)
            # check if the standard deviation is below the threshold, and if the number of axes the standard deviation is below equals the std_min_num_axes threshold
            if (result[i,1] < thresh):
                non_activity_vector1[start:end] = 0
    
        return non_activity_vector1
    
    def delete_nonactivity_periods(self,data):
        
        non_wear_vector=_hees_2013_calculate_non_wear_time(data[:,1:])
        index_non_wear=np.where(non_wear_vector==1)[0]
        

        consecutive_index_non_wear=_find_consecutive_index_ranges(index_non_wear, increment = 1)
        
        
        activity_time = np.full(shape = [len(data), 1], fill_value = 0, dtype = 'uint8')
        
        print("the data has a length of %d"%len(data))
        
        for i in range(0,len(consecutive_index_non_wear)):
            print("finished %d out of %d"%(i,len(consecutive_index_non_wear)))
            
            print("the data has a length of %d"%(consecutive_index_non_wear[i][-1]-consecutive_index_non_wear[i][0]))
            
            results=self.calculate_svm(data[consecutive_index_non_wear[i][0]:consecutive_index_non_wear[i][-1],:],
                                  epoch_time_interval=1,truncate=True)
    
    
            non_activity_vector=self.detect_inactivity_periods(
                consecutive_index_non_wear[i][-1]-consecutive_index_non_wear[i][0],
                results,thresh=7)
            
            
            index_non_activity=np.where(non_activity_vector==1)[0]
            
            consecutive_index_non_activity=_find_consecutive_index_ranges(index_non_activity, increment = 60*1000)

            
            print("found %d active periods"%(len(consecutive_index_non_activity)))
            consecutive_index_non_activity=[act for act in consecutive_index_non_activity if len(act)>=60*100]
            print("kept %d"%(len(consecutive_index_non_activity)))
            
            for k in range(0,len(consecutive_index_non_activity)):
                activity_time[consecutive_index_non_activity[k][0]+consecutive_index_non_wear[i][0]:consecutive_index_non_activity[k][-1]+consecutive_index_non_wear[i][0]]=1

        return(activity_time)
    

    
    def cal_feature_predict_step(self,data,data_filtered,t):
        
        data=data*9.8
        
        data_filtered=data_filtered*9.8

        maxprob=0.5
        
        

        # y_prob=DETECT_STEP().LGBM(imudata=data,iterate=t,model_acc=self.sess_acc,input_acc=self.input_acc,B=B,A=A)
        
        y_prob=DETECT_STEP().LGBM(imudata=data,imudata_filtered=data_filtered,
                                  iterate=t,model_acc=self.sess_acc,input_acc=self.input_acc,B=B,A=A)
        
        
        
        
        step_instant=int(np.where(y_prob>maxprob,1,0))
        
        return step_instant
        
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
       
if __name__=="__main__":
    
    plt.close('all')
    filename="d://Users//al-abiad//Desktop//kim//Naima & Thomas//sig_try.mat"
    
    amb_data=Ambulatory_IMU(filename)
    
    amb_data.detect_walking_period()

    