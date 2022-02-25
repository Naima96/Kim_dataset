# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:41:18 2022

@author: al-abiad
"""

import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

from utility_functions2 import _find_consecutive_index_ranges,_hees_2013_calculate_non_wear_time,loop_svm,_detect_walk_sixty_sec


# import time
# from progressbar import ProgressBar

import warnings
warnings.filterwarnings('ignore')
# from mat4py import loadmat 
import scipy.io




# def lowpass(signal, dt, fc):
#     """Low pass filter. Based on constant sampling hypothesis.
    
#     :param array signal: signal
#     :param float dt: sampling period [s]
#     :param float fc: cutoff frequency for the low pass filter [Hz]
#     """
#     RC = 1/fc
#     a = dt / (RC + dt)
#     outsig = np.zeros(len(signal))
#     outsig[0] = signal[0]
#     # for tt=2:size(sig,1)
#     # pbsig(tt) = a*sig(tt) + (1-a)*pbsig(tt-1)
    
#     for tt in range(len(signal)-1):
#         outsig[tt+1] = a*signal[tt+1]+ (1-a)*outsig[tt]
    
#     return outsig 


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
            
    # def detect_walk_sixty_sec(self,sixty_sec_active):
    #     return _detect_walk_sixty_sec(sixty_sec_active)
        


    def detect_walking_period(self):
        
                    
        self.all_gait_segment=[None] * len(self.active_periods_IMU)
        pool=Pool(8)   
        l=0
        for active_period in self.active_periods_IMU:


            
            active_period_split=np.array_split(active_period, 
                                               len(active_period)//(60*100))
            
            
            
            print("we found %d number of 60 sec slices in active period"%(len(active_period_split)))
            
            self.gait_segment = []
            

            # wind_step_active=np.full(shape = len(active_period), fill_value = 0, dtype = 'uint8')
            
            self.results=pool.map_async(_detect_walk_sixty_sec, active_period_split ).get()

                
            self.all_gait_segment[l]=self.results
            
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
    

    

        

    
       
if __name__=="__main__":
    
    plt.close('all')
    filename="d://Users//al-abiad//Desktop//kim//Naima & Thomas//sig_try.mat"
    
    amb_data=Ambulatory_IMU(filename)
    
    amb_data.detect_walking_period()

    amb_data.results
