# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:00:46 2022

@author: al-abiad
"""
from ambulatory_data import Ambulatory_IMU as amb
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit,vectorize 
import scipy.io
from scipy import signal

filename="d://Users//al-abiad//Desktop//kim//Naima & Thomas//sig_try.mat"

A = np.asarray([1, -8.99594505535417, 36.4633075498862, -87.6908102873418, 138.559912556039, -150.302116273606, 113.348650426591, -58.6790854259776, 19.9562604962225, -4.02604302102184, 0.365869040210561]);
B = np.asarray([5.51456216845228e-12, 5.51456216845228e-11, 2.48155297580353e-10, 6.61747460214274e-10, 1.15805805537498e-09, 1.38966966644997e-09, 1.15805805537498e-09, 6.61747460214274e-10, 2.48155297580353e-10, 5.51456216845228e-11, 5.51456216845228e-12]);
   

x = scipy.io.loadmat(filename)['signal']
x=np.c_[ np.arange(0,len(x)), x ]  



def split_into_epochs(sample_values, epoch_time_interval=5, return_indices=False):
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
    
def calculate_svm(sample_values, epoch_time_interval=5, truncate=False):
    """
    Calculate the mean(abs(SVM-1)) value for the given sample ndarray [[time_seconds, accel_x, accel_y, accel_z]]
    
    :param epoch_time_interval: seconds per epoch (typically 60 seconds)
    :param relative_to_time: None=align epochs to start of data, 0=align epochs to natural time, other=custom alignment
    :param truncate: If true, use max(SVM-1,0) rather than abs(SVM-1)
    :returns: ndarray of [time,svm]
    """

    # Split samples into epochs
    epochs = split_into_epochs(sample_values, epoch_time_interval)

    # Calculate each epoch

    
    result=loop_svm(epochs)
    # for epoch_index in range(num_epochs):
    #     this_epoch = epochs[epoch_index]

    #     # Epoch start time and sample data
    #     epoch_time = this_epoch[0,0]
    #     samples = this_epoch[:,1:]
        
    #     # Calculate Euclidean norm minus one 
    #     samples_enmo = np.sqrt(np.sum(samples * samples, axis=1)) - 1
    #     # samples_enmo = np.linalg.norm(samples,ord=2,axis=1)  - 1

    #     # This scalar vector magnitude approach takes the absolute value
    #     if truncate:
    #         samples_svm = samples_enmo
    #         samples_svm[samples_svm < 0] = 0
    #     else:
    #         samples_svm = np.abs(samples_enmo)

    #     # Mean of the value
    #     epoch_value = np.mean(samples_svm)

    #     # Result
    #     result[epoch_index,0] = epoch_time
    #     result[epoch_index,1] = epoch_value

    return result

@njit
def loop_svm(epochs):
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

@njit
def _find_consecutive_index_ranges(vector, increment = 1):
	return np.split(vector, np.where(np.diff(vector) > increment)[0]+1) 


def _hees_2013_calculate_non_wear_time(data, hz = 100, min_non_wear_time_window = 60, window_overlap = 15, std_mg_threshold = 3, std_min_num_axes = 1,
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
    # convert the value range threshold from g to m/s^2
# 	value_range_mg_threshold=value_range_mg_threshold*9.80665 

	# new array to record non-wear time. The default behavior is 0 = non-wear time, and 1 = wear time. Since we create a new array filled with wear time encoding, we only have to 
	# deal with non-wear time, since everything else is already set as wear-time.
	non_wear_vector = np.full(shape = data.shape[0], fill_value = wt_encoding, dtype = 'uint8')

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


        
		std=np.array([np.std(subset_data[:,0]),
                      np.std(subset_data[:,1]),
                      np.std(subset_data[:,2])])


		# check if the standard deviation is below the threshold, and if the number of axes the standard deviation is below equals the std_min_num_axes threshold
		if (std < std_mg_threshold).sum() >= std_min_num_axes:

			# at least 'std_min_num_axes' are below the standard deviation threshold of 'std_min_num_axes', now set this subset of the data to the non-wear time encoding.
			# Note that the full 'new_wear_vector' is pre-populated with the wear time encoding, so we only have to set the non-wear time.
			non_wear_vector[start:end] = nwt_encoding

		# calculate the value range (difference between the min and max) (here the point-to-point numpy method is used) for each column



		value_range=np.array([np.ptp(subset_data[:,0]),np.ptp(subset_data[:,1]),np.ptp(subset_data[:,2])])
		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
		if (value_range < value_range_mg_threshold).sum() >= value_range_min_num_axes:

			# set the non wear vector to non-wear time for the start to end slice of the data
			non_wear_vector[start:end] = nwt_encoding

	return non_wear_vector

wt=_hees_2013_calculate_non_wear_time(x[:,1:])

index_non_wear=np.where(wt==1)[0]



epochs = split_into_epochs(x[:30000,], 5)


calculate_svm(x[:30000,1:])

z=np.array([0,0,0,1,1,1,0,0,0,1,0,1])
yp=np.where(y==1)[0]

_find_consecutive_index_ranges(index_non_wear, increment = 3)

sample_values=x[:30000,1:]

timestamps=sample_values[:,0]/100

epoch_time_offset = timestamps[0]

# Quantize into interval numbers
epoch_time_index = (timestamps - epoch_time_offset) // epoch_time_interval


hees_2013_calculate_non_wear_time(x[:300,1:])
subset_data=x[:300,1:]
std=np.array([np.std(subset_data[:,0]),np.std(subset_data[:,1]),np.std(subset_data[:,2])])

np.std(x[:300,1:], axis=0)

calc_sma(data)

_dominant_frequency(data)

list(map(abs, x[:300,1:2]))

np.argmax(data)

data=x[:300,1:2]
# calculate_norm_accandgyro(self,acc=None)



# amb_data.detect_walking_period()

# x=amb_data.gait_segment

# np.where(x[0]==1)[0]




# plt.figure()
# period=amb_data.active_periods_IMU[0][6030:6030+6030,1]
# plt.plot(period)
# plt.scatter(np.where(x[1]==1)[0],period[np.where(x[1]==1)[0]])

# amb_data.active_periods_IMU[0]

# np.repeat(amb_data.all_gait_segment[0], 60*100)

# plt.figure()
# plt.plot(amb_data.active_periods_IMU[0])



# active_period_split=np.array_split(amb_data.active_periods_IMU[0]
# , len(amb_data.active_periods_IMU[0])//(60*100))




























non_wear_vector=self.hees_2013_calculate_non_wear_time(data[:,1:])


