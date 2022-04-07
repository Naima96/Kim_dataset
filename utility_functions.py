# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:48:52 2022

@author: al-abiad
"""

import numpy as np
from scipy import signal
from numba import njit


def calc_skew(a):

    mean = np.mean(a)
    m2 = _moment(a, 2,mean=mean)
    m3 = _moment(a, 3,mean=mean)

    vals = m3 / m2**1.5
    skew=vals

    return skew


def calc_kurt(a):

    mean = np.mean(a)
    m2 = _moment(a, 2,mean=mean)
    m4 = _moment(a, 4, mean=mean)
    valsk = m4 / m2**2.0
    kurt=valsk - 3 
    return kurt

def dominant_frequency(signal_x): #100
    sampling_rate=100
    nfft=1024
    nfft2=512
    fmin=0.5
    fmax=4

    signal_x = signal_x-np.mean(signal_x)
    dim = signal_x.shape
    
    freq = (np.fft.fftfreq(nfft) * sampling_rate)[0:nfft2]
     
    lowind=np.where(freq>fmin)[0][0]
    upind=np.max(np.where(freq<fmax))

    haming= np.hamming(dim[0])
    sp_hat = np.fft.fft(signal_x*haming, nfft)
    
    furval = sp_hat[0:nfft2] * np.conjugate(sp_hat[0:nfft2])

    
    ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
    domfreq=freq[ind] # it is equal to the maximum frequency ==Max_freq 
    
    return domfreq

@njit 
def calculate_norm_accandgyro(acc=None):
     return np.sqrt((acc**2).sum(axis=1)) 


@njit
def findMiddle(input_list):
    middle = float(len(input_list))/2
    return np.where(middle % 2 != 0,int(middle - .5),int(middle))


@njit
def calc_index_min(data):
   return np.argmax(data)

@njit
def calc_sma(data):
   return sum(list(map(abs, data)))



def filter_data(acc=[0],B=[],A=[]):

    y=np.copy(acc)

    y=signal.filtfilt(B, A, y,axis=0) # fix
   
    return y

@njit
def calc_median(mag):
    return np.median(mag)

@njit
def _moment(a, moment, mean=None):

    
    if moment==2:
        n_list=np.array([2])
    elif moment==3:
       n_list=np.array([3,1])
       
    elif moment==4:
       n_list=np.array([4,2]) 
    

    a_zero_mean = a - mean
    
    if n_list[-1] == 1:
        s = a_zero_mean.copy()
    else:
        s = a_zero_mean**2

    # Perform multiplications
    for n in n_list[-2::-1]:
        s = s**2
        if n % 2:
            s *= a_zero_mean
    return np.mean(s)