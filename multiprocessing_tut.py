# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:14:47 2022

@author: al-abiad
"""

import multiprocessing as mp
import time

#Queue: FIFO data structure allows us to perform interprocess communication
# put to inset data to the queue
# get data from the queue

#pool: is data parallelism 

#Process: create a process identifier to run as independent OS

#lock lock the code during execution by a process






start= time.perf_counter()

def do_something():
    print('Sleeping 1 second..')
    time.sleep(1)
    print("Done Sleeping...")



def lang_func(lang):
    print(lang)

def sqr(x,q):
    q.put(x*x)
    
    
    
if __name__=="__main__":
    
    start= time.perf_counter()
    
    do_something()
    
    do_something() 
    
    finish = time.perf_counter()
    
    print(finish-start)
    
    
    # q=mp.Queue()
    
    
    # processes=[mp.Process(target=sqr,args=(i,q)) for i in range(2,10)]
    
    # for proc in processes:
    #     proc.start()
        
    # for proc in processes:
        
    #     proc.join()
        
    
    
    # result = [q.get() for p in processes]
    
    # print(result)
    
    
    
    
    # langs = ['C','Python','JAVA','PHP']
    # processes = []
    
    # for l in langs:

    #     proc=Process(target=lang_func,args=(l,))
    #     processes.append(proc)
        
    #     proc.start()
        
    # for p in processes:
    #     p.join()
    
