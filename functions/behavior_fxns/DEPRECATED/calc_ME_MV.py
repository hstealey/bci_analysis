# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:26:18 2025

@author: hanna
"""


import os
import pickle
import numpy as np
import pandas as pd


def calc_error(px,py,target):
    
    
    y_errors = []
    for x,y in zip(px,py):
        
        if (target == 0) or (target == 180):
            # m = 0
            # y_hat = m*x
            y_err = y
            
        elif (target == 45) or (target == 225):
            # m = 1
            # y_hat = m*x
            y_err = y-x
    
        elif (target == 135) or (target == 315):
            # m = -1
            # y_hat = m*x
            y_err = y-(-1*x)
        
        elif (target == 90) or (target == 270):
            # m = inf.
            y_err = x
        
    
        y_errors.append(y_err)

    
    return(y_errors)

def calc_ME_MV_full(dfi, blockType, mode):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]
        
    try:
        if mode == 'rotation':
            n1 = 41
            n2 = 41
        elif mode == 'shuffle':
            n1 = 20
            n2 = 41
    except Exception:
        print('INVALID MODE FOR calc_ME_MV_full')
        return(None, None)
    

    if blockType == 1:
        trial_inds = np.zeros((8,n1))
        ME = np.zeros((8,n1))
        MV = np.zeros((8,n1))
    else:
        trial_inds = np.zeros((8,n2))
        ME = np.zeros((8,n2))
        MV = np.zeros((8,n2))


    for j, deg in enumerate(np.arange(0,360,45)):
        
        if blockType == 1:
            trial_inds[j,:] = df.loc[df['target']==deg].index[1:]
        elif blockType == 2:
            trial_inds[j,:] = df.loc[df['target']==deg].index[:n2]
            
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            px = df.loc[trial]['decoder_py']
            py = df.loc[trial]['decoder_px']  

            y_errors = calc_error(px,py,deg)
            
            MEi = np.sqrt(np.sum(np.abs(y_errors)))/len(y_errors)
            y_mean = np.mean(y_errors)
            MVi = np.sqrt(np.sum((y_errors - y_mean)**2)/len(y_errors))
            
            ME[j,ti] = MEi
            MV[j,ti] = MVi
            
    return(trial_inds, ME, MV)





def calc_ME_MV_window(dfi, blockType, mode, trial_inds, window_len=None):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]

    n = np.shape(trial_inds)[1]

    ME = np.zeros((8,n))
    MV = np.zeros((8,n))
    for j, deg in enumerate(np.arange(0,360,45)):
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            if window_len == None:
                px = df.loc[trial]['decoder_py']
                py = df.loc[trial]['decoder_px']  
            
            else:
                px = df.loc[trial]['decoder_py'][:window_len]
                py = df.loc[trial]['decoder_px'][:window_len]
            

            y_errors = calc_error(px,py,deg)
            
            MEi = np.sqrt(np.sum(np.abs(y_errors)))/len(y_errors)
            y_mean = np.mean(y_errors)
            MVi = np.sqrt(np.sum((y_errors - y_mean)**2)/len(y_errors))
            
            ME[j,ti] = MEi
            MV[j,ti] = MVi
            
   
    return(ME, MV)




# def calc_ME_MV_window_TEST(dfi, blockType, mode, trial_inds, window_len=None):

#     df_ = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]
#     df = df_ #df_.reset_index() #
    
#     if mode == 'rotation':
#         n = 41
#     elif mode == 'shuffle':
#         n = 20
#     else:
#         print('INVALID MODE FOR calc_ME_MV_full')


#     errors = []
#     trials = np.sort(np.concatenate((trial_inds)))

    
#     for trial in trials:
            
#         if window_len == None:
#             px = df.loc[trial]['decoder_py']
#             py = df.loc[trial]['decoder_px']  
        
#         else:
#             px = df.loc[trial]['decoder_py'][:window_len]
#             py = df.loc[trial]['decoder_px'][:window_len]
        

#         y_errors = calc_error(px,py,df.loc[trial]['target'])
        
#         errors.append(y_errors)

    
#     return(errors)






