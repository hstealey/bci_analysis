# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 00:25:49 2025

@author: hanna
"""


import os
import pickle
import numpy as np
import pandas as pd


def calc_vel_full(dfi, blockType, mode):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]
    
    try:
        if mode == 'rotation':
            n1 = 41
            n2 = 41
        elif mode == 'shuffle':
            n1 = 20
            n2 = 41
    except Exception:
        print('INVALID MODE FOR calc_vel_full')
        return(None, None)
    

    if blockType == 1:
        trial_inds = np.zeros((8,n1))
        vel_full = np.zeros((8,n1,200))
        vel_mean = np.zeros((8,n1))
        
    else:
        trial_inds = np.zeros((8,n2))
        vel_full = np.zeros((8,n2,200))
        vel_mean = np.zeros((8,n2))
        
    for j, deg in enumerate(np.arange(0,360,45)):
        
        if blockType == 1:
            trial_inds[j,:] = df.loc[df['target']==deg].index[1:]
        elif blockType == 2:
            trial_inds[j,:] = df.loc[df['target']==deg].index[:n2]
            
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            vx = df.loc[trial]['decoder_vy']
            vy = df.loc[trial]['decoder_vx'] 
            magv_ = np.sqrt(vx**2 + vy**2)


            vel_full[j,ti,:len(magv_)] = magv_
            vel_mean[j,ti] = np.mean(magv_)

            
   
    return(trial_inds, vel_full, vel_mean)





def calc_vel_window(dfi, blockType, mode, trial_inds, window_len):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]
    
    n = np.shape(trial_inds)[1]

    vel_full_neural = np.zeros((8,n,200))
    vel_mean_neural = np.zeros((8,n))
    vel_mean_neural_window = np.zeros((8,n))
    
    for j, deg in enumerate(np.arange(0,360,45)):
    
        for ti, trial in enumerate(trial_inds[j,:]):
            vx = df.loc[trial]['decoder_vy']
            vy = df.loc[trial]['decoder_vx']
            magv_ = np.sqrt(vx**2 + vy**2)
            
            vel_full_neural[j,ti,:len(magv_)] = magv_
            vel_mean_neural[j,ti] = np.mean(magv_)
            vel_mean_neural_window[j,ti] = np.mean(magv_[:window_len])
        
                
    return(vel_full_neural, vel_mean_neural, vel_mean_neural_window)









