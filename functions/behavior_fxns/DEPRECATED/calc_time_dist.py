# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:56:49 2025

@author: hanna
"""


import os
import pickle
import numpy as np
import pandas as pd



def calc_time_dist_full(dfi, blockType, mode):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]
    
    try:
        if mode == 'rotation':
            n1 = 41
            n2 = 41
        elif mode == 'shuffle':
            n1 = 20
            n2 = 41
    except Exception:
        print('INVALID MODE FOR calc_time_dist_full')
        return(None, None)

    if blockType == 1:
        trial_inds = np.zeros((8,n1))
        time = np.zeros((8,n1))
        dist = np.zeros((8,n1))
    else:
        trial_inds = np.zeros((8,n2))
        time = np.zeros((8,n2))
        dist = np.zeros((8,n2))
        
    for j, deg in enumerate(np.arange(0,360,45)):
        
        if blockType == 1:
            trial_inds[j,:] = df.loc[df['target']==deg].index[1:]
        elif blockType == 2:
            trial_inds[j,:] = df.loc[df['target']==deg].index[:n2]
            
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            time[j,ti] = df.loc[trial]['trial_time']
            dist[j,ti] = df.loc[trial]['decoder_distance']
            
    return(trial_inds, time, dist)





def calc_time_dist_window(dfi, blockType, mode, trial_inds, window_len=None):

    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)]

    n = np.shape(trial_inds)[1]

    time = np.zeros((8,n))
    dist = np.zeros((8,n))
    for j, deg in enumerate(np.arange(0,360,45)):
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            if window_len == None:
                time[j,ti] = df.loc[trial]['trial_time']
                dist[j,ti] = df.loc[trial]['decoder_distance']
            
            else:
                px = df.loc[trial]['decoder_py']
                py = df.loc[trial]['decoder_px']
                
                time[j,ti] = 100*(window_len/len(px))
                dist[j,ti] = np.linalg.norm(np.vstack((px[:window_len], py[:window_len])))

   
    return(time, dist)









