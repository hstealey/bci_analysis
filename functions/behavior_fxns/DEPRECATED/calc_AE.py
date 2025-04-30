# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:29:02 2025

@author: hanna
"""

import os
import pickle
import numpy as np
import pandas as pd


def degChange(ANG1, ANG2):
    
    """
    Computes the minimum circular (angular) distance between two angles (ANG1, ANG2)
    
    inputs
        ANG1:float, first angle (in degrees)
        ANG2: float, second angle (in degrees)
    
    returns
        delta: float, the minimum angle between the two input angles (in degrees)
    
    """
    
    #Convert angles from degrees to radians.
    radA = ANG1 * (np.pi/180)
    radB = ANG2 * (np.pi/180)
    
    #Convert angles into cosine and sine components.
    a = np.array( (np.cos(radA),np.sin(radA)) )
    b = np.array( (np.cos(radB),np.sin(radB)) )
    
    #Inverse cosine converted to degrees.
    delta = np.arccos(np.matmul(a,b)) * (180/np.pi)

    return(delta)

def calc_AE_full(dfi, blockType, mode):
    
    """
    Computes the angular error (AE) at half the distance from the target
        [REF] (Chase JNP 2012) "Behavioral and neural correlates of visuomotor adaptation observed through a brain-computer interface in primary motor cortex"
        Half distance is hard-coded as 5 cm (target is centered at 10cm)
        Error clamp trials are not analyzed

    inputs
        dfi: pandas DataFrame, task and spike information extracted from HDF files
        blockType: int, indicates block number (1: baseline, 2: perturbation, 3: perturbation (with error clamp), 4: washout)
            Note: [3] and [4] may not exist or exist with various numbers of trials between sessions
        mode: string, indicates if the data is from a 'rotation' or 'shuffle' perturbation BCI task
        
    
    returns
        trial_inds: 2D numpy array (8 targets x n trials) of trial numbers used to compute statistics
            #TODO: impose type as int?
        AE: 2D numpy array (8 targets x n trials) of angular errors
    
    """
    
    #Dataframe from all non-error clamp trials in a given block (blockType)
    df = dfi.loc[(dfi['errorClamp']==0) & (dfi['blockType']==blockType)] 
    
    try:
        if mode == 'rotation':
            n1 = 41
            n2 = 41
        elif mode == 'shuffle':
            n1 = 20
            n2 = 41
    except Exception:
        print('INVALID MODE FOR calc_AE_full')
        return(None, None)

    if blockType == 1:
        trial_inds = np.zeros((8,n1))
        AE = np.zeros((8,n1))
    else:
        trial_inds = np.zeros((8,n2))
        AE = np.zeros((8,n2))
        
    for j, deg in enumerate(np.arange(0,360,45)):
        
        if blockType == 1:
            #Exclude first trial set to eliminate "start up" effects (e.g., subject does not recognize that task has started so they are not properly engaged.
            trial_inds[j,:] = df.loc[df['target']==deg].index[1:]
        elif blockType == 2:
            trial_inds[j,:] = df.loc[df['target']==deg].index[:n2]
            
    
        for ti, trial in enumerate(trial_inds[j,:]):
            
            target = df.loc[trial]['target']
                
            px = df.loc[trial]['decoder_py']
            py = df.loc[trial]['decoder_px']
            
            magp_ = np.sqrt(px**2 + py**2)

            half_ind = np.where(magp_ >= 5)[0][0] 
            px_mid = px[half_ind]
            py_mid = py[half_ind]
            
            ang = np.array([pd+2*np.pi if pd < 0 else pd for pd in [np.arctan2(py_mid,px_mid)]])*(180/np.pi)
            ang_err = degChange(ang[0],target)
            
            AE[j,ti] = ang_err
        
   
    return(trial_inds, AE)



# AE over time within trial
# ang_err = []
# for x,y in zip(px,py):
#     ang_i = np.arctan2(y,x)
#     if ang_i < 0:
#         ang_i+=2*np.pi
#     ang_err_ = degChange(ang_i*(180/np.pi), target)    
#     ang_err.append(ang_err_)
    

# dAE[i][date][blockLabel].append(ang_err)
