# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:11:09 2025

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



def compute_behavior(df, dfINDS, window_len):
    
    """

    
    """
    
  
    tn   = []
    deg  = []
    
    time = []
    time_window = [] #actually percent complete of trial
    dist = []
    dist_window = []
    
    vel_mag   = []
    vel_mean  = []
    vel_mean_window = []
    vel_min   = []
    vel_min_window = []
    vel_max   = []
    vel_max_window = []
    vel_range = []
    vel_range_window = []
    
    px_all = []
    py_all = []
    
    vx_all = []
    vy_all = []
    
    AE = []
    vel_at_AE = []
    
    ME = []
    ME_window = []
    MV = []
    MV_window = []

    for trial in dfINDS['tn']:

        px = df.loc[trial]['decoder_py']
        py = df.loc[trial]['decoder_px']
        magp_ = np.sqrt(px**2 + py**2)
        
        vx = df.loc[trial]['decoder_vy']
        vy = df.loc[trial]['decoder_vx'] 
        magv_ = np.sqrt(vx**2 + vy**2)

        tn.append(trial)
        deg.append(dfINDS.loc[dfINDS['tn']==trial, 'deg'].values[0])
        
        
        'Trial Time, Cursor Path Length (Distance), Velocity'
        time.append(df.loc[trial]['trial_time'])
        time_window.append(100*(window_len/len(px)))
        dist.append(df.loc[trial]['decoder_distance'])
        dist_window.append(np.linalg.norm(np.vstack((px[:window_len], py[:window_len]))))
        
        vel_mag.append(magv_)
        vel_mean.append(np.mean(magv_))
        vel_mean_window.append(np.mean(magv_[:window_len]))
        vel_min.append(np.min(magv_))
        vel_min_window.append(np.min(magv_[:window_len]))
        vel_max.append(np.max(magv_))
        vel_max_window.append(np.max(magv_[:window_len]))
        vel_range.append(np.max(magv_)-np.min(magv_))
        vel_range_window.append(np.max(magv_[:window_len])-np.min(magv_[:window_len]))
        
        px_all.append(px)
        py_all.append(py)
    
        vx_all.append(vx)
        vy_all.append(vy)
        
        
        
        'Angular Error (AE)'
        half_ind = np.where(magp_ >= 5)[0][0] 
        px_mid = px[half_ind]
        py_mid = py[half_ind]
        
        ang = np.array([pd+2*np.pi if pd < 0 else pd for pd in [np.arctan2(py_mid,px_mid)]])*(180/np.pi)
        ang_err = degChange(ang[0],deg[-1])
        AE.append(ang_err)
        
        vel_at_AE.append(magv_[half_ind])
        

        'Movement Error (ME) | Movement Variability (MV'
        y_errors = calc_error(px,py,deg[-1])
        MEi = np.sqrt(np.sum(np.abs(y_errors)))/len(y_errors)
        y_mean = np.mean(y_errors)
        MVi = np.sqrt(np.sum((y_errors - y_mean)**2)/len(y_errors))
        
        ME.append(MEi)
        MV.append(MVi)
        
        
        y_errors_window = y_errors[:window_len]
        MEi_window = np.sqrt(np.sum(np.abs(y_errors_window)))/len(y_errors_window)
        y_mean_window = np.mean(y_errors_window)
        MVi_window = np.sqrt(np.sum((y_errors_window - y_mean_window)**2)/len(y_errors_window))
        
        ME_window.append(MEi_window)
        MV_window.append(MVi_window)
        
   
    dfBEH = pd.DataFrame({'tn': tn, 'deg': deg, 
         'time': time, 'time_window': time_window, 
         'dist': dist, 'dist_window': dist_window,
         'vel_mag': vel_mag, 
          'vel_mean': vel_mean, 'vel_mean_window': vel_mean_window,
          'vel_min': vel_min, 'vel_min_window': vel_min_window,
          'vel_max': vel_max, 'vel_max_window': vel_max_window,
          'vel_range': vel_range, 'vel_range_window': vel_range_window,
          'px_all': px_all, 'py_all': py_all,
          'vx_all': vx_all, 'vy_all': vy_all,
          'AE': AE, 'vel_at_AE': vel_at_AE,
          'ME': ME, 'ME_window': ME_window,
          'MV': MV, 'MV_window': MV_window}).sort_values(by='tn').reset_index(drop=True)
    
        
    return(dfBEH)










# def compute_behavior2(df, dfINDS, window_len):
    
#     """

#     Same function as compute_behavior(), except the first two samples of each trial are removed.
#     See TODOs belows to see changes
    
#     """
    
  
#     tn   = []
#     deg  = []
    
#     time = []
#     time_window = [] #actually percent complete of trial
#     dist = []
#     dist_window = []
    
#     vel_mag   = []
#     vel_mean  = []
#     vel_mean_window = []
#     vel_min   = []
#     vel_min_window = []
#     vel_max   = []
#     vel_max_window = []
#     vel_range = []
#     vel_range_window = []
    
    
    
#     px_all = []
#     py_all = []
    
#     vx_all = []
#     vy_all = []
    
#     AE = []
#     vel_at_AE = []
    
#     ME = []
#     ME_window = []
#     MV = []
#     MV_window = []

#     for trial in dfINDS['tn']:

#         px = df.loc[trial]['decoder_py']
#         py = df.loc[trial]['decoder_px']
#         magp_ = np.sqrt(px**2 + py**2)
        
#         vx = df.loc[trial]['decoder_vy']
#         vy = df.loc[trial]['decoder_vx'] 
#         magv_ = np.sqrt(vx**2 + vy**2)

#         tn.append(trial)
#         deg.append(dfINDS.loc[dfINDS['tn']==trial, 'deg'].values[0])
        
        
#         'Trial Time, Cursor Path Length (Distance), Velocity'
#         time.append(df.loc[trial]['trial_time'])
#         time_window.append(100*(window_len/len(px)))
#         dist.append(df.loc[trial]['decoder_distance'])
#         dist_window.append(np.linalg.norm(np.vstack((px[2:window_len], py[2:window_len])))) #TODO: changed to remove first two samples
        
#         vel_mag.append(magv_)
#         vel_mean.append(np.mean(magv_))
#         vel_mean_window.append(np.mean(magv_[2:window_len]))  # #TODO: changed to remove first two samples
#         vel_min.append(np.min(magv_))
#         vel_min_window.append(np.min(magv_[2:window_len])) #TODO: changed to remove first two samples
#         vel_max.append(np.max(magv_))
#         vel_max_window.append(np.max(magv_[2:window_len]))  #TODO: changed to remove first two samples
#         vel_range.append(np.max(magv_)-np.min(magv_))
#         vel_range_window.append(np.max(magv_[2:window_len])-np.min(magv_[2:window_len])) #TODO: changed to remove first two samples
        
#         px_all.append(px)
#         py_all.append(py)
    
#         vx_all.append(vx)
#         vy_all.append(vy)
        
        
#         #NOT CHANGED
#         'Angular Error (AE)'
#         half_ind = np.where(magp_ >= 5)[0][0] 
#         px_mid = px[half_ind]
#         py_mid = py[half_ind]
        
#         ang = np.array([pd+2*np.pi if pd < 0 else pd for pd in [np.arctan2(py_mid,px_mid)]])*(180/np.pi)
#         ang_err = degChange(ang[0],deg[-1])
#         AE.append(ang_err)
#         vel_at_AE.append(magv_[half_ind])
        
        
        
#         #TODO: changed ALL!!! to remove first two samples
#         'Movement Error (ME) | Movement Variability (MV'
#         y_errors = calc_error(px[2:],py[2:],deg[-1]) 
#         MEi = np.sqrt(np.sum(np.abs(y_errors)))/len(y_errors)
#         y_mean = np.mean(y_errors)
#         MVi = np.sqrt(np.sum((y_errors - y_mean)**2)/len(y_errors))
        
#         ME.append(MEi)
#         MV.append(MVi)
        
        
#         y_errors_window = calc_error(px[2:window_len],py[2:window_len],deg[-1]) 
#         MEi_window = np.sqrt(np.sum(np.abs(y_errors_window)))/len(y_errors_window)
#         y_mean_window = np.mean(y_errors_window)
#         MVi_window = np.sqrt(np.sum((y_errors_window - y_mean_window)**2)/len(y_errors_window))
        
#         ME_window.append(MEi_window)
#         MV_window.append(MVi_window)
        
   
#     dfBEH = pd.DataFrame({'tn': tn, 'deg': deg, 
#          'time': time, 'time_window': time_window, 
#          'dist': dist, 'dist_window': dist_window,
#          'vel_mag': vel_mag, 
#           'vel_mean': vel_mean, 'vel_mean_window': vel_mean_window,
#           'vel_min': vel_min, 'vel_min_window': vel_min_window,
#           'vel_max': vel_max, 'vel_max_window': vel_max_window,
#           'vel_range': vel_range, 'vel_range_window': vel_range_window,
#           'px_all': px_all, 'py_all': py_all,
#           'vx_all': vx_all, 'vy_all': vy_all,
#           'AE': AE, 
#           'vel_at_AE': vel_at_AE,
#           'ME': ME, 'ME_window': ME_window,
#           'MV': MV, 'MV_window': MV_window}).sort_values(by='tn').reset_index(drop=True)
    

        
#     return(dfBEH)




# # AE over time within trial
# # ang_err = []
# # for x,y in zip(px,py):
# #     ang_i = np.arctan2(y,x)
# #     if ang_i < 0:
# #         ang_i+=2*np.pi
# #     ang_err_ = degChange(ang_i*(180/np.pi), target)    
# #     ang_err.append(ang_err_)
    

# # dAE[i][date][blockLabel].append(ang_err)





