# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:30:42 2025

@author: hanna
"""

import numpy as np
from scipy.optimize import curve_fit

def degChange(PD1, PD2):
    radA = PD1 * (np.pi/180)
    radB = PD2 * (np.pi/180)
    
    a = np.array( (np.cos(radA),np.sin(radA)) )
    b = np.array( (np.cos(radB),np.sin(radB)) )
    
    delta = np.arccos(np.matmul(a,b)) * (180/np.pi)

    return(delta)


def cosine_model(theta, b1, PD, b0):
    """
    Cosine function for firing rate fitting.
    theta: angles (in radians)
    b1: modulation (M)
        Modulation Depth (MD): b1/b0
    PD: phase shift (preferred direction)
    b0: baseline firing rate
    """
    return b0 + b1 * np.cos(theta - PD)


def fit_cosine_model(theta_rad, m):
    
    """
    constraints: 
        b1 (modulation): 0 to +inf
        PD (preferred direction): -2pi to +2pi
        b0 (mean firing rate): 0 to +inf
    """

    'initial guesses'
    #modulation
    p0_b1 = (np.max(m) - np.min(m))/2 
    #preferred direction
    p0_PD = theta_rad[np.where(m == np.max(m))[0][0]]
    #mean firing rate
    p0_b0 = np.mean(m)
    p0 = [p0_b1, p0_PD, p0_b0]

    'fit model'
    params, pcov = curve_fit(cosine_model, theta_rad, m, p0, bounds=([0,-2*np.pi,0],[np.inf,2*np.pi,np.inf]))
    M, PD, meanFR = params
    y_est = cosine_model(theta_rad, *params)
    
    'return parameters'
    return(M, M/meanFR, PD, meanFR, y_est)






def compute_signed_degChange(PD1, PD2):
    
    #TODO: need to update for signed changes....?
     
    # if PD1 < 0:
    #     PD1+=360
    
    # if PD2 < 0:
    #     PD2+=360
    
    delta1 = PD1 - PD2
    
    if delta1 < 0:
        delta2 = delta1+360
    else:
        delta2 = delta1
    
    if delta2 == 0:
        sign_ = 0 #'NO DIFF'
    elif delta2 == 180.0:
        sign_ = 'DIIF EXACTLY 180'
    elif delta2 < 180:
        sign_ = -1 #cw
    elif delta2 > 180:
        sign_ = 1 #'ccw'
    
    return(sign_, degChange(PD1, PD2))    




