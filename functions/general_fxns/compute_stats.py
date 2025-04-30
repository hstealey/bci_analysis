# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:31:14 2025

@author: hanna
"""


import pandas as pd
import numpy as np
from scipy import stats


from scipy.optimize import curve_fit

        

def stats_star(p, trending=True):

    if trending == True:
        if p < 0.001:
            star = '***'
        elif (p<0.01) and (p>0.001):
            star = '**'
        elif (p<0.05) and (p>0.01):
            star = '*'
        elif (p<0.1) and (p>0.05):
            star = 't.'
        else:
            star = 'n.s.'
        
    else:
        if p < 0.001:
            star = '***'
        elif (p<0.01) and (p>0.001):
            star = '**'
        elif (p<0.05) and (p>0.01):
            star = '*'
        # elif (p<0.1) and (p>0.05):
        #     star = 't.'
        else:
            star = 'n.s.'

    return(star)


def compute_one_sample_ttest(v1, popmean=0, trending=True):
    
    t,p = stats.ttest_1samp(v1,popmean)
    star = stats_star(p,trending)
    
    return(t,p,star)

        
def compute_two_sample_ttest(v1, v2, trending=True):
    
    var1 = np.var(v1, ddof=1)
    var2 = np.var(v2, ddof=2)
    
    var_ratio = np.max([var1/var2, var2/var1])
    
    if var_ratio > 3:
        equal_var = False
    else:
        equal_var = True
    
    t,p = stats.ttest_ind(v1,v2,equal_var=equal_var)
    star = stats_star(p, trending)
    
    return(t,p,equal_var,star)





def compute_corr(v1, v2, trending=True):
    
    """
    Compute Pearson's correlation coefficient (r)
    
    """
    
    r,p = stats.pearsonr(v1,v2)
    star = stats_star(p, trending)
    
    return(r,p,star)
        


def compute_r_squared(y_true, y_predicted):
    """
    Calculate the R-squared (coefficient of determination).

    Args:
        y_true (array-like): Actual values.
        y_predicted (array-like): Predicted values.

    Returns:
        float: R-squared value.
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    
    sse = np.sum((y_true - y_predicted)**2)     #SSE: sume of squared errors
    sst = np.sum((y_true - np.mean(y_true))**2) #SST: total sum of squares
    
    r2 = 1 - (sse / sst) #R-squared
    
    return(r2)


def linear_func1(x, b1, b0):
    return(b1*x + b0)

def linear_func2(X, b1, b2, b0):
    x, y = X
    return(b1*x + b2*y + b0)

def linear_func2i(X, b1, b2, b3, b0):
    x, y = X
    return(b1*x + b2*y +b3*x*y +b0)

# def linear_func3(X, b1, b2, b3, b0):
#     x, y, z = X
#     return(b1*x + b2*y + b3*z + b0)

# def linear_func3i(X, b1, b2, b3, b4, b5, b6, b7, b0):
#     x, y, z = X
#     return(b1*x + b2*y + b3*z + b4*x*y +b5*x*z + b6*y*z + b7*x*y*z + b0)

    
def bin_data(v1, v2):
    
    """
        v1: list of independent values (e.g., neural command magnitude, communality)
        v2: list of dependent values (e.g., "accuracy")
        
    """
    n_bin = []
    
    x_mean   = []
    bin_mean = []
    bin_sem  = []
    
    x_sem = []
    
    TEMP = []
    
    bins_lo = np.arange(0,91,10)#/100
    bins_hi = np.arange(10,101,10)#/100
    
    #for lo_percVal,hi_percVal in zip(bins_lo, bins_hi):
    for j,k in zip(bins_lo, bins_hi):
        
        lo_percVal = np.percentile(v1,j)
        hi_percVal = np.percentile(v1,k)

        if j!=bins_hi[-1]:
            'Bins are inclusive of lowest value.'
            ind = np.where((v1 >= lo_percVal) & (v1 < hi_percVal))[0]
        else:
            'Highest bin is all inclusinve of highest value.'
            ind = np.where((v1 >= lo_percVal) & (v1 <= hi_percVal))[0]
        
        n_bin.append(len(ind))
        x_mean.append(np.mean(v1[ind]))
        bin_mean.append(np.mean(v2[ind]))
        bin_sem.append(stats.sem(v2[ind]))
        
        # #x_sem.append(stats.sem(v1[ind]))
        # xmin = np.min(v1[ind])
        # xmax = np.max(v1[ind])
        # x_sem.append((xmax-xmin)/2)
        # # TEMP.append(v2[ind])
       
    return(np.array(x_mean), np.array(bin_mean), np.array(bin_sem))#, np.array(x_sem))#, TEMP)
        



def fit_linear_model(xdata, ydata, model_type='L1'):
    
    if model_type == 'L1':
        model = linear_func1
    elif model_type == 'L2':
        model = linear_func2
    elif model_type == 'L21':
        model = linear_func2i
        
    params, pcov = curve_fit(model, xdata, ydata)#, maxfev=5000)
    perr = np.sqrt(np.diag(pcov)) #standard errors
    tvalues = params / perr
    dof = len(ydata) - len(params)
    pvalues = 2 * stats.t.sf(np.abs(tvalues), dof) # p-values (two-sided test)
    

    return(params, perr, tvalues, pvalues)
