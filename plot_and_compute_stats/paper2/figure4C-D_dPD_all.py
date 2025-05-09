# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:10:09 2025

@author: hanna
"""



"""

[Figure 4C, 4D] changes in preferred of individual BCI neurons - summary of all sessions

"""

#%%

import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy import stats


import warnings
warnings.filterwarnings("ignore")


import statsmodels.api as sm 
from statsmodels.formula.api import ols 

from statsmodels.stats.multicomp import pairwise_tukeyhsd

"""Plotting Parameters"""
fontsize = 12
mpl.rcParams["font.size"] = fontsize
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.spines.right"] = "False"
mpl.rcParams["axes.spines.top"] = "False"


'IBM Color Scheme - color blind friendly'
blue    = [100/255, 143/255, 255/255]
yellow  = [255/255, 176/255, 0/255]
purple  = [120/255, 94/255, 240/255]
orange  = [254/255, 97/255, 0/255]
magenta = [220/255, 38/255, 127/255]

palette_BEH = {50: yellow, 90: blue}
palette_ROT = {50: orange, 90: purple}
palette_SHU = {True: magenta, False: 'grey'}
palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'} #palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple}

dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')  
pickle_path_dTC = os.path.join(pickle_path, 'tuning') 
              

'------------------------------'
'Loading Custom Functions'
os.chdir(os.path.join(root_path, 'functions', 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star, compute_r_squared


#%%


"""

Combined Tuning Dictionary Keys

dPD_all[i]:         dPD_abs*, dPD_mean, dPD_median, dPD_16, dPD_84

dPD_each[i][date]:  shuff, MD_BL, MD_PE, dPD_lo, dPD_hi, dPD_median, 
                     dPD_abs, sig,
                     assigned_PDBL, assigned_PDPE, assigned_dPD


"""


mode = 'rotation'
fn = f'tuning_ns0_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()

mode = 'shuffle'
fn = f'tuning_ns0_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle, dPD_all_shuffle, dPD_each_shuffle = pickle.load(open_file)
open_file.close()


#%%



"""

##############################################################################

[Figure 4C] all ROTATION sessions

##############################################################################

"""

PD_KEY = 'dPD_mean'

fontsize = 17
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(16,6))
fig.subplots_adjust(wspace=0.0)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])
    ax[i].set_title(f'{subject}', loc='left')


    ax[i].set_ylim([-190,190])
    ax[i].set_yticks([-180,-90,0,90,180])
    
    ax[i].axhline(0, color='grey', zorder=0, lw=2)
   

    count = 0
    for DEG in [-90,-50,50,90]:
        
        color = palette_ROT4[DEG]
        
        DATES_DEG = dDates[i][np.where(assigned == DEG)[0]]
        
        for date in DATES_DEG[:]:
            
            sig  = dPD_each[i][date]['sig']
      
            if np.sum(sig) == 0:
                print(i, date,'ROTATION: NONE SIGNIFICANT')
            
            else:
    
                sigT = np.where(sig==1)[0]
                dPD_sesh = dPD_each[i][date]['dPD_median'][sigT]
           
                ax[i].axvline(count,color='grey', zorder=0, lw=0.4)
                ax[i].scatter(np.ones(len(dPD_sesh))*count, dPD_sesh, color=color,alpha=0.3)
                ax[i].scatter(count,np.mean(dPD_sesh), zorder=100, color='k', marker='s', s=10)
        
            count+=1
        
ax[0].set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
ax[0].set_xlabel('session (sorted by applied rotation)', fontsize=fontsize+2)   
ax[1].set_xlabel('session (sorted by applied rotation)', fontsize=fontsize+2) 
      

#%%

"""

##############################################################################

[Figure 4D] all SHUFFLE sessions

##############################################################################

"""

PD_KEY = 'dPD_mean'

fontsize = 17
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(16,8))
fig.subplots_adjust(wspace=0, hspace=0.4)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])
    ax[0][i].set_title(f'{subject} | non-shuffled', loc='left')
    ax[1][i].set_title(f'{subject} | shuffled', loc='left')


    ax[0][i].set_ylim([-195,195])
    ax[0][i].set_yticks([-180,-90,0,90,180])
    
    ax[0][i].axhline(0, color='grey', zorder=0, lw=2)
    ax[1][i].axhline(0, color='grey', zorder=0, lw=2)
    

    percShuff = []
    for d,date in enumerate(dDates_shuffle[i]):
        
        shuff = dPD_each_shuffle[i][date]['shuff']
        
        percShuff.append( 100* (np.sum(shuff)/len(shuff)) )
        
    dfDS = pd.DataFrame({'date': dDates_shuffle[i], 'perc': percShuff}).sort_values(by=['perc']).reset_index(drop=True)
        
    count = 0
    for d, date in enumerate(dfDS['date'].tolist()[:]):
        
        dPD_sesh = dPD_each_shuffle[i][date]['dPD_median']
        
        shuff = dPD_each_shuffle[i][date]['shuff']
        shuffT = np.where(shuff==True)[0]
        shuffF = np.where(shuff==False)[0]
        
        
        sig  = dPD_each_shuffle[i][date]['sig']
        
        if np.sum(sig) == 0:
            pass
  
        
        else:

            dPD_sesh_not = []
            dPD_sesh_shu = []
            for neuron in range(len(sig)):
                
                if (shuff[neuron] == False) and (sig[neuron] == 1):
                    dPD_sesh_not.append(dPD_sesh[neuron])
                   
                    
                elif (shuff[neuron] == True) and (sig[neuron] == 1):
                    dPD_sesh_shu.append(dPD_sesh[neuron])
                    
               
            ax[0][i].axvline(d,color='grey', zorder=0, lw=0.4)
            ax[1][i].axvline(d,color='grey', zorder=0, lw=0.4)
            
            if len(dPD_sesh_not) == 0:
                ax[0][i].scatter(d, 0, marker='x', color=magenta, s=60) 
            else:
                ax[0][i].scatter(np.ones(len(dPD_sesh_not))*d, dPD_sesh_not, color='k',alpha=0.5)
                
            
            if len(dPD_sesh_shu) == 0:
                ax[1][i].scatter(d, 0, marker='x', color=magenta, s=60)
            else:
                ax[1][i].scatter(np.ones(len(dPD_sesh_shu))*d, dPD_sesh_shu, color='k',alpha=0.5)
                
            
            if (len(dPD_sesh_not)!=0) and (len(dPD_sesh_shu)!=0):
                U, p = stats.mannwhitneyu(np.abs(dPD_sesh_not), np.abs(dPD_sesh_shu))
                if p < 0.05:
                    ax[0][i].scatter(d,170,color='blue', marker='d', s=60)
                    ax[1][i].scatter(d,170,color='blue', marker='d', s=60)
            else:
                count+=1
   
ax[0][0].set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
ax[1][0].set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
ax[1][0].set_xlabel('session (sorted by % shuffled)', fontsize=fontsize+2)   
ax[1][1].set_xlabel('session (sorted by % shuffled)', fontsize=fontsize+2) 

legend_elements = [Line2D([0],[0], marker='o', markersize=10, color='grey', lw=0, label='ΔPD'),
                   Line2D([0],[0], marker='x', markersize=10, color=magenta, lw=0, label='no sig. ΔPD'),
                   Line2D([0],[0], marker='d', markersize=10, color='blue', lw=0, label='sig. diff.')]


ax[1][0].legend(handles=legend_elements, ncol=3, columnspacing=0.3, handletextpad=0.0, loc='lower left')

          


