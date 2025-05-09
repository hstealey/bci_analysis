# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:00:52 2025

@author: hanna
"""


"""

[Figure 4A, 4B] changes in preferred of individual BCI neurons from example sessions


"""



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


# import warnings
# warnings.filterwarnings("ignore")


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

##############################################################################

[Figure 4A] example ROTATION session

##############################################################################

"""


mode = 'rotation'


fn = f'tuning_ns0_{mode}.pkl' #alt: fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()


palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'}


fontsize = 14
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8,8))

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    if i == 0:
        d = 47
    
    if i == 1:
        d = 17
    date = dDates[i][d]
    
    assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])

    ax[i].set_title(f'{subject}', loc='left')
    

    sig = dPD_each[i][date]['sig']
    
    dPD_lo  = dPD_each[i][date]['dPD_lo']
    dPD_med = dPD_each[i][date]['dPD_median']
    dPD_hi  = dPD_each[i][date]['dPD_hi']
    
    MD_BL = dPD_each[i][date]['MD_BL']
    MD_PE = dPD_each[i][date]['MD_PE']
    
    dfPLOT = pd.DataFrame({'sig': sig, 
                           'dPD_med': dPD_med, 'dPD_lo': dPD_lo, 'dPD_hi': dPD_hi,
                           'MD_BL': MD_BL, 'MD_PE': MD_PE}).sort_values(by=['sig', 'dPD_med']).reset_index(drop=True)


    axis1 = ax[i]
    axis1.axhline(0, color='grey', zorder=0, lw=1, ls='--')
    
    for l, dPD_lo_j, dPD_j, dPD_hi_j, sig in zip(np.arange(len(dfPLOT)), dfPLOT['dPD_lo'], dfPLOT['dPD_med'], dfPLOT['dPD_hi'], dfPLOT['sig']):
        if sig == 1:
            axis1.plot([l,l],[dPD_lo_j, dPD_hi_j], color='k', lw=2)
            axis1.scatter(l, dPD_j, color='k', marker='s', s=10)
        else:
            axis1.plot([l,l],[dPD_lo_j, dPD_hi_j], color='grey', ls='-')
            axis1.scatter(l, dPD_j, color='w', edgecolor='k', s=20)
        
    sig_true  = dfPLOT.loc[dfPLOT['sig']==1, 'dPD_med'].values
    sig_false = dfPLOT.loc[dfPLOT['sig']==0, 'dPD_med'].values
    
    axis1.text(40,180, f'applied: {assigned[d]}°', color='k')
    axis1.scatter(38,185, marker='s', s=100, color=palette_ROT4[assigned[d]])
    axis1.text(40,150, f'reported session mean: {np.mean(dfPLOT["dPD_med"]):.2f}°', color='k')
    axis1.text(40,120, f'average of significant only: {np.mean(sig_true):.2f}°', color='k')
 
ax[0].set_ylim([-190,190]) 
ax[1].set_ylim([-190,190]) 
ax[0].set_yticks(np.arange(-180,181,90))

ax[0].set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
ax[1].set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)

ax[1].set_xlabel('neuron (sorted)', fontsize=fontsize+2)   

legend_elements = [Line2D([0],[0], marker='o', markersize=10, color='w', lw=0, markeredgecolor='k', label='n.s. ΔPD'),
                   Line2D([0],[0], marker='s', markersize=10, color='k', lw=0, markeredgecolor='w', label='sig. ΔPD')]

ax[0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc='lower right')


#%%


"""

##############################################################################

[Figure 4B] example SHUFFLE session

##############################################################################

"""


mode = 'shuffle'

fn = f'tuning_ns0_{mode}.pkl' #alt: fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()


fontsize = 14
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10,10))
fig.subplots_adjust(hspace=0.3, wspace=0.4)


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    if i == 0:
        d = 30#4, 24 #30
        
    if i == 1:
        d = 0
        
    date = dDates[i][d]
    

    assigned = dPD_each[i][date]['assigned_dPD']
    shuff = dPD_each[i][date]['shuff']


    ax[i][0].set_title(f'{subject} | non-shuffled', loc='left')
    ax[i][0].text(-2,170, 'no assigned changes', fontstyle='italic')
    ax[i][1].set_title(f'{subject} | shuffled', loc='left')
    

    sig = dPD_each[i][date]['sig']
    
    dPD_lo  = dPD_each[i][date]['dPD_lo']
    dPD_med = dPD_each[i][date]['dPD_median']
    dPD_hi  = dPD_each[i][date]['dPD_hi']
    
    MD_BL = dPD_each[i][date]['MD_BL']
    MD_PE = dPD_each[i][date]['MD_PE']
    
    dfPLOT = pd.DataFrame({ 'assigned': assigned, 
                            'shuffled': shuff,
                            'sig': sig, 
                           'dPD_med': dPD_med, 'dPD_lo': dPD_lo, 'dPD_hi': dPD_hi,
                           'MD_BL': MD_BL, 'MD_PE': MD_PE}).sort_values(by=['sig', 'dPD_med']).reset_index(drop=True)


    axis1 = ax[i][0]
    axis1.axhline(0, color='grey', zorder=0, ls='--')
    
    shuffF = dfPLOT.loc[dfPLOT['shuffled']==False]
    
    for j, PD, PD1, PD2, sig in zip(np.arange(len(shuffF)), shuffF['dPD_med'], shuffF['dPD_lo'], shuffF['dPD_hi'], shuffF['sig']):
    
        if sig == 1:
            axis1.scatter(j, PD, color='k', marker='s')
            axis1.plot([j,j],[PD1, PD2], color='k')
        elif sig == 0:
            axis1.scatter(j, PD, color='w',edgecolor='grey', marker='o')
            axis1.plot([j,j],[PD1, PD2], color='grey')
            

    
    
    axis2 = ax[i][1]
    axis2.axhline(0, color='grey', zorder=0, ls='--')
    axis2.axvline(0, color='grey', zorder=0, ls='--')
    
    shuffT = dfPLOT.loc[dfPLOT['shuffled']==True]
    
    for j, assign, PD, PD1, PD2, sig in zip(np.arange(len(shuffT)), shuffT['assigned'], shuffT['dPD_med'], shuffT['dPD_lo'], shuffT['dPD_hi'], shuffT['sig']):
    
        if sig == 1:
            marker = 's'
            color = 'k'
            edgecolor=None
            color2 = 'k'
            
        elif sig == 0:
            marker = 'o'
            color = 'w'
            edgecolor='grey'
            color2 = 'grey'
            
        axis2.scatter(assign, PD, color=color, marker=marker, edgecolor=edgecolor)
        axis2.plot([assign,assign],[PD1, PD2], color=color2)
    
    
   
    axis1.set_xlabel('neuron (sorted)', fontsize=fontsize+2)
    axis1.set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
    axis1.set_ylim([-185,185])
    axis1.set_yticks(np.arange(-180,181,90))
    
    axis2.set_xlabel('assigned ΔPD (°)', fontsize=fontsize+2)
    axis2.set_xlim([-185,185])
    axis2.set_xticks(np.arange(-180,181,90))
    axis2.set_ylabel('measured ΔPD (°)', fontsize=fontsize+2)
    axis2.set_ylim([-185,185])
    axis2.set_yticks(np.arange(-180,181,90))



ax[0][0].set_xlim([-3, len(shuffF)])
ax[0][0].set_xticks(np.arange(0, len(shuffF), 10))

ax[1][0].set_xlim([-3, len(shuffF)])
ax[1][0].set_xticks(np.arange(0, len(shuffF), 10))




legend_elements = [Line2D([0],[0], marker='o', markersize=10, color='w', lw=0, markeredgecolor='k', label='n.s. ΔPD'),
                   Line2D([0],[0], marker='s', markersize=10, color='k', lw=0, markeredgecolor='w', label='sig. ΔPD')]

ax[0][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc='lower right')
ax[0][1].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc='lower right')
ax[1][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc=(0.6,0.01))
ax[1][1].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc='lower right')



