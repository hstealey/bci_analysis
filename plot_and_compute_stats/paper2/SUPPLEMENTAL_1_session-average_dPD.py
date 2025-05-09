# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:25:00 2025

@author: hanna
"""

"""


Supplemental Figure 1
    session-average changes for ROTATION and SHUFFLE sessions
        averages are over all neurons (including non-significant)


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
os.chdir(os.path.join(root_path,'functions', 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star


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

[Dissertation Figure 4]

    Session-averaged dPD
    
        - ROTATION + SHUFFLE
        - CW/CCW separated
        - session-average dPD (including sig and n.s.)
            NOTE -> "N.S." CHANGES REPORTED AS 0
        
"""


PD_KEY = 'dPD_mean'

fontsize = 17
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(11,12))
fig.subplots_adjust(wspace=0.1)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])


    assigned_both = np.concatenate((assigned, 4*np.ones(len(dPD_all_shuffle[i][PD_KEY]))))
    dPD_both = np.concatenate((dPD_all[i][PD_KEY], dPD_all_shuffle[i][PD_KEY]))




    ax[i].set_title(f'{subject}', loc='left')
    sb.swarmplot(x=assigned_both, y=dPD_both, ax=ax[i], order=[-90,-50, 50, 90,4], palette=palette_ROT4, s=8, alpha=0.7, edgecolor='grey')
    ax[i].axhline(0, color='grey', lw=0.75, ls='--', zorder=0)


    
    ax[i].set_xticks([0,1,2,3,4.5])
    ax[i].set_xticklabels(['-90°', '-50°', '+50°', '+90°', 'shuffle'], rotation=0) 
    ax[i].set_xlabel('applied perturbation')
    
    ind90_cw =  np.where(assigned == -90)[0] 
    ind50_cw =  np.where(assigned == -50)[0] 
    ind50_ccw = np.where(assigned == 50)[0]  
    ind90_ccw = np.where(assigned == 90)[0]  
    
    v1 = dPD_all[i][PD_KEY][ind90_cw] 
    v2 = dPD_all[i][PD_KEY][ind50_cw]
    v3 = dPD_all[i][PD_KEY][ind50_ccw]
    v4 = dPD_all[i][PD_KEY][ind90_ccw]
    v5 = dPD_all_shuffle[i][PD_KEY]
    
    

  
    ax[i].scatter([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=70, zorder=10) #linewidth=2.5, s=75
    (_, caps, _) = ax[i].errorbar([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4), np.mean(v5)], yerr=[np.std(v1), np.std(v2), np.std(v3), np.std(v4), np.std(v5)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
    for cap in caps:
        cap.set_markeredgewidth(2.5)
        
    #print(i, np.min(v5), np.max(v5))
        
    
    size = 150
    
    for j, DEG, VAR_J in zip(np.arange(4), [-90, -50, 50, 90], [v1,v2,v3,v4]):
        ax[i].scatter(j, DEG, color=palette_ROT4[DEG], marker='d', s=size, edgecolor='k')
        ax[i].plot([j,j],[DEG, np.mean(VAR_J)], zorder=0, ls='-.', color=palette_ROT4[DEG])
        ax[i].plot([j,j],[0, np.mean(VAR_J)], color='grey')#, ls='--')
        
        if DEG < 0:
            r,p,star = compute_one_sample_ttest(VAR_J, DEG)
            ax[i].text(j,DEG-6,f'{star}', ha='center', va='top', color=palette_ROT4[DEG])
        else:
            r,p,star = compute_one_sample_ttest(VAR_J, DEG)
            ax[i].text(j,DEG+2,f'{star}', ha='center', va='bottom', color=palette_ROT4[DEG])


        r,p,star = compute_one_sample_ttest(VAR_J, 0)
        ax[i].text(j,-3,f'{star}', ha='center', va='bottom', color='grey')

    marker = 's'
    size2 = 50

    base = 95
    adder = 4
    
    '-90/-50'
    ax[i].plot([0,1],[base+(adder*0),base+(adder*0)], color='k', lw=1, zorder=0)
    ax[i].scatter(0,base+(adder*0), color=palette_ROT4[-90], marker=marker, s=size2) 
    ax[i].scatter(1,base+(adder*0), color=palette_ROT4[-50], marker=marker, s=size2) 

   
    '-90/+50'
    ax[i].plot([0,2],[base+(adder*1),base+(adder*1)], color='k', lw=1, zorder=0)
    ax[i].scatter(0,base+(adder*1), color=palette_ROT4[-90], marker=marker, s=size2) 
    ax[i].scatter(2,base+(adder*1), color=palette_ROT4[50], marker=marker, s=size2) 

    '+/-90'
    ax[i].plot([0,3],[base+(adder*2),base+(adder*2)], color='k', lw=1, zorder=0)
    ax[i].scatter(0,base+(adder*2), color=palette_ROT4[-90], marker=marker, s=size2) 
    ax[i].scatter(3,base+(adder*2), color=palette_ROT4[90], marker=marker, s=size2) 


    '+/-50'
    ax[i].plot([1,2],[base+(adder*3),base+(adder*3)], color='k', lw=1, zorder=0)
    ax[i].scatter(1,base+(adder*3), color=palette_ROT4[-50], marker=marker, s=size2) 
    ax[i].scatter(2,base+(adder*3), color=palette_ROT4[50], marker=marker, s=size2) 


    '-50,+90'
    ax[i].plot([1,3],[base+(adder*4), base+(adder*4)], color='k', lw=1, zorder=0)
    ax[i].scatter(1,base+(adder*4), color=palette_ROT4[-50], marker=marker, s=size2) 
    ax[i].scatter(3,base+(adder*4), color=palette_ROT4[90],  marker=marker, s=size2) 

    
    '+50,+90'
    ax[i].plot([2,3],[base+(adder*5), base+(adder*5)], color='k', lw=1, zorder=0)
    ax[i].scatter(2,base+(adder*5), color=palette_ROT4[50],  marker=marker, s=size2) 
    ax[i].scatter(3,base+(adder*5), color=palette_ROT4[90],  marker=marker, s=size2) 




    # dfANOVA = pd.DataFrame({'dPD': dPD_all[i][PD_KEY], 'rot': assigned}) 
    
    
    # model = ols('dPD ~ C(rot) ', data=dfANOVA).fit() 
    # result = sm.stats.anova_lm(model, type=1) 
      
    # print(result) 
    
    # tukey = pairwise_tukeyhsd(endog=dfANOVA['dPD'],
    #                           groups=dfANOVA['rot'],
    #                           alpha=0.05)
    
    # print(tukey)
    
    r,p,star = compute_one_sample_ttest(v5,0)
    ax[i].plot([4,5],[15,15], color='grey', zorder=0)
    ax[i].plot([4,4],[0,15], color='grey', zorder=0)
    ax[i].plot([5,5],[0,15], color='grey', zorder=0)
    
    if p < 0.05:
        ax[i].text(4.5, 13, f'{star}', fontstyle='italic', color='grey', va='bottom', ha='center', fontsize=fontsize+2)
    
    else:
        ax[i].text(4.5, 15, f'{star}', fontstyle='italic', color='grey', va='bottom', ha='center')


    ax[i].axvline(3.4, ymin=0, ymax=0.98, lw=1, ls='--', zorder=0, color='k')


ax[0].set_xlim([-0.5,5.25])  
   
ax[0].set_ylim([-110,120])
ax[0].set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
  
ax[0].set_ylabel('average measured ΔPD (°)', fontsize=fontsize+2)

