# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:15:26 2025

@author: hanna
"""


"""

[Figure 5] changes in preferred of individual BCI neurons - an extended look into shuffle sessions

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


mode = 'shuffle'
fn = f'tuning_ns0_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle, dPD_all_shuffle, dPD_each_shuffle = pickle.load(open_file)
open_file.close()


#%%

"""

##############################################################################

[Figure 5A] non-shuffled neurons across all sessions

##############################################################################

"""

'---non-shuffle---'

bins = np.arange(-180,181,20)

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7,4))
fig.subplots_adjust(wspace=0.075, hspace=0.0)

for i in [0,1]:
    
    subj, subject = dSubject[i]


    axis = ax[i]
    axis.set_title(f'{subject} | non-shuffled', loc='left')
    axis.text(-185, 0.022, 'assigned ΔPD = 0°', fontstyle='italic', fontsize=fontsize-1)

    m_sF = []
    a_sF = []
    
    ns = []
    
    for d, date in enumerate(dDates_shuffle[i][:]): 
        
        dfSESH = pd.DataFrame({'shuff': dPD_each_shuffle[i][date]['shuff'], 'sig': dPD_each_shuffle[i][date]['sig'], 'dPD': dPD_each_shuffle[i][date]['dPD_median'], 'adPD': dPD_each_shuffle[i][date]['assigned_dPD']})
        
        m_sF_ = dfSESH.loc[(dfSESH['shuff']==False)&(dfSESH['sig']==1),'dPD'].values
        a_sF_ = dfSESH.loc[(dfSESH['shuff']==False)&(dfSESH['sig']==1),'adPD'].values
        
        m_sF.append(m_sF_)
        a_sF.append(a_sF_)
        
        ns.append(len(dfSESH.loc[(dfSESH['shuff']==False)&(dfSESH['sig']==0),'dPD'].values) + len(m_sF_))


    mF = np.concatenate((m_sF))

    axis.hist(mF, bins=bins, color='grey', edgecolor='w', density=True)#, s=6, alpha=0.5)
    axis.set_xlim([-190,190])
    axis.set_xticks(bins[::3])
    axis.set_xticklabels(bins[::3], rotation=0, fontsize=10)
   
 
    axis.set_ylim([0,0.023])
    axis.set_yticks([0,0.0050,0.01,0.0150,0.02])
 
    axis.set_yticklabels((np.array([0,0.0050,0.01,0.0150,0.02])*1e3).astype(int), fontsize=10)#fontsize-1)
    
    axis.axvline(0, color='k', ls='-.', lw=2, ymin=0, ymax=0.021/0.023)
   
    axis.tick_params(axis='x', length=3) 
    axis.tick_params(axis='y', length=2) 
    

ax[0].set_ylabel('density (1e3)', fontsize=fontsize-1)
ax[0].set_xlabel('measured ΔPD (°)', fontsize=fontsize-1)
ax[1].set_xlabel('measured ΔPD (°)', fontsize=fontsize-1)


#%%


"""

##############################################################################

[Figure 5B] shuffled neurons across all sessions

##############################################################################

"""

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7,4))
fig.subplots_adjust(wspace=0.075, hspace=0.0)

for i in [0,1]:
    
    subj, subject = dSubject[i]


    axis = ax[i]
    axis.set_title(f'{subject} | shuffled', loc='left')
    
    m_sT = []
    a_sT = []
    

    ns = []
    
    for d, date in enumerate(dDates_shuffle[i][:]): 
        
        dfSESH = pd.DataFrame({'shuff': dPD_each_shuffle[i][date]['shuff'], 'sig': dPD_each_shuffle[i][date]['sig'], 'dPD': dPD_each_shuffle[i][date]['dPD_median'], 'adPD': dPD_each_shuffle[i][date]['assigned_dPD']})
        
        m_sT_ = dfSESH.loc[(dfSESH['shuff']==True)&(dfSESH['sig']==1),'dPD'].values
        a_sT_ = dfSESH.loc[(dfSESH['shuff']==True)&(dfSESH['sig']==1),'adPD'].values
        
        
        m_sT.append(m_sT_)
        a_sT.append(a_sT_)
        
        ns.append(len(dfSESH.loc[(dfSESH['shuff']==True)&(dfSESH['sig']==0),'dPD'].values) + len(m_sT_))

        
    
    mT = np.concatenate((m_sT))
    aT = np.concatenate((a_sT))
    
    
    r,p,star = compute_corr(mT, aT)
    print(i,len(mT), , len(dDates_shuffle[i]), f'r={r:.2f}, p={p:.2f}')

    
    axis.scatter(aT, mT, color=magenta, s=50, alpha=0.5)
    axis.set_xlim([-190,190])
    axis.set_xticks(np.arange(-180,181,45))
    axis.set_xticklabels(np.arange(-180,181,45), fontsize=fontsize-1)
    axis.set_ylim([-190,190])
    axis.set_yticklabels(np.arange(-180,181,45), fontsize=fontsize-1)
    axis.set_yticks(np.arange(-180,181,45))
    axis.axhline(0, color='grey', ls='--', zorder=0)
    axis.axvline(0, color='grey', ls='--', zorder=0)
    
    axis.axhline(45,color='k', ls='-.', zorder=0, lw=1)
    axis.axhline(-45,color='k', ls='-.', zorder=0, lw=1)
    

ax[0].set_xlabel('assigned ΔPD (°)', fontsize=fontsize+1)
ax[1].set_xlabel('assigned ΔPD (°)', fontsize=fontsize+1)
ax[0].set_ylabel('measured ΔPD (°)', fontsize=fontsize+1)

