# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:21:36 2025

@author: hanna
"""

import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from scipy import stats
from scipy.optimize import curve_fit

# import warnings
# warnings.filterwarnings("ignore")

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def linear_model(x,m,b):
    return(m*x+b)

def exponential_model(x,a,b,c):
    return(a*np.exp(-b*x)+c)


import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple}


dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')  
pickle_path_BEH = os.path.join(pickle_path, 'dfBEH')
pickle_path_dTC = os.path.join(pickle_path, 'FA_tuning') 
              

'------------------------------'
'Loading Custom Functions'
os.chdir(os.path.join(root_path, 'functions', 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star, compute_r_squared




#%%

BEH_VAR_LIST = ['time','dist']

dBEH = {}

for mode in ['rotation', 'shuffle']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dBEH[mode] = {}


    for i in [0,1]:
    
        dBEH[mode][i] = {}
        
        dBEH[mode][i]['time'] = np.zeros((len(dDates[i]),41))
        dBEH[mode][i]['dist'] = np.zeros((len(dDates[i]),41))
        
        subj, subject = dSubject[i]
        
        
        for d in np.arange(len(dDates[i])):
            
            date = dDates[i][d]

            open_file = open(os.path.join(pickle_path_BEH, f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
            dfBEH_BL, dfBEH_PE = pickle.load(open_file)
            open_file.close()
            

            if mode == 'rotation':
                n_setsBL = 41
            elif mode == 'shuffle':
                n_setsBL = 20
                
            for BEH_VAR in BEH_VAR_LIST:
                dfBEH = dfBEH_BL
                bBL = np.zeros((8,n_setsBL))
                for di,deg in enumerate(np.arange(0,360,45)):
                    bBL[di,:] =  dfBEH.loc[dfBEH['deg'] == deg, BEH_VAR].values[-n_setsBL:]
                
                dfBEH = dfBEH_PE
                bPE = np.zeros((8,41))
                for di,deg in enumerate(np.arange(0,360,45)):
                    bPE[di,:] =  dfBEH.loc[dfBEH['deg'] == deg, BEH_VAR].values[:41]
                

                mBL = np.mean(bBL)
                mPE = np.mean(bPE, axis=0)
                
                deltaBEH = 100*((mPE-mBL)/mBL)

                dBEH[mode][i][BEH_VAR][d,:] = deltaBEH
               
  
                





#%%

"""

##############################################################################

[Figure 3A] behavior over sets of trials

##############################################################################

"""

fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=False,figsize=(10,5))
fig.subplots_adjust(wspace=0.05, hspace=0.3, left=0.05)

mode = 'rotation'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]

    ax1 = ax[i][0]
    ax2 = ax[i][1]
    
    ax1.set_title(f'{subject} | trial time', loc='left')
    ax2.set_title(f'{subject} | distance', loc='left')
    
    ax1.set_ylabel('% Δ from BL', fontsize=fontsize+2)


 
    for DEG, indDEG, marker in zip([50,90],[ind50,ind90],['d', 'D']):
        

        'Time'
        mT = np.mean(dBEH[mode][i]['time'][indDEG,:], axis=0)
        sT = stats.sem(dBEH[mode][i]['time'][indDEG,:], axis=0)
        ax1.fill_between(np.arange(41), mT-sT, mT+sT, color=palette_ROT4[DEG], alpha=0.3)
        ax1.scatter(np.arange(41),mT, color=palette_ROT4[DEG], marker=marker, edgecolor='w')
        
        'Distance'
        mD = np.mean(dBEH[mode][i]['dist'][indDEG,:], axis=0)
        sD = stats.sem(dBEH[mode][i]['dist'][indDEG,:], axis=0)
        ax2.fill_between(np.arange(41), mD-sD, mD+sD, color=palette_ROT4[DEG], alpha=0.3)
        ax2.scatter(np.arange(41),mD, color=palette_ROT4[DEG], marker=marker, edgecolor='w')
        
      


mode = 'shuffle'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax1 = ax[i][0]
    ax2 = ax[i][1]
        
    'Time'
    mT = np.mean(dBEH[mode][i]['time'], axis=0)
    sT = stats.sem(dBEH[mode][i]['time'], axis=0)
    ax1.scatter(np.arange(41),mT, color='grey', marker='o', edgecolor='w')
    ax1.fill_between(np.arange(41), mT-sT, mT+sT, color='grey', alpha=0.5)
    
    'Distance'
    mD = np.mean(dBEH[mode][i]['dist'], axis=0)
    sD = stats.sem(dBEH[mode][i]['dist'], axis=0)
    ax2.scatter(np.arange(41),mD, color='grey', marker='o', edgecolor='w')
    ax2.fill_between(np.arange(41), mD-sD, mD+sD, color='grey', alpha=0.4)
    
    
    
    ax1.axhline(0, color='grey', ls='--', lw=0.75, zorder=0)
    ax2.axhline(0, color='grey', ls='--', lw=0.75, zorder=0)
    
    ax1.axvline(0, color='grey', ls='--', lw=0.75, zorder=0)
    ax2.axvline(0, color='grey', ls='--', lw=0.75, zorder=0)
    
    
    'Fit linear regression for TIME.'
    xdata = np.arange(41)
    ydata = mT
    popt, pcov = curve_fit(linear_model, xdata, ydata)
    
    # Calculate degrees of freedom
    n = len(ydata)  # Number of data points
    k = len(popt)   # Number of parameters
    df = n - k
    
    standard_errors = np.sqrt(np.diag(pcov))
    t_values = popt / standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
    
    #print(f'{subject}| shuffle-time | m={popt[0]:.3f} p={p_values[0]:.2f}')
    
    
    'Fit linear regression from DISTANCE.'
    ydata = mD
    popt, pcov = curve_fit(linear_model, xdata, ydata)
    
    # Calculate degrees of freedom
    n = len(ydata)  # Number of data points
    k = len(popt)   # Number of parameters
    df = n - k
    
    standard_errors = np.sqrt(np.diag(pcov))
    t_values = popt / standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
    
    #print(f'{subject}| shuffle-dist | m={popt[0]:.3f} p={p_values[0]:.2e}')


legend_elements = [Line2D([0],[0], marker='d', markersize=10, color=orange, lw=0, markeredgecolor='w', label='rotation, easy'),
                   Line2D([0],[0], marker='D', markersize=8, color=purple, lw=0, markeredgecolor='w', label='rotation, hard'),
                   Line2D([0],[0], marker='o', markersize=9, color='grey', lw=0, markeredgecolor='w', label='shuffle')]
    


ax[0][1].legend(handles=legend_elements, ncol=1, fontsize=fontsize, columnspacing=0.3, handletextpad=0.0, loc=(0.55,0.65))#'upper right')


ax[0][0].set_ylim([-5,120])
ax[0][0].set_yticks(np.arange(0,121,20))
ax[0][0].set_yticklabels(['BL', 20, 40, 60, 80, 100, 120])
ax[0][1].set_ylim([-5,120])
ax[0][1].set_yticks(np.arange(0,121,20))
ax[0][1].set_yticklabels([])


ax[1][0].set_ylim([-5,100])
ax[1][0].set_yticks(np.arange(0,101,20))
ax[1][0].set_yticklabels(['BL', 20, 40, 60, 80, 100])
ax[1][1].set_ylim([-5,100])
ax[1][1].set_yticks(np.arange(0,101,20))
ax[1][1].set_yticklabels([])


ax[1][0].set_xticks([0,10-1,20-1,30-1,40-1])
ax[1][0].set_xticklabels([r'${1^{st}}$', r'${10^{th}}$', '$20^{th}$', '$30^{th}$', '$40^{th}$'], rotation=0, fontstyle='italic', fontname='Arial') #\mathregular

ax[1][1].set_xticks([0,10-1,20-1,30-1,40-1])
ax[1][1].set_xticklabels([r'${1^{st}}$', r'${10^{th}}$', '$20^{th}$', '$30^{th}$', '$40^{th}$'], rotation=0, fontstyle='italic', fontname='Arial') #\mathregular


ax[1][0].set_xlabel('perturbation trial set', fontsize=fontsize+2)
ax[1][1].set_xlabel('perturbation trial set', fontsize=fontsize+2)



#%%

"""

##############################################################################

[Figure 4B] behavior at select timepoints

##############################################################################

"""



fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(3.5,11))
fig.subplots_adjust(wspace=0.1, hspace=0.15)

BEH_KEY = 'dist'

mode = 'rotation'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

mode = 'shuffle'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle = pickle.load(open_file)
open_file.close()

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[i][0].set_title(f'{subject}', loc='left')
    
    
    for j,TP in enumerate(['first', 'best']):
        axis = ax[i][j]
        axis.text(0.6,227,f'{TP}', ha='left', fontstyle='italic', fontsize=fontsize+2)
        
        xdata = []
        ydata = []
        for mode in ['rotation', 'shuffle']:
            if mode == 'rotation':
                ind50 = np.where(dDegs[i]==50)[0]
                ind90 = np.where(dDegs[i]==90)[0]
                xdata.append([50]*len(ind50))
                xdata.append([90]*len(ind90))
            else:
                xdata.append(np.ones(len(dDates_shuffle[i]))*3)

            if TP == 'first':
                if mode == 'rotation':
                    ydata.append( dBEH[mode][i][BEH_KEY][ind50,0])
                    ydata.append( dBEH[mode][i][BEH_KEY][ind90,0])
                else:
                    ydata.append( dBEH[mode][i][BEH_KEY][:,0])

            elif TP == 'best':
                if mode == 'rotation':
                    ydata.append( np.min(dBEH[mode][i][BEH_KEY][ind50,:], axis=1))
                    ydata.append( np.min(dBEH[mode][i][BEH_KEY][ind90,:], axis=1))
                else:
                    ydata.append( np.min(dBEH[mode][i][BEH_KEY][:,:], axis=1) )
        
        
                
        
        bplot = axis.boxplot(ydata, showfliers=False, patch_artist=True)
        axis.axhline(0, color='grey', zorder=0, ls='--', lw=1)
        
        for patch, color in zip(bplot['boxes'], [orange, purple, 'grey']):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        for patch, color in zip(bplot['medians'], [orange, purple, 'grey']):
            patch.set_color('k')
            
        
        dfANOVA = pd.DataFrame({'pert': np.concatenate((xdata)), 'val': np.concatenate((ydata))})
            

        
        ##Uncomment following to print ANOVA results:
            
        # v1 = dfANOVA.loc[dfANOVA['pert']==50, 'val'].values
        # v2 = dfANOVA.loc[dfANOVA['pert']==90, 'val'].values
        # v3 = dfANOVA.loc[dfANOVA['pert']==3, 'val'].values
   
        # print(i,TP, 'easy', compute_one_sample_ttest(v1,0))
        # print(i,TP, 'hard', compute_one_sample_ttest(v2,0))
        # print(i,TP, 'shuffle', compute_one_sample_ttest(v3,0))
        
        # anova_res = f_oneway(v1, v2, v3)
        
        # num_groups = 3
        # df_between  = num_groups -1
        # df_within = len(v1) + len(v2) + len(v3) - num_groups
        # F,p = anova_res
        # #print(subject, TP, BEH_KEY, f'F{df_between},{df_within}={F:.2f}, p={p:.1e}')

    
        # dfANOVA = pd.DataFrame({'pert': [50]*len(v1) + [90]*len(v2) + [3]*len(v3), 'val': np.concatenate(([v1,v2,v3])) })
        # tukey = pairwise_tukeyhsd(endog=dfANOVA['val'],
        #                   groups=dfANOVA['pert'],
        #                   alpha=0.05)
        
        # print(tukey)

   


        axis.set_xticks([1,2,3])
        axis.set_xticks([1,2,3])
        
for j in [0,1]:
    ax[1][j].set_xticks([1,2,3])
    ax[1][j].set_xticklabels(['rot.,\neasy', 'rot.,\nhard', 'shuffle'])
        


ax[0][0].set_ylim([-35,235])
ax[0][0].set_yticks(np.arange(-25,226,25))
ax[0][0].set_yticklabels([-25, 'BL', 25, 50, 75, 100, 125, 150, 175, 200,225])



if BEH_KEY == 'dist':

    ax[0][0].set_ylabel('% Δ in distance from BL', fontsize=fontsize+2)
    ax[1][0].set_ylabel('% Δ in distance from BL', fontsize=fontsize+2)    
    
    """
    
    SIG BARS for "dist"
    
    """
    
    'first'
        
    '---50 vs 90'
    '-----Monkey A | 0.001; 90 > 50'
    ax[0][0].plot([1,2],[200,200], color='k')
    ax[0][0].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[0][0].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.1779; 90 ! 50'
    ax[1][0].plot([1,2],[200,200], color='grey', ls='-.', lw=0.75)
    ax[1][0].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[1][0].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    
    
    
    
    '---50 v shuff'
    '-----Monkey A | 0.001; 50 > shuff'
    ax[0][0].plot([1,3],[210,210], color='k')
    ax[0][0].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[0][0].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
        
    '-----Monkey B | 0.001; 50 > shuff'
    ax[1][0].plot([1,3],[210,210], color='k')
    ax[1][0].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[1][0].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    
    
    
    '---90 v shuff'
    '-----Monkey A | 0.001; 90 > shuff'
    ax[0][0].plot([2,3],[220,220], color='k')
    ax[0][0].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[0][0].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 90 > shuff'
    ax[1][0].plot([2,3],[220,220], color='k')
    ax[1][0].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[1][0].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    
    
  



    

    
    
    'best'
    '---50 vs 90'
    '-----Monkey A | 0.001; 90 > 50'
    ax[0][1].plot([1,2],[200,200], color='k')
    ax[0][1].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[0][1].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 90 > 50'
    ax[1][1].plot([1,2],[200,200], color='k')
    ax[1][1].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[1][1].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '---50 vs shuff'
    '-----Monkey A | 0.0.001; 50 > shuff'
    ax[0][1].plot([1,3],[210,210], color='grey', ls='-.', lw=0.75)
    ax[0][1].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[0][1].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 50 < shuff'
    ax[1][1].plot([1,3],[210,210], color='k')
    ax[1][1].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[1][1].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    '---90 vs shuff'
    '-----Monkey A | 0.0.001; 90 > shuff'
    ax[0][1].plot([2,3],[220,220], color='k')
    ax[0][1].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[0][1].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.0869 (t.); 90 > shuff'
    ax[1][1].plot([2,3],[220,220], color='k', ls='--')
    ax[1][1].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[1][1].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    
    

elif BEH_KEY == 'time':
    
    
    ax[0][0].set_ylabel('% Δ in trial time from BL', fontsize=fontsize+2)
    ax[1][0].set_ylabel('% Δ in trial time from BL', fontsize=fontsize+2)


    """
    
    SIG BARS for "time"
    
    """
    
    'first'
    '---50 vs 90'
    '-----Monkey A | 0.001; 90 > 50'
    ax[0][0].plot([1,2],[200,200], color='k')
    ax[0][0].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[0][0].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.0508 (t.); 90 > 50'
    ax[1][0].plot([1,2],[200,200], color='k', ls='--')
    ax[1][0].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[1][0].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '---50 vs shuff'
    '-----Monkey A | 0.0146; 50 > shuff'
    ax[0][0].plot([1,3],[210,210], color='k')
    ax[0][0].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[0][0].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 50 > shuff'
    ax[1][0].plot([1,3],[210,210], color='k')
    ax[1][0].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[1][0].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    
    '---90 vs shuff'
    '-----Monkey A | 0.001; 90 > shuff'
    ax[0][0].plot([2,3],[220,220], color='k')
    ax[0][0].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[0][0].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 90 > shuff'
    ax[1][0].plot([2,3],[220,220], color='k')
    ax[1][0].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[1][0].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    
    
    
    'best'
    '---50 vs 90'
    '-----Monkey A | 0.001; 90 > 50'
    ax[0][1].plot([1,2],[200,200], color='k')
    ax[0][1].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[0][1].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 90 > 50'
    ax[1][1].plot([1,2],[200,200], color='k')
    ax[1][1].scatter(1,200,color=orange, marker='s', s=40, zorder=10)
    ax[1][1].scatter(2,200,color=purple, marker='s', s=40, zorder=10)
    
    '---50 vs shuff'
    '-----Monkey A | 0.2091; 50 ! shuff'
    ax[0][1].plot([1,3],[210,210], color='grey', ls='-.', lw=0.75)
    ax[0][1].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[0][1].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.001; 50 < shuff'
    ax[1][1].plot([1,3],[210,210], color='k')
    ax[1][1].scatter(1,210,color=orange, marker='s', s=40, zorder=10)
    ax[1][1].scatter(3,210,color='grey', marker='s', s=40, zorder=10)
    
    '---90 vs shuff'
    '-----Monkey A | 0.0.001; 90 > shuff'
    ax[0][1].plot([2,3],[220,220], color='k')
    ax[0][1].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[0][1].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    '-----Monkey B | 0.9; 90 ! shuff'
    ax[1][1].plot([2,3],[220,220], color='grey', ls='-.', lw=0.75)
    ax[1][1].scatter(2,220,color=purple, marker='s', s=40, zorder=10)
    ax[1][1].scatter(3,220,color='grey', marker='s', s=40, zorder=10)
    
    





                
                