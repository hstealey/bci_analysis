# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:30:30 2025

@author: hanna
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

from scipy.optimize import curve_fit

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
palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple}
palette_SHU = {True: magenta, False: 'grey'}

'Dark to Light Teal'
teal100 = [8/255, 26/255, 28/255]
teal90  = [2/255, 43/255, 48/255]
teal80  = [0/255, 65/255, 68/255]
teal70  = [0/255, 93/255, 93/255]
teal60  = [0/255, 125/255, 121/255]
teal50  = [0/255, 157/255, 154/255]
teal40  = [8/255, 189/255, 186/255]
teal30  = [61/255, 219/255, 217/255]
teal20  = [158/255, 240/255, 240/255]
teal10  = [217/255, 251/255, 251/255]

'Dark to Light Magenta'
magenta100 = [42/255, 10/255, 24/255]
magenta90  = [81/255, 2/255, 36/255]
magenta80  = [116/255, 9/255, 55/255]
magenta70  = [159/255, 24/255, 83/255]
magenta60  = [208/255, 38/255, 112/255]
magenta50  = [238/255, 83/255, 150/255]
magenta40  = [255/255, 126/255, 182/255]
magenta30  = [255/255, 175/255, 210/255]
magenta20  = [255/255, 214/255, 232/255]
magenta10  = [255/255, 240/255, 247/255]

dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')  
                 

'------------------------------'
'Loading Custom Functions'
os.chdir(os.path.join(root_path, functions, 'neural_fxns'))
from tuning_curve_fxns import degChange, compute_signed_degChange

os.chdir(os.path.join(root_path_functions, 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star


#%%

pickle_path_BEH = os.path.join(pickle_path, 'dfBEH', 'compute_behavior_results')


path_trials = 'latePE_160only' #alt: 'earlyPE_160only'
path_window = f'fixed_window_2-6'


pickle_path_dTC = os.path.join(pickle_path, 'FA_tuning', 'TTT', path_trials, path_window) #previously: #pickle_path_dTC = os.path.join(pickle_path,'tuning', 'BLOCK', 'earlyPE_160only', 'fit') 
pickle_path_VAR = os.path.join(pickle_path, 'FA_tuning', 'TTT', path_trials, path_window)


#%%


def linear_model(x,m,b):
    return(m*x+b)





#%%

BEH_VAR_LIST = ['time','dist']#, 'ME', 'MV'] 

dBEH = {}

for mode in ['rotation', 'shuffle']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dBEH[mode] = {}


    for i in [0,1]:
        
        dBEH[mode][i] = {'time': {'delta': np.zeros(len(dDates[i])),
                                  'best': np.zeros(len(dDates[i])),
                                  'aol': np.zeros(len(dDates[i])),
                                   'init': np.zeros(len(dDates[i]))},
                         
                        'dist': {'delta': np.zeros(len(dDates[i])),
                                  'best': np.zeros(len(dDates[i])),
                                  'aol': np.zeros(len(dDates[i])),
                                  'init': np.zeros(len(dDates[i]))}} 
                        
                        # 'ME':   {'delta': np.zeros(len(dDates[i])),
                        #           'best': np.zeros(len(dDates[i]))},
                        
                        # 'MV':    {'delta': np.zeros(len(dDates[i])),
                        #           'best': np.zeros(len(dDates[i]))} }
        
        
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

                dBEH[mode][i][BEH_VAR]['init'][d] = deltaBEH[0] #
                #dBEH[mode][i][BEH_VAR]['delta'][d] = np.mean(deltaBEH)
                dBEH[mode][i][BEH_VAR]['best'][d] = np.min(deltaBEH)
                #dBEH[mode][i][BEH_VAR]['aol'][d] = n.s. for most (deltaBEH[0]-np.min(deltaBEH))/deltaBEH[0]


#%%

"""
##############################################################################
##############################################################################
##############################################################################
"""

#%%


"""
ROTATION
"""

mode = 'rotation'

fn = f'tuning_ns0_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  #previously: open_file = open(os.path.join(pickle_path, 'postprocessing', 'paper2' ,fn), "rb") 
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()


#dPD_all[i]:  dPD_abs, dPD_mean, dPD_median, dPD_16, dPD_84
#dPD_each[i][date]:  shuff, MD_BL, MD_PE, dPD_lo, dPD_hi, dPD_median, dPD_abs, sig,
# assigned_PDBL, assigned_PDPE, assigned_dPD



#%%


"""

"BEST"

ROTATION - Mean of each "mismatch"
    mean(ABS(PD_each - PD_assigned)) 

"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,  sharey=True, figsize=(4,6))
fig.subplots_adjust(hspace=0.1,wspace=0.1)

dPD_mismatch = {}

for mode in ['rotation']:
    
    dPD_mismatch[mode] = {}
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')

        assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])

        dPD_mismatch[mode][i] = np.zeros(len(dDates[i]))
        for d, date in enumerate(dDates[i]):
            
            
            dPD_mismatch[mode][i][d] = np.mean( np.abs( dPD_each[i][date]['dPD_median'] - assigned[d] ) )
            #dPD_mismatch[mode][i][d] = np.abs(np.mean(dPD_each[i][date]['dPD_median'])-assigned[d])
    
    
        v1 = dPD_mismatch[mode][i]
        
        for j, BEH_VAR in enumerate(['time', 'dist']):#enumerate(['delta', 'best']):
            KEY = 'best'
            if KEY == 'init':
                color=teal30
            else:
                color=teal60
                
            
            v2 = dBEH[mode][i][BEH_VAR][KEY]
            
            
            
            ax[j][i].scatter(v1,v2, color=color, alpha=0.6, edgecolor='k')
            
            r,p,star = compute_corr(v1,v2)

            if p > 0.1:
                ls='--'
                lc='k'
               
            else:
                ls='-'
                lc='k'

            params, _ = curve_fit(linear_model, v1, v2)
            
            x1 = np.min(v1)
            x2 = np.max(v1)
            
            y1 = linear_model(x1,*params)
            y2 = linear_model(x2,*params)
            ax[j][i].plot([x1,x2],[y1, y2], color=lc, lw=2, ls=ls, zorder=100)
            
            ax[j][i].set_ylabel('')
            
            ax[j][i].axhline(0, color='k', zorder=0, ls='--', lw=0.75)
            
            ax[j][i].set_xlim([-3,78])
            ax[j][i].set_xticks([0,25,50,75])
            
            
            if KEY == 'init':
                ax[j][i].set_ylim([-15,275])  
                ax[j][i].set_yticks([0,50,100,150,200,250])
                if i == 0:
                    ax[j][0].set_yticklabels(['BL',50,100,150,200,250])
                
                ax[j][i].text(5,250,f'r={r:.2f} ({star})')
            
            if KEY == 'best':
                ax[j][i].set_ylim([-20,55])  
                ax[j][i].set_yticks([-15,0,15,30,45])
                if i == 0:
                    ax[j][0].set_yticklabels([-15,'BL',15,30,45])
                
                ax[j][i].text(5,50,f'r={r:.2f} ({star})')
                

ax[1][0].set_xlabel('ΔPD mismatch (°)')  
ax[1][1].set_xlabel('ΔPD mismatch (°)')  


if KEY == 'init':
    ax[0][0].set_ylabel(f'first\n%Δ in trial time from BL')
    ax[1][0].set_ylabel(f'first\n%Δ in distance from BL')
    
elif KEY == 'best':
    ax[0][0].set_ylabel(f'best\n%Δ in trial time from BL')
    ax[1][0].set_ylabel(f'best\n%Δ in distance from BL')






#%%

"""
##############################################################################
##############################################################################
##############################################################################
"""



#%%


"""

SHUFFLE

"""

mode = 'shuffle'

fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  #previously: open_file = open(os.path.join(pickle_path, 'postprocessing', 'paper2' ,fn), "rb") 
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()






#%%


"""

"BEST"

SHUFFLE

"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,  sharey=True, figsize=(4,6))
fig.subplots_adjust(hspace=0.1,wspace=0.1)

dPD_mismatch = {}

for mode in ['shuffle']:
    
    dPD_mismatch[mode] = {}
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')

        assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])

   
        popMean = []
        for date in dDates[i][:]:
            
            PD = dPD_each[i][date]['dPD_median']
            assigned = dPD_each[i][date]['assigned_dPD']
            
        
            #popMean.append(  np.mean( np.abs(PD - assigned) )  )
            popMean.append(np.abs(np.mean(np.abs(PD)))-np.mean(np.abs(assigned)))
                  
        
        dPD_mismatch[mode][i] = np.array(popMean)#np.abs(np.array(popMean))
    
        
        
        for j, BEH_VAR in enumerate(['time', 'dist']):#enumerate(['delta', 'best']):
            KEY = 'init'
            v1 = dPD_mismatch[mode][i]
            v2 = dBEH[mode][i][BEH_VAR][KEY]
            
            r,p,star = compute_corr(v1,v2)

            if p > 0.1:
                ls='--'
                lc='grey'
               
            else:
                ls='-'
                lc='k'

            params, _ = curve_fit(linear_model, v1, v2)

            x1 = np.min(v1)
            x2 = np.max(v1)
            
            y1 = linear_model(x1,*params)
            y2 = linear_model(x2,*params)
            ax[j][i].plot([x1,x2],[y1, y2], color=lc, lw=2, ls=ls, zorder=100)
            
            
            ax[j][i].axhline(0, color='grey', zorder=0, ls='--', lw=0.75)
            
            ax[j][i].set_xlim([-3,78])
            ax[j][i].set_xticks([0,25,50,75])
            
            
            inds = np.where(v2<200)[0]
            
            if len(inds) != len(v2):
                print(i, KEY, len(v2)-len(inds))
            
            v1 = v1[inds]
            v2 = v2[inds]
            
            if KEY == 'init':
                color=magenta30
            else:
                color=magenta60
            
            ax[j][i].scatter(v1,v2, color=color, alpha=0.6, edgecolor='k')
            
            if KEY == 'init':
                ax[j][i].set_ylim([-50,100])  
                ax[j][i].set_yticks([-30,0,30,60,90])
                if i == 0:
                    ax[j][0].set_yticklabels([-30,'BL',30,60,90])
                
                ax[j][i].text(5,90,f'r={r:.2f} ({star})')
                
            elif KEY == 'best':
                ax[j][i].set_ylim([-50,100])  
                ax[j][i].set_yticks([-30,0,30,60,90])
                if i == 0:
                    ax[j][0].set_yticklabels([-30,'BL',30,60,90])
                
                ax[j][i].text(5,90,f'r={r:.2f} ({star})')

ax[1][0].set_xlabel('ΔPD mismatch (°)')  
ax[1][1].set_xlabel('ΔPD mismatch (°)')  


if KEY == 'init':
    ax[0][0].set_ylabel(f'first\n%Δ in trial time from BL')
    ax[1][0].set_ylabel(f'first\n%Δ in distance from BL')
    
elif KEY == 'best':
    ax[0][0].set_ylabel(f'best\n%Δ in trial time from BL')
    ax[1][0].set_ylabel(f'best\n%Δ in distance from BL')




#%%

"""

"DELTA"

ROTATION - Mean of each "mismatch"
    mean(ABS(PD_each - PD_assigned)) 

"""

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,  sharey=True, figsize=(4,6))
# fig.subplots_adjust(hspace=0.1,wspace=0.1)

# dPD_mismatch = {}

# for mode in ['rotation']:
    
#     dPD_mismatch[mode] = {}
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    
    
#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
#         ax[0][i].set_title(f'{subject}', loc='left')

#         assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])

#         dPD_mismatch[mode][i] = np.zeros(len(dDates[i]))
#         for d, date in enumerate(dDates[i]):
            
            
#             #TODO: !!!
#             #dPD_mismatch[mode][i][d] = np.mean( np.abs( dPD_each[i][date]['dPD_median'] - assigned[d] ) )
#             dPD_mismatch[mode][i][d] = np.abs(np.mean(dPD_each[i][date]['dPD_median']) -assigned[d])
    
#         v1 = dPD_mismatch[mode][i]
        
#         for j, BEH_VAR in enumerate(['time', 'dist']):#enumerate(['delta', 'best']):
#             v2 = dBEH[mode][i][BEH_VAR]['delta']
            
#             ax[j][i].scatter(v1,v2, color='grey', alpha=0.3, edgecolor='k')
            
#             r,p,star = compute_corr(v1,v2)

#             if p > 0.1:
#                 ls='--'
#                 lc='k'
               
#             else:
#                 ls='-'
#                 lc='b'

#             params, _ = curve_fit(linear_model, v1, v2)
            
#             x1 = np.min(v1)
#             x2 = np.max(v1)
            
#             y1 = linear_model(x1,*params)
#             y2 = linear_model(x2,*params)
#             ax[j][i].plot([x1,x2],[y1, y2], color=lc, lw=2, ls=ls, zorder=100)
            
#             ax[j][i].set_ylabel('')
            
#             ax[j][i].axhline(0, color='grey', zorder=0, ls='--', lw=0.75)
            
#             ax[j][i].set_xlim([-3,78])
#             ax[j][i].set_xticks([0,25,50,75])
            
#             ax[j][i].set_ylim([-4,105])  
#             ax[j][i].set_yticks([0,20,40,60,80,100])
#             if i == 0:
#                 ax[j][0].set_yticklabels(['BL',20,40,60,80,100])
            
#             ax[j][i].text(5,95,f'r={r:.2f} ({star})')
            

# ax[1][0].set_xlabel('ΔPD mismatch (°)')  
# ax[1][1].set_xlabel('ΔPD mismatch (°)')  

# ax[0][0].set_ylabel('Δ time (%)')
# ax[1][0].set_ylabel('Δ distance (%)')

#%%

"""

"DELTA"

SHUFFLE

"""

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,  sharey=True, figsize=(4,6))
# fig.subplots_adjust(hspace=0.1,wspace=0.1)

# dPD_mismatch = {}

# for mode in ['shuffle']:
    
#     dPD_mismatch[mode] = {}
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    
    
#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
#         ax[0][i].set_title(f'{subject}', loc='left')


#         popMean = []
#         for date in dDates[i][:]:
            
#             PD = dPD_each[i][date]['dPD_median']
#             assigned = dPD_each[i][date]['assigned_dPD']
            
            
#             #TODO: !!!
#             popMean.append(  np.mean( np.abs(PD - assigned) )  )
#             #popMean.append(np.abs(np.mean(np.abs(PD)))-np.mean(np.abs(assigned)))
            
        
   
#         dPD_mismatch[mode][i] = np.array(popMean)#np.abs(popMean)
    
#         v1 = dPD_mismatch[mode][i]
        
#         for j, BEH_VAR in enumerate(['time', 'dist']):#enumerate(['delta', 'best']):
#             v2 = dBEH[mode][i][BEH_VAR]['delta']
            
#             ax[j][i].scatter(v1,v2, color='grey', alpha=0.3, edgecolor='k')
            
#             r,p,star = compute_corr(v1,v2)

#             if p > 0.1:
#                 ls='--'
#                 lc='k'
               
#             else:
#                 ls='-'
#                 lc='b'

#             params, _ = curve_fit(linear_model, v1, v2)
            
#             x1 = np.min(v1)
#             x2 = np.max(v1)
            
#             y1 = linear_model(x1,*params)
#             y2 = linear_model(x2,*params)
#             ax[j][i].plot([x1,x2],[y1, y2], color=lc, lw=2, ls=ls, zorder=100)
            
#             ax[j][i].set_ylabel('')
            
#             ax[j][i].axhline(0, color='grey', zorder=0, ls='--', lw=0.75)
            
#             ax[j][i].set_xlim([-3,78])
#             ax[j][i].set_xticks([0,25,50,75])
            
            
#             ax[j][i].set_ylim([-40,160])  
#             ax[j][i].set_yticks([-30,0,30,60,90,120,150])
#             if i == 0:
#                 ax[j][0].set_yticklabels([-30,'BL',30,60,90,120,150])
            
#             ax[j][i].text(5,145,f'r={r:.2f} ({star})')
            

# ax[1][0].set_xlabel('ΔPD mismatch (°)')  
# ax[1][1].set_xlabel('ΔPD mismatch (°)')  

# ax[0][0].set_ylabel('Δ time (%)')
# ax[1][0].set_ylabel('Δ distance (%)')



