# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:28:19 2025

@author: hanna
"""

"""


    Figure 6: relationships between behavior and neural activity


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

# palette_BEH = {50: yellow, 90: blue}
# palette_NEU = {50: orange, 90: purple}


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


# dColor = {'50': magenta60, 'neg50': magenta30, '90': teal60, 'neg90': teal60}
# palette_COM_PRIVATE = {'50': teal30,    '90': teal60}
# palette_COM_SHARED  = {'50': magenta30, '90': magenta60}


dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'           
            
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
os.chdir(os.path.join(custom_functions_path,'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star


def linear_model(x,m,b):
    return(m*x+b)


pickle_root_save_path = os.path.join(pickle_path, 'FA')
pickle_path_BEH = os.path.join(pickle_path, 'dfBEH', 'compute_behavior_results')


#%%

""" blockwise changes in neural variance metrics """

dV = {}

for mode in ['rotation']:
    
    dV[mode] = {}

    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()

    
    for i in [0,1]:
        
        dV[mode][i] = {'tv': np.zeros((len(dDates[i]))),
                           'sv': np.zeros((len(dDates[i]))),
                           'pv': np.zeros((len(dDates[i])))}#,
                           # 'dFR_all': np.zeros((len(dDates[i]))),
                           # 'dFR_each': np.zeros((len(dDates[i])))}
    
        
        subj, subject = dSubject[i]
        
        for d, date in enumerate(dDates[i]):
      
            fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
            _, dSC = pickle.load(open_file)
            open_file.close()
            
            'Neural'
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA6', fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            
            k = 'BL'
            # tvBL = dVAR[k]['total']
            svBL = dVAR[k]['shared']
            pvBL = np.sum(dVAR[k]['MAT_private']/np.mean(dSC[k], axis=0))
            tvBL = svBL + pvBL
            
            k = 'PE'
            # tvPE = dVAR[k]['total']
            svPE = dVAR[k]['shared']
            pvPE = np.sum(dVAR[k]['MAT_private']/np.mean(dSC[k], axis=0))
            tvPE = svPE + pvPE
            
            
            deltaTV = 100*((tvPE-tvBL)/tvBL)
            deltaSV = 100*((svPE-svBL)/svBL)
            deltaPV = 100*((pvPE-pvBL)/pvBL)
            

            '-delta'
            dV[mode][i]['tv'][d] = deltaTV
            dV[mode][i]['sv'][d] = deltaSV
            dV[mode][i]['pv'][d] = deltaPV

               


#%%


""" blockwise changes in behavior (trial time) """



dB = {}


for mode in ['rotation']:

    dB[mode] = {}
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
        
        dB[mode][i] = {'time': {
                                    'delta': np.zeros((len(dDates[i]))), 
                                    'init': np.zeros((len(dDates[i]))), 
                                    'best': np.zeros((len(dDates[i]))), 
                                    'aol': np.zeros((len(dDates[i]))), } }



        subj, subject = dSubject[i]

        for d in tqdm(range(len(dDates[i][:]))):
    
            
            date = dDates[i][d]
            
            open_file = open(os.path.join(pickle_path_BEH, f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
            dfBEH_BL, dfBEH_PE = pickle.load(open_file)
            open_file.close()
            
            for BEH_KEY in ['time']:
                
                PE_BEH = np.zeros((8,len(dfBEH_BL)//8))
                for j, deg in enumerate(np.arange(0,360,45)):
                    PE_BEH[j,:] = dfBEH_PE.loc[dfBEH_PE['deg'] == deg, BEH_KEY][:len(dfBEH_BL)//8]
                
                mBL = np.mean(dfBEH_BL[BEH_KEY])
                mPE_ = np.mean(PE_BEH)#, axis=0)
                
                
                dB[mode][i][BEH_KEY]['delta'][d] = 100*((mPE_-mBL)/mBL)
                
                
                mPE = np.mean(PE_BEH, axis=0)
                deltaBEH = 100*((mPE-mBL)/mBL)
                
               
                init = deltaBEH[0]
                best = np.min(deltaBEH)
                AOL  =  (init-best)/init
                best_TSN = np.where(deltaBEH == best)[0][0]
                
                dB[mode][i][BEH_KEY]['init'][d] = init 
                dB[mode][i][BEH_KEY]['best'][d] = best
                dB[mode][i][BEH_KEY]['aol'][d]  = AOL

#%%











#%%

"""

[Figure 6A] SHARED variance

"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(5,8))
fig.subplots_adjust(wspace=0.1, hspace=0.1)


for mode in ['rotation']:
 
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')
        
        # open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        # dDates, dDegs, dDegs2 = pickle.load(open_file)
        # open_file.close()
        
        # ind50 = np.where(dDegs[i]==50)[0]
        # ind90 = np.where(dDegs[i]==90)[0]
        
        for j,BEH_KEY in enumerate(['best', 'aol']):   #'delta',

            v1 = dV[mode][i]['sv']#[ind50]#[inds]
            v2 = dB[mode][i]['time'][BEH_KEY]#[ind50]#[inds]
                

            r,p,star = compute_corr(v1,v2)

            if BEH_KEY == 'best':
                yp = 58
                ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
            elif BEH_KEY == 'aol':
                yp = 2.45
                ax[j][i].axhline(1, color='grey', zorder=0, ls='--')

            ax[j][i].text(-50,yp,f'r={r:.2f} ({star})')#, color='red')
            ax[j][i].scatter(v1,v2, color=teal60, s=25, alpha=0.5)
            
            ax[j][i].axvline(0, color='grey', zorder=0, ls='--')
            
            
            params, _ = curve_fit(linear_model, v1, v2)
    
            x1 = np.min(v1)+2
            x2 = np.max(v1)-2
            
            y1 = linear_model(x1,*params)
            y2 = linear_model(x2,*params)
            ax[j][i].plot([x1,x2],[y1, y2], color='k', lw=2, ls='-', zorder=100)
            
            
            ax[j][i].set_xlim([-55,40])
            ax[j][i].set_xticks([-45,-30,-15,0,15,30])
            ax[j][i].set_xticklabels([-45,-30,-15,'BL',15,30], rotation=0, fontsize=11)

# ax[0][0].set_ylabel(f'Δ time (%)')
ax[0][0].set_ylabel(f'best performance\n(%Δ in trial time from BL)', fontsize=fontsize+1)  
ax[1][0].set_ylabel(f'amount of recovery', fontsize=fontsize+1)  

ax[j][0].set_xlabel(f'Δ shared variance (%)', fontsize=fontsize+1)
ax[j][1].set_xlabel(f'Δ shared variance (%)', fontsize=fontsize+1) 


for i in [0,1]:

    ax[0][i].set_ylim([-30,65])
    ax[0][i].set_yticks([-20,0,20,40,60])
    ax[0][i].set_yticklabels([-20,'BL',20,40,60], fontsize=11)


    ax[1][i].set_ylim([-0.1,2.6])
    ax[1][i].set_yticks([0,0.5,1,1.5,2.0,2.5])
    ax[1][i].set_yticklabels([0,0.5,1,1.5,2.0,2.5], fontsize=11)
    
    
    
    if i == 1:
        ax[0][i].set_yticklabels([])
        ax[1][i].set_yticklabels([])
    
    

#%%

"""

[Figure 6A] PRIVATE variance

"""
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(5,8))
fig.subplots_adjust(wspace=0.1, hspace=0.1)


for mode in ['rotation']:
 
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')


        for j,BEH_KEY in enumerate(['best', 'aol']):

            v1 = dV[mode][i]['pv']#[ind50]#[inds]
            v2 = dB[mode][i]['time'][BEH_KEY]#[ind50]#[inds]
                

            r,p,star = compute_corr(v1,v2, trending=False)

            if BEH_KEY == 'best':
                yp = 58
                ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
            elif BEH_KEY == 'aol':
                yp = 2.45
                ax[j][i].axhline(1, color='grey', zorder=0, ls='--')
            
            
            if p < 0.05:

                ax[j][i].text(-50,yp,f'r={r:.2f} ({star})')#, color='red')
            else:
                ax[j][i].text(-50,yp,f'n.s.', fontstyle='italic')
            ax[j][i].scatter(v1,v2, color=magenta60, s=25, alpha=0.5)
            
            ax[j][i].axvline(0, color='grey', zorder=0, ls='--')
            
            
            params, _ = curve_fit(linear_model, v1, v2)
    
            x1 = np.min(v1)+2
            x2 = np.max(v1)-2
            
            y1 = linear_model(x1,*params)
            y2 = linear_model(x2,*params)
            
            if p > 0.05:
                # ls = '-.'
                # lc = 'grey'
                pass
            else:
                ls = '-'
                lc = 'k'
                
                ax[j][i].plot([x1,x2],[y1, y2], color=lc, lw=2, ls=ls, zorder=100)
            
            
            ax[j][i].set_xlim([-55,40])
            ax[j][i].set_xticks([-45,-30,-15,0,15,30])
            ax[j][i].set_xticklabels([-45,-30,-15,'BL',15,30], rotation=0, fontsize=11)

# ax[0][0].set_ylabel(f'Δ time (%)')
ax[0][0].set_ylabel(f'best performance\n(%Δ in trial time from BL)', fontsize=fontsize+1)  
ax[1][0].set_ylabel(f'amount of recovery', fontsize=fontsize+1)  
 

ax[j][0].set_xlabel(f'Δ private variance (%)', fontsize=fontsize+1)
ax[j][1].set_xlabel(f'Δ private variance (%)', fontsize=fontsize+1) 


for i in [0,1]:

    ax[0][i].set_ylim([-30,65])
    ax[0][i].set_yticks([-20,0,20,40,60])
    ax[0][i].set_yticklabels([-20,'BL',20,40,60], fontsize=11)


    ax[1][i].set_ylim([-0.1,2.6])
    ax[1][i].set_yticks([0,0.5,1,1.5,2.0,2.5])
    ax[1][i].set_yticklabels([0,0.5,1,1.5,2.0,2.5], fontsize=11)
    
    
    
    if i == 1:
        ax[0][i].set_yticklabels([])
        ax[1][i].set_yticklabels([])
    
    
    
    
#%%

'STATS ONLY - Figure 6'

for mode in ['rotation']:
    
    for VAR_KEY in ['sv', 'pv']:
 
        for i in [0,1]:
            
            subj, subject = dSubject[i]
    
            for j,BEH_KEY in enumerate(['best', 'aol']): #'delta',
    
                v1 = dV[mode][i][VAR_KEY]
                v2 = dB[mode][i]['time'][BEH_KEY]
                    
    
                r,p,star = compute_corr(v1,v2)
                
                
                if p > 0.01:
                    print(f'{subject} | {VAR_KEY} v. {BEH_KEY}: r={r:.2f}, p={p:.2f}, {star}')
                else:
                    print(f'{subject} | {VAR_KEY} v. {BEH_KEY}: r={r:.2f}, p={p:.1e}, {star}')


        print('')

#%%

'STATS ONLY - Figure 6 - BY SESSION [Easy/50]'

for mode in ['rotation']:
    
    for VAR_KEY in ['sv', 'pv']:
    
        for BEH_KEY in ['best', 'aol']: #'delta'
     
            for i in [0,1]:
                
                subj, subject = dSubject[i]
        
                open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
                dDates, dDegs, dDegs2 = pickle.load(open_file)
                open_file.close()
                
                ind50 = np.where(dDegs[i]==50)[0]
                ind90 = np.where(dDegs[i]==90)[0]
                
                
                for LAB, INDS in zip(['easy', 'hard'],[ind50, ind90]):
    
                    v1 = dV[mode][i]['sv'][INDS]
                    v2 = dB[mode][i]['time'][BEH_KEY][INDS]
                        
        
                    r,p,star = compute_corr(v1,v2)
                    
                    
                    if p > 0.01:
                        print(f'{subject} | {VAR_KEY} ({LAB}) v. {BEH_KEY}: r={r:.2f}, p={p:.2f}, {star}')
                    else:
                        print(f'{subject} | {VAR_KEY} ({LAB}) v. {BEH_KEY}: r={r:.2f}, p={p:.1e}, {star}')
        
            print('')
    


    
#%%

"""

Figure 7????

"""
    


#%%

# """

# TEST
#     Does the amount of baseline variability (normalized by the number of neurons)

# """


# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(5,8))
# fig.subplots_adjust(wspace=0.1, hspace=0.1)


# for mode in ['rotation']:
 
#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
#         ax[0][i].set_title(f'{subject}', loc='left')
        

#         for j,BEH_KEY in enumerate(['init','best', 'aol']):

            
#             v1 = dBL[mode][i]['sv']/dBL[mode][i]['nU']
#             v2 = dV_BEH[mode][i]['time'][BEH_KEY]
                
#             axis = ax[j][i]
#             axis.scatter(v1,v2, color='k', s=6, alpha=0.5)
#             r,p,star = compute_corr(v1,v2)
#             axis.text(10,0,f'{star}')

            
            
#             params, _ = curve_fit(linear_model, v1, v2)
    
#             x1 = np.min(v1)+2
#             x2 = np.max(v1)-2
            
#             y1 = linear_model(x1,*params)
#             y2 = linear_model(x2,*params)
#             axis.plot([x1,x2],[y1, y2], color='k', lw=2, ls='-', zorder=100)
            
       



#%%
"""

Is SSA correlation with any behavior?
    predictably (bc deltaSV results above) - yes

"""

# mode = 'rotation'

# open_file = open(os.path.join(pickle_path_VAR, f'dSSA_{mode}.pkl'), "rb")
# dSSA = pickle.load(open_file)
# open_file.close()
    
   

# for i in [0]:
    
#     subj,subject = dSubject[i]
#     fig, ax = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=False, figsize=(4,12))
#     ax[0][0].set_title(f'{subject}', loc='left')
    
#     ind50 = np.where(dDegs[i]==50)[0]
#     ind90 = np.where(dDegs[i]==90)[0]

    
   
#     for j, SSA_KEY in zip([0,1],['BL_to_PE', 'PE_to_BL']):
        

#         v1 = dSSA[i][SSA_KEY]
        
        
#         for k,BEH_KEY in enumerate(['best', 'aol']):

      
#             v2 = dV_BEH[mode][i]['time'][BEH_KEY]#[ind90]#[inds]
#             ax[k][j].scatter(v1,v2, color='k', s=6, alpha=0.5)
#             ax[k][j].set_xlim([0.5,1.1])
                
#             r,p,star = compute_corr(v1,v2)
            
#             print(f'{subject} | {SSA_KEY}, {BEH_KEY} | r={r:.2f} p={p:.2f} ({star})')
        

#         print('')
    
#     print('')
    
                
 
            
            
            
            
            
            