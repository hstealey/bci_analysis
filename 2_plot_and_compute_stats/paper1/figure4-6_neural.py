# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:24:20 2025

@author: hanna
"""

"""

    Figure 3: Blockwise Changes in Neural Activity
    
        tv (include?)
        sv ("main shared"; no mean-matching)
        
            TODO: FR PLOT
        
        pv (normalize by mean firing rate?)

    Figure 4: dshared, SSA
        dshared
        SSA
    
    
    
    
    [loading similarity]
    [pairwise metrics]

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
palette_ROT4 = {-50: yellow, 50: orange, -90: blue, 90: purple}

dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'           
            

'------------------------------'
'Loading Custom Functions'
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
os.chdir(os.path.join(custom_functions_path, 'neural_fxns'))
from factor_analysis_fxns import fit_fa_model
os.chdir(os.path.join(custom_functions_path, 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star





#%%


n_window = 9
path_window = f'fixed_window_{n_window}' #alt: 'fixed_window_0-6' 

pickle_path_VAR = os.path.join(pickle_path, 'FA', 'TTT', path_window, 'FA5', 'eachNF')
pickle_path_VAR_FA6 = os.path.join(pickle_path_VAR, 'FA6')
pickle_path_SC = os.path.join(pickle_path,  'FA', 'TTT', path_window, 'FA2')

pickle_path_FA1 = os.path.join(pickle_path,  'FA', 'TTT', path_window, 'FA1')

#%%

"""

STATS: number of neurons included/exlcuded

"""


nU = {}

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:

        nU[i] = {'pre': np.zeros(len(dDates[i])),
                 'post': np.zeros(len(dDates[i])),
                 'diff': np.zeros(len(dDates[i]))}        

        subj, subject = dSubject[i]

        
        for d, date in enumerate(tqdm(dDates[i])): 

            fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_FA1, fn), "rb")
            delU, dNU, dParams1, _, _ = pickle.load(open_file)
            open_file.close()
            
            n_sets = dParams1['n_sets']
            n_trials = dParams1['n_trials']
            window_len = dParams1['window_len']
            
            if (n_sets!=34) or (n_trials!=272) or (window_len!=9):
                print("ERROR")
                
            
            nU[i]['pre'][d]  = dNU['nU_pre']
            nU[i]['post'][d] = dNU['nU_post']
            nU[i]['diff'][d] = dNU['nU_pre'] - dNU['nU_post']

#%%

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]

        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        

        
        
        
        v1 = nU[i]['post'][ind50]
        v2 = nU[i]['post'][ind90]
        
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        n1 = len(ind50)
        n2 = len(ind90)
        
        dof = n1+n2-2
        
        print(f'{subject}| t{dof}={t:.2f}, p={p:.2f} {star}, equal_var={equal_var}')
        
        print(f'{subject}: easy = {np.mean(v1):.2f} +/- {np.std(v1):.2f}')
        print(f'{subject}: hard = {np.mean(v2):.2f} +/- {np.std(v2):.2f}')
        
        
        v1_ex = nU[i]['diff'][ind50]
        v2_ex = nU[i]['diff'][ind90]
        
        print(f'\t{subject}: EXCLUDE easy = {np.mean(v1_ex):.2f} +/- {np.std(v1_ex):.2f}')
        print(f'\t{subject}: EXCLUDE hard = {np.mean(v2_ex):.2f} +/- {np.std(v2_ex):.2f}')
        
        


#%%


"""


[Dissertation Figure 4]

    "main" shared (using nf=dshared)
    normalized private by mean firing rate
    total = sum of shared + private (as above)


"""


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dRES = {}
    
    for i in [0,1]:
        

        dRES[i] = {'BL':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))},
                   'PE':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))},
                   'delta':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))}}

        subj, subject = dSubject[i]
        
        for d, date in enumerate(tqdm(dDates[i])): 
            
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_VAR_FA6, fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()

     
            fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_SC, fn), "rb")
            _, dSC = pickle.load(open_file)
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
            
            dRES[i]['BL']['tv'][d] = tvBL
            dRES[i]['BL']['sv'][d] = svBL
            dRES[i]['BL']['pv'][d] = pvBL
            
            dRES[i]['PE']['tv'][d] = tvPE
            dRES[i]['PE']['sv'][d] = svPE
            dRES[i]['PE']['pv'][d] = pvPE
            
            dRES[i]['delta']['tv'][d] = deltaTV
            dRES[i]['delta']['sv'][d] = deltaSV
            dRES[i]['delta']['pv'][d] = deltaPV
            

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(5,10))         
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    
    for i in [0,1]:
        

        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        VAR_ = dRES[i]['delta']
        
        for j, VAR_KEY in zip([0,1,2], ['tv', 'sv', 'pv']):
 
            VAR = VAR_[VAR_KEY]
            
            ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
            sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[j][i], palette=palette_ROT, alpha=0.8, edgecolor='k', lw=3)
            #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[j][i], jitter=0.1, palette=palette_ROT)
                        
            v1 = VAR[ind50]
            v2 = VAR[ind90]
            ax[j][i].scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=40, zorder=10)
            (_, caps, _) = ax[j][i].errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=8, zorder=9, ls='')
            for cap in caps:
                cap.set_markeredgewidth(2.5)
            

                
            t,p,equal_var,star = compute_two_sample_ttest(v1,v2,trending=False)
            ax[j][i].plot([0,1],[35,35],color='k')
            
            if p < 0.05:
                fontsize_p = fontsize+5
                y_p = 35-3
                
            else:
                fontsize_p = fontsize+2
                y_p = 35
            
            ax[j][i].text(0.5,y_p,f'{star}',ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p)
            
        
            t,p,star = compute_one_sample_ttest(v1,0,trending=False)
            if p < 0.05:
                y_p = -3
            else:
                y_p = 0.5
                
            if (j==2) and (i==0): 
                y_p = 4
                fontsize_p_v1 = fontsize+4
            
            if (j==2) and (i==1):
                y_p = 3
                
            ax[j][i].plot([0,0.4],[np.mean(v1), np.mean(v1)],  color='grey')#, ls='--')
            ax[j][i].plot([0.4,0.4],[0, np.mean(v1)],  color='grey')#, ls='--')
            ax[j][i].text(0.37,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)


            t,p,star = compute_one_sample_ttest(v2,0,trending=False)
            if p < 0.05:
                y_p = -3
            else:
                y_p = 0.5
            
            if (j==2) and (i==0): 
                y_p = 4
                
            
            
            if (j==2) and (i==1):
                y_p = 1.5
                
                
            ax[j][i].plot([1,0.6],[np.mean(v2), np.mean(v2)],  color='grey')#, ls='--')
            ax[j][i].plot([0.6,0.6],[0, np.mean(v2)],  color='grey')#, ls='--')
            ax[j][i].text(0.62,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)

            ax[j][i].set_ylim([-50,50])
        
        
            
            ax[j][i].set_yticks([-40,-20,0,20,40])

        
        ax[2][i].set_xticks([0,1])
        ax[2][i].set_xticklabels(['easy', 'hard'])
        ax[2][i].set_xlabel('rotation condition')
                
            
ax[0][0].set_ylabel('Δ total variance (%)')
ax[1][0].set_ylabel('Δ shared variance (%)')          
ax[2][0].set_ylabel('Δ private variance (%)')    



ax[0][0].set_xlim([-0.4,1.4])   

for j in [0,1,2]:
    ax[j][0].set_yticklabels([-40,-20,'BL',20,40])
        


#%%

"""

blockwise - print STATS only

"""


mode = 'rotation'
    
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()
    

for VAR_KEY in ['tv', 'sv', 'pv']:
    
    for i in [0,1]:
        
        subj,subject = dSubject[i]
        
        VAR = dRES[i]['delta'][VAR_KEY]
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]

      
        v1 = VAR[ind50]
        v2 = VAR[ind90]
     
        #all are equal var
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2,trending=False)
        n1 = len(v1)
        n2 = len(v2)
        dof = n1+n2-2
        if p > 0.05:
            print(f'{VAR_KEY} | {subject}, t{dof}={t:.2f}, p={p:.2f}')#' equal_var={equal_var}')
        else:
            print(f'{VAR_KEY} | {subject}, t{dof}={t:.2f}, p={p:.1e}')#' equal_var={equal_var}')
    
    
        t,p,star = compute_one_sample_ttest(v1,0,trending=False)
        dof = n1-1
        if p > 0.05:
            print(f'\t{VAR_KEY} | E {subject}, t{dof}={t:.2f}, p={p:.2f}')
        else:
            print(f'\t{VAR_KEY} | E {subject}, t{dof}={t:.2f}, p={p:.1e}')
    
    
    
        t,p,star = compute_one_sample_ttest(v2,0,trending=False)
        dof = n2-1
        if p > 0.05:
            print(f'\t{VAR_KEY} | H {subject}, t{dof}={t:.2f}, p={p:.2f}')
        else:
            print(f'\t{VAR_KEY} | H {subject}, t{dof}={t:.2f}, p={p:.1e}')
        
        print('')
 
        

#%%


"""


[Dissertation Figure 5]

    Firing rate changes are not significant
        mean differences, equal variances (?), K-S test? (or difference between...)

        "paired" but do sorted???


"""



fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(12,6))         
fig.subplots_adjust(hspace=0.15)


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()

    
    for i in [0,1]:
        

        subj, subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left', fontsize=fontsize+2)
        
        #for d, date in enumerate(tqdm(dDates[i][:30])): 
            
        ind50 = np.where(dDegs[i] == 50)[0]
        ind90 = np.where(dDegs[i] == 90)[0]    
        
        count=0
        for indDEG, color in zip([ind50, ind90],[orange,purple]):
        #for indDEG, color in zip([ind90],[purple]):   
            # m_condish_BL = []
            # m_condish_PE = []
            
            for date in dDates[i][indDEG][:]:
                fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_path_SC, fn), "rb")
                _, dSC = pickle.load(open_file)
                open_file.close()
                
                
                mBL = np.mean(dSC['BL'], axis=0)
                sBL = np.std(mBL)
                mPE = np.mean(dSC['PE'], axis=0)
                sPE = np.std(mPE)
                
                # if np.mean(mBL)-sBL < 0:
                #     yerr_lo = 0
                # else:
                #     yerr_lo = np.mean(mBL)-sBL
                
                
                #ax[i].scatter(np.ones(len(mBL))*d-0.25, mBL, color='k', alpha=0.1, s=2, zorder=0)
                'Baseline'
                ax[i].scatter(count-0.5, np.mean(mBL), marker='s', color='k', s=15, zorder=2)
                ax[i].errorbar(count-0.5, y=np.mean(mBL), yerr=sBL,color='k', zorder=1)
                
                'Perturbation'
                ax[i].scatter(count+0.5, np.mean(mBL), marker='s', color=color,  s=15, zorder=2)
                ax[i].errorbar(count+0.5, y=np.mean(mPE), yerr=sPE,color=color, zorder=1)
 
                count+=3
                
                
                t,p,equal_var,star = compute_two_sample_ttest(mBL, mPE)
                
                if equal_var == False:
                    print(i,date)
          
                
                if p < 0.05:
                    print(i,date)
            
            
                _, p = stats.ks_2samp(mBL, mPE)
                
                if p < 0.05:
                    print(i, date)
                
                
            if color==orange:
                ax[i].axvline(count-1.5, color='grey', lw=2, ls='-' )
                #count+=3
                ax[i].text(count+1, 30, 'hard', color=purple, fontsize=fontsize+4)
                           

        
        ax[i].text(-1, 30, 'easy', color=orange, fontsize=fontsize+4)
        ax[i].set_ylabel('firing rate (Hz)', fontsize=fontsize+2)
        ax[i].set_yticks([0,5,10,15,20,25,30])
        
        if i == 0:
            ax[i].set_ylim([-8,35])
        else:
            ax[i].set_ylim([-2,35])
        
        ax[i].set_xlim([-3,190])
        ax[i].set_xticks(np.arange(0,189,3*10))
        ax[i].set_xticklabels(np.arange(0,63,10))

ax[1].set_xlabel('session (sorted by rotation condition)', fontsize=fontsize+2)

#%%


"""

TODO: maybe later???


    [Dissertation Figure 5 cont.]
    
        example distributions

"""



# fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12,4))         
# fig.subplots_adjust(hspace=0.15)


# mode = 'rotation'
    
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates, dDegs, dDegs2 = pickle.load(open_file)
# open_file.close()


# i = 0


# subj, subject = dSubject[i]

            
# dates50 = dDates[i][np.where(dDegs[i] == 50)[0]]
# dates90 = dDates[i][np.where(dDegs[i] == 90)[0]]    

# j = 0
# for date, color in zip([dates50[0], dates90[0]],[orange,purple]):
           
#     fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
#     open_file = open(os.path.join(pickle_path_SC, fn), "rb")
#     _, dSC = pickle.load(open_file)
#     open_file.close()
    
    
#     mBL = np.mean(dSC['BL'], axis=0)
#     #sBL = np.std(mBL)
#     mPE = np.mean(dSC['PE'], axis=0)
#     #sPE = np.std(mPE)
    
#     # ax[j].hist(mBL, color='k', zorder=0, density=True)
#     # ax[j].hist(mPE, color=color, zorder=1, density=True, alpha=0.5)
    
#     #sb.swarmplot(x=np.zeros(len(mBL)), y=mBL, color=orange, ax=ax)
#     ax.plot(np.sort(mBL), color='k')
#     ax.plot(np.sort(mPE), color=orange)
    
    
#     j+=1
    



#%%


"""

[Dissertation Figure 6] Shared Space Alignment

PE->BL

"""

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(4,8))

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    open_file = open(os.path.join(pickle_path_VAR_FA6, f'dSSA_{mode}.pkl'), "rb")
    dSSA = pickle.load(open_file)
    open_file.close()



    for i in [0,1]:
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        subj,subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left', fontsize=fontsize)
    
        VAR = dSSA[i]['PE_to_BL'] 
        sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[i], order=[50, 90], palette=palette_ROT, alpha=0.7 )#, order=[-50, 50, -90, 90, -999], palette=palette_COMB)
        
        ax[i].set_xticks([0,1])
        ax[i].set_xticklabels(['easy', 'hard'])
        ax[i].set_xlabel('rotation condition')
        
        
        v1 = VAR[ind50]
        v2 = VAR[ind90]
    
    
        ax[i].scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2,  s=40, zorder=10)
        (_, caps, _) = ax[i].errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        n1 = len(v1)
        n2 = len(v2)
        dof = n1 + n2 - 2
        print(f'{subject} | PE to BL | t{dof}={t:.2f}, p={p:.1e}, equal_var={equal_var}')
        
        ax[i].set_ylim([0.45,1.03])#ax[i].set_ylim([0,1.03])
        #ax[i].plot([0,1],[1.01,1.01], color='k')
        ax[i].text(0.5,1.01,f'{star}', fontsize=fontsize+3, ha='center', va='center', fontstyle='italic')
        #ax[i].set_yticks(np.arange(0,1.1,0.1))
         
    ax[0].set_ylabel('shared space alignment', fontsize=fontsize+1)
    
    

#%%

"""

[Figure 4a] dshared (number of factors to explain 95% of the shared variance)

BY BLOCK

"""
import warnings
warnings.filterwarnings("ignore")


bins = np.arange(-3,4,1)


fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(5.5,8))#9))         
            


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')
        ax[1][i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        nfBL_95 = np.zeros(len(dDates[i]))
        nfPE_95 = np.zeros(len(dDates[i]))
        
        
        delta = np.zeros(len(dDates[i]))
        

        for d, date in enumerate(dDates[i]):
            
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_VAR_FA6, fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            nfBL_95[d] = dVAR['BL']['dshared95']
            nfPE_95[d] = dVAR['PE']['dshared95']
            
            nf1 = dVAR['BL']['dshared95']
            nf2 = dVAR['PE']['dshared95']
            
            delta[d] = nf2 - nf1 #100*((nf2-nf1)/nf1)
            
            color = palette_ROT[dDegs[i][d]]
            if dDegs[i][d] == 50:
                ax[0][i].plot([0,2],[nf1, nf2], color=color, alpha=0.1)
                ax[0][i].scatter([0,2],[nf1, nf2], zorder=10, color=color, alpha=0.2, edgecolor='k')
            elif dDegs[i][d] == 90:
                ax[0][i].plot([4,6],[nf1, nf2], color=color, alpha=0.1)
                ax[0][i].scatter([4,6],[nf1, nf2], zorder=10, color=color, alpha=0.2, edgecolor='k')

        ax[0][i].set_ylim([0,10])
        ax[0][i].set_xlim([-1,7])
        ax[0][i].set_xticks([0,2,4,6])
        ax[0][i].set_xticklabels(['BL\neasy', 'PE\neasy', 'BL\nhard', 'PE\nhard'])
        

        
        
        sb.swarmplot(x=dDegs[i], y=delta, ax=ax[1][i], palette=palette_ROT)
        ax[1][i].set_xticks([0,1])
        ax[1][i].set_xticklabels(['easy', 'hard'])
        ax[1][i].set_xlabel('rotation condition')
        
        v1 = delta[ind50]
        v2 = delta[ind90]
        
        ax[1][i].scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2,  s=40, zorder=10)
        (_, caps, _) = ax[1][i].errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
            
        t,p,equal_var,star = compute_two_sample_ttest(delta[ind50], delta[ind90])
        ax[1][i].plot([0,1],[3.3,3.3],color='k')
        ax[1][i].text(0.5, 3.35, f'{star}', ha='center', va='bottom', fontstyle='italic')
        
        ax[1][i].set_ylim([-3.5,3.75])
        ax[1][i].set_yticks([-3,-2,-1,0,1,2,3])
        
        ax[1][i].axhline(0, color='grey', ls='--', zorder=0, lw=1)
        
        if i == 1:
            ax[0][1].set_yticklabels([])
            ax[1][1].set_yticklabels([])
        
                
   
ax[0][0].set_ylabel(f'$d_{{shared}}$', fontsize=fontsize+2)

fig.subplots_adjust(hspace=0.35,wspace=0.1)#, bottom=0.1)


#%%


"""

nf stats

""" 


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]

        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        nfBL_95 = np.zeros(len(dDates[i]))
        nfPE_95 = np.zeros(len(dDates[i]))
        
        
        delta = np.zeros(len(dDates[i]))
        

        for d, date in enumerate(dDates[i]):
            
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_VAR_FA6, fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            nfBL_95[d] = dVAR['BL']['dshared95']
            nfPE_95[d] = dVAR['PE']['dshared95']
            
            nf1 = dVAR['BL']['dshared95']
            nf2 = dVAR['PE']['dshared95']
            
            delta[d] = nf2 - nf1 #100*((nf2-nf1)/nf1)
            

        
        v1 = delta[ind50]
        v2 = delta[ind90]
        
        # # t,p,equal_var,star = compute_two_sample_ttest(delta[ind50], delta[ind90])
        # n1=len(v1)
        # n2=len(v2)
        # # dof = n1+n2-2
        # # print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}, equal_var={equal_var}')
        
        # t,p,star = compute_one_sample_ttest(delta[ind50], 0)
        # dof = n1-1
        # print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}')
        
        # t,p,star = compute_one_sample_ttest(delta[ind90], 0)
        # dof = n2-1
        # print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}')
        







#%%

"""

STATS - number of factors identified

    2x2-way ANOVA

"""

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


mode = 'rotation'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

#TODO: double check that model was fit using dshared95 (FA6)

for i in [0,1]:
    
    subj, subject = dSubject[i]
        
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]
    
    nfALL = {'BL':np.zeros(len(dDates[i])),
             'PE':np.zeros(len(dDates[i]))}

    for d, date in enumerate(dDates[i]):
        fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_path_VAR_FA6, fn), "rb")
        _, dVAR = pickle.load(open_file)
        open_file.close()
        
        nfALL['BL'][d] = dVAR['BL']['dshared95']
        nfALL['PE'][d] = dVAR['PE']['dshared95']
        
        # nf1 = dVAR['BL']['dshared95']
        # nf2 = dVAR['PE']['dshared95']
        
        #delta[d] = nf2 - nf1 #100*((nf2-nf1)/nf1)
    
    
    # NF  = []
    # DEG = []
    # BLOCK = []
    
    # for ind_DEG, deg_LAB in zip([ind50, ind90],['50', '90']):
    
    #     for block_LAB in ['BL', 'PE']:
        
    #             NF.append(nfALL[block_LAB][ind_DEG])
    #             DEG.append([deg_LAB]*len(ind_DEG))
    #             BLOCK.append([block_LAB]*len(ind_DEG))
            
            
    # NF  = np.concatenate((NF))
    # DEG = np.concatenate((DEG))
    # BLOCK   = np.concatenate((BLOCK))        
    
    # dfANOVA = pd.DataFrame({'NF': NF, 'deg': DEG, 'block': BLOCK})
    
    # print(i, len(dfANOVA), len(dfANOVA)/3)
    
    # formula = 'NF ~ C(deg) + C(block) + C(deg):C(block)'
    # model = ols(formula, data=dfANOVA).fit()
    
    # # Perform ANOVA and print the table
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

   

#%%


"""

Loading Similarity...

"""

import warnings
warnings.filterwarnings("ignore")

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,3))         
            
for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left', fontsize=fontsize)
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
    

        sim1 = np.zeros(len(dDates[i]))
        sim2 = np.zeros(len(dDates[i]))

        for d, date in enumerate(dDates[i]):
            
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path_VAR_FA6, fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            
            # UA = dVAR['BL']['loadings'].T
            # cov_shared_A = np.dot(UA,UA.T)
            # rankA = np.linalg.matrix_rank(cov_shared_A)
            # _,aS,_ = np.linalg.svd(cov_shared_A)
            

            
            # UB = dVAR['PE']['loadings'].T
            # cov_shared_B = np.dot(UB,UB.T)
            # rankB = np.linalg.matrix_rank(cov_shared_B)
            # _,bS,_ = np.linalg.svd(cov_shared_B)
            
            lsBL = np.mean(dVAR['BL']['loading_sim'][:3])
            lsPE = np.mean(dVAR['PE']['loading_sim'][:3])
            sim1[d] = lsBL#/lsBL #100*((lsPE-lsBL)/lsBL) #np.log10() #lsPE/lsBL #np.log10(lsPE/lsBL) #100*((lsPE-lsBL)/lsBL)
            sim2[d] = lsPE#/lsBL #100*((lsPE-lsBL)/lsBL) #np.log10() #lsPE/lsBL #np.log10(lsPE/lsBL) #100*((lsPE-lsBL)/lsBL)

        
  
        axis = ax[i]
        x = ['BL/50']*len(ind50)  + ['PE/50']*len(ind50) +  ['BL/90']*len(ind90) + ['PE/90']*len(ind90)
        
        v1 = sim1[ind50]
        v2 = sim1[ind90]
        
        v3 = sim2[ind50]
        v4 = sim2[ind90]
        
        VAR = np.concatenate((v1,v3,v2,v4))
        sb.swarmplot(x=x, y=VAR, ax=axis)#, palette=palette_ROT, alpha=0.6, s=5)
        
        axis.axhline(0, color='grey', zorder=0, ls='--')
        

        # ind50 = np.where(dDegs[i]==50)[0]
        # ind90 = np.where(dDegs[i]==90)[0]
        
        # v1 = VAR[ind50]
        # v2 = VAR[ind90]
        
    #     print(np.mean(v1), np.mean(v2))
    
        
    #     # axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=50, zorder=10)
    #     # (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=6, zorder=9, ls='')
    #     # for cap in caps:
    #     #     cap.set_markeredgewidth(2.5)
            
    #     # t,p,equal_var,star = compute_two_sample_ttest(v1,v2,trending=False)
    #     # axis.plot([0,1],[0,0], color='k')
    #     # axis.text(0.5,0,f'{star}', ha='center', va='bottom', fontstyle='italic')
        
    #     # t,p,star = compute_one_sample_ttest(v1,1,trending=False)
    #     # axis.text(-0.5,0,f'{star}', va='top', ha='center', fontstyle='italic')
        
    #     # t,p,star = compute_one_sample_ttest(v2,1,trending=False)
    #     # axis.text(1.5,0,f'{star}', va='top', ha='center', fontstyle='italic')

            
            
    #     axis.set_ylim([-0.1,0.15])
    
    # axis.set_xticks([0,1])
    # axis.set_xticklabels(['easy', 'hard'])
    # axis.set_xlabel('rotation condition')
 

# ax[0].set_ylabel('$log_{10}$(loading similarity ratio)')                   
# ax[0][0].set_ylabel('factor 1 loading similarity\n(log ratio)')
# ax[1][0].set_ylabel('factor 2 loading similarity\n(log ratio)')
# ax[2][0].set_ylabel('factor 3 loading similarity\n(log ratio)')

fig.subplots_adjust(hspace=0.1, wspace=0.1)

                
            
#%%

# UA = dVAR['BL']['loadings'].T
# cov_shared_A = np.dot(UA,UA.T)
# rankA = np.linalg.matrix_rank(cov_shared_A)

# U,S,VT = np.linalg.svd(cov_shared_A)
# cS = np.sum(S[:3])/np.sum(S[:rankA])

# print(cS)




        

#%%

"""

...pairwise metrics....


"""


# for mode in ['rotation']:
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    
#     for i in [0]:#,1]:
        
#         subj, subject = dSubject[i]
#         #ax[0][i].set_title(f'{subject}', loc='left')
        
#         ind50 = np.where(dDegs[i]==50)[0]
#         ind90 = np.where(dDegs[i]==90)[0]
        
#         for d in [3]:    
#         #for d, date in enumerate(dDates[i]):
            
#             date = dDates[i][d]
            
            
#             fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
#             open_file = open(os.path.join(pickle_path_VAR_save, fn), "rb")
#             _, dVAR_90 = pickle.load(open_file)
#             open_file.close()
            
            
#             fig, ax = plt.subplots(figsize=(6,6))
            
#             rsc_mean = dVAR_90['BL']['rsc_mean']
#             rsc_sd  = dVAR_90['BL']['rsc_sd']
            
#             plt.scatter(rsc_mean, rsc_sd, color='k')
            
            
#             rsc_mean = dVAR_90['PE']['rsc_mean']
#             rsc_sd  = dVAR_90['PE']['rsc_sd']
            
#             plt.scatter(rsc_mean, rsc_sd, color='b')
            
            
#             # ax.set_ylim([0,5])
#             # ax.set_xlim([0,5])
            
            
#             for l in range(1,6):
#                 center = (0, 0)
#                 width = 2*l
#                 height = 2*l
#                 angle = 0
#                 theta1 = 0
#                 theta2 = 90
                
    
#                 arc = patches.Arc(center, width, height, angle=angle, theta1=theta1, theta2=theta2, color='grey', ls='--', linewidth=1)
#                 ax.add_patch(arc)
                        
#             print('dim:', np.sign( dVAR_90['BL']['dshared95'] - dVAR_90['PE']['dshared95']) )
#             print('ls:', np.sign( dVAR_90['BL']['loading_sim'][0] - dVAR_90['PE']['loading_sim'][0] ) )
#             print('sv:',np.sign( np.mean(dVAR_90['BL']['%sv']) - np.mean(dVAR_90['PE']['%sv']) ))
#             # print()
#             # print()
#             # print()
            
            

# #%%

# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches

# # # Create a figure and axes object
# # fig, ax = plt.subplots(1)



# # # Set the limits of the plot
# # ax.set_xlim([0, 6])
# # ax.set_ylim([0, 6])
# # ax.set_aspect('equal', adjustable='box')

# # # Add labels and title
# # ax.set_xlabel('x-axis')
# # ax.set_ylabel('y-axis')
# # ax.set_title('Arc Example')

# # # Show the plot
# # plt.show()


# #%%# for mode in ['rotation']:#, 'shuffle']:
    
# #     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# #     dDates, dDegs, dDegs2 = pickle.load(open_file)
# #     open_file.close()
    

    
# #     for i in [0,1]:#,1]:
        
# #         TEMP1 = np.zeros(len(dDates[i]))
# #         TEMP2 = np.zeros(len(dDates[i]))

# #         subj, subject = dSubject[i]
        
# #         for d, date in enumerate(dDates[i][:]): #enumerate(tqdm(dDates[i])): #enumerate(dDates[i]): #: 

            
# #             date = dDates[i][d]
            
# #             fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
# #             open_file = open(os.path.join(pickle_path_SC, fn), "rb")
# #             _, dSC = pickle.load(open_file)
# #             open_file.close()
            
# #             fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
# #             open_file = open(os.path.join(pickle_path_VAR_save, fn), "rb")
# #             _, dVAR_90 = pickle.load(open_file)
# #             open_file.close()
            
# #             bins = dVAR_90['bins']
        
            
# #             dALL = np.zeros(50)
# #             dEST = np.zeros(50)
# #             for TEST in range(50):
            
# #                 mBL = dVAR_90['BL']['mFR_est'][TEST,:]
# #                 mPE = dVAR_90['PE']['mFR_est'][TEST,:]
                
# #                 if np.shape(mBL)[0] != np.shape(mPE)[0]:
# #                     print(TEST, 'ERROR', np.shape(mBL), np.shape(mPE))
                
                
                
# #                 mBL_sv = dVAR_90['BL']['shared_est'][TEST]
# #                 mPE_sv = dVAR_90['PE']['shared_est'][TEST]
        
        
# #                 delta_EST = 100* ((mPE_sv - mBL_sv)/ mBL_sv)
# #                 dEST[TEST] = delta_EST
    
                
# #                 delta = []
# #                 for j in range(len(bins)-1):
                    
# #                     bin_j = bins[j]
# #                     bin_j1 = bins[j+1]
                    
# #                     indsBL = np.where( (mBL > bin_j) & (mBL <= bin_j1))[0]
# #                     indsPE = np.where( (mPE > bin_j) & (mPE <= bin_j1))[0]
                    
# #                     if len(indsBL) != len(indsPE):
# #                         print(d, "ERROR", len(indsBL), len(indsPE))
                        
# #                     else:
# #                         if len(indsBL) != 0:
# #                             m1 = np.mean(mBL[indsBL])
# #                             m2 = np.mean(mPE[indsPE])
                            
# #                             delta_ = 100*((m2-m1)/m1)
                            
# #                             delta.append(delta_)
                            
                            
# #                 dALL[TEST] = np.mean(delta)
                        
            
# #             TEMP1[d] = np.mean(dALL)
# #             TEMP2[d] = np.mean(dEST)

            

# #         ind50 = np.where(dDegs[i]==50)[0]
# #         ind90 = np.where(dDegs[i]==90)[0]
        
# #         v1 = TEMP1[ind50]
# #         v2 = TEMP2[ind50]
        
# #         print(compute_corr(v1,v2))
        
# #         v1 = TEMP1[ind90]
# #         v2 = TEMP2[ind90]
        
# #         print(compute_corr(v1,v2))

# #%%
# """
# SHUFFLE
# """

# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(5,10))         
# fig.subplots_adjust(hspace=0.1, wspace=0.1)

# for mode in ['shuffle']:
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    
    
#     for i in [0,1]:
        

#         subj, subject = dSubject[i]
#         ax[0][i].set_title(f'{subject}', loc='left')
        

        
#         VAR_ = dRES[i]['delta']
        
#         for j, VAR_KEY in zip([0,1,2], ['tv', 'sv', 'pv']):
 
#             VAR = VAR_[VAR_KEY]
            
#             ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
#             sb.swarmplot(x=np.zeros(len(VAR)), y=VAR, ax=ax[j][i])#, palette=palette_ROT, alpha=0.7, edgecolor='k', lw=3)
#             #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[j][i], jitter=0.1, palette=palette_ROT)
            
#             print(VAR_KEY, compute_one_sample_ttest(VAR,0))

                        
            
#%%  

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(5,10))         
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    
    for i in [0,1]:
        

        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        VAR_ = dRES[i]['delta']
        
        for j, VAR_KEY in zip([0,1,2], ['tv', 'sv', 'pv']):
 
            VAR = VAR_[VAR_KEY]
            
            ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
            sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[j][i], palette=palette_ROT, alpha=0.7, edgecolor='k', lw=3)
            #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[j][i], jitter=0.1, palette=palette_ROT)
                        
            v1 = VAR[ind50]
            v2 = VAR[ind90]
            ax[j][i].scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=40, zorder=10)
            (_, caps, _) = ax[j][i].errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=8, zorder=9, ls='')
            for cap in caps:
                cap.set_markeredgewidth(2.5)
            

                
            t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
            ax[j][i].plot([0,1],[35,35],color='k')
            
            if p < 0.05:
                fontsize_p = fontsize+5
                y_p = 35-3
                
            else:
                fontsize_p = fontsize+2
                y_p = 35
            
            ax[j][i].text(0.5,y_p,f'{star}',ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p)
            
        
            t,p,star = compute_one_sample_ttest(v1,0)
            if p < 0.05:
                y_p = -3
            else:
                y_p = 0.5
                
            if (j==2) and (i==0): 
                y_p = 3
                fontsize_p_v1 = fontsize+4
                
            ax[j][i].plot([0,0.4],[np.mean(v1), np.mean(v1)],  color='grey')#, ls='--')
            ax[j][i].plot([0.4,0.4],[0, np.mean(v1)],  color='grey')#, ls='--')
            ax[j][i].text(0.37,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)


            t,p,star = compute_one_sample_ttest(v2,0)
            if p < 0.05:
                y_p = -3
            else:
                y_p = 0.5
            
            if (j==2) and (i==0): 
                y_p = 4
            ax[j][i].plot([1,0.6],[np.mean(v2), np.mean(v2)],  color='grey')#, ls='--')
            ax[j][i].plot([0.6,0.6],[0, np.mean(v2)],  color='grey')#, ls='--')
            ax[j][i].text(0.62,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)

            ax[j][i].set_ylim([-50,50])
        
        
            
            ax[j][i].set_yticks([-40,-20,0,20,40])

        
        ax[2][i].set_xticks([0,1])
        ax[2][i].set_xticklabels(['easy', 'hard'])
        ax[2][i].set_xlabel('rotation condition')
                
            
ax[0][0].set_ylabel('Δ total variance (%)')
ax[1][0].set_ylabel('Δ shared variance (%)')          
ax[2][0].set_ylabel('Δ private variance (%)')    



ax[0][0].set_xlim([-0.4,1.4])   

for j in [0,1,2]:
    ax[j][0].set_yticklabels([-40,-20,'BL',20,40])
        







