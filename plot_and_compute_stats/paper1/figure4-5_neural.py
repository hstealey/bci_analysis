# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:24:20 2025

@author: hanna
"""

"""

    Figure 4: Blockwise Changes in Neural Activity

    Figure 5: shared space alignment (SSA)
              

    
    Additional stats:
        -number of factors (dshared)
        -number of excluded, included BCI neurons
        -population firing rate changes (baseline, perturbation)


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
#blue    = [100/255, 143/255, 255/255]
#yellow  = [255/255, 176/255, 0/255]
purple  = [120/255, 94/255, 240/255]
orange  = [254/255, 97/255, 0/255]
#magenta = [220/255, 38/255, 127/255]


palette_NEU = {50: orange, 90: purple}


dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}


root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')
pickle_root_save_path = os.path.join(pickle_path, 'FA')

#n_sets_dir = 14
#pickle_root_save_path = os.path.join(pickle_path, 'SUPPLEMENTARY', 'pickles', 'FA', f'test_{n_sets_dir}_sets')

'------------------------------'
'Loading Custom Functions'
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
# os.chdir(os.path.join(custom_functions_path, 'neural_fxns'))
# from factor_analysis_fxns import fit_fa_model
os.chdir(os.path.join(custom_functions_path, 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star







#%%


"""

[Figure 4] Blockwise Changes in Neural Activity
        
        shared variance  (sv) - "main shared", no mean-matching
        private variance (pv) - normalzied by mean firing rate
        total variance   (tv) - sv + pv
        
"""


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dRES = {}
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]

        dRES[i] = {'BL':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))},
                   'PE':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))},
                   'delta':{'tv': np.zeros(len(dDates[i])),
                         'sv': np.zeros(len(dDates[i])),
                         'pv': np.zeros(len(dDates[i]))}}

        
        for d, date in enumerate(tqdm(dDates[i])): 
            
            fn = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA6', fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()

     
            fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
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
            sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[j][i], palette=palette_NEU, alpha=0.8, edgecolor='k', lw=3)
            #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[j][i], jitter=0.1, palette=palette_NEU)
                        
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

""" print STATS only - [Figure 4] Blockwise Changes in Neural Activity """


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




#%%

"""

[Figure 5] shared space alignment (SSA)

PE->BL

"""

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(4,8))

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    open_file = open(os.path.join(pickle_root_save_path, 'FA', f'dSSA_{mode}.pkl'), "rb")
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
    
    

# #%%

             
            
# #%%  

# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(5,10))         
# fig.subplots_adjust(hspace=0.1, wspace=0.1)

# for mode in ['rotation']:
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    
    
#     for i in [0,1]:
        

#         subj, subject = dSubject[i]
#         ax[0][i].set_title(f'{subject}', loc='left')
        
#         ind50 = np.where(dDegs[i]==50)[0]
#         ind90 = np.where(dDegs[i]==90)[0]
        
#         VAR_ = dRES[i]['delta']
        
#         for j, VAR_KEY in zip([0,1,2], ['tv', 'sv', 'pv']):
 
#             VAR = VAR_[VAR_KEY]
            
#             ax[j][i].axhline(0, color='grey', zorder=0, ls='--')
#             sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[j][i], palette=palette_NEU, alpha=0.7, edgecolor='k', lw=3)
#             #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[j][i], jitter=0.1, palette=palette_NEU)
                        
#             v1 = VAR[ind50]
#             v2 = VAR[ind90]
#             ax[j][i].scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=40, zorder=10)
#             (_, caps, _) = ax[j][i].errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=8, zorder=9, ls='')
#             for cap in caps:
#                 cap.set_markeredgewidth(2.5)
            

                
#             t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
#             ax[j][i].plot([0,1],[35,35],color='k')
            
#             if p < 0.05:
#                 fontsize_p = fontsize+5
#                 y_p = 35-3
                
#             else:
#                 fontsize_p = fontsize+2
#                 y_p = 35
            
#             ax[j][i].text(0.5,y_p,f'{star}',ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p)
            
        
#             t,p,star = compute_one_sample_ttest(v1,0)
#             if p < 0.05:
#                 y_p = -3
#             else:
#                 y_p = 0.5
                
#             if (j==2) and (i==0): 
#                 y_p = 3
#                 fontsize_p_v1 = fontsize+4
                
#             ax[j][i].plot([0,0.4],[np.mean(v1), np.mean(v1)],  color='grey')#, ls='--')
#             ax[j][i].plot([0.4,0.4],[0, np.mean(v1)],  color='grey')#, ls='--')
#             ax[j][i].text(0.37,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)


#             t,p,star = compute_one_sample_ttest(v2,0)
#             if p < 0.05:
#                 y_p = -3
#             else:
#                 y_p = 0.5
            
#             if (j==2) and (i==0): 
#                 y_p = 4
#             ax[j][i].plot([1,0.6],[np.mean(v2), np.mean(v2)],  color='grey')#, ls='--')
#             ax[j][i].plot([0.6,0.6],[0, np.mean(v2)],  color='grey')#, ls='--')
#             ax[j][i].text(0.62,y_p,f'{star}', ha='center', va='bottom', fontstyle='italic', fontsize=fontsize_p-4, rotation=0)

#             ax[j][i].set_ylim([-50,50])
        
        
            
#             ax[j][i].set_yticks([-40,-20,0,20,40])

        
#         ax[2][i].set_xticks([0,1])
#         ax[2][i].set_xticklabels(['easy', 'hard'])
#         ax[2][i].set_xlabel('rotation condition')
                
            
# ax[0][0].set_ylabel('Δ total variance (%)')
# ax[1][0].set_ylabel('Δ shared variance (%)')          
# ax[2][0].set_ylabel('Δ private variance (%)')    



# ax[0][0].set_xlim([-0.4,1.4])   

# for j in [0,1,2]:
#     ax[j][0].set_yticklabels([-40,-20,'BL',20,40])
        



#%%



"""

ADDITIONAL STATS

"""


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
            open_file = open(os.path.join(pickle_root_save_path, 'FA1',  fn), "rb")
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

[Supplementary Figure] population firing rate changes


    Firing rate changes are not significant
        mean differences, equal variances (?), K-S test? (or difference between...)

        "paired" but do sorted???




            
            # mBL = np.sort( np.mean(dSC['BL'], axis=0) )
            # mPE = np.sort( np.mean(dSC['PE'], axis=0) )
            
            # # K, p = stats.ks_2samp(mBL, mPE) 
            # # star = stats_star(p)
            # # print(f'{K:.2f}', star)
            
            # dDELTA[mode][i]['dFR_all'][d]  = K#100*(np.mean(mPE)-np.mean(mBL))/np.mean(mBL)
            # dDELTA[mode][i]['dFR_each'][d] = np.mean(100*((mPE-mBL)/mBL))
            
            



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
                open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
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

dshared (number of factors to explain 95% of the shared variance)

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
            open_file = open(os.path.join(pickle_root_save_path, 'FA6', fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            nfBL_95[d] = dVAR['BL']['dshared95']
            nfPE_95[d] = dVAR['PE']['dshared95']
            
            nf1 = dVAR['BL']['dshared95']
            nf2 = dVAR['PE']['dshared95']
            
            delta[d] = nf2 - nf1 #100*((nf2-nf1)/nf1)
            
            color = palette_NEU[dDegs[i][d]]
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
        

        
        
        sb.swarmplot(x=dDegs[i], y=delta, ax=ax[1][i], palette=palette_NEU)
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
            open_file = open(os.path.join(pickle_root_save_path, 'FA6', fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            nfBL_95[d] = dVAR['BL']['dshared95']
            nfPE_95[d] = dVAR['PE']['dshared95']
            
            nf1 = dVAR['BL']['dshared95']
            nf2 = dVAR['PE']['dshared95']
            
            delta[d] = nf2 - nf1 #100*((nf2-nf1)/nf1)
            

        
        v1 = delta[ind50]
        v2 = delta[ind90]
        
        # t,p,equal_var,star = compute_two_sample_ttest(delta[ind50], delta[ind90])
        n1=len(v1)
        n2=len(v2)
        # dof = n1+n2-2
        # print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}, equal_var={equal_var}')
        
        t,p,star = compute_one_sample_ttest(delta[ind50], 0)
        dof = n1-1
        print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}')
        
        t,p,star = compute_one_sample_ttest(delta[ind90], 0)
        dof = n2-1
        print(f'{subject} | t{dof}={t:.2f}, p={p:.2f}')
        


#%%



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
            open_file = open(os.path.join(pickle_root_save_path, 'FA6', fn), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            nfBL_95[d] = dVAR['BL']['dshared95']
            nfPE_95[d] = dVAR['PE']['dshared95']
            
            # nf1 = dVAR['BL']['dshared95']
            # nf2 = dVAR['PE']['dshared95']
            
        
        nE_BL = nfBL_95[ind50]
        nE_PE = nfPE_95[ind50]
        
        nH_BL = nfBL_95[ind90]
        nH_PE = nfPE_95[ind90]
        
        print(f'{subject} - Easy | BL, {np.mean(nE_BL):.2f} ({np.std(nE_BL):.2f})')
        print(f'{subject} - Easy | PE, {np.mean(nE_PE):.2f} ({np.std(nE_PE):.2f})')
        print(f'{subject} - Hard | BL, {np.mean(nH_BL):.2f} ({np.std(nH_BL):.2f})')
        print(f'{subject} - Hard | PE, {np.mean(nH_PE):.2f} ({np.std(nH_PE):.2f})')
        
            
            

