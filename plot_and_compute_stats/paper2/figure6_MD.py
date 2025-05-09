# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:21:51 2025

@author: hanna
"""

#%%


"""

[Figure 6] changes in modulation depth

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

palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'}



dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}



root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')  
pickle_path_dTC = os.path.join(pickle_path, 'tuning') 
pickle_path_FA_shuffle = os.path.join(pickle_path, 'FA_tuning') 
              

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
fn = f'tuning_ns0_{mode}.pkl' #alt: fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()

mode = 'shuffle'
fn = f'tuning_ns0_{mode}.pkl' #alt: fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle, dPD_all_shuffle, dPD_each_shuffle = pickle.load(open_file)
open_file.close()



#%%

"""

##############################################################################

[Figure 6A] % of BCI neurons that are significant per session over sessions

##############################################################################

"""


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,6))


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[i].set_title(f'{subject}', loc='left')
    
    assigned_ROT = np.abs(np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]]))
    sig_ROT = np.array([np.sum(dPD_each[i][date]['sig'])/len(dPD_each[i][date]['sig']) for date in dDates[i]])
    sig_SHU = np.array([np.sum(dPD_each_shuffle[i][date]['sig'])/len(dPD_each_shuffle[i][date]['sig']) for date in dDates_shuffle[i]])


    assigned_both = np.concatenate(( assigned_ROT, 4*np.ones(len(dDates_shuffle[i])) ))
    sig_both = 100*np.concatenate(( sig_ROT, sig_SHU ))
    
    
    sb.swarmplot(x=assigned_both, y=sig_both, ax=ax[i], order=[50, 90, 4], palette=palette_ROT4)

    ax[i].set_ylabel('')
    ax[i].set_xticks([0,1,2])
    ax[i].set_xticklabels(['rotation,\neasy', 'rotation,\nhard', 'shuffle'], rotation=0) 
    ax[i].set_xlabel('applied perturbation')
    
    
    ind50_ccw = np.where(assigned_both == 50)[0]  #+/-
    ind90_ccw = np.where(assigned_both == 90)[0]  #+/-
    ind_shuffle = np.where(assigned_both == 4)[0]
    
    
    v3 = sig_both[ind50_ccw]
    v4 = sig_both[ind90_ccw]
    v5 = sig_both[ind_shuffle]
    

    
    ax[i].scatter([0,1,2], [ np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=70, zorder=10) #linewidth=2.5, s=75
    (_, caps, _) = ax[i].errorbar([0,1,2], [ np.mean(v3), np.mean(v4), np.mean(v5)], yerr=[ np.std(v3), np.std(v4), np.std(v5)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
    for cap in caps:
        cap.set_markeredgewidth(2.5)
        
    
    ax[i].axhline(0,color='grey',ls='--', zorder=0)
    
    
    
    
    dfANOVA = pd.DataFrame({'sig': sig_both,  'rot': assigned_both}) 
    
    
    model = ols('sig ~ C(rot) ', data=dfANOVA).fit() 
    result = sm.stats.anova_lm(model, type=1) 
      
    print(result) 
    
    tukey = pairwise_tukeyhsd(endog=dfANOVA['sig'],
                              groups=dfANOVA['rot'],
                              alpha=0.05)
    
    print(tukey)
        
        

ax[0].set_ylabel('% neurons with sig. measured ΔPD')


ax[0].set_ylim([-5,108])
ax[0].set_yticks(np.arange(0,101,20))


for i in [0,1]:

    ax[i].plot([0,1],[100,100],color='k', zorder=0)
    ax[i].scatter(0,100,marker='s', color=orange,s=25)
    ax[i].scatter(1,100,marker='s', color=purple,s=25)
    
    ax[i].plot([0,2],[103,103],color='k', zorder=0)
    ax[i].scatter(0,103,marker='s', color=orange,s=25)
    ax[i].scatter(2,103,marker='s', color='grey',s=25)
    
    ax[i].plot([1,2],[106,106],color='k', zorder=0)
    ax[i].scatter(1,106,marker='s', color=purple,s=25)
    ax[i].scatter(2,106,marker='s', color='grey',s=25)









#%%


"""

##############################################################################

[Figure 6B] average measured modulation depth: baseline block (top row), perturbation block (bottom row)

##############################################################################

"""



fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(4,6))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[0][i].set_title(f'{subject}', loc='left')
    
    assigned_ROT = np.abs(np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]]))
    
    
    MD_BL_ROT = np.array([np.mean(dPD_each[i][date]['MD_BL']) for date in dDates[i]])
    MD_PE_ROT = np.array([np.mean(dPD_each[i][date]['MD_PE']) for date in dDates[i]])

    MD_BL_SHU = np.array([np.mean(dPD_each_shuffle[i][date]['MD_BL']) for date in dDates_shuffle[i]])
    MD_PE_SHU = np.array([np.mean(dPD_each_shuffle[i][date]['MD_PE']) for date in dDates_shuffle[i]])


    assigned_both = np.concatenate(( assigned_ROT, 4*np.ones(len(dDates_shuffle[i])) ))
    MD_BL_BOTH = np.concatenate(( MD_BL_ROT, MD_BL_SHU ))
    MD_PE_BOTH = np.concatenate(( MD_PE_ROT, MD_PE_SHU ))
    
    
    sb.swarmplot(x=assigned_both, y=MD_BL_BOTH, ax=ax[0][i], order=[ 50, 90, 4], palette=palette_ROT4, s=4, alpha=0.7)
    sb.swarmplot(x=assigned_both, y=MD_PE_BOTH, ax=ax[1][i], order=[50, 90, 4], palette=palette_ROT4, s=4, alpha=0.7)


    ax[0][i].set_ylim([0,1])

    ax[1][i].set_xticks([0,1,2])
    ax[1][i].set_xticklabels(['rot.,\neasy', 'rot.,\nhard', 'shuffle'], rotation=0) 
    ax[1][i].set_xlabel('applied perturbation')
    
    
    for j, MD_BLOCK in zip([0,1], [MD_BL_BOTH, MD_PE_BOTH]):
    
        ind50_ccw = np.where(assigned_both == 50)[0]  #+/-
        ind90_ccw = np.where(assigned_both == 90)[0]  #+/-
        ind_shuffle = np.where(assigned_both == 4)[0]
        
        v3 = MD_BLOCK[ind50_ccw]
        v4 = MD_BLOCK[ind90_ccw]
        v5 = MD_BLOCK[ind_shuffle]

        
        ax[j][i].scatter([0,1,2], [ np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=40, zorder=10) #linewidth=2.5, s=75
        (_, caps, _) = ax[j][i].errorbar([0,1,2], [np.mean(v3), np.mean(v4),np.mean(v5)], yerr=[ np.std(v3), np.std(v4), np.std(v5)], color='k', lw=1.5, capsize=6, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
        

        dfANOVA = pd.DataFrame({'sig': MD_BL_BOTH,  'rot': assigned_both}) 
        
        
        # model = ols('sig ~ C(rot) ', data=dfANOVA).fit() 
        # result = sm.stats.anova_lm(model, type=1) 
          
        # print(result) 
        
        tukey = pairwise_tukeyhsd(endog=dfANOVA['sig'],
                                  groups=dfANOVA['rot'],
                                  alpha=0.05)
        
        print(tukey)
        

for i in [0,1]:
    for j in [0,1]:
        axis = ax[j][i]
        
        axis.plot([0,1],[0.85,0.85],color='grey', ls='-.', lw=0.75,zorder=0)
        axis.scatter(0,0.85,marker='s', color=orange,s=25)
        axis.scatter(1,0.85,marker='s', color=purple,s=25)
        
        axis.plot([0,2],[0.9,0.9],color='k', zorder=0)
        axis.scatter(0,0.9,marker='s', color=orange,s=25)
        axis.scatter(2,0.9,marker='s', color='grey',s=25)
        
        axis.plot([1,2],[0.95,0.95],color='k', zorder=0)
        axis.scatter(1,0.95,marker='s', color=purple,s=25)
        axis.scatter(2,0.95,marker='s', color='grey',s=25)





ax[0][0].set_ylabel('average MD (baseline)')
ax[1][0].set_ylabel('average MD (perturbation)')



#%%



"""

##############################################################################

[Figure 6C] changes in measured modulation depth - ROTATION

##############################################################################

"""
fontsize = 12


fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6,6))


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[0][i].set_title(f'{subject}', loc='left')

    assigned_ROT = np.abs(np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]]))
    

    dMD_50_sig = []
    dMD_90_sig = []
    
    dMD_50_ns = []
    dMD_90_ns = []
    for d,date in enumerate(dDates[i]):
        
        sig = dPD_each[i][date]['sig']

        MD_BL_ = dPD_each[i][date]['MD_BL']
        MD_PE_ = dPD_each[i][date]['MD_PE']
        
        ratio = np.log10(MD_BL_/MD_PE_) 
        
       
        dfTEMP = pd.DataFrame({'sig': sig, 'ratio': ratio})
        
        if assigned_ROT[d] == 50:
            dMD_50_sig.append(dfTEMP.loc[(dfTEMP['sig']==1), 'ratio'].values)
            dMD_50_ns.append(dfTEMP.loc[(dfTEMP['sig']==0), 'ratio'].values)
            
        elif assigned_ROT[d] == 90:
            dMD_90_sig.append(dfTEMP.loc[(dfTEMP['sig']==1), 'ratio'].values)
            dMD_90_ns.append(dfTEMP.loc[(dfTEMP['sig']==0), 'ratio'].values)
    
        else:
           print("ERROR")



    dMD_F_sig = np.concatenate((dMD_50_sig))
    dMD_T_sig = np.concatenate((dMD_90_sig))
    
    dMD_F_ns = np.concatenate((dMD_50_ns))
    dMD_T_ns = np.concatenate((dMD_90_ns))
    
    
    # print(i, 50, len(dMD_F_sig), len(dMD_F_ns))
    # print(i, 90, len(dMD_T_sig), len(dMD_T_ns))
      
     
    bins = np.linspace(-1,1,30)
    ax[0][i].hist(dMD_F_sig, bins=bins, edgecolor='k', color=orange, density=True, alpha=1)
    ax[0][i].hist(dMD_T_sig, bins=bins, edgecolor='k', color=purple, density=True, alpha=0.8)

    ax[0][i].set_ylim([0,6])
    ax[0][i].set_yticks(np.arange(0,6.1,1))
    ax[0][i].set_yticklabels(np.arange(0,6.1,1).astype(int), fontsize=fontsize-1)
    ax[0][i].set_xticks(np.arange(-1,1.01,0.5))
    ax[0][i].set_xticklabels(np.arange(-1,1.01,0.5), fontsize=fontsize-1)
    
    ax[0][i].axvline(0, color='k', zorder=0, ls='-.', lw=1, ymin=0, ymax=5.4/6)
    ax[0][i].scatter(np.mean(dMD_T_sig), 5, marker='d', s=30, alpha=0.6, color=purple, zorder=1)
    ax[0][i].scatter(np.mean(dMD_F_sig), 5, marker='d', s=30, alpha=0.6, color=orange, zorder=2)

    ax[1][i].hist(dMD_F_ns, bins=bins, edgecolor='k', color=orange, density=True, alpha=1)
    ax[1][i].hist(dMD_T_ns, bins=bins, edgecolor='k', color=purple, density=True, alpha=0.8)

    ax[1][i].axvline(0, color='k', zorder=0, ls='-.', lw=1, ymin=0, ymax=5.4/6)
    ax[1][i].scatter(np.mean(dMD_T_ns), 5, marker='d', s=30,alpha=0.6, color=purple, zorder=1 )
    ax[1][i].scatter(np.mean(dMD_F_ns), 5, marker='d', s=30, alpha=0.6,color=orange, zorder=2)


    x = dMD_F_sig
    y = dMD_T_sig
    n1 = len(x)
    n2 = len(y)
    s1 = np.std(dMD_F_sig, ddof=1)  # Sample standard deviation (Bessel's correction)
    s2 = np.std(dMD_T_sig, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (np.mean(x) - np.mean(y)) / pooled_sd
    
    print(i, f'effect size: {cohens_d:.2f}')


    #print(10**np.mean(dMD_T_sig) - 10**np.mean(dMD_F_sig))
    
    # print(compute_one_sample_ttest(dMD_F_ns, 0))
    # print(compute_one_sample_ttest(dMD_T_ns, 0))
    
    
    # print(compute_two_sample_ttest(dMD_F_sig, dMD_T_sig))
    # print(compute_two_sample_ttest(dMD_F_ns, dMD_T_ns))
    
    
    ax[0][i].text(-0.95,5.75, 'sig. ΔPD only', fontsize=fontsize-3, fontstyle='italic')
    ax[1][i].text(-0.95,5.75, 'n.s. ΔPD only', fontsize=fontsize-3, fontstyle='italic')



ax[0][0].set_ylabel('density', fontsize=fontsize)
ax[1][0].set_ylabel('density', fontsize=fontsize)
ax[1][0].set_xlabel('log$_{10}$(MD$_{PE}$/MD$_{BL}$)', fontsize=fontsize-1)
ax[1][1].set_xlabel('log$_{10}$(MD$_{PE}$/MD$_{BL}$)', fontsize=fontsize-1)



legend_elements = [Line2D([0],[0], marker='s', markersize=8, color=orange, lw=0, markeredgecolor='k', label='easy'),
                   Line2D([0],[0], marker='s', markersize=8, color=purple, lw=0, markeredgecolor='k', label='hard')]

ax[0][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc=(0.65,0.85), fontsize=fontsize-2)

fig.subplots_adjust(wspace=0.15, hspace=0.1)



ax[0][0].text(0.05/2, 5.2, '***', ha='center', fontstyle='italic', fontsize=fontsize)
ax[0][1].text(0.05/2, 5.2, '***', ha='center', fontstyle='italic', fontsize=fontsize)


ax[1][0].text(0.05, 5.25, 'n.s.', ha='center', fontstyle='italic', fontsize=fontsize-2)
ax[1][1].text(0.05, 5.25, 'n.s.', ha='center', fontstyle='italic', fontsize=fontsize-2)





#%%

"""

##############################################################################

[Figure 6D] changes in measured modulation depth - SHUFFLE (shuffled vs non-shuffled)

##############################################################################

"""

fontsize = 12

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6,6))


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[0][i].set_title(f'{subject}', loc='left', fontsize=fontsize)


    dMD_T_sig = []
    dMD_F_sig = []
    
    dMD_T_ns = []
    dMD_F_ns = []
    for date in dDates_shuffle[i]:
        
       '---dfKG---'
       fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
       open_file = open(os.path.join(pickle_path_FA_shuffle, 'FA1', fn), "rb")
       delU, dNU, dParams1, _, _ = pickle.load(open_file)
       open_file.close()        
        
       open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
       dfKG = pickle.load(open_file)
       open_file.close()
       
       sig = dPD_each_shuffle[i][date]['sig']

       
        
       shuff = np.delete(dfKG['shuffled'].values, delU)
       indT = np.where(shuff==True)[0]
       indF = np.where(shuff==False)[0]
       
       MD_BL_ = dPD_each_shuffle[i][date]['MD_BL']#[indT]
       MD_PE_ = dPD_each_shuffle[i][date]['MD_PE']#[indT]
       
       ratio = np.log10(MD_PE_/MD_BL_)
       
       dfTEMP = pd.DataFrame({'sig': sig, 'shuff': shuff, 'ratio': ratio})
        
       dMD_T_sig.append(dfTEMP.loc[(dfTEMP['sig']==1)&(dfTEMP['shuff']==True), 'ratio'].values)
       dMD_F_sig.append(dfTEMP.loc[(dfTEMP['sig']==1)&(dfTEMP['shuff']==False), 'ratio'].values)
       
       dMD_T_ns.append(dfTEMP.loc[(dfTEMP['sig']==0)&(dfTEMP['shuff']==True), 'ratio'].values)
       dMD_F_ns.append(dfTEMP.loc[(dfTEMP['sig']==0)&(dfTEMP['shuff']==False), 'ratio'].values)

       
      
    dMD_T_sig = np.concatenate((dMD_T_sig))
    dMD_F_sig = np.concatenate((dMD_F_sig))
    
    dMD_T_ns = np.concatenate((dMD_T_ns))
    dMD_F_ns = np.concatenate((dMD_F_ns))
      
     
    bins = np.linspace(-1,1,30)
    ax[0][i].hist(dMD_F_sig, bins=bins, edgecolor='k', color='k', density=True, alpha=1)
    ax[0][i].hist(dMD_T_sig, bins=bins, edgecolor='k', color=magenta, density=True, alpha=0.8)

    ax[0][i].set_yticks(np.arange(0,8.1,1))
    ax[0][i].set_yticklabels(np.arange(0,8.1,1).astype(int), fontsize=fontsize-1)
    ax[0][i].set_xticks(np.arange(-1,1.01,0.5))
    ax[0][i].set_xticklabels(np.arange(-1,1.01,0.5), fontsize=fontsize-1)
    
    ax[0][i].axvline(0, color='k', zorder=0, ls='-.', lw=1)
    ax[0][i].scatter(np.mean(dMD_T_sig), 6, marker='D', s=30, color=magenta, edgecolor=magenta, zorder=1 )
    ax[0][i].scatter(np.mean(dMD_F_sig), 6, marker='o', s=10, color='k', zorder=2)




    ax[1][i].hist(dMD_F_ns, bins=bins, edgecolor='k', color='k', density=True, alpha=1)
    ax[1][i].hist(dMD_T_ns, bins=bins, edgecolor='k', color=magenta, density=True, alpha=0.8)

    ax[1][i].axvline(0, color='k', zorder=0, ls='-.', lw=1)
    ax[1][i].scatter(np.mean(dMD_T_ns), 6, marker='D', s=30, color=magenta, edgecolor=magenta, zorder=1 )
    ax[1][i].scatter(np.mean(dMD_F_ns), 6, marker='o', s=10, color='k', zorder=2)


    
    # print(compute_two_sample_ttest(dMD_F_sig, dMD_T_sig))
    # print(compute_two_sample_ttest(dMD_F_ns, dMD_T_ns))
    
    # print(compute_one_sample_ttest(dMD_F_sig, 0))
    # print(compute_one_sample_ttest(dMD_F_ns, 0))
    
    # print(compute_one_sample_ttest(dMD_T_sig, 0))
    # print(compute_one_sample_ttest(dMD_T_ns, 0))
    
    
    ax[0][i].text(-0.95,6, 'sig. ΔPD only', fontsize=fontsize-3, fontstyle='italic')
    ax[1][i].text(-0.95,6, 'n.s. ΔPD only', fontsize=fontsize-3, fontstyle='italic')

    ax[0][i].text(0.1, 5.75, 'n.s.', fontsize=fontsize-3, fontstyle='italic')
    ax[1][i].text(0.1, 5.75, 'n.s.', fontsize=fontsize-3, fontstyle='italic')


ax[0][0].set_ylabel('density', fontsize=fontsize)
ax[1][0].set_ylabel('density', fontsize=fontsize)
ax[1][0].set_xlabel('log$_{10}$(MD$_{PE}$/MD$_{BL}$)', fontsize=fontsize-1)
ax[1][1].set_xlabel('log$_{10}$(MD$_{PE}$/MD$_{BL}$)', fontsize=fontsize-1)



legend_elements = [Line2D([0],[0], marker='s', markersize=8, color='k', lw=0, markeredgecolor='k', label='non-shuffled'),
                   Line2D([0],[0], marker='s', markersize=8, color=magenta, lw=0, markeredgecolor='k', label='shuffled')]

ax[0][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc=(0.56,0.65), fontsize=fontsize-2)

fig.subplots_adjust(wspace=0.15, hspace=0.1)



