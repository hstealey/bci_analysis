# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:42:50 2025

@author: hanna
"""



"""
Figure 3 - Behavior
    [3A] exponential (averaged by condition); fit exponential curve
    [3B] performance during first PE TSN, second PE TSN, and best performance (variable TSN) 
    [3C] "AoL"

        "rates"? (amount recovered from initial to second; does it make sense to approximate initial to best???)
    
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
palette_SHU = {True: magenta, False: 'grey'}


dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'           
            
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
os.chdir(os.path.join(custom_functions_path,'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star, compute_r_squared


#%%

def exp_model(t,a,b,c):
    return(a*np.exp(-b*t)+c)


#%%

pickle_path_BEH = os.path.join(pickle_path, 'dfBEH', 'compute_behavior_results')

BEH_KEY_LIST = ['time']#, 'dist', 'ME', 'MV']#, 'vel_at_AE', 'vel_max', 'AE', 'ME', 'MV']


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dT = {}
    
    for i in [0,1]:
        

        dT[i] = {'full': np.zeros((len(dDates[i]), 41)),
                'delta': np.zeros((len(dDates[i]))), 
                'init': np.zeros((len(dDates[i]))), 
                'init2': np.zeros((len(dDates[i]))),
                'best': np.zeros((len(dDates[i]))), 
                'aol': np.zeros((len(dDates[i]))),
                'TSN': np.zeros((len(dDates[i]))),
                'delta1': np.zeros((len(dDates[i]))),
                'delta2': np.zeros((len(dDates[i])))} 



        subj, subject = dSubject[i]

        for d in tqdm(range(len(dDates[i][:]))):
    
            
            date = dDates[i][d]
            
            'V2: Behavior'
            open_file = open(os.path.join(pickle_path_BEH, f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
            dfBEH_BL, dfBEH_PE = pickle.load(open_file)
            open_file.close()
            
            for BEH_KEY in BEH_KEY_LIST:
                
                PE_BEH = np.zeros((8,len(dfBEH_BL)//8))
                for j, deg in enumerate(np.arange(0,360,45)):
                    PE_BEH[j,:] = dfBEH_PE.loc[dfBEH_PE['deg'] == deg, BEH_KEY][:len(dfBEH_BL)//8]
                
                mBL = np.mean(dfBEH_BL[BEH_KEY])
                mPE_ = np.mean(PE_BEH)#, axis=0)
                
                
                dT[i]['delta'][d] = 100*((mPE_-mBL)/mBL)
                
                
                mPE = np.mean(PE_BEH, axis=0)
                deltaBEH = 100*((mPE-mBL)/mBL)
                
               
                init = deltaBEH[0]
                init2 = deltaBEH[1]
                best = np.min(deltaBEH)
                AOL  =  (init-best)/init
                best_TSN = np.where(deltaBEH == best)[0][0]
                
                dT[i]['full'][d,:] = deltaBEH
                dT[i]['init'][d] = init 
                dT[i]['init2'][d] = init2
                dT[i]['best'][d] = best
                dT[i]['aol'][d]  = AOL

                dT[i]['TSN'][d] = best_TSN
                dT[i]['delta1'][d] = init2-init
                dT[i]['delta2'][d] = (best-init)/best_TSN 
                
#%%

'"BEST" TSN'
for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]
    
    BST = dT[i]['TSN']
    
    BST_E = BST[ind50]
    BST_H = BST[ind90]

    print(f'{subject} - Easy, {np.mean(BST_E):.2f} ({np.std(BST_E):.2f})')
    print(f'{subject} - Hard, {np.mean(BST_H):.2f} ({np.std(BST_H):.2f})')

#%%

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]
    
    init = dT[i]['init']
    best = dT[i]['best']
    
    init50 = init[ind50]
    init90 = init[ind90]
    
    best50 = best[ind50]
    best90 = best[ind90]
    
    axis = ax[i]
    
    axis.scatter(init50, (init50-best50)/init50, color=yellow)
    axis.scatter(init90, (init90-best90)/init90, color=blue)
    
    
    
    
    



#%%

"""

STATS - support for combining rotation conditions

    2x2-way ANOVA

"""

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

path_trials = 'latePE_160only' 
path_window = f'fixed_window_2-6'
pickle_path_dTC = os.path.join(pickle_path, 'FA_tuning', 'TTT', path_trials, path_window) #previously: #pickle_path_dTC = os.path.join(pickle_path,'tuning', 'BLOCK', 'earlyPE_160only', 'fit') 

mode = 'rotation'


fn = f'tuning_ns0_{mode}.pkl' ## fn = f'tuning_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])

    for deg in [50,90]:
        
        ind_pos = np.where(assigned==deg)[0]
        ind_neg = np.where(assigned==-1*deg)[0]
        
        BEH  = []
        SIGN = []
        TP   = []
        
        for ind_LAB, ind_SIGN in zip(['POS', 'NEG'], [ind_pos, ind_neg]):
            
            for tp_KEY in ['init', 'init2', 'best']:
                
                BEH.append(dT[i][tp_KEY][ind_SIGN])
                SIGN.append([ind_LAB]*len(ind_SIGN))
                TP.append([tp_KEY]*len(ind_SIGN))
                
                
        BEH  = np.concatenate((BEH))
        SIGN = np.concatenate((SIGN))
        TP   = np.concatenate((TP))        
        
        dfANOVA = pd.DataFrame({'BEH': BEH, 'sign': SIGN, 'TP': TP})
        
        print(i, deg, len(dfANOVA), len(dfANOVA)/3)

        formula = 'BEH ~ C(sign) + C(TP) + C(sign):C(TP)'
        model = ols(formula, data=dfANOVA).fit()
        
        # Perform ANOVA and print the table
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)



                
                

#%%


"""

Exponential

"""


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,3.5))
fig.subplots_adjust(wspace=0.05, hspace=0.075)

xdata = np.arange(41)

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left', fontsize=fontsize+2)
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        for indDEG, color, marker, DEG in zip([ind50,ind90],[yellow,blue], ['^', 's'], ['50' ,'90']):
        

            m = np.mean(dT[i]['full'][indDEG,:], axis=0)
            s = stats.sem(dT[i]['full'][indDEG,:], axis=0)

            ax[i].errorbar(x=xdata, y=m, yerr=s, color=color, marker=marker, markersize=7, markeredgecolor='w', markeredgewidth=0.4)
            
            p0_a = m[0] - np.min(m)
            p0_b = 0.5
            p0_c = np.min(m)
            
            p0 = [p0_a, p0_b, p0_c]
            
            params, _ = curve_fit(exp_model, xdata, m, p0)
            y_est = exp_model(xdata, *params)
            ax[i].plot(xdata,y_est,color='k', ls='-',  lw=2, zorder=2)
           
            a,b,c = params
            print(f'{subject}, {DEG} || {a:.2f}*e(-{b:.2f}t) + {c:.2f}') #formula, R2, sig?
            
            R2 = compute_r_squared(m, y_est)
            print(f'\tR2: {R2:.2f}')

     
            #Linear approximation of 0-1: ax[i].plot([0,1],[m[0],m[1]], color='r', zorder=100)


        ax[i].axhline(0, color='grey', ls='--', lw=1, zorder=0)

        ax[i].set_xlabel('perturbation trial set', fontsize=fontsize+2)
        
        ax[i].set_yticks([0,20,40,60,80,100,120])
        
        ax[i].axvline(0,color='grey', ls='--', lw=1, zorder=0)
        ax[i].axvline(1,color='grey', ls='--', lw=1, zorder=0)
        
        ax[i].set_xticks([0,10-1,20-1,30-1,40-1])
        ax[i].set_xticklabels([r'${1^{st}}$', r'${10^{th}}$', '$20^{th}$', '$30^{th}$', '$40^{th}$'], rotation=0, fontstyle='italic', fontname='Arial', fontsize=fontsize) #\mathregular

ax[0].set_ylabel('performance\n(% Δ in trial time from BL)', fontsize=fontsize+2)
ax[0].set_yticklabels(['BL',20,40,60,80,100,120], fontsize=fontsize+1)
        

legend_elements = [Line2D([0],[0], marker='^', markersize=9, color=yellow, lw=0, label='easy (50°)'),
                    Line2D([0],[0], marker='s', markersize=8, color=blue, lw=0, label='hard (90°)')]
    


ax[1].legend(handles=legend_elements, ncol=1, fontsize=fontsize+1, columnspacing=0.3, handletextpad=0.0, loc=(0.59,0.8))#'upper right')#, title='rotation')



#%%


"""

"timepoints"

"""


fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(6,10))
fig.subplots_adjust(wspace=0.075, hspace=0.125)

xdata = np.arange(41)

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[i][0].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]

        for j,KEY,lab in zip([0,1,2],['init', 'init2', 'best'],['first', 'second', 'best']):
            VAR = dT[i][KEY]
            
            #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[i][j], jitter=0.1, palette=palette_ROT, alpha=0.8)
            
            sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[i][j], palette=palette_BEH, alpha=0.8)
            v1 = VAR[ind50]
            v2 = VAR[ind90]
            
            axis = ax[i][j]
            axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2,  s=50, zorder=10)
            (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=6, zorder=9, ls='')
            for cap in caps:
                cap.set_markeredgewidth(2.5)
                
            t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
            ax[i][j].plot([0,1],[225,225], lw=1, color='k')
            ax[i][j].text(0.5,225,f'{star}',ha='center',va='center',fontstyle='italic',fontsize=fontsize+4)
            
            # n1 = len(v1)
            # n2 = len(v2)
            # dof = n1+n2-2
            # print(f'{subject}| {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}, equal_var={equal_var}')
            
                
            
            if (i == 1) and (j == 2):
                t,p,star = compute_one_sample_ttest(v1,0)
                ax[i][j].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
                ax[i][j].plot([0.4,0.4],[0,np.mean(v1)],color='grey')
                ax[i][j].text(0.5,8,f'{star}',ha='right',va='center',fontstyle='italic',fontsize=fontsize,color='grey')
                
                n1 = len(v1)
                dof = n1-1
                #print(f'\t{subject}| easy, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
                
                    
                    
            else:
                t,p,star = compute_one_sample_ttest(v1,0)
                ax[i][j].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
                ax[i][j].plot([0.4,0.4],[0,np.mean(v1)],color='grey')
                ax[i][j].text(0.41,np.mean(v1),f'{star}',ha='right',va='center',fontstyle='italic',fontsize=fontsize+2,color='grey')
                
                n1 = len(v1)
                dof = n1-1
                #print(f'\t{subject}| easy, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
                
                    
                
            
            t,p,star = compute_one_sample_ttest(v2,0)
            ax[i][j].plot([0.6,1],[np.mean(v2), np.mean(v2)],color='grey')
            ax[i][j].plot([0.6,0.6],[0,np.mean(v2)],color='grey')
            ax[i][j].text(0.53,np.mean(v2),f'{star}',ha='left',va='center',fontstyle='italic',fontsize=fontsize+2, color='grey')
            n2 = len(v2)
            dof = n2-1
            print(f'\t{subject}| hard, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
            
                

        
            ax[i][j].axhline(0, color='grey', ls='--', zorder=0, lw=0.75)        
            if i == 1:
                ax[i][j].set_xticks([0,1])
                ax[i][j].set_xticklabels(['easy', 'hard'])
                ax[i][j].set_xlabel('rotation condition')
                
            
            ax[i][j].set_xlim([-0.5,1.5])
            
 
            ax[i][j].set_ylim([-25,260])
            ax[i][j].text(-0.4,250,f'{lab}', fontstyle='italic', ha='left',fontsize=fontsize+1)
            ax[i][j].set_yticks([0,50,100,150,200,250])
            ax[i][j].set_yticklabels(['BL', 50, 100, 150, 200, 250])

            if j!=0:
                ax[i][j].set_yticklabels([])
                
            
            # if j==2:
            #     print(f'{subject} | best')
            #     len1 = len(np.where(v1 < 0)[0])
            #     len2 = len(np.where(v2 < 0)[0])
                
            #     print(f'\t easy: {len1}, {len(v1)}, {100*(len1/len(v1)):.1f}')
            #     print(f'\t hard: {len2}, {len(v2)}, {100*(len2/len(v2)):.1f}')

        print('')

ax[0][0].set_ylabel('performance\n(%Δ in trial time from BL)', fontsize=fontsize+2)
ax[1][0].set_ylabel('performance\n(%Δ in trial time from BL)', fontsize=fontsize+2)
         


#%%



def exp_model(t,a,b,c):
    return(a*np.exp(-b*t)+c)


y_examp = exp_model(xdata, 78, 0.4, 35)


y_ex = [y_examp[0]]

for y in y_examp[1:]:
    y_ex.append(y+(0.6*np.random.randint(-20,5)))

y_ex = np.array(y_ex)
y_ex[1:] = y_ex[1:]-0
y_ex[25] = y_ex[25]-10

#%%

fontsize = 10

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25*2,2*2))

plt.plot(xdata, y_ex, color='grey', zorder=0, alpha=0.3)
plt.scatter(xdata, y_ex, color='grey', zorder=1, alpha=0.3, )

plt.scatter(0,y_ex[0], marker='D', color='k', zorder=10, s=30)
plt.scatter(1,y_ex[1], marker='v', color='k', zorder=10, s=50)
plt.scatter(25,y_ex[25], marker='*', color='k', zorder=10, s=75)

plt.ylim([0,125])
plt.xlim([-1,42])


plt.text(24.5, 110, 'amount of recovery', ha='center', fontstyle='italic', fontsize=fontsize+2)

plt.scatter(23,100, marker='D', color='k', zorder=10, s=25)
plt.plot([24.2,25-0.2], [100,100], color='grey')
plt.scatter(26,100, marker='*', color='k', zorder=10, s=75)
plt.plot([22,27], [95,95], color='grey')
plt.scatter(22+2.5,90, marker='D', color='k', zorder=10, s=25)


plt.text(0.757, y_ex[0]+0, '$performance_{first}$', ha='left', va='center', fontstyle='italic', fontsize=fontsize+4)
plt.text(1.75, y_ex[1]+0,'$performance_{second}$', ha='left', va='center', fontstyle='italic', fontsize=fontsize+4)
plt.text(25.75, y_ex[25]+0, '$performance_{best}$', ha='left', va='center', fontstyle='italic',fontsize=fontsize+4)

ax.tick_params(axis='x', length=3)
ax.tick_params(axis='y', length=3)

plt.ylabel('performance (%Δ in trial time from BL)', fontsize=fontsize)
plt.xlabel('perturbation trial set', fontsize=fontsize)

plt.yticks(np.arange(0,121,20), ['BL', 20, 40, 60, 80, 100,120], fontsize=fontsize)
plt.xticks([0,10-1,20-1,30-1,40-1], [r'${1^{st}}$', r'${10^{th}}$', '$20^{th}$', '$30^{th}$', '$40^{th}$'], rotation=0, fontstyle='italic', fontname='Arial', fontsize=fontsize) #\mathregular



#%%

"""

"AOL"

"""



fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(3,5.5))



for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        sb.swarmplot(x=dDegs[i], y=dT[i]['aol'], ax=ax[i], palette=palette_BEH)
        
        v1 = dT[i]['aol'][ind50]
        v2 = dT[i]['aol'][ind90]
        
        axis = ax[i]
        axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=75, zorder=10)
        (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
        
        ax[i].axhline(1, color='grey', zorder=0, ls='--')#, lw=0.75)
        
     
        y1=2.4
        y2=2.35
        
        y50=np.mean(v1)+0.05
        y90=np.mean(v2)-0.05

        
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        ax[i].plot([0,1],[y1,y1],color='k')
        ax[i].text(0.5,y2,f'{star}', ha='center', va='bottom', fontsize=fontsize+4)
        
        n1 = len(v1)
        n2 = len(v2)
        dof = n1+n2-2
        print(f'{subject}| t{dof}={t:.2f}, p={p:.1e}, {star}, equal_var={equal_var}')
        
        
        t,p,star = compute_one_sample_ttest(v1,1)
        ax[i].plot([0,0.5],[np.mean(v1), np.mean(v1)], color='grey')
        ax[i].plot([0.5,0.5],[1,np.mean(v1)], color='grey')
        ax[i].text(0.6,y50,f'{star}', ha='right',va='center',fontstyle='italic')
        print(f'\t{subject}| t{n1-1}={t:.2f}, p={p:.1e}, {star}')
        
        
        
        t,p,star = compute_one_sample_ttest(v2,1)
        ax[i].plot([0.65,1],[np.mean(v2), np.mean(v2)], color='grey')
        ax[i].plot([0.65,0.65],[1,np.mean(v2)], color='grey')
        ax[i].text(0.4,y90,f'{star}', ha='left',va='center',fontstyle='italic')
        print(f'\t{subject}| t{n2-1}={t:.2f}, p={p:.1e}, {star}')
        
        
        
        
    
        ax[i].set_ylim([0,2.7])
        ax[i].set_yticks(np.arange(0,2.6,0.25))

        # elif i == 1:
        #     ax[i].set_ylim([0,2])
        #     ax[i].set_yticks([0,0.5,1,1.5,2])
        
        
        
        ax[i].set_xlim([-0.5,1.5])  
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['easy', 'hard'])

fig.text(0.5,0,'rotation condition', ha='center')

fig.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.075)
ax[0].set_ylabel('amount of recovery',fontsize=fontsize+2)

    
        
        
#%%



"""

"TSN"

"""


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(7/3,10))
fig.subplots_adjust(wspace=0.1, hspace=0.15)


for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        TEST = dT[i]['TSN']#dT[i]['TSN']#(dT[i]['init2'] - dT[i]['init'])/dT[i]['init']
        
        sb.swarmplot(x=dDegs[i], y=TEST, ax=ax[i], palette=palette_BEH)
        
        v1 = TEST[ind50]
        v2 = TEST[ind90]
        
        axis = ax[i]
        axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=75, zorder=10)
        (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        n1 = len(v1)
        n2 = len(v2)
        dof = n1+n2-2
        print(f'{subject} | t{dof}={t:.2f}, p={p:.3f} ({star}), equal_var={equal_var}')
            
        
        # ax[i].axhline(1, color='grey', zorder=0, ls='--')#, lw=0.75)
        
        # if i==0:
        #     y1=2.4
        #     y2=2.35
            
        #     y50=np.mean(v1)+0.075
        #     y90=np.mean(v2)-0.075
        # else:
        #     y1=1.65
        #     y2=1.625
            
        #     y50=np.mean(v1)+0.05
        #     y90=np.mean(v2)-0.05
        
        # t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        # ax[i].plot([0,1],[y1,y1],color='k')
        # ax[i].text(0.5,y2,f'{star}', ha='center', va='bottom', fontsize=fontsize+4)
        
        # t,p,star = compute_one_sample_ttest(v1,1)
        # ax[i].plot([0,0.4],[np.mean(v1), np.mean(v1)], color='grey')
        # ax[i].plot([0.4,0.4],[1,np.mean(v1)], color='grey')
        # ax[i].text(0.44,y50,f'{star}', ha='right',va='center',fontstyle='italic')
        
        # t,p,star = compute_one_sample_ttest(v2,1)
        # ax[i].plot([0.6,1],[np.mean(v2), np.mean(v2)], color='grey')
        # ax[i].plot([0.6,0.6],[1,np.mean(v2)], color='grey')
        # ax[i].text(0.53,y90,f'{star}', ha='left',va='center',fontstyle='italic')
        
        # if i == 0:
        #     ax[i].set_ylim([0.3,2.7])
        #     ax[i].set_yticks([0.5,1,1.5,2,2.5])
        
        # elif i == 1:
        #     ax[i].set_ylim([0.5,1.75])
        #     #ax[i].set_yticks([0.5,1,1.5,2,2.5])
        
        
        
        
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['easy', 'hard'])
ax[1].set_xlabel('rotation condition')


# ax[0].set_ylabel('trial set of best',fontsize=fontsize+2)
# ax[1].set_ylabel('trial set of best',fontsize=fontsize+2)
    
#%%

"""

RATE

"""

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(6,10))
# fig.subplots_adjust(wspace=0.075, hspace=0.125)

# for mode in ['rotation']:
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()
    

#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
#         ax[i][0].set_title(f'{subject}', loc='left')
        
#         ind50 = np.where(dDegs[i]==50)[0]
#         ind90 = np.where(dDegs[i]==90)[0]

#         for j,KEY,lab in zip([0,1],['delta1', 'delta2'],['delta1', 'delt2']):
#             VAR = dT[i][KEY]
            
#             #sb.stripplot(x=dDegs[i], y=VAR, ax=ax[i][j], jitter=0.1, palette=palette_ROT, alpha=0.8)
            
#             sb.swarmplot(x=dDegs[i], y=VAR, ax=ax[i][j], palette=palette_BEH, alpha=0.8)
#             v1 = VAR[ind50]
#             v2 = VAR[ind90]
            
#             axis = ax[i][j]
#             axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2,  s=50, zorder=10)
#             (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2, capsize=6, zorder=9, ls='')
#             for cap in caps:
#                 cap.set_markeredgewidth(2.5)
                
#             t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
#             ax[i][j].plot([0,1],[225,225], lw=1, color='k')
#             ax[i][j].text(0.5,225,f'{star}',ha='center',va='center',fontstyle='italic',fontsize=fontsize+4)
            
#             # n1 = len(v1)
#             # n2 = len(v2)
#             # dof = n1+n2-2
#             # print(f'{subject}| {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}, equal_var={equal_var}')
            
                
            
#             if (i == 1) and (j == 2):
#                 t,p,star = compute_one_sample_ttest(v1,0)
#                 ax[i][j].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
#                 ax[i][j].plot([0.4,0.4],[0,np.mean(v1)],color='grey')
#                 ax[i][j].text(0.5,8,f'{star}',ha='right',va='center',fontstyle='italic',fontsize=fontsize,color='grey')
                
#                 n1 = len(v1)
#                 dof = n1-1
#                 #print(f'\t{subject}| easy, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
                
                    
                    
#             else:
#                 t,p,star = compute_one_sample_ttest(v1,0)
#                 ax[i][j].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
#                 ax[i][j].plot([0.4,0.4],[0,np.mean(v1)],color='grey')
#                 ax[i][j].text(0.41,np.mean(v1),f'{star}',ha='right',va='center',fontstyle='italic',fontsize=fontsize+2,color='grey')
                
#                 n1 = len(v1)
#                 dof = n1-1
#                 #print(f'\t{subject}| easy, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
                
                    
                
            
#             t,p,star = compute_one_sample_ttest(v2,0)
#             ax[i][j].plot([0.6,1],[np.mean(v2), np.mean(v2)],color='grey')
#             ax[i][j].plot([0.6,0.6],[0,np.mean(v2)],color='grey')
#             ax[i][j].text(0.53,np.mean(v2),f'{star}',ha='left',va='center',fontstyle='italic',fontsize=fontsize+2, color='grey')
#             n2 = len(v2)
#             dof = n2-1
#             print(f'\t{subject}| hard, {lab} | t{dof}={t:.2f}, p={p:.1e}, {star}')
            
                

        
#             ax[i][j].axhline(0, color='grey', ls='--', zorder=0, lw=0.75)        
#             if i == 1:
#                 ax[i][j].set_xticks([0,1])
#                 ax[i][j].set_xticklabels(['easy', 'hard'])
#                 ax[i][j].set_xlabel('rotation condition')
                
            
#             ax[i][j].set_xlim([-0.5,1.5])
            
 
#             # ax[i][j].set_ylim([-25,260])
#             # ax[i][j].text(-0.4,250,f'{lab}', fontstyle='italic', ha='left',fontsize=fontsize+1)
#             # ax[i][j].set_yticks([0,50,100,150,200,250])
#             # ax[i][j].set_yticklabels(['BL', 50, 100, 150, 200, 250])

#             if j!=0:
#                 ax[i][j].set_yticklabels([])
                
            
#             # if j==2:
#             #     print(f'{subject} | best')
#             #     len1 = len(np.where(v1 < 0)[0])
#             #     len2 = len(np.where(v2 < 0)[0])
                
#             #     print(f'\t easy: {len1}, {len(v1)}, {100*(len1/len(v1)):.1f}')
#             #     print(f'\t hard: {len2}, {len(v2)}, {100*(len2/len(v2)):.1f}')

#         print('')

# # ax[0][0].set_ylabel('performance\n(%Δ in trial time from BL)', fontsize=fontsize+2)
# # ax[1][0].set_ylabel('performance\n(%Δ in trial time from BL)', fontsize=fontsize+2)
         




#%%


"""
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
"""



#%%

"""

DEPRECATED

"""

# import os
# import pickle
# import numpy as np
# import pandas as pd

# from tqdm import tqdm

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sb

# from matplotlib.lines import Line2D
# # from matplotlib.patches import Patch

# from scipy import stats

# # import warnings
# # warnings.filterwarnings("ignore")

# """Plotting Parameters"""
# fontsize = 12
# mpl.rcParams["font.size"] = fontsize
# mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams["font.sans-serif"] = ["Arial"]
# mpl.rcParams["axes.spines.right"] = "False"
# mpl.rcParams["axes.spines.top"] = "False"


# 'IBM Color Scheme - color blind friendly'
# blue    = [100/255, 143/255, 255/255]
# yellow  = [255/255, 176/255, 0/255]
# purple  = [120/255, 94/255, 240/255]
# orange  = [254/255, 97/255, 0/255]
# magenta = [220/255, 38/255, 127/255]


# palette_BEH = {50: yellow, 90: blue}

# # palette_ROT = {50: orange, 90: purple}
# # palette_SHU = {True: magenta, False: 'grey'}

# dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

# pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'  

# 'Loading Custom Functions'
# custom_functions_path  = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions\general_fxns'
# os.chdir(os.path.join(custom_functions_path, 'general_fxns'))
# from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star




# #%%

# mode = 'rotation'
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates, dDegs, dDegs2 = pickle.load(open_file)
# open_file.close()



# #%%


# """

# .....preprocessing.....

# """

# dN = {'rotation': {1:41,2:41}}#, 'shuffle': {1:20,2:41}}

# n_setsBL = dN[mode][1] - 1 
# n_setsPE = dN[mode][2] - 1

# deltaBEH = {}

# for i in [1]:
    
#     deltaBEH[i] = {}
    
#     subj, subject = dSubject[i]
    
#     for key in ['time', 'dist', 'AE', 'MV', 'ME']:
        
#         deltaBEH[i][key] = {'BL': {'mean': np.zeros((len(dDates[i]),n_setsBL)),
#                                    'std': np.zeros((len(dDates[i]),n_setsBL)),
#                                    'var': np.zeros((len(dDates[i]),n_setsBL))},
#                             'PE': {'mean': np.zeros((len(dDates[i]),n_setsPE)),
#                                    'std': np.zeros((len(dDates[i]),n_setsBL)),
#                                    'var': np.zeros((len(dDates[i]),n_setsBL))}}
        
#         for d, date in enumerate(tqdm(dDates[i][:10])):
        
#             open_file = open(os.path.join(pickle_path, f'FA2_inputs1_{subj}_{mode}_{date}.pkl'), "rb")
#             dNF, delU, dNU, dParams, check_shape, tnBL_set, tnPE_set = pickle.load(open_file)
#             open_file.close()
            
#             open_file = open(os.path.join(pickle_path, f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
#             dfBEH_BL, dfBEH_PE = pickle.load(open_file)
#             open_file.close()
            
            
#             for block_key, trial_set, dfBEH in zip(['BL', 'PE'],[tnBL_set, tnPE_set],[dfBEH_BL, dfBEH_PE]):

#                 for count in trial_set.keys():

#                     BEH_date = np.array([dfBEH.loc[dfBEH['tn']==trial, key].values[0] for trial in trial_set[count]])
                        
#                     deltaBEH[i][key][block_key]['mean'][d,count] = np.mean(BEH_date)
#                     deltaBEH[i][key][block_key]['std'][d,count]  = np.std(BEH_date)
#                     deltaBEH[i][key][block_key]['var'][d,count]  = np.var(BEH_date)
                
                
            
# #%%


# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8,8))

# for i in [1]:
    
#     subj,subject = dSubject[i]
    
#     ind50 = np.where(dDegs[i]==50)[0]
#     ind90 = np.where(dDegs[i]==90)[0]
    
#     for j, key in enumerate(['ME', 'AE']):
        
#         for color, indDEG in zip([palette_BEH[50], palette_BEH[90]],[ind50, ind90]):
        
#             ax[j][i].plot( np.mean( deltaBEH[i][key]['BL']['mean'][indDEG,:], axis=0 ), color=color )
#             ax[j][i].plot( np.mean( deltaBEH[i][key]['PE']['mean'][indDEG,:], axis=0 ), color=color )
            


     
        
        
        
        
        
    

                

# #%%

# """

# [Figure 2A]
#     Behavior of Sets of Trials

# """

# VAR =  dT_all
# VAR_LAB = 'time (% change)'

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3,4))
# fig.subplots_adjust(hspace=0.3)

# for i in [0,1]:
    
#     subj, subject = dSubject[i]
#     ax[i].set_title(f'{subject}', loc='left', fontsize=fontsize)
    
#     overall = np.zeros((len(dDates[i]), 41))
    
#     count=0
#     for d, date in enumerate(tqdm(dDates[i][:])):

        
#         mBL = np.mean(VAR[i][date]['BL'])
#         mPE = np.mean(VAR[i][date]['PE'], axis=0)
        
#         delta = 100*((mPE-mBL)/mBL)
#         overall[count,:] = delta

#         count+=1
    
#     m50 = np.mean(overall[ind50,:], axis=0)
#     s50 = stats.sem(overall[ind50,:], axis=0)
#     m90 = np.mean(overall[ind90,:], axis=0)
#     s90 = stats.sem(overall[ind90,:], axis=0)
    
#     ax[i].fill_between(np.arange(41), m50-s50, m50+s50, color=palette_ROT[50])
#     ax[i].fill_between(np.arange(41), m90-s90, m90+s90, color=palette_ROT[90])


#     #ax[i].plot(np.mean(overall[ind50,:], axis=0), color=palette_ROT[50], lw=4, zorder=100)
#     #ax[i].plot(np.mean(overall[ind90,:], axis=0), color=palette_ROT[90], lw=4, zorder=100)
#     ax[i].axhline(0, lw=1, zorder=0, color='grey', ls='--')  
    
#     ax[i].set_ylabel(VAR_LAB)
    
#     ax[i].set_ylim([-10,125])
#     ax[i].set_yticks([0,25,50,75,100,125])
#     ax[i].set_yticklabels(['BL', 25, 50, 75, 100, 125])

# ax[1].set_xlabel('perturbation trial set')
        

# legend_elements = [Line2D([0],[0], marker='s', markersize=10, color=orange, lw=0, markeredgecolor='w', label='easy (50°)'),
#                    Line2D([0],[0], marker='s', markersize=10, color=purple, lw=0, markeredgecolor='w', label='hard (90°)')]
    


# ax[0].legend(handles=legend_elements, ncol=1, fontsize=8, columnspacing=0.3, handletextpad=0.0, loc='upper right')



        
            


# #%%

# """
# ROTATION: average velocity profile (window_len)


# example session, early vs late; 50 vs 90

# """
# #n = window_len


# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,6))
# fig.subplots_adjust(wspace=0.05, bottom=0.075)

# for i in [0,1]:

#     subj, subject = dSubject[i]
#     ax[i].set_title(f'{subject}', loc='left')

#     ind50 = np.where(dDegs[i]==50)[0]
#     ind90 = np.where(dDegs[i]==90)[0]
    
#     date = dDates[i][ind50[0]]
#     BL50 = np.concatenate((dVel_full_neural[i][date]['BL'][:,:,:n]) , axis=0)
#     PE50 = np.concatenate((dVel_full_neural[i][date]['PE'][:,:,:n]) , axis=0)
#     for date in dDates[i][ind50[1:]]:

#             BL50_ = np.concatenate((dVel_full_neural[i][date]['BL'][:,:,:n]) , axis=0)
#             BL50 = np.vstack((BL50,BL50_))
            
#             PE50_ = np.concatenate((dVel_full_neural[i][date]['PE'][:,:,:n]) , axis=0)
#             PE50 = np.vstack((PE50,PE50_))
            
#     m1 = np.mean(BL50, axis=0)
#     s1 = stats.sem(BL50, axis=0)
    
#     m2 = np.mean(PE50, axis=0)
#     s2 = stats.sem(PE50, axis=0)
    
#     ax[i].fill_between(np.arange(n), m1-s1, m1+s1, color='k')
#     ax[i].plot(m1, color=orange, ls='--')
#     ax[i].fill_between(np.arange(n), m2-s2, m2+s2, color=orange)

    

#     date = dDates[i][ind90[0]]
#     BL90 = np.concatenate((dVel_full_neural[i][date]['BL'][:,:,:n]) , axis=0)
#     PE90 = np.concatenate((dVel_full_neural[i][date]['PE'][:,:,:n]) , axis=0)
#     for date in dDates[i][ind90[1:]]:

#             BL90_ = np.concatenate((dVel_full_neural[i][date]['BL'][:,:,:n]) , axis=0)
#             BL90 = np.vstack((BL90,BL90_))
            
#             PE90_ = np.concatenate((dVel_full_neural[i][date]['PE'][:,:,:n]) , axis=0)
#             PE90 = np.vstack((PE90,PE90_))
            
#     m1 = np.mean(BL90, axis=0)
#     s1 = stats.sem(BL90, axis=0)
    
#     m2 = np.mean(PE90, axis=0)
#     s2 = stats.sem(PE90, axis=0)
    
#     ax[i].fill_between(np.arange(n), m1-s1, m1+s1, color='k')
#     ax[i].plot(m1, color=purple, ls='--')
#     ax[i].fill_between(np.arange(n), m2-s2, m2+s2, color=purple)
 
#     ax[i].set_xticks(np.arange(0,n,1))
#     ax[i].set_xticklabels(np.arange(0,n,1)/10, rotation=0, fontsize=10)
#     #ax[i].set_xlabel()
    

# legend_elements = [Line2D([0],[0], marker='s', markersize=15, color=orange, lw=0, markeredgecolor='w', label='easy (50°)'),
#                    Line2D([0],[0], marker='s', markersize=15, color=purple, lw=0, markeredgecolor='w', label='hard (90°)'), 
#                    Line2D([0],[0], marker='s', markersize=15, color='k', lw=0, markeredgecolor='w', label='baseline')]
    


# ax[0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc='upper left')

    

# fig.text(0.5,0,'time from movement onset (s)', ha='center')
# ax[0].set_ylabel('velocity (||cm/s||)')
    




            

