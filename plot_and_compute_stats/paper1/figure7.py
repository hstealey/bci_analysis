# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:00:16 2025

@author: hanna
"""




# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:50:55 2025

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

import statistics
import math


from scipy.optimize import curve_fit

def linear_model(x,m,b):
    return(m*x+b)

            

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

#pickle_root_save_path = os.path.join(pickle_path, 'FA', 'DTS') #'FA_DTS_COVAR')#, 'TEST_TTT')
pickle_path_BEH  = os.path.join(pickle_path, 'dfBEH', 'compute_behavior_results')


pickle_root_save_path = os.path.join(r'C:\Users\hanna\OneDrive\Documents\bci_analysis','bci_analysis_DEPRECATED', 'zDEPRECATED', 'DEPRECATED_PICKLES','DEPRECATED_pickles_FA', 'FA_DTS_covar')


'------------------------------'
'Loading Custom Functions'
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
os.chdir(os.path.join(custom_functions_path, 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star





#%%


# """
# """

# """
# ###############################################################################

# [1] Data Formatting

# ###############################################################################
# """


# n_sets = 41#dN[mode]
# n_trials = n_sets*8


# # window_len = 9 #900ms

# firing_rate_threshold = 1#Hz


# for mode in ['rotation']:

#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, _, _ = pickle.load(open_file)
#     open_file.close()

#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
        
#         for d in tqdm(range(len(dDates[i]))):
#         #for d in [10]:

#             date = dDates[i][d]

#             fn = f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'
#             open_file = open(os.path.join(pickle_path,'trial_inds',fn), "rb")
#             dfBL, dfPE, _ = pickle.load(open_file) #window_len
#             open_file.close()
            
#             dNU = {}
#             dSC_pre = {} 
#             dSC     = {} 


                
#             'Pull same number of trials for each target location.'
#             tnBL = np.zeros((8, n_sets)) 
          
#             for j, deg in enumerate(np.arange(0,360,45)):
#                 tnBL[j,:] = dfBL.loc[dfBL['deg']==deg, 'tn'].values[:n_sets]
     
        
#             nU_pre = np.shape(dfBL.loc[dfBL['tn']==tnBL[0,0], 'spikes'].values[0])[1]
#             dNU['nU_pre'] = nU_pre
            
            
#             units_to_del = []
           
            
#             'Baseline'
#             tnBL_all = np.sort(np.concatenate((tnBL)))
#             sc = dfBL.loc[dfBL['tn']==tnBL_all[0], 'spikes'].values[0]  
#             for tn in tnBL_all:
#                   spikes_tn = dfBL.loc[dfBL['tn']==tn, 'spikes'].values[0]
#                   sc = np.vstack((sc, spikes_tn))

                     

#             dSC_pre['BL'] = sc
#             dSC_pre['mBL'] = np.mean(sc/0.1, axis=0)
#             dSC_pre['target_degs_BL'] = np.array([dfBL.loc[dfBL['tn']==tn, 'deg'].values[0] for tn in  tnBL_all]).astype(int)
          
#             units_to_del.append(np.where(dSC_pre['mBL']< firing_rate_threshold)[0]) 
              
            
#             'Remove neurons with average firing rates below the firing rate threshold (Hz) during any set of trials.'
#             delU = np.unique(np.concatenate((units_to_del)))
            
#             'Baseline'
#             sc = np.delete(dSC_pre['BL'], delU, axis=1)
#             dSC['BL']  = sc
#             dSC['mBL'] = np.mean(sc/0.1, axis=0)
#             dSC['target_degs_BL'] = dSC_pre['target_degs_BL']
            

#             dNU['nU_post'] = np.shape(dSC['BL'])[1]
              
#             dParams1 = {}#'n_setsBL': n_setsBL, 'n_setsPE': n_setsPE,'n_trialsBL': n_trialsBL, 'n_trialsPE': n_trialsPE,'window_start': window_start, 'window_end_not-incusive': window_end,'n_window': n_window}
   

#             os.chdir(os.path.join(pickle_root_save_path, 'FA1'))
#             filename = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
#             obj_to_pickle = [delU, dNU]#, dParams1, tnBL_all, tnPE_all] 
#             open_file = open(filename, "wb")
#             pickle.dump(obj_to_pickle, open_file)
#             open_file.close()


#             os.chdir(os.path.join(pickle_root_save_path, 'FA2'))
#             filename = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
#             obj_to_pickle = [dSC_pre, dSC]
#             open_file = open(filename, "wb")
#             pickle.dump(obj_to_pickle, open_file)
#             open_file.close()





            

#%%


""" extract behavior """


BEH_KEY = 'time'

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    

    dBEH  = {}

    for i in [0,1]:
        
        subj, subject = dSubject[i]

        dBEH[i] = {'BL': np.zeros((len(dDates[i]))),
                   'PE_aol': np.zeros((len(dDates[i]))),
                   'PE_best': np.zeros((len(dDates[i])))}


        for d in tqdm(range(len(dDates[i]))):
            
            date = dDates[i][d]
            
            open_file = open(os.path.join(pickle_path_BEH, f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
            dfBEH_BL, dfBEH_PE = pickle.load(open_file)
            open_file.close()
            
            BL = np.zeros((8,41))
            PE = np.zeros((8,41))
            for deg_i, deg in enumerate(np.arange(0,360,45)):
                
                BL[deg_i,:] = dfBEH_BL.loc[dfBEH_BL['deg']==deg, BEH_KEY].values
                PE[deg_i,:] = dfBEH_PE.loc[dfBEH_PE['deg']==deg, BEH_KEY].values
            
        
            mBL = np.mean(BL)
            mPE = np.mean(PE, axis=0)
            
            delta = 100*((mPE-mBL)/mBL)
            
            best = np.min(delta)
            init = delta[0]
          
            dBEH[i]['BL'][d] = mBL
            dBEH[i]['PE_aol'][d] = (init-best)/init 
            dBEH[i]['PE_best'][d] = best 




#%%



""" ALL TARGETS TOGETHER - average magnitude of BL patterns under PE mapping """


dRES_BL_ALL = {}
dRES_ALL = {}
for i in [0,1]:
    
    dRES_BL_ALL[i] = np.zeros((len(dDates[i])))
    dRES_ALL[i] = np.zeros((len(dDates[i])))
    subj,subject = dSubject[i]
    
    for d, date in enumerate(dDates[i][:]):
    #for d in [18]:
    
        date = dDates[i][d]
        
        fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
        scBL, scPE = pickle.load(open_file)
        open_file.close()
      
         
        fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
        delU, dNU,_,_,_= pickle.load(open_file)
        open_file.close()
        
        
        open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
        dfKG = pickle.load(open_file)
        open_file.close()
        
        
        #Yes, this is correct.  Values were initially stored as (y,x).
        xBL = np.delete(dfKG.yBL.values, delU)
        yBL = np.delete(dfKG.xBL.values, delU)
        xPE = np.delete(dfKG.yPE.values, delU)
        yPE = np.delete(dfKG.xPE.values, delU)
        
        
        
        
        sc = scBL[0].T
        for j,deg in enumerate([45,90,135,180,225,270,315]):#enumerate(np.arange(0,360,45)):
        
            
            sc_ = scBL[deg] 
            
            #sc = sc_.T
            sci = sc_.T
            
            sc = np.hstack((sc, sci))
            
           
        
        KYx_BL = np.matmul(xBL, sc)
        KYy_BL = np.matmul(yBL, sc)
        magBL = np.sqrt(KYx_BL**2 +KYy_BL**2)
        
        KYx_PE = np.matmul(xPE, sc)
        KYy_PE = np.matmul(yPE, sc)
        magPE = np.sqrt(KYx_PE**2 +KYy_PE**2)
        
        
        degBL = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_BL, KYx_BL)])*(180/np.pi) 
        degPE = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_PE, KYx_PE)])*(180/np.pi)
        
        
        bins_plot = np.arange(-1,360,20)
        bins_kde = np.arange(-1,361,1)
        


        kern1 = stats.gaussian_kde(degBL)#, weights=magBL) #can add weights for each point....
        k1 = kern1(bins_kde)
  
        kern2 = stats.gaussian_kde(degPE)#, weights=magPE) #can add weights for each point....
        k2 = kern2(bins_kde)

        min_overlap = np.minimum(k1,k2)
        
        # plt.hist(degBL, color='k')
        # plt.hist(degPE, color='r')
        

        
        kde_overlap = np.trapz(min_overlap, bins_kde)
        
        #print(kde_overlap)
            

        dRES_BL_ALL[i][d] = np.std(degBL) #stats.kurtosis(degBL) #np.var(degBL) #np.std(degBL)
        dRES_ALL[i][d] = kde_overlap #overlap_area
        
        
#%%


'BL | BL'

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    
    v1 = dRES_BL_ALL #dRES_ALL[i] 
    v2 = dBEH[i]['BL'] 
    
    r,p,star = compute_corr(v1,v2)
    print(f'{subject} | BL-BL | r={r:.2f}, p={p:.1e} ({star})')
    
    plt.scatter(v1,v2)
    
    
    
    
    # v1 = dRES_ALL[i] #dRES_ALL[i] 
    # v2 = dBEH[i]['PE_aol'] #PE_aol 
    
    # r,p,star = compute_corr(v1,v2)
    # print(f'{subject} | BL-PE_best | r={r:.2f}, p={p:.1e} ({star})')
    
    # plt.scatter(v1,v2)
    
    
      
    

#%%

""" average magnitude of BL patterns under PE mapping """


dRES_BL = {}
dRES = {}
for i in [0,1]:
    
    dRES_BL[i] = np.zeros((8,len(dDates[i])))
    dRES[i] = np.zeros((8,len(dDates[i])))
    subj,subject = dSubject[i]
    
    for d, date in enumerate(dDates[i][:]):
    #for d in [18]:
    
        #date = dDates[i][d]
        
        fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
        scBL, scPE = pickle.load(open_file)
        open_file.close()
      
         
        fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
        delU, dNU,_,_,_= pickle.load(open_file)
        open_file.close()
        
        
        open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
        dfKG = pickle.load(open_file)
        open_file.close()
        
        
        #Yes, this is correct.  Values were initially stored as (y,x).
        xBL = np.delete(dfKG.yBL.values, delU)
        yBL = np.delete(dfKG.xBL.values, delU)
        xPE = np.delete(dfKG.yPE.values, delU)
        yPE = np.delete(dfKG.xPE.values, delU)
        
        for j,deg in enumerate(np.arange(0,360,45)):
        #for j, deg in zip([0],[0]):#
            
            sc_ = scBL[deg] #(scBL[deg]/(0.9))*0.1
            
            sc = sc_.T
            
            
            #sc = spikesBL[i][date].T
            
            KYx_BL = np.matmul(xBL, sc)
            KYy_BL = np.matmul(yBL, sc)
            magBL = np.sqrt(KYx_BL**2 +KYy_BL**2)
            
            KYx_PE = np.matmul(xPE, sc)
            KYy_PE = np.matmul(yPE, sc)
            magPE = np.sqrt(KYx_PE**2 +KYy_PE**2)
            

            if deg > 175:
                degBL = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_BL, KYx_BL)])*(180/np.pi) 
                degPE = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_PE, KYx_PE)])*(180/np.pi)
                bins_plot = np.arange(-1,360,20)
                bins_kde = np.arange(-1,361,1)
   
            else:
                degBL = np.array([rad if rad<0 else rad for rad in np.arctan2(KYy_BL, KYx_BL)])*(180/np.pi) 
                degPE = np.array([rad if rad<0 else rad for rad in np.arctan2(KYy_PE, KYx_PE)])*(180/np.pi)
                bins_plot = np.arange(-181,181,20)
                bins_kde = np.arange(-181,181,1)


            kern1 = stats.gaussian_kde(degBL)#, weights=magBL) #can add weights for each point....
            k1 = kern1(bins_kde)
  
            kern2 = stats.gaussian_kde(degPE)#, weights=magPE) #can add weights for each point....
            k2 = kern2(bins_kde)

            min_overlap = np.minimum(k1,k2)
            
            # lo = int(deg-23)
            # hi = int(deg+23)
            
            # ind_lo = np.where(bins_kde >= lo)[0][0]
            # ind_hi = np.where(bins_kde >= hi)[0][0]
            
            
            
            # kde_overlap = np.trapz(min_overlap[ind_lo:ind_hi], bins_kde[ind_lo:ind_hi])
            
                        
            
            kde_overlap = np.trapz(min_overlap, bins_kde)
            

            dRES_BL[i][j,d] = np.std(degBL) #stats.kurtosis(degBL) #np.var(degBL) #np.std(degBL)
            dRES[i][j,d] = kde_overlap #overlap_area
            
            
         
  
            # plt.hist(degBL, bins=bins_plot, density=True, color='k', alpha=0.5)
            # plt.hist(degPE, bins=bins_plot, density=True, color='r', alpha=0.5)
          
            # plt.plot(bins_kde, kern1(bins_kde),color='k')
            # plt.plot(bins_kde, kern2(bins_kde),color='r')
            
            # lo = int(deg-23)
            # hi = int(deg+23)
            
            # ind_lo = np.where(bins_kde >= lo)[0][0]
            # ind_hi = np.where(bins_kde >= hi)[0][0]
            
            # plt.plot(bins_kde[ind_lo:ind_hi], min_overlap[ind_lo:ind_hi], color='w')
            
            # plt.axvline(deg)
            # # plt.scatter(deg-22.5, deg+22.5)
            
            

#%%




"""

Predicting behavior from baseline information...
    delta, init, best, AOL, TSN? 

"""

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(5,10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]
    
    ax[0][i].set_title(f'{subject}', loc='left')

#     axis=ax[0][i]
#     v1 = np.mean(dRES_BL[i], axis=0)
#     v2 = np.mean(dRES[i], axis=0)
#     axis.scatter(v1,v2)

# #%%
    'Baseline | Baseline'
    axis = ax[0][i]
    v1 = np.mean(dRES_BL[i], axis=0)
    v2 = dBEH[i]['BL'] 
    
    axis.scatter(v1,v2, color='w', edgecolor='k', alpha=1, s=20)
    r,p,star = compute_corr(v1,v2)
    #axis.text(42,1.93,f'{subject} | r={r:.2f} ({star})')
    #print(f'{subject} | BL-BL | r={r:.2f}, p={p:.1e} ({star})')
    axis.text(42,2, f'r={r:.2f} ({star})')
    axis.set_xlim([40,85])
    axis.set_ylim([0.9,2.1])
    params, _ = curve_fit(linear_model, v1, v2)

    x1 = np.min(v1)
    x2 = np.max(v1)
    
    y1 = linear_model(x1,*params)
    y2 = linear_model(x2,*params)
    axis.plot([x1,x2],[y1, y2], color='k', lw=2, ls='-', zorder=100)


    '-----'
    axis = ax[1][i]
    v1 = np.mean(dRES[i], axis=0)
    v2 = dBEH[i]['PE_best']
    axis.scatter(v1,v2, color='grey', alpha=0.6, s=15, edgecolor='k')
    r,p,star = compute_corr(v1,v2)
    #print(f'{subject} | BL-PE best| r={r:.2f}, p={p:.1e} ({star})')
    axis.text(0.05,55, f'r={r:.2f} ({star})')
    params, _ = curve_fit(linear_model, v1, v2)

    x1 = np.min(v1)
    x2 = np.max(v1)
    
    y1 = linear_model(x1,*params)
    y2 = linear_model(x2,*params)
    axis.plot([x1,x2],[y1, y2], color='k', lw=2, ls='-', zorder=100)
    
    '----'
    axis = ax[2][i]
    v1 = np.mean(dRES[i], axis=0)
    v2 = dBEH[i]['PE_aol']
    axis.scatter(v1,v2,color='grey', alpha=0.6, s=15, edgecolor='k')
    r,p,star = compute_corr(v1,v2)
    #axis.text(42,1.93,f'{subject} | r={r:.2f} ({star})')
    #print(f'{subject} | BL-PE aol | r={r:.2f}, p={p:.1e} ({star})')
    axis.text(0.05,2.5, f'r={r:.2f} ({star})')



    params, _ = curve_fit(linear_model, v1, v2)

    x1 = np.min(v1)
    x2 = np.max(v1)
    
    y1 = linear_model(x1,*params)
    y2 = linear_model(x2,*params)
    axis.plot([x1,x2],[y1, y2], color='k', lw=2, ls='-', zorder=100)
   
ax[0][0].set_ylabel(f'mean BL trial time (s)', fontsize=fontsize+1) 
ax[1][0].set_ylabel(f'best performance\n(%Δ in trial time from BL)', fontsize=fontsize+1)  
ax[2][0].set_ylabel(f'amount of recovery', fontsize=fontsize+1)    

for i in [0,1]:

    ax[1][i].set_ylim([-30,65])
    ax[1][i].set_yticks([-20,0,20,40,60])
    ax[1][i].set_yticklabels([-20,'BL',20,40,60], fontsize=11)
    ax[1][i].axhline(0, color='grey', ls='--', zorder=0)
    
    ax[1][i].set_xlim([0,1])
    ax[1][i].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[1][i].set_xticklabels([0,25, 50, 75, 100], fontsize=10)



    ax[2][i].set_ylim([-0.1,2.6])
    ax[2][i].set_yticks([0,0.5,1,1.5,2.0,2.5])
    ax[2][i].set_yticklabels([0,0.5,1,1.5,2.0,2.5], fontsize=11)
    ax[2][i].axhline(1, color='grey', ls='--', zorder=0)
    
    ax[2][i].set_xlim([0,1])
    ax[2][i].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[2][i].set_xticklabels([0,25, 50, 75, 100], fontsize=10)
    
    if i == 1:
        ax[0][i].set_yticklabels([])
        ax[1][i].set_yticklabels([])
        ax[2][i].set_yticklabels([])
        


    ax[0][i].set_xlabel('STDDEV of BL distribution')
    ax[1][i].set_xlabel('% overlap')
    ax[2][i].set_xlabel('% overlap')






#%%

"""

Example of "overlap"

"""

dRES_BL = {}
dRES = {}
for i in [0]:
    
    dRES_BL[i] = np.zeros((8,len(dDates[i])))
    dRES[i] = np.zeros((8,len(dDates[i])))
    subj,subject = dSubject[i]
    
    #for d, date in enumerate(dDates[i]):
    for d in [9]:
    
        date = dDates[i][d]
        
        fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_path, 'FA2', fn), "rb")
        scBL, scPE = pickle.load(open_file)
        open_file.close()
        
         
        fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_path, 'FA1', fn), "rb")
        delU, dNU, dParams1, _, _ = pickle.load(open_file)
        open_file.close()
        
        
        open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
        dfKG = pickle.load(open_file)
        open_file.close()
        
        
        #Yes, this is correct.  Values were initially stored as (y,x).
        xBL = np.delete(dfKG.yBL.values, delU)
        yBL = np.delete(dfKG.xBL.values, delU)
        xPE = np.delete(dfKG.yPE.values, delU)
        yPE = np.delete(dfKG.xPE.values, delU)
        
        for j,deg in zip([0],[225]): #enumerate(np.arange(0,360,45)): #
        
            sc = scBL[deg].T
            #sc = spikesBL[i][date].T
            
            KYx_BL = np.matmul(xBL, sc)
            KYy_BL = np.matmul(yBL, sc)
            magBL = np.sqrt(KYx_BL**2 +KYy_BL**2)
            
            KYx_PE = np.matmul(xPE, sc)
            KYy_PE = np.matmul(yPE, sc)
            magPE = np.sqrt(KYx_PE**2 +KYy_PE**2)
            

            if deg > 175:
                degBL = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_BL, KYx_BL)])*(180/np.pi) 
                degPE = np.array([rad+(2*np.pi) if rad<0 else rad for rad in np.arctan2(KYy_PE, KYx_PE)])*(180/np.pi)
                bins_plot = np.arange(-1,360,20)
                bins_kde = np.arange(-1,361,1)
   
            else:
                degBL = np.array([rad if rad<0 else rad for rad in np.arctan2(KYy_BL, KYx_BL)])*(180/np.pi) 
                degPE = np.array([rad if rad<0 else rad for rad in np.arctan2(KYy_PE, KYx_PE)])*(180/np.pi)
                bins_plot = np.arange(-181,181,20)
                bins_kde = np.arange(-181,181,1)
            

            kern1 = stats.gaussian_kde(degBL)#, weights=magBL) #can add weights for each point....
            k1 = kern1(bins_kde)
  
            kern2 = stats.gaussian_kde(degPE)#, weights=magPE) #can add weights for each point....
            k2 = kern2(bins_kde)

            min_overlap = np.minimum(k1,k2)
            kde_overlap = np.trapz(min_overlap, bins_kde)
            
            
  
            #plt.hist(degBL, bins=bins_plot, density=True, color='k', alpha=0.5)
            #plt.hist(degPE, bins=bins_plot, density=True, color='r', alpha=0.5)
          
            plt.plot(bins_kde, kern1(bins_kde),color='k',lw=1, ls='--', zorder=10, label='BL spikes -> BL map')
            plt.plot(bins_kde, kern2(bins_kde),color='blue', lw=1,ls='--', zorder=10, label='BL spikes -> PE map')
            plt.plot(bins_kde, min_overlap, color='k', label='overlap')
            plt.fill_between(bins_kde, 0, min_overlap, color='w', alpha=0.4, zorder=10)
            
            
            plt.fill_between(bins_kde, 0, kern1(bins_kde), color='k', alpha=1, zorder=2)
            plt.fill_between(bins_kde, 0, kern2(bins_kde), color='b', alpha=0.7, zorder=1)
            
            
            
            plt.xlabel('direction (°) of velocity command')
            plt.xticks(np.arange(0,361,45))
            plt.ylabel('density (1e3)')
            plt.yticks(np.arange(0,0.011, 0.002), (np.arange(0,0.011, 0.002)* 1e3).astype(int))
            
            #plt.legend(fontsize=10)
            
            #plt.title(f'Example cloud (0 deg. target) from example session ', fontsize=12)
            
            # plt.axvline(deg, color='k', lw=2, zorder=10)
            # plt.axvline(np.mean(degBL), color='k', lw=2, zorder=10)




#%%
"""
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
"""


#%%

"""
ROTATION 
    OVERALL
"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(6,6))

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        ax[0][i].set_title(f'{subject}', loc='left')
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        # deg_i = 0
        
        v1_a = dProj_ALL[i]['BL'][ind50] #dProj[i]['BL'][deg_i,ind50]  
        v2_a = dProj_ALL[i]['PE'][ind50] #dProj[i]['PE'][deg_i,ind50] 
        delta1 = 100*((v2_a-v1_a)/v1_a)

        v1 = dProj_ALL[i]['BL'][ind90] #dProj[i]['BL'][deg_i,ind90]
        v2 = dProj_ALL[i]['PE'][ind90] #dProj[i]['PE'][deg_i,ind90]
        delta = 100*((v2-v1)/v1)

        #print(compute_two_sample_ttest(delta1,delta))

        xdata = [50]*len(ind50) + [90]*len(ind90)
        ydata = np.concatenate((delta1,delta))
        sb.swarmplot(x=xdata, y=ydata, ax=ax[0][i], palette=palette_ROT)
        
        ax[0][i].axhline(0, color='grey', zorder=0, ls='--')
        
        v1 = delta1
        v2 = delta
        
        
        axis = ax[0][i]
        axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=75, zorder=10)
        (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        ax[0][i].plot([0,1],[22,22],color='k')
        ax[0][i].text(0.5,22,f'{star}',ha='center',va='bottom', fontstyle='italic')
       
        t,p,star = compute_one_sample_ttest(v1,0)
        ax[0][i].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
        ax[0][i].plot([0.4,0.4],[0, np.mean(v1)],color='grey')
        ax[0][i].text(0.4,0,f'{star}',ha='center',va='bottom', fontstyle='italic')
       
        t,p,star = compute_one_sample_ttest(v2,0)
        ax[0][i].plot([0.6,1],[np.mean(v2), np.mean(v2)],color='grey')
        ax[0][i].plot([0.6,0.6],[0, np.mean(v2)],color='grey')
        ax[0][i].text(0.6,0,f'{star}',ha='center',va='bottom', fontstyle='italic')
       

        
        
        
        v1_a = dProj_ALL_NULL[i]['BL'][ind50]
        v2_a = dProj_ALL_NULL[i]['PE'][ind50]
        delta1 = 100*((v2_a-v1_a)/v1_a)

        v1 = dProj_ALL_NULL[i]['BL'][ind90]
        v2 = dProj_ALL_NULL[i]['PE'][ind90]
        delta = 100*((v2-v1)/v1)

        xdata = [50]*len(ind50) + [90]*len(ind90)
        ydata = np.concatenate((delta1,delta))
        sb.swarmplot(x=xdata, y=ydata, ax=ax[1][i], palette=palette_ROT)
        
        ax[1][i].axhline(0, color='grey', zorder=0, ls='--')
        
        v1 = delta1
        v2 = delta
        
        
        axis = ax[1][i]
        axis.scatter([0,1], [np.mean(v1), np.mean(v2)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=75, zorder=10)
        (_, caps, _) = axis.errorbar([0,1], [np.mean(v1), np.mean(v2)], yerr=[np.std(v1), np.std(v2)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
        t,p,equal_var,star = compute_two_sample_ttest(v1,v2)
        ax[1][i].plot([0,1],[22,22],color='k')
        ax[1][i].text(0.5,22,f'{star}',ha='center',va='bottom', fontstyle='italic')
       
     
        t,p,star = compute_one_sample_ttest(v1,0)
        ax[1][i].plot([0,0.4],[np.mean(v1), np.mean(v1)],color='grey')
        ax[1][i].plot([0.4,0.4],[0, np.mean(v1)],color='grey')
        ax[1][i].text(0.4,0,f'{star}',ha='center',va='bottom', fontstyle='italic')
       
     
        t,p,star = compute_one_sample_ttest(v2,0)
        ax[1][i].plot([0.6,1],[np.mean(v2), np.mean(v2)],color='grey')
        ax[1][i].plot([0.6,0.6],[0, np.mean(v2)],color='grey')
        ax[1][i].text(0.6,0,f'{star}',ha='center',va='bottom', fontstyle='italic')
       
      
        
        ax[1][i].set_xticks([0,1])
        ax[1][i].set_xticklabels(['easy', 'hard'])
        ax[1][i].set_xlabel('rotation condition')

ax[0][0].set_ylabel('Δ covariability\nalong mapping (%)')
ax[1][0].set_ylabel('Δ covariability\nalong null space (%)')


#%%

'OVERALL: Correlation of changes in potent space and changes in null space'

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,3))

for i in [0,1]:
    
    subj,subject = dSubject[i]
    ax[i].set_title(f'{subject}', loc='left')

    v1 = dProj_ALL[i]['BL']
    v2 = dProj_ALL[i]['PE']
    delta1 = 100*((v2-v1)/v1)
    
    v1 = dProj_ALL_NULL[i]['BL']
    v2 = dProj_ALL_NULL[i]['PE']
    
    delta2 = 100*((v2-v1)/v1)
    
    ax[i].scatter(delta1, delta2, color='k', alpha=0.8)
    
    r,p,star = compute_corr(delta1,delta2)
    ax[i].text(-25,21,f'r={r:.2f} ({star})')
    
    r,p,star = compute_corr(delta1[ind50],delta2[ind50])
    ax[i].text(5,21,f'r={r:.2f} ({star})', color=orange)
    r,p,star = compute_corr(delta1[ind90],delta2[ind90])
    ax[i].text(5,18,f'r={r:.2f} ({star})', color=purple)
    
    ax[i].axvline(0,color='grey', zorder=0, ls='--')
    ax[i].axhline(0,color='grey',zorder=0, ls='--')
    
ax[0].set_ylabel('Δ covariability\nalong null space(%)')
ax[0].set_xlabel('Δ covariability\nalong mapping (%)')
ax[1].set_xlabel('Δ covariability\nalong mapping (%)')
            
#%%


avgBL = {}
avgPE = {}

kNN = 5


for i in [0]:#,1]:
    
    avgBL[i] = {}
    avgPE[i] = {}
    
    subj, subject = dSubject[i]

    #for d in tqdm(range(len(dDates[i]))):
    for d in [8]:
        
        date = dDates[i][d]
        
        avgBL[i][date] = np.zeros((8,8))
        avgPE[i][date] = np.zeros((8,8))
        
        S_before = Sbefore_all[i][date]#[deg]
        inv_cov = np.linalg.inv(S_before)
        
        fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_root_path, 'FA2', fn), "rb")
        scBL, scPE = pickle.load(open_file)
        open_file.close()
        
        for deg1_ind,deg1 in enumerate(np.arange(0,360,45)):
            
            sc1 = scBL[deg1].astype(float)
            nT, nU = np.shape(sc1)
            
            sc1_PE = scPE[deg1].astype(float)
            
            for deg2_ind,deg2 in enumerate(np.arange(0,360,45)):
    
                sc2 = scBL[deg2].astype(float)
                
                sc2_PE = scPE[deg2].astype(float)
    
                D1 = np.ones((nT,nT))
                for j in range(nT):
                    for k in range(nT):
                        dy = sc1[j,:] - sc2[k,:]
                        D1[j,k] = np.sqrt( np.dot(dy,np.dot(inv_cov,dy.T)) ) 


                D2 = np.ones((nT,nT))
                for j in range(nT):
                    for k in range(nT):
                        dy = sc1_PE[j,:] - sc2_PE[k,:]
                        D2[j,k] = np.sqrt( np.dot(dy,np.dot(inv_cov,dy.T)) ) 


                mKNN_ = np.mean(np.partition(D1,kth=kNN+1,axis=0)[:kNN+1,:], axis=0)
                avgBL[i][date][deg1_ind, deg2_ind] = np.mean(mKNN_)
                
                mKNN_ = np.mean(np.partition(D2,kth=kNN,axis=0)[:kNN,:], axis=0)
                avgPE[i][date][deg1_ind, deg2_ind] = np.mean(mKNN_)
                
                print(deg1_ind, deg2_ind, avgBL[i][date][deg1_ind, deg2_ind] ,avgPE[i][date][deg1_ind, deg2_ind] )

#%%

D_BL = avgBL[i][date]
D_PE = avgPE[i][date]

sb.heatmap((D_PE/D_BL)-1, vmin=-0.1, vmax=0.1, cmap=plt.cm.PRGn)#, vmin=0.75, vmax=1.1, cmap=plt.cm.Purples_r)                

                
                #%%
        
                # D = np.ones((nT,nT))
                # for j in range(nT):
                #     for k in range(nT):
                #         dy = sc2[j,:] - sc1[k,:]
                #         D[j,k] = np.sqrt( np.dot(dy,np.dot(inv_cov,dy.T)) ) 
                
        
        

                
                num = np.mean(avgBL)
                den = np.mean(avgPE)
                
                lam = (nT-1)/nT
                dt = (lam*(num/den)) - 1
                
            
                
      
            




#%%
lam = (nT-1)/nT


dt = (lam*(num/den)) - 1

#%%

# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8,4))

# for i in [0,1]:
    
#     v1 = dProj_ALL[i]['BL'] #np.concatenate((dProj[i]['BL']))
#     v2 = np.mean(dBEH[i]['PE'], axis=0) #np.concatenate((dBEH[i]['BL']))
    
#     ax[i].scatter(v1,v2,alpha=0.5)
    
#     print(compute_corr(v1,v2))
    
            
            
#%%


"""
SHUFFLE
"""

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(6,6))

# for mode in ['shuffle']:
    
#     open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
#     dDates, dDegs, dDegs2 = pickle.load(open_file)
#     open_file.close()

#     for i in [0,1]:
        
#         subj, subject = dSubject[i]
        

#         v1 = np.mean(dProj[i]['BL'][:,:], axis=0)
#         v2 = np.mean(dProj[i]['PE'][:,:], axis=0)
        
#         s1 = stats.sem(dProj[i]['BL'][:,:], axis=0)
#         s2 = stats.sem(dProj[i]['PE'][:,:], axis=0)
        
#         ax[0][i].errorbar(x=v1,y=v2,xerr=s1,yerr=s2,color=magenta, alpha=0.5)
#         ax[0][i].plot([0,50],[0,50],color='grey',ls='--', zorder=0)
        
        
#         delta = 100*((v2-v1)/v1)
#         ax[1][i].hist(delta)
        
        



#%%





#%%
"""
ROTATION
"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(6,6))

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()

    for i in [0,1]:
        
        subj, subject = dSubject[i]
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]
        
        v1 = np.mean(dProj[i]['BL'][:,ind50], axis=0)#[:5]
        v2 = np.mean(dProj[i]['PE'][:,ind50], axis=0)#[:5]
        
        s1 = stats.sem(dProj[i]['BL'][:,ind50], axis=0)#[:5]
        s2 = stats.sem(dProj[i]['PE'][:,ind50], axis=0)#[:5]
        
        ax[0][i].plot([0,50],[0,50],color='grey', alpha=0.5, zorder=0, ls='--')
        ax[0][i].errorbar(x=v1,y=v2,xerr=s1,yerr=s2, fmt='none', color=orange, alpha=0.5)

        delta1 = 100*((v2-v1)/v1)
        ax[1][i].hist(delta1)
        
        # print(compute_one_sample_ttest(v1,0))
        # print(compute_one_sample_ttest(v2,0))

        # print(np.min(delta), np.max(delta))


        v1 = np.mean(dProj[i]['BL'][:,ind90], axis=0)#[:5]
        v2 = np.mean(dProj[i]['PE'][:,ind90], axis=0)#[:5]
        
        s1 = stats.sem(dProj[i]['BL'][:,ind90], axis=0)#[:5]
        s2 = stats.sem(dProj[i]['PE'][:,ind90], axis=0)#[:5]
        
        #ax[0][i].plot([0,50],[0,50],color='grey', alpha=0.5, zorder=0, ls='--')
        ax[0][i].errorbar(x=v1,y=v2,xerr=s1,yerr=s2, fmt='none', color=purple, alpha=0.5)
        
        
        delta = 100*((v2-v1)/v1)
        ax[1][i].hist(delta)
        
        
        # print(np.min(delta), np.max(delta))
        
        print(compute_two_sample_ttest(delta1,delta))


        # print(compute_one_sample_ttest(v1,0))
        # print(compute_one_sample_ttest(v2,0))



#%%


"""

....

"""


fig, ax = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(6,3))

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    
    for i in [0,1]:
        
        subj, subject = dSubject[i]
        
        ind50 = np.where(dDegs[i]==50)[0]
        ind90 = np.where(dDegs[i]==90)[0]

        v1 = dProj_ALL[i]['BL']/dProj_ALL_NULL[i]['BL']
        v2 = np.mean(dBEH[i]['PE'], axis=0)

        ax[i].scatter(v1[ind50],v2[ind50], color='k')
        
        print(compute_corr(v1[ind50], v2[ind50]))
        


      
