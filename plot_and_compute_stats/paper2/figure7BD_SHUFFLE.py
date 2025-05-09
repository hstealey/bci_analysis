# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 21:29:32 2025

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

# from scipy.optimize import curve_fit

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
palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'}

dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles') 
pickle_path_VAR = os.path.join(pickle_path, 'FA_tuning')  
pickle_path_dTC = os.path.join(pickle_path, 'tuning') 
# pickle_path_FA_shuffle = os.path.join(pickle_path, 'FA_tuning') 
              

'------------------------------'
'Loading Custom Functions'
os.chdir(os.path.join(root_path, 'functions', 'neural_fxns'))
from tuning_curve_fxns import degChange, compute_signed_degChange

os.chdir(os.path.join(root_path, 'functions', 'general_fxns'))
from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star
from compute_stats import compute_r_squared, linear_func1, linear_func2, linear_func2i
from compute_stats import bin_data, fit_linear_model





#%%

"""

Load data.

"""

mode = 'shuffle'

fn = f'tuning_ns0_{mode}.pkl' # alt: fn = f'tuning_ns0_{mode}.pkl'
open_file = open(os.path.join(pickle_path_dTC, fn), "rb")  #previously: open_file = open(os.path.join(pickle_path, 'postprocessing', 'paper2' ,fn), "rb") 
dDates, dDegs, dDegs2, dPD_all, dPD_each = pickle.load(open_file)
open_file.close()

n_window = 4  #number of samples in the window




#%%



"""

    ...collecting metrics...

"""



dRES = {}



for i in [0,1]:

    subj, subject = dSubject[i]
    
    a_dPD  = []
    dPD    = []
    sig    = []
    shuff   = []
    
    svBL = []
    MD_BL = []
    KY = []
    
    session = []
    
    mFR = []
    
    
    
    nU_pre = []
    nU_post = []
    
    
    
    TF = []
    
    for d, date in enumerate(dDates[i][:]):
        
        '---tuning---'    
        #a_dPD.append(dPD_each[i][date]['assigned_dPD'])
        a_dPD.append(np.round(dPD_each[i][date]['assigned_dPD']).astype(int))
        dPD.append(dPD_each[i][date]['dPD_median'])
        sig.append(dPD_each[i][date]['sig'])
        
        MD_BL.append(dPD_each[i][date]['MD_BL'])
        

        '---%svBL---'
        fn = f'FA5_dVAR_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_path_VAR, 'FA5', fn), "rb")
        dVAR = pickle.load(open_file)
        open_file.close()
        
        svBL.append(100*dVAR['BL']['%sv'])
    

        '---dSC---'
        fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_path_VAR, 'FA2', fn), "rb")
        _, dSC = pickle.load(open_file)
        open_file.close()
        
        
        scBL = np.mean(dSC['BL']/(n_window*0.1), axis=0)#*0.1 #average spike counts in 100ms bin
        mFR.append(scBL)
        
        '---dfKG---'
        fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
        open_file = open(os.path.join(pickle_path_VAR, 'FA1', fn), "rb")
        delU, dNU, dParams1, _, _ = pickle.load(open_file)
        open_file.close()     
        
        
        
        open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
        dfKG = pickle.load(open_file)
        open_file.close()
        
        shuff.append(np.delete(dfKG['shuffled'].values, delU))
    
                
        #Yes, these are supposed to be "swapped" like this (x=y).
        xBL = np.delete(dfKG['yBL'].values, delU) 
        yBL = np.delete(dfKG['xBL'].values, delU) 
        

        scBL_100ms = np.mean(dSC['BL']/(n_window*0.1), axis=0)*0.1
        KY_ = np.sqrt((xBL*scBL_100ms)**2 + (yBL*scBL_100ms)**2)
        KY.append(100*(KY_/np.sum(KY_))) 
        
       
        session.append(np.ones(len(xBL))*d)    
    
    a_dPD  = np.concatenate((a_dPD))  
    dPD = np.concatenate((dPD))
    sig =  np.concatenate((sig))
    shuff   = np.concatenate((shuff))
    
    MD_BL = np.concatenate((MD_BL))
    svBL = np.concatenate((svBL))
    KY =  np.concatenate((KY))
    
    mFR = np.concatenate((mFR))
    
    session = np.concatenate((session))


    dfPARAMS = pd.DataFrame({'session': session,
                           'a_dPD': a_dPD,'dPD': dPD, 
                           'shuff': shuff, 'sig': sig, 
                           'MD_BL': MD_BL,'svBL': svBL, 'KY': KY,
                           'mFR': mFR})
    
    dRES[i] = dfPARAMS
  



#%%


"""

##############################################################################

[Figure 7B] 

log of % shared variance & measured change in preferred direction
log of % contribution & measured change in preferred direction


##############################################################################

"""

palette_SHU = {True: magenta, False: 'k'}


fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(6,6))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

dBS = {}

for i in [0,1]:
    
    dBS[i] = {}
    
    subj, subject = dSubject[i]
    ax[0][i].set_title(f'{subject}', loc='left')
    
    
    dfPARAMS = dRES[i]
    dfSIG_ = dfPARAMS.loc[(dfPARAMS['sig']==1)] 
    
    
    for q, DEG in zip([0,1],[False, True]):
        
        dBS[i][DEG] = {}
        
        color = palette_SHU[DEG]

        dfSIG = dfSIG_.loc[(dfPARAMS['shuff']==DEG) ]
        
        
        n_sig = []
        for sesh in np.unique(dfSIG['session'].values):
            n_sig.append(len(dfSIG.loc[dfSIG['session']==sesh]))
    
       
        for j,KEY in enumerate(['svBL', 'KY']):
            axis = ax[j][i]
            xdata = np.log(dfSIG[KEY].values) #np.log10(dfSIG[KEY].values)
            ydata = np.abs(dfSIG['dPD'].values)
            
            dBS[i][DEG][KEY] = {'xdata': xdata, 'ydata': ydata, 'n_sig':n_sig}
            
            
            
            
            if i==0:
                axis.set_ylabel('measured ΔPD (°)')
            axis.set_ylim([0,180])
            axis.set_yticks(np.arange(0,181,40))
    
            

            '---BINNED DATA---'
            #xbin, ybin, ysem = bin_data(xdata, ydata)
            #axis.scatter(xbin,ybin, color=color, edgecolor='k', zorder=100)
            #axis.fill_between(xbin, ybin-ysem, ybin+ysem, color=color, alpha=0.85)

            # params, perr, tvalues, pvalues = fit_linear_model(xdata, ydata, model_type='L1')        
            # x_est = np.array([np.min(xbin), np.max(xbin)])
            # y_est = linear_func1(x_est, *params)
            

            '---FULL DATA---'
            axis.scatter(xdata,ydata, color=color, alpha=0.3, zorder=0, s=12)
            params, perr, tvalues, pvalues = fit_linear_model(xdata, ydata, model_type='L1')        
            x_est = np.array([np.min(xdata), np.max(xdata)])
            y_est = linear_func1(x_est, *params)
            axis.plot(x_est,y_est,color=color, ls='--', zorder=99)
            
            print(params[1])
            
            
            if KEY == 'svBL':
                xloc = 0.1
                axis.set_xlim([-0.15,5])
                axis.set_xticks(np.arange(0,5.1,1))
                #print(np.min(xdata))
            elif KEY == 'KY':
                xloc = -4.4
                axis.set_xlim([-4.64,4.64])
                axis.set_xticks([-4,-2,0,2,4])
                #x_est = np.array([-2,2.1])
                
            
            
                

            r,p,star = compute_corr(xdata,ydata)
            axis.text(xloc, 170-(q*15), f'r={r:.2f} ({star})', color=color)#, {stats_star(pvalues[0])}') #{params[0]:.2f} ± {np.sqrt(pcov[0, 0]
            
            axis.set_ylim([-5,185])
            axis.set_yticks(np.arange(0,181,45))


ax[0][0].set_xlabel('ln(%sv)')
ax[0][1].set_xlabel('ln(%sv)')

ax[1][0].set_xlabel('ln(%contribution)')
ax[1][1].set_xlabel('ln(%contribution)')



legend_elements = [Line2D([0],[0], marker='s', markersize=8, color='k', lw=0, label='non-shuffled'),
                   Line2D([0],[0], marker='s', markersize=8, color=magenta, lw=0, label='shuffled')]#,
                   # Line2D([0],[0], marker='^', markersize=15, color='w', lw=0, markeredgecolor='k', label='applied'),
                   # Line2D([0],[0], marker='*', markersize=15, color='k', lw=0, markeredgecolor='k', label='mean')]
    

legend = ax[0][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc=(0.57,0.825), fontsize=fontsize-2)

frame = legend.get_frame()
frame.set_alpha(1)


fig.subplots_adjust(wspace=0.15, hspace=0.3)


for i in [0,1]:
    ax[0][i].axvline(np.log(100),ls='--',color='grey',zorder=0)
    ax[0][i].text(np.log(100),110,'100%',color='grey', rotation=-90, fontsize=fontsize)#-1)
    ax[0][i].axvline(np.log(50),ls='--',color='grey',zorder=0)  
    ax[0][i].text(np.log(50),117,'50%',color='grey', rotation=-90, fontsize=fontsize)#-1)
    ax[0][i].axvline(np.log(25),ls='--',color='grey',zorder=0)
    ax[0][i].text(np.log(25),117,'25%',color='grey', rotation=-90, fontsize=fontsize)#-1)
    ax[0][i].axvline(np.log(10),ls='--',color='grey',zorder=0)  
    ax[0][i].text(np.log(10),117,'10%',color='grey', rotation=-90, fontsize=fontsize)#-1)


    ax[1][i].axvline(np.log(1),ls='--',color='grey',zorder=0)
    ax[1][i].text(np.log(1),120,'1%',color='grey', rotation=-90, fontsize=fontsize)#-1)
    ax[1][i].axvline(np.log(5),ls='--',color='grey',zorder=0)
    ax[1][i].text(np.log(5),120,'5%',color='grey', rotation=-90, fontsize=fontsize)#-1)
    ax[1][i].axvline(np.log(20),ls='--',color='grey',zorder=0)  
    ax[1][i].text(np.log(20),117,'20%',color='grey', rotation=-90, fontsize=fontsize)#-1)


  
  

#%%      

"""

   ...boostrapping...

"""

n_random_repeats = 1000

dBS_RES = {}

for i in [0,1]:
    
    dBS_RES[i] = {}
    
    subj, subject = dSubject[i]

    
    dfPARAMS = dRES[i]
    dfSIG_ = dfPARAMS.loc[ (dfPARAMS['sig']==1)]
    
    
    for q, DEG in zip([0,1],[False,True]):

        dBS_RES[i][DEG] = {}        

        dfSIG = dfSIG_.loc[(dfPARAMS['shuff']==DEG)]
    
        for j,KEY in enumerate(['svBL', 'KY']):

            v1 = dBS[i][DEG][KEY]['xdata']
            v2 = dBS[i][DEG][KEY]['ydata']
            
            #print(np.mean(dBS[i][DEG][KEY]['n_sig']))

            inds = np.arange(len(v1))
            n_size = len(v1)

            rs = []
            for repeat in range(n_random_repeats):
                #np.random.seed(42)
                ind_BS = np.random.choice(inds, size=n_size, replace=True)
                v1_BS = v1[ind_BS]
                v2_BS = v2[ind_BS]

                r,p,star = compute_corr(v1_BS, v2_BS)

                rs.append(r)

            dBS_RES[i][DEG][KEY] = rs

#%%

"""

##############################################################################

[Figure 7D] 

bootsrapped distrubtions

log of % shared variance & measured change in preferred direction
log of % contribution & measured change in preferred direction


##############################################################################

"""

fontsize=14

dADJ = {0:{True:{'svBL':0.1, 'KY':0.1},
           False:{'svBL':-0.1, 'KY':-0.1}},
        
        1:{True:{'svBL':0.1, 'KY':-0.1},
           False:{'svBL':-0.1, 'KY':0.1}}}

bins = np.linspace(-1,0.1,40)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8))
fig.subplots_adjust(hspace=0.15, wspace=0.1)

for i in [0,1]:
    
    subj, subject = dSubject[i]
    ax[0][i].set_title(f'{subject}', loc='left', fontsize=fontsize+2)
    
    for q, DEG in zip([0,1],[False,True]):
        
        color = palette_SHU[DEG]
        
        for j,KEY in enumerate(['svBL', 'KY']):
            
            axis = ax[j][i]
            axis.set_ylim([0,15])
            axis.set_xlim([-1,0.1])
            axis.set_xticks([-1.0, -0.75, -0.5, -0.25, 0])
     
    
            rs = dBS_RES[i][DEG][KEY]
            axis.hist(rs, bins=bins, color=color, edgecolor='w', lw=0.5, alpha=0.7, density=True)
            axis.axvline(0, color='grey',  ls='--',lw=1)#, ymin=0, ymax=(y_loc*0.9)/y_loc, zorder=0)

            axis.scatter(np.mean(rs), 11, color=color, s=50)
            
            x_adj = dADJ[i][DEG][KEY]
            axis.text(np.mean(rs)+x_adj,11.5, f'{np.mean(rs):.2f}', color=color, ha='center', va='bottom', fontsize=fontsize+2)
    
            if (i==0) and (KEY=='KY'):
                axis.plot([-0.68,-0.46],[11,11], color='k', lw=0.75, zorder=0)
            
            if (i==0) and (KEY=='svBL'):
                axis.plot([-0.75,-0.70],[11,11], color='grey', ls='-.', lw=0.75, zorder=0)
            
            if (i==1) and (KEY=='svBL'):
                axis.plot([-0.69,-0.58],[11,11], color='grey', ls='-.', lw=0.75, zorder=0)
    
            
            if (i==1) and (KEY=='KY'):
                axis.plot([-0.42,-0.37],[11,11], color='grey', ls='-.', lw=0.75, zorder=0)
               
         
         

ax[0][0].set_ylabel('density', fontsize=fontsize)
ax[1][0].set_ylabel('density', fontsize=fontsize)       
ax[1][0].set_xlabel('r$^2$', fontsize=fontsize)   
ax[1][1].set_xlabel('r$^2$', fontsize=fontsize)   

ax[0][0].text(-0.95,14, 'ln(%sv)', fontsize=fontsize) 
ax[0][1].text(-0.95,14,'ln(%sv)', fontsize=fontsize) 

ax[1][0].text(-0.95,14,'ln(%contribution)', fontsize=fontsize) 
ax[1][1].text(-0.95,14,'ln(%contribution)', fontsize=fontsize) 



legend_elements = [Line2D([0],[0], marker='s', markersize=8, color='k', lw=0, label='non-shuffled'),
                   Line2D([0],[0], marker='s', markersize=8, color=magenta, lw=0, label='shuffled')]#,
                   # Line2D([0],[0], marker='^', markersize=15, color='w', lw=0, markeredgecolor='k', label='applied'),
                   # Line2D([0],[0], marker='*', markersize=15, color='k', lw=0, markeredgecolor='k', label='mean')]
    

legend = ax[0][0].legend(handles=legend_elements, ncol=1, columnspacing=0.3, handletextpad=0.0, loc=(0.55,0.75), fontsize=fontsize-1)

frame = legend.get_frame()
frame.set_alpha(1)

#%%

for i in [0,1]:
    for KEY in ['svBL', 'KY']:
        v1 = dBS_RES[i][False][KEY]
        v2 = dBS_RES[i][True][KEY]
        
        TEST = []
        for j in range(len(v1)):
            if v1[j] < v2[j]:
                TEST.append(1)
            else:
                TEST.append(0)
                
        print(i,KEY, (n_random_repeats-np.sum(TEST))/n_random_repeats  )
                
                
            
#%%

for i in [0,1]:
    for DEG in [False,True]:
        for KEY in ['svBL', 'KY']:
            TEST = [dBS_RES[i][DEG][KEY][j] < 0 for j in range(n_random_repeats)]
            print(i, DEG, KEY, (n_random_repeats-np.sum(TEST))/n_random_repeats)
            
    # for KEY in ['svBL', 'KY']:
    #     v50 = dBS_RES[i][False][KEY]
    #     v90 = dBS_RES[i][True][KEY]
        
    #     TEST = []
    #     for j in range(n_random_repeats):
    #         TEST.append(v50[j] < v90[j])
    #     print(i, 'vs', KEY,  (n_random_repeats-np.sum(TEST))/n_random_repeats)
                


                
            
              