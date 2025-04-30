# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:18:45 2025

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

"""Plotting Parameters"""
fontsize = 12
mpl.rcParams["font.size"] = fontsize
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.spines.right"] = "False"
mpl.rcParams["axes.spines.top"] = "False"
mpl.rcParams["axes.spines.left"] = "False"
mpl.rcParams["axes.spines.bottom"] = "False"


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


root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'pickles')  




def plot_cursor(axis, x, y):
    axis.scatter(x,y, color='grey', alpha=0.8, edgecolor='k', s=20, zorder=1)
    axis.plot(x,y, color='grey', alpha=0.6, zorder=0)
    return()




fontsize = 20

fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for mode in ['rotation']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, _ = pickle.load(open_file)
    open_file.close()

    for i in [0]:

        subj, subject = dSubject[i]

        for j,d in zip([0],[17]): #15#4/36, 14/42
            
            date = dDates[i][d]
  
            open_file = open(os.path.join(pickle_path, 'df', f'df_{subj}_{mode}_{date}.pkl'), "rb")
            df = pickle.load(open_file)
            open_file.close()
            
            
            dfBL = df.loc[ (df['blockType']==1) & (df['errorClamp']==0)]
            dfPE = df.loc[ (df['blockType']==2) & (df['errorClamp']==0)]
            
            tnBL = []
            tnBL_late = []
            tnPE = []
            tnPE_late = []
            for deg in np.arange(0,360,45):
                tnBL.append(dfBL.loc[dfBL['target']==deg].index.tolist()[1])
                tnBL_late.append(dfBL.loc[dfBL['target']==deg].index.tolist()[10])
                tnPE.append(dfPE.loc[dfPE['target']==deg].index.tolist()[0])
                tnPE_late.append(dfPE.loc[dfPE['target']==deg].index.tolist()[30])
                
            axis1 = ax[j][0]
            axis2 = ax[j][1]
            axis3 = ax[j][2]
            axis4 = ax[j][3]
            
            if j == 0:
                axis1.set_title('early baseline trials', loc='center', fontsize=fontsize)
                axis2.set_title('late baseline trials', loc='center', fontsize=fontsize)
                axis3.set_title('early perturbation trials', loc='center', fontsize=fontsize)
                axis4.set_title('late perturbation trials', loc='center', fontsize=fontsize)
                
                
            for axis, tns, dfK in zip([axis1, axis2, axis3, axis4], [tnBL, tnBL_late, tnPE, tnPE_late], [dfBL, dfBL, dfPE, dfPE]):
                

                for tn in tns:
                    #yes, this "reversal" is correct.
                    x = dfK.loc[tn]['decoder_py']
                    y = dfK.loc[tn]['decoder_px']
                    plot_cursor(axis, x, y)
                
                
                for deg in np.arange(0,360,45):
                    
                    rad = deg*(np.pi/180)
                    x_loc = 10*np.cos(rad)
                    y_loc = 10*np.sin(rad)
                    
                    axis.scatter(x_loc, y_loc, color='r', alpha=0.3, zorder=0, s=600)
                
                axis.scatter(0,0, color='r', alpha=0.2, zorder=0, s=600)




for mode in ['shuffle']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, _ = pickle.load(open_file)
    open_file.close()

    for i in [0]:

        subj, subject = dSubject[i]

        for j,d in zip([1],[26]):
        #for d in [26]:#range(len(dDates[i])):
            
            date = dDates[i][d]
  
            open_file = open(os.path.join(pickle_path, 'df', f'df_{subj}_{mode}_{date}.pkl'), "rb")
            df = pickle.load(open_file)
            open_file.close()
            
            
            dfBL = df.loc[ (df['blockType']==1) & (df['errorClamp']==0)]
            dfPE = df.loc[ (df['blockType']==2) & (df['errorClamp']==0)]
            
            tnBL = []
            tnBL_late = []
            tnPE = []
            tnPE_late = []
            for deg in np.arange(0,360,45):
                tnBL.append(dfBL.loc[dfBL['target']==deg].index.tolist()[1])
                tnBL_late.append(dfBL.loc[dfBL['target']==deg].index.tolist()[10])
                tnPE.append(dfPE.loc[dfPE['target']==deg].index.tolist()[0])
                tnPE_late.append(dfPE.loc[dfPE['target']==deg].index.tolist()[30])
           
            timeBL = []
            for tn in tnBL_late:
                #yes, this "reversal" is correct.
                time = dfBL.loc[tn]['trial_time']
                timeBL.append(time)
           
            
            timePE = []
            for tn in tnPE:
                #yes, this "reversal" is correct.
                time = dfPE.loc[tn]['trial_time']
                timePE.append(time)
                

            axis1 = ax[j][0]
            axis2 = ax[j][1]
            axis3 = ax[j][2]
            axis4 = ax[j][3]
            
            if j == 0:
                axis1.set_title('early baseline trials', loc='center', fontsize=fontsize)
                axis2.set_title('late baseline trials', loc='center', fontsize=fontsize)
                axis3.set_title('early perturbation trials', loc='center', fontsize=fontsize)
                axis4.set_title('late perturbation trials', loc='center', fontsize=fontsize)
                
                
            for axis, tns, dfK in zip([axis1, axis2, axis3, axis4], [tnBL, tnBL_late, tnPE, tnPE_late], [dfBL, dfBL, dfPE, dfPE]):
                

                for tn in tns:
                    #yes, this "reversal" is correct.
                    x = dfK.loc[tn]['decoder_py']
                    y = dfK.loc[tn]['decoder_px']
                    plot_cursor(axis, x, y)
                
                
                for deg in np.arange(0,360,45):
                    
                    rad = deg*(np.pi/180)
                    x_loc = 10*np.cos(rad)
                    y_loc = 10*np.sin(rad)
                    
                    axis.scatter(x_loc, y_loc, color='r', alpha=0.3, zorder=0, s=600)
                
                axis.scatter(0,0, color='r', alpha=0.2, zorder=0, s=600)

axis.set_xlim([-16,16])
axis.set_ylim([-16,16])   
axis.set_xticks([])
axis.set_yticks([])

ax[0][0].set_ylabel('rotation (90°)', fontsize=fontsize)
ax[1][0].set_ylabel('shuffle', fontsize=fontsize)              
                 
            