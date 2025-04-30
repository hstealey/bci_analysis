# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:52:03 2025

@author: hanna
"""

import numpy as np
import pandas as pd 


def get_trial_inds(df, blockType, mode, n_sets):
    
    """
    Finds the putative trials (and their lengths) within a block for use in analysis.
    Ensures that the number of sets of 8 trials needed (n_sets) exist.
    
    NOTE: Error clamp trials are EXCLUDED.
        
    
    inputs:
        df: pandas dataframe, see bci_analysis\preprocessing\main_process_hdf for details
        blockType: int, indicates the block to process
            Possible values: 1: baseline (BL), 2: perturbation (PE), 3: perturbation 2 (PL), 4: washout (WO)
        mode: str, rotation perturbation type (i.e., 'rotation', 'shuffle')
        n_sets: the number of sets of 8 trials to include based on mode and blockType
    
    
    returns:
        None, None if the number of trials (for any target direction ('deg')) does not match the number of sets set in dN
        dfInds: pandas dataframe of trial numbers, target locations, number of samples within trial, 2D array of spike counts
            Sorted by trial number
        
        minimum number of samples on any given trial
        
    
    """

    dfBLOCK = df.loc[(df['errorClamp']==0) & (df['blockType']==blockType)]

    n_bins    = []
    trial_num = []
    trial_deg = []
    spikes    = []
    
    vx = []
    vy = []
    
    for j, deg in enumerate(np.arange(0,360,45)):
        
        if blockType==1:
            #Because the subject may not immediately recognize that the task has started immediately when the task is launched, we do not consider the first set of 8 trials (1 per target location).
            ind_degs = dfBLOCK.loc[dfBLOCK['target']==deg].index.tolist()[1:]
        else:
            ind_degs = dfBLOCK.loc[dfBLOCK['target']==deg].index.tolist()[:n_sets]
        
        if len(ind_degs) != n_sets:
            return(None, None)

        for k,ind in enumerate(ind_degs):
            n_bins.append(np.shape(dfBLOCK.loc[ind]['spikes'])[0])
            trial_num.append(ind)
            trial_deg.append(deg)
            spikes.append( np.array(dfBLOCK.loc[ind]['spikes']) )
            
            vx.append(np.array(dfBLOCK.loc[ind]['decoder_vx']))
            vy.append(np.array(dfBLOCK.loc[ind]['decoder_vy']))
    
    dfInds = pd.DataFrame({'tn': trial_num, 'deg': trial_deg, 'n': n_bins, 'vx': vx, 'vy': vy, 'spikes': spikes}).sort_values(by='tn').reset_index(drop=True)

    return(dfInds, np.min(dfInds['n'].values))






