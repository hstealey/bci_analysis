# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:52:01 2025

@author: hanna
"""

# from datetime import datetime

import os
import pickle
import numpy  as np

from tqdm import tqdm



def run_main_get_trial_inds(root_path, pickle_path, modes, dSubject):

    #File path to save results. 
    # root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
    # pickle_path = os.path.join(root_path, 'DEMO_pickles')
    #dSubject = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}
    
    '------------------------------'
    'Loading Custom Functions'
    os.chdir(os.path.join(root_path, 'functions', 'general_fxns'))
    from get_trial_inds import get_trial_inds
    
    
    
    dN = {'rotation': {1:41,2:41}, 'shuffle': {1:20,2:41}}
    
    
    # start_time = datetime.now()
    
    
    
    """
    ###############################################################################
    
    Pull trial numbers and window length to be used for analysis.
    
    ###############################################################################
    """
    
    for mode in modes:
        
        FAILED = {}
        
        dLEN = {}
        dMIN = {}
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, _, _ = pickle.load(open_file)
        open_file.close()
        
        for i in range(len(dSubject)):
            
            FAILED[i] = []
            
            dLEN[i] = {}
            dMIN[i] = {}
            
            subj, subject = dSubject[i]
    
            for date in tqdm(dDates[i]):
    
                open_file = open(os.path.join(pickle_path,'df',f'df_{subj}_{mode}_{date}.pkl'), "rb")
                df = pickle.load(open_file)
                open_file.close()
                
                dfBL, minBL = get_trial_inds(df, 1, mode, dN[mode][1])
                dfPE, minPE = get_trial_inds(df, 2, mode, dN[mode][2])
                
                dMIN[i][date] = {'BL': minBL, 'PE': minPE, 'all': np.min([minBL, minPE])}
                
                dLEN[i][date] = {}
                dLEN[i][date]['nPL'] = len(df.loc[df['blockType']==3])
                dLEN[i][date]['nWO'] = len(df.loc[df['blockType']==4])
                
                
                if (minBL is None) or (minPE is None):
                    FAILED[i].append(date)
    
                else:
                    os.chdir(os.path.join(pickle_path,'trial_inds'))
                    filename = f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'
                    obj_to_pickle = [dfBL, dfPE, np.min([minBL, minPE])]
                    open_file = open(filename, "wb")
                    pickle.dump(obj_to_pickle, open_file)
                    open_file.close()
                    
    
        os.chdir(os.path.join(pickle_path,'trial_inds'))
        filename = f'trial_inds_FAILED_dMIN-dLEN_{mode}.pkl'
        obj_to_pickle = [FAILED, dMIN, dLEN]
        open_file = open(filename, "wb")
        pickle.dump(obj_to_pickle, open_file)
        open_file.close()
    
    
    
    
    
    """
    #####################################################################################################
    
    Check for Failures
        Did any of the sessions fail to meet the threshold for number of sets of trials (defined in dN)?
    
    #####################################################################################################
    """
    
    
    for mode in modes: #['rotation', 'shuffle']:
       
        fn = f'trial_inds_FAILED_dMIN-dLEN_{mode}.pkl'
        open_file = open(os.path.join(pickle_path, 'trial_inds', fn), "rb")
        FAILED, _, _ = pickle.load(open_file) #dMIN, dLEN
        open_file.close()
    
        for i in [0,1]:
            
            subj, subject = dSubject[i]
            
            if len(FAILED[i]) != 0:
                print(f'FAILED: {subject} | {mode} | {FAILED[i]}')
            else:
                print(f'passed: {subject} | {mode} | {FAILED[i]}')
            
    
    
    # end_time = datetime.now()
    
    # elapsed_time = end_time - start_time
    
    # print(elapsed_time)
    
