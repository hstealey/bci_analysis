# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:37:34 2025

@author: hanna
"""


"""

Main script for computing behavioral metrics over all sessions and subjects for BCI tasks
    Hard-coded subjects: 0 (Monkey A), 1 (Monkey B)
    Hard-coded BCI task perturbation types: rotation, shuffle
    Error clamp trials are excluded.


    Behavioral Metrics
        Trial Time (Time) & Cursor Path Length (Distance)
        Velocity
        Movement Error (ME) and Movement Variability (MV)
        Angular Error (AE)
        

pre-reqs: 
    bci_analysis/1_processing/main_process_hdf.py
        dDates_dDegs_HDF_{mode}.pkl
        df_{subj}_{mode}_{date}.pkl
        dfKG_{subj}_{mode}_{date}.pkl
        
    
    bci_analysis/1_processing/main_get_trial_inds.py
        trial_inds_BL-PE_{subj}_{mode}_{date}.pkl

"""
#from datetime import datetime

import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

# dSubject = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

# root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
# pickle_path = os.path.join(root_path, 'DEMO_pickles')


#start_time = datetime.now()

def run_main_behavior(root_path, pickle_path, modes, dSubject):
    
    '------------------------------'
    'Loading Custom Functions'
    os.chdir(os.path.join(root_path, 'functions', 'behavior_fxns'))
    from compute_behavior import compute_behavior
    

    window_len = 9 
    
    for mode in modes:
        
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, _ = pickle.load(open_file)
        open_file.close()
    
        
        for i in range(len(dSubject)):
            
            subj, subject = dSubject[i]
    
    
            for d, date in enumerate(dDates[i][:]):
    
        
                open_file = open(os.path.join(pickle_path, 'df', f'df_{subj}_{mode}_{date}.pkl'), "rb")
                df = pickle.load(open_file)
                open_file.close()
                
                open_file = open(os.path.join(pickle_path,'trial_inds', f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'), "rb")
                dfINDS_BL, dfINDS_PE, _ = pickle.load(open_file) #window_len
                open_file.close()
                
                        
    
                dfBEH_BL = compute_behavior(df, dfINDS_BL, window_len)
                dfBEH_PE = compute_behavior(df, dfINDS_PE, window_len)
    
                '''Saving values to .pkl'''     
                os.chdir(os.path.join(pickle_path,'dfBEH', 'compute_behavior_results'))
                obj_to_pickle = [dfBEH_BL, dfBEH_PE]
                filename = f'dfBEH_BL-PE_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()     
            
    
    # end_time = datetime.now()
    
    # elapsed_time = end_time - start_time
    
    # print(elapsed_time)



