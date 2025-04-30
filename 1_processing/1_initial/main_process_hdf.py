# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:36:49 2025

@author: hanna
"""

"""

PREPROCESSING STARTS WITH THIS FILE: ALL SUBSEQUENT PREPROCESSING RELIES ON THE RESULTS SAVED FROM THIS STEP

Main script for extracting behavioral data and related decoder information for each BCI session

    Required custom function: process_hdf (bci_analysis\functions\general_fxns\process_hdf\process_hdf.py)
    Loads behavioral (.hdf) files and decoder files (.pkl) and extracts
        df: pandas dataframe that contains behavioral information & aligned spike counts
        dfKG: pandas dataframe that contains the baseline and perturbation Kalman gain for each neuron and where it was rotated/shuffled

    Hard-coded path to save pickles: pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'
    Hard_coded subjects: 'airp', 'braz'
    Hard_coded modes: 'rotation', 'shuffle'

    For each date, a separate .pkl file is saved for df and dfK in the pickles folder.
        f'df_{subj}_{mode}_{date}.pkl'
        f'dfKG_{subj}_{mode}_{date}.pkl'
        
        subj: str, four-letter code for subject (i.e., 'airp' or 'braz')
        mode: str, BCI perturbation type (i.e., 'rotation' or 'shuffle')
        date: str, date of session in format YYYYMMDD (e.g., '20250205')

    
    For all sessions for all subjects in a perturbation type ('mode'), an additional file
    is saved that contains all of the dates and perturbation degrees.
        dDates, dDegs, dDegs2 || f'dDates_dDegs_HDF_{mode}.pkl'
        
        mode: str, BCI perturbation type (i.e., 'rotation' or 'shuffle')
        

            dDates: dict, contains a list of all dates processed (date: str, YYYYMMDD)
                keys: 0:'airp', 1:'braz'
            dDegs: dict, contains a numpy array of degree of perturbation 
                keys: 0:'airp', 1:'braz'
                
                If mode is 'rotation', then the list contains integers representing the 
                easy (50) or hard (90) rotation condition.
                    Note: does not indicate whether clockwise or counterclockwise (see 'rotation', dDegs2)
                
                If mode is 'shuffle', then the list contains floats representing the 
                fraction of neurons shuffled.
            
            dDegs2: dict, contains a numpy array of degree of perturbation that was entered into task GUI
                keys: 0:'airp', 1:'braz'
                
                If mode is 'rotation;, then the list contains integers representing the 
                easy (-50/50) or hard (-90/90) rotation condition. 
                    Note: does indicate whether clockwise or counterclockwise
              
                
                If mode is 'shuffle;, then the list contains the approximate percentage of
                neurons shuffled (30%, 50%). 
                    Note: precise amount varies by number of units (see 'shuffle', dDegs)
                    
            
"""

#%%

#from datetime import datetime

import os
import pickle

from tqdm import tqdm
from glob import glob

import numpy  as np
import pandas as pd

root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'

#Root path containing (specifically formatted) data folders.
# dRoot = {'rotation': r'C:\BMI_Rotation\Data',
#          'shuffle':  r'C:\BMI_Shuffle\Data'}

dRoot = {'rotation': os.path.join(root_path, 'DEMO_data', 'bci_rotation'), 
         'shuffle':  os.path.join(root_path, 'DEMO_data', 'bci_shuffle')}

#File path to save results. 
pickle_path = os.path.join(root_path, 'DEMO_pickles')


'------------------------------'
#Load custom function to process behavioral (.hdf) and decoder files (.pkl): process_hdf
os.chdir(os.path.join(root_path, 'functions', 'general_fxns'))
from process_hdf import process_hdf


#start_time = datetime.now()


dictRot2  = {'50': 50, '90': 90, 'neg50': 50, 'neg90': 90}
dictRot4  = {'50': 50, '90': 90, 'neg50':-50, 'neg90':-90}

dictShuff2 = {'30': 30, '50': 50}

for mode in ['rotation','shuffle']: 
    
    data_root_path_ = dRoot[mode]

    dDegs  = {}
    dDegs2 = {}
    dDates = {}
    
    for subject_ind, subj, subject in zip([0,1],['airp', 'braz'],['Airport', 'Brazos']):
      
        deg_list = []
        date_list = []
        
        data_root_path = os.path.join(data_root_path_, subject)
    
        os.chdir(data_root_path)
        degFolders = glob('*')
        
        for degFolder in degFolders:
            os.chdir(os.path.join(data_root_path, degFolder))
            dates = glob('*')
            
            for date in tqdm(dates):
                df, dfKG = process_hdf(os.path.join(data_root_path,degFolder,date), mode)
                deg_list.append(degFolder)
                date_list.append(date)
                
                os.chdir(os.path.join(pickle_path, 'df'))
                obj_to_pickle = df 
                filename = f'df_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
                
                os.chdir(os.path.join(pickle_path, 'dfKG'))
                obj_to_pickle = dfKG
                filename = f'dfKG_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()

            print(f'Complete: {subject}, {mode}, {degFolder}')
                

        if mode == 'rotation':
            deg_list1 = np.array([dictRot2[deg] for deg in deg_list])
            deg_list2 = np.array([dictRot4[deg] for deg in deg_list])
        
            dDegs[subject_ind]  = deg_list1
            dDegs2[subject_ind] = deg_list2

        if mode == 'shuffle':
            deg_list1 = []
            for date in date_list:  
                open_file = open(os.path.join(pickle_path,'dfKG',f'dfKG_{subj}_{mode}_{date}.pkl'), "rb")
                dfKG = pickle.load(open_file)
                open_file.close()
                deg_list1.append(np.sum(dfKG['shuffled'])/len(dfKG))
                
            deg_list2 = np.array([dictShuff2[deg] for deg in deg_list])
            
            dDegs[subject_ind]  = np.array(deg_list1)
            dDegs2[subject_ind] = deg_list2
            
        dDates[subject_ind] = np.array(date_list)
     
      
    '''Saving values to .pkl'''     
    os.chdir(pickle_path)
    obj_to_pickle = [dDates, dDegs, dDegs2]
    filename = f'dDates_dDegs_HDF_{mode}.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()


# end_time = datetime.now()

# elapsed_time = end_time - start_time

# print(elapsed_time)
    
