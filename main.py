# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:56:53 2025

@author: hanna
"""

"""

Main script for executing all processing scripts on (sample) data


    [Step 1] process behavioral+neural data files (.hdf) & decoder files (.pkl).
    [Step 2] determine trial lengths for each trial
    [Step 3] compute behavioral metrics
    [Step 4] all steps for factor analysis (neural metric: %sv)
    [Step 5] fit tuning curves


    Results from each step are saved in pickle files (.pkl).
    
    [Example] How to open .pkl files:
        file_name = f'dDates_dDegs_HDF_{mode}.pkl'
        open_file = open(os.path.join(pickle_path, file_name), "rb")
        dDates, dDegs, dDegs2 = pickle.load(open_file)
        open_file.close()
        





"""




import os

root_path = os.path.dirname( os.path.abspath(__file__) )
os.chdir(root_path)


#Root path containing (specifically formatted) data folders.
dRoot = {'rotation': os.path.join(root_path, 'data', 'bci_rotation'), 
          'shuffle': os.path.join(root_path, 'data', 'bci_shuffle')}

#File path to save results. 
pickle_path = os.path.join(root_path, 'results')#'DEMO_pickles')


modes = ['rotation', 'shuffle']
dSubject = {0:['airp', 'Airport'], 1:['braz', 'Brazos']}




#%%

"""
Step 1: process behavioral+neural data files (.hdf) & decoder files (.pkl).
    output: extracted data in pandas dataframe format, saved as .pkl
"""

from processing.initial.main_process_hdf import run_main_process_hdf
run_main_process_hdf(root_path, dRoot, pickle_path, modes, dSubject)


#%%

"""
Step 2: determine trial lengths for each trial
    output: extracted data in pandas dataframe format, saved as .pkl
"""

from processing.initial.main_get_trial_inds import run_main_get_trial_inds
run_main_get_trial_inds(root_path, pickle_path, modes, dSubject)


#%%

"""
Step 3: compute behavioral metrics
    output: pandas dataframe, saved as .pkl
"""

from processing.behavior.main_behavior import run_main_behavior
run_main_behavior(root_path, pickle_path, modes, dSubject)



#%%

"""
Step 4: all steps for factor analysis (single model per session fit on baseline block data)
    output: multiple .pkl files


    [1] data formatting
    [2] cross-validation
    [3] fit models
    [4] compute variance metrics
    


"""

from processing.FA_tuning.main_FA_TTT_for_tuning import run_main_FA_TTT
run_main_FA_TTT(root_path, pickle_path, modes, dSubject)


#%%


"""
Step 5: fit (standard cosine) tuning curve
    output: multiple .pkl files
        
        
    Sections of code:
    [1] generate bootstrapped means
    [2] fit cosine tuning curves
    [3] compute changes in preferred direction (measured vs "assinged" by decoder gain)

    
"""

from processing.tuning.main_tuning_curve import run_tuning_curve
run_tuning_curve(root_path, pickle_path, modes, dSubject)










