# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:08:08 2025

@author: hanna
"""


"""

Shared Space Alignment (SSA)


    "We used the “shared space alignment” to measure the similarity between the 
    shared variance (or main shared variance) of Epoch A and Epoch B.
    
    The shared space alignment is the fraction of epoch A shared variance 
    captured in epoch B’s shared space and thus ranges from 0 to 1. 
    
        For some geometric intuition, in the one-dimensional case 
        (i.e., rank⁡(𝛴𝐴,shared)=rank⁡(𝛴𝐵,shared)=1), the space alignment is equivalent to cos⁡𝜃,
        where 𝜃 is the angle between epoch A and epoch B’s one-dimensional shared space. 
    
    We note that the shared space alignment is asymmetric when shared 
    dimensionality is greater than 1, such that alignment of A with B 
        ---need not be equal---   
   to the alignment of B with A." 


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
palette_ROT4 = {-50: yellow, 50: orange, -90: blue, 90: purple}

dSubject  = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

          
pickle_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\pickles'   
path_ext = r'FA\TTT\fixed_window_9\FA5\eachNF\FA6'       
            
pickle_save_path = os.path.join(pickle_path, path_ext)

if os.path.exists(pickle_save_path) == True:
    print(pickle_save_path)
else:
    print('SAVE PATH DOES NOT EXIST!')

'------------------------------'
'Loading Custom Functions'
custom_functions_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis\functions'
os.chdir(os.path.join(custom_functions_path, 'neural_fxns'))
from shared_space_alignment_fxns import calcSSA
# os.chdir(os.path.join(custom_functions_path, 'general_fxns'))
# from compute_stats import compute_one_sample_ttest, compute_two_sample_ttest, compute_corr, stats_star




#%%

"""

Shared Space Alignment

"""


for mode in ['rotation', 'shuffle']:

    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    dSSA = {}

    for i in [0,1]:
        
        dSSA[i] = {'BL_to_PE':np.zeros(len(dDates[i])),
                   'PE_to_BL':np.zeros(len(dDates[i]))}
        
        subj,subject = dSubject[i]
        
        for d, date in enumerate(dDates[i][:]):
        
            open_file = open(os.path.join(pickle_save_path, f'FA6_dVAR_{subj}_{mode}_{date}.pkl'), "rb")
            _, dVAR = pickle.load(open_file)
            open_file.close()
            
            'SSA | BL_to_PE: the fraction of BL shared varianace explained by the column space of PE'
            dSSA[i]['BL_to_PE'][d] = calcSSA(dVAR['BL']['loadings'].T , dVAR['PE']['loadings'].T )
            
            'SSA | PE_to_BL: the fraction of PE shared varianace explained by the column space of BL'
            dSSA[i]['PE_to_BL'][d] = calcSSA(dVAR['PE']['loadings'].T , dVAR['BL']['loadings'].T )
            

    '''Saving values to .pkl'''     
    os.chdir( pickle_save_path )
    obj_to_pickle = dSSA
    filename = f'dSSA_{mode}.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()


