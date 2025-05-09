# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:49:03 2025

@author: hanna
"""



"""

FACTOR ANALYSIS || Trial-to-Trial (TTT) Models || single model per session (baseline (BL) block data only)
   Spike counts summed over a fixed window of 200ms to 600ms for each neuron on each trial.

    [1] data formatting
    [2] cross-validation
    [3] fit models
    [4] compute variance metrics
    
    

pre-reqs: 
    bci_analysis/1_processing/main_process_hdf.py
        dDates_dDegs_HDF_{mode}.pkl
        df_{subj}_{mode}_{date}.pkl
        dfKG_{subj}_{mode}_{date}.pkl
        
    
    bci_analysis/1_processing/main_get_trial_inds.py
        trial_inds_BL-PE_{subj}_{mode}_{date}.pkl


"""


# from datetime import datetime

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


# dSubject = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

# root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
# pickle_path = os.path.join(root_path, 'DEMO_pickles')


def run_main_FA_TTT(root_path, pickle_path, modes, dSubject):


    '------------------------------'
    'Loading Custom Functions'
    os.chdir(os.path.join(root_path, 'functions', 'neural_fxns'))
    from factor_analysis_fxns import perform_cv, fit_fa_model
    
    
    firing_rate_threshold = 1 #Hz 
    
    dN = {'rotation': {1:20, 2:20}, 'shuffle': {1:20, 2:20}}
    
    window_start = 2
    window_end   = 6 #non-inclusive
    n_window = window_end - window_start
    
    
    pickle_root_save_path = os.path.join(pickle_path, 'FA_tuning')
    
    if os.path.exists(pickle_root_save_path) == False:
        print("FILE SAVE PATH DOES NOT EXISTS")
    else:
        print(pickle_root_save_path)
    
    
    
    
    """
    ###############################################################################
    
    [1] Data Formatting
    
    ###############################################################################
    """
    
    # start_time = datetime.now()
    
    
    
    for mode in modes: #['rotation', 'shuffle']:
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, _, _ = pickle.load(open_file)
        open_file.close()
        
        n_sets_BL = dN[mode][1]
        n_sets_PE = dN[mode][2]
        
        n_trials_BL = n_sets_BL*8
        n_trials_PE = n_sets_PE*8
    
        for i in range(len(dSubject)):#[0,1]:
            
            subj, subject = dSubject[i]
    
            for date in tqdm(dDates[i][:]):
    
                fn = f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_path,'trial_inds',fn), "rb")
                dfBL, dfPE, _ = pickle.load(open_file) #window_len
                open_file.close()
                
                dNU = {}
                dSC_pre  = {'BL':{}, 'PE':{}, 'mBL':{}, 'sBL':{}, 'vBL':{}, 'mPE':{}, 'sPE':{}, 'vPE':{}}
                dSC      = {'BL':{}, 'PE':{}, 'mBL':{}, 'sBL':{}, 'vBL':{}, 'mPE':{}, 'sPE':{}, 'vPE':{}}
    
                tnBL = np.zeros((8,n_sets_BL))
                tnPE = np.zeros((8,n_sets_PE))
                for j, deg in enumerate(np.arange(0,360,45)):
                
                    tnBL[j,:] = dfBL.loc[dfBL['deg']==deg, 'tn'].values[-n_sets_BL:]
                    tnPE[j,:] = dfPE.loc[dfPE['deg']==deg, 'tn'].values[-n_sets_PE:]
                    
                
                nU_pre = np.shape(dfBL.loc[dfBL['tn']==tnBL[0,0], 'spikes'].values[0])[1]
                dNU['nU_pre'] = nU_pre
                
                units_to_del = []
       
                
                
                'Baseline'
                tnBL_all = np.sort(np.concatenate((tnBL)))
                sc = dfBL.loc[dfBL['tn']==tnBL_all[0], 'spikes'].values[0][window_start:window_end,:]  
                for tn in tnBL_all[1:]:
                      spikes_tn = dfBL.loc[dfBL['tn']==tn, 'spikes'].values[0][window_start:window_end,:]
                      sc = np.vstack((sc, spikes_tn))
                     
                      if np.shape(spikes_tn)[0] != n_window:
                          print(mode, subject, date, 'BL', ti)
                         
                         
                '-summed spike counts over the window (for fitting tuning curves)'         
                sc_summed = np.zeros((n_trials_BL, nU_pre)) 
                for ti, tn in enumerate(tnBL_all):
                      sc_summed[ti,:] = np.sum(dfBL.loc[dfBL['tn']==tn, 'spikes'].values[0][window_start:window_end,:], axis=0)
    
    
                dSC_pre['BL']  = sc_summed
                dSC_pre['mBL'] = np.mean( sc_summed/(n_window*0.1), axis=0)
                dSC_pre['target_degs_BL'] = np.array([dfBL.loc[dfBL['tn']==tn, 'deg'].values[0] for tn in  tnBL_all]).astype(int)
              
                units_to_del.append(np.where( dSC_pre['mBL'] < firing_rate_threshold)[0])              
    
    
    
    
    
                'Perturbation'
                tnPE_all = np.sort(np.concatenate((tnPE)))
                sc = dfPE.loc[dfPE['tn']==tnPE_all[0], 'spikes'].values[0][window_start:window_end,:]  
                for tn in tnPE_all[1:]:
                      spikes_tn = dfPE.loc[dfPE['tn']==tn, 'spikes'].values[0][window_start:window_end,:]
                      sc = np.vstack((sc, spikes_tn))
                     
                      if np.shape(spikes_tn)[0] != n_window:
                          print(mode, subject, date, 'PE', ti)
                         
                         
                '-summed spike counts over the window (for fitting tuning curves)'         
                sc_summed = np.zeros((n_trials_PE, nU_pre)) 
                for ti, tn in enumerate(tnPE_all):
                      sc_summed[ti,:] = np.sum(dfPE.loc[dfPE['tn']==tn, 'spikes'].values[0][window_start:window_end,:], axis=0)
    
    
                dSC_pre['PE']  = sc_summed
                dSC_pre['mPE'] = np.mean( sc_summed/(n_window*0.1), axis=0)
                dSC_pre['target_degs_PE'] = np.array([dfPE.loc[dfPE['tn']==tn, 'deg'].values[0] for tn in  tnPE_all]).astype(int)
              
                
                units_to_del.append(np.where( dSC_pre['mPE'] < firing_rate_threshold)[0])              
    
            
    
                'Remove neurons with average firing rates below the firing rate threshold (Hz) during any set of trials.'
                delU = np.unique(np.concatenate((units_to_del)))
    
    
    
                'Baseline'
                sc_ = np.delete(dSC_pre['BL'], delU, axis=1)
                sc  = sc_#stats.zscore(sc_,axis=0)
                dSC['BL']  = sc
                dSC['target_degs_BL'] = dSC_pre['target_degs_BL']
        
                
                'Perturbation'
                sc_ = np.delete(dSC_pre['PE'], delU, axis=1)
                sc  = sc_#stats.zscore(sc_,axis=0)
                dSC['PE']  = sc
                dSC['target_degs_PE'] = dSC_pre['target_degs_PE']
    
    
    
                dNU['nU_post'] = np.shape(dSC['BL'])[1]
                  
                
                dParams1 = {'n_sets_BL': n_sets_BL, 'n_sets_PE': n_sets_PE,  
                            'n_trials_BL': n_trials_BL, 'n_trials_PE': n_trials_PE,
                            'n_window': n_window, 'window_start': window_start, 'window_end': window_end}
                
           
                os.chdir(os.path.join(pickle_root_save_path,'FA1'))
                filename = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                obj_to_pickle = [delU, dNU, dParams1, np.sort(np.concatenate((tnBL))), np.sort(np.concatenate((tnPE)))] 
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
                
                os.chdir(os.path.join(pickle_root_save_path,'FA2'))
                filename = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
                obj_to_pickle = [dSC_pre, dSC]
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
                
    
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
    
    
    
    """
    ###############################################################################
    
    [2] Cross Validation
    
    ###############################################################################
    """
    
    # start_time = datetime.now()
    
    
    
    max_n_components = 9 #inclusive
    n_splits  = 10
    n_repeats = 10 #20
    
    for mode in modes: #['rotation', 'shuffle']:
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, _, _ = pickle.load(open_file)
        open_file.close()
    
        for i in range(len(dSubject)):
            
            subj, subject = dSubject[i]
    
            for date in tqdm(dDates[i][:]):
                
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, tnBL_all, tnPE_all = pickle.load(open_file)
                open_file.close()
                
                fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
                dSC_pre, dSC = pickle.load(open_file)
                open_file.close()
    
    
                dNF = {}
                dFAscores = {'BL':{}, 'PE':{}}
                
                for blockKey in ['BL', 'PE']:
                
                    dNF[blockKey], dFAscores[blockKey]['mean'], dFAscores[blockKey]['all']= perform_cv(dSC[blockKey], np.arange(0,max_n_components+1,1), n_splits, n_repeats, False)
                
                    if dNF[blockKey] == max_n_components:
                        dNF[f'd_{blockKey}_REDO'] = 'FAIL'
                    else:
                        dNF[f'd_{blockKey}_REDO'] = 'pass'
                        
                
    
                dParams2 = {'firing_rate_threshold_Hz': firing_rate_threshold, 'max_n_components': max_n_components, 'n_splits': n_splits, 'n_repeats': n_repeats}
    
                os.chdir(os.path.join(pickle_root_save_path, 'FA3'))
                filename = f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
                obj_to_pickle = [dNF, dParams2]
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
                
                
                os.chdir(os.path.join(pickle_root_save_path, 'FA4'))
                filename = f'FA4_scores_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
    
    
                print(dNF)
    
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
    
    
    
    
    
    
    """
    ###############################################################################
    
    [2a] Cross-Validation Checks
    
    CHECKS
        [#1] Determine if the max number of components could be greater than the number of components tested (i.e., nf==max number of compnents tested).
        [#2] Determine if  the number of samples per training model fit was sufficient (>#neurons).
    
    ###############################################################################
    """
    
    
    dREDO = {}
    
    
    for mode in modes: #['rotation', 'shuffle']:
        
        dREDO[mode] = {}
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, _, _ = pickle.load(open_file)
        open_file.close()
    
        for i in range(len(dSubject)):
            
            dREDO[mode][i] = []
            
            subj, subject = dSubject[i]
    
            for date in dDates[i]:
                
                fn = f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA3', fn), "rb")
                dNF, dParams2 = pickle.load(open_file)
                open_file.close()
                
    
                'Check #1'
                pass_check_BL = dNF['d_BL_REDO']
                pass_check_PE = dNF['d_PE_REDO']
                
                
                if ( pass_check_BL == 'FAIL') or ( pass_check_PE == 'FAIL' ):
                    print(f'REDO: max_n_components: {mode}, {subj}, {date} || {pass_check_BL}, {pass_check_PE}')  
                    dREDO[mode][i].append(date)
                
                'Check #2'
                n_splits = dParams2['n_splits']
                
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, _,_ = pickle.load(open_file)
                open_file.close()
                
                n_trials = dParams1['n_trials_BL'] #TODO: change if not supposed to be the same
                
                if dNU['nU_post'] > (n_trials)*((n_splits-1)/n_splits):
                      print(f'REDO: ratio <<: {mode}, {subj}, {date}') 
                      dREDO[mode][i].append(date)
                      
            dREDO[mode][i] = np.unique(dREDO[mode][i])
    
    print(dREDO)
                
    
    
    """
    ###############################################################################
    
    [3] Fit FA Models
    [4] Compute Variance Metrics
    
    ###############################################################################
    
    """
    
    # start_time = datetime.now()
    
    
    
    
    dAllNF = {} 
    
    for mode in modes: #['rotation', 'shuffle']:
        
        dAllNF[mode] = {}
        
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, dDegs2 = pickle.load(open_file)
        open_file.close()
        
        for i in range(len(dSubject)):
            
            dAllNF[mode][i] = {'BL': np.zeros(len(dDates[i])),  'PE': np.zeros(len(dDates[i]))}
            
            subj, subject = dSubject[i]
    
            for d in tqdm(range(len(dDates[i][:]))):
    
                
                date = dDates[i][d]
    
                dVAR = {}
                  
                'Load Data'
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, _,_  = pickle.load(open_file)
                open_file.close()
                
                nU_pre, nU = dNU['nU_pre'], dNU['nU_post']
     
                fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
                dSC_pre, dSC = pickle.load(open_file)
                open_file.close()
                
                if (np.shape(dSC['BL'])) != (np.shape(dSC['PE'])):
                    print(f'{subj}, {mode}, {d}, {date}: FAILED CHECK: INPUT (SC) DIMENSIONS ARE NOT THE SAME')
            
                
                fn = f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'FA3', fn), "rb")
                dNF, dParams2 = pickle.load(open_file)
                open_file.close()
                
        
                if (dNF['d_BL_REDO'] == 'FAIL') or (dNF['d_PE_REDO'] == 'FAIL'):
                    print(f'{subj}, {mode}, {d}, {date}: FAILED CHECK: NEED TO REPEAT CV')
    
                    
                    
                dAllNF[mode][i]['BL'][d] = dNF['BL']
                dAllNF[mode][i]['PE'][d] = dNF['PE']
                
                
                
                """
                
                Fit FA models
                
                """
                
                for key in ['BL', 'PE']:
    
                    dVAR[key] = {}
                    
         
                    
                    _, _, _, loadings_, _, _ = fit_fa_model(dSC[key], dNF[key]) 
                    
                    
                    # test1 = np.shape(dSC[key])[1]
                    # test2 = nU
    
    
                    # if test1 != test2:
                    #     print('ERROR')
    
                    
                    'Compute "shared dimensionality" the number of factors needed to explain > 95% of the shared variance explained by the model fit using the number of factors identified from the CV.'
                    U = loadings_.T
                    cov_shared = np.dot(U,U.T)
                    rank = np.linalg.matrix_rank(cov_shared) #also just number of factors
                    _, S, _ = np.linalg.svd(cov_shared)
                    cS = np.cumsum(S[:rank])/np.sum(S[:rank])
                    nf95 = np.where(cS > 0.95)[0][0] +1 #add one to account for factor 1 index is 0
               
     
                    'Refit model & re-compute metrics'
                    total_, shared_, private_, loadings_, MAT_shared_, MAT_private_ = fit_fa_model(dSC[key], nf95) 
                    
                    '--orthonormalize loadings---'
                    rank = nf95
                    U = loadings_.T
                    shared_cov = np.dot(U,U.T)
                    U_, S_, VT_ = np.linalg.svd(shared_cov)
                    U = U_[:,:rank]
    
    
      
                    dVAR[key]['nf']          = nf95
                    dVAR[key]['total']       = total_
                    dVAR[key]['shared']      = shared_
                    dVAR[key]['private']     = private_
                    dVAR[key]['loadings']    = loadings_
                    dVAR[key]['U']           = U
                    dVAR[key]['MAT_shared']  = MAT_shared_
                    dVAR[key]['MAT_private'] = np.diag(MAT_private_)
                    dVAR[key]['%sv']         = np.diag(MAT_shared_)/( np.diag(MAT_shared_ )+ np.diag(MAT_private_))
                    
                    # shared_cov  = np.diag(np.dot(U,U.T))
                    # private_cov = np.diag(MAT_private_)/(np.mean(dSC[key]/(n_window*0.1), axis=0))
                    
                    # dVAR[key]['%sv_norm'] = shared_cov/(shared_cov+private_cov)
                    
                    # mBL = np.mean(dSC[key]/(n_window*0.1), axis=0)
                    # svNORM = shared_cov/(shared_cov+private_cov)
                    
    
                os.chdir(os.path.join(pickle_root_save_path,'FA5')) 
                filename = f'FA5_dVAR_{subj}_{mode}_{date}.pkl'
                obj_to_pickle = dVAR
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
         
    
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
