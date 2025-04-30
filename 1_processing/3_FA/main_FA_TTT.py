# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:02:24 2025

@author: hanna
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:49:03 2025

@author: hanna
"""



"""

FACTOR ANALYSIS || Trial-to-Trial (TTT) Models || single model per block (baseline (BL), perturbation (PE))
    One model is fit per block (2 models per session).
    Spike counts summed over a fixed window of 900ms from each trial.


    [1] data formatting
    [2] cross-validation
    [3] fit models
    [4] compute variance metrics
    

pre-reqs: 
    bci_analysis/1_processing/main_process_hdf.py
        f'dDates_dDegs_HDF_{mode}.pkl'
        f'df_{subj}_{mode}_{date}.pkl'
        f'dfKG_{subj}_{mode}_{date}.pkl'
        
    
    bci_analysis/1_processing/main_get_trial_inds.py
        f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'


"""

#%%

import datetime

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



dSubject = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}

root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
pickle_path = os.path.join(root_path, 'DEMO_pickles')


'------------------------------'
'Loading Custom Functions'
os.chdir(os.path.join(root_path, 'functions', 'neural_fxns'))
from factor_analysis_fxns import perform_cv, fit_fa_model


firing_rate_threshold = 1 #Hz 

window_len = 9 #900ms

dN = {'rotation': 34, 'shuffle': 18} ##dN = {'rotation': {1:41, 2:41}, 'shuffle': {1:20, 2:41}}


pickle_root_save_path = os.path.join(pickle_path, 'FA')

if os.path.exists(pickle_root_save_path) == False:
    print("FILE SAVE PATH DOES NOT EXISTS")
else:
    print(pickle_root_save_path) 
    
    


#%%

"""
###############################################################################

[1] Data Formatting

FA1: delU, dNU, dParams1, tnBL_all, tnPE_all  > f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
FA2: dSC_pre, dSC > f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'

###############################################################################
"""

#start_time = datetime.now()



for mode in ['rotation']:#,'shuffle']:

    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, _, _ = pickle.load(open_file)
    open_file.close()
    
    n_sets = dN[mode]
    n_trials = n_sets*8

    for i in [0,1]:

        subj, subject = dSubject[i]
        
        for d, date in enumerate(tqdm(dDates[i][:])): 
        
            fn = f'trial_inds_BL-PE_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_path,'trial_inds',fn), "rb")
            dfBL, dfPE, _ = pickle.load(open_file) #window_len
            open_file.close()
            
            dNU = {}
            dSC_pre  = {'BL':{}, 'PE':{}, 'mBL':{}, 'sBL':{}, 'vBL':{}, 'mPE':{}, 'sPE':{}, 'vPE':{}}
            dSC      = {'BL':{}, 'PE':{}, 'mBL':{}, 'sBL':{}, 'vBL':{}, 'mPE':{}, 'sPE':{}, 'vPE':{}}

            tnBL = np.zeros((8,n_sets))
            tnPE = np.zeros((8,n_sets))
            
            for j, deg in enumerate(np.arange(0,360,45)):
                tnBL_ = dfBL.loc[dfBL['deg']==deg, 'tn'].values[:]
                tnPE_ = dfPE.loc[dfPE['deg']==deg, 'tn'].values[:]
                

                'Baseline'
                inds = []
                count = 0
                for tn_ in tnBL_:
                    nb_tn_ = np.shape(dfBL.loc[dfBL['tn']==tn_, 'spikes'].values[0])[0]
                    
                    if nb_tn_ >= window_len:
                        inds.append(tn_)
                
                if len(inds) < n_sets:
                    print(i,d,date,len(inds))
                
                tnBL[j,:] = np.array(inds)[:n_sets]
                    
                'Perturbation'
                inds = []
                count = 0
                for tn_ in tnPE_:
                    nb_tn_ = np.shape(dfPE.loc[dfPE['tn']==tn_, 'spikes'].values[0])[0]
                    
                    if nb_tn_ >= window_len:
                        inds.append(tn_)
                
                if len(inds) < n_sets:
                    print(i,d,date,len(inds))
                
                tnPE[j,:] = np.array(inds)[:n_sets]
                        
            
                
            '-----------------------------------------------'
            nU_pre = np.shape(dfBL.loc[dfBL['tn']==tnBL[0,0], 'spikes'].values[0])[1]
            dNU['nU_pre'] = nU_pre
            
            units_to_del = []
            
            'Baseline'
            sc = np.zeros((n_trials,nU_pre))
            degsBL = np.zeros((n_trials))
            for ti, tn in zip(np.arange(n_trials),np.sort(np.concatenate((tnBL)))):
                  sc[ti,:] = np.sum(dfBL.loc[dfBL['tn']==tn, 'spikes'].values[0][:window_len,:], axis=0)
                  degsBL[ti] = dfBL.loc[dfBL['tn']==tn, 'deg'].values[0]

            dSC_pre['BL']  = sc
            dSC_pre['mBL'] = np.mean(sc/(window_len*0.1), axis=0)
            dSC_pre['sBL'] = np.std(sc/(window_len*0.1), axis=0)
            dSC_pre['vBL'] = np.var(sc/(window_len*0.1), axis=0)
            dSC_pre['target_degs_BL'] = degsBL.astype(int)
          
            units_to_del.append(np.where(dSC_pre['mBL']< firing_rate_threshold)[0])               


            'Perturbation'
            sc = np.zeros((n_trials,nU_pre))
            degsPE = np.zeros((n_trials))
            for ti, tn in zip( np.arange(n_trials),np.sort(np.concatenate((tnPE))) ):
                  sc[ti,:] = np.sum(dfPE.loc[dfPE['tn']==tn, 'spikes'].values[0][:window_len,:], axis=0)
                  degsPE[ti] = dfPE.loc[dfPE['tn']==tn, 'deg'].values[0]
    
            dSC_pre['PE']  = sc
            dSC_pre['mPE'] = np.mean(sc/(window_len*0.1), axis=0)
            dSC_pre['sPE'] = np.std(sc/(window_len*0.1), axis=0)
            dSC_pre['vPE'] = np.var(sc/(window_len*0.1), axis=0)
            dSC_pre['target_degs_PE'] = degsPE.astype(int)
          
            units_to_del.append(np.where(dSC_pre['mPE']< firing_rate_threshold)[0]) 
            
            'Remove neurons with average firing rates below the firing rate threshold (Hz) during any set of trials.'
            delU = np.unique(np.concatenate((units_to_del)))
            
            'Baseline'
            sc = np.delete(dSC_pre['BL'], delU, axis=1)
            dSC['BL']  = sc
            dSC['mBL'] = np.mean(sc/(window_len*0.1), axis=0)
            dSC['sBL'] = np.std(sc/(window_len*0.1), axis=0)
            dSC['vBL'] = np.var(sc/(window_len*0.1), axis=0)
            dSC['target_degs_BL'] = degsBL.astype(int)
    
            
            'Perturbation'
            sc = np.delete(dSC_pre['PE'], delU, axis=1)
            dSC['PE']  = sc
            dSC['mPE'] = np.mean(sc/(window_len*0.1), axis=0)
            dSC['sPE'] = np.std(sc/(window_len*0.1), axis=0)
            dSC['vPE'] = np.var(sc/(window_len*0.1), axis=0)
            dSC['target_degs_PE'] = degsPE.astype(int)
    


            dNU['nU_post'] = np.shape(dSC['BL'])[1]
              
            
            dParams1 = {'n_sets': n_sets, 'n_trials': n_trials, 'window_len': window_len}
            
       
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



#%%

"""
###############################################################################

[2] Cross Validation

FA3: dNF, dParams2 >  f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
FA4: dFAscores > f'FA4_scores_{subj}_{mode}_{date}.pkl'


###############################################################################
"""


max_n_components = 13#inclusive
n_splits = 10
n_repeats = 10#20


for mode in ['rotation']:#, 'shuffle']:

    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, _, _ = pickle.load(open_file)
    open_file.close()

    for i in [0,1]:
        
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
            obj_to_pickle = dFAscores
            open_file = open(filename, "wb")
            pickle.dump(obj_to_pickle, open_file)
            open_file.close()


            print(dNF)
            

#%%




"""
###############################################################################

[2a] Cross-Validation Checks

CHECKS
    [#1] Determine if the max number of components could be greater than the number of components tested (i.e., nf==max number of compnents tested).
    [#2] Determine if  the number of samples per training model fit was sufficient (>#neurons).

###############################################################################
"""



for mode in ['rotation']:#, 'shuffle']:

    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, _, _ = pickle.load(open_file)
    open_file.close()

    for i in [0,1]:
        
        subj, subject = dSubject[i]

        for d,date in enumerate(dDates[i]):
            
            fn = f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA3', fn), "rb")
            dNF, dParams2 = pickle.load(open_file)
            open_file.close()
            

            'Check #1'
            pass_check_BL = dNF['d_BL_REDO']
            pass_check_PE = dNF['d_PE_REDO']
            
            
            if ( pass_check_BL == 'FAIL') or ( pass_check_PE == 'FAIL' ):
                print(f'REDO: max_n_components: {mode}, {subj}, {d}, {date}')

            
            'Check #2'
            n_splits = dParams2['n_splits']
            
            fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA1', fn), "rb")
            delU, dNU, dParams1, _,_ = pickle.load(open_file)
            open_file.close()
            
            n_trials = dParams1['n_trials']
            
            if dNU['nU_post'] > (n_trials)*((n_splits-1)/n_splits):
                  print(f'REDO: ratio <<: {mode}, {subj}, {date}') 
            
#%%


"""
###############################################################################

[3] Fit FA Models

FA5: dVAR > f'FA5_dVAR_{subj}_{mode}_{date}.pkl'

###############################################################################
"""


for mode in ['rotation']:#,'shuffle']:

    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:
    
        
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
        
            
            fn = f'FA3_inputs_dNF_dParams2_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA3', fn), "rb")
            dNF, dParams2 = pickle.load(open_file)
            open_file.close()
            
            if np.shape(dSC['BL']) != np.shape(dSC['PE']):
                print(f'{subj}, {mode}, {d}, {date}: FAILED CHECK: INPUT DATA SHAPE UNEQUAL')
            # else:
            #     print( np.shape(dSC['BL']), np.shape(dSC['PE']))
    
            if (dNF['d_BL_REDO'] == 'FAIL') or (dNF['d_PE_REDO'] == 'FAIL'):
                print(f'{subj}, {mode}, {d}, {date}: FAILED CHECK: NEED TO REPEAT CV')

                

            """
            
            Fit FA models
            
            """
            
            for blockKey in ['BL', 'PE']:

                dVAR[blockKey] = {}
                
                total_, shared_, private_, loadings_, MAT_shared_, MAT_private_ = fit_fa_model(dSC[blockKey], dNF[blockKey])
  
    
                dVAR[blockKey]['nf'] = dNF[blockKey]
                dVAR[blockKey]['total'] = total_
                dVAR[blockKey]['shared'] = shared_
                dVAR[blockKey]['private'] = private_
                dVAR[blockKey]['loadings'] = loadings_
                dVAR[blockKey]['MAT_shared'] = MAT_shared_
                dVAR[blockKey]['MAT_private']  = np.diag(MAT_private_)
                
            

            os.chdir(os.path.join(pickle_root_save_path,'FA5')) 
            filename = f'FA5_dVAR_{subj}_{mode}_{date}.pkl'
            obj_to_pickle = dVAR 
            open_file = open(filename, "wb")
            pickle.dump(obj_to_pickle, open_file)
            open_file.close()




#%%

"""

[4] Compute Variance Metrics

FA6: dVAR, dVAR_95 > f'FA6_dVAR_{subj}_{mode}_{date}.pkl'

"""



for mode in ['rotation']:#, 'shuffle']:
    
    open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
    dDates, dDegs, dDegs2 = pickle.load(open_file)
    open_file.close()
    
    for i in [0,1]:

        subj, subject = dSubject[i]
        
        for d, date in enumerate(tqdm(dDates[i])):

            
            fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
            open_file = open(os.path.join(pickle_root_save_path, 'FA2', fn), "rb")
            _, dSC = pickle.load(open_file)
            open_file.close()

            open_file = open(os.path.join(pickle_root_save_path, 'FA5', f'FA5_dVAR_{subj}_{mode}_{date}.pkl'), "rb")
            dVAR = pickle.load(open_file)
            open_file.close()
            
            
            dVAR_95 = {'BL':{}, 'PE':{}}
            
            for k in ['BL', 'PE']:
                
                n_trials, nU = np.shape(dSC[k])
                
                'Compute "shared dimensionality" the number of factors needed to explain > 95% of the shared variance explained by the model fit using the number of factors identified from the CV.'
                U = dVAR[k]['loadings'].T
                cov_shared = np.dot(U,U.T)
                rank = np.linalg.matrix_rank(cov_shared) #also just number of factors
                _, S, _ = np.linalg.svd(cov_shared)
                cS = np.cumsum(S[:rank])/np.sum(S[:rank])
                nf95 = np.where(cS > 0.95)[0][0] +1 #add one to account for factor 1 index is 0
           
 
                'Refit model & re-compute metrics'
                total_, shared_, private_, loadings_, MAT_shared_, MAT_private_ = fit_fa_model(dSC[k], nf95) 
                
                U = loadings_.T
                cov_FA = np.matmul(U,U.T) + np.diag(MAT_private_) #cov = components_.T * components_ + diag(noise_variance)


                '---orthonormalize loadings and compute loading similarity, #Umakantha'
                U_orth_, _, _ = np.linalg.svd(cov_shared)
                U_orth = U_orth_[:,:rank]
                loading_similarity = ((1/nU) - np.var(U_orth, axis=0))*nU #np.var(U_orth,1,1)
                        
                        
                dVAR_95[k]['nf_full']     = rank
                dVAR_95[k]['dshared95']   = nf95
                dVAR_95[k]['total']       = total_
                dVAR_95[k]['shared']      = shared_
                dVAR_95[k]['private']     = private_
                dVAR_95[k]['loadings']    = loadings_
                dVAR_95[k]['MAT_shared']  = MAT_shared_
                dVAR_95[k]['MAT_private'] = np.diag(MAT_private_)
                dVAR_95[k]['%sv']         = np.diag(MAT_shared_)/( np.diag(MAT_shared_ )+ np.diag(MAT_private_))
                dVAR_95[k]['loading_sim'] = loading_similarity
                
   
                """
                Pairwise Metrics
                
                    Pearsons correlation coefficient: 
                        mean (rsc_mean), standard deviation (rsc_sd)
                    
                    Mean and variance in spike counts
                    
                    Fano factor
                
                """
        
                dVAR_95[k]['cov_FA']   = cov_FA
                cov_SC = np.cov(dSC[k].T)
                dVAR_95[k]['cov']      = cov_SC

                rsc = []
                rsc2 = []
                for j in range(nU):
                    for kj in range(j):
                
                        rsc.append(cov_FA[j,kj])
                        rsc2.append(cov_SC[j,kj])
                
                
                rsc = np.array(rsc)
                rsc2 = np.array(rsc2)
                
                dVAR_95[k]['FA_rsc'] = rsc
                dVAR_95[k]['FA_rsc_mean'] = np.mean(rsc)
                dVAR_95[k]['FA_rsc_sd']   = np.std(rsc)
                
                dVAR_95[k]['rsc'] = rsc2
                dVAR_95[k]['rsc_mean'] = np.mean(rsc2)
                dVAR_95[k]['rsc_sd']   = np.std(rsc2)
                
                dVAR_95[k]['meanFR'] = np.mean(dSC[k]/(window_len*0.1), axis=0) 
                dVAR_95[k]['varFR']  = np.var(dSC[k]/(window_len*0.1), axis=0) 
                dVAR_95[k]['FF'] =  dVAR_95[k]['varFR'] / dVAR_95[k]['meanFR']  
                

            os.chdir(os.path.join(pickle_root_save_path,'FA6')) 
            filename = f'FA6_dVAR_{subj}_{mode}_{date}.pkl'
            obj_to_pickle = [dVAR, dVAR_95] 
            open_file = open(filename, "wb")
            pickle.dump(obj_to_pickle, open_file)
            open_file.close()
 



