# -*- coding: utf-8 -*-
"""
Updated February 10, 2025

@author: hanna
"""



"""

Fit cosine tuning curves to each BCI neuron and assess significance using bootstrapping


Full description of methods can be found here: 
    Population-level constraints on single neuron tuning changes during behavioral adaptation (Stealey et al., under review 2025) https://doi.org/10.1101/2025.04.17.649401
		

    Sections of code:
        [1] generate bootstrapped means
        [2] fit cosine tuning curves
        [3] compute changes in preferred direction (measured vs "assinged" by decoder gain)

"""



#from datetime import datetime

import os
import pickle

import numpy  as np
from tqdm import tqdm


# root_path = r'C:\Users\hanna\OneDrive\Documents\bci_analysis'
# pickle_path = os.path.join(root_path, 'DEMO_pickles')


def run_tuning_curve(root_path, pickle_path, modes, dSubject):

    '------------------------------'
    'Loading Custom Functions'
    os.chdir(os.path.join(root_path, 'functions', 'neural_fxns'))
    from tuning_curve_fxns import degChange, compute_signed_degChange, cosine_model, fit_cosine_model
    
 
    
    'Adjustable parameters'
    
    # dSubject = {0: ['airp', 'Monkey A'], 1: ['braz', 'Monkey B']}
    
    nb = 8 #number of bins to split stimulus space (360°)
    n_random_repeats = 1000 #for bootstrapping procedure
    window_start = 2 #200ms from trial start
    window_end   = 6 #600ms from trial start; non-inclusive value
    n_window = window_end - window_start #indices 2,3,4,5; 400ms
    
    pickle_root_pull_path = os.path.join(pickle_path, 'FA_tuning')
    pickle_root_save_path = os.path.join(pickle_path, 'tuning')
    
    set_ns_delta_0 = True
    
    
   
    
    """
    ###############################################################################
    ###############################################################################
    
    [1] Generate Bootstrapped Means
        
    ###############################################################################
    ###############################################################################
    """
    
    #start_time = datetime.now()
    
    
    
    for mode in modes: #['rotation','shuffle']:
        
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, dDegs2 = pickle.load(open_file)
        open_file.close()
    
        for i in range(len(dSubject)): #[0,1]:
    
            subj, subject = dSubject[i]
    
            for d, date in enumerate(tqdm(dDates[i][:])):
    
                'Load Data'
    
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_pull_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, _, _ = pickle.load(open_file)
                open_file.close()
                
                fn = f'FA2_inputs2_dSCpre_dSC_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_pull_path, 'FA2', fn), "rb")
                _, dSC = pickle.load(open_file)
                open_file.close()
                
                if dParams1['n_sets_BL'] == dParams1['n_sets_PE']:
                    n_sets   = dParams1['n_sets_BL'] 
                    n_trials = dParams1['n_trials_BL']
                    
                n_window = dParams1['n_window'] 
                nU = dNU['nU_post']
    
    
    
                dMean = {'BL':{}, 'PE':{}}
                for k in ['BL','PE']:
                    sc    = dSC[k]
                    targs = dSC[f'target_degs_{k}']
                    
                    dMean[k] = {'mean': {} }#, 'all': {} } #, 'spike_count': {}}
                    
                    for neuron in range(nU): 
                        dMean[k]['mean'][neuron] = np.zeros((n_random_repeats,nb))
                        #dMean[k]['all'][neuron] = np.zeros((n_random_repeats,nb))
                        
                    for bootstrap_sample in range(n_random_repeats):
                        np.random.seed(bootstrap_sample) 
                        
                        rand_inds = np.random.choice(np.arange(n_sets), size=n_sets, replace=True)
                        
                        for j, deg in enumerate(range(0,360,45)):
                            bin_inds = np.where(targs == deg)[0][rand_inds]
    
                            for neuron in range(nU): 
                                sci = sc[:,neuron]
                                dMean[k]['mean'][neuron][bootstrap_sample, j] = np.mean(sci[bin_inds]/(n_window*0.1))
                                #dMean[k]['all'][neuron][bootstrap_sample, j]  = sci[bin_inds]/(n_window*0.1)
                                #dMean[k]['spike_count'][neuron][bootstrap_sample, j] = np.sum(sci[bin_inds])
    
                '''Saving values to .pkl'''     
                os.chdir(os.path.join(pickle_root_save_path, 'bootstrap'))
                obj_to_pickle = dMean
                filename = f'dMean_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
    
    
    
    # end_time = datetime.now()
    
    # elapsed_time = end_time - start_time
    
    # print(elapsed_time)
    

    
    """
    ###############################################################################
    ###############################################################################
    
    [2] Fit {Standard} Cosine Tuning Curves
        
    ###############################################################################
    ###############################################################################
    """
    
    #start_time = datetime.now()
    
    xdata = np.arange(0,360,45)*(np.pi/180)
    
    
    for mode in modes: #['rotation', 'shuffle']:
        
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, dDegs2 = pickle.load(open_file)
        open_file.close()
    
        for i in range(len(dSubject)):#[0,1]:
    
            subj, subject = dSubject[i]
    
            for d, date in enumerate(tqdm(dDates[i][:])):
                
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_pull_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, _,_ = pickle.load(open_file)
                open_file.close()
                
                
                nU = dNU['nU_post']
                
                fn = f'dMean_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'bootstrap', fn), "rb")
                dMean = pickle.load(open_file)
                open_file.close()
    
                dTC = {}
                            
                for k in ['BL', 'PE']:
                    
                    dTC[k] = {}
                    
                    for neuron in range(nU): #tqdm(range(nU)):
                        
                        dTC[k][neuron] = {'TC': np.zeros((n_random_repeats, 360)),
                                          'PD_est': np.zeros((n_random_repeats)),
                                          'MD_est': np.zeros((n_random_repeats)),
                                          'b0_est': np.zeros((n_random_repeats)),
                                          'b1_est': np.zeros((n_random_repeats))}
                                
                
                        for samp in range(n_random_repeats):
                            m = dMean[k]['mean'][neuron][samp,:]
                            M, MD, PD, meanFR, y_est = fit_cosine_model(xdata, m)
                            
                            if PD < 0:
                                PD += 2*np.pi
                            
                            dTC[k][neuron]['TC'][samp,:]   = cosine_model(np.arange(0,360,1)*(np.pi/180), *(M, PD, meanFR))
                            dTC[k][neuron]['PD_est'][samp] = PD*(180/np.pi)
                            dTC[k][neuron]['MD_est'][samp] = MD
                            dTC[k][neuron]['b0_est'][samp] = meanFR
                            dTC[k][neuron]['b1_est'][samp] = M
                            
                '''Saving values to .pkl'''     
                os.chdir(os.path.join(pickle_root_save_path, 'fit'))
                obj_to_pickle = dTC
                filename = f'dTC_{subj}_{mode}_{date}.pkl'
                open_file = open(filename, "wb")
                pickle.dump(obj_to_pickle, open_file)
                open_file.close()
                            
    
    # end_time = datetime.now()
    
    # elapsed_time = end_time - start_time
    
    # print(elapsed_time)
    
    

    
    """
    
    ###############################################################################
    ###############################################################################
    
    [3] Changes in Preferred Direction
    
        dPD_all: session averaged changes in preferred direction
        dPD_each: median tuning changes (and assigned tuning changes)
        
    ###############################################################################
    ###############################################################################
    
    """
    
    #start_time = datetime.now()
    
    for mode in modes: #['rotation', 'shuffle']: 
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, dDegs2 = pickle.load(open_file)
        open_file.close()
        
        dPD_all  = {}
        dPD_each = {}
        dKG_each = {}
    
        for i in range(len(dSubject)): #[0,1]:
        
            subj, subject = dSubject[i]
            
            dPD_all[i] = {'dPD_abs': np.zeros(len(dDates[i])),
                          'dPD_mean': np.zeros(len(dDates[i])),
                          'dPD_median': np.zeros(len(dDates[i])),
                          'dPD_16': np.zeros(len(dDates[i])),
                          'dPD_84': np.zeros(len(dDates[i]))}
            
            
            dPD_each[i] = {}
            dKG_each[i] = {}
            
            
            '#############################################################'
            for d, date in enumerate(tqdm(dDates[i][:])):
    
                
                fn = f'FA1_inputs1_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_pull_path, 'FA1', fn), "rb")
                delU, dNU, dParams1, _, _ = pickle.load(open_file) 
                open_file.close()
                
                nU = dNU['nU_post']
                
                fn = f'dfKG_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_path, 'dfKG', fn), "rb")
                dfKG = pickle.load(open_file)
                open_file.close()
                
                shuff = np.delete(dfKG['shuffled'].values, delU)
                
                #Yes, this is correct. Values were initially stored (y,z,x) - so inverting them now.
                xBL = np.delete(dfKG['yBL'].values, delU)
                yBL = np.delete(dfKG['xBL'].values, delU)
                assigned_PDBL = np.array([rad+2*np.pi if rad < 0 else rad for rad in np.arctan2(yBL,xBL)])*(180/np.pi)
                
                xPE = np.delete(dfKG['yPE'].values, delU)
                yPE = np.delete(dfKG['xPE'].values, delU)
                assigned_PDPE = np.array([rad+2*np.pi if rad < 0 else rad for rad in np.arctan2(yPE,xPE)])*(180/np.pi)
                
                assigned_dPD = np.zeros(nU)
                for j, PD1, PD2 in zip(np.arange(nU), assigned_PDBL, assigned_PDPE):
                    
                    if PD1 != PD2:
                        sign_, dPD_ = compute_signed_degChange(PD1, PD2)
                        assigned_dPD[j] = sign_*dPD_
                    else:
                        assigned_dPD[j] = 0 
    
    
                'dTC'
                fn = f'dTC_{subj}_{mode}_{date}.pkl'
                open_file = open(os.path.join(pickle_root_save_path, 'fit',  fn), "rb") 
                dTC = pickle.load(open_file)
                open_file.close()
        
                MD_BL = np.zeros(nU)
                MD_PE = np.zeros(nU)
                
                dPD_lo = np.zeros(nU)
                dPD_hi = np.zeros(nU)
                dPD = np.zeros(nU)
                dPD_abs = np.zeros(nU)
                sig = np.zeros(nU)
                for neuron in range(nU):
                    
                    'Modulation Depth'
                    MD_BL_ = dTC['BL'][neuron]['MD_est']
                    MD_PE_ = dTC['PE'][neuron]['MD_est']
                    
                    MD_BL[neuron] = np.percentile(MD_BL_, 50)
                    MD_PE[neuron] = np.percentile(MD_PE_, 50)
    
           
                    'Preferred Direction'
                    PD_BL = dTC['BL'][neuron]['PD_est']
                    PD_PE = dTC['PE'][neuron]['PD_est']
                
                    sign = np.zeros(n_random_repeats)
                    deltaPD = np.zeros(n_random_repeats)
                    for samp, PD1, PD2 in zip(np.arange(n_random_repeats), PD_BL, PD_PE):
                        
                        delta1 = PD1 - PD2
                        
                        sign_, dPD_ = compute_signed_degChange(PD1, PD2)
                            
                        sign[samp] = sign_
                        deltaPD[samp] = dPD_
                    
                    dPD_lo_ = np.percentile(sign*deltaPD,2.5)
                    dPD_hi_ = np.percentile(sign*deltaPD,97.5)
                    
                    
                    if (0 > dPD_lo_) and (0 < dPD_hi_):
                        sig[neuron] = False
                        
                        if set_ns_delta_0 == True:
                            dPD[neuron] = 0
                            dPD_abs[neuron] = 0
                        else:
                            dPD[neuron] = np.percentile(sign*deltaPD,50)
                            dPD_abs[neuron] = np.percentile(deltaPD,50)
                    else:
                        sig[neuron] = True
                        dPD[neuron] = np.percentile(sign*deltaPD,50)
                        dPD_abs[neuron] = np.percentile(deltaPD,50)
                        
                    dPD_lo[neuron] = dPD_lo_
                    dPD_hi[neuron] = dPD_hi_
    
    
                dPD_all[i]['dPD_abs'][d]    = np.mean(dPD_abs)
                dPD_all[i]['dPD_mean'][d]   = np.mean(dPD)
                dPD_all[i]['dPD_median'][d] = np.percentile(dPD, 50) 
                dPD_all[i]['dPD_16'][d]     = np.percentile(dPD, 16)
                dPD_all[i]['dPD_84'][d]     = np.percentile(dPD, 84)
             
                dPD_each[i][date] = {}
                dPD_each[i][date]['shuff']  = shuff
                dPD_each[i][date]['MD_BL']  = MD_BL #median from bootstrap distribution
                dPD_each[i][date]['MD_PE']  = MD_PE #median from bootstrap distribution
                dPD_each[i][date]['dPD_lo'] = dPD_lo
                dPD_each[i][date]['dPD_hi'] = dPD_hi
                dPD_each[i][date]['dPD_median'] = dPD #median
                dPD_each[i][date]['dPD_abs'] = dPD_abs
                dPD_each[i][date]['sig']    = sig
                dPD_each[i][date]['assigned_PDBL'] = assigned_PDBL
                dPD_each[i][date]['assigned_PDPE'] = assigned_PDPE
                dPD_each[i][date]['assigned_dPD']  = assigned_dPD
                
                dKG_each[i][date] = {}
                dKG_each[i][date]['xBL'] = xBL
                dKG_each[i][date]['yBL'] = yBL
                dKG_each[i][date]['xPE'] = xPE
                dKG_each[i][date]['yPE'] = yPE
    
                
        '''Saving values to .pkl'''     
        os.chdir( pickle_root_save_path )
        obj_to_pickle = [dDates, dDegs, dDegs2, dPD_all, dPD_each]
        if set_ns_delta_0 == True:
            filename = f'tuning_ns0_{mode}.pkl'
        else:
            filename = f'tuning_{mode}.pkl'
        open_file = open(filename, "wb")
        pickle.dump(obj_to_pickle, open_file)
        open_file.close()
    
    
    # end_time = datetime.now()
    
    # elapsed_time = end_time - start_time
    
    # print(elapsed_time)