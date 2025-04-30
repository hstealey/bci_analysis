# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:42:30 2025

@author: hanna
"""

from tqdm import tqdm

import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def perform_cv(data, n_components, n_splits, n_repeats, tqdm_n_components=False):
    
    '''
    Performs k-fold cross validation to determine the number of factors that reliably fit the data based on log likelihood of fits.
        Hard-coded for factor analysis
            Default number of ierations (max_iter=1000).
            No rotation.
    
    inputs:
        data: 2D numpy array, number of samples x number of neurons
            Each element of the matrix is the number of spikes within a specific window for a given neuron.
                Summation (also called 'integration') window:
                    decoder time-scale: 100ms
                    trial-to-trial: window may vary, but typically within 500ms - 1000ms
                    
        n_components: 1D numpy array, array of values for number of factors (nf) used to fit the model
        n_splits: int, k; the number of groups to divide the data for k-fold cross validation
            k-1 groups are used for training, and 1 group is used for testing
        n_repeats: int, number of iterations to re-do k-fold cross validation
            The samples in each fold are different (shuffle=True).
    
        
    
    returns:
        nf: float, the number of factors that maximizes the log likelihood of the fit
        m: 1D array, the average log likelihood for each number of factors across all folds and iterations
        fa_scores: 2D array, the log likelihood value for each number of factors across all folds and interations
 
            
    Note on 'n_jobs=-1': number of jobs to run in parallel; -1 uses all processors...may "overwhelm" computer but speeds up calculations
         
    '''
    
    if tqdm_n_components == False:
    
        fa = FactorAnalysis()
        
        fa_scores = []
        
        for n in n_components: #tqdm(n_components):
         
            fa.n_components = n
            fa_scores_n = []
        
            for repeats in range(n_repeats):
                k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=repeats)
                fa_scores_ = cross_val_score(fa, data, cv=k_fold, n_jobs=-1)
                fa_scores_n.append(fa_scores_)
                
            fa_scores.append(np.concatenate((fa_scores_n)))
            
        mean_scores = np.mean(fa_scores, axis=1)
        nf = np.where(mean_scores == np.max(mean_scores))[0][0]
    
    else:
    
        fa = FactorAnalysis()
        
        fa_scores = []
        
        for n in tqdm(n_components):
         
            fa.n_components = n
            fa_scores_n = []
        
            for repeats in range(n_repeats):
                k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=repeats)
                fa_scores_ = cross_val_score(fa, data, cv=k_fold, n_jobs=-1)
                fa_scores_n.append(fa_scores_)
                
            fa_scores.append(np.concatenate((fa_scores_n)))
            
        mean_scores = np.mean(fa_scores, axis=1)
        nf = np.where(mean_scores == np.max(mean_scores))[0][0]
        
            
            
            
    return(nf, mean_scores, fa_scores)







def fit_fa_model(data, nf):

    '''
    Fits factor analysis model to 'data' using the number of factors defined by 'nf'
        Default number of ierations (max_iter=1000).
        No rotation.
    
    inputs:
        data: 2D numpy array, number of samples x number of neurons
            Each element of the matrix is the number of spikes within a specific window for a given neuron.
                Summation (also called 'integration') window:
                    decoder time-scale: 100ms
                    trial-to-trial: window may vary, but typically within 500ms - 1000ms
        nf: int, number of factors used to fit the mode
            Typically determined by cross-validation.
    
    returns:
        total:    float, total estimated variance (shared + private)
        shared:   float, estimated amount of shared variance
        private:  float, estimated amount of private variance
        loadings: 2D numpy array, factor weights for each neuron and each factor 
            Commonly mathematically deonted as L or U
        shared_variance: 2D numpy array, squared factor loadings; diagonal entries represent the shared variance of each neuron explained by all factors 
            Commonly matematically denoted as LL^T or UU^T
        private_variance: 2D numpy array (diagonal), private variance estimated for each neuron
            Commonly matematically denoted as psi
            
    
    Note: The total amount of variance does not change based on the the number of factors.  
    Rather, the partiotioning of the shared and private portions do.
    
    '''
    
    FA = FactorAnalysis(n_components=nf)
    FA.fit(data)
    
    shared_variance = FA.components_.T.dot(FA.components_)
    private_variance = np.diag(FA.noise_variance_)

    total   = np.trace(shared_variance) + np.trace(private_variance) 
    shared  = np.trace(shared_variance) 
    private = np.trace(private_variance)                
    
    loadings = FA.components_

    return(total, shared, private, loadings, shared_variance, private_variance)#, FA.loglike_) 




