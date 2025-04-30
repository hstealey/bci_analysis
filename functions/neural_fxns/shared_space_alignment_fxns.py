# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:29:25 2025

@author: hanna
"""


import numpy as np

def calcSSA(UA,UB):
    
    cov_shared_A = np.dot(UA,UA.T)
    
    cov_shared_B = np.dot(UB,UB.T)
    rB = np.linalg.matrix_rank(cov_shared_B)
    
    U, S, VT = np.linalg.svd(cov_shared_B)
    bB = U[:,:rB] #orthonormal basis for the column space of B # V = VT.T  bB = V[:,:rB]
    PB = np.dot(bB,bB.T) #projection matrix into the column space of B
    
    num = np.trace(np.matmul(PB, cov_shared_A, PB.T))
    den  = np.trace(cov_shared_A)
    
    ssa_ = num/den
    
    # rA = np.linalg.matrix_rank(cov_shared_A)
    # U, S, VT = np.linalg.svd(cov_shared_A)
    # Sigma = np.diag(S)#[2:4, 2:4] #Sigma = np.diag(S)[:rA, :rA]
    # cov_shared_A = np.dot( np.dot(U[:,:rA], np.diag(S)[:rA,:rA]), VT[:rA,:] )
    

    return(ssa_)

