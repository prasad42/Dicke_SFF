'''
-------------------------------------------------------------------------------------
				        # Import Libraries #
-------------------------------------------------------------------------------------
'''

import numpy as np
import qutip as qt
import multiprocessing as mp
import time
from tqdm import tqdm
import os
from parameters import *

'''
-------------------------------------------------------------------------------------
				        # Functions #
-------------------------------------------------------------------------------------
'''

def DH_fun(w, w0, g, M, j):
    
    '''
    This function returns the Dicke Hamiltonian for the following parameters.
    
    Parameters
    ----------
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    g : Coupling strength
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    
    '''
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    
    H0 = w * a.dag() * a + w0 * Jz
    H1 = 1.0 / np.sqrt(2*j) * (a + a.dag()) * (Jp + Jm)
    H = H0 + g * H1
    
    return H

def par_op_fun(M, j):
    '''
    This function retruns the parity operator.
    
    Parameters
    ----------
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    
    '''
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    J2 = Jz*Jz - Jz + Jp*Jm
    # Dimension of Pseudospin
    n = 2*j + 1
    iden = qt.tensor(qt.qeye(M), qt.qeye(int(n)))
    
    par_op = Jz + a.dag()*a + (J2 + 1/4 * iden).sqrtm() - 1/2 * iden
    par_op = (np.pi*1j*par_op).expm()
    
    return par_op

def proj_ev_H(w, w0, g, M, j):
    
    '''
    Gives even subspace Dicke Hamiltonian
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    g : Coupling strength
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    '''
    
    print("Project to even subspace")
    H = DH_fun(w, w0, g, M, j)
    P = par_op_fun(M, j)
    # Project to even subspace 
    Pi_pl = (1+P)/2
    H = Pi_pl*H*Pi_pl.dag()
    # Convert to a numpy array
    H = np.array(H.full())
    
    tol = 1e-10
    # Delete zero rows
    for i in tqdm(range(int(len(H)/2)+1)):
        count = 0
        for j in range(int(len(H))):
            if H[i][j] < tol:
                count += 1
        if count == len(H):
            H = np.delete(H,i,axis=0)

    # Delete zero columns
    H_t = np.transpose(H)
    for i in tqdm(range(int(len(H_t)/2)+1)):
        count = 0
        for j in range(int(len(H_t)/2)):
            if H_t[i][j] < tol:
                count += 1
        if count == int(len(H_t)/2):
            H_t = np.delete(H_t,i,axis=0)

    return H_t


def Ham_eig_str(w, w0, g_arr, M, j):
    '''
    The function stores the eigenvalues of the Hamiltonain in an npy file
    
    parameters
    ----------
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    g_arr : Coupling strengths
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    
    '''
    if not os.path.exists("evals_par"):
        os.mkdir(f"evals_par")
    for g in tqdm(g_arr):
        print(f"g={g}\n")
        H = proj_ev_H(w, w0, g, M, j)
        eval_list = np.linalg.eigvals(H)
        eval_list = np.sort(eval_list)
        np.save(f"evals_par_test/evals_g={g}_j={j}_M={M}.npy",eval_list)
        
    return eval_list

'''
-------------------------------------------------------------------------------
			        # Calculate/Remove SFF#
-------------------------------------------------------------------------------	
'''
# Start the timer
InitialTime = time.time()

# Evaluate the eigenvalues of the even parity Hamiltonian
Ham_eig_str(w, w0, g_arr, M, j)

#end the timer                       
FinalTime = time.time()
print("\n Total time elapsed: " + str((FinalTime - InitialTime)/60) + "mins")

'''
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
'''
