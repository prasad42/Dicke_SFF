'''
-------------------------------------------------------------------------------------
				        # Import Libraries #
-------------------------------------------------------------------------------------
'''

import numpy as np
import qutip as qt
import multiprocessing as mp
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

    # Create the directories if they do not exists
    if not os.path.exists("evals"):
        os.mkdir("evals")
    if not os.path.exists("ev"):
        os.mkdir("ev")

    for g in tqdm(g_arr):

        # Calculate the eigenvalues and the eigenvectors
        H = np.array(DH_fun(w, w0, g, M, j).full())
        eval_list, ev_list = np.linalg.eig(H)
        idx = eval_list.argsort()  # Sort indices
        eval_list = np.sort(eval_list)
        eval_list = eval_list[idx]
        ev_list = ev_list[:, idx]

        # Save the eigenvalues and the eigenvectors
        np.save(f"evals/evals_g={g}_j={j}_M={M}.npy",eval_list)
        np.save(f"ev/ev_g={g}_j={j}_M={M}.npy",ev_list)
        
    return eval_list

'''
-------------------------------------------------------------------------------
			        # Calculate/Remove SFF#
-------------------------------------------------------------------------------	
'''

# Evaluate the eigenvalues of the even parity Hamiltonian
print("Generating data for eigenvalues and eigenvectors of the full spectrum")
Ham_eig_str(w, w0, g_arr, M, j)

'''
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
'''