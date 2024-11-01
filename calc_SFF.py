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

def loc_den(v, i, lvl_arr):
    '''
    This function gives local density of states.
    
    Parameters
    ----------
    
    v : Local unfolding parameter
    
    i : index of the energy level
    
    lvl_arr : Array of energy eigenvalues
    
    '''
    N = len(lvl_arr)
    if (v < 1 or v > int(N-1)):
        raise Exception(f"Enter number v between 0 and {N}")
    
    if (i < v):
        rho_L = 2 * v /(lvl_arr[v+v]-lvl_arr[0])
    elif (i > N-1-v):
        rho_L = 2 * v /(lvl_arr[N-1]-lvl_arr[N-1-v-v])
    else:
        rho_L = 2 * v /(lvl_arr[i+v]-lvl_arr[i-v])
    
    return rho_L

def filter_fun(E, E_avg, wid, eta):
    
    '''
    This function filters the middle part of the spectrum
    
    eval_list : energy array
    
    E : Energy value
    
    eta : filter width
    
    '''
    
    val = np.exp(-(E-E_avg)**2/(2*wid*eta**2))
    
    return val

def SFF_list(eval_list, tlist, beta, g, M, j, eta):
    
    '''
    This function returns the SFF with normal energies at each time step for a single trajectory.
    
    parameters
    ----------
    
    eval_list : Unfolded energies
    
    tlist : pass time list as an array
    
    beta : Finite T
    
    g : Coupling strength
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    
    '''
    if not os.path.exists("SFF"):
        os.mkdir("SFF")

    with open(f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta}.dat", 'a') as file:
        # average energy of the spectrum
        #E_avg = np.average(eval_list)
        # width of the spectrum
        # wid = np.var(eval_list)
        # Select only middle states
        #eval_list = eval_list[int(0.2*len(eval_list)):int(0.8*len(eval_list))]
        for t_ind, t in tqdm(enumerate(tlist)):
            SFF = 0
            norm = 0
            for i,eval1 in enumerate(eval_list):
                # f = filter_fun(eval1, E_avg, wid, eta)
                # rho = loc_den(v, i, eval_list)
                SFF += np.exp(-(beta+1j*t)*(eval1))
                norm += np.exp(-beta*eval1)
            SFF1 = SFF/norm
            SFF = np.conjugate(SFF)*SFF/(norm**2)
            file.write("{}".format(t))
            file.write("\t{}".format(SFF))
            file.write("\t{}".format(SFF1))
            file.write("\n")
    
    return 0

'''
-------------------------------------------------------------------------------
			        # Calculate/Remove SFF#
-------------------------------------------------------------------------------	
'''
# Start the timer
InitialTime = time.time()

# Find the SFF of the spectrum
# protect the entry point
if __name__ == '__main__':
    # create and configure the process pool
    with mp.Pool(nproc) as pool:
        args_list = []
        for g in g_arr:
            eval_list = np.load(f"evals_par/evals_g={np.round(g,2)}_j={j}_M={M}.npy")            
            # Prepare arguments for parallelization
            args_list.append([eval_list, tlist, beta, g, M, j, eta])
        # execute tasks and process results in order
        for result in pool.starmap(SFF_list, args_list, chunksize = 1):
            print(f'Result: {result}')

#end the timer                       
FinalTime = time.time()
print("\n Total time elapsed: " + str((FinalTime - InitialTime)/60) + "mins")

'''
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
'''
