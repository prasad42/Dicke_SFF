import numpy as np
import qutip as qt
import os
import tqdm

def DH_fun(ω, ω0, g, M, j):
    
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
    
    H0 = ω * a.dag() * a + ω0 * Jz
    H1 = 1.0 / np.sqrt(2*j) * (a + a.dag()) * (Jp + Jm)
    H = H0 + g * H1
    
    H_even = H[::2,::2]

    return H_even

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

    with open(f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta}.dat", 'w') as file:
        for t_ind, t in tqdm(enumerate(tlist)):
            SFF = 0
            norm = 0
            for i,eval1 in enumerate(eval_list):
                SFF += np.exp(-(beta+1j*t)*(eval1))
                norm += np.exp(-beta*eval1)
            SFF1 = SFF/norm
            SFF = np.conjugate(SFF)*SFF/(norm**2)
            file.write("{}".format(t))
            file.write("\t{}".format(SFF))
            file.write("\t{}".format(SFF1))
            file.write("\n")
    
    return 0