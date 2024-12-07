import numpy as np
import qutip as qt
import os
from tqdm import tqdm
import scipy.linalg as sl

def DH_fun(ω, ω0, j, M, g):
    
    '''
    This function returns the even parity Dicke Hamiltonian for the following parameters.
    
    Parameters
    ----------
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin

    g : Coupling strength
    
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

def loc_den(v, i, eigvals):
    '''
    This function gives local density of states.
    
    Parameters
    ----------
    
    v : Local unfolding parameter
    
    i : index of the energy level
    
    eigvals : Array of energy eigenvalues
    
    '''
    N = len(eigvals)
    if (v < 1 or v > int(N-1)):
        raise Exception(f"Enter number v between 0 and {N}")
    
    if (i < v):
        rho_L = 2 * v /(eigvals[v+v]-eigvals[0])
    elif (i > N-1-v):
        rho_L = 2 * v /(eigvals[N-1]-eigvals[N-1-v-v])
    else:
        rho_L = 2 * v /(eigvals[i+v]-eigvals[i-v])
    
    return rho_L

def unf_eigval_fun(v, eigvals):

    """
    
    Unfolds the even spectrum locally and returns the unfolded spectrum
    
    parameters:
    -----------
    
    v : spread of eigenvalues taken into consideration while local unfolding
    
    eigvals: list of eigenvalues
    
    """
    
    # Unfolded levels
    lvl_unf = []
    unf_val = 0
    for i in range(len(eigvals)):
        # Unfolded value of energy
        unf_val = 0
        for m in range(len(eigvals[:i])):
            # Local density of states
            rho_L = loc_den(v, m, eigvals)
            unf_val += rho_L * (eigvals[m]-eigvals[m-1])
        lvl_unf.append(unf_val)
    lvl_unf = np.sort(lvl_unf)
    
    return lvl_unf

def eigval_sp_fun(ω, ω0, j, M, g, v):

    '''
    The function returns the spacings between the unfolded eigenvalues

    Parameters
    ----------
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin

    g : Coupling strength

    '''
    # Calculate Level Statistics
    # Load the energies
    file_path = f"evals_par/evals_j={j}_M={M}_g={g}.npy"
    if not os.path.exists(file_path):
        dicke_eigvals_fun(ω, ω0, j, M, g)        
    eigvals = np.load(file_path)

    # Unfold the energies
    eigvals = unf_eigval_fun(v, eigvals)
    eigvals_sp = []
    for i in range(len(eigvals)-1):
        lvl_sp = eigvals[i+1]-eigvals[i]
        eigvals_sp.append(lvl_sp)
    eigvals_sp = np.sort(eigvals_sp)

    return eigvals_sp

def dicke_eigvals_fun(ω, ω0, j, M, g):
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

    file_path = f"evals_par/evals_j={j}_M={M}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = DH_fun(ω, ω0, j, M, g)
        eigvals = sl.eigvals(H)
        eigvals = np.sort(eigvals)
        np.save(file_path,eigvals)
    else:
        print(f"{file_path} already exists.")
        
    return 0


def SFF_list(j, M, g, beta, tlist, eval_list):
    
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

    file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta}.dat"
    if not os.path.exists(file_path):
        print("Data does not exist, generating data.")
        with open(file_path, 'w') as file:
            for t in tlist:
                SFF = 0
                norm = 0
                for eval in eval_list:
                    SFF += np.exp(-(beta+1j*t)*(eval))
                    norm += np.exp(-beta*eval)
                SFF1 = SFF/norm
                SFF = np.conjugate(SFF)*SFF/(norm**2)
                file.write(f"{t}\t{SFF}\t{SFF1}\n")
    else:
        print("Data already exists.")
    
    return 0