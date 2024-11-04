import numpy as np
import qutip as qt
import time
import multiprocessing as mp
from tqdm import tqdm
import os
from parameters import *
import runpy

def DH(w, w0, g, M, j):
    
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

def SFF_fun(psi0, psi):
    
    '''
    This function returns the spectral form factor for a given state psi which is an overlap of psi with psi0
    
    parameters
    ----------
    
    H : Hamiltonian
    
    psi0 : Initial state
    
    c_op : Collapse operator
    
    '''
    
    SFF = psi.overlap(psi0)
    SFF = np.abs(SFF) * np.abs(SFF)
    
    return SFF

def SFF_list(psi0, psi_list, tlist):
    
    '''
    This function returns the SFF at each time step for a single trajectory.
    
    parameters
    ----------
    
    psi0 : Initial state
    
    psi_list : psi for each time
    
    tlist : pass time list as an array
    
    '''
    
    SFF_list = []
    for t_ind,t in enumerate(tlist):
        psi = psi_list[t_ind]
        SFF = SFF_fun(psi0, psi)
        SFF_list.append(SFF)
    SFF_list = np.array(SFF_list)
    return SFF_list

def my_mcsolve(w, w0, g, M, j, psi0, tlist, kappa, ntraj):
    
    '''
    MCWF for Dicke Model with a Kraus operator jump. Returns WF for each time. 
    
    Parameters
    ----------
    
    w : frequency of the bosonic field
    
    w0 : Energy difference in spin states
    
    g : Coupling strength
    
    M : Upper limit of bosonic fock states
    
    j : Pseudospin
    
    psi0 : Initial state
    
    tlist : pass time list as an array
    
    kappa : Jump strength
    
    ntraj : number of trajectories
    
    '''
    
    # Dicke hamiltonian
    H = qt.Qobj(np.array(DH(w, w0, g, M, j).full()))

    # Measurement collapse operators(cavity decay)
    c_op = np.sqrt(kappa) * qt.tensor(qt.destroy(M),qt.qeye(int(2*j+1)))
    c_op = qt.Qobj(np.array(c_op.full()))

    # Expectation value
    e_op = psi0 * psi0.dag()
    
    result = qt.mcsolve(H, psi0, tlist, c_op, e_op, ntraj, options={"map":"loky"})
    
    SFF_list = result.expect
    
    SFF_list = np.abs(SFF_list) * np.abs(SFF_list)
    
    return SFF_list


def integrate(g, tlist, kappa, beta, ntraj):
    
    '''
    
    Intergrate the Dicke Hamiltonian with damping of bosonic modes
    
    parameters
    ----------
    
    g : Coupling strength
    
    kappa : damping strength
    
    beta : Inverse temperature

    ntraj : Number of trajectories
        
    '''
    
    print(f'j={j}, M={M}, g={g}, beta={beta}, kappa={kappa}')

    # Use Master equation solver to get list of psi for each time step
    
    ev_list = np.load(f"ev/ev_g={g}_j={j}_M={M}.npy",allow_pickle=True)
    evals_list = np.load(f"evals/evals_g={g}_j={j}_M={M}.npy",allow_pickle=True)

    # Initial state is the CGS state
    psi0 = CGS_fun(ev_list, evals_list, beta)

    #SFF_avg_list = SFF_list(psi0, rho_list, tlist)
    file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta},kappa={kappa},ntraj={ntraj}.dat"
    if not os.path.exists(file_path):
        SFF_list = my_mcsolve(w, w0, g, M, j, psi0, tlist, kappa, ntraj)
        if not os.path.exists('SFF'):
            os.mkdir("SFF")
        with open(file_path, 'w') as file:
            for t_ind, t in enumerate(tlist):
                file.write("{}".format(t))
                SFF = SFF_list[:,t_ind][0]
                file.write("\t{}".format(SFF))
                file.write("\n")
        return 0
    else:
        raise Exception("File already exists.")
  
def CGS_fun(ev_list, evals_list, beta):

    '''
    
    Returns CGS function
    
    prameters
    ---------
    
    ev_list : Eigenstates
    
    evals_list : Eigenvalues
    
    beta : Inverse temperatures
    
    '''
    
    psi0 = np.exp(-beta/2*evals_list[0]) * ev_list[0]
    for i in range(len(ev_list[1:])):
        psi0 += np.exp(-beta/2*evals_list[i+1]) * ev_list[i+1]
        
    norm = 0
    for i in range(len(evals_list)):
        norm += np.exp(-beta*evals_list[i])
    norm = np.sqrt(norm)
        
    psi0 = psi0/norm
    
    # psi0 = qt.Qobj(np.array(qt.Qobj(psi0).full()))

    return qt.Qobj(psi0)

def main():
    for beta in beta_arr:
        for g in g_arr: 
            integrate(g, tlist_open, kappa, beta, ntraj)

    return 0

if __name__=='__main__':
    main()