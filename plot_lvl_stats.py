# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from parameters import *
import os
import runpy

# Functions
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

def unf_lvl(v, eval_list):

    """
    
    Unfolds the even spectrum locally and returns the unfolded spectrum
    
    parameters:
    -----------
    
    v : spread of eigenvalues taken into consideration while local unfolding
    
    eval_list: list of eigenvalues
    
    """
    
    print("Unfolding Initiated")
    # Unfolded levels
    lvl_unf = []
    unf_val = 0
    for i in tqdm(range(len(eval_list))):
        # Unfolded value of energy
        unf_val = 0
        for m in range(len(eval_list[:i])):
            # Local density of states
            rho_L = loc_den(v, m, eval_list)
            unf_val += rho_L * (eval_list[m]-eval_list[m-1])
        lvl_unf.append(unf_val)
        '''
        rho_L = loc_den(v, i, eval_list)
        unf_val = rho_L*eval_list[i]
        lvl_unf.append(unf_val)
        '''
    lvl_unf = np.sort(lvl_unf)
    
    return lvl_unf

# Plot Level Statistics
dataList = []
for g in g_arr:
    print(f'g={g}')
    # Load the energies
    if not os.path.exists(f"evals_par/evals_g={g}_j={j}_M={M}.npy"):
        runpy.run_path("calc_evals_par.py",init_globals={'__name__': '__main__'})
    evals = np.load(f"evals_par/evals_g={g}_j={j}_M={M}.npy")
    # Unfold the energies
    evals_unfl = unf_lvl(v, evals)
    lvl_sp_arr = []
    for i in range(len(evals_unfl)-1):
        lvl_sp = evals_unfl[i+1]-evals_unfl[i]
        lvl_sp_arr.append(lvl_sp)
    lvl_sp_arr = np.sort(lvl_sp_arr)
    dataList.append(lvl_sp_arr)

# Plot
plt.figure(figsize=(9,7))
for g_ind, lvl_sp in enumerate(dataList):
    # Reference plots
    x_val = np.arange(0,4,0.1)
    y_gauss = np.pi/2 * x_val * np.exp(-np.pi*x_val**2/4)
    y_poi = np.exp(-x_val)
    # Create a histogram
    #plt.suptitle(f'Lower Energy Levels\n j={j}, M={M}')
    plt.subplot(4,3,g_ind+1)
    plt.title(f'g={g_arr[g_ind]}')
    plt.plot(x_val, y_gauss, linestyle = '--', label = 'Gaussian', linewidth = 1)
    plt.plot(x_val, y_poi, label = 'Poisson', linewidth = 1)
    hist_values,bin_edges, _ = plt.hist(lvl_sp, bins=60, histtype= 'step', density=True)

    # Add labels and title
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.xlim([0,4])
    #plt.legend()
    plt.grid()
    plt.tight_layout()

# Show the plot
if not os.path.exists("plots"):
    os.mkdir("plots")
plt.savefig('plots/Level_Statistics')
plt.show()