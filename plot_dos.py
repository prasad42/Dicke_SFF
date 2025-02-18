from dicke_sff_lib import *
from parameters import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_g = len(g_arr)
    num_rows = (num_g + 1) // 2
    plt.figure(figsize=(10,4*num_rows))
    for g_ind, g in enumerate(g_arr):
        plt.subplot(num_rows,2,g_ind+1)
        plt.title(f"g={g}")
        eigvals = dicke_eigvals_fun(ω, ω0, j, M, g)
        dos = dos_fun(eigvals)
        plt.xlabel(r"$E$")
        plt.plot(eigvals[:-1], dos)
    # plt.show()

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def density_of_states_kde(eigvals, bandwidth=0.1, num_points=500):
    """
    Computes and plots the density of states (DOS) using Kernel Density Estimation (KDE).
    
    Args:
    - eigvals (array-like): Sorted eigenvalues of the system.
    - bandwidth (float): Bandwidth for KDE (controls smoothness).
    - num_points (int): Number of points for evaluating the density.

    Returns:
    - E_vals (np.ndarray): Energy values where DOS is computed.
    - dos (np.ndarray): Density of states values.
    """
    eigvals = np.sort(eigvals)  # Ensure eigenvalues are sorted

    # Kernel Density Estimation (KDE)
    kde = gaussian_kde(eigvals, bw_method=bandwidth)
    E_vals = np.linspace(min(eigvals), max(eigvals), num_points)
    dos = kde(E_vals)

    return E_vals, dos

num_g = len(g_arr)
num_rows = (num_g + 1) // 2
plt.figure(figsize=(10,4*num_rows))
for g_ind, g in enumerate(g_arr):
    plt.subplot(num_rows,2,g_ind+1)
    plt.title(f"g={g}")
    eigvals = dicke_eigvals_fun(ω, ω0, j, M, g)
    E_vals, dos = density_of_states_kde(eigvals, bandwidth=0.1)
    plt.xlabel(r"$E$")
    plt.plot(E_vals, dos)
plt.show()