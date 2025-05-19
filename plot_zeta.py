import matplotlib.pyplot as plt
import numpy as np
from dicke_sff_lib import *
from parameters import *
import os
from scipy.integrate import simpson  # Import Simpson's rule for better integration

g_arr = [0.1, 1.0]
g_arr = np.round(np.arange(0.1, 1.05, 0.05), 2)
j_arr = [10, 20]
tlist = np.linspace(0, 10, 100000)  # Time list for SFF calculation

α = 0.6  # Filtering parameter for Dicke model

p = 2 # degree of distance

def compute_sff_distance(tlist, dicke_sff, goe_sff, p):
    """Compute the squared distance between the Dicke SFF and GOE SFF using Simpson's rule."""
    # Handle potential NaN or Inf values by replacing them with zeros
    dicke_sff = np.nan_to_num(dicke_sff)
    goe_sff = np.nan_to_num(goe_sff)

    # Use Simpson's rule for better numerical integration
    distance = (simpson((dicke_sff - goe_sff)**p, x = tlist))**(1/p)  # Integrate the squared difference

    return distance

def main():
    sff_zeta = np.zeros((len(j_arr), len(g_arr)))  # Rows: different j, Columns: different g

    for g_ind, g in enumerate(g_arr):
        for j_ind, j in enumerate(j_arr):
            # Compute SFF for Dicke model
            sff_list, eig_d = sff_list_fun_finer_tlist(ω, ω0, j, M, g, β, tlist, v, deg, unfl_proc, α = α)
            sff_rl = sff_rl_fun(tlist, sff_list)

            N = eig_d

            # Compute GOE and Poisson predictions
            Kgoe = K_GOE(tlist, N) / N**2
            Kpoi = np.abs(K_Poisson(tlist, N)) / N**2

            # Compute SFF-Zeta
            zeta = compute_sff_distance(tlist, sff_rl, Kgoe, p)
            sff_zeta[j_ind, g_ind] = zeta

    # Now plot
    fig, ax = plt.subplots(figsize=(4,3))

    for j_ind, j in enumerate(j_arr):
        ax.plot(g_arr/gc, sff_zeta[j_ind, :], 'o-', label=fr"$j={j}$", markersize=4, linewidth=1)

    ax.set_xlabel(r"$g/g_c$", fontsize=10)
    ax.set_ylabel(r"SFF-$\zeta$", fontsize=10)

    ax.set_yscale('log')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(direction='in', length=4, width=1, colors='black')

    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/Dicke_SFF_zeta_vs_g.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
