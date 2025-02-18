import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_sff_lib import *

def main():
    num_g = len(g_arr)
    num_rows = (num_g+1) // 3
    plt.figure(figsize=(10,4*num_rows))
    for g_ind, g in enumerate(g_arr):
        eigvals = dicke_eigvals_fun(ω, ω0, j, M, g)
        eigval_sp = eigval_sp_poly_fun(eigvals, α, deg)
        # Reference plots
        x_val = np.arange(0,4,0.1)
        y_gauss = np.pi/2 * x_val * np.exp(-np.pi*x_val**2/4)
        y_poi = np.exp(-x_val)
        # Create a histogram
        plt.subplot(num_rows,3,g_ind+1)
        plt.title(f'g={g}')
        plt.plot(x_val, y_gauss, linestyle = '--', label = 'Gaussian', linewidth = 1)
        plt.plot(x_val, y_poi, label = 'Poisson', linewidth = 1)
        hist_values,bin_edges, _ = plt.hist(eigval_sp, bins=70, histtype= 'step',label="Dicke Model", density=True)
        plt.xlabel('s')
        plt.ylabel('P(s)')
        plt.xlim([0,4])
        plt.grid()
        plt.tight_layout()
    plt.legend()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f'plots/Dicke_NNSD_poly_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}_α={α}_deg={deg}.jpg')
    plt.show()

if __name__=="__main__":
    main()