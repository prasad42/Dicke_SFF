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
        eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
        # eigvals = unf_eigval_poly_fun(deg, eigvals)
        plt.xlabel(r"$E$")
        plt.hist(eigvals, histtype= 'step', bins=60)
    plt.show()

if __name__ == '__main__':
    main()
