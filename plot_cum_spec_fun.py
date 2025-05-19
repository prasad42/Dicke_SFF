from dicke_sff_lib import *
from parameters import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    for g in g_arr:
        eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
        eigvals = np.sort(eigvals)  # Ensure eigenvalues are sorted
        
        # Create the staircase function (indices as rank of eigenvalues)
        indices = np.arange(1, len(eigvals) + 1)

        plt.ylabel(r"$I(E)$")
        plt.xlabel(r"$E$")
        
        # Use `plt.step()` for a proper staircase plot
        plt.step(eigvals, indices, where="post", label=f"g={g}")
        unfolded_eigvals = unf_eigval_poly_fun(deg, eigvals)
        plt.plot(eigvals, unfolded_eigvals)
        plt.grid()

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
