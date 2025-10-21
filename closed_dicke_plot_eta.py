import numpy as np
import matplotlib.pyplot as plt
from closed_dicke_parameters import *
from closed_dicke_lib import *
import os

j_arr = [10, 50, 100]
gc = np.sqrt(ω*ω0)/2
g_arr = {
    10:  np.round(np.arange(0.1, 1.05, 0.05),2),
    15:  np.round(np.arange(0.1, 1.05, 0.05),2),
    20:  np.round(np.arange(0.1, 1.05, 0.05),2),
    50:  np.round(np.arange(0.1, 1.05, 0.05),2),
    100: np.round(np.arange(0.1, 1.05, 0.05),2),
}
# j_arr = [10, 15, 20]
M=400
N_goe = 20000
α = 0.6

def main():
    # Set up APS-style figure
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)

    markers = ['o', 's', '^', 'D', 'v', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, len(j_arr)))

    for i, j in enumerate(j_arr):
        eta_arr = []
        g_arr_for_j = g_arr[j]

        for g in g_arr_for_j:
            eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α=α)
            unfolded_spacings = eigval_sp_poly_fun(eigvals)
            eta = compute_eta(unfolded_spacings, bins=100)
            eta_arr.append(eta)

        # Plot
        ax.plot(g_arr_for_j / gc, eta_arr, markersize=5, marker=markers[i],
                color=colors[i], label=fr"$j={j}$")

    # Axis formatting
    ax.set_xlabel(r"$g / g_c$", fontsize=10)
    ax.set_ylabel(r"$\eta$", fontsize=10)
    ax.set_ylim(-0.1, 1.25)
    ax.tick_params(direction='in', length=4, width=1, labelsize=8)

    # Reference lines
    ax.axhline(1, linestyle='--', color="k", label="GOE")
    ax.axhline(0, linestyle=':', color="r", label="Poisson")
    ax.axvline(1, linestyle='--', color="black", alpha=0.3)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend above, but within figure boundary
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))

    # Ensure all fits well
    fig.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for legend

    # Save neatly
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f'plots/Dicke_eta_M={M}_α={np.round(α,2)}.pdf',
                dpi=600, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()
