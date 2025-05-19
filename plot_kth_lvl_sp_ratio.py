import numpy as np
import matplotlib.pyplot as plt
import os
from dicke_sff_lib import *
from parameters import *
import warnings
warnings.filterwarnings('ignore')

# Parameters
j_arr = [10, 50, 100]
M_arr = np.arange(400, 450, 50)
g_arr = {
    10:  np.round(np.arange(0.1, 1.05, 0.05), 2),
    50:  np.round(np.arange(0.1, 1.05, 0.05), 2),
    100: np.round(np.arange(0.1, 1.05, 0.05), 2),
}
k_arr = [1, 10, 20, 30]
N_goe = 1000
α = 0.6
colors = plt.cm.viridis(np.linspace(0, 1, len(j_arr)))
markers = ['o', 's', '^']  # Marker styles for each j

def main():
    fig, axs = plt.subplots(1, 4, figsize=(6.8, 1.8), sharex=True, sharey=False)
    axs = axs.flatten()

    for idx, k in enumerate(k_arr):
        ax = axs[idx]
        N = N_goe
        r_avg_goe, _ = rk_avg_goe_fun(N, ntraj, k)
        r_avg_poi, _ = rk_avg_poi_fun(N, ntraj, k)

        ax.axhline(y=r_avg_goe, linestyle='--', color='k', alpha=0.8, label='GOE' if idx == 0 else "")
        ax.axhline(y=r_avg_poi, linestyle=':', color='r', alpha=0.8, label='Poisson' if idx == 0 else "")
        ax.axvline(x=1, linestyle='--', color='gray', alpha=0.3)

        for i, j in enumerate(j_arr):
            r_avg_arr = []
            g_arr_for_j = g_arr[j]
            for g in g_arr_for_j:
                r_avg, _, _ = rk_avg_fun(ω, ω0, j, M, g, k, α=α)
                r_avg_arr.append(r_avg)
            ax.plot(g_arr_for_j / gc, r_avg_arr, marker=markers[i], color=colors[i], label=rf"$j={j}$" if idx == 0 else "")

        ax.text(0.5, 0.5, rf"$\langle r_{{{k}}} \rangle$", transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='grey'))
        ax.tick_params(labelsize=7, direction='in', length=3, width=0.8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Global labels
    fig.text(0.5, 0.01, r"$g/g_c$", ha='center', va='bottom', fontsize=9)
    fig.text(0.05, 0.5, r"$\langle r_k \rangle$", ha='center', va='center', rotation='vertical', fontsize=9)

     # Legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.1))

    fig.tight_layout(rect=[0.05, 0.05, 1, 1])
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f'plots/Dicke_Level_spacing_ratio_subplots_M={M}_α={np.round(α,2)}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
