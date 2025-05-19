import numpy as np
import matplotlib.pyplot as plt
import os

from dicke_sff_lib import *
from parameters import *

# Global settings (APS-like)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1,
    "lines.markersize": 4,
})

α = 0.6  # Filtering parameter for Dicke model
tol = 1e-4  # Tolerance for Heisenberg time detection

# Define g values to scan
g_arr = np.round(np.arange(0.1, 1.05, 0.05), 2)

# j values
j_arr = [10, 15, 20]  # Add more if you want

def find_heisenberg_time(tlist, sff_avg, t_min=1.0, tolerance=1e-4):
    """ Find Heisenberg time where SFF saturates near its late-time value after t_min. """
    tlist = np.array(tlist)
    sff_avg = np.array(sff_avg)

    mask = tlist >= t_min
    tlist_filtered = tlist[mask]
    sff_avg_filtered = sff_avg[mask]

    # Late-time average
    late_time_mask = tlist_filtered >= (0.9 * tlist_filtered[-1])
    sff_late_avg = np.mean(sff_avg_filtered[late_time_mask])

    # Find first time where SFF is within tolerance of late-time average
    diff = np.abs(sff_avg_filtered - sff_late_avg)
    within_tol_indices = np.where(diff <= tolerance)[0]

    if len(within_tol_indices) == 0:
        return np.nan
    else:
        return tlist_filtered[within_tol_indices[0]]

def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    markers = ['o', 's', '^']  # Different markers for different j
    colors = plt.cm.viridis(np.linspace(0, 1, len(j_arr)))  # Different colors for different j

    for j_ind, j in enumerate(j_arr):
        t_H_list = []

        for g in g_arr:
            print(f"Computing for g = {g:.2f}, j = {j}...")

            # Compute SFF
            sff_list, eig_d = sff_list_fun(ω, ω0, j, M, g, β, tlist, v, deg, unfl_proc, α=α)
            sff_rl = sff_rl_fun(tlist, sff_list)
            N = eig_d

            # Find Heisenberg time
            t_H = find_heisenberg_time(tlist, sff_rl, t_min=1.0, tolerance=tol)
            t_H_list.append(t_H)

        # Plot for this j
        ax.plot(g_arr/gc, t_H_list, marker=markers[j_ind], color=colors[j_ind], label=fr"$j={j}$")

    # Plot GOE Heisenberg time
    ax.axhline(2*np.pi, color='gray', linestyle='--', linewidth=1, label=r"$2\pi$ (GOE)")
    ax.axvline(1, color='gray', linestyle=':', linewidth=1, label=f"critical coupling")
    fig.text(0.5, 0.01, r'$g/g_c$', ha='center', va='bottom', fontsize=11)  # x-axis at bottom
    fig.text(0.01, 0.5, r'$t_H$', ha='center', va='center', rotation='vertical', fontsize=11)
    # ax.set_title(r"Heisenberg time $t_H$ vs $g$")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=True, loc='best', ncol=1, fontsize = 8)

    # Layout tight
    plt.tight_layout()

    # Save
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/Dicke_tH_vs_g_j={j_arr}_M={M}_β={β}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
