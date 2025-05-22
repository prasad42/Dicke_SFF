import matplotlib.pyplot as plt
from closed_dicke_lib import *
from closed_dicke_parameters import *
import os
import numpy as np

g_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0]
# g_arr = [0.2, 0.85]
# g_arr = [0.2, 1.0]
# g_arr = [1.0, 2.0, 3.0, 4.0]
j_arr = [50]
M = 400
unfl_proc = "poly"
α = 0.6  # Filtering parameter for Dicke model

def main():
    num_g = len(g_arr)//2

    # Create subplots with shared x and y axes
    fig, axes = plt.subplots(2, num_g, figsize=(3.4*2, 3.3), sharex='col', sharey='row')
    axes = axes.flatten()

    for g_ind, g in enumerate(g_arr):
        ax = axes[g_ind]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-7,1.1e0)
        ax.set_xlim(0.5e-3,1e3)

        # Place the label inside the subplot
        ax.text(0.5, 0.9, r"$g/g_c=$" + f"{g/gc:.1f}", transform=ax.transAxes,
                fontsize=8, va='top', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5))

        for j in j_arr:
            # Compute quantities for each j separately
            sff_list, eig_d = sff_list_fun(ω, ω0, j, M, g, β, tlist, v, deg, unfl_proc, α = α, tol=0.1)
            sff_rl = sff_rl_fun(tlist, sff_list)
            N = eig_d

            # Plot raw SFF data (grey background curve)
            ax.plot(tlist, sff_list, color='0.8', linewidth=1)

            # Plot Dicke model moving average
            ax.plot(tlist, sff_rl, label=fr"Dicke, $j={j}$", linewidth=0.8)

        # Plot GOE and Poisson theories (only once per g)
        Kgoe = K_GOE(tlist, N) / N**2
        Kpoi = np.abs(K_Poisson(tlist, N)) / N**2

        ax.plot(tlist, Kgoe, '--k', label="GOE Theory", linewidth=0.8)
        ax.plot(tlist, Kpoi, ':r', label="Poisson Theory", linewidth=0.8)

        # Grid, ticks
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(direction='in', length=4, width=1, colors='black', grid_color='gray', grid_alpha=0.5, labelsize=8)

    # Global labels
    fig.text(0.5, 0.07, 'Time t', ha='center', va='center', fontsize=9)
    fig.text(0.07, 0.5, fr'SFF(t)', ha='center', va='center', rotation='vertical', fontsize=9)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.98))

    # Layout adjustments
    plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.15, top=0.85)

    # Save
    os.makedirs("plots", exist_ok=True)
    if unfl_proc == "local":
        plt.savefig(f'plots/Dicke_sff_j={j_arr}_M={M}_β={β}_v={v}.pdf', dpi=300, bbox_inches='tight')
    elif unfl_proc == "poly":
        plt.savefig(f'plots/Dicke_sff_j={j_arr}_M={M}_β={β}_deg={deg}.pdf', dpi=300, bbox_inches='tight')
    elif unfl_proc == None:
        plt.savefig(f'plots/Dicke_sff_j={j_arr}_M={M}_β={β}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
