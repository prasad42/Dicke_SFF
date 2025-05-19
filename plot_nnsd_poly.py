import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from parameters import *
from dicke_sff_lib import *
from matplotlib.ticker import MaxNLocator, MultipleLocator  # Import Locators
from matplotlib.lines import Line2D

j = 50
g_arr = [0.1, 0.2, 0.4, 0.5, 0.7, 1.0]
α = 0.6

def main():
    num_g = len(g_arr)
    num_cols = 3
    num_rows = (num_g + num_cols - 1) // num_cols
    fig_height = num_rows * fig_height_per_row
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.4, 2.2),
                              sharex='col', sharey='row')

    axes = np.atleast_2d(axes)

    x_val = np.linspace(0, 4, 200)
    y_gauss = np.pi/2 * x_val * np.exp(-np.pi * x_val**2 / 4)
    y_poi = np.exp(-x_val)

    for g_ind, g in enumerate(g_arr):
        row, col = divmod(g_ind, num_cols)
        ax = axes[row, col]

        eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α = α)
        eigval_sp = eigval_sp_poly_fun(eigvals, deg)

        ax.plot(x_val, y_gauss, '--', color='k', label='GOE', linewidth=1)
        ax.plot(x_val, y_poi, ':', color='r', label='Poisson', linewidth=1)
        # ax.hist(eigval_sp, bins=70, histtype='step', density=True, label='Dicke Model')
        n, bins, _ = ax.hist(eigval_sp, bins=70, histtype='step', density=True, linewidth=1)

        ax.set_xlim([0, 4])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(direction='in')

        ax.text(0.25, 0.95, rf'$g/g_c = {g/gc:.2f}$',
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='grey', alpha=0.8))
         # Control the x and y ticks
        # ax.xaxis.set_major_locator(MultipleLocator(1))  # Set x-tick interval to 1
        # ax.yaxis.set_major_locator(MultipleLocator(1))  # 3 ticks on y-axis

    # Hide unused axes
    for g_ind in range(len(g_arr), num_rows * num_cols):
        fig.delaxes(axes.flat[g_ind])

    # Global labels
    fig.text(0.5, 0.02, r'$s$', ha='center', va='center', fontsize=8)
    fig.text(0.02, 0.5, r'$P(s)$', ha='center', va='center', rotation='vertical', fontsize=8)

    custom_lines = [
    Line2D([0], [0], linestyle='--', color='k', linewidth=1, label='GOE'),
    Line2D([0], [0], linestyle=':', color='r', linewidth=1, label='Poisson'),
    Line2D([0], [0], linestyle='-', linewidth=1, label='Dicke Model')
    ]
    fig.legend(custom_lines, ['GOE', 'Poisson', 'Dicke Model'], loc='upper center', ncol=3,
            frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=8)
    # # Legend
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=8)

    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15)

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f'plots/Dicke_NNSD_poly_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}_deg={deg}.pdf',
                format='pdf', bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == "__main__":
    main()