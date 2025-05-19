import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *
import os

N_goe = 6607
deg = 20

def main():
    num_g = len(g_arr)
    num_rows = (num_g + 1) // 2

    # Create subplots with shared x and y axes
    fig, axes = plt.subplots(1, 1, figsize=(3.4, 3.3), sharex='col', sharey='row')

    sff_goe_list = sff_goe_list_fun(N_goe, β, tlist, v, deg, unfl_proc, ntraj)
    # sff_poi_list = sff_poi_list_fun(N_goe, β, tlist, v, deg, unfl_proc, ntraj)

    # sff_goe_list = sff_goe_list_fun(N, β, tlist, v, deg, unfl_proc, ntraj)

    Kgoe = K_GOE(tlist, N_goe) / N_goe**2
    Kpoi = np.abs(K_Poisson(tlist, N_goe)) / N_goe**2  # Poisson is complex-valued, take real part

    ax = axes
    # Place the label inside the subplot (adjust x, y for positioning)
    # ax.text(0.5, 0.9, r"${g}/{g_c}=$" + f"{g/gc}", ha='center', va='center', fontsize=8, transform=ax.transAxes)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # ax.set_xlabel('Time', fontsize=8)
    # ax.set_ylabel('SFF', fontsize=8)

    # Plot goe data
    ax.plot(tlist, sff_goe_list, label='GOE Numerical', linewidth=1)
    ax.plot(tlist, Kgoe, '--k', label='GOE Theory', linewidth=0.6)  # Dashed line for GOE
    # ax.plot(tlist, Kpoi, ':r', label='Poisson Theory', linewidth=0.6)  # Dotted line for Poisson

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Set ticks to be directed inward (as per APS style)
    ax.tick_params(direction='in', length=4, width=1, colors='black', grid_color='gray', grid_alpha=0.5)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Global figure labels (these are outside of individual subplots)
    fig = plt.gcf()
    fig.text(0.5, 0.02, 'Time', ha='center', va='center', fontsize=8)
    fig.text(0.02, 0.5, 'SFF', ha='center', va='center', rotation='vertical', fontsize=8)

    # Add legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    plt.legend()

    ax.set_xlim(1e-4, 1e3)  # Set x-axis limits
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15, top=0.9)  # Adjust space between subplots


    # Save the figure as a PDF with APS-style adjustments
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f'plots/GOE_sff_N={N_goe}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()