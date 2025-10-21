import numpy as np
import matplotlib.pyplot as plt
from closed_dicke_parameters import *
from closed_dicke_lib import *
import os

j = 10
gc = np.sqrt(ω*ω0)/2
g_arr = np.round(np.arange(0.1, 1.05, 0.05),2)
# j_arr = [10, 15, 20]
M_arr = [80, 400]
N_goe = 20000
α = 0.6

def main():
    # Set up the figure in APS style (about 3.4 inches wide)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    # Define marker and color styles for the different gamma values
    markers = ['o', 's', '^', 'D', 'v', '*']  # Circle, square, triangle_up, diamond, triangle_down, star
    colors = plt.cm.viridis(np.linspace(0, 1, len(M_arr)))  # Mild colormap

    for i, M in enumerate(M_arr):
        eta_arr = []
        
        for g in g_arr:
            eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α = α)
            unfolded_spacings = eigval_sp_poly_fun(eigvals)
            eta = compute_eta(unfolded_spacings)
            eta_arr.append(eta)

        # Plot the eta vs g/gc curve
        plt.plot(g_arr / gc, eta_arr, markersize=6, marker=markers[i], color=colors[i], label=f"M={M}")

        # Add a grid with dashed lines and reduced opacity
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Set ticks and their parameters to be APS-friendly (inward direction, size 4, etc.)
        plt.tick_params(direction='in', length=4, width=1, colors='black', grid_color='gray', grid_alpha=0.5, labelsize=8)

        # Adjust layout to ensure everything fits neatly
        plt.tight_layout()

    # Labels and title with APS-friendly font sizes
    plt.xlabel(r"$g / g_c$", fontsize=10)
    plt.ylabel(r"$\eta$", fontsize=10)
    plt.ylim(-0.1,1.55)
    # Add horizontal lines for GOE and Poisson
    plt.axhline(1, linestyle='--', color="k", label="GOE")
    plt.axhline(0, linestyle=':', color="r", label="Poisson")

    # Add vertical line for g = gc
    plt.axvline(1, linestyle='--', color="black", alpha=0.3)
    
    # Set the legend
    # plt.legend(fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.7, 0.5), fontsize=8)

    # Save the plot as a high-quality PDF
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f'plots/Dicke_eta_j={j}_M={M}_α={np.round(α,2)}.pdf', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
