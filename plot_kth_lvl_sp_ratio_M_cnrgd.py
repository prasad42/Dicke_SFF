import numpy as np
import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *
import warnings
warnings.filterwarnings('ignore')

def main():
    g = g_cnrgd
    # Generate distinct colors for each j using a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(j_arr)))  # You can change 'viridis' to other colormaps
    plt.title(f"k={k}, g={g}, α={α}")
    for i, j in enumerate(j_arr):  # Iterate over j values with an index
        color = colors[i]  # Assign a unique color to this iteration

        r_avg_arr = []
        r_dicke_arr = []
        
        for M in M_arr:
            dM = int(dM_per*M)
            r_avg, r_dicke, eig_d = rk_avg_fun(ω, ω0, j, M, g, α, k, dM, tol)
            r_avg_arr.append(r_avg)
            r_dicke_arr.append(r_dicke)
            print(f"g={g}, r={r_avg}")
        N = N_goe
        r_avg_goe, r_goe = rk_avg_goe_fun(N, ntraj, k)
        r_avg_poi, r_poi = rk_avg_poi_fun(N, ntraj, k)
        print(r_avg_goe, r_avg_poi)

        # Use the same color for each j
        plt.plot(M_arr, r_avg_arr, '-o', color=color, label=f"j={j}")

    plt.axhline(y=r_avg_goe, linestyle='-', color=color, alpha=0.8, 
        label=rf'$\left\langle r\right\rangle_{{\text{{GOE Num}}}}$')
    plt.axhline(y=r_avg_poi, linestyle='--', color=color, alpha=0.8, 
        label=rf'$\left\langle r\right\rangle_{{\text{{Poi Num}}}}$')

    # Constant reference lines (not changing with j)
    if k == 1:
        plt.axhline(y=0.536, linestyle='--', color='r', label=r'$\left\langle r\right\rangle_{\text{GOE}}$')
        plt.axhline(y=0.386, linestyle='--', color='k', label=r'$\left\langle r\right\rangle_{\text{Poi}}$')

    plt.xlabel('M')
    plt.ylabel(r'$\left\langle r\right\rangle$')
    plt.legend()
    plt.grid()

    # Save and show the plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f'plots/Dicke_Level_spacing_ratio_M={M}_α={α}_k={k}_dM={dM}_tol={tol}.png')
    plt.show()

if __name__=="__main__":
    main()