import numpy as np
import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *
import warnings
warnings.filterwarnings('ignore')

def main():
    for j in j_arr:
        r_avg_arr = []
        plt.title(f"k={k}, M={M}")
        r_dicke_arr = []
        for g in g_arr:
            r_avg, r_dicke = rk_avg_fun(ω, ω0, j, M, g, α, k)
            r_avg_arr.append(r_avg)
            r_dicke_arr.append(r_dicke)
            print(f"g={g}, r={r_avg}")
        plt.plot(g_arr,r_avg_arr,'-o',label = f"j={j}")
    ntraj = 100
    N = 2000
    r_avg_goe, r_goe = rk_avg_goe_fun(N, ntraj, k)
    r_avg_poi, r_poi = rk_avg_poi_fun(N, ntraj, k)
    print(r_avg_goe, r_avg_poi)
    plt.axhline(y=r_avg_goe,linestyle='-',color='r',label=r'$\left\langle r\right\rangle_{\text{GOE Num}}$')
    plt.axhline(y=r_avg_poi,linestyle='-',color='k',label=r'$\left\langle r\right\rangle_{\text{Poi num}}$')
    if k==1:
        plt.axhline(y=0.536,linestyle='--',color='r',label=r'$\left\langle r\right\rangle_{\text{GOE}}$')
        plt.axhline(y=0.386,linestyle='--',color='k',label=r'$\left\langle r\right\rangle_{\text{Poi}}$')
    plt.xlabel('g')
    plt.ylabel(r'$\left\langle r\right\rangle$')
    plt.legend()
    plt.grid()
    os.makedirs("plots",exist_ok=True)
    plt.savefig(f'plots/Dicke_Level_spacing_ratio_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}_α={α}_k={k}.png')
    plt.show()

    # Plot Distribution
    # num_g = len(g_arr)
    # num_rows = (num_g+1) // 2
    # plt.figure(figsize=(10,5*num_rows))
    # for g_ind, g in enumerate(g_arr):
    #     plt.subplot(num_rows, 2, g_ind+1)
    #     plt.title(f"g={g}")
    #     bins = 200
    #     plt.ylabel(r"$P(r)$")
    #     plt.xlabel("kth Level Spacing Ratio" + r" $(r_k)$")
    #     plt.hist(r_dicke_arr[g_ind], label="Dicke Model", histtype= 'step', density=True, bins=bins)
    #     plt.hist(r_goe, label="GOE", histtype= 'step', density=True, bins=bins)
    #     plt.hist(r_poi, label="Poi", histtype= 'step', density=True, bins=bins)
    #     plt.grid()
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'plots/Dicke_Level_spacing_ratio_distribution_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}_α={α}_k={k}.png')
    # plt.show()

if __name__=="__main__":
    main()