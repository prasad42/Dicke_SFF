import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    sff_goe_list = sff_goe_list_fun(N, β, tlist, v, ntraj)
    sff_poi_list = sff_poi_list_fun(N, β, tlist, v, ntraj)
    num_g = len(g_arr)
    num_rows = (num_g + 1) // 2
    plt.figure(figsize=(10,5*num_rows))
    plt.suptitle(f"j={j},M={M},α={α},v={v}")
    for g_ind, g in enumerate(g_arr):
        sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist, α, v)
        sff_rl = sff_rl_fun(tlist, sff_list)
        plt.subplot(num_rows,2,g_ind+1)
        plt.title(f"g={g}")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Time"); plt.ylabel("sff")
        plt.xlim(1e-3,1e3); plt.ylim(1e-8,1)
        # Plot raw data
        plt.plot(tlist,sff_list,color='0.8')
        # Plot GOE
        plt.plot(tlist, sff_goe_list,'--k',label=f"GOE")
        # Plot POI
        plt.plot(tlist, sff_poi_list,'--r',label=f"Poi")
        # Plot moving average
        plt.plot(tlist,sff_rl,label=f"Dicke Model")
        plt.tight_layout()
        plt.grid(True)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.legend()
    plt.savefig(f'plots/Dicke_sff_j={j}_M={M}_β={β}_gc={gc}_v={v}.png')
    plt.show()
    # g = 0.1
    # eigvals = dicke_eigvals_fun(ω, ω0, j, M, g)
    # plt.hist(eigvals, bins=60)
    # plt.show()

if __name__ == '__main__':
    main()