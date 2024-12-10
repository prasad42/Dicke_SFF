import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    sff_goe_list = sff_goe_list_fun(j, M, β, tlist, ntraj)
    plt.figure(figsize=(int(len(g_arr)),int(len(g_arr))))
    for g_ind, g in enumerate(g_arr):
        sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist)
        sff_rl = sff_rl_fun(ω, ω0, j, M, g, β, tlist)
        plt.subplot(4,3,g_ind+1)
        plt.title(f"g={g}")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Time"); plt.ylabel("sff")
        plt.xlim(1e-3,1e3); plt.ylim(1e-8,1)
        # Plot raw data
        plt.plot(tlist,sff_list,color='0.8')
        # Plot GOE
        plt.plot(tlist, sff_goe_list,'--k',label=f"GOE")
        # Plot moving average
        plt.plot(tlist,sff_rl,label=f"Dicke Model")
        plt.tight_layout()
        
        plt.grid(True)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.legend()
    plt.savefig(f'plots/Dicke_sff_j={j}_M={M}_β={β}_gc={gc}.png')
    plt.show()

if __name__ == '__main__':
    main()