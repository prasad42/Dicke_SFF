import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    num_g = len(g_arr)
    num_rows = (num_g + 1) // 2
    plt.figure(figsize=(10,5*num_rows))

    # if unfl_proc == "local":
    #     plt.suptitle(f"j={j},M={M},α={α},v={v},dM={dM},tol={tol}")
    # elif unfl_proc == "poly":
    #     plt.suptitle(f"j={j},M={M},α={α},deg={deg},dM={dM},tol={tol}")
    # elif unfl_proc == None:
    #     plt.suptitle(f"j={j},M={M},α={α},dM={dM},tol={tol}")
    for g_ind, g in enumerate(g_arr):
        sff_list, eig_d = sff_list_fun(ω, ω0, j, M, g, β, tlist, α, v, deg, unfl_proc, dM, tol)
        sff_rl = sff_rl_fun(tlist, sff_list)

        N = eig_d

        sff_goe_list = sff_goe_list_fun(N, β, tlist, v, deg, unfl_proc, ntraj)
        sff_poi_list = sff_poi_list_fun(N, β, tlist, v, deg, unfl_proc, ntraj)

        plt.subplot(num_rows,2,g_ind+1)
        plt.title(f"g={g}")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Time"); plt.ylabel("SFF")
        # plt.xlim(1e-3,1e3); plt.ylim(1e-8,1)
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

    if unfl_proc == "local":
        plt.savefig(f'plots/Dicke_sff_j={j}_M={M}_β={β}_α={α}_v={v}_dM={dM}_tol={tol}.png')
    elif unfl_proc == "poly":
        plt.savefig(f'plots/Dicke_sff_j={j}_M={M}_β={β}_α={α}_deg={deg}_dM={dM}_tol={tol}.png')
    elif unfl_proc == None:
        plt.savefig(f'plots/Dicke_sff_j={j}_M={M}_β={β}_α={α}_dM={dM}_tol={tol}.png')

    plt.show()

if __name__ == '__main__':
    main()