import numpy as np
import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *
import warnings
warnings.filterwarnings('ignore')

def main():
    r_avg_arr = []
    for g in g_arr:
        r_avg = r_avg_fun(ω, ω0, j, M, g)
        r_avg_arr.append(r_avg)
    plt.plot(g_arr,r_avg_arr,'-o',label = f"M={M}")
    plt.axhline(y=0.386,linestyle='--',color='k',label=r'$\left\langle r\right\rangle_{\text{Poi}}$')
    plt.axhline(y=0.536,linestyle='-',color='r',label=r'$\left\langle r\right\rangle_{\text{GOE}}$')
    
    plt.xlabel('g')
    plt.ylabel(r'$\left\langle r\right\rangle$')
    plt.legend()
    plt.grid()
    # plt.ylim(0.38,0.54)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f'plots/Dicke_Level_spacing_ratio_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={gc}.png')
    plt.show()

if __name__=="__main__":
    main()