# Libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from parameters import *
import os
import calc_SFF_open

# Data unpack
if __name__=='__main__':
    dataList = []
    for beta in beta_arr:
        dataList1 = []
        for g in g_arr:
            file_path = f"SFF/SFFvsTime,kappa={kappa},j={j},M={M},g={g},beta={beta},ntraj={ntraj}.dat"
            if not os.path.exists(file_path):
                print("Data file not found. Generating data...")
                calc_SFF_open.main_par()
                print("Data generation complete.")
            data = np.loadtxt(file_path)
            dataList1.append(data)
        dataList.append(dataList1)

    plt.figure(figsize=(8,7))
    for beta_ind, beta in enumerate(dataList):
        for g_ind, data in enumerate(dataList[beta_ind]):
            data = np.column_stack(data)
            #plt.title(f"Open Dicke Model SFF (MCWF) j={j} M={M} " r"$\beta$" + f"={beta} "  r"$\gamma$" + f"={kappa} " r"$g_c$" + f"={0.7}")
            plt.subplot(1,1,beta_ind+1)
            plt.title(r"$\beta$" + f"={beta_arr[beta_ind]}")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(1e-2,1e2)
            plt.xlabel("Time")
            plt.ylabel("SFF")
            #plt.ylim(1e-10,0.1e2)
            plt.plot(data[0],data[1],label=f"g={g_arr[g_ind]}")
            plt.tight_layout()
        plt.grid()
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig("plots/2_SFF_mcsolve_b_vals")
    plt.legend()
    plt.show()