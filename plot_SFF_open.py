# Libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from parameters import *
import os
import calc_SFF_open
import math 

# Data unpack
if __name__=='__main__':
    dataList = []
    for beta in beta_arr:
        dataList1 = []
        for g in g_arr:
            file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta},kappa={kappa},ntraj={ntraj}.dat"
            if not os.path.exists(file_path):
                print("Data file not found. Generating data...")
                print(file_path)
                calc_SFF_open.integrate(g, tlist, kappa, beta, ntraj)
                print("Data generation complete.")
            data = np.loadtxt(file_path)
            dataList1.append(data)
        dataList.append(dataList1)

    plt.figure(figsize=(8,7))
    for beta_ind, beta in enumerate(dataList):
        plt.subplot(math.ceil(len(beta_arr)/2),2,beta_ind+1)
        for g_ind, data in enumerate(dataList[beta_ind]):
            data = np.column_stack(data)
            plt.title(r"$\beta$" + f"={beta_arr[beta_ind]}")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(1e-2,1e2)
            plt.ylim(1e-11,3)
            plt.xlabel("Time")
            plt.ylabel("SFF")
            #plt.ylim(1e-10,0.1e2)
            plt.plot(data[0],data[1],label=f"g={g_arr[g_ind]}")
            plt.tight_layout()
        plt.grid()
        if not os.path.exists("plots"):
            os.mkdir("plots")
    plt.legend()
    plt.savefig(f"plots/SFF_mcsolve_beta,kappa={kappa}.png")
    plt.show()