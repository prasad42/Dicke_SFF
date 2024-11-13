# Libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from parameters import *
import os
import calc_SFF_open
import calc_SFF_GOE
import math 

if __name__=='__main__':

    ##################################### Varying g for different beta for a given kappa #####################################

    # kappa = kappa_arr[-1]
    # dataList = []
    # for beta in beta_arr:
    #     dataList1 = []
    #     for g in g_arr:
    #         file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta},kappa={kappa},ntraj={ntraj}.dat"
    #         if not os.path.exists(file_path):
    #             print("Data file not found. Generating data...")
    #             print(file_path)
    #             calc_SFF_open.integrate(g, tlist, kappa, beta, ntraj)
    #             print("Data generation complete.")
    #         data = np.loadtxt(file_path)
    #         dataList1.append(data)
    #     dataList.append(dataList1)

    # plt.figure(figsize=(8,7))
    # for beta_ind, beta in enumerate(dataList):
    #     plt.subplot(math.ceil(len(beta_arr)/2),2,beta_ind+1)
    #     plt.suptitle(r"$\gamma$"+f"={kappa}")
    #     for g_ind, data in enumerate(dataList[beta_ind]):
    #         data = np.column_stack(data)
    #         plt.title(r"$\beta$" + f"={beta_arr[beta_ind]}")
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         # plt.xlim(1e-2,1e3)
    #         # plt.ylim(1e-11,3)
    #         plt.xlabel("Time")
    #         plt.ylabel("SFF")
    #         plt.plot(data[0],data[1],label=f"g={g_arr[g_ind]}")
    #         plt.tight_layout()
    #     plt.grid()
    #     if not os.path.exists("plots"):
    #         os.mkdir("plots")            
    # plt.legend()
    # plt.savefig(f"plots/SFF_mcsolve_vary_g_for_different_beta_kappa={kappa}.png")
    # plt.show()

    ##################################### Varying g for different kappas for a given betas #####################################

    beta = beta_arr[-1]
    dataList = []
    for kappa in kappa_arr:
        if kappa == 2.0:
            # g_arr = np.array([0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
            g_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
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

    plt.figure(figsize=(8,5))
    for kappa_ind, kappa in enumerate(dataList):
        plt.subplot(math.ceil(len(kappa_arr)/2),2,kappa_ind+1)
        plt.suptitle(r"$\beta$"+f"={beta}")
        for g_ind, data in enumerate(dataList[kappa_ind]):
            data = np.column_stack(data)
            plt.title(r"$\gamma$" + f"={kappa_arr[kappa_ind]}")
            plt.xscale('log')
            plt.yscale('log')
            # plt.xlim(1e-2,1e3)
            # plt.ylim(1e-11,3)
            plt.xlabel("Time")
            plt.ylabel("SFF")
            plt.plot(data[0],data[1],label=f"g={g_arr[g_ind]}")
            plt.tight_layout()
        plt.grid()
        if not os.path.exists("plots"):
            os.mkdir("plots")
    plt.legend()
    plt.savefig(f"plots/SFF_mcsolve_vary_g_for_different_kappas_beta={beta}.png")
    plt.show()

    ##################################### Varying beta for different kappas for a given g #####################################

    # g = g_arr[9]
    # dataList = []
    # for kappa in kappa_arr:
    #     dataList1 = []
    #     for beta in beta_arr:
    #         file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta},kappa={kappa},ntraj={ntraj}.dat"
    #         if not os.path.exists(file_path):
    #             print("Data file not found. Generating data...")
    #             print(file_path)
    #             calc_SFF_open.integrate(g, tlist, kappa, beta, ntraj)
    #             print("Data generation complete.")
    #         data = np.loadtxt(file_path)
    #         dataList1.append(data)
    #     dataList.append(dataList1)

    # plt.figure(figsize=(8,7))
    # plt.suptitle(f"g={g}")
    # for kappa_ind, kappa in enumerate(kappa_arr):
    #     plt.subplot(math.ceil(len(kappa_arr)/2),2,kappa_ind+1)
    #     for beta_ind, data in enumerate(dataList[kappa_ind]):
    #         data = np.column_stack(data)
    #         plt.title(r"$\gamma$" + f"={kappa_arr[kappa_ind]}")
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.xlim(1e-2,1e2)
    #         # plt.ylim(1e-11,3)
    #         plt.xlabel("Time")
    #         plt.ylabel("SFF")
    #         plt.plot(data[0],data[1],label=r"$\beta$"+f"={beta_arr[beta_ind]}")
    #         plt.tight_layout()
    #     plt.grid()
    #     if not os.path.exists("plots"):
    #         os.mkdir("plots")
    # plt.legend()
    # plt.savefig(f"plots/SFF_mcsolve_vary_beta_for_different_kappa_g={g}.png")

    # plt.show()

##################################### Varying kappa for different betas for a given g #####################################

    # g = g_arr[6]
    # dataList = []
    # for beta in beta_arr:
    #     dataList1 = []
    #     for kappa in kappa_arr:
    #         file_path = f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta},kappa={kappa},ntraj={ntraj}.dat"
    #         if not os.path.exists(file_path):
    #             print("Data file not found. Generating data...")
    #             print(file_path)
    #             calc_SFF_open.integrate(g, tlist, kappa, beta, ntraj)
    #             print("Data generation complete.")
    #         data = np.loadtxt(file_path)
    #         dataList1.append(data)
    #     dataList.append(dataList1)

    # plt.figure(figsize=(8,7))
    # plt.suptitle(f"g={g}")
    # for beta_ind, beta in enumerate(beta_arr):
    #     plt.subplot(math.ceil(len(beta_arr)/2),2,beta_ind+1)
    #     for kappa_ind, data in enumerate(dataList[beta_ind]):
    #         data = np.column_stack(data)
    #         plt.title(r"$\beta$" + f"={beta_arr[beta_ind]}")
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.xlim(1e-2,1e2)
    #         # plt.ylim(1e-11,3)
    #         plt.xlabel("Time")
    #         plt.ylabel("SFF")
    #         plt.plot(data[0],data[1],label=r"$\kappa$"+f"={kappa_arr[kappa_ind]}")
    #         plt.tight_layout()
    #     plt.grid()
    #     if not os.path.exists("plots"):
    #         os.mkdir("plots")
    # plt.legend()
    # plt.savefig(f"plots/SFF_mcsolve_vary_kappa_for_different_beta_g={g}.png")

    # plt.show()