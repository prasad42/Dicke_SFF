# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from parameters import *
import os
import runpy

# Calculate Level Spacing Ratio
dataList = []
for g in g_arr:
    if os.path.exists(f"evals_par/evals_g={g}_j={j}_M={M}.npy"):
        evals = np.load(f"evals_par/evals_g={g}_j={j}_M={M}.npy")
    else:
        runpy.run_path("calc_SFF_GOE")
        evals = np.load(f"evals_par/evals_g={g}_j={j}_M={M}.npy")
    data = np.real(evals)
    dataList.append(data)

delta_arr = []
for g_ind, data in enumerate(dataList):
    lvl_sp_arr = []
    delta = []
    for i in range(len(data)-1):
        lvl_sp_arr.append(data[i+1]-data[i])
    for i in range(len(data)-2):
        delta.append(lvl_sp_arr[i+1]/lvl_sp_arr[i])
    for i in range(len(data)-2):
        if delta[i] > 1:
            delta[i] = 1/ delta[i]
        else:
            delta[i] = delta[i]
    delta_arr.append(delta)
# Average Level Spacing Ratio
delta_avg_arr = []
for g_ind, g in enumerate(g_arr):
    delta_avg_arr.append(np.average(delta_arr[g_ind]))

# Plot
plt.plot(g_arr,delta_avg_arr,'-o')
plt.axhline(y=0.386,linestyle='--',color='k',label=r'$\left\langle r\right\rangle_{\text{Poi}}$')
plt.axhline(y=0.536,linestyle='-',color='r',label=r'$\left\langle r\right\rangle_{\text{GOE}}$')
plt.xlabel('g')
plt.ylabel(r'$\left\langle r\right\rangle$')
plt.legend()
plt.grid()
plt.ylim(0.38,0.54)

if not os.path.exists("plots"):
    os.mkdir("plots")
plt.savefig('plots/Level_spacing_ratio')
plt.show()