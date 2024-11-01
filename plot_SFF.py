# Libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# parameters
g_arr = [0.1, 0.2, 0.3, 0.4]
# g_arr = [0.6, 0.7, 0.9, 1.0]
j = 20
M = 400
N = int((2*j+1)*M/2)
beta = 0


# Data unpack
dataList_raw = []
for g in g_arr:
    data_raw = np.loadtxt(f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta}.dat", dtype=complex)
    dataList_raw.append(data_raw)

# Rolling average data
dataList = []
for g in g_arr:
    data = np.loadtxt(f"SFF/SFFvsTime,j={j},M={M},g={g},beta={beta}.dat", dtype=complex)
    data_rl = []
    # Window size for rolling average
    win = 50
    for data_ind in tqdm(range(0,len(data),1)):
        win_start = int(data_ind)
        win_end = int(data_ind+win)
        data_rl_val = np.average(data[win_start:win_end], axis=0)
        data_rl.append(data_rl_val)
    dataList.append(data_rl)

plt.figure(figsize=(10,8))
for g_ind, data in enumerate(dataList):
#     plt.suptitle(f"Dicke Model SFF (j={j},M={M})")
    plt.subplot(2,2,g_ind+1)
    plt.title(f"g={np.round(g_arr[g_ind],2)}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("SFF")
    #plt.title(f"g={g_arr[g_ind]}")
    # Plot raw data
    data_raw = dataList_raw[g_ind]
    data_raw = np.column_stack(data_raw)
    plt.plot(data_raw[0],data_raw[1],color='0.8')
    # Plot GOE 
    data1 = np.loadtxt(f"SFF_GOE/goe_sff_data,j={j},M={M},N={N},ntraj=1000.dat",dtype=complex)
    data1 = np.column_stack(data1)    
    plt.plot(data1[0],data1[1],'--k',label=f"GOE")
    # Plot moving average
    data = np.column_stack(data)
    plt.plot(data[0],data[1],label=f"Dicke Model")
    plt.xlim(1e-3,1e3)
    plt.ylim(1e-8,1)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
# plt.savefig('SFF closed Dicke chaotic phase compared to GOE')
plt.savefig('SFF closed Dicke normal phase compared to GOE')
plt.show()