# Libraries
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
import os
import runpy

# Combine t_vals and sff_vals into a 2D array
if os.path.exists(f"SFF_GOE/goe_sff_data_j={j},M={M},N={N}_ntraj={num_realizations}.dat"):
    data = np.loadtxt(f"SFF_GOE/goe_sff_data_j={j},M={M},N={N}_ntraj={num_realizations}.dat")
else:
    runpy.run_path("calc_SFF_GOE.py")
    data = np.loadtxt(f"SFF_GOE/goe_sff_data_j={j},M={M},N={N}_ntraj={num_realizations}.dat")

data = np.column_stack(data)
t_vals = data[0]
sff_vals = data[1]

# Plot the results
plt.plot(t_vals, sff_vals, label=f'GOE (N={N})')
plt.xlabel('T')
plt.ylabel('Spectral Form Factor (SFF)')
plt.title('Spectral Form Factor for GOE')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
if not os.path.exists("plots"):
    os.mkdir("plots")
plt.savefig(f"plots/SFF_GOE")
plt.show()