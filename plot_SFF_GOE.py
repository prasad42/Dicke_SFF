# Libraries
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
import os
import runpy

# Check if data file exists; generate data if not
data_file = f"SFF/goe_sff_data_j={j},M={M},N={N}_ntraj={num_realizations}.dat"
if not os.path.exists(data_file):
    print("Data file not found. Generating data...")
    runpy.run_path("calc_SFF_GOE.py",init_globals={'__name__': '__main__'})
    print("Data generation complete.")

data = np.loadtxt(f"SFF/goe_sff_data_j={j},M={M},N={N},ntraj={num_realizations}.dat")

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