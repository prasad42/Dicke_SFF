# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Parameters
j = 2
M = 4
N = int((2*j+1)*M/2)  # Size of GOE matrix
num_realizations = 1  # Number of random matrices to average over

# Combine t_vals and sff_vals into a 2D array
data = np.loadtxt(f"SFF_GUE/gue_sff_data_j={j},M={M},N={N}_ntraj={num_realizations}.dat")
data = np.column_stack(data)
t_vals = data[0]
sff_vals = data[1]

# Plot the results
plt.plot(t_vals, sff_vals, label=f'GUE (N={N})')
plt.xlabel('T')
plt.ylabel('Spectral Form Factor (SFF)')
plt.title('Spectral Form Factor for GUE')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()