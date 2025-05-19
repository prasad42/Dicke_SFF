import numpy as np
import matplotlib.pyplot as plt	
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

ω  = 1.0; ω0 = 1.0; j = 20; M = 400; β=0; gc=np.round((np.sqrt(ω*ω0)/2),2)
g_arr = np.round(np.arange(0.1, 1.05, 0.05),2)

# Select unfolding procedure
# unfl_proc = "local"
unfl_proc = "poly"
# unfl_proc = None

# Degree of local unfolding
v = 5

# Degree of polynomial unfolding
deg = 20


# Number of random matrices to average over in GOE and GUE
ntraj = 100

# Filter the eigenvalue as per the following parameter
# One more filtering to check the convergence of the eigenvalues. (E_M1-E_M2)/Average < tol
# N = int(α*(2*j+1)*M/2)

# Number of Processes
nproc = 8

# Time list
t_vals_0_to_01 = np.linspace(0, 0.1, 1000, endpoint=False)
t_vals_01_to_1 = np.linspace(0.1, 1, 1000, endpoint=False)
t_vals_1_to_10 = np.linspace(1, 10, 1000, endpoint=False)
t_vals_10_to_100 = np.linspace(10, 100, 1000, endpoint=False)
t_vals_100_to_1000 = np.linspace(100, 1000, 1000)
tlist = np.concatenate([t_vals_0_to_01, t_vals_01_to_1, t_vals_1_to_10, t_vals_10_to_100, t_vals_100_to_1000])

# Time list for open model with MCWF method
StartTime = 0
LateTime = 100
tlist_open = np.arange(StartTime, LateTime, 0.01)

# Apply APS-friendly plot settings
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'pdf.fonttype': 42,  # Embed fonts
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
})

# APS size: one column = 8.6 cm = 3.3858 inches
fig_width = 3.3858  # inches
fig_height_per_row = 1.2  # adjust for your plots
