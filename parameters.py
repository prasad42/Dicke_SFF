import numpy as np
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

ω  = 1.0; ω0 = 1.0; j = 20; M = 400; β=0; gc={np.round((np.sqrt(ω*ω0)/2),2)}
g_arr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
g_arr = np.round(np.arange(0.1, 1.05, 0.05),2)
# g_arr = [0.1, 0.2, 0.4, 0.5, 0.7, 1.0]
# g_arr = [0.1, 0.2, 0.3, 0.4]
# g_arr = [0.5, 0.7, 0.8, 1.0]

# Degree of level spacing ratio
k = 10

# Select unfolding procedure
# unfl_proc = "local"
unfl_proc = "poly"
# unfl_proc = None

# Degree of local unfolding
v = 5

# Degree of polynomial unfolding
deg = 20

# For Level Spacing Ratio
j_arr = [10, 15, 20]
M_arr = np.arange(400,450,50)
g_cnrgd = 1.0
N_goe = 20000

# Number of random matrices to average over in GOE and GUE
ntraj = 100

# Filter the eigenvalue as per the following parameter
α = 0.9 # Use only this much of the eigenvalues in the center of the spectrum (discard the edge spectrum)
# One more filtering to check the convergence of the eigenvalues. (E_M1-E_M2)/Average < tol
tol = 0.1
# (M2-M1)/M2 = 0.1 
dM_per = 0.1   # Percent difference in M and M-dM
dM = int(dM_per*M)

#
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