import numpy as np
'''
-------------------------------------------------------------------------------------
				        # parameters #
-------------------------------------------------------------------------------------
'''

# SET UP THE CALCULATION

w  = 1.0
w0 = 1.0
# Coupling strengths
g_arr = np.array([0.1,0.2,0.3,0.4,0.5])
g_arr = np.concatenate((g_arr,np.array([0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3])))
# g_arr = [0.1, 1.0]
# g_arr = np.array([0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8])
# Pseudospin
j = 20
# Upper limit of bosonic fock states
M = 80
# local unfolding parameter
v = 30

# Dimension of Pseudospin
N_dim = (2*j+1)*M
# Dimension of the even subspace
N = int((2*j+1)*M/2)
# Inverse temperature for closed model
beta = 0
# Decay rate
kappa_arr = [0.01, 0.1, 1.0, 10]
kappa_arr = [1.0, 2.0]
kappa_arr = [2.0]

# Number of trajectories of MCWF method
ntraj = 100
# Values of beta for open model
beta_arr = [0.1, 2, 5, 10]
beta_arr = [10]

# Number of random matrices to average over in GOE and GUE
num_realizations = 1000

# Number of Processes
nproc = 8

# Time list
# Generate the intervals with the specified number of points
t_vals_0_to_01 = np.linspace(0, 0.1, 1000, endpoint=False)
t_vals_01_to_1 = np.linspace(0.1, 1, 1000, endpoint=False)
t_vals_1_to_10 = np.linspace(1, 10, 1000, endpoint=False)
t_vals_10_to_100 = np.linspace(10, 100, 1000, endpoint=False)
t_vals_100_to_1000 = np.linspace(100, 1000, 1000)

# Concatenate them into a single array
tlist = np.concatenate([t_vals_0_to_01, t_vals_01_to_1, t_vals_1_to_10, t_vals_10_to_100, t_vals_100_to_1000])

# Time list for open model with MCWF method
StartTime = 0
LateTime = 100
tlist_open = np.arange(StartTime, LateTime, 0.01)

# gaussian unfolding parameter
sig = 10
# ?
eta = 1