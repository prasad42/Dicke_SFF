'''
-------------------------------------------------------------------------------------
				        # Import Libraries #
-------------------------------------------------------------------------------------
'''

import numpy as np
import qutip as qt
import multiprocessing as mp
import time
from tqdm import tqdm
import os
from parameters import *
import runpy

'''
-------------------------------------------------------------------------------------
				        # Functions #
-------------------------------------------------------------------------------------
'''

def main():
    # create and configure the process pool
    with mp.Pool(nproc) as pool:
        args_list = []
        for g in g_arr:
            if not os.path.exists(f"evals_par/evals_g={np.round(g,2)}_j={j}_M={M}.npy"):
                runpy.run_path("calc_evals_par.py")
            eval_list = np.load(f"evals_par/evals_g={np.round(g,2)}_j={j}_M={M}.npy")
            # Prepare arguments for parallelization
            beta=0
            args_list.append([eval_list, tlist, beta, g, M, j, eta])
        # execute tasks and process results in order
        for result in pool.starmap(SFF_list, args_list, chunksize = 1):
            print(f'Result: {result}')

    return 0

'''
-------------------------------------------------------------------------------
			        # Calculate SFF#
-------------------------------------------------------------------------------	
'''

# Find the SFF of the spectrum
if __name__=='__main__':
    main()

'''
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
'''
