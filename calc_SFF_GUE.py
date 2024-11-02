import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from parameters import *

def generate_goe_matrix(N):
    """
    Generate an NxN Gaussian Orthogonal Ensemble (GOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    A = (A + A.T) / 2
    return A

def spectral_form_factor(N, t_vals, num_realizations):
    """
    Compute the Spectral Form Factor (SFF) for GOE matrices of size N,
    averaged over `num_realizations` random GOE matrices.
    
    Args:
    - N: Size of the GOE matrix.
    - t_vals: Array of time values (T) for which to compute SFF.
    - num_realizations: Number of GOE realizations to average over.
    
    Returns:
    - sff_vals: Array of SFF values for each T.
    """
    sff_vals = np.zeros_like(t_vals, dtype=np.float64)
    
    for _ in tqdm(range(num_realizations)):
        # Generate a GOE matrix
        H = generate_goe_matrix(N)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(H)
        
        # Loop over time values (T)
        for i, T in enumerate(t_vals):
            # Compute SFF(T) = |sum_n exp(-i * lambda_n * T)|^2
            exp_sum = np.sum(np.exp(-1j * eigenvalues * T))
            sff_vals[i] += np.abs(exp_sum)**2
    
    # Average over realizations
    sff_vals /= num_realizations * N**2  # Normalize by the matrix size N
    
    return sff_vals

def main():
    
    # Generate the intervals with the specified number of points
    t_vals_0_to_01 = np.linspace(0, 0.1, 1000, endpoint=False)
    t_vals_01_to_1 = np.linspace(0.1, 1, 1000, endpoint=False)
    t_vals_1_to_10 = np.linspace(1, 10, 1000, endpoint=False)
    t_vals_10_to_100 = np.linspace(10, 100, 1000, endpoint=False)
    t_vals_100_to_1000 = np.linspace(100, 1000, 1000)
    
    # Concatenate them into a single array
    t_vals = np.concatenate([t_vals_0_to_01, t_vals_01_to_1, t_vals_1_to_10, t_vals_10_to_100, t_vals_100_to_1000])
    
    # Compute Spectral Form Factor
    sff_vals = spectral_form_factor(N, t_vals, num_realizations)

    # Combine t_vals and sff_vals into a 2D array
    data_to_save = np.column_stack((t_vals, sff_vals))

    if not os.path.exists(f"SFF/gue_sff_data,j={j},M={M},N={N},ntraj={num_realizations}.dat"):
        # Save to a text file
        np.savetxt(f"SFF/gue_sff_data,j={j},M={M},N={N},ntraj={num_realizations}.dat", data_to_save, comments="", fmt="%.8e")
    
if __name__ == "__main__":
    main()