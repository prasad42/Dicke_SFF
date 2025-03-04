import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def DH_par_fun(ω, ω0, j, M, g):
    '''
    Even parity Dicke Hamiltonian for the following parameters.
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    H0 = ω * a.dag() * a + ω0 * Jz
    H1 = 1.0 / np.sqrt(2*j) * (a + a.dag()) * (Jp + Jm)
    H = H0 + g * H1
    H_even = H[::2,::2]
    
    return H_even

def eigvals_for_M_fun(ω, ω0, j, M, g):
    """
    Loads or generates the eigenvalues for a given M.
    """
    os.makedirs("evals_par", exist_ok=True)
    file_path = f"evals_par/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data for M={M}.")
        H = DH_par_fun(ω, ω0, j, M, g)
        eigvals = np.sort(sl.eigvalsh(H))
        np.save(file_path, eigvals)
    else:
        print(f"{file_path} already exists.")
        eigvals = np.load(file_path)

    return eigvals

def central_eigenvals_fun(eigvals, α):
    """
    Returns the central α% of the eigenvalue spectrum.
    """
    eig_d = len(eigvals)
    start_idx = int((1 - α) / 2 * eig_d)
    end_idx = int((1 + α) / 2 * eig_d)
    return eigvals[start_idx:end_idx]

def compute_energy_lists(ω, ω0, j, M_vals, g, α):
    """
    For each M value in M_vals, obtain the central eigenvalues.
    Returns a list of arrays.
    """
    energy_lists = []
    for M_iter in M_vals:
        eigvals = eigvals_for_M_fun(ω, ω0, j, M_iter, g)
        central_eigvals = central_eigenvals_fun(eigvals, α)
        energy_lists.append(central_eigvals)
    return energy_lists

def compute_relative_differences(energy_lists):
    """
    Given a list of energy arrays, computes the relative differences between consecutive arrays.
    Returns a 2D array of shape (n-1, min_len) where n is the number of arrays.
    """
    # Ensure all arrays have the same length (truncate to the smallest length)
    lengths = [len(arr) for arr in energy_lists]
    min_len = min(lengths)
    truncated = np.array([arr[:min_len] for arr in energy_lists])
    
    rel_diffs = []
    for i in range(len(truncated) - 1):
        E_lower = truncated[i]
        E_upper = truncated[i+1]
        # Relative difference element-wise:
        diff = np.abs(E_upper - E_lower) / ((np.abs(E_upper) + np.abs(E_lower)) / 2)
        rel_diffs.append(diff)
    return np.array(rel_diffs)

def select_converged_eigenvals(energy_lists, tol):
    """
    From a list of energy arrays, computes the maximum relative difference across
    consecutive arrays and returns the eigenvalues (from the largest M) that
    are converged (i.e. max relative difference < tol).
    """
    rel_diffs = compute_relative_differences(energy_lists)
    max_rel_diff = np.max(rel_diffs, axis=0)  # maximum diff per eigenvalue index
    converged_indices = np.where(max_rel_diff < tol)[0]
    # Return the converged eigenvalues from the largest M (last array)

    return energy_lists[-1][converged_indices]

def dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol):
    """
    Computes eigenvalues of the even parity Hamiltonian and selects converged ones
    using a relative tolerance criterion across 5 different M values.
    
    We use 5 values: [M - 4dM, M - 3dM, M - 2dM, M - dM, M].
    
    Args:
      - ω : Frequency of the bosonic field.
      - ω0 : Energy difference in spin states.
      - j : Pseudospin.
      - M : Target upper limit of bosonic Fock states.
      - g : Coupling strength.
      - α : Fraction of the spectrum to consider from the middle.
      - dM : Step size for M values.
      - tol : Relative convergence tolerance.
    
    Returns:
      - converged_eigvals : Array of converged eigenvalues (from the largest M).
    """
    # Define the 5 M values
    M_vals = [M - 4*dM, M - 3*dM, M - 2*dM, M - dM, M]
    
    # Path to save converged eigenvalues
    os.makedirs("evals_cnrgd", exist_ok=True)
    file_path = f"evals_cnrgd/evals_cnrgd_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_g={g}_α={α}_dM={dM}_tol={tol}.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, computing converged eigenvalues.")
        # Get the central eigenvalues for each M value
        energy_lists = compute_energy_lists(ω, ω0, j, M_vals, g, α)
        
        # Select and return the converged eigenvalues (from the largest M)
        converged_eigvals = select_converged_eigenvals(energy_lists, tol)
        
        # Save converged eigenvalues
        np.save(file_path, converged_eigvals)
    else:
        print(f"{file_path} already exists. Loading saved data.")
        converged_eigvals = np.load(file_path)

    eigvals = np.load(f"evals_par/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_g={g}.npy")
    print(f"Number of eigenvalues before filtrering: {len(eigvals)}")
    print(f"Number of eigenvalues after filtering: {len(converged_eigvals)}")

    return converged_eigvals

def eigval_sp_fun(eigvals, v):

    '''
    The function returns the spacings between the locally unfolded eigenvalues
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - v : Local unfolding parameter
    '''
    
    eig_d = len(eigvals)
    eigvals = unf_eigval_fun(v, eigvals)
    eigvals_sp = []
    for i in range(len(eigvals)-1):
        lvl_sp = eigvals[i+1]-eigvals[i]
        eigvals_sp.append(lvl_sp)
    eigvals_sp = np.sort(eigvals_sp)

    return eigvals_sp

def loc_den(v, i, eigvals):
    '''
    This function gives local density of states.
    Args:    
    - v : Local unfolding parameter
    - i : index of the energy level
    - eigvals : Array of energy eigenvalues
    '''
    N = len(eigvals)
    if (v < 1 or v > int(N-1)):
        raise Exception(f"Enter number v between 0 and {N}")
    
    if (i < v):
        rho_L = 2 * v /(eigvals[v+v]-eigvals[0])
    elif (i > N-1-v):
        rho_L = 2 * v /(eigvals[N-1]-eigvals[N-1-v-v])
    else:
        rho_L = 2 * v /(eigvals[i+v]-eigvals[i-v])
    
    return rho_L

def unf_eigval_fun(v, eigvals):

    """
    Unfolds the even spectrum locally and returns the unfolded spectrum
    Args:
    - v : spread of eigenvalues taken into consideration while local unfolding
    - eigvals: list of eigenvalues
    """
    # Unfolded levels
    lvl_unf = []
    unf_val = 0
    for i in (range(len(eigvals))):
        # Unfolded value of energy
        unf_val = 0
        for m in range(len(eigvals[:i])):
            # Local density of states
            rho_L = loc_den(v, m, eigvals)
            unf_val += rho_L * (eigvals[m]-eigvals[m-1])
        lvl_unf.append(unf_val)
    lvl_unf = np.sort(lvl_unf)
    
    return lvl_unf

def unf_eigval_poly_fun(deg, eigvals):
    """
    Unfolds the eigenvalue spectrum polynomially and returns the unfolded spectrum.
    
    Args:
    - deg (int): Degree of the polynomial fit.
    - eigvals (array-like): Sorted list of eigenvalues.

    Returns:
    - unfolded (np.ndarray): Unfolded spectrum.
    """
    eigvals = np.sort(eigvals)  # Ensure eigenvalues are sorted
    
    # Create the staircase function (indices as rank of eigenvalues)
    indices = np.arange(1, len(eigvals) + 1)

    # Fit a polynomial of degree `deg` to the staircase function
    coeffs = np.polyfit(eigvals, indices, deg)
    poly = np.poly1d(coeffs)

    # Compute unfolded eigenvalues
    unfolded = poly(eigvals)

    return unfolded

def eigval_sp_poly_fun(eigvals, deg):

    '''
    The function returns the spacings between the locally unfolded eigenvalues
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - v : Local unfolding parameter
    '''
    eig_d = len(eigvals)
    unfolded = unf_eigval_poly_fun(deg, eigvals)
    # Compute the nearest-neighbor spacings from the unfolded spectrum
    eigvals_sp = np.diff(unfolded)

    return eigvals_sp

def r_avg_fun(ω, ω0, j, M, g, α, dM, tol):

    '''
    Calculates the average eigenvalue spacing ratio of the spectrum
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    eigval_sp_arr = []
    r = []
    eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
    for i in range(len(eigvals)-1):
        eigval_sp_arr.append(eigvals[i+1]-eigvals[i])
    for i in range(len(eigvals)-2):
        r.append(eigval_sp_arr[i+1]/eigval_sp_arr[i])
    for i in range(len(eigvals)-2):
        if r[i] > 1:
            r[i] = 1/ r[i]
        else:
            r[i] = r[i]

    return np.average(r)

def rk_avg_fun(ω, ω0, j, M, g, α, k, dM, tol):

    '''
    Calculates the average eigenvalue spacing ratio of the spectrum
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    
    os.makedirs("r_avg",exist_ok=True)
    os.makedirs("r",exist_ok=True)

    eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
    eig_d = len(eigvals)

    file_path = f"r_avg/r_avg_ω={ω}_ω0={ω0}_j={j}_M={M}_g={g}_α={α}_k={k}_dM={dM}_tol={tol}.npy"
    file_path1 = f"r/r_ω={ω}_ω0={ω0}_j={j}_M={M}_g={g}_α={α}_k={k}_dM={dM}_tol={tol}.npy"
    if not (os.path.exists(file_path) and os.path.exists(file_path1)):
        r = np.zeros(len(eigvals)-(2*k))
        for i in range(len(eigvals)-(2*k)):
            r_val = (eigvals[i+2*k]-eigvals[i+k])/(eigvals[i+k]-eigvals[i])
            r[i] = min(r_val,1/r_val)
            # r[i] = r_val
        r_avg = np.average(r)
        np.save(file_path,r_avg)
        np.save(file_path1, r)
    else:
        r_avg = np.load(file_path)
        r = np.load(file_path1)

    return r_avg, r, eig_d

def rk_avg_goe_fun(N, ntraj, k):

    '''
    Calculates the average eigenvalue spacing ratio of the spectrum
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    
    os.makedirs("r_avg",exist_ok=True)
    os.makedirs("r",exist_ok=True)
    file_path = f"r_avg/r_avg_goe_N={N}_k={k}_ntraj={ntraj}.npy"
    file_path1 = f"r/r_goe_N={N}_k={k}.npy"
    if not (os.path.exists(file_path) and os.path.exists(file_path1)):
        print(f"{file_path} does not exist. Generating data.")
        r_traj = np.zeros(ntraj)
        for traj_ind in tqdm(range(ntraj)):
            eigvals = calc_goe_eigvals(N, traj_ind)
            # eigvals = calc_goe_eigvals_wigner(N, traj_ind)
            r = np.zeros(len(eigvals)-(2*k))
            for i in range(len(eigvals)-(2*k)):
                r_val = (eigvals[i+2*k]-eigvals[i+k])/(eigvals[i+k]-eigvals[i])
                r[i] = min(r_val,1/r_val)
            r_traj[traj_ind] = np.average(r)
        r_avg = np.average(r_traj)
        np.save(file_path,r_avg)
        np.save(file_path1, r)
    else:
        print(f"{file_path} already exists.")
        r_avg = np.load(file_path)
        r = np.load(file_path1)

    return r_avg, r

def rk_avg_poi_fun(N, ntraj, k):
    os.makedirs("r_avg",exist_ok=True)
    file_path = f"r_avg/r_avg_poi_N={N}_k={k}_ntraj={ntraj}.npy"
    file_path1 = f"r/r_poi_N={N}_k={k}.npy"
    if not (os.path.exists(file_path) and os.path.exists(file_path1)):
        print(f"{file_path} does not exist. Generating data.")
        # Construct the spectrum by cumulative summing the spacings
        r_traj = np.zeros(ntraj)
        for traj_ind in tqdm(range(ntraj)):
            eigvals = calc_poi_eigvals(N, traj_ind)
            r = np.zeros(len(eigvals)-(2*k))
            for i in range(len(eigvals)-(2*k)):
                r_val = (eigvals[i+2*k]-eigvals[i+k])/(eigvals[i+k]-eigvals[i])
                r[i] = min(r_val,1/r_val)
            r_traj[traj_ind] = np.average(r)
        r_avg = np.average(r_traj)
        np.save(file_path,r_avg)
        np.save(file_path1,r)
    else:
        print(f"{file_path} already exists.")
        r_avg = np.load(file_path)
        r = np.load(file_path1)

    return r_avg, r

def sff_list_fun(ω, ω0, j, M, g, β, tlist, α, v, deg, unfl_proc, dM, tol):
    '''
    Calculates the sff with energies of the Dicke Hamiltonian at each time step for a single trajectory.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    '''
    os.makedirs("sff",exist_ok=True)
    eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
    eig_d = len(eigvals)

    if unfl_proc=="local":
        eigvals = unf_eigval_fun(v, eigvals)
        file_path = f"sff/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_β={β}_g={g}_α={α}_v={v}_dM={dM}_tol={tol}.npy"
    elif unfl_proc == "poly":
        eigvals = unf_eigval_poly_fun(deg, eigvals)
        file_path = f"sff/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_β={β}_g={g}_α={α}_deg={deg}_dM={dM}_tol={tol}.npy"
    elif unfl_proc == None:
        eigvals = eigvals
        file_path = f"sff/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω*ω0)/2,2)}_β={β}_g={g}_α={α}_dM={dM}_tol={tol}.npy"
    
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = []
        for t in tqdm(tlist):
            sff = 0
            norm = 0
            for eigval in eigvals:
                sff += np.exp(-(β+1j*t)*(eigval))
                norm += np.exp(-β*eigval)
            sff = np.conjugate(sff)*sff/(norm**2)
            sff_list.append(sff)
            np.save(file_path,np.array(sff_list))
    else:
        print(f"{file_path} already exists.")
    sff_list = np.load(file_path)

    return sff_list, eig_d

def sff_rl_fun(tlist, sff_list, win = 50):
    '''
    This function returns the rolling average of the sff over time.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    - win: Window size for rolling average
    '''
    sff_rl = []
    for t_ind in range(0,len(tlist),1):
        win_start = int(t_ind)
        win_end = int(t_ind+win)
        sff_rl_val = np.average(sff_list[win_start:win_end], axis=0)
        sff_rl.append(sff_rl_val)

    return sff_rl

def generate_goe_matrix(N):
    """
    Generate an NxN Gaussian Orthogonal Ensemble (GOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    A = (A + A.T)/2
    return A

def calc_goe_eigvals(N, traj_ind):
    os.makedirs("evals_goe",exist_ok=True)
    file_path = f"evals_goe/evals_goe_N={N}_traj={traj_ind}"
    if not os.path.exists(file_path):
        A = generate_goe_matrix(N)
        eigvals = sl.eigvalsh(A)
        np.save(file_path,eigvals)
    else:
        eigvals = np.load(file_path)
    return eigvals

def calc_goe_eigvals_wigner(N, traj_ind):
    """
    Generate eigenvalues for a GOE matrix using the Wigner surmise.
    
    Args:
    - N : Number of eigenvalues.
    - traj_ind : Index of the trajectory (for saving/loading).

    Returns:
    - eigvals : Array of eigenvalues.
    """
    os.makedirs("evals_goe_winger", exist_ok=True)
    file_path = f"evals_goe_wigner/evals_goe_N={N}_traj={traj_ind}_wigner.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data")

        # Generate spacings using Wigner surmise distribution
        spacings = np.random.rayleigh(scale=np.sqrt(4 / np.pi), size=N)

        # Construct spectrum by cumulative sum
        eigvals = np.cumsum(spacings)
        
        # Normalise the spectrum to have zero mean
        eigvals -= np.mean(eigvals)

        np.save(file_path, eigvals)
    else:
        print(f"{file_path} already exists.")
    
    eigvals = np.load(file_path)

    return eigvals

def sff_goe_list_fun(N, β, tlist, v, deg, unfl_proc, ntraj):
    """
    Compute or reuse Spectral Form Factor (sff) for GOE matrices of size N,
    while covering nearby values using interpolation to avoid redundant calculations.
    """
    os.makedirs("sff", exist_ok=True)
    
    # Define file path for the base N computation
    if unfl_proc=="local":
        file_path = f"sff/sff_goe_N={N}_β={β}_ntraj={ntraj}_v={v}.npy"
    elif unfl_proc == "poly":
        file_path = f"sff/sff_goe_N={N}_β={β}_ntraj={ntraj}_deg={deg}.npy"
    elif unfl_proc == None:
        file_path = f"sff/sff_goe_N={N}_β={β}_ntraj={ntraj}.npy"
    
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = np.zeros_like(tlist, dtype=np.float64)
        for traj_ind in tqdm(range(ntraj)):
            eigvals = calc_goe_eigvals(N, traj_ind)
            if unfl_proc == "local":
                eigvals = unf_eigval_fun(v, eigvals)
            elif unfl_proc == "poly":
                eigvals = unf_eigval_poly_fun(deg, eigvals)
            for i, t in (enumerate(tlist)):
                exp_sum = np.sum(np.exp(-(β + 1j * t) * eigvals))
                sff_list[i] += np.abs(exp_sum) ** 2
        sff_list /= ntraj * N**2
        np.save(file_path, sff_list)
    else:
        print(f"{file_path} already exists.")
    
    # Load the stored SFF data
    sff_list = np.load(file_path)
    
    return sff_list

def calc_poi_eigvals(N, traj_ind):
    os.makedirs("evals_poi",exist_ok=True)
    file_path = f"evals_poi/evals_poi_N={N}_traj={traj_ind}"
    if not os.path.exists(file_path):
        # Parameters
        E0 = 0            # starting energy
        mean_spacing = 1  # mean spacing
        # Generate spacings: s = -mean_spacing * ln(U)
        spacings = -mean_spacing * np.log(np.random.rand(N))
        eigvals = E0 + np.cumsum(spacings)
        np.save(file_path, eigvals)
    else:
        eigvals = np.load(file_path)
        
    return eigvals

def sff_poi_list_fun(N, β, tlist, v, deg, unfl_proc, ntraj):
    """
    Compute the Spectral Form Factor (sff) for GOE matrices of size N,
    averaged over `ntraj` random GOE matrices.
    
    Args:
    - N: Size
    - β : Inverse Temperature
    - tlist: Array of time values (T) for which to compute sff.
    - ntraj: Number of GOE realizations to average over.
    
    Returns:
    - sff_list: Array of sff values for each T.
    """
    os.makedirs("sff",exist_ok=True)

    if unfl_proc=="local":
        file_path = f"sff/sff_poi_N={N}_β={β}_ntraj={ntraj}_v={v}.npy"
    elif unfl_proc == "poly":
        file_path = f"sff/sff_poi_N={N}_β={β}_ntraj={ntraj}_deg={deg}.npy"
    elif unfl_proc == None:
        file_path = f"sff/sff_poi_N={N}_β={β}_ntraj={ntraj}.npy"

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = np.zeros_like(tlist, dtype=np.float64)
        for traj_ind in tqdm(range(ntraj)):
            eigvals = calc_poi_eigvals(N, traj_ind)
            if unfl_proc=="local":
                eigvals = unf_eigval_fun(v, eigvals)
            elif unfl_proc == "poly":
                eigvals = unf_eigval_poly_fun(deg, eigvals)
            elif unfl_proc == None:
                eigvals = eigvals
            for i, t in enumerate(tlist):
                exp_sum = np.sum(np.exp(-(β + 1j*t) * eigvals))
                sff_list[i] += np.abs(exp_sum)**2
        sff_list /= ntraj * N**2 
        np.save(file_path,sff_list)
    else:
        print(f"{file_path} already exists.")

    sff_list = np.load(file_path)

    return sff_list