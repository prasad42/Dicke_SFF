import numpy as np
from open_dicke_Liou_parameters import *
from open_dicke_Liou_lib import *

# --- Parameters ---
# System parameters
ω = 1.0
ω0 = 1.2

# Dissipation rate
γ_arr = [2.2] 

# Coupling strengths to test, mapped by gamma value
g_map = {
    2.2: [0.2],
}

# Truncation dimension for the photon basis
j = 5
M_arr = [40]

def main():
    """
    Calculates and prints eigenvalues for the Dicke model using the specified parameters.
    """
    print("--- Starting Eigenvalue Calculation ---")

    # The single value for M to be used, from the M_arr list
    M = M_arr[0]
    
    # Iterate over each specified dissipation rate (gamma)
    for γ in γ_arr:
        # Get the list of coupling strengths (g) for the current gamma
        g_values_for_gamma = g_map.get(γ)
        
        if not g_values_for_gamma:
            print(f"No 'g' values found for γ = {γ}. Skipping.")
            continue
            
        # Iterate over each coupling strength (g)
        for g in g_values_for_gamma:
            print(f"\nCalculating for parameters: γ={γ}, g={g}, M={M}, j={j}")
            
            # --- Function Call ---
            # This is the specific function requested to be run.
            eigvals = Dicke_Lop_even_evals_fun(ω, ω0, j, M, g, γ)
            
            # --- Output Results ---
            print(f"Calculation complete. Found {len(eigvals)} eigenvalues.")
            # Uncomment the line below if you want to see all the eigenvalues
            print("g:\n", g)

            plt.plot(eigvals.real, eigvals.imag, 'o', label=f'g={g}, γ={γ}')
    plt.show()

    print("\n--- All calculations finished. ---")


if __name__ == "__main__":
    main()