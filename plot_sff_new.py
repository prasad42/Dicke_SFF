import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

# Shared variable for clicked x-coordinate
x_click = None

# Click handler
def on_click(event):
    global x_click
    if event.inaxes:  # Check if the click is inside the plot axes
        x_click = event.xdata  # Store the x-coordinate
        plt.close()  # Close the plot after clicking

def ramp_slope_fun(tlist, sff_list, sff_goe_list):
    """
    Returns the optimal slope for the ramp of the SFF.
    """
    global x_click

    # Click on the start of the ramp
    x_click = None
    plt.figure(figsize=(8, 5))
    plt.title("Select the start of the ramp")
    plt.plot(tlist, sff_goe_list, "--k")
    plt.plot(tlist, sff_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise RuntimeError("No click detected for ramp start.")
    ramp_start = x_click
    ramp_start_idx = int(np.argmin(np.abs(tlist - ramp_start)))  # Find the closest index
    print(f"Ramp start: x={ramp_start:.2f}, index={ramp_start_idx}")

    # Click on the end of the ramp
    x_click = None
    plt.figure(figsize=(8, 5))
    plt.title("Select the end of the ramp")
    plt.plot(tlist, sff_goe_list, "--k")
    plt.plot(tlist, sff_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise RuntimeError("No click detected for ramp end.")
    ramp_end = x_click
    ramp_end_idx = int(np.argmin(np.abs(tlist - ramp_end)))  # Find the closest index
    print(f"Ramp end: x={ramp_end:.2f}, index={ramp_end_idx}")

    # Ensure valid indices
    if ramp_end_idx <= ramp_start_idx:
        raise ValueError("Ramp end index must be greater than ramp start index.")

    # Slope of the ramp
    m, b = np.polyfit(
        np.log10(tlist[ramp_start_idx:ramp_end_idx]),
        np.log10(sff_list[ramp_start_idx:ramp_end_idx]),
        1
        )
    print(f"Slope of the ramp: {m}")

    return m

def select_ramp_fun(tlist, sff_list):
    """
    Returns the optimal slope for the ramp of the SFF.
    """
    global x_click

    # Click on the start of the ramp
    x_click = None
    plt.figure(figsize=(8, 5))
    plt.title("Select the start of the ramp")
    plt.plot(tlist, sff_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise RuntimeError("No click detected for ramp start.")
    ramp_start = x_click
    ramp_start_idx = int(np.argmin(np.abs(tlist - ramp_start)))  # Find the closest index
    print(f"Ramp start: x={ramp_start:.2f}, index={ramp_start_idx}")

    # Click on the end of the ramp
    x_click = None
    plt.figure(figsize=(8, 5))
    plt.title("Select the end of the ramp")
    plt.plot(tlist, sff_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if x_click is None:
        raise RuntimeError("No click detected for ramp end.")
    ramp_end = x_click
    ramp_end_idx = int(np.argmin(np.abs(tlist - ramp_end)))  # Find the closest index
    print(f"Ramp end: x={ramp_end:.2f}, index={ramp_end_idx}")

    # Ensure valid indices
    if ramp_end_idx <= ramp_start_idx:
        raise ValueError("Ramp end index must be greater than ramp start index.")

    return ramp_start_idx, ramp_end_idx

def plot_sff(M, g, tlist, sff_goe_list):

    plt.figure(figsize=(5, 20))
    sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist)
    sff_rl_list = sff_rl_fun(tlist, sff_list)
    plt.title(f"g={g}")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Time"); plt.ylabel("sff")
    plt.xlim(1e-3,1e3); plt.ylim(1e-8,1)
    # Plot raw data
    plt.plot(tlist,sff_list,color='0.8')
    # Plot GOE
    plt.plot(tlist, sff_goe_list,'--k',label=f"GOE")
    # Plot moving average
    plt.plot(tlist,sff_rl_list,label=f"Dicke Model")
    plt.tight_layout()
    plt.grid(True)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.legend()
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    return 0

def Dicke_sff_ramp_fun():
    # Find the optimal ramp slope
    m_arr_arr = []
    for M in M_arr:
        sff_goe_list = sff_goe_list_fun(j, M, β, tlist)
        m_arr = []
        for g in g_arr:
            sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist)
            sff_rl_list = sff_rl_fun(tlist, sff_list)
            m = ramp_slope_fun(tlist, sff_rl_list, sff_goe_list)
            m_arr.append(m)
        m_arr_arr.append(m_arr)
    
    return m_arr_arr

def sff_distance_fun():
    """
    Computes the distance between raw SFF values (GOE vs Dicke) over a specified range.

    Parameters:
    - tlist: Array of time points.
    - sff_goe: GOE raw SFF values corresponding to tlist.
    - sff_dicke: Dicke raw SFF values corresponding to tlist.
    - ramp_start_idx: Start index for the ramp comparison.
    - ramp_end_idx: End index for the ramp comparison.

    Returns:
    - pointwise_distances: Pointwise differences between the two SFF arrays.
    - total_distance: Euclidean distance (L2 norm) between the raw SFF values.
    """

    dist_arr_arr = []
    for M in M_arr:
        sff_goe_list = sff_goe_list_fun(j, M, β, tlist)
        # Subset the SFF and tlist to the desired range
        ramp_start_idx, ramp_end_idx = select_ramp_fun(tlist, sff_goe_list)
        sff_goe_subset = sff_goe_list[ramp_start_idx:ramp_end_idx]
        dist_arr = []
        for g in g_arr:
            sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist)
            sff_rl_list = sff_rl_fun(tlist, sff_list)
            # ramp_start_idx, ramp_end_idx = select_ramp_fun(tlist, sff_rl_list)
            sff_dicke_subset = sff_list[ramp_start_idx:ramp_end_idx]
            # Compute total distance (L2 norm)
            dist = np.linalg.norm(sff_goe_subset - sff_dicke_subset)
            dist_arr.append(dist)
        dist_arr_arr.append(dist_arr)

    return dist_arr_arr

def main():
    m_arr_arr = Dicke_sff_ramp_fun()
    # Plot slope of the ramp as function of g for various M's
    plt.figure(figsize=(8,5))
    plt.title("distance between GOE and Dicke SFF ramps")
    for idx, M in enumerate(M_arr):
        m_arr = m_arr_arr[idx]
        plt.axhline(1, linestyle='--', color="k", label="GOE")
        plt.plot(g_arr,m_arr,'.-', label=f'M={M_arr[idx]}')
        plt.xlabel('g')
        plt.ylabel('slope of the ramp')
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/Dicke_sff_j={j}_slope_ramp_vs_g_new")
    plt.show()

# def main():
#     dist_arr_arr = sff_distance_fun()
#     # Plot slope of the ramp as function of g for various M's
#     plt.figure(figsize=(8,5))
#     plt.title("distance between GOE and Dicke SFF ramps")
#     for idx, M in enumerate(M_arr):
#         dist_arr = dist_arr_arr[idx]
#         plt.plot(g_arr, dist_arr)
#         plt.xlabel('g')
#         plt.ylabel('distance')
#         plt.grid()
#         plt.savefig(f"plots/Dicke_goe_sff_dist_j={j}_slope_ramp_vs_g_new")
#     plt.show()

if __name__ == '__main__':
    main()