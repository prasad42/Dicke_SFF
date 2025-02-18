import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    sff_goe_list = sff_goe_list_fun(N, β, tlist, v, ntraj)
    sff_poi_list = sff_poi_list_fun(N, β, tlist, v, ntraj)
    plt.figure(figsize=(10,5))
    plt.suptitle(f"j={j},M={M},α={α},v={v}")

    mse_list = []
    for g_ind, g in enumerate(g_arr):
        sff_list = sff_list_fun(ω, ω0, j, M, g, β, tlist, α, v)
        sff_rl = sff_rl_fun(tlist, sff_list)
        # Compute MSE
        mse = np.mean((sff_rl - sff_goe_list) ** 2)
        mse_list.append(mse)

    plt.plot(g_arr,mse_list)
    plt.show()

if __name__ == '__main__':
    main()