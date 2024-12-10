import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    for g in g_arr:
        sff_open_list = sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist, ntraj)
        plt.plot(tlist,sff_open_list)
    plt.show()

if __name__ == "__main__":
    main()