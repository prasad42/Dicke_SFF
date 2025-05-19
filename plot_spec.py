import matplotlib.pyplot as plt
from parameters import *
from dicke_sff_lib import *

eigvals_arr = []
for g_ind, g in enumerate(g_arr):
    eigvals = dicke_eigvals_fun(ω, ω0, j, M, g, α, dM, tol)
    cum_spec_fun = np.cumsum(eigvals)
    eigvals1 = unf_eigval_poly_fun(deg, cum_spec_fun)
    plt.plot(cum_spec_fun, label="eigvals cumulative sum")
    plt.plot(eigvals1, label="unfolded eigvals")
    # eigvals_arr.append(eigvals[1:100]-eigvals[0])

print(np.shape(eigvals_arr))

# plt.ylim(0,2)
plt.grid()
# plt.plot(g_arr, eigvals_arr)
plt.legend()
plt.show()