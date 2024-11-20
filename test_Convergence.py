import matplotlib.pyplot as plt
import numpy as np
from parameters import *

M1 = 80
M2 = 400
g = 1.0

eval_list1 = np.load(f"evals_par/evals_g={np.round(g,2)}_j={j}_M={M1}.npy")
eval_list2 = np.load(f"evals_par/evals_g={np.round(g,2)}_j={j}_M={M2}.npy")

# eval_list_cut = []

# for i in range(len(eval_list2)):
#     E = eval_list2[i]
#     if E/(2*j)>0.4 and E/(2*j)<4:
#         eval_list_cut.append(E)
#     else:
#         eval_list_cut.append(0)

plt.scatter(eval_list1,eval_list2[0:len(eval_list1)])
plt.show()