from scipy import *
from matplotlib import pyplot as plt
import numpy as np
import sys
###
# Usage: python plot_flock_diam <parameter to vary from default>
#    ex: python plot_flock_diam gamma_2
# Valid
# Params: birds, alpha, gamma_1, gamma_2, kappa, rho, delta, food_flag
###

param_dict = { 'birds': 0.0, 'alpha': 0.2, 'gamma_1': 1.0, 'gamma_2': 4.0, \
               'kappa': 2.0, 'rho': 1.0, 'delta': 0.25, 'food_flag' : -0.0}

offset_dict = { 'birds'  : (11,13),
                'alpha'  : (14,17),
                'gamma_1': (18,21),
                'gamma_2': (22,25),
                'kappa'  : (26,29),
                'rho'    : (30,33),
                'delta'  : (34,35)}

default_filename = 'flock_diam_30_0.4_2.0_8.0_4.0_2.0_0.5_1.txt'

if len(sys.argv) != 2:
    sys.stderr.write('no valid param specified\n')
    sys.exit()

try:
    param_base = param_dict[sys.argv[1]]
except KeyError:
    sys.stderr.write('invalid param specified\n')
    sys.exit()

# iterate through increasing exponentiations of 2
# i.e. in param_base*[2**0 ... 2**param_iter_range]
# unless food_flag or birds is file param

data = []
param_list = []
param_iter_range = 5
if param_base == 0.0:
    param_list = [10, 30, 100]
elif param_base == -0.0:
    param_list = [0, 1]
else:
    param_list = [2.0**float(i) for i in range(param_iter_range)]
    print(param_list)
    param_list = param_base*np.asarray(param_list)
    print(param_list)

begin_slice, end_slice = offset_dict[sys.argv[1]]

for p in param_list:
    prefix = default_filename[:begin_slice]
    suffix = default_filename[end_slice:]
    data.append(loadtxt(prefix + str(p) + suffix))

ns = arange(len(data[0]))

f, fplot = plt.subplots()
for i, d in enumerate(data):
    fplot.plot(ns, d, label = str(sys.argv[1])+' = '+str(param_list[i]))

plt.legend()

plt.savefig('flock_diam_plots.png',dpi=600,bbox_inches = None, pad_inches = 0.1)
