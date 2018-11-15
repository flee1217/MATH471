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

param_dict = { 'birds': 0.0, 'alpha': 0.1, 'gamma_1': 0.5, 'gamma_2': 0.5, \
               'kappa': 0.25, 'rho': 0.5, 'delta': 0.125, 'food_flag' : -1.0}

offset_dict = { 'birds'    : (11,13),
                'alpha'    : (14,17),
                'gamma_1'  : (18,21),
                'gamma_2'  : (22,25),
                'kappa'    : (26,29),
                'rho'      : (30,33),
                'delta'    : (34,37),
                'food_flag': (38,39)}

symbol_dict = { 'birds'     : '$Birds$',
                'alpha'     : '$\\alpha$',
                'gamma_1'   : '$\\gamma_{1}$',
                'gamma_2'   : '$\\gamma_{2}$',
                'kappa'     : '$\\kappa$',
                'rho'       : '$\\rho$',
                'delta'     : '$\\delta$',
                'food_flag' : '$Food$ $Flag$'}

default_filename = 'flock_diam_30_0.4_2.0_8.0_4.0_2.0_0.5_1.txt'

if len(sys.argv) != 2:
    sys.stderr.write('no valid param specified\n')
    sys.exit()

pp = sys.argv[1]
param_base = 0.0
try:
    param_base = param_dict[pp]
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
elif param_base == -1.0:
    param_list = [0, 1]
else:
    param_list = [2.0**float(i) for i in range(param_iter_range)]
    param_list = param_base*np.asarray(param_list)

begin_slice, end_slice = offset_dict[pp]

# finding the desired files (which hopefully exist)
for p in param_list:
    prefix = default_filename[:begin_slice]
    suffix = default_filename[end_slice:]
    print(prefix)
    print(suffix)
    print(prefix+str(p)+suffix)
    data.append(loadtxt(prefix + str(p) + suffix))

begin_time = 0.0
end_time = 10.0
dt = end_time - begin_time
ns = arange(len(data[0]))

# create time vector values
N = [dt * float(n)/len(ns) for n in ns]

f, fplot = plt.subplots()
for i, d in enumerate(data):
    if pp == 'gamma_2':
        fplot.plot(N, d, label = symbol_dict[pp]+' = '+str(param_list[i]),
                   linewidth = 2.0)
    else:
        fplot.plot(N, d, label = symbol_dict[pp]+' = '+str(param_list[i]),
                   linewidth = 2.0)

# plot formatting
plt.title('Flock Diameter vs. Time')
plt.xlabel('$t$',
           fontsize = 16)
plt.ylabel('$D$',
           rotation = 0,
           rotation_mode = 'anchor',
           fontsize = 16)
plt.legend(loc=2)

plt.savefig('flock_diams_'+str(pp)+'.png',dpi=600,bbox_inches = None, pad_inches = 0.1)
