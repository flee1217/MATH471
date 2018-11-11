from scipy import *
from matplotlib import pyplot as plt
import sys
###
# Usage: python plot_flock_diam <list of flock sizes to process>
#    ex: python plot_flock_diam 10 30 100
###

data = []
for i in range(len(sys.argv)-1):
    data.append(loadtxt('flock_diam'+str(sys.argv[i+1])+'.txt'))

ns = arange(len(data[0]))

f, fplot = plt.subplots()
for i, d in enumerate(data,1):
    fplot.plot(ns, d, label = 'N = '+str(sys.argv[i]))

plt.legend()

plt.savefig('flock_diam_plots.png',dpi=600,bbox_inches = None, pad_inches = 0.1)
