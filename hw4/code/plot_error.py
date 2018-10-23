import numpy as np
from matplotlib import pyplot as plt

error = np.loadtxt('error.txt')
ns = np.arange(1,6)

f, fplot = plt.subplots()

for i,e in enumerate(error):
    fplot.loglog(ns, e, label= 'threads: ' + str(i+1) + '', linewidth = 2)

# reference plots
#fplot.loglog(ns, ns**(-1.)/10.**8.)
#fplot.loglog(ns, ns**(-2.)/10.**8.)

# nice formatting
fplot.legend()

plt.show()
