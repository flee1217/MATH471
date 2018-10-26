import numpy as np
from matplotlib import pyplot as plt

error = np.loadtxt('error.txt')
ns = np.arange(1,6)

f, fplot = plt.subplots()

for i,e in enumerate(error):
    fplot.loglog(ns, e, label= '$Threads:$ $' + str(i+1) + '$', linewidth = 2)

# reference plots
fplot.loglog(ns, ns**(-1.)/10.**8,
             label = '$O(n^{-1})$')
fplot.loglog(ns, ns**(-2.)/10.**8,
             label = '$O(n^{-2})$')

# nice formatting
plt.title('$Error$ $vs$ $N$',
          fontsize = 16)
fplot.set_xlabel('$N$',
                 fontsize = 16)
fplot.set_ylabel('$Error$',
                 fontsize = 16,
                 rotation = 0,
                 rotation_mode = 'anchor')
fplot.legend()

plt.savefig('error_thread.png',dpi=600,bbox_inches = None, pad_inches = 0.1)
