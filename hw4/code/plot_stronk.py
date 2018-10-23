import numpy as np
from matplotlib import pyplot as plt

################################
# Plotting timings
################################

# timing data from CARC run
timings = np.loadtxt('timings.txt')
ns = np.arange(1,len(timings)+1)

# creating new plotting objects
f, fplot = plt.subplots()

# plot data
fplot.plot(ns, timings,
           linewidth = 2.,
           color = 'c')

# formatting
fplot.set_xlim(.5, 8.5)
fplot.set_ylim(6.8,15.5)
plt.xlabel('$Threads$',
           fontsize = 16)
plt.ylabel('$Time$        \n$(s)$        ',
           rotation = 0,
           rotation_mode = 'anchor',
           fontsize = 16)
plt.title('Compute Time vs Thread Count',
          fontsize = 16)

plt.savefig('strong.png',dpi = 600,
            bbox_inches = 'tight',
            pad_inches = .1)

################################
# Plotting efficiency
################################

# calculating efficiency values
efficiencies = timings[0]/(ns*timings)

g, gplot = plt.subplots()

gplot.plot(ns, efficiencies,
           linewidth = 2.,
           color = 'c')

gplot.set_xlim(.5, 8.5)
gplot.set_ylim(.2, 1.1)
plt.xlabel('$Threads$',
           fontsize = 16)
plt.ylabel('$Efficiency$        ',
           rotation = 0,
           rotation_mode = 'anchor',
           fontsize = 16)
plt.title('Efficiency vs Thread Count',
          fontsize = 16)

plt.savefig('efficient.png', dpi = 600,
            bbox_inches = 'tight',
            pad_inches = .1)
