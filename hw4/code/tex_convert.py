import numpy as np
e = np.loadtxt('error.txt')
np.savetxt('error.tex',e,fmt='%.6e', delimiter='  &  ',newline=' \\\\\n')
