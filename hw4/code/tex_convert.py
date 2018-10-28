import sys
import numpy as np
s = str(sys.argv[1])
e = np.loadtxt(s)
np.savetxt(s + '.tex',e,fmt='%.6e', delimiter='  &  ',newline=' \\\\\n')
