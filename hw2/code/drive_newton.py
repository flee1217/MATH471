from scipy import *
from newton import newton
import matplotlib.pyplot as plt

# Define your functions to test Newton on.  You'll need to 
# define two other functions for f(x) = x and f(x) = sin(x) + cos(x**2)

def f1(x, d):
    if d == 0:
        return x**2
    elif d == 1:
        return 2*x

# f(x) = x
def f2(x, d):
    if d == 0:
        return x
    elif d == 1:
        return 1

# f(x) = sin(x) + cos(x**2)
def f3(x, d):
    if d == 0:
        return sin(x) + cos(x**2)
    elif d == 1:
        return cos(x) - 2*x*sin(x**2)


fcn_list = [f1,f2,f3]

# tunable parameters
x0 = .5
maxIterations = 30
tolerance = 10 ** (-10)

for i in range(len(fcn_list)):
    data = newton(fcn_list[i], x0, maxIterations, tolerance )
    print(data)
    print('number of iterations: %d' %len(data))

    # this compound plot holds two vertically arranged subplots
    f, plots = plt.subplots(2, sharex = True)

    # the first subplot graphs the function value as a function
    # of the current root approximation as the root-finder
    # iterates
    plots[0].plot(range(len(data)), data[:,4])
    # the second subplot graphs the alpha value which is the
    # result of the expression ( log (A) / log (B) ), where
    # A = | X_(n+1) - X_n     | / | X_n     - X_(n-1) |
    # B = | X_n     - X_(n-1) | / | X_(n-1) - X_(n-2) |
    plots[1].plot(range(len(data)),data[:,5])
    # showing the compound plot
    plt.show()
    
    f.savefig('newton_fig%d.png' %(i+1), dpi=None, format='png', bbox_inches='tight',pad_inches=0.1,)
    savetxt('newton_data%s.tex' %(i+1),data,fmt='%.5e',delimiter='  &  ',newline=' \\\\\n')
    # Here, you can post-process data, and test convergence rates 

    # Also, use your skills from the last lab, and use savetxt() to dump 
    # data and/or convergence rates to a text file


