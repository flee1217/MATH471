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

for fcn in fcn_list:
    data = newton(fcn, x0, maxIterations, tolerance )
    print(data)
    print('number of iterations: %d' %len(data))
    f, plots = plt.subplots(2, sharex = True)
    plots[0].plot(range(len(data)), data[:,4])
    plots[1].plot(range(len(data)),data[:,5])

    
    plt.show()
    # Here, you can post-process data, and test convergence rates 

    # Also, use your skills from the last lab, and use savetxt() to dump 
    # data and/or convergence rates to a text file


