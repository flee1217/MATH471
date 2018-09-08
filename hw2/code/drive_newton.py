from scipy import *
from newton import newton

# Define your functions to test Newton on.  You'll need to 
# define two other functions for f(x) = x and f(x) = sin(x) + cos(x**2)

def f1(x, d):
    if d == 0:
        return x**2
    elif d == 1:
        return 2*x

def f2(x, d):
    if d == 0:
        return x
    elif d == 1:
        return 1

def f3(x, d):
    if d == 0:
        return sin(x) + cos(x**2)
    elif d == 1:
        return cos(x) - 2*x*sin(x**2)

fcn_list = [f1,f2,f3]

x0 = -0.5
for fcn in fcn_list:
    data = newton(fcn, x0)

    # Here, you can post-process data, and test convergence rates 

    # Also, use your skills from the last lab, and use savetxt() to dump 
    # data and/or convergence rates to a text file


