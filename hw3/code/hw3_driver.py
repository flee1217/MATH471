from scipy import *
# from lglnodes import lglnodes
import numpy as np
import matplotlib.pyplot as plt

def int_trap(f, a, b, n):
    h = float(b - a) / n
    xs = np.linspace(a, b, n + 1)
    sum = 0.0
    for i in range( 1, len(xs) - 1):
        sum += f(xs[i])

    return (2.0 * sum + f(xs[0]) + f(xs[-1])) * (h / 2.0)

def f1(k):
    def f1_actual(x):
        return np.exp( np.cos( k * x ) )
    return f1_actual

# change k to np.pi to see minimal number of partitions N
# needed to get within some tolerance tol
# d is absolute difference between trapezoidal integration
# given N+1 and N partitions
#   d = | I_T(n+1) - I_T(n) |
# as given in HW3, problem 1, (the little kronecker delta)

k = np.pi * np.pi
func = f1(k)
tol = 10 ** (-10)
'''
for N in range(1,3000):
    I_n1 = int_trap(func, -1, 1, N+1)
    I_n = int_trap(func, -1, 1, N)
    d = abs(I_n1-I_n)
    print ('N:%d I_n1:%.20f I_n:%.20f d:%.20f' %(N, I_n1, I_n, d))
    if (d <= tol):
        print(N)
        break
'''

# now we want to plot the error
#
# error here is defined as presented in the assignment specification
# for problem 1
# error_N+1 = | I_T_(N+1) - I_T_(N) |

fcn_list = [f1(np.pi), f1(np.pi ** 2)]

partition_limit = 1000

errors = [np.zeros(partition_limit), np.zeros(partition_limit)]

for i in range(len(fcn_list)):
    for N in range(1,partition_limit):
        a = int_trap(fcn_list[i], -1, 1, N+1)
        b = int_trap(fcn_list[i], -1, 1, N)
        errors[i][N-1] = abs(a - b)
    
    f, fplot = plt.subplots()
    ns = range(1,partition_limit + 1)

    fplot.loglog(ns, errors[i],
                 ns, [1.0/n for n in ns],
                 ns, [1.0/(n**2) for n in ns],
                 ns, [1.0/(n**3) for n in ns],
                 ns, [1.0/(n**4) for n in ns])
    plt.show()
