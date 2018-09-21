from scipy import *
# from lglnodes import lglnodes
import numpy as np

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
for N in range(1,3000):
    a = int_trap(func, -1, 1, N+1)
    b = int_trap(func, -1, 1, N)
    d = abs(a-b)
    print ('N:%d a:%.20f b:%.20f d:%.20f %(N, a, b, d)')
    if (d <= tol):
        print(N)
        break
