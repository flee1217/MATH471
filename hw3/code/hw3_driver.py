
from scipy import *
from lglnodes import lglnodes
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

################################################################################
# Finding N that gives successive iteration error under a given tolerance
#
# d is absolute difference between trapezoidal integration
# given N+1 and N partitions
#   d = | I_T(n+1) - I_T(n) |
# as given in HW3, problem 1, (the little kronecker delta)

k = np.pi * np.pi
tol = 10 ** (-10)
t=zeros((2,2))

fcn_list = [f1(np.pi), f1(np.pi ** 2)]
fcn_names = ['pi','pi^2']

initial_N = 1
print('Iterating through N to find when d = abs(I_(N+1) - I_(N)) < 10^-10')
for i in range(len(fcn_list)):
    print('For k = %s...' %fcn_names[i])
    if (i == 1):
        initial_N = 2800
    for N in range(initial_N,3000):
        I_n1 = int_trap(fcn_list[i], -1, 1, N+1)
        I_n = int_trap(fcn_list[i], -1, 1, N)
        d = abs( I_n1 - I_n )
        if (d <= tol):
            t[0][i]=N+1
            print ('N:%d d:%.15f' %(N, d))
            break

################################################################################
# Plotting Error for Trapezoid Quadrature
#
# error is defined as presented in the assignment specification
# for problem 1
# error_N+1 = | I_T_(N+1) - I_T_(N) |



partition_limit = 1000

errors = [np.zeros(partition_limit), np.zeros(partition_limit)]

print('generating trapezoid integration error plot...')
# computing the successive iteration errors and storing data
# in errors[]
for i in range(len(fcn_list)):
    for N in range(1,partition_limit):
        a = int_trap(fcn_list[i], -1, 1, N+1)
        b = int_trap(fcn_list[i], -1, 1, N)
        errors[i][N-1] = abs(a - b)


# creating a figure and axes object (from a factory?)
# to plot our error data using trapezoid quadrature
f, fplot = plt.subplots()
ns = range(1,partition_limit + 1)

# plotting data along with guiding convergence lines for
# ease of interpretation
for i in range(len(fcn_list)):
    fplot.loglog(ns, errors[i], linewidth = 2)

# adding guiding plots using list comprehensions
fplot.loglog(ns, [1.0/n for n in ns],
             ns, [1.0/(n**2) for n in ns],
             ns, [1.0/(n**3) for n in ns],
             ns, [1.0/(n**4) for n in ns])

# formatting
plt.grid(True)
fplot.set_title('Trapezoid Rule Error vs $n$',
                fontsize = 16)
fplot.set_ylim( 10 ** -16, 5)
fplot.set_xlabel('$n$',
                 fontsize = 16)
fplot.set_ylabel('$(\delta_{abs})_{n+1}$',
                 fontsize = 16,
                 rotation = 0,
                 verticalalignment = 'center',
                 rotation_mode = 'anchor')

# adding a legend
fplot.legend(['$k= \pi$','$k = \pi^{2}$','$1/n$','$1/n^{2}$','$1/n^{3}$','$1/n^{4}$'],
             loc = (0.05,0.05))
# saving plot
f.savefig('int_trap_error_plot.png',
          dpi = None,
          format = 'png',
          bbox_inches = 'tight',
          pad_inches = 0.1,)
print('Done!')

################################################################################
# Plotting Error using Gauss Quadrature as a function of N, the argument to
# lglnodes(n) found in lglnodes.py, which returns a tuple of function inputs and
# associated weights
#
#
n_upper = 100
LGL_ints = [np.zeros(n_upper), np.zeros(n_upper)]
LGL_errors = [np.zeros(n_upper), np.zeros(n_upper)]

print('generating gauss quadrature error plot...')
# computing integration error over successive integrations
# using lglnodes (gauss quadrature)
for i in range(len(fcn_list)):
    for N in range(1,n_upper + 1):
        (nodes_,weights_) = lglnodes(N+1)
        (nodes,weights) = lglnodes(N)
        f_ = fcn_list[i](nodes_)
        f = fcn_list[i](nodes)
        approx_integral_ = sum(f_ * weights_ )
        approx_integral = sum(f * weights)
        LGL_ints[i][N-1] = approx_integral
        LGL_errors[i][N-1] = abs(approx_integral_ - approx_integral)

# getting a figure and axes to work with in generating
# the gauss quadrature error
g, gplot = plt.subplots()
ns = range(1, n_upper + 1)

# plotting data with guiding exponential convergence lines
# for ease of interpretation
for i in range(len(fcn_list)):
    gplot.loglog(ns, LGL_errors[i], linewidth = 2)
gplot.loglog(ns, [np.e ** (-.5*n) for n in ns],
             ns, [np.e ** (-1*n) for n in ns],
             ns, [np.e ** (-1.5*n) for n in ns])
  
# formatting
gplot.set_title('Gaussian Quadrature Error vs $n$',
                fontsize = 16)
gplot.set_ylim( 10 ** -16, 5)
gplot.set_xlabel('$n$',
                 fontsize = 16)
gplot.set_ylabel('$(\delta_{abs})_{n+1}$',
                 fontsize = 16,
                 rotation = 0,
                 verticalalignment = 'center',
                 rotation_mode = 'anchor')
plt.grid(True)

# more formatting
gplot.legend(['$k = \pi$','$k = \pi^{2}$','$e^{-n/2}$','$e^{-n}$','$e^{-3n/2}$'],
             loc = (.05,0.05))

# saving plot
g.savefig('lgl_error_plot.png',
          dpi = None,
          format = 'png',
          bbox_inches = 'tight',
          pad_inches = 0.1,)
print('Done!')

################################################################################   
# Table for Task 3

for i in range(len(fcn_list)):
    for x in range(len(LGL_errors[0])):
        if LGL_errors[i][x]<=10**-10:
            print x+2
            t[1][i]=x+2
            print LGL_errors[i][x]
            break 
print t
savetxt('table1.tex',t,fmt=['%d','%d'], delimiter='  &  ',newline=' \\\\\n')

                                                                       
# End
################################################################################
