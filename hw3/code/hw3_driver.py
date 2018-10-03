from scipy import *
from lglnodes import lglnodes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

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

# t holds the min N to get quadrature error within the given tolerance 10^-10
t = np.zeros((2,2))

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


INTEGRATION_COLOR = (1.0,
                     0.27059,
                     0.0)

xs = np.linspace(-1.0, 1.0, 1000)
ys = fcn_list[0](xs)
t_xs = np.linspace(-1.0, 1.0, t[0][0])
t_ys = fcn_list[0](t_xs)

j, jplot = plt.subplots()

x_names = ['$x_{%d}$' %i for i in range(12)]

jplot.plot(xs, ys, color = 'b')
jplot.plot(t_xs, t_ys, color = INTEGRATION_COLOR)
jplot.scatter(t_xs, t_ys, color = INTEGRATION_COLOR)

plt.xticks(t_xs, x_names)

width = t_xs[1] - t_xs[0]

for k in range(int(t[0][0]) - 1):
    p = Polygon(xy= [(t_xs[k]  , 0         ),
                     (t_xs[k]  , t_ys[k]   ),
                     (t_xs[k+1], t_ys[k+1] ),
                     (t_xs[k+1], 0         )],
                alpha = 0.05,
                color = INTEGRATION_COLOR,
                )
    jplot.add_patch(p)

# formatting
jplot.set_xlim(-1.0,1.0)
jplot.set_title('Trapezoid Zones')
jplot.set_xlabel('$x$')
jplot.set_ylabel('$y$',
                 rotation=0)
plt.grid(True)

j.savefig('trapezoid_plot.png',
          dpi = None,
          format = 'png',
          bbox_inches = 'tight',
          pad_inches = 0.1,)

################################################################################
# Plotting the visualization of gauss quadrature on k = pi^2

k, kplot = plt.subplots()

f_xs = np.linspace(-1.1, 1.1, 1000)
f_ys = fcn_list[0](f_xs)

(g_xs, g_ws) = lglnodes(19)
g_ys = fcn_list[0](g_xs)

tick_xs = np.linspace(-1.0,1.0,5)

plt.xticks(tick_xs)

kplot.set_title('Gaussian Quadrature')

kplot.set_ylabel('$f(x) \n$',
                 rotation = 0,
                 verticalalignment = 'center',
                 rotation_mode = 'anchor',
                 color = 'b')
kplot.plot(f_xs, f_ys,
           color = 'b')
kplot.tick_params('y',
                  colors = 'b')
kplot.set_xlim(-1.1,1.1)

lplot = kplot.twinx()
lplot.set_ylabel('$f(x) w(x)$',
                 rotation = 0,
                 verticalalignment = 'center',
                 rotation_mode = 'anchor',
                 color = 'k')

for i in range(len(g_xs)):
    lplot.plot([g_xs[i], g_xs[i]],[0,g_ys[i]*g_ws[i]],
               color = 'k')
'''
lplot.bar(g_xs,g_ys*g_ws,
          width = 0.01,
          align = 'center',
          linewidth = 0,
          color = 'k')
'''
lplot.tick_params('y',
                  colors = 'k')
lplot.set_xlim(-1.1,1.1)

plt.grid(True,
         which = 'both',
         axis = 'both')

k.savefig('lgl_plot.png',
          dpi = None,
          format = 'png',
          bbox_inches = 'tight',
          pad_inches = 0.1,)

# end gaussian line plotting
################################################################################

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
fplot.set_title('Trapezoidal Rule: Error vs $n$',
                fontsize = 16)
fplot.set_xlim(1, partition_limit)
fplot.set_ylim( 10 ** -16, 5)
fplot.set_xlabel('$n$',
                 fontsize = 16)
fplot.set_ylabel('$(\delta_{abs})_{n+1}$\n',
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
gplot.loglog(ns, [2 ** ((1)(4-n)) for n in ns],
             ns, [2 ** ((2)(4-n)) for n in ns],
             ns, [2 ** ((3)(4-n)) for n in ns])
  
# formatting
gplot.set_title('Gaussian Quadrature: Error vs $n$',
                fontsize = 16)
gplot.set_ylim( 10 ** -16, 5)
gplot.set_xlabel('$n$',
                 fontsize = 16)
gplot.set_ylabel('$(\delta_{abs})_{n+1}$\n',
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

s1 = [str(int(i)) for i in t[0]]
s2 = [str(int(i)) for i in t[1]]
s = [['Trapezoid'] + s1,
     ['Gaussian'] + s2]

print s
savetxt('table1.tex',s,fmt='%s', delimiter='  &  ',newline=' \\\\\n')

                                                          
# End
################################################################################
