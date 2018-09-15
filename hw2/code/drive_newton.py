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

# fcn plot parameters
plot_scale_factor = 0.1
plot_res     = 1000

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
    # saving the compound plot
    f.savefig('newton_fig%d.png' %(i+1), dpi=None, format='png', bbox_inches='tight',pad_inches=0.1,)
    # dumping root finding data into .tex file
    savetxt('newton_data%s.tex' %(i+1),data,fmt='%.5e',delimiter='  &  ',newline=' \\\\\n')

    # now, we plot the function

    xs = data[:,0].tolist()
    xs.insert(0, x0)
    ys = data[:,4].tolist()
    ys.insert(0, fcn_list[i](x0,0))
    
    # we would like to plot only parts of our function that are necessary
    # for visualizing the iterated root approximations, so we set the xlimits
    # and ylimits for our plots to 
    y_max = max(ys)
    y_min = min(ys)
    x_max = max(xs)
    x_min = min(xs)
    
    # differences between smallest and largest values in ys and xs
    y_d = y_max - y_min
    x_d = x_max - x_min
    
    # more book-keeping: max_diff holds the largest difference between x_d and y_d
    max_diff = max(y_d, x_d)
    
    # x_center and y_center are the linear midpoints between extremum values in x & y, respectively
    x_center = x_min + x_d/2
    y_center = y_min + y_d/2

    x_ = linspace(x_min-max_diff*plot_scale_factor, x_max+max_diff*plot_scale_factor,plot_res + 1)
    y_ = fcn_list[i](x_,0)
    g, gplot = plt.subplots()
    gplot.plot(x_,y_)
    
    # plotting the x and f(x) iterates
    gplot.scatter(x0,fcn_list[i](x0,0),c='red')
    gplot.scatter(data[:,0],data[:,4],c='red')

    gplot.set_xlim(x_center - (max_diff / 2) * (1.1), x_center + (max_diff / 2) * (1.1))
    gplot.set_ylim(y_center - (max_diff / 2) * (1.1), y_center + (max_diff / 2) * (1.1))

    gplot.grid(True, linestyle='--')
    g.savefig('newton_plot%d.png' %(i+1), dpi=None, format='png', bbox_inches='tight',pad_inches=0.1,)
    plt.show()

