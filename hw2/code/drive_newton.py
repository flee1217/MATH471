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
x0 = 0.5
maxIterations = 30
tolerance = 10 ** (-10)

# fcn plot parameters
plot_lim_scale_factor = 1.1
plot_res     = 1000

init_root_label_x = [ 0.65, 0.65, 0.39 ]
init_root_label_y = [ 0.8, 0.85, 0.6  ]
final_root_label_x = [ 0.4, 0.5, 0.35 ]
final_root_label_y = [ 0.6, 0.4, 0.44 ]
plot_titles = ['$f(x) = x^2$', '$f(x) = x$', '$f(x) = \sin(x) + \cos(x^2)$']

for i in range(len(fcn_list)):
    # for this lab, we want to use a initial guess of -0.5 for the third
    # function, mostly to showcase the fast convergence using newton's
    # method
    if (i == 2):
        x0 = -x0
    data = newton(fcn_list[i], x0, maxIterations, tolerance )
    print(data)
    print('number of iterations: %d' %len(data))
    
    '''
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
    '''
    # dumping root finding data into .tex file
    savetxt('newton_data%s.tex' %(i+1),data,fmt='%.5e',delimiter='  &  ',newline=' \\\\\n')
    
    
    # now, we plot the function on a linear x-y graph
    xs = data[:,0].tolist()
    xs.insert(0, x0)
    ys = data[:,4].tolist()
    ys.insert(0, fcn_list[i](x0,0))
    
    # we would like to plot only parts of our function that are necessary
    # for visualizing the iterated root approximations, so we set the xlimits
    # and ylimits for our plots to values close to the extremum x or y values
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

    x_lower_bound = x_min - max_diff * (plot_lim_scale_factor - 1)
    x_upper_bound = x_max + max_diff * (plot_lim_scale_factor - 1)
    
    g, gplot = plt.subplots()
    




    
    # plotting the x and f(x) iterates
    gplot.plot(xs[:], ys[:], 'bo--', c='red')
    
    # setting the dynamically determined plot extrema
    x_lower_lim = x_center - (max_diff / 2) * (plot_lim_scale_factor)
    x_upper_lim = x_center + (max_diff / 2) * (plot_lim_scale_factor)
    y_lower_lim = y_center - (max_diff / 2) * (plot_lim_scale_factor)
    y_upper_lim = y_center + (max_diff / 2) * (plot_lim_scale_factor)

    lim_max = max(abs(x_lower_lim),abs(x_upper_lim),abs(y_lower_lim),abs(y_upper_lim))
    # gplot.set_xlim(x_lower_lim, x_upper_lim)
    # gplot.set_ylim(y_lower_lim, y_upper_lim)
    
    gplot.set_xlim(-lim_max, lim_max)
    gplot.set_ylim(-lim_max, lim_max)

    # we would like to superimpose the root approximations over a graph of the function in question
    # x_ = linspace(x_lower_bound, x_upper_bound,plot_res + 1)
    x_ = linspace(-lim_max, lim_max, plot_res + 1)
    
    y_ = fcn_list[i](x_,0)
    gplot.plot(x_,y_,c='black')

    gplot.axvline(linewidth = 1, color = 'k')
    gplot.axhline(linewidth = 1, color = 'k')
    
    gplot.set_xlabel('$x$')
    gplot.set_ylabel('$f(x)$')
    gplot.set_title('%s' %plot_titles[i])
    gplot.grid(True, linestyle='--')
    gplot.set_aspect(aspect='equal')
    
    # these two functions create and place the labels for the initial and final root approximations
    gplot.annotate('$(x_0,f(x_0))$', xy=(xs[0],ys[0]),xytext=(init_root_label_x[i],init_root_label_y[i]),
                   textcoords='figure fraction',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),size=14)

    gplot.annotate('$(x_n,f(x_n))$', xy=(xs[-1],ys[-1]),xytext=(final_root_label_x[i],final_root_label_y[i]),
                   textcoords='figure fraction',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),size=14)

    g.savefig('newton_plot%d.png' %(i+1), dpi=None, format='png', bbox_inches='tight',pad_inches=0.1,)
    plt.show()

