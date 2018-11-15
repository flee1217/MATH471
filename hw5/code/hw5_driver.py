from scipy import *
from scipy import stats
from matplotlib import pyplot
from operator import itemgetter
import numpy as np
import matplotlib.animation as manimation
import os, sys, random

# HW5 solution by Aaron and Fred

# Usage:
# Expected command line param format
# python hw5_driver.py <movie_option>    : [0, 1]
#                      <run_option>      : [0, 1, 2, 3]
#                      <number of birds> : [10, 30, 100]
#                      <alpha>           : [0.2, 0.4, 0.8, 1.6, 3.2]
#                      <gamma_1>         : [0.5, 1, 2, 4, 8]
#                      <gamma_2>         : [4, 8, 16, 32, 64]
#                      <kappa>           : [2, 4, 8, 16, 32]
#                      <rho>             : [1, 2, 4, 8, 16]
#                      <delta>           : [0.25, 0.5, 1.0, 2.0, 4.0]
#                      <food_flag>       : [0, 1]
#                      <sigma>           : [1.0, 2.0, 4.0, 8.0, 16.0]
#                      <pi_>             : [1.0, 2.0, 4.0, 8.0, 16.0]
#
# default params: 0 0 30 0.2 2.0 8.0 4.0 2.0 0.5 1 8.0 4.0
#
# Proper usage is assumed (i.e. there are no safeguards in place for running
# non-listed param sets, this is more to facilitate compatibility with the
# flock diameter plotting file)
#
# Parameters are processed in order, meaning the second argv entry (sys.argv[2])
# is expected to be number of birds
#
# Additional params (after food_flag) are ignored



def RK4(f, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta,\
        sigma, pi_, run_option):
    '''
    Carry out a step of RK4 from time t using dt
    
    Input
    -----
    f:  Right hand side value function
    y:  state vector
    t:  current time
    dt: time step size
    
    food_flag:  0 or 1, depending on the food location
    alpha:      Parameter from homework PDF
    gamma_1:    Parameter from homework PDF
    gamma_2:    Parameter from homework PDF
    kappa:      Parameter from homework PDF
    rho:        Parameter from homework PDF
    delta:      Parameter from homework PDF
    sigma:      Smelly bird repulsive force scalar factor
    pi_:        Predator avoidance force factor
    run_option: flag dictating which forces to apply for extra credit
    

    Output
    ------
    Return updated vector y, according to RK4 formula
    '''
    
    # Task: Fill in the RK4 formula
    k1 = dt*f(y,t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta,\
              sigma, pi_, run_option)
    k2 = dt*f(y+k1/2.,t+dt/2., food_flag, alpha, gamma_1, gamma_2, kappa,\
              rho, delta, sigma, pi_, run_option)
    k3 = dt*f(y+k2/2.,t+dt/2., food_flag, alpha, gamma_1, gamma_2, kappa,\
              rho, delta, sigma, pi_, run_option)
    k4 = dt*f(y+k3,t+dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho,\
              delta, sigma, pi_, run_option)

    y = y + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)

    return y


def C(t, food_flag):
    if food_flag == 1:
        return (sin(alpha*t),cos(alpha*t))
    else:
        return (0.0, 0.0)


def P(t):
    rho = 2.0*np.sqrt(1j*cos(2.0*t))
    r = np.sqrt(rho.real**2 + rho.imag**2)
    x,y = (r*np.cos(t),r*np.sin(t))
    return (x,y)

def nearest_neighbors(b, c, k):
    '''
    Give locations of the nearest k neighbors of bird b
    in the form of a list of k rows and 2 columns (x and y locations)
    from the community c of birds

    Assumptions
    -----------
    - 2D euclidean space (x and y locations for each bird)
    - 1 <= k < c
    - b is contained within c
    - len(c) > 1

    Input
    -----
    b: target bird
    k: number of nearest neighbors desired
    c: community of birds to choose from

    Output
    ------
    n: list of nearest k neighbor locations
    '''

    n = np.zeros((k,2))

    # first column of d holds squared dist between
    # target bird b and bird i in community c
    # second column holds index i for each bird in c
    d = np.zeros_like(c)
    for i in range(len(c)):
        d[i][0] = (c[i][0]-b[0])**2 + (c[i][1]-b[1])**2
        d[i][1] = i

    # sort this "tuple" (not really a tuple) list by
    # squared distance (in ascending order)
    d = array(sorted(d,key=itemgetter(0)))

    # the first entry in d is b so it will be
    # the first item in the sorted dist,index list.
    # That is why the row index in d is (i+1)
    for i in range(k):
        index = int(d[i+1][1])
        n[i][0] = c[index][0]
        n[i][1] = c[index][1]

    return n


def flock_diameter_linregress(c):
    '''
    Return the linregress version of flock diameter
    evaluation, where diameter here is defined as the
    euclidean distance between two points (A and B) along
    its linear regression line

    Input
    -----
    c: the community (flock) of birds

    Output
    _____
    (distance, ndarray([A_x, A_y],
                       [B_x, B_y])

    - a tuple containing:
    -   - the distance calculation between
    -     the two points on the linear regression line
    -   - a NumPy array containing the x and y values
    -   - of each point
    '''
    # extremum points to return
    extremum_pts = np.zeros((2,2))
    
    c_x = c[:,0]
    c_y = c[:,1]

    # using a helpful scipy library function
    slope, intercept, r, p, std_err = stats.linregress(c_x,c_y)

    max_dx = max(c_x) - min(c_x)
    max_dy = max(c_y) - min(c_y)
    
    dist = np.sqrt(max_dx**2.0 + max_dy**2.0)

    x1 = 0.0
    x2 = 0.0

    if max_dx >= max_dy:
        x1 = min(c_x)
        x2 = max(c_x)
    else:
        x1 = [c_x[i] for i in range(len(c_x)) if c_y[i] == min(c_y)]
        x2 = [c_x[i] for i in range(len(c_x)) if c_y[i] == max(c_y)]

    # linear regression line:
    # f(x) = slope * x_value + intercept
    y1 = slope*x1 + intercept
    y2 = slope*x2 + intercept

    extremum_pts[0] = [x1, y1]
    extremum_pts[1] = [x2, y2]

    return (dist, extremum_pts)


def flock_diameter(c):
    '''
    Return the diameter of a flock (community) c of birds
    where flock diameter is the distance between the
    center of the flock and the bird furthest from the
    center (multiplied by 2) (using euclidean distance)

    Assumptions
    -----------
    - 2D locations
    - c is non-zero (there are more then 0 zeros in c)

    Input
    -----
    c: the flock (community) of birds

    Output
    ------
    d: the diameter of the flock c of birds
    '''
    center = flock_center(c)
    d = np.zeros(len(c))
    radius = 0.0
    
    for i in range(len(c)):
        current_radius = (c[i][0]-center[0][0])**2 + (c[i][1]-center[0][1])**2
        current_radius = max(radius, current_radius)

    return 2.*np.sqrt(current_radius)


def flock_center(c):
    '''
    Give the x,y location of the center of the flock
    (community) c of birds

    Assumptions
    -----------
    - 2D locations (x and y)
    - c is an size N x 2 matrix of birds where the x
       locations of the birds are held in column 1
       and the y locations are held in column 2
    
    Input
    -----
    c: the flock (community) of birds

    Output
    ------
    (ndarray)
    - center: the center location of the flockk
    '''

    center = np.zeros((1,2))
    # N: the number of birds
    N = y.shape[0]
    for i in range(N):
        center[0][0] += y[i][0]
        center[0][1] += y[i][1]

    center /= N
    return center    


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta,\
        sigma, pi_, run_option):
    '''
    Define the right hand side of the ODE

    input
    -----
    y: current flock state
    t: current time
    
    output
    ------
    f: y'

    '''
    N = y.shape[0]
    f = np.zeros_like(y)

    # leader bird calculation
    f[0] = gamma_1*(C(t,food_flag) - y[0])

    # flock calculations
    
    # follow force
    f[1:] = gamma_2*(y[0] - y[1:])

    # centering force
    f[1:] += kappa*(flock_center(y) - y[1:])

    # repelling force
    for k in range(1,N):
        nn = nearest_neighbors(y[k],y,neighbors)
        repelling = 0.0
        for n in nn:
            repelling += ((y[k] - n)/((y[k] - n)**2 + delta))
        f[k] += repelling*rho


    # smelly bird repulsive force
    if run_option == 1 or run_option == 3:
        f[2:] += sigma*(y[2:] - y[1])/((y[2:] - y[1])**2 + delta)
    
    # predator avoidance force
    if run_option == 2 or run_option == 3:
        f[:] += pi_*(y[:] - P(t))/((y[:] - P(t))**2 + delta)
    return f


##
# Set up problem domain
t0     = 0.0   # start time
T      = 10.0  # end time
nsteps = 50    # number of time steps

np.random.seed(2018)

# Task:  Experiment with the problem parameters, and understand what they do
# Task:  Experiment with N, number of birds
birds = 30

dt = (T - t0) / (nsteps-1.0)
alpha = 0.4
gamma_1 = 2.0
gamma_2 = 8.0
kappa = 4.0
rho = 2.0
delta = 0.5
food_flag = 1
# food_flag == 0: C(x,y) = (0.0, 0.0)
# food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))
neighbors = 5
# number of nearest neighbors to consider for repulsive force
# calculations

movie_option = 0
# movie option == 0: no movie
#              == 1: here's a movie

run_option = 0
# run_option == 0: regular sim
#            == 1: smelly bird
#            == 2: predator
#            == 3: smelly + predator

# EXTRA CREDIT
sigma = 8.0 # smelly bird repulsion force scaling term
pi_   = 4.0 # predator repulsion force scaling term

# Apply Command Line Params
# Assume the user isn't a jerk trying to break the parser
p = sys.argv

for i in range(1,len(p)):
    param = p[i]
    if i == 1:                    # movie_option
        movie_option = int(param)
    if i == 2:                    # run_option
        run_option = int(param)
    if i == 3:                    # birds
        birds = int(param)
    if i == 4:                    # alpha
        alpha = float(param)
    if i == 5:                    # gamma_1
        gamma_1 = float(param)
    if i == 6:                    # gamma_2
        gamma_2 = float(param)
    if i == 7:                    # kappa
        kappa = float(param)
    if i == 8:                    # rho
        rho = float(param)
    if i == 9:                    # delta
        delta = float(param)
    if i == 10:                   # food_flag
        food_flag = int(param)
    if i == 11:                   # sigma
        sigma = float(param)
    if i == 12:                   # pi
        pi_ = float(param)


# Intialize problem

y = np.random.rand(birds,2)
flock_diam = np.zeros((nsteps,))

# Initialize the Movie Writer


# Begin writing movie frames (if a render is wanted)
if (movie_option == 1):
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=12)
    fig = pyplot.figure(0)
    ax = pyplot.subplot(111)


    # flock plotting data
    # cc: flock w/o leader
    # ll: leader bird
    # ff: food
    # ss: smelly bird
    # pp: predator
    cc, = pyplot.plot([],[], 'k.', label = '$Flock$')
    ll, = pyplot.plot([],[], 'ro', label = '$Leader$')
    ff, = pyplot.plot([],[], 'y*', label = '$Food$')
    ss, = pyplot.plot([],[], 'bP', label = '$Smelly$')
    pp, = pyplot.plot([],[], 'mX', label = '$Predator$')
    pyplot.xlabel(r'$X$', fontsize='large')
    pyplot.ylabel(r'$Y$', fontsize='large')
    pyplot.xlim(-3,3)       # may need to adjust if birds fly outside box
    pyplot.ylim(-3,3)       # may need to adjust if birds fly outside box
    
    # legend formatting
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                 
    ax.legend(loc='center left', bbox_to_anchor = (1, 0.5))

    with writer.saving(fig, "movie.mp4",dpi=648):

    # First frame
        p_x,p_y = P(t0)
        c_x,c_y = C(t0,food_flag)
        cc.set_data(y[2:,0], y[2:,1]) 
        ll.set_data(y[0,0], y[0,1])
        ff.set_data(c_x,c_y)
        if (run_option == 1 or run_option == 3):
            ss.set_data(y[1,0], y[1,1])
        if (run_option == 2 or run_option == 3):
            pp.set_data(p_x,p_y)
        writer.grab_frame()

        t = t0
        i = 1
        for step in range(nsteps):
            if i % 10 == 0:
                print('frame: '+str(i)+'/'+str(nsteps))

            i += 1
        
            # Task: Fill these two lines in
            flock_diam[step] = flock_diameter(y)
            y = RK4(RHS, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho,\
                        delta, sigma, pi_, run_option)
            t += dt
            p_x,p_y = P(t)
            c_x,c_y = C(t,food_flag)
        
            # Movie frame
            cc.set_data(y[2:,0], y[2:,1]) 
            ll.set_data(y[0,0], y[0,1])
            ff.set_data(c_x,c_y)
            if (run_option == 1 or run_option == 3):
                ss.set_data(y[1,0], y[1,1])
            if (run_option == 2 or run_option == 3):
                pp.set_data(p_x,p_y)
            writer.grab_frame()
# no movie? only time stepping happens
else:
    t = t0
    i = 1
    for step in range(nsteps):
        if i % 10 == 0:
            print('frame: '+str(i)+'/'+str(nsteps))
        i += 1
        
        flock_diam[step] = flock_diameter(y)
        y = RK4(RHS, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho,\
                    delta, sigma, pi_, run_option)
        t += dt
        
        
       
savetxt('flock_diam_'+str(birds)+'_'+str(alpha)+'_'+str(gamma_1)+'_'+\
        str(gamma_2)+'_'+str(kappa)+'_'+str(rho)+'_'+str(delta)+'_'+\
        str(food_flag)+'_'+str(sigma)+'_'+str(pi_)+'.txt',flock_diam)
