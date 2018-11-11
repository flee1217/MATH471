from scipy import *
from matplotlib import pyplot
from operator import itemgetter
import matplotlib.animation as manimation
import os, sys

def RK4(f, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
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
    

    Output
    ------
    Return updated vector y, according to RK4 formula
    '''
    
    # Task: Fill in the RK4 formula
    k1 = dt*f(y,t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k2 = dt*f(y+k1/2.,t+dt/2., food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k3 = dt*f(y+k2/2.,t+dt/2., food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k4 = dt*f(y+k3,t+dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)

    y = y + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)

    return y


def C(t, food_flag):
    if food_flag == 1:
        return (sin(alpha*t),cos(alpha*t))
    else:
        return (0.0, 0.0)


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

    n = zeros((k,2))

    # first column of d holds squared dist between
    # target bird b and bird i in community c
    # second column holds index i for each bird in c
    d = zeros_like(c)
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


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
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
    f = zeros_like(y)

    # leader bird calculation
    f[0] = gamma_1*(C(t,food_flag) - y[0])

    # flock calculations
    
    # follow force
    f[1:] = gamma_2*(y[0] - y[1:])

    # centering force
    center = zeros((1,2))
    for i in range(len(y)):
        center[0][0] += y[i][0]
        center[0][1] += y[i][1]
        
    center = center/N
    f[1:] += kappa*(center - y[1:])

    # repelling force
    for k in range(1,N):
        nn = nearest_neighbors(y[k],y,neighbors)
        repelling = 0.0
        for n in nn:
            repelling += ((y[k] - n)/((y[k] - n)**2 + delta))
        f[k] += repelling*rho
        
    return f


##
# Set up problem domain
t0 = 0.0        # start time
T = 10.0        # end time
nsteps = 500    # number of time steps

# Task:  Experiment with N, number of birds
N = 100

# Task:  Experiment with the problem parameters, and understand what they do
dt = (T - t0) / (nsteps-1.0)
gamma_1 = 2.0
gamma_2 = 8.0
alpha = 0.4
kappa = 4.0
rho = 22.0
delta = 5.0
food_flag = 1   # food_flag == 0: C(x,y) = (0.0, 0.0)
                # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))
neighbors = 5
# Intialize problem
y = rand(N,2)
flock_diam = zeros((nsteps,))

# Initialize the Movie Writer
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=12)
fig = pyplot.figure(0)
pp, = pyplot.plot([],[], 'k+') 
rr, = pyplot.plot([],[], 'r+')
ff, = pyplot.plot([],[], 'c+')
pyplot.xlabel(r'$X$', fontsize='large')
pyplot.ylabel(r'$Y$', fontsize='large')
pyplot.xlim(-3,3)       # may need to adjust if birds fly outside box
pyplot.ylim(-3,3)       # may need to adjust if birds fly outside box


# Begin writing movie frames
with writer.saving(fig, "movie.mp4",dpi=648):

    # First frame
    pp.set_data(y[1:,0], y[1:,1]) 
    rr.set_data(y[0,0], y[0,1])
    ff.set_data([C(t0,food_flag)[0]],[C(t0,food_flag)[1]])
    writer.grab_frame()

    t = t0
    for step in range(nsteps):
        
        # Task: Fill these two lines in
        y = RK4(RHS, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
#        flock_diam[step] = ... 
        t += dt
        
        # Movie frame
        pp.set_data(y[:,0], y[:,1]) 
        rr.set_data(y[0,0], y[0,1])
        ff.set_data([C(t,food_flag)[0]],[C(t,food_flag)[1]])
        writer.grab_frame()
        

# Task: Plot flock diameter
#plot(..., flock_diam, ...)

