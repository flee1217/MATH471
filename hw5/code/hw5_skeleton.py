from scipy import *
from matplotlib import pyplot
import matplotlib.animation as manimation
import os, sys

# HW5 solution by Jacob

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
    k1 = f(...)
    k2 = f(...)
    k3 = f(...)
    k4 = f(...)

    return ... 


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Define the right hand side of the ODE

    '''
    N = y.shape[0]
    f = zeros_like(y)
    
    # Task:  Fill this in by assigning values to f
    
    return f


##
# Set up problem domain
t0 = 0.0        # start time
T = 10.0        # end time
nsteps = 50     # number of time steps

# Task:  Experiment with N, number of birds
N = 30

# Task:  Experiment with the problem parameters, and understand what they do
dt = (T - t0) / (nsteps-1.0)
gamma_1 = 2.0
gamma_2 = 8.0
alpha = 0.4
kappa = 4.0
rho = 2.0
delta = 0.5
food_flag = 1   # food_flag == 0: C(x,y) = (0.0, 0.0)
                # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))

# Intialize problem
y = rand(N,2)
flock_diam = zeros((nsteps,))

# Initialize the Movie Writer
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=6)
fig = pyplot.figure(0)
pp, = pyplot.plot([],[], 'k+') 
rr, = pyplot.plot([],[], 'r+') 
pyplot.xlabel(r'$X$', fontsize='large')
pyplot.ylabel(r'$Y$', fontsize='large')
pyplot.xlim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!
pyplot.ylim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!


# Begin writing movie frames
with writer.saving(fig, "movie.mp4", dpi=1000):

    # First frame
    pp.set_data(y[1:,0], y[1:,1]) 
    rr.set_data(y[0,0], y[0,1]) 
    writer.grab_frame()

    t = t0
    for step in range(nsteps):
        
        # Task: Fill these two lines in
        y = RK4(...) 
        flock_diam[step] = ... 
        t += dt
        
        # Movie frame
        pp.set_data(y[:,0], y[:,1]) 
        rr.set_data(y[0,0], y[0,1]) 
        writer.grab_frame()
        

# Task: Plot flock diameter
plot(..., flock_diam, ...)

