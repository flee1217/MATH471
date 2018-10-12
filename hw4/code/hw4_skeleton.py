from scipy import *
from scipy import sparse
import time
import numpy as np
from threading import Thread
from matplotlib import pyplot
from poisson import poisson
from scipy.linalg import norm as norm

def l2norm(e, h):
    '''
    Take L2-norm of e
    '''
    # ensure e has a compatible shape for taking a dot-product
    e = e.reshape(-1,) 

    # Task:
    # Return the L2-norm, i.e., the square roof of the integral of e^2
    # Assume a uniform grid in x and y, and apply the midpoint rule.
    # Assume that each grid point represents the midpoint of an equally sized region
    return norm(e)


def compute_fd(n, nt, k, f, fpp_num):
    '''
    Compute the numeric second derivative of the function 'f'

    Input
    -----
    n   <int>       :   Number of grid points in x and y for global problem
    nt  <int>       :   Number of threads
    k   <int>       :   My thread number
    f   <func>      :   Function to take second derivative of
    fpp_num <array> :   Global array of size n**2


    Output
    ------
    fpp_num will have this thread's local portion of the second derivative
    written into it


    Notes
    -----
    We do a 1D domain decomposition.  Each thread 'owns' the k(n/nt) : (k+1)(n/nt) rows
    of the domain.  
    
    For example, 
    Let the global points in the x-dimension be [0, 0.33, 0.66, 1.0] 
    Let the global points in the y-dimension be [0, 0.33, 0.66, 1.0] 
    Let the number of threads be two (nt=2)
    
    Then for the k=0 case (for the 0th thread), the rows of the 
    domain 'owned' are
    y = 0,    and x = [0, 0.33, 0.66, 1.0]
    y = 0.33, and x = [0, 0.33, 0.66, 1.0]
    
    Then For the k = 1, case, the rows of the domain owned are
    y = 0.66, and x = [0, 0.33, 0.66, 1.0]
    y = 1.0,  and x = [0, 0.33, 0.66, 1.0]

    We assume that n/nt divides evenly.

    '''
    
    # Task: 
    # Compute start, end, start_halo, and end_halo
    start = <first row owned by thread k>
    end = <first row owned by thread k+1>
    
    # Task:
    # Halo regions essentially expand a thread's local domain to include enough
    # information from neighboring threads to carry out the needed computation.
    # For the above example, that means including 
    #   - the y=0.66 row of points for the k=0 case
    #   - the y=0.33 row of points for the k=1 case
    start_halo = <start - 1, unless k == 0>
    end_halo = <end + 1, unless k == (nt-1)>

    
    # Construct local CSR matrix.  Here, you're given that function for free
    A = poisson((end_halo - start_halo, n), format='csr')
    h = 1./(n-1)
    A *= 1/h**2

    # Task:
    # Inspect a row or two of A, and verify that it's the correct 5 point stencil

    # Task:
    # Construct grid of evenly spaced points over this thread's halo region 
    x_pts = linspace(0,1,n)
    y_pts = <linspace(start_halo*h, ... )> 
    
    # Task:
    # There is no coding to do here, but examime how meshgrid works and 
    # understand how it gives you the correct uniform grid.
    X,Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,) 

    # Compute local portion of f 
    f_vals = f(X,Y)
    
    # Task:
    # Compute the correct range of output values for this thread
    output = A*f_vals
    < output array should be truncated to exclude values in the halo region >
    < you will need special cases to account for the first and last threads >
    
    # Task:
    # Set the output array
    fpp_num[ <insert indices> ] = <insert output> 


def fcn(x,y):
    '''
    This is the function we are studying
    '''
    return cos((x+1)**(1./3.) + (y+1)**(1./3.)) + sin((x+1)**(1./3.) + (y+1)**(1./3.))

def fcnpp(x,y):
    '''
    This is the second derivative of the function we are studying
    '''
    # Task:
    # Fill this function in with the correct second derivative.  You end up with terms like
    # -cos((x+1)**(1./3.) + (y+1)**(1./3.))*(1./9.)*(x+1)**(-4./3)
    return <insert code> 


##
# Here are three problem size options for running.  The instructor has chosen these
# for you.
option = 3
if option == 1:
    # Choose this if doing a final run on CARC
    NN = array([840*6])
    num_threads = [1,2,3,4,5,6,7,8]
elif option == 2:
    # Choose this for printing convergence plots on your laptop/lab machine,
    # and for initial runs on CARC.  
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = 210*arange(1,6)
    num_threads = [1] #eventually include 2, 3, 4
elif option == 3:
    # Choose this for code development and debugging on your laptop/lab machine
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = array([6])
    num_threads = [1] #eventually include 2,3
else:
    print("Incorrect Option!")

##
# Begin main computation loop
##

# Task:
# Initialize your data arrays
error = np.zeros((num_threads[0],NN[0]))   # <array of zeros of size num_threads X size of NN >
timings = np.zeros((num_threads[0],NN[0])) # <array of zeros of size num_threads X size of NN> 

# Loop over various numbers of threads
for i,nt in enumerate(num_threads):
    # Loop over various problem sizes 
    for j,n in enumerate(NN):
        
        # Task:
        # Initialize output array
        fpp_numeric = np.zeros((NN-2,NN-2)) # <array of zeros of appropriate size> 
        
        # Task:
        # Choose the number of timings to do for each thread number, domain size combination 
        ntimings = 10 # <insert>

        # Time over 10 tries
        min_time = 10000
        for m in range(ntimings):

            # Compute fpp numerically in the interior of each thread's domain
            #compute_fd(n,nt,0,fcn,fpp_numeric)
            t_list = []
            for k in range(nt):
                # Task:
                # Finish this call to Thread(), passing in the correct target and arguments
                t_list.append(Thread(target=compute_fd, args=(NN,num_threads,i,fcn,fpp_numeric) )) 

            start = time.time()
            # Launch and Join Threads
            for t in t_list:
                t.start()
            for t in t_list:
                t.join()
            end = time.time()
            min_time = min(end-start, min_time)
        ##
        # End loop over timings

        # Construct grid of evenly spaced points for a reference evaluation of
        # the double derivative
        h = 1./(n-1)
        pts = linspace(0,1,n)
        X,Y = meshgrid(pts, pts)
        X = X.reshape(-1,)
        Y = Y.reshape(-1,) 
        fpp = fcnpp(X,Y)

        # Account for domain boundaries.  
        #
        # The boundary_points array is a Boolean array, that acts like a
        # mask on an array.  For example if boundary_points is True at 10
        # points and False at 90 points, then x[boundar_points] will be a
        # length 10 array at those True locations
        boundary_points = (Y == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points]-h)
        
        # Task:
        # Account for the domain boundaries at Y == 1, X == 0, X == 1
        
        # Task:
        # Compute error
        h = # <insert> 
        e = # <insert> 
        error[i,j] = <insert l2norm> 
        timings[i,j] = min_time
        print min_time
    ##
    # End Loop over various grid-sizes
    print " "
    
    # Task:
    # Generate and save plot showing convergence for this thread number
    # Do not try to plot on CARC. 
    pyplot.loglog(NN, <array slice of error values>) 
    pyplot.loglog(NN, <reference quadratic convergence data>)
    <insert nice formatting options with large axis labels, tick fontsizes, and large legend labels>
    pyplot.savefig('error'+str(i)+'.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)


# Save timings for future use
savetxt('timings.txt', timings)

