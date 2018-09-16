from scipy import *
import numpy as np

def newton(f, x0, maxiter, tol):
    '''
    Newton's method of root finding
    
    input
    -----
    f           : function to find zero of
                  Interface for f:  f(x, 0)  returns f(x)
                                    f(x, 1)  returns f'(x)
    x0          : initial guess
    maxiter     : the max number of iterations allowed before
                  halting
    tol         : the minimum allowed absolute difference
                  between the two most recent root guesses
                  before halting                  
    
    output
    ------
    data   : array of approximate roots
    '''
    
    # data is a table with maxiter rows and 4 columns
    # column 1 will hold root guesses (data[0,0] is the first
    # iterated root guess)
    # column 2 will hold absolute error information
    # column 3 will hold linear convergence factors
    # column 4 will hold quadratic convergence factors
    # column 5 will hold f evaluated at the i-th x iterate
    # column 6 will hold the approximate rate of convergence
    # of the sequence
    data = zeros((maxiter,6))
    x = x0
    
    print("initial guess: %f" %x)

    for i in range(maxiter):
        xold = x
        x = x - f(x, 0)/f(x, 1)

        # do err calc here
        
        s = "Iter  " + str(i+1) + "   Approx. root  " + str(x)
        print(s)
    
        data[i,0] = x
        data[i,4] = f(x, 0)
        # delta is the absolute difference between the previous
        # and current root guess.
        delta = abs(x - xold)
        data[i,1] = delta
        # we ignore calculating the convergence factors for the 1st
        # iteration because the calculations are not well defined
        # The convergence formulas on page 2 of the HW2 pdf do not
        # allow calculating convergence factors after a single
        # iteration using newton's method
        if (i >= 1):
            data[i,2] = data[i,1] / data[ i-1 , 1]
            data[i,3] = data[i,1] / (data[ i-1 , 1] ** 2)
        
        # here, we are calculating alpha, the approximate rate of
        # convergence at each iteration
        # the value for alpha cannot be calculated without
        # two previous iterations
        if (i >= 2):
            #data[i,5] = np.log(data[i,2])/np.log(data[i-1,2])
            data[i,5] = np.log(1.0/data[i,2])/np.log(1.0/data[i-1,2])

        # if our guess difference is less than the defined tolerance
        if (delta < tol):
            break
    
    ##
    # end for-loop

    # the indices here constrain the size of the returned data set
    # because we preallocate maxiter rows but may not need all of them
    return data[0:(i+1)]

