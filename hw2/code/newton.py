from scipy import *

def newton(f, x0):
    '''
    Newton's method of root finding
    
    input
    -----
    f           : function to find zero of
                  Interface for f:  f(x, 0)  returns f(x)
                                    f(x, 1)  returns f'(x)
    x0          : initial guess
    <maxiter>   :
    <tol>       :
    
    output
    ------
    data   : array of approximate roots
    '''
    

    maxiter = 10        # delete this line after you add a maxiter parameter
    data = zeros((maxiter,1))
    x = x0
    
    for i in range(maxiter):
        xold = x
        x = x - f(x, 0)/f(x, 1)
        
        s = "Iter  " + str(i) + "   Approx. root  " + str(x)
        print(s)
    
        data[i,0] = x

        # Insert some logic for halting, if the tolerance is met
        #if some_condition :
        #   break
    
    ##
    # end for-loop

    return data[0:(i+1)]

