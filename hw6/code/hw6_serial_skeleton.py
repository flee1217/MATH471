from scipy import *
from matplotlib import pyplot 
from poisson import poisson

# speye generates a sparse identity matrix
from scipy.sparse import eye as speye
from numpy.linalg import inv

######################
######################
#
### Later On, you'll need to use MPI.  Here's most of what you'll need.
#
### Import MPI at start of program
# from mpi4py import MPI
#
### Initialize MPI
# comm = MPI.COMM_WORLD
#
### Get your MPI rank
# rank = comm.Get_rank()
#
### Send an array of doubles to rank 1 (from rank 0)
# comm.Send([data, MPI.DOUBLE], dest=1, tag=77)
#
### Receive that array of doubles from rank 0 (on rank 1)
# comm.Recv([data, MPI.DOUBLE], source=0, tag=77)
#
### Carry out an all-reduce, to sum (as opposed to multiple, subtract, etc...)
### part_norm and global_norm are length (1,) arrays.  The result of the global
### sum will reside in global_norm.
# comm.Allreduce(part_norm, global_norm, op=MPI.SUM)
#
### For instance, a simple Allreduce program is
# from scipy import *
# from mpi4py import MPI
# 
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# 
# part_norm = array([1.0])
# global_norm = zeros_like(part_norm) 
# 
# comm.Allreduce(part_norm, global_norm, op=MPI.SUM)
# 
# 
# if (rank == 0):
#     print global_norm
#
######################
######################


def jacobi(A, b, x0, tol, maxiter, start, start_halo, end, end_halo, N, comm):
    '''
    Carry out the Jacobi method to invert A

    Input
    -----
    A          <CSR matrix> : Matrix to invert
    b          <array>      : Right hand side
    x0         <array>      : Initial solution guess
    tol        <float>      : Halting tolerance for Jacobi
    maxiter    <int>        : Max number of allowed Jacobi iterations
    start      <int>        : First domain row owned by this processor (same as hw4)
    start_halo <int>        : Halo region 'below' this processor (same as hw4)
    end        <int>        : Last domain row owned by this processor (same as hw4)
    end_halo   <int>        : Halo region 'above' this processor (same as hw4)
    N          <int>        : Size of a domain row, excluding boundary points
                              ==> N = n-2
    comm       <MPI comm>   : MPI communicator

    Output
    ------
    x <array>       : Solution to A x = b
    '''

    # This useful function returns an array containing diag(A)
    D = A.diagonal()
    Dinv = reciprocal(D)
    
    # compute initial residual norm
    r0 = ravel(b - A*x0)
    r0 = sqrt(dot(r0, r0))

    I = speye(A.shape[0], format='csr')
    r = zeros_like(r0)
    x = x0

    x    = x.reshape(   (A.shape[0],-1))
    Dinv = Dinv.reshape((A.shape[0],-1))
    b    = b.reshape(   (A.shape[0],-1))

    x1 = I - A.multiply(Dinv)
    x2 = Dinv*b

    # Start Jacobi iterations 
    # Task in serial: implement Jacobi formula and halting if tolerance is satisfied
    # Task in parallel: extend the matrix-vector multiply to the parallel setting.  
    #                   Additionally, you'll have to compute a vector norm in parallel.
    #                   
    #                   It is suggested to write separate subroutines for the norm and 
    #                   matrix-vector product, as this will make your code much, much 
    #                   easier to debug and extend to CG.
    for i in range(maxiter):
        # Carry out Jacobi 
        x = x1*x + x2
        r = ravel(b-A*x)
        r = sqrt(dot(r,r))
        if (r/r0) < tol:
            print(str(i) + ' iterations to satisfy tolerance')
            return x

    # Task: Print out if Jacobi did not converge. In parallel, you'll want only rank 0 to print these out.    
    if (r/r0) < tol:
        print('Jacobi did not converge within' + str(maxiter) + ' steps')
    
    return x # Task: your solution vector

def euler_backward(A, u, ht, f, g, start, start_halo, end, end_halo, N, comm):
    '''
    Carry out backward Euler's method for one time step
    
    Input
    -----
    A          <CSR matrix> : Discretization matrix of Poisson operator
    u          <array>      : Current solution vector at previous time step
    ht         <scalar>     : Time step size
    f          <array>      : Current forcing vector
    g          <array>      : Current vector containing boundary condition information
    start      <int>        : First domain row owned by this processor (same as hw4)
    start_halo <int>        : Halo region 'below' this processor (same as hw4)
    end        <int>        : Last domain row owned by this processor (same as hw4)
    end_halo   <int>        : Halo region 'above' this processor (same as hw4)
    N          <int>        : Size of a domain row, excluding boundary points
                              ==> N = n-2
    comm       <MPI comm>   : MPI communicator

    Output
    ------
    u at the next time step

    '''
    
    # Task: Form the system matrix for backward Euler
    I = speye(A.shape[0], format='csr')
    G = I - ht*A

    # b is current state + ht*forcing function + ht*boundary conditions
    b = u + ht*f + ht*g
    
    # initial solution guess to G*x = u + ht*f + ht*g = b
    x0 = u.copy()

    # tolerance & maxiter get set here to match instructor func signatures in /hw6_hints.txt
    tol = 10.0**(-10)
    # 200 as maxiter is seen (probably in lecture slides somewhere)
    maxiter = 200

    # Task: return solution from Jacobi 
    k = jacobi(G, b, x0, tol, maxiter, start, start_halo, end, end_halo, N, comm)
    return k.reshape((-1,))


##
# Problem Definition
# Task: figure out the (1) exact solution, (2) f(t,x,y), and (3) g(t,x,y)
#
# u_t = u_xx + u_yy + f,    on the unit box [0,1] x [0,1] and t in a user-defined interval
#
# with an exact solution of
# u(t,x,y) = ...
#
# which in turn, implies a forcing term of f, where
# f(t,x,y) = ...
#
# u(t=0,x,y) = ...           zero initial condition
#
# g(t,x,y) = u(t,x,y) = ...    for x,y on the boundary
#



# Declare the problem
def uexact(t,x,y):
    # Task: fill in exact solution
    return sin(pi*t)*sin(pi*x)*sin(pi*y) #...

def f(t,x,y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy
    
    # Task: fill in forcing term
    return pi*cos(pi*t)*sin(pi*x)*sin(pi*y) + 2.0*pi*pi*uexact(t,x,y) #...

# Loop over various numbers of time points (nt) and spatial grid sizes (n)
error = []

######
# Choose final time and problem sizes to loop over by commenting lines in
#
# Task: Change T and the sizes for the weak and strong scaling studies
#Nt_values = ...
#N_values = ...
#T = ...
#
# Some larger sizes for a convergence study
#Nt_values = array([8, 8*4, 8*4*4, 8*4*4*4])
#N_values = array([8, 16, 32, 64, ])
#T = 0.5
#
# Two very small sizes for debugging
Nt_values = array([8, 8*4])
N_values = array([8,8])
T = 0.5
######

for (nt, n) in zip(Nt_values, N_values):
    
    # In Parllel: Adding the below line may help you have problem sizes that
    #          divide easily by numbers of processors like 8, 16, 32, and 64.  
    #n = n + 2
    
    # Declare time domain
    t0 = 0.0
    ht = (T - t0)/float(nt-1)

    # Declare spatial domain grid spacing
    h = 1.0 / (n-1.0)
    
    # Task in parallel: Compute which portion of the spatial domain the current
    #         MPI rank owns, i.e., compute start, end, start_halo, and end_halo

    # Remember, we assume a Dirichlet boundary condition, which simplifies
    # things a bit.  Thus, we only want a spatial grid from
    # [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h].  
    # We know what the solution looks like on the boundary, and don't need to solve for it.
    #
    # Task: fill in the spatial grid vector "pts"
    # Task: in parallel, you'll need this to be just the local grid plus halo region.  Mimic HW4 here.
    pts = linspace( h, 1.-h, n-2) #...)
    X,Y = meshgrid(pts, pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,)


    # Declare spatial discretization matrix
    # Task: what dimension should A be?  remember the spatial grid is from 
    #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
    #       Pass in the right size to poisson.
    # Task: in parallel, A will be just a processors local part of A
    A = poisson((n-2, n-2), format='csr')

    # Task: scale A by the grid size
    A = (1.0/h**2.0)*A

    # Declare initial condition
    #   This initial condition obeys the boundary condition.
    u0 = uexact(0, X, Y) 


    # Declare storage 
    # Task: what size should u and ue be?  u will store the numerical solution,
    #       and ue will store the exact solution.
    # Task in parallel: the size should only be for this processor's local portion of the domain.
    u = zeros((nt,A.shape[0]))
    ue = zeros((nt,A.shape[0]))

    # Set initial condition
    u[0,:] = u0
    ue[0,:] = u0

    # Run time-stepping
    for i in range(1,nt):
        
        # Task: We need to store the exact solution so that we can compute the error
        ue[i,:] = uexact(t0+i*ht,X,Y) #...
        
        # Task: Compute boundary contribution vector for the current time i*ht
        g = zeros((A.shape[0],))

        y_lower = abs(Y - h) < 10.**(-12)
        g[y_lower] += (1/h**2)*(uexact(t0+i*ht, X[y_lower], Y[y_lower] - h))

        y_upper = abs(Y - (1. - h)) < 10.**(-12)
        g[y_upper] += (1/h**2)*(uexact(t0+i*ht, X[y_upper], Y[y_upper] + h))

        x_lower = abs(X - h) < 10.**(-12)
        g[x_lower] += (1/h**2)*(uexact(t0+i*ht, X[x_lower] - h, Y[x_lower]))

        x_upper = abs(X - (1. - h)) < 10.**(-12)
        g[x_upper] += (1/h**2)*(uexact(t0+i*ht, X[x_upper] + h, Y[x_upper]))


        # Backward Euler
        # Task: fill in the arguments to backward Euler
        u[i,:] = euler_backward(A, u[i-1,:], ht, f(t0+i*ht,X,Y), g, 0, 0, n, n, n-2, 0.0)

    # Compute L2-norm of the error at final time
    e = (u[-1,:] - ue[-1,:]).reshape(-1,)
    enorm = sqrt(dot(e,e)*h**2) # Task: compute the L2 norm over space-time here.  In serial this is just one line.  In parallel, write a helper function.
    print "Nt, N, Error is:  " + str(nt) + ",  " + str(n) + ",  " + str(enorm)
    error.append(enorm)


    # You can turn this on to visualize the solution.  Possibly helpful for debugging.
    if False: 
        pyplot.figure(1)
        pyplot.imshow(u[0,:].reshape(n-2,n-2), origin='lower', extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Initial Condition")
        
        pyplot.figure(3)
        pyplot.imshow(u[-1,:].reshape(n-2,n-2))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Solution at final time")
       
        pyplot.figure(4)
        pyplot.imshow(uexact(T,X,Y).reshape(n-2,n-2))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Exact Solution at final time")
        
        pyplot.show()


# Plot convergence 
if False:
    pyplot.loglog(1./N_values, 1./N_values**2, '-ok')
    pyplot.loglog(1./N_values, array(error), '-sr')
    pyplot.tick_params(labelsize='large')
    pyplot.xlabel(r'Spatial $h$', fontsize='large')
    pyplot.ylabel(r'$||e||_{L_2}$', fontsize='large')
    pyplot.legend(['Ref Quadratic', 'Computed Error'], fontsize='large')
    pyplot.show()
    #pyplot.savefig('error.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)




