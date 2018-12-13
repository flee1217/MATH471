from scipy import *
from matplotlib import pyplot as plt
from poisson import poisson
from mpi4py import MPI

# speye generates a sparse identity matrix
from scipy.sparse import eye as speye
from numpy.linalg import inv

# stderr for debugging
import sys
# using sys.stderr.write() to print stuff to console

# MPI stuff for communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.size

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

def vector_norm(v, comm):
    '''
    carry out ||v||_2
    '''
    # compute each proc's local portion of L2-norm of v
    # then do global all_reduce
    v = v.reshape(-1,)
    
    part_norm = dot(v,v)
    global_norm = zeros(1)

    comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    return sqrt(global_norm)


def mat_vec(A, x, start, start_halo, end, end_halo, N, comm):
    '''
    carry out a matvec A*x over the domain rows [start, end)
    '''
    # proc above is 'my rank + 1' and proc below is 'my rank - 1'

    # if my_rank is not 0, then send second domain row in x to the proc below
    #   This fills in the halo region on the proc below.

    # if my_rank is not nprocs-1, then send second-to-last domain row in x to the proc above
    #   This fills in the halo region on the proc above.

    # if my_rank is not 0, then receive my bottom halo row from the proc below
    #   This fills in my lower halo row (first row in x)

    # if my_rank is not nprocs-1, then receive my top halo row from the next proc
    #   This fills in my upper halo row (last row in x)
                                              


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
    r0 = vector_norm(r0, comm)

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
# f(t,x,y) = u_t - u_xx - u_yy
#
# u(t=0,x,y) = cos(pi*(0)/2.)*cos(pi*x/2.)*cos(pi*y/2.)  zero initial condition
#
# g(t,x,y) = u(t,x,y) = ...    for x,y on the boundary
#



# Declare the problem
def uexact(t,x,y):
    # Task: fill in exact solution
    return cos(pi*t/2.)*cos(pi*x/2.)*cos(pi*y/2.)

def f(t,x,y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy
    
    # Task: fill in forcing term
    return -pi/2.*sin(pi*t/2.)*cos(pi*x/2.)*cos(pi*y/2.) + (.5)*(pi**2)*uexact(t,x,y)

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
Nt_values = array([8,32])
N_values = array([8,16])
T = 0.5
######

for (nt, n) in zip(Nt_values, N_values):
    
    # In Parllel: Adding the below line may help you have problem sizes that
    #          divide easily by numbers of processors like 8, 16, 32, and 64.  
    n = n + 2
    
    # Declare time domain
    t0 = 0.0
    ht = (T - t0)/float(nt-1)

    # Declare spatial domain grid spacing
    h = 1.0 / (n-1.0)

    # own [ start, end ) rows (not including end)
    start = rank * (n-2) / num_ranks
    end = (rank + 1) * (n - 2) / num_ranks

    # determining halo rows
    start_halo = start - 1 if rank != 0 else start
    end_halo = end + 1     if rank != (num_ranks - 1) else end
    # Task in parallel: Compute which portion of the spatial domain the current
    #         MPI rank owns, i.e., compute start, end, start_halo, and end_halo

    # Remember, we assume a Dirichlet boundary condition, which simplifies
    # things a bit.  Thus, we only want a spatial grid from
    # [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h].  
    # We know what the solution looks like on the boundary, and don't need to solve for it.
    #
    # Task: fill in the spatial grid vector "pts"
    # Task: in parallel, you'll need this to be just the local grid plus halo region.
    # Mimic HW4 here.
    x_pts = linspace( h, 1.-h, n-2)
    y_pts = linspace( start_halo * h + h, (end_halo-1) * h + h, end_halo - start_halo)
    
    X,Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,)

    sys.stderr.write('proc: '+str(rank)+' X:\n' + str(X) + '\n')
    sys.stderr.write('proc: '+str(rank)+ 'Y:\n' + str(Y) + '\n')

    # Declare spatial discretization matrix
    # Task: what dimension should A be?  remember the spatial grid is from 
    #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
    #       Pass in the right size to poisson.
    # Task: in parallel, A will be just a processors local part of A
    # (x_pts, y_pts)
    A = poisson((end_halo - start_halo, n-2), format='csr')

    # Task: scale A by the grid size
    A = (1.0/h**2.0)*A

    # Declare initial condition
    #   This initial condition obeys the boundary condition.
    sys.stderr.write('rank: ' + str(rank) + ', # of grid points: ' + str(len(X)) + '\n')
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
        ue[i,:] = uexact(t0 + i*ht,X,Y) #...
        
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
        print('at time: ' + str(i))
        u[i,:] = euler_backward(A, u[i-1,:], ht, f(t0+i*ht,X,Y), g, start, start_halo, end, end_halo, n-2, comm)

    # Compute L2-norm of the error at final time
    e = (u[-1,:] - ue[-1,:]).reshape(-1,)
    enorm = sqrt(dot(e,e)*h**2)
    # Task: compute the L2 norm over space-time here.
    # In serial this is just one line.  In parallel, write a helper function.
    print "Nt, N, Error is:  " + str(nt) + ",  " + str(n-2) + ",  " + str(enorm)
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
        pyplot.imshow(u[-1,:].reshape(n-2,n-2), origin='lower', extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Solution at final time")
       
        pyplot.figure(4)
        pyplot.imshow(uexact(T,X,Y).reshape(n-2,n-2), origin='lower', extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Exact Solution at final time")

        pyplot.figure(5)
        pyplot.imshow((u[-1,:]-u[0,:]).reshape(n-2,n-2), origin='lower', extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Numerical Difference at final time")

        pyplot.figure(6)
        pyplot.imshow((abs(uexact(T,X,Y)-u[0,:])).reshape(n-2,n-2), origin='lower', extent=(0, 1, 0, 1))
        pyplot.colorbar()
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title("Exact Difference at final time")
        
        pyplot.show()


# Plot convergence 
if True and rank == 0:
    f, fplot = plt.subplots()
    fplot.loglog(1./N_values, 1./N_values**2, '-ok')
    fplot.loglog(1./N_values, array(error), '-sr')
    fplot.tick_params(labelsize='large')
    fplot.set_xlabel(r'Spatial $h$', fontsize='large')
    fplot.set_ylabel(r'$||e||_{L_2}$', fontsize='large')
    fplot.legend(['Ref Quadratic', 'Computed Error'], fontsize='large', loc = 2)
    fplot.set_title('L2-Norm of Error e vs Spatial $h$', fontsize = 'large')
    #plt.show()
    f.savefig('error.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)
