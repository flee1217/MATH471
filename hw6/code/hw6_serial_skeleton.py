from scipy import *
from matplotlib import pyplot 
from poisson import poisson

# speye generates a sparse identity matrix
from scipy.sparse import eye as speye

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


def jacobi(A, b, x0, tol, maxiter):
    '''
    Carry out the Jacobi method to invert A

    Input
    -----
    A <CSR matrix>  : Matrix to invert
    b <array>       : Right hand side
    x0 <array>      : Initial solution guess

    Output
    ------
    x <array>       : Solution to A x = b
    '''

    # This useful function returns an array containing diag(A)
    D = A.diagonal()
    
    # compute initial residual norm
    r0 = ravel(b - A*x0)
    r0 = sqrt(dot(r0, r0))

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


    # Task: Print out if Jacobi did not converge. In parallel, you'll want only rank 0 to print these out.

    return # Task: your solution vector

def euler_backward(A, u, ht, f, g):
    '''
    Carry out backward Euler's method for one time step
    
    Input
    -----
    A <CSR matrix>  : Discretization matrix of Poisson operator
    u <array>       : Current solution vector at previous time step
    ht <scalar>     : Time step size
    f <array>       : Current forcing vector
    g <array>       : Current vector containing boundary condition information

    Output
    ------
    u at the next time step

    '''
    
    # Task: Form the system matrix for backward Euler
    I = speye(A.shape[0], format='csr') 
    G = I - #... 
    
    # Task: return solution from Jacobi 
    return jacobi(G, #... 


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
    return #...

def f(t,x,y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy
    
    # Task: fill in forcing term
    return #...

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
N_values = array([8,16])
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
    pts = linspace(#...)
    X,Y = meshgrid(pts, pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,)


    # Declare spatial discretization matrix
    # Task: what dimension should A be?  remember the spatial grid is from 
    #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
    #       Pass in the right size to poisson.
    # Task: in parallel, A will be just a processors local part of A
    A = poisson((sizex, sizey, format='csr')

    # Task: scale A by the grid size
    A = #... 


    # Declare initial condition
    #   This initial condition obeys the boundary condition.
    u0 = uexact(0, X, Y) 


    # Declare storage 
    # Task: what size should u and ue be?  u will store the numerical solution,
    #       and ue will store the exact solution.
    # Task in parallel: the size should only be for this processor's local portion of the domain.
    u = zeros(#...
    ue = zeros(#...

    # Set initial condition
    u[0,:] = u0
    ue[0,:] = u0

    # Run time-stepping
    for i in range(1,nt):
        
        # Task: We need to store the exact solution so that we can compute the error
        ue[i,:] = uexact(#...
        
        # Task: Compute boundary contribution vector for the current time i*ht
        g = zeros((A.shape[0],)) 
        boundary_points = (Y == h)
        #g[boundary_points] += ... 

        # Backward Euler
        # Task: fill in the arguments to backward Euler
        u[i,:] = euler_backward(A, u[i-1,:], #... 
        

    # Compute L2-norm of the error at final time
    e = (u - ue).reshape(-1,)
    enorm = # Task: compute the L2 norm over space-time here.  In serial this is just one line.  In parallel, write a helper function.
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




