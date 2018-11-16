+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for Homework 4, CS/MATH 471, Fall 2018, Aaron Segura, Fredrick Lee
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- There is a homework report located in the /report subdirectory
- There is code for homework 5 so a hw5/code subdirectory was made
- There are MPEG-4 (mp4) encoded movies that are in the hw5/movies
  subdirectory. You might need ffmpeg to watch them.

- In this assignment, we explore numerical modelling of a system of
  ordinary differential equations.

- This README file contains the instructions to reproduce results
  found in the report document.

**************************************************************************
NOTE: All development and testing of this software took place on the Mac
      computers in the loboLAB in UNM's student union building.
     -We cannot guarantee that this software system works on any other
      machine at this time.
**************************************************************************


- To reproduce the results in the report, do the following

  1) Clone the repo on a local machine with the following software packages
     installed:
       -python2.7
       -scipy
       -matplotlib

  2) Enter the /code subdirectory of this repository folder
    - $ cd code
    
  3) Run the following command:
    - $ python hw5_driver.py <arg_list>
    
    where arg_list is the list of parameters you'd like to initialize
    the system with.
    
    - example: $ python hw5_driver.py 1 0 100
    - This runs the simulation for the default scenario (no smelly bird
      or predator) with 100 birds in the flock. A movie is generated
      with the name movie.mp4
      
    - example: $ python hw5_driver.py 0 0 30 1.6
    - Runs a default sim (no smelly bird or predator) with all settings
      default except alpha == 1.6
      
    NOTE: this simulator uses an arbitrarily chosen seed value of 2018
          to generate the initial conditions using NumPy random number
          generation.
          Additionally, the default sim params that are not set using
          command line arguments are of the following values:
          
          t0 (time begin) : 0.0 (seconds)
          T  (time end)   : 10.0 (seconds)
          n  (# of time
              steps    )  : 50 steps
          dt (step size)  : (T - t0)/n == 0.2 (seconds)
    
    
    default arg_list: 0 0 30 0.4 2.0 8.0 4.0 2.0 0.5 1 8.0 4.0
      
    Expected command line param format     : [ options for values ]
    python hw5_driver.py <movie_option>    : [0, 1]
                         <run_option>      : [0, 1, 2, 3]
                         <number of birds> : [10, 30, 100]
                         <alpha>           : [0.1, 0.2, 0.4, 0.8, 1.6]
                         <gamma_1>         : [0.5, 1.0, 2.0, 4.0, 8.0]
                         <gamma_2>         : [0.5, 1.0, 2.0, 4.0, 8.0]
                         <kappa>           : [0.25, .5, 1.0, 2.0, 4.0]
                         <rho>             : [0.5, 1.0, 2.0, 4.0, 8.0]
                         <delta>           : [.125, .25, .5, 1.0, 2.0]
                         <food_flag>       : [0, 1]
                         <sigma>           : [1.0, 2.0, 4.0, 8.0, 16.0]
                         <pi_>             : [1.0, 2.0, 4.0, 8.0, 16.0]


    Proper usage is assumed (i.e. there are no safeguards in place for running
    non-listed param sets, this is more to facilitate compatibility with the
    flock diameter plotting file)

    Parameters are processed in order, meaning the fourth argv entry (sys.argv[3])
    is expected to be number of birds
    
    Additional params (after food_flag) are ignored

- There will be 2 files that will be created as a result of running the
    above command. The file names are as follows:
    - movie.mp4
    - flock_diam_<# of birds>_<alpha>_<gamma_1>_<gamma_2>_<kappa>_<rho>_<delta>_<food_flag>_<sigma>_<pi>.txt
    
- The flock_diam_....txt file contains only the flock diameter information
  for that particular run