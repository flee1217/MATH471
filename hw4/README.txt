+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for Homework 4, CS/MATH 471, Fall 2018, Aaron Segura, Fredrick Lee
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- There is a homework report located in the /report subdirectory
- There is code for homework 4 so a hw4/code subdirectory was made

- In this assignment, we explore parallelization and numerical differentiation
  of a function of two variables using finite difference stencils.

- This README file contains the instructions to reproduce these results 

**************************************************************************
NOTE: All development and testing of this software took place on the Mac
      computers in the loboLAB in UNM's student union building.
     -We cannot guarantee that this software system works on any other
      machine at this time.
**************************************************************************


- To reproduce the results in the report, do the following

  1) Clone the repo on a local machine with the following software packages
     installed:
       -python 2
       -scipy
       -matplotlib

  2) Enter the /code subdirectory of this repository folder
    - $ cd code
    
  3) Run the following command:
    - $ python hw4_driver.py

- There will be 5 files that will be created as a result of running the
    above command. The file names are as follows:
    - error.txt
    - timings.txt
    
- There are 3 problem size option numbers to explore and set in the
  hw4_driver.py file: 1, 2, and 3
    - Running the hw4_driver file will output error and timing information
      for the given problem size option number, which will be located
      in the mentioned files:
        - error.txt
        - timings.txt
    - It is important to note that running hw4_driver.py multiple times
      will result in error.txt and timings.txt containing error and timing
      information for the most recent run.