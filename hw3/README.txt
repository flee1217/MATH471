+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for Homework 3, CS/MATH 471, Fall 2018, Aaron Segura, Fredrick Lee
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- There is a homework report located in the /report subdirectory
- There is code for homework 2 so a hw2/code subdirectory was made

- In this assignment, two quadrature methods are applied to numerically
    approximate the definite integral of the function
        
        f(x) = e^( cos ( k * x ) )
        
    for values of k = { pi , pi ^ 2 }

- The two methods in question are the Trapezoidal Rule and Gaussian Quadrature
- This README file contains the instructions to reproduce these results 

**************************************************************************
NOTE: All development and testing of this software took place on the Mac
      computers in the loboLAB in UNM's student union building.
     -We cannot guarantee that this software system works on any other
      machine at this time.
**************************************************************************


- To reproduce the results in the report, do the following

  1) Clone the repo on a local machine with python 3.7, scipy, & pdfTeX installed
    -python can be installed from: https://www.python.org/downloads/
    -pdfTeX can be installed from: https://www.tug.org/applications/pdftex/
    -SciPy  can be installed from: https://scipy.org/scipylib/

  2) Enter the /code subdirectory of this repository folder
    - $ cd code
    
  3) Run the following command:
    - $ python hw3_driver.py

- There will be 5 files that will be created as a result of running the
    above command. The file names are as follows:
    - table1.tex
    - int_trap_error_plot.png
    - lgl_error_plot.png
    - lgl_plot.png
    - trapezoid_plot.png
    
- These 5 files are used in the report located in the /report subdirectory
  using the pdflatex command