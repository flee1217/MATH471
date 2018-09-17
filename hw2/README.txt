+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for Homework 2, Math 471, Fall 2018, Aaron Segura, Fredrick Lee
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- There is a homework report is in the directory.
- There is code for homework 2 so a hw2/code subdirectory was made
- The purpose of this homework is to familarize the class with managing computations through scripting and automation.
- Here we will automate computations to find the roots of a function f(x), 
- i.e., approximate  solutions  to  the  equation f(x)  =  0.   
- We  will  consider  three  different  functions:
    -f(x) = x
    -f(x) = x^2
    -f(x) = sin(x) + cos(x^2)
- We will use Newton’s method to find the roots.
- This README file contains the instructions to reproduce these results 

- To reproduce the results in the report, do the following

  1) Clone the repo on a local machine with python 3.7, scipy, & pdfTeX installed
    -python can be installed from: https://www.python.org/downloads/
    -pdfTeX can be installed from: https://www.tug.org/applications/pdftex/
    -SciPy  can be installed from: https://scipy.org/scipylib/

  2) Enter the /code subdirectory of this repository folder
    - $ cd code
    
  3) Run the following command:
    - $ python drive_newton.py

- There will be 9 files that will be created as a result of running that
    command. The file names are of the form:
    - newton_data*.tex
    - newton_fig*.png
    - newton_plot*.png
        where * is the index of the function in fcn_list
        NOTE: fcn_list in drive_newton.py is of the order
            1 :: f(x) = x^2
            2 :: f(x) = x
            3 :: f(x) = sin(x) + cos(x^2)

- These 9 files are used in the report located in the /report subdirectory
  using the pdflatex command