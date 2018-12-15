from scipy import *
import sys

'''
expects 2 command line args

python testdata.py <name> <num_procs>

name: tag-like name of data (testing matvec(A,x,..) might warrant a name of 'Ax')

num_procs: # of procs used for parallel computation
'''

name = str(sys.argv[1])
num_procs = int(sys.argv[2])

A = loadtxt('s'+name+'.txt')
B = []
for i in range(num_procs):
    B.append(loadtxt('p'+str(i)+str(name)+'.txt'))

print('desired '+str(name))
print(A)
for i in range(num_procs):
    print('p'+str(i)+' '+str(name))
    print(B[i])

print('diff')
C = B[0]
for i in range(1,len(B)):
    C = concatenate((C,B[i]), axis = 0)
print(A-C)
