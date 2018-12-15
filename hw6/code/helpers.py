from scipy import *
import sys

def savedata_thenexit(sender,name,d):
    savetxt(str(sender)+str(name)+'.txt',d,delimiter=',',newline='\n')
    sys.exit()
