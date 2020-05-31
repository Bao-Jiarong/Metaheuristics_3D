'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-05-28
 *  Modified On: 2020-05-28
 '''
import math
import random
import numpy as np

def randfloat(low,high):
    return low+((high-low)*random.random())

class Algo:
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,f,n,T,verbose=False):
        self.f      =  f
        self.n      =  n
        self.T      =  T
        self.verbose=  verbose

    #----------------------------------------------------------
    def set_verbose(self,status):
        self.verbose = status
