import numpy as np
from math import floor
import datetime
import matplotlib.pyplot as plt


def shape2size(shape):
    res = 1
    for d in shape:
        res*=d
    return res     
    
def stamp_now(format="%Y_%m_%d_%H_%M_%S"):
    return datetime.datetime.strftime(datetime.datetime.now(), format)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num, base, digs):
    """ convert base-10 integer to base-n array of fixed no. of digits 
    return array (of length = digs)"""
    res = np.zeros(digs, dtype=np.int32)
    q = num
    for i in range(digs):
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(arr, base):
    """ convert array from given base to base-10  --> return integer"""
    res = 0
    for i in range(len(arr)):
        res+=(base**i)*arr[i]
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="[", sep=",", end="]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for i in range(len(arr)):
        res+=str(arr[i])+sep
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=   
def strD(arr, sep="\n", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-="+sep+"DICT: "+caption+sep+"=-=-=-=-==-=-=-=-="+sep
    for i in arr:
        res+=str(i) + "\t\t" + str(arr[i]) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strV(SV):
    """ converts state vector to appropiate string for storing in Q-dictionary --> return string """
    if (type(SV)==np.ndarray):
        return strA(SV)
    else:
        return str(SV)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


from scipy.sparse.csgraph import floyd_warshall
#~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+
def effective_bandwidth(Sparse_M):
    """ function for finding shortest path and effective Data Rate (bandwidth) b/w edge or cloud servers 
        return dist_matrix """
    M=np.copy(Sparse_M)
    lm = len(M)
    for i in range(lm):
        for j in range(i+1, lm):
            if M[i,j]!=0:
                M[i,j] = 1/M[i,j]
            M[j,i] = M[i,j]
    dist_matrix, predecessors = floyd_warshall(csgraph=M, 
                directed=False,return_predecessors=True)
    return dist_matrix #<--- this turns the diagonals to inf  #print(dist_matrix)
#~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+

    
    
    
