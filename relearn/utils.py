import numpy as np
import datetime
import torch
import matplotlib.pyplot as plt
from math import floor
# from torch._C import T

def compare_weights(w1, w2):
    for i,(k1,k2) in enumerate(zip(w1,w2)):
        #print(i,k1,k2, w1[k1]-w2[k2])
        if not torch.equal(w1[k1],w2[k2]):
            return False
    return True

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



            
def prepare_batchT(device, memory, batch):
    #batch = memory.sample(size)
    steps = len(batch)
    cS, nS, act, reward, done = [], [], [], [], []
    for i in batch:
        cSi, nSi, acti, rewardi, donei, _ = memory.mem[i]
        cS.append(cSi)
        nS.append(nSi)
        act.append(acti)
        reward.append(rewardi)
        done.append(donei)
    return  steps, np.arange(steps), \
            torch.stack(cS).to(device), \
            torch.stack(nS).to(device), \
            np.array(act), \
            torch.stack(reward).to(device), \
            torch.stack(done).to(device)