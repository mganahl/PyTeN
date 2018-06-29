#!/usr/bin/env python
import numpy as np
import sys
import itertools
#import utils
herm=lambda x:np.conj(np.transpose(x))
#from exceptions import TensorKeyError
#from exceptions import TensorSizeError

def getKeyShapePairs(n,indices,keylist,shapelist,key,shape):
    if n<len(indices)-1:
        for q in indices[n]._Q:
            if n==0:
                key=[None]*len(indices)
                shape=[None]*len(indices)
            key[n]=q
            shape[n]=indices[n]._Q[q]
            getKeyShapePairs(n+1,indices,keylist,shapelist,key,shape)
    else:
        Q=key[0]*indices[0]._flow
        for p in range(1,len(indices)-1):
            Q+=(key[p]*indices[p]._flow)
        if Q*(-indices[n]._flow) in indices[n]._Q:
            key[n]=Q*(-indices[n]._flow)
            shape[n]=indices[n]._Q[Q*(-indices[n]._flow)]

            keylist.append(tuple(key))
            shapelist.append(tuple(shape))            

def tupmult(t1,t2):
    assert(len(t1)==len(t2))
    mul=t1
    for n in range(len(mul)):
        mul[n]*=t2[n]
    return mul
    
def tupsignmult(t1,t2):
    assert(len(t1)==len(t2))
    mul=list(t1)
    for n in range(len(mul)):
        mul[n]*=np.sign(t2[n])
    return tuple(mul)


#flattens an arbitrarily nested tuple of tuples
def flatten(data):
    r=tuple([])
    if type(data)==tuple:
        for d in data:
            if type(d)==tuple:
                r+=flatten(d)
            if type(d)!=tuple:
                r+=tuple([d])
        return r

    elif type(data)!=tuple:
        return tuple([data])

def prod(data):
    z=1
    for d in data:
        z*=d
    return z


def flipsigns(nestedtuple):
    if isinstance(nestedtuple,tuple):
        newtup=tuple([])
        for n in range(len(nestedtuple)):
            if type(nestedtuple[n])==tuple:
                newtup+=tuple([flipsigns(nestedtuple[n])])
            else:
                newtup+=tuple([-nestedtuple[n]])
        return newtup
    else:
        return -nestedtuple

#returns a boolean and a dict of the different values for the common keys of d1 and d2
def dictcomp(d1,d2,verbose=True):
    diff1=set(d1.keys()).difference(d2.keys())
    diff2=set(d2.keys()).difference(d1.keys())
    isidentical=False
    d3={}
    if verbose:
        print ('dictcomp comparing d1 and d2:')
    if len(diff1)>0:
        if verbose:
            print ('following keys are in d1 and not in d2: ',list(diff1))
    if len(diff2)>0:
        if verbose:
            print ('following keys are in d2 and not in d1: ',list(diff2))
    if (len(diff1)==0) and (len(diff2)==0):
        if verbose:
            print ('d1 and d2 have identical keys')
        intersection=sorted(list(set(d2.keys()).intersection(d1.keys())))

        for comk in intersection:
            if d1[comk]!=d2[comk]:
                d3[comk]=tuple([d1[comk],d2[comk]])
        if len(d3.keys())!=0:
            if verbose:
                print ('d1 and d2 differ in the following common keys:', d3)
        if len(d3.keys())==0:
            if verbose:
                print ('d1 and d2 are identical')
            isidentical=True
    return isidentical,d3
