#!/usr/bin/env python
import time,random
from sys import stdout
import functools as fct
import numpy as np
import cython_modules.cython_modules.XXZED as ed
import scipy as sp
import lib.Lanczos.LanczosEngine as lanEn
from lib.utis.binaryoperations import getBit,flipBit
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix



def XXZGrid(Jxy,Jz,N,basis,grid):
    num2ind={}
    for n in range(len(basis)):
        num2ind[basis[n]]=n
    Jp=Jxy/2.0
    diag={}
    nondiag={}
    N0=len(basis)

    for n in range(len(basis)):
        if n%1000==0:
            stdout.write("\r building Hamiltonian ... %2.2f percent done" %(100.0*n/N0))
            stdout.flush()
        state=basis[n]
        szsz=0

        for s in range(N):
            sz=(getBit(state,s)-0.5)*2
            for p in range(len(grid[s])):
                nei=grid[s][p]
                szsz+=sz*(getBit(state,nei)-0.5)*2*Jz[s,p]/4.0
        if (abs(szsz)>1E-5):
            diag[(n,n)]=szsz
        for s in range(N):
            for p in range(len(grid[s])):
                nei=grid[s][p]
                if getBit(state,s)!=getBit(state,nei):
                    newstate=flipBit(flipBit(state,s),nei)
                    index=(num2ind[newstate],n)
                    nondiag[index]=Jp[s,p]

    indxoffdiag=np.zeros(len(nondiag)).astype(int)
    indyoffdiag=np.zeros(len(nondiag)).astype(int)
    indxdiag=np.zeros(len(diag)).astype(int)
    indydiag=np.zeros(len(diag)).astype(int)
    valdiag=np.zeros(len(diag))
    valoffdiag=np.zeros(len(nondiag))

    n=0
    for inds,v in zip(diag.keys(),diag.values()):
        indxdiag[n]=inds[0]
        indydiag[n]=inds[1]
        valdiag[n]=v
        n+=1
    n=0
    for inds,v in zip(nondiag.keys(),nondiag.values()):
        indxoffdiag[n]=inds[0]
        indyoffdiag[n]=inds[1]
        valoffdiag[n]=v
        n+=1
    return csc_matrix((valdiag,(indxdiag, indydiag)),shape=(len(basis), len(basis)))+csc_matrix((valoffdiag,(indxoffdiag, indyoffdiag)),shape=(len(basis), len(basis)))

def periodicXXZchain(Jxy,Jz,N,basis):
    num2ind={}
    for n in range(len(basis)):
        num2ind[basis[n]]=n

    Jp=Jxy/2.0
    diag={}
    nondiag={}
    N0=len(basis)
    
    for n in range(len(basis)):
        if n%1000==0:
            stdout.write("\r building Hamiltonian ... %2.2f percent done" %(100.0*n/N0))
            stdout.flush()
        state=basis[n]
        binrep=bin(state)
        if (len(binrep)==N+2) and binrep[2]=='1':
            shifted=((np.uint64(state-2**(N-1))<<np.uint64(1))+np.uint64(1))
        else:
            shifted=np.uint64(state<<np.uint64(1))

        minus=bin(state^shifted).count('1')
        plus=bin(~(state^shifted))[-N:].count('1')

        if (plus!=minus):
            diag[(n,n)]=(plus-minus)*Jz/4.0

        #case 1: binrep is shorter than N (it has a leading 0), but the first bit (on the left side, i.e. site 0) is 1
        #->flip the last and the first bit
        if binrep[-1]=='1' and (len(binrep)-2)<N:
            newstate=flipBit(flipBit(state,N-1),0)
            index=(num2ind[newstate],n)
            nondiag[index]=Jp

        #run through binrep, starting at the right end; note that flipBit starts counting bits from the right, i.e.
        #the rightmost bit in binrep has index 0 for flipBit
        #this means that pos=len(binrep)-1-s, i.e. the last bit in binrep (s=len(bitprep)-1) is mapped to pos=0
        #if a domain wall is found, flip the spins on either side
        #this loop stops at s=3, such that pos+1=len(binrep)-3 is the first significant entry of binrep
        #note that binrep has entries of the form '0bxxx', where xxx are binary numbers (0,1) and 0b signifies that 
        #binrep is a str. 0b is always ignored
        for s in range(len(binrep)-1,2,-1):
            if binrep[s]!=binrep[s-1]:
                pos=len(binrep)-1-s
                newstate=flipBit(flipBit(state,pos+1),pos)
                index=(num2ind[newstate],n)
                nondiag[index]=Jp

        #now if Nbin<N, which means that binrep has some 0 after 0b that have been omitted by the python, and the first
        #non-trivial entry in binrep (start counting from the left this is at s=2) is a 1, we need to flip that last entry to 0 and introduce
        #a 1 to the left of it (at pos=len(binrep)-2 for flipBit, when starting to count from the right).
        if (len(binrep)-2)<N:
            newstate=flipBit(flipBit(state,len(binrep)-2),len(binrep)-3)
            index=(num2ind[newstate],n)
            nondiag[index]=Jp

        #final case: len(binrep)-2==N (that is, there is no leading 0 left), and the right-most index is a 0, then flip the first and the last bits
        if ((len(binrep)-2)==N) and (binrep[-1]=='0'):
            newstate=flipBit(flipBit(state,N-1),0)
            index=(num2ind[newstate],n)
            nondiag[index]=Jp

    indxoffdiag=np.zeros(len(nondiag)).astype(int)
    indyoffdiag=np.zeros(len(nondiag)).astype(int)
    indxdiag=np.zeros(len(diag)).astype(int)
    indydiag=np.zeros(len(diag)).astype(int)
    valdiag=np.zeros(len(diag))
    valoffdiag=np.zeros(len(nondiag))

    n=0
    for inds,v in zip(diag.keys(),diag.values()):
        indxdiag[n]=inds[0]
        indydiag[n]=inds[1]
        valdiag[n]=v
        n+=1
    n=0
    for inds,v in zip(nondiag.keys(),nondiag.values()):
        indxoffdiag[n]=inds[0]
        indyoffdiag[n]=inds[1]
        valoffdiag[n]=v
        n+=1
    return csc_matrix((valdiag,(indxdiag, indydiag)),shape=(len(basis), len(basis)))+csc_matrix((valoffdiag,(indxoffdiag, indyoffdiag)),shape=(len(basis), len(basis)))


