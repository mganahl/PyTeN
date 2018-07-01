#!/usr/bin/env python
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)
import numpy as np
import scipy as sp
import math
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":
    D=100
    D0=10
    d=2
    N=4
    Jz=np.ones(N)
    Jxy=np.ones(N)
    #initialize MPS with bond dimension D0
    mps=mpslib.MPS.random(N=N,D=D0,d=d,obc=False,dtype=float)

    #set the bond-dimension cutoff to D
    #mps._D=D
    mps.__position__(N)
    mps.__position__(0)
    mpoobc=H.XXZ(Jz,Jxy,np.zeros(N),False)
    lb=np.ones((D0,D0,1))
    rb=np.ones((D0,D0,1))
    idmrg=en.IDMRGengine(mps,mpoobc,'blabla')
    idmrg.__simulateTwoSite__(Nmax=100,NUC=2,Econv=1E-10,tol=1E-6,ncv=20,cp=None,verbose=1,truncation=1E-8)
    #idmrg.__simulate__(Nmax=100,NUC=2,Econv=1E-10,tol=1E-6,ncv=20,cp=None,verbose=1)

