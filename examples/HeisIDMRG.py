"""
@author: Martin Ganahl
"""
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
    D=50
    d=2
    N=2
    Jz=np.ones(N)
    Jxy=np.ones(N)
    #initialize MPS with bond dimension D
    dtype=complex
    mps=mpslib.MPS.random(N=N,D=D,d=d,obc=False,dtype=dtype)
    mpo=H.XXZ(Jz,Jxy,np.zeros(N),False,dtype=dtype)
    idmrg=en.IDMRGengine(mps,mpo,'blabla')
    
    #idmrg.__simulateTwoSite__(Nmax=1000,NUC=1,Econv=1E-10,tol=1E-4,ncv=10,cp=None,verbose=1,truncation=1E-8,regaugestep=3) #two site idmrg
    print(idmrg.__doc__)
    print(idmrg.__simulate__.__doc__)
    idmrg.simulate(Nmax=100,NUC=2,solver='LOBPCG',Econv=1E-10,tol=1E-6,ncv=20,cp=None,verbose=1,regaugestep=0) #single site idmrg

    
