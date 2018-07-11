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
    D=32
    d=2
    N=2
    Jx=np.ones(N)
    Bz=0.5*np.ones(N)
    #initialize MPS with bond dimension D
    dtype=float
    mps=mpslib.MPS.random(N=N,D=D,d=d,obc=False,dtype=dtype)
    mpo=H.TFI(Jx,Bz,False)
    idmrg=en.IDMRGengine(mps,mpo,'blabla')
    
    #idmrg.__simulateTwoSite__(Nmax=1000,NUC=1,Econv=1E-10,tol=1E-4,ncv=10,cp=None,verbose=1,truncation=1E-8,regaugestep=0) #two site idmrg
    print(idmrg.__doc__)
    print(idmrg.__simulate__.__doc__)
    idmrg.__simulate__(Nmax=1000,NUC=2,solver='LAN',Econv=1E-10,tol=1E-6,ncv=20,cp=None,verbose=1,regaugestep=0) #single site idmrg
    
