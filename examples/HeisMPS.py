"""
@author: Martin Ganahl
"""

import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import numpy as np
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib
import matplotlib.pyplot as plt
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
plt.ion()
if __name__ == "__main__":
    D=60        #final bond dimension
    d=2         #local hilbert space dimension
    N=100        #number of sites
    Jz=np.ones(N) #Hamiltonian parameters
    Jxy=np.ones(N)

    mps=mpslib.MPS.random(N=N,D=10,d=d,obc=True,dtype=float)  #initialize a random MPS with bond dimension D'=10
    print(mpslib.MPS.random.__doc__)
    mps._D=D     #set the mps final bond-dimension parameter to D; mps._D is the maximally allowed dimension of the mps
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.__position__(N)
    mps.__position__(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n
    mpo=H.XXZ(Jz,Jxy,Bz=np.zeros(N),obc=True)

    #initialize a DMRGEngine with an mps and an mpo
    dmrg=en.DMRGengine(mps,mpo,'blabla')
    print(dmrg.__doc__)
    print(dmrg.__simulate__.__doc__)
    
    print(dmrg.__simulateTwoSite__.__doc__)
    
    #start with a two site simulation with the state with bond dimension D'=10; the bond-dimension will
    #grow until it reaches mps._D
    dmrg.simulateTwoSite(1,1e-10,1e-6,40,verbose=1,solver='LAN')
    #now switch to a single site DMRG (faster) to further converge state state
    dmrg.simulate(4,1e-10,1e-10,30,verbose=1,solver='LAN')


    #a list of measurement operators to measured (here sz)
    Sz=[np.diag([0.5,-0.5]) for n in range(N)]
    
    #measure the local spin-density
    meanSz=dmrg._mps.__measureList__(Sz)

    #measure the Sz-Sz correlations
    meanSzSz=[]        
    for n in range(N):
        meanSzSz.append(dmrg._mps.__measure__([Sz[0],Sz[0]],sorted([int(N/2),n])))        


    #now truncate the state to Dt=20
    Dt=20
    dmrg._mps.truncate(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)

    #check if bond dimension has indeed been truncated
    print()
    print (dmrg._mps.D)
    #measure again ,now with truncated state
    meanSztrunc=dmrg._mps.measureList(Sz)

    meanSzSztrunc=[]        
    for n in range(N):
        meanSzSztrunc.append(dmrg._mps.measure([Sz[0],Sz[0]],sorted([int(N/2),n])))        


    #compare results before and after truncation

    plt.figure(1)
    plt.plot(range(len(meanSz)),meanSz,range(len(meanSztrunc)),meanSztrunc,'--')
    plt.ylabel(r'$\langle S^z_i\rangle$')
    plt.xlabel(r'$i$')    
    plt.legend(['before truncation (D={0})'.format(D),'after truncation (D={0})'.format(Dt)])

    plt.figure(2)
    plt.plot(range(len(meanSzSz)),meanSzSz,range(len(meanSzSztrunc)),meanSzSztrunc,'--')
    plt.ylabel(r'$\langle S^z_{N/2} S^z_{i}\rangle$')
    plt.xlabel(r'$i$')        
    plt.legend(['before truncation (D={0})'.format(D),'after truncation (D={0})'.format(Dt)])    

    plt.draw()
    plt.show()
    
    plt.draw()
    plt.show()
    input()
