#!/usr/bin/env python
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import numpy as np
import matplotlib.pyplot as plt
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":
    D=100        #final bond dimension
    d=2         #local hilbert space dimension
    N=101        #number of sites
    Jz=np.ones(N).astype(float) #Hamiltonian parameters
    Jxy=np.ones(N).astype(float)

    mps=mpslib.MPS.random(N=N,D=10,d=d,obc=True,dtype=float)  #initialize a random MPS with bond dimension D'=10
    mps._D=D     #set the mps final bond-dimension parameter to D; mps._D is the maximally allowed dimension of the mps
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.position(N)
    mps.position(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n
    mpo=H.XXZ(Jz,Jxy,Bz=np.zeros(N),obc=True)

    #initialize a DMRGEngine with an mps and an mpo
    dmrg=en.DMRGengine(mps,mpo,'blabla')
    #start with a two site simulation with the state with bond dimension D'=10; the bond-dimension will
    #grow until it reaches mps._D
    dmrg.simulateTwoSite(2,1e-10,1e-6,40,verbose=1,solver='LAN')
    #now switch to a single site DMRG (faster) to further converge state state
    dmrg.simulate(3,1e-10,1e-10,30,verbose=1,solver='LAN')
    dmrg._mps.position(0)

    #initialize a TimeEvolutionEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    dmrg._mps.applyOneSiteGate(np.asarray([[0.0,1.],[0.0,0.0]]),50)
    dmrg._mps.position(N)
    dmrg._mps.position(0)

    #initialize a TEBDEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    engine=en.TimeEvolutionEngine(dmrg._mps,mpo,"insert_name_here")
    dt=-1j*0.05  #time step
    numsteps=20  #numnber of steps to be taken in between measurements
    Dmax=40      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                 #adapted to this value in the TEBDEngine
    thresh=1E-8  #truncation threshold
    Nmax=1000    #number of measurements
    SZ=np.zeros((Nmax,N)) #container for holding the measurements
    plt.ion()
    sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
    it=0  #counts the total iteration number
    tw=0  #accumulates the truncated weight (see below)
    for n in range(Nmax):

        #measure the operators 
        L=engine.measureLocal(sz)
        #store result for later use
        SZ[n,:]=L
        tw,it=engine.doTEBD(dt=dt,numsteps=numsteps,Dmax=Dmax,tr_thresh=thresh,cnterset=it,tw=tw)
        #plot 
        plt.figure(1)
        plt.clf()
        plt.plot(SZ[n,:])
        plt.ylim([-0.5,0.5])
        plt.draw()
        plt.show()
        plt.pause(0.01)
