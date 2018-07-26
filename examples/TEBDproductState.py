"""
@author: Martin Ganahl
"""
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


    d=2            #local hilbert space dimension
    N=100          #system size 
    Jz=np.ones(N)  #Hamiltonian parameters
    Jxy=np.ones(N) #Hamiltonian parameters


    #initializes a random MPS with bond dimension D for open boundary conditions
    #D=10
    #mps=mpslib.MPS.random(N=N,D=D,d=[d]*N,obc=True)

    #initializes a product state with a specific arrangement of up and down spins
    #state is an array of length N defining which state should be put at each site;
    #values at state[n] can be in {0,1,..,d-1}
    state=[np.asarray([1,0])]*N              #initialize with all spins down
    state[int(N/2)-10]=np.asarray([0,1])        #put an up spin at site int(N/2)-10
    state[int(N/2)]=np.asarray([0,1])        #put an up spin at site int(N/2)
    state[int(N/2)+10]=np.asarray([0,1])     #put an up spin at site int(N/2)+10
    mps=mpslib.MPS.productState(state,obc=True) #this mps for now has a maximally allowed bond dimension mps._D=1;

    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.position(N)
    mps.position(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n
    mpo=H.XXZ(Jz,Jxy,np.zeros(N),obc=True)


    #initialize a TEBDEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    engine=en.TimeEvolutionEngine(mps,mpo,"insert_name_here")
    dt=-1j*0.05  #time step
    numsteps=20  #numnber of steps to be taken in between measurements
    Dmax=10      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                 #adapted to this value in the TEBDEngine
    thresh=1E-8  #truncation threshold
    Nmax=1000    #number of measurements
    SZ=np.zeros((Nmax,N)) #container for holding the measurements
    plt.ion()
    sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
    it=0  #counts the total iteration number
    tw=0  #accumulates the truncated weight (see below)
    for n in range(Nmax):
        #do numsteps TEBD steps 
        tw,it=engine.doTEBD(dt=dt,numsteps=numsteps,Dmax=Dmax,tr_thresh=thresh,cnterset=it,tw=tw)
        #measure the operators 
        L=engine._mps.measureList(sz)
        #store result for later use
        SZ[n,:]=L

        #plot 
        plt.figure(1)
        plt.clf()
        plt.plot(SZ[n,:])
        plt.ylim([-0.5,0.5])
        plt.draw()
        plt.show()
        plt.pause(0.01)

