"""
@author: Martin Ganahl
"""
import sys,os
# root=os.getcwd()
# os.chdir('../')
# sys.path.append(os.getcwd())#add parent directory to path
# os.chdir(root)

import numpy as np
import matplotlib.pyplot as plt
import lib.mpslib.SimContainer as SCT
import lib.mpslib.MPO as MPO
from lib.mpslib.Tensor import Tensor
import lib.mpslib.TensorNetwork as TN

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":


    d = 2            #local hilbert space dimension
    N = 100          #system size 
    Jz = np.ones(N)  #Hamiltonian parameters
    Jxy = np.ones(N) #Hamiltonian parameters


    #initializes a product state with a specifipc arrangement of up and down spins
    A = Tensor.zeros((1, 1, 2))
    A[0,0,0] = 1
    tensors = [A.copy() for _ in range(N)]
    mps=TN.FiniteMPS(tensors=tensors)

    Sp = np.array([[0, 0],[1, 0]]).view(Tensor)
    mps.apply_1site_gate(Sp, N//2)
    mps.apply_1site_gate(Sp, N//2 - 8)
    mps.apply_1site_gate(Sp, N//2 + 8)
    
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.position(N)
    mps.position(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n
    mpo=MPO.FiniteXXZ(Jz,Jxy,np.zeros(N))

    #initialize a TEBDEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    tebd = SCT.FiniteTEBDEngine(mps,mpo,"insert_name_here")
    dt = -1j*0.05  #time step
    numsteps = 20  #numnber of steps to be taken in between measurements
    Dmax = 10      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
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
        tw,it = tebd.do_steps(dt=dt, numsteps=numsteps,
                              D=Dmax,tr_thresh=thresh)
        #measure the operators 
        L=tebd.mps.measure_1site_ops(sz, range(N))
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

