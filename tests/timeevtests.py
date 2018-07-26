#!/usr/bin/env python
import sys,os,copy
from sys import stdout
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)
import unittest
import numpy as np
import scipy as sp
import math
import functools as fct
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib
import lib.utils.binaryoperations as bb
import tests.HeisED.XXZED as ed
import scipy as sp
import matplotlib.pyplot as plt
import lib.Lanczos.LanczosEngine as lanEn
from scipy.sparse import csc_matrix
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

"""
helper function for measuring Sz in the ED case

"""
def measureSz(state,basis,N):
    assert(len(state)==len(basis))
    SZexact=np.zeros((N))
    for m in range(len(basis)):
        b=basis[m]
        for n in range(N):
            spin=bb.getBit(b,n)-0.5
            SZexact[n]+=(np.abs(state[m])**2)*spin
    return SZexact

def measureSy(state,basis,N):
    assert(len(state)==len(basis))
    SYexact=np.zeros((N))
    for m in range(len(basis)):
        b=basis[m]
        for n in range(N):
            other=bb.flipBit(b,n)
            if np.nonzero(basis==other)[0].size>0:
                amp=state[np.nonzero(basis==other)]
                if bb.getBit(b,n)==1:
                    SYexact[n]+=amp*(-1j)*state[m]
                elif bb.getBit(b,n)==0:
                    SYexact[n]+=amp*(1j)*state[m]
    return SYexact

def measureSx(state,basis,N):
    assert(len(state)==len(basis))
    SXexact=np.zeros((N))
    for m in range(len(basis)):
        b=basis[m]
        for n in range(N):
            other=bb.flipBit(b,n)
            if np.nonzero(basis==other)[0].size>0:
                amp=state[np.nonzero(basis==other)]
                SXexact[n]+=amp*state[m]
    return SXexact

"""
runs tests fon TEBD and TDVP: the method does the following computations:

using exact diagonalization, calculate ground-state for a N=10 sites Heisenberg model, apply S+ to site n=5, and evolve 
the state forward;
using dmrg, calculate ground-state for a N=10 sites Heisenberg model, apply S+ to site n=5
evolve the state using TDVP and TEBD; compare the results with ED

"""
class TestTimeEvolution(unittest.TestCase):        
    def setUp(self):
        D=32
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0
        self.Nmax=10
        self.dt=-1j*0.05  #time step
        self.numsteps=10  #numnber of steps to be taken in between measurements        
        Jz=np.ones(self.N)*jz
        Jxy=np.ones(self.N)*jxy

        N=self.N #system size
        Nup=int(self.N/2) #number of up-spins
        Z2=-10
        Jzar=[0.0]*N
        Jxyar=[0.0]*N
        grid=[None]*N
        for n in range(N-1):
            grid[n]=[n+1]
            Jzar[n]=np.asarray([jz])
            Jxyar[n]=np.asarray([jxy])
            
        grid[N-1]=[]
        Jzar[N-1]=np.asarray([0.0])
        Jxyar[N-1]=np.asarray([0.0])
        np.asarray(Jxyar)

        Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
        Hsparse1=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid) #get the ground state
        e,v=sp.sparse.linalg.eigsh(Hsparse1,k=1,which='SA',maxiter=1000000,tol=1E-8,v0=None,ncv=40)
        #the Hamiltonian in the new sector
        Hsparse2=ed.XXZSparseHam(Jxyar,Jzar,N,Nup+1,Z2,grid)
        
        #flip the spin at the center of the ground state

        basis1=ed.binarybasisU1(self.N,self.N/2)
        assert(len(basis1)==Hsparse1.shape[0])
        
        basis2=ed.binarybasisU1(self.N,self.N/2+1)
        vnew=np.zeros(len(basis2),v.dtype)                
        assert(len(basis2)==Hsparse2.shape[0])        
        for n in range(len(basis1)):
            state1=basis1[n]
            splus=bb.setBit(state1,(N-1)/2)
            for m in range(len(basis2)):            
                if splus==basis2[m]:
                    vnew[m]=v[n]


        def matvec(mat,vec):
            return mat.dot(vec)
        mv=fct.partial(matvec,*[Hsparse2])
        
        lan=lanEn.LanczosTimeEvolution(mv,np.dot,np.zeros,ncv=20,delta=1E-8)
        self.Szexact=np.zeros((self.Nmax,self.N))
        self.Syexact=np.zeros((self.Nmax,self.N))        
        vnew/=np.linalg.norm(vnew)
        for n in range(self.Nmax):
            self.Szexact[n,:]=measureSz(vnew,basis2,self.N)[::-1]
            self.Syexact[n,:]=measureSy(vnew,basis2,self.N)[::-1]            
            stdout.write("\rED time evolution at t/T= %4.4f/%4.4f"%(n*self.numsteps*np.abs(self.dt),self.Nmax*self.numsteps*np.abs(self.dt)))
            stdout.flush()
            for step in range(self.numsteps):            
                vnew=lan.__doStep__(vnew,self.dt)
                vnew/=np.linalg.norm(vnew)

        mps=mpslib.MPS.random(self.N,10,d,obc=True)
        mps._D=D
        mps.__position__(self.N)
        mps.__position__(0)
        self.mpo=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        self.dmrg=en.DMRGengine(mps,self.mpo,'testXXZ',lb,rb)
        self.eps=1E-5

    def testTEBD(self):
        N=self.N
        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
        self.dmrg._mps.__position__(self.N)
        self.dmrg._mps.__position__(0)
        self.dmrg._mps.resetZ() #don't forget to normalize the state again after application of the gate        
        engine=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"insert_name_here")

        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
        #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold

        SZ=np.zeros((self.Nmax,N)) #container for holding the measurements

        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        for n in range(self.Nmax):
            #measure the operators 
            #L=[engine._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
            L=engine.measureLocal(sz)            
            #store result for later use
            SZ[n,:]=L
            
            tw,it2=engine.doTEBD(dt=self.dt,numsteps=self.numsteps,Dmax=Dmax,tr_thresh=thresh,\
                                     cnterset=it2,tw=tw)
        print("                : ",np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N))
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-5)        
        
    def testTDVP_LAN(self):
        N=self.N
        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
        self.dmrg._mps.__position__(self.N)
        self.dmrg._mps.__position__(0)
        self.dmrg._mps.resetZ() #don't forget to normalize the state again after application of the gate
        
        engine=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"TDVP_insert_name_here")        
        
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
        #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold

        SZ=np.zeros((self.Nmax,N)) #container for holding the measurements

        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        solver='LAN'        
        for n in range(self.Nmax):
            #measure the operators
            L=engine.measureLocal(sz)
            #store result for later use
            SZ[n,:]=L 
            it1=engine.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
        print("                : ",np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N))            
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-8)
        
    def testTDVP_RK45(self):
        N=self.N
        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
        self.dmrg._mps.__position__(self.N)
        self.dmrg._mps.__position__(0)
        self.dmrg._mps.resetZ() #don't forget to normalize the state again after application of the gate
        
        engine=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"TDVP_insert_name_here")        
        
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
        #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold

        SZ=np.zeros((self.Nmax,N)) #container for holding the measurements

        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        solver='RK45'        
        for n in range(self.Nmax):
            #measure the operators
            L=engine.measureLocal(sz)
            #store result for later use
            SZ[n,:]=L 
            it1=engine.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
        print("                : ",np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N))            
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-8)
        
    def testTDVP_SEXPMV(self):
        N=self.N
        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
        self.dmrg._mps.__position__(self.N)
        self.dmrg._mps.__position__(0)
        self.dmrg._mps.resetZ() #don't forget to normalize the state again after application of the gate
        
        engine=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"TDVP_insert_name_here")        
        
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
        #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold

        SZ=np.zeros((self.Nmax,N)) #container for holding the measurements

        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        solver='SEXPMV'        
        for n in range(self.Nmax):
            #measure the operators
            L=engine.measureLocal(sz)
            #store result for later use
            SZ[n,:]=L 
            it1=engine.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
        print("                : ",np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N))            
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-8)            


"""
runs tests fon TEBD and TDVP: the method does the following computations:

using exact diagonalization, calculate ground-state for a N=10 sites Heisenberg model, apply S+ to site n=5, and evolve 
the state forward;
using dmrg, calculate ground-state for a N=10 sites Heisenberg model, apply S+ to site n=5
evolve the state using TDVP and TEBD; compare the results with ED
plots the results 

"""
        
class TestPlot(unittest.TestCase):        
    def setUp(self):
        D=32
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0
        self.Nmax=10
        self.dt=-1j*0.05  #time step
        self.numsteps=10  #numnber of steps to be taken in between measurements        
        Jz=np.ones(self.N)*jz
        Jxy=np.ones(self.N)*jxy

        N=self.N #system size
        Nup=int(self.N/2) #number of up-spins
        Z2=-10
        Jzar=[0.0]*N
        Jxyar=[0.0]*N
        grid=[None]*N
        for n in range(N-1):
            grid[n]=[n+1]
            Jzar[n]=np.asarray([jz])
            Jxyar[n]=np.asarray([jxy])
            
        grid[N-1]=[]
        Jzar[N-1]=np.asarray([0.0])
        Jxyar[N-1]=np.asarray([0.0])
        np.asarray(Jxyar)

        Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
        Hsparse1=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid) #get the ground state
        e,v=sp.sparse.linalg.eigsh(Hsparse1,k=1,which='SA',maxiter=1000000,tol=1E-8,v0=None,ncv=40)
        #the Hamiltonian in the new sector
        Hsparse2=ed.XXZSparseHam(Jxyar,Jzar,N,Nup+1,Z2,grid)
        
        #flip the spin at the center of the ground state

        basis1=ed.binarybasisU1(self.N,self.N/2)
        assert(len(basis1)==Hsparse1.shape[0])
        
        basis2=ed.binarybasisU1(self.N,self.N/2+1)
        vnew=np.zeros(len(basis2),v.dtype)                
        assert(len(basis2)==Hsparse2.shape[0])        
        for n in range(len(basis1)):
            state1=basis1[n]
            splus=bb.setBit(state1,(N-1)/2)
            #splus=bb.setBit(state1,1)            
            for m in range(len(basis2)):            
                if splus==basis2[m]:
                    vnew[m]=v[n]


        def matvec(mat,vec):
            return mat.dot(vec)
        mv=fct.partial(matvec,*[Hsparse2])
        
        lan=lanEn.LanczosTimeEvolution(mv,np.dot,np.zeros,ncv=20,delta=1E-8)
        self.Szexact=np.zeros((self.Nmax,self.N))
        self.Syexact=np.zeros((self.Nmax,self.N))
        self.Sxexact=np.zeros((self.Nmax,self.N))                
        vnew/=np.linalg.norm(vnew)
        for n in range(self.Nmax):
            self.Szexact[n,:]=measureSz(vnew,basis2,self.N)[::-1]
            self.Syexact[n,:]=measureSy(vnew,basis2,self.N)[::-1]
            self.Sxexact[n,:]=measureSx(vnew,basis2,self.N)[::-1]                                    
            stdout.write("\rED time evolution at t/T= %4.4f/%4.4f"%(n*self.numsteps*np.abs(self.dt),self.Nmax*self.numsteps*np.abs(self.dt)))
            stdout.flush()
            for step in range(self.numsteps):            
                vnew=lan.__doStep__(vnew,self.dt)
                vnew/=np.linalg.norm(vnew)

        mps=mpslib.MPS.random(self.N,10,d,obc=True,dtype=complex,schmidt_thresh=1E-16)
        mps._D=D
        mps.__position__(self.N)
        mps.__position__(0)
        self.mpo=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))

        self.dmrg=en.DMRGengine(mps,self.mpo,'testXXZ',lb,rb)
        self.eps=1E-5

        
    def test_plot_LAN(self):
        
        def run_sim(dmrgcontainer,solver):
            dmrgcontainer._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
            dmrgcontainer._mps.__position__(self.N)
            dmrgcontainer._mps.__position__(0)

            engine1=en.TimeEvolutionEngine(dmrgcontainer._mps,self.mpo,"insert_name_here")
            engine2=en.TimeEvolutionEngine(dmrgcontainer._mps.__copy__(),self.mpo,"TDVP_insert_name_here")        
            engine1._mps.resetZ()
            engine2._mps.resetZ()        
            
            Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
            #adapted to this value in the TimeEvolutionEngine
            thresh=1E-16  #truncation threshold
            
            SZ1=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements
            SZ2=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements
            SX1=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements
            SX2=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements
            SY1=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements
            SY2=np.zeros((self.Nmax,N)).astype(complex) #container for holding the measurements        
            
            plt.ion()
            sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
            sy=[np.asarray([[0,-0.5j],[0.5j,0]]) for n in range(N)]  #a list of local operators to be measured
            sx=[np.asarray([[0,1.0],[1.0,0]]) for n in range(N)]  #a list of local operators to be measured                    
            it1=0  #counts the total iteration number
            it2=0  #counts the total iteration number
            tw=0  #accumulates the truncated weight (see below)

            for n in range(self.Nmax):
                #measure a list of local operators
                #L=[engine1._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
                SZ1[n,:],SY1[n,:],SX1[n,:]=engine1.measureLocal(sz),engine1.measureLocal(sy),engine1.measureLocal(sx)
                #store result for later use
            
                #note: when measuring with measureLocal function, one has to update the simulation container after that.
                #this  is because measureLocal shifts the center site of the mps around, and that causes inconsistencies
                #in the simulation container with the left and right environments
                #L=[engine2._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
                #engine2.update()
                SZ2[n,:],SY2[n,:],SX2[n,:]=engine2.measureLocal(sz),engine2.measureLocal(sy),engine2.measureLocal(sx)
                tw,it2=engine1.doTEBD(dt=self.dt,numsteps=self.numsteps,Dmax=Dmax,tr_thresh=thresh,\
                                         cnterset=it2,tw=tw)
            
                it1=engine2.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
            
                plt.figure(1,figsize=(10,8))
                plt.clf()
                plt.subplot(3,3,1)
                plt.plot(range(self.N),self.Szexact[n,:],range(self.N),np.real(SZ2[n,:]),'rd',range(self.N),np.real(SZ1[n,:]),'ko',Markersize=5)
                plt.ylim([-0.5,0.5])
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\langle S^z_n\rangle$')
                plt.legend(['exact','TDVP ('+solver+')','TEBD'])
                plt.subplot(3,3,4)
                plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-np.real(SZ2[n,:])))
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^z_n\rangle_{tdvp}-\langle S^z_n\rangle_{exact}|$')            
                plt.subplot(3,3,7)
                plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-np.real(SZ1[n,:])))            
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^z_n\rangle_{tebd}-\langle S^z_n\rangle_{exact}|$')
                plt.tight_layout()
            
                plt.subplot(3,3,2)
                plt.plot(range(self.N),self.Syexact[n,:],range(self.N),np.real(SY2[n,:]),'rd',range(self.N),np.real(SY1[n,:]),'ko',Markersize=5)
                plt.ylim([-0.5,0.5])
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\langle S^y_n\rangle$')
                plt.legend(['exact','TDVP ('+solver+')','TEBD'])                
                plt.subplot(3,3,5)
                plt.semilogy(range(self.N),np.abs(self.Syexact[n,:]-np.real(SY2[n,:])))
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^y_n\rangle_{tdvp}-\langle S^y_n\rangle_{exact}|$')            
                plt.subplot(3,3,8)
                plt.semilogy(range(self.N),np.abs(self.Syexact[n,:]-np.real(SY1[n,:])))            
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^y_n\rangle_{tebd}-\langle S^y_n\rangle_{exact}|$')
                plt.tight_layout()
            


                plt.subplot(3,3,3)
                plt.plot(range(self.N),self.Sxexact[n,:],range(self.N),np.real(SX2[n,:]),'rd',range(self.N),np.real(SX1[n,:]),'ko',Markersize=5)
                plt.ylim([-0.5,0.5])
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\langle S^x_n\rangle$')
                plt.legend(['exact','TDVP ('+solver+')','TEBD'])                
                plt.subplot(3,3,6)
                plt.semilogy(range(self.N),np.abs(self.Sxexact[n,:]-np.real(SX2[n,:])))
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^x_n\rangle_{tdvp}-\langle S^x_n\rangle_{exact}|$')            
                plt.subplot(3,3,9)
                plt.semilogy(range(self.N),np.abs(self.Sxexact[n,:]-np.real(SX1[n,:])))            
                plt.xlabel('lattice site n')
                plt.ylabel(r'$|\langle S^x_n\rangle_{tebd}-\langle S^x_n\rangle_{exact}|$')
                plt.tight_layout()
            
                plt.figure(2)
                plt.clf()
                plt.subplot(3,1,1)
                plt.plot(range(self.N),np.imag(SZ2[n,:]),'rd',range(self.N),np.imag(SZ1[n,:]),'ko',Markersize=5)
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\Im\langle S^z_n\rangle$')
                plt.legend(['TDVP ('+solver+')','TEBD'])                                
                plt.tight_layout()
            
                plt.subplot(3,1,2)
                plt.plot(range(self.N),np.imag(SY2[n,:]),'rd',range(self.N),np.imag(SY1[n,:]),'ko',Markersize=5)
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\Im\langle S^y_n\rangle$')
                plt.legend(['TDVP ('+solver+')','TEBD'])                                                
                plt.tight_layout()
                
                plt.subplot(3,1,3)
                plt.plot(range(self.N),np.imag(SX2[n,:]),'rd',range(self.N),np.imag(SX1[n,:]),'ko',Markersize=5)
                plt.xlabel('lattice site n')
                plt.ylabel(r'$\Im\langle S^x_n\rangle$')
                plt.legend(['TDVP ('+solver+')','TEBD'])                                                
                plt.tight_layout()
                
                
                plt.draw()
                plt.show()
                plt.pause(0.01)
                #input()

        N=self.N
        
        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')
                
        for solver in ['LAN','RK45','SEXPMV','RK23']:
            run_sim(copy.deepcopy(self.dmrg),solver)
            
#    def test_plot_SEXPMV(self):
#        
#        N=self.N
#        
#        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
#        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')
#
#        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
#        self.dmrg._mps.__position__(self.N)
#        self.dmrg._mps.__position__(0)
#
#        engine1=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"insert_name_here")
#        engine2=en.TimeEvolutionEngine(self.dmrg._mps.__copy__(),self.mpo,"TDVP_insert_name_here")        
#        
#        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
#        #adapted to this value in the TimeEvolutionEngine
#        thresh=1E-16  #truncation threshold
#
#        SZ1=np.zeros((self.Nmax,N)) #container for holding the measurements
#        SZ2=np.zeros((self.Nmax,N)) #container for holding the measurements        
#        plt.ion()
#        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
#        it1=0  #counts the total iteration number
#        it2=0  #counts the total iteration number
#        tw=0  #accumulates the truncated weight (see below)
#        solver='LAN'
#        engine1._mps.resetZ()
#        engine2._mps.resetZ()        
#        for n in range(self.Nmax):
#            #measure a list of local operators
#            #L=[engine1._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
#            L=engine1.measureLocal(sz)
#            #store result for later use
#            SZ1[n,:]=L
#            #note: when measuring with measureLocal function, one has to update the simulation container after that.
#            #this  is because measureLocal shifts the center site of the mps around, and that causes inconsistencies
#            #in the simulation container with the left and right environments
#            #L=[engine2._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
#            #engine2.update()
#            L=engine2.measureLocal(sz)
#            SZ2[n,:]=L
#            tw,it2=engine1.doTEBD(dt=self.dt,numsteps=self.numsteps,Dmax=Dmax,tr_thresh=thresh,\
#                                     cnterset=it2,tw=tw)
#
#            it1=engine2.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
#
#            plt.figure(1,figsize=(10,8))
#            plt.clf()
#            plt.subplot(3,1,1)
#            plt.plot(range(self.N),self.Szexact[n,:],range(self.N),SZ2[n,:],'rd',range(self.N),SZ1[n,:],'ko',Markersize=5)
#            plt.ylim([-0.5,0.5])
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$\langle S^z_n\rangle$')
#            plt.legend(['exact','TDVP (SEXPMV)','TEBD'])
#            plt.subplot(3,1,2)
#            plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-SZ2[n,:]))
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$|\langle S^z_n\rangle_{tdvp}-\langle S^z_n\rangle_{exact}|$')            
#            plt.subplot(3,1,3)
#            plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-SZ1[n,:]))            
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$|\langle S^z_n\rangle_{tebd}-\langle S^z_n\rangle_{exact}|$')
#            plt.tight_layout()
#            plt.draw()
#            plt.show()
#            plt.pause(0.01)
#
#            
#
#
#    def test_plot_RK45(self):
#        
#        N=self.N
#        
#        self.dmrg.__simulateTwoSite__(4,1e-10,1e-6,40,verbose=1,solver='LAN')    
#        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')
#
#        self.dmrg._mps.__applyOneSiteGate__(np.asarray([[0.0,1.],[0.0,0.0]]),int(self.N/2))
#        self.dmrg._mps.__position__(self.N)
#        self.dmrg._mps.__position__(0)
#
#        engine1=en.TimeEvolutionEngine(self.dmrg._mps,self.mpo,"insert_name_here")
#        engine2=en.TimeEvolutionEngine(self.dmrg._mps.__copy__(),self.mpo,"TDVP_insert_name_here")        
#        
#        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
#        #adapted to this value in the TEBDEngine
#        thresh=1E-16  #truncation threshold
#
#        SZ1=np.zeros((self.Nmax,N)) #container for holding the measurements
#        SZ2=np.zeros((self.Nmax,N)) #container for holding the measurements        
#        plt.ion()
#        sz=[np.diag([-0.5,0.5]) for n in range(N)]  #a list of local operators to be measured
#        it1=0  #counts the total iteration number
#        it2=0  #counts the total iteration number
#        tw=0  #accumulates the truncated weight (see below)
#        solver='RK45'
#        engine2._mps.resetZ()
#        engine1._mps.resetZ()        
#        for n in range(self.Nmax):
#            #measure a list of local operators
#            #L=[engine1._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
#            L=engine1.measureLocal(sz)            
#            #store result for later use
#            SZ1[n,:]=L
#            #note: when measuring with measureLocal function, one has to update the simulation container after that.
#            #this  is because measureLocal shifts the center site of the mps around, and that causes inconsistencies
#            #in the simulation container with the left and right environments
#            #L=[engine2._mps.measureLocal(np.diag([-0.5,0.5]),site=n).real for n in range(N)]
#            #engine2.update()
#            L=engine2.measureLocal(sz)                        
#            SZ2[n,:]=L
#            tw,it2=engine1.doTEBD(dt=self.dt,numsteps=self.numsteps,Dmax=Dmax,tr_thresh=thresh,\
#                                     cnterset=it2,tw=tw)
#
#            it1=engine2.doTDVP(self.dt,numsteps=self.numsteps,krylov_dim=10,cnterset=it1,solver=solver)
#
#            plt.figure(1,figsize=(10,8))
#            plt.clf()
#            plt.subplot(3,1,1)
#            plt.plot(range(self.N),self.Szexact[n,:],range(self.N),SZ2[n,:],'rd',range(self.N),SZ1[n,:],'ko',Markersize=5)
#            plt.ylim([-0.5,0.5])
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$\langle S^z_n\rangle$')
#            plt.legend(['exact','TDVP (RK45)','TEBD'])
#            plt.subplot(3,1,2)
#            plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-SZ2[n,:]))
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$|\langle S^z_n\rangle_{tdvp}-\langle S^z_n\rangle_{exact}|$')            
#            plt.subplot(3,1,3)
#            plt.semilogy(range(self.N),np.abs(self.Szexact[n,:]-SZ1[n,:]))            
#            plt.xlabel('lattice site n')
#            plt.ylabel(r'$|\langle S^z_n\rangle_{tebd}-\langle S^z_n\rangle_{exact}|$')
#            plt.tight_layout()
#            plt.draw()
#            plt.show()
#            plt.pause(0.01)
#


if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTimeEvolution)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestPlot)    
    #unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)     

