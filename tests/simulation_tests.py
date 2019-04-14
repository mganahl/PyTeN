#!/usr/bin/env python
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)
import unittest
import numpy as np
import scipy as sp
import math
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.Container as CO
import lib.mpslib.TensorNetwork as TN
import lib.mpslib.MPO as MPO
import lib.mpslib.SimContainer as SCT
from lib.mpslib.Tensor import Tensor
import lib.Lanczos.LanczosEngine as LZ
import lib.ncon as ncon
import lib.HeisED.XXZED as ed
import lib.utils.binaryoperations as bb
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import functools as fct
import pickle
from sys import stdout
from scipy.sparse import csc_matrix

from scipy.sparse import csc_matrix
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

plot = True
class TestFiniteDMRGXXZ(unittest.TestCase):        
    def setUp(self):
        self.D=60
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0
        self.Jz=np.ones(self.N)*jz
        self.Jxy=np.ones(self.N)*jxy

        N=self.N #system size
        Nup=int(self.N/2) #number of up-spins
        Z2=-1
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
        Hsparse=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid)
        e,v=sp.sparse.linalg.eigsh(Hsparse,k=1,which='SA',maxiter=1000000,tol=1E-8,v0=None,ncv=40)
        self.Eexact=e[0]
        self.eps = 1E-5


    def test_one_site_dmrg_float_ar(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2,precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_one_site_dmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2,precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)



    def test_one_site_dmrg_float_lan(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_one_site_dmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)



    def test_two_site_dmrg_float_ar(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_two_site_dmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_two_site_dmrg_float_lan(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')

        
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_two_site_dmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

class TestMPSTrunc(unittest.TestCase):        
    def setUp(self):
        self.D=64
        d=2
        self.N=32
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        N=self.N #system size
        self.eps = 1E-5

    def test_mps_trunc_float(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')        
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        Sz=np.diag([0.5,-0.5])
        meanSz = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N))
        meanSzSz = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))
        
        Dt=32
        dmrg.mps.truncate(schmidt_thresh=1E-8,D=Dt)
        meanSztrunc = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N) )  
        meanSzSztrunc = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))

        self.assertTrue(np.linalg.norm(meanSztrunc-meanSz)< 1E-8)
        self.assertTrue(np.linalg.norm(meanSzSztrunc-meanSzSz)< 1E-4)        

        
    def test_mps_trunc_complex(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')                
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        Sz=np.diag([0.5,-0.5])

        meanSz = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N))
        meanSzSz = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))
        
        Dt=20
        dmrg.mps.truncate(schmidt_thresh=1E-8,D=Dt)
        meanSztrunc = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N) )  
        meanSzSztrunc = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))

        self.assertTrue(np.linalg.norm(meanSztrunc-meanSz)< 1E-8)
        self.assertTrue(np.linalg.norm(meanSzSztrunc-meanSzSz)< 1E-4)        

        
class TestInfiniteDMRGXXZ(unittest.TestCase):        
    def setUp(self):
        self.D=60
        d=2
        self.N=2
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        self.Eexact = 0.25 - np.log(2) #exact GS energy of XXX model in thermodynamic limit
        self.eps = 1E-4
        self.Nsweeps=100

        
    def test_one_site_idmrg_float_ar(self):
        dtype = np.float64
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2] * self.N, dtype=dtype, minval=-0.3, maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_one_site_idmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)
        
    def test_one_site_idmrg_float_lan(self):
        dtype = np.float64
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_one_site_idmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


class TestITEBD(unittest.TestCase):        
    def setUp(self):
        self.D=40
        d=2
        self.N=2
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        self.Eexact = 0.25 - np.log(2) #exact GS energy of XXX model in thermodynamic limit
        self.eps = 1E-4
        self.Nsweeps=100
        self.Eexact = 0.25 - np.log(2)
        
    def test_itebd_imag_float(self):
        dtype = np.float64
        imps = TN.MPS.random(D=[self.D]*(self.N + 1),d=[2] * self.N ,minval=-0.5,maxval=0.5)
        impo = MPO.InfiniteXXZ(Jz=np.ones(len(imps)),Jxy = np.ones(len(imps)), Bz=np.zeros(len(imps)), dtype=imps.dtype)
        itebd = SCT.InfiniteTEBDEngine(imps,impo)
        itebd.canonize_mps()
        Nmx = 500
        Sz = np.diag([-0.5, 0.5])
        Sx = np.array([[0,0.5],[0.5,0]])
        Sy = np.array ([[0, 0.5j],[-0.5j, 0]]) 
        from sys import stdout
        dt = 0.01
        numsteps=20
        for n in range(Nmx):
            itebd.do_steps(dt=-dt, numsteps=numsteps, D = self.D,recanonize = None, verbose=0)
            imps.canonize(precision=1E-12)
            energy = imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1])  #the minus accounts for the missing i above
            imps.roll(1)
            energy += imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1]) #the minus accounts for the missing i above           
            
            stdout.write('\r time: %0.4f/%0.4f, energy: %.8f (exact: %0.8f)'%(itebd.time, dt*Nmx*numsteps, np.real(energy[0])/2.0, 0.25 -np.log(2)))
        self.assertTrue(np.abs(energy[0] - self.Eexact) < 1E-4)
        
    def test_itebd_imag_complex(self):
        dtype = np.complex128
        imps = TN.MPS.random(D=[self.D]*(self.N + 1),d=[2] * self.N ,minval=-0.5,maxval=0.5)
        impo = MPO.InfiniteXXZ(Jz=np.ones(len(imps)),Jxy = np.ones(len(imps)), Bz=np.zeros(len(imps)), dtype=imps.dtype)
        itebd = SCT.InfiniteTEBDEngine(imps,impo)
        itebd.canonize_mps()
        Nmx = 500
        Sz = np.diag([-0.5, 0.5])
        Sx = np.array([[0,0.5],[0.5,0]])
        Sy = np.array ([[0, 0.5j],[-0.5j, 0]]) 
        from sys import stdout
        dt = 0.01
        numsteps=20
        for n in range(Nmx):
            itebd.do_steps(dt=-dt, numsteps=numsteps, D = self.D,recanonize = None, verbose=0)
            imps.canonize(precision=1E-12)
            energy = imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1])  
            imps.roll(1)
            energy += imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1]) 
            
            stdout.write('\r time: %0.4f/%0.4f, energy: %.8f (exact: %0.8f)'%(itebd.time, dt*Nmx*numsteps, np.real(energy[0])/2.0, 0.25 -np.log(2)))
        self.assertTrue(np.abs(energy[0] - self.Eexact) < 1E-4)




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

def Sz(b,N):
    sz=0
    for n in range(N):
        spin=bb.getBit(b,n)-0.5
        sz+=spin
    return sz

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
class TimeEvolutionTests(unittest.TestCase):        
    def setUp(self):
        self.D = 32
        D = self.D
        d = 2
        self.N = 10
        jz = 1.0 
        jxy = 1.0
        Bz = 1.0
        self.Nmax = 10  #leave this value, has to match size of loaded test data
        self.dt = -1j*0.05  #time step
        self.numsteps = 10  #numnber of steps to be taken in between measurements
        print()
        print('running time evolution test for total time T = {0}'.format(self.Nmax*self.dt*self.numsteps))
        Jz = np.ones(self.N) * jz
        Jxy = np.ones(self.N) * jxy

        N = self.N #system size
        Nup = int(self.N / 2) #number of up-spins
        Z2 = -10
        Jzar = [0.0] * N
        Jxyar = [0.0] * N
        grid = [None] * N
        for n in range(N - 1):
            grid[n] = [n + 1]
            Jzar[n] = np.asarray([jz])
            Jxyar[n] = np.asarray([jxy])
            
        grid[N - 1]=[]
        Jzar[N - 1]=np.asarray([0.0])
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


        diag=[Sz(b,self.N)*Bz for b in basis2]

        inddiag=list(range(len(basis2)))
        HB=csc_matrix((diag,(inddiag,inddiag)),shape=(len(basis2),len(basis2)))
        Hsparse2+=HB
        
        vnew=np.zeros(len(basis2),v.dtype)
        assert(len(basis2)==Hsparse2.shape[0])
        for n in range(len(basis1)):
            state1=basis1[n]
            splus=bb.setBit(state1,int((N-1)/2))            
            for m in range(len(basis2)):            
                if splus==basis2[m]:
                    vnew[m]=v[n]

        def matvec(mat,vec):
            return mat.dot(vec)
        mv=fct.partial(matvec,*[Hsparse2])
        def scalar_product(a,b):
            return np.dot(np.conj(a),b)
        lan=LZ.LanczosTimeEvolution(mv,scalar_product,ncv=20,delta=1E-8)
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
                vnew=lan.do_step(vnew,self.dt,zeros=np.zeros(vnew.shape,dtype=np.complex128))
                vnew/=np.linalg.norm(vnew)

        mps=TN.FiniteMPS.random(d=self.N*[2],D=[D]*(N-1))
        self.mpo=MPO.FiniteXXZ(Jz,Jxy,np.zeros(self.N))
        mps=TN.FiniteMPS.random(d=[2]*self.N,D=[D]*(N-1),dtype=complex)
        mps.position(self.N)
        mps.position(0)

        self.timeevmpo=MPO.FiniteXXZ(Jz,Jxy,Bz*np.ones(self.N))                
        self.dmrg=SCT.FiniteDMRGEngine(mps,self.mpo,'testXXZ')
        self.eps=1E-5
        
        self.dmrg.run_two_site(Nsweeps=6,
                               precision=1E-10,
                               ncv=40,
                               cp=None,
                               verbose=0,
                               Ndiag=10,
                               landelta=1E-8,
                               landeltaEta=1E-8,
                               solver='LAN')
        
        self.dmrg.mps.position(self.N // 2)
        self.dmrg.mps.apply_1site_gate(np.array([[0.0,0.0], [1.0,0.0]]).view(Tensor),self.N // 2)
        self.dmrg.mps.position(self.N)
        self.dmrg.mps.position(0)
        self.dmrg.mps.normalize()

    def test_expvals(self):
        Szexact_test=np.load('testfiles/Szexact.npy')
        with open('testfiles/XXZ_gates.pickle','rb') as f:
            gates_test=pickle.load(f)
        self.assertTrue(np.linalg.norm(Szexact_test-self.Szexact)<1E-10)
        for s in range(self.N-1):
            self.assertTrue(np.linalg.norm(gates_test[(s,s+1)]-self.timeevmpo.get_2site_gate(s,s+1,self.dt))<1E-10)


class FiniteTEBDTests(TimeEvolutionTests):
    def testTEBD(self):
        
        tebd=SCT.FiniteTEBDEngine(self.dmrg.mps,self.timeevmpo,"TEBD_unittest")
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
        #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold
        SZ=np.zeros((self.Nmax,self.N)) #container for holding the measurements
        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(self.N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        sites=range(self.N)
        for n in range(self.Nmax):
            #measure the operators
            L=tebd.mps.measure_1site_ops(sz,range(len(sz)))
            SZ[n,:]=L
            tw,it2=tebd.do_steps(dt=self.dt,numsteps=self.numsteps,D=Dmax,tr_thresh=thresh, verbose = 0)
            if plot:
                plt.figure(1)
                plt.clf()                
                plt.title('TEBD vs ED')
                plt.plot(sites,L,sites,self.Szexact[n,:],'o')
                plt.ylim([-0.5,0.5])
                plt.xlabel('n',fontsize=20)
                plt.ylabel(r'$\langle S^z_n\rangle$',fontsize=20)
                plt.tight_layout()                
                plt.draw()
                plt.show()
                plt.pause(0.01)
        print()
        print("normalized difference between TEBD and ED:{} ".format(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)))
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-5)        

class FiniteTDVPTests(TimeEvolutionTests):

    def test_1site_FiniteTDVP(self):
        
        tdvp=SCT.FiniteTDVPEngine(self.dmrg.mps,self.timeevmpo,"TDVP_unittest")
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                     #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold
        SZ=np.zeros((self.Nmax,self.N)) #container for holding the measurements
        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(self.N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        sites=range(self.N)
        for n in range(self.Nmax):
            #measure the operators
            L=tdvp.mps.measure_1site_ops(sz,range(len(sz)))
            SZ[n,:]=L
            t = tdvp.run_one_site(dt=self.dt,numsteps=self.numsteps, krylov_dim=10, verbose = 1)
            if plot:
                plt.figure(1)
                plt.clf()
                plt.title('one-site TDVP vs ED')                
                plt.plot(sites,L,sites,self.Szexact[n,:],'o')
                plt.ylim([-0.5,0.5])
                plt.xlabel('n',fontsize=20)
                plt.ylabel(r'$\langle S^z_n\rangle$',fontsize=20)
                plt.tight_layout()                
                plt.draw()
                plt.show()
                
                plt.pause(0.01)

        print()
        print("normalized difference between TEBD and ED:{} ".format(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)))
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-5)        
        
    def test_2site_FiniteTDVP(self):
        
        tdvp=SCT.FiniteTDVPEngine(self.dmrg.mps,self.timeevmpo,"TDVP_unittest")
        Dmax=32      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                     #adapted to this value in the TimeEvolutionEngine
        thresh=1E-16  #truncation threshold
        SZ=np.zeros((self.Nmax,self.N)) #container for holding the measurements
        plt.ion()
        sz=[np.diag([-0.5,0.5]) for n in range(self.N)]  #a list of local operators to be measured
        it1=0  #counts the total iteration number
        it2=0  #counts the total iteration number
        tw=0  #accumulates the truncated weight (see below)
        sites=range(self.N)
        for n in range(self.Nmax):
            #measure the operators
            L=tdvp.mps.measure_1site_ops(sz,range(len(sz)))
            SZ[n,:]=L
            t = tdvp.run_two_site(dt=self.dt,numsteps=self.numsteps, Dmax = self.D, tr_thresh=1E-10,
                                  krylov_dim=10, verbose = 1)
            if plot:
                plt.figure(1)
                plt.clf()                
                plt.title('two-site TDVP vs ED')
                plt.plot(sites,L,sites,self.Szexact[n,:],'o')
                plt.ylim([-0.5,0.5])
                plt.xlabel('n',fontsize=20)
                plt.ylabel(r'$\langle S^z_n\rangle$',fontsize=20)
                plt.tight_layout()
                plt.draw()
                plt.show()
                plt.pause(0.01)
        print()
        print("normalized difference between TEBD and ED:{} ".format(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)))
        self.assertTrue(np.linalg.norm(SZ-self.Szexact)/(self.Nmax*self.N)<1E-5)        




        
if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFiniteDMRGXXZ)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestMPSTrunc)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestInfiniteDMRGXXZ)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestITEBD)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(FiniteTEBDTests)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(FiniteTDVPTests)    

    # unittest.TextTestRunner(verbosity=2).run(suite1)
    # unittest.TextTestRunner(verbosity=2).run(suite2)
    # unittest.TextTestRunner(verbosity=2).run(suite3)
    # unittest.TextTestRunner(verbosity=2).run(suite4)
    unittest.TextTestRunner(verbosity=2).run(suite5)
    unittest.TextTestRunner(verbosity=2).run(suite6)
