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
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib
import tests.HeisED.XXZED as ed
import scipy as sp
import lib.Lanczos.LanczosEngine as lanEn
from scipy.sparse import csc_matrix
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


class TestXXZFloat(unittest.TestCase):        
    def setUp(self):
        D=60
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0

        Jz=np.ones(self.N)*jz
        Jxy=np.ones(self.N)*jxy

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

        mps=mpslib.MPS.random(self.N,10,d,obc=True)
        mps._D=D
        mps.position(self.N)
        mps.position(0)
        mpoobc=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        self.dmrg=en.DMRGengine(mps,mpoobc,'testXXZ',lb,rb)
        self.eps=1E-5

    def testXXZAR(self):
        self.dmrg.simulateTwoSite(2,1e-10,1e-6,40,verbose=1)    
        edmrg=self.dmrg.simulate(2,1e-10,1e-10,30,verbose=1)

        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N)
        meanSz=np.zeros(self.N)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.truncate(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N)
        meanSztrunc=np.zeros(self.N)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))

    def testXXZLAN(self):
        self.dmrg.simulateTwoSite(2,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.simulate(2,1e-10,1e-10,30,verbose=1,solver='LAN')
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N)
        meanSz=np.zeros(self.N)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.truncate(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N)
        meanSztrunc=np.zeros(self.N)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))


class TestXXZComplex(unittest.TestCase):        
    def setUp(self):
        D=60
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0
        Jz=np.ones(self.N)*jz
        Jxy=np.ones(self.N)*jxy
        


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
        Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
        Hsparse=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid)
        e,v=sp.sparse.linalg.eigsh(Hsparse,k=1,which='SA',maxiter=1000000,tol=1E-8,v0=None,ncv=40)
        self.Eexact=e[0]

        mps=mpslib.MPS.random(self.N,10,d,obc=True,dtype=complex)
        mps._D=D
        mps.position(self.N)
        mps.position(0)
        mpoobc=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        self.dmrg=en.DMRGengine(mps,mpoobc,'testXXZ',lb,rb)
        self.eps=1E-5

    def testXXZAR(self):
        self.dmrg.simulateTwoSite(2,1e-10,1e-6,40,verbose=1)    
        edmrg=self.dmrg.simulate(2,1e-10,1e-10,30,verbose=1)
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N).astype(complex)
        meanSz=np.zeros(self.N).astype(complex)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.truncate(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N).astype(complex)
        meanSztrunc=np.zeros(self.N).astype(complex)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))

    def testXXZLAN(self):
        self.dmrg.simulateTwoSite(2,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.simulate(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N).astype(complex)
        meanSz=np.zeros(self.N).astype(complex)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.truncate(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N).astype(complex)
        meanSztrunc=np.zeros(self.N).astype(complex)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.measure(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.measure([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))




if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSIAM)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestXXZFloat)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestXXZComplex)
    unittest.TextTestRunner(verbosity=2).run(suite1) 
    unittest.TextTestRunner(verbosity=2).run(suite2) 
    unittest.TextTestRunner(verbosity=2).run(suite3) 
