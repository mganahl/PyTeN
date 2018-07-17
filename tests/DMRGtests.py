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


class TestSIAM(unittest.TestCase):
    def setUp(self):
        N=11
        dtype=float
        U=np.zeros(N)
        U[0]=0.0
        hop_u=(-1.0)*np.ones(N-1)
        hop_d=(-1.0)*np.ones(N-1)
        mu_u=(-0.1)*np.ones(N)
        mu_d=(-0.8)*np.ones(N)
    
        """ 
        ===============================================================
        diagonalization of free Hamiltonian
        """
        Ham_up=np.diag(mu_u)+np.diag(-hop_u,1)+np.diag(-hop_u,-1)
        eta_up,UNIT_up=np.linalg.eig(Ham_up)
        eta_up[np.abs(eta_up)<1E-12]=0.0
        mF_up=np.nonzero(eta_up<=0.0)[0][-1]
        UNIT_up_=np.copy(UNIT_up[:,0:mF_up+1])
        M_up=UNIT_up_.dot(herm(UNIT_up_))
        
        Ham_down=np.diag(mu_d)+np.diag(-hop_d,1)+np.diag(-hop_d,-1)
        eta_down,UNIT_down=np.linalg.eig(Ham_down)
        eta_down[np.abs(eta_down)<1E-12]=0.0
        mF_down=np.nonzero(eta_down<=0.0)[0][-1]
        UNIT_down_=np.copy(UNIT_down[:,0:mF_down+1])
        M_down=UNIT_down_.dot(herm(UNIT_down_))
        
        self.eexact=np.sum(eta_up[eta_up<0])+np.sum(eta_down[eta_down<0])
        """ 
        ===============================================================
        """
        mpoobc=H.HubbardChain(U,hop_u,hop_d,mu_u,mu_d,obc=True,dtype=dtype)
        chi=10
        d=4
        mps=mpslib.MPS.random(N,chi,d,obc=True,dtype=dtype)
        mps.__position__(N)
        mps.__position__(0)
        mps._D=60
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))

        self.dmrg=en.DMRGengine(mps,mpoobc,'testSIAM',lb,rb)
        self.eps=1E-5
    def testSIAMAR(self):
        edmrg=self.dmrg.__simulateTwoSite__(Nmax=2,Econv=1e-10,tol=1e-6,ncv=40,cp=None,verbose=1,numvecs=1)
        edmrg=self.dmrg.__simulate__(4,1e-10,1e-6,40,cp=10,verbose=1,numvecs=1)
        self.assertTrue(abs(self.eexact-edmrg)/np.abs(edmrg)<self.eps)
    def testSIAMLAN(self):
        edmrg=self.dmrg.__simulateTwoSite__(Nmax=2,Econv=1e-10,tol=1e-6,ncv=40,cp=None,verbose=1,numvecs=1,solver='LAN')
        edmrg=self.dmrg.__simulate__(4,1e-10,1e-6,40,cp=10,verbose=1,numvecs=1,solver='LAN')
        self.assertTrue(abs(self.eexact-edmrg)/np.abs(edmrg)<self.eps)

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
        mps.__position__(self.N)
        mps.__position__(0)
        mpoobc=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        self.dmrg=en.DMRGengine(mps,mpoobc,'testXXZ',lb,rb)
        self.eps=1E-5

    def testXXZAR(self):
        self.dmrg.__simulateTwoSite__(2,1e-10,1e-6,40,verbose=1)    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1)

        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N)
        meanSz=np.zeros(self.N)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.__truncate__(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N)
        meanSztrunc=np.zeros(self.N)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))

    def testXXZLAN(self):
        self.dmrg.__simulateTwoSite__(2,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N)
        meanSz=np.zeros(self.N)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.__truncate__(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N)
        meanSztrunc=np.zeros(self.N)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))

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
        mps.__position__(self.N)
        mps.__position__(0)
        mpoobc=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        self.dmrg=en.DMRGengine(mps,mpoobc,'testXXZ',lb,rb)
        self.eps=1E-5

    def testXXZAR(self):
        self.dmrg.__simulateTwoSite__(2,1e-10,1e-6,40,verbose=1)    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1)
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N).astype(complex)
        meanSz=np.zeros(self.N).astype(complex)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.__truncate__(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N).astype(complex)
        meanSztrunc=np.zeros(self.N).astype(complex)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))

    def testXXZLAN(self):
        self.dmrg.__simulateTwoSite__(2,1e-10,1e-6,40,verbose=1,solver='LAN')    
        edmrg=self.dmrg.__simulate__(2,1e-10,1e-10,30,verbose=1,solver='LAN')

        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

        Sz=np.diag([0.5,-0.5])

        meanSzSz=np.zeros(self.N).astype(complex)
        meanSz=np.zeros(self.N).astype(complex)
  
        for n in range(self.N):
            meanSz[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSz[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))
        
        Dt=20
        self.dmrg._mps.__truncate__(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        meanSzSztrunc=np.zeros(self.N).astype(complex)
        meanSztrunc=np.zeros(self.N).astype(complex)
        for n in range(self.N):
            meanSztrunc[n]=self.dmrg._mps.__measure__(Sz,n)
        for n in range(self.N):
            meanSzSztrunc[n]=self.dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n]))

        print(np.linalg.norm(meanSztrunc-meanSz))




if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSIAM)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestXXZFloat)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestXXZComplex)
    unittest.TextTestRunner(verbosity=2).run(suite1) 
    unittest.TextTestRunner(verbosity=2).run(suite2) 
    unittest.TextTestRunner(verbosity=2).run(suite3) 
