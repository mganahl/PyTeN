#!/usr/bin/env python

#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code

import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import unittest
import numpy as np
import scipy as sp
import os,random
import math
import datetime as dt
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import lib.mpslib.mps as mpslib
import lib.ncon as ncon
import lib.mpslib.engines as en
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.Hamiltonians as H
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

plt.ion()
class TestTMeigs(unittest.TestCase):
    def setUp(self):
        #initialize a CMPS by loading it from a file 
        self.D=64
        self.d=2
        #self.dtype='complex128'
        self.eps=1E-10

    def test_TMeigs_left_complex(self):
        tensor=(np.random.rand(self.D,self.D,self.d)-0.5)+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        eta,v,numeig=mf.TMeigs(tensor,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        out1=mf.GeneralizedMatrixVectorProduct(1,tensor,tensor,v)
        l=np.reshape(v,(self.D,self.D))
        out2=mf.GeneralizedMatrixVectorProduct(1,tensor,tensor,np.reshape(l,(self.D**2)))
        self.assertTrue(np.linalg.norm(out1-eta*v)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(out2-eta*v)/self.D**2<self.eps)

        A,r,Z=mf.prepareTensor(tensor,1)
        eta,v,numeig=mf.TMeigs(A,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        self.assertTrue(np.linalg.norm(l-np.eye(self.D)/self.D)/self.D**2<self.eps)

    def test_TMeigs_left_real(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        eta,v,numeig=mf.TMeigs(tensor,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        out1=mf.GeneralizedMatrixVectorProduct(1,tensor,tensor,v)
        l=np.reshape(v,(self.D,self.D))
        out2=mf.GeneralizedMatrixVectorProduct(1,tensor,tensor,np.reshape(l,(self.D**2)))
        self.assertTrue(np.linalg.norm(out1-eta*v)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(out2-eta*v)/self.D**2<self.eps)

        A,r,Z=mf.prepareTensor(tensor,1)
        eta,v,numeig=mf.TMeigs(A,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        self.assertTrue(np.linalg.norm(l-np.eye(self.D)/self.D)/self.D**2<self.eps)

    def test_TMeigs_right_complex(self):
        tensor=(np.random.rand(self.D,self.D,self.d)-0.5)+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        dtype='complex128'
        eta,v,numeig=mf.TMeigs(tensor,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        out1=mf.GeneralizedMatrixVectorProduct(-1,tensor,tensor,v)
        l=np.reshape(v,(self.D,self.D))
        out2=mf.GeneralizedMatrixVectorProduct(-1,tensor,tensor,np.reshape(l,(self.D**2)))
        self.assertTrue(np.linalg.norm(out1-eta*v)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(out2-eta*v)/self.D**2<self.eps)
        B,r,Z=mf.prepareTensor(tensor,-1)
        eta,v,numeig=mf.TMeigs(B,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=r/np.trace(r)
        self.assertTrue(np.linalg.norm(r-np.eye(self.D)/self.D)/self.D**2<self.eps)


    def test_TMeigs_right_real(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        dtype='float64'
        eta,v,numeig=mf.TMeigs(tensor,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        out1=mf.GeneralizedMatrixVectorProduct(-1,tensor,tensor,v)
        l=np.reshape(v,(self.D,self.D))
        out2=mf.GeneralizedMatrixVectorProduct(-1,tensor,tensor,np.reshape(l,(self.D**2)))
        self.assertTrue(np.linalg.norm(out1-eta*v)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(out2-eta*v)/self.D**2<self.eps)
        B,r,Z=mf.prepareTensor(tensor,-1)
        eta,v,numeig=mf.TMeigs(B,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=r/np.trace(r)
        self.assertTrue(np.linalg.norm(r-np.eye(self.D)/self.D)/self.D**2<self.eps)


class TestRegauge(unittest.TestCase):
    def setUp(self):
        #initialize a CMPS by loading it from a file 
        self.D=64
        self.d=2
        #self.dtype='complex128'
        
        self.eps=1E-10

    def test_regauge_left_complex(self):
        tensor=(np.random.rand(self.D,self.D,self.d)-0.5)+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        dtype='complex128'
        A,x=mf.regauge(tensor,gauge='left',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14)
        self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(self.D))/self.D**2<self.eps)

    def test_regauge_left_real(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        dtype='float64'
        A,x=mf.regauge(tensor,gauge='left',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14)
        self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(self.D))/self.D**2<self.eps)

    def test_regauge_right_complex(self):
        tensor=(np.random.rand(self.D,self.D,self.d)-0.5)+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        dtype='complex128'
        A,x=mf.regauge(tensor,gauge='right',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14)        
        self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([1,2],[1,2]))-np.eye(self.D))/self.D**2<self.eps)


    def test_regauge_right_real(self):
        tensor=(np.random.rand(self.D,self.D,self.d)-0.5)
        dtype='float64'
        A,x=mf.regauge(tensor,gauge='right',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14)        
        self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([1,2],[1,2]))-np.eye(self.D))/self.D**2<self.eps)


class TestRENORMBLOCKHAMGMRES(unittest.TestCase):
    def setUp(self):
        self.D=4
        self.d=2
        self.eps=1E-10
        N=2
        Delta=1.0
        Jz=Delta*np.ones(N-1)
        Jxy=1.0*np.ones(N-1)
        B=np.zeros(N)
        self.mpo=H.XXZ(Jz,Jxy,B,obc=True)


    def test_RENORMBLOCKHAMGMRES_left_real_vs_complex(self):
        np.random.seed(10)
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        A,rest,Z=mf.prepareTensor(tensor,1)
        dtype='float64'
        eta,v,numeig=mf.TMeigs(A,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=np.real(r/np.trace(r))
        L=mf.initializeLayer(A,np.eye(self.D),A,self.mpo[0],1)
        L=np.reshape(mf.addLayer(L,A,self.mpo[1],A,1),(self.D,self.D))
        h=np.tensordot(L,r,([0,1],[0,1]))
        inhom=np.reshape(L-h*np.eye(self.D),self.D**2)
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        kreal=mf.RENORMBLOCKHAMGMRES(A,A,r,np.eye(self.D),inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=1)

        dtype='complex128'
        eta,v,numeig=mf.TMeigs(A,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=r/np.trace(r)
        L=mf.initializeLayer(A,np.eye(self.D),A,self.mpo[0],1)
        L=np.reshape(mf.addLayer(L,A,self.mpo[1],A,1),(self.D,self.D))
        h=np.tensordot(L,r,([0,1],[0,1]))
        inhom=np.reshape(L-h*np.eye(self.D),self.D**2)
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        kcomplex=mf.RENORMBLOCKHAMGMRES(A,A,r,np.eye(self.D),inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=1)
        self.assertTrue(np.linalg.norm(kreal-kcomplex)/self.D**2<self.eps)




    def test_RENORMBLOCKHAMGMRES_right_real_vs_complex(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        B,rest,Z=mf.prepareTensor(tensor,-1)
        dtype='float64'
        eta,v,numeig=mf.TMeigs(B,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        R=mf.initializeLayer(B,np.eye(self.D),B,self.mpo[1],-1)
        R=np.reshape(mf.addLayer(R,B,self.mpo[0],B,-1),(self.D,self.D))
        h=np.tensordot(l,R,([0,1],[0,1]))
        inhom=np.reshape(R-h*np.eye(self.D),self.D**2)
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        kreal=mf.RENORMBLOCKHAMGMRES(B,B,np.eye(self.D),l,inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=-1)

        dtype='complex128'
        eta,v,numeig=mf.TMeigs(B,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        R=mf.initializeLayer(B,np.eye(self.D),B,self.mpo[1],-1)
        R=np.reshape(mf.addLayer(R,B,self.mpo[0],B,-1),(self.D,self.D))
        h=np.tensordot(l,R,([0,1],[0,1]))
        inhom=np.reshape(R-h*np.eye(self.D),self.D**2)
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        kcomplex=mf.RENORMBLOCKHAMGMRES(B,B,np.eye(self.D),l,inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=-1)

        self.assertTrue(np.linalg.norm(kreal-kcomplex)/self.D**2<self.eps)




    def test_RENORMBLOCKHAMGMRES_left_real(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        A,rest,Z=mf.prepareTensor(tensor,1)
        dtype='float64'
        eta,v,numeig=mf.TMeigs(A,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=r/np.trace(r)
        L=mf.initializeLayer(A,np.eye(self.D),A,self.mpo[0],1)
        L=np.reshape(mf.addLayer(L,A,self.mpo[1],A,1),(self.D,self.D))
        h=np.tensordot(L,r,([0,1],[0,1]))

        inhom=np.reshape(L-h*np.eye(self.D),self.D**2)
        x=np.copy(inhom)
        y=np.copy(inhom)
        converged=False


        while not converged:
            x=mf.pseudoTransferOperator(A,r,np.eye(self.D),direction=1,vector=x)
            ynew=y+x
            if np.linalg.norm(y-ynew)<1E-13:
                converged=True
            y=np.copy(ynew)
        
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        k=mf.RENORMBLOCKHAMGMRES(A,A,r,np.eye(self.D),inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=1)


        #checks
        self.assertTrue(np.linalg.norm(k-herm(k))/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(np.reshape(k,self.D**2)-y)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(A,A,r,np.eye(self.D),direction=1,momentum=0.0,vector=np.reshape(k,self.D**2))-inhom)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(A,A,r,np.eye(self.D),direction=1,momentum=0.0,vector=y)-inhom)/self.D**2<self.eps)

    def test_RENORMBLOCKHAMGMRES_right_real(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5
        B,rest,Z=mf.prepareTensor(tensor,-1)
        dtype='float64'
        eta,v,numeig=mf.TMeigs(B,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        R=mf.initializeLayer(B,np.eye(self.D),B,self.mpo[1],-1)
        R=np.reshape(mf.addLayer(R,B,self.mpo[0],B,-1),(self.D,self.D))
        h=np.tensordot(l,R,([0,1],[0,1]))
        inhom=np.reshape(R-h*np.eye(self.D),self.D**2)
        x=np.copy(inhom)
        y=np.copy(inhom)
        converged=False
        while not converged:
            x=mf.pseudoTransferOperator(B,np.eye(self.D),l,direction=-1,vector=x)
            ynew=y+x
            if np.linalg.norm(y-ynew)<1E-13:
                converged=True
            y=np.copy(ynew)
    
        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        k=mf.RENORMBLOCKHAMGMRES(B,B,np.eye(self.D),l,inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=-1)
        #checks
        self.assertTrue(np.linalg.norm(k-herm(k))/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(np.reshape(k,self.D**2)-y)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(B,B,np.eye(self.D),l,direction=-1,momentum=0.0,vector=np.reshape(k,self.D**2))-inhom)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(B,B,np.eye(self.D),l,direction=-1,momentum=0.0,vector=y)-inhom)/self.D**2<self.eps)


    def test_RENORMBLOCKHAMGMRES_left_complex(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        A,rest,Z=mf.prepareTensor(tensor,1)
        dtype='complex128'
        eta,v,numeig=mf.TMeigs(A,direction=-1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        r=np.reshape(v,(self.D,self.D))
        r=r/np.trace(r)
        L=mf.initializeLayer(A,np.eye(self.D),A,self.mpo[0],1)
        L=np.reshape(mf.addLayer(L,A,self.mpo[1],A,1),(self.D,self.D))
        h=np.tensordot(L,r,([0,1],[0,1]))
        inhom=np.reshape(L-h*np.eye(self.D),self.D**2)
        x=np.copy(inhom)
        y=np.copy(inhom)
        converged=False
        while not converged:
            
            x=mf.pseudoTransferOperator(A,r,np.eye(self.D),direction=1,vector=x)
            ynew=y+x
            if np.linalg.norm(y-ynew)<1E-13:
                converged=True
            y=np.copy(ynew)

        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        k=mf.RENORMBLOCKHAMGMRES(A,A,r,np.eye(self.D),inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=1)
        #checks
        self.assertTrue(np.linalg.norm(k-herm(k))/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(np.reshape(k,self.D**2)-y)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(A,A,r,np.eye(self.D),direction=1,momentum=0.0,vector=np.reshape(k,self.D**2))-inhom)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(A,A,r,np.eye(self.D),direction=1,momentum=0.0,vector=y)-inhom)/self.D**2<self.eps)



    def test_RENORMBLOCKHAMGMRES_right_complex(self):
        tensor=np.random.rand(self.D,self.D,self.d)-0.5+1j*(np.random.rand(self.D,self.D,self.d)-0.5)
        B,rest,Z=mf.prepareTensor(tensor,-1)
        dtype='complex128'
        eta,v,numeig=mf.TMeigs(B,direction=1,numeig=1,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR')
        l=np.reshape(v,(self.D,self.D))
        l=l/np.trace(l)
        R=mf.initializeLayer(B,np.eye(self.D),B,self.mpo[1],-1)
        R=np.reshape(mf.addLayer(R,B,self.mpo[0],B,-1),(self.D,self.D))
        h=np.tensordot(l,R,([0,1],[0,1]))
        inhom=np.reshape(R-h*np.eye(self.D),self.D**2)

        x=np.copy(inhom)
        y=np.copy(inhom)
        
        converged=False
        while not converged:
            x=mf.pseudoTransferOperator(B,np.eye(self.D),l,direction=-1,vector=x)
            ynew=y+x
            if np.linalg.norm(y-ynew)<1E-13:
                converged=True
            y=np.copy(ynew)

        #solve equation system (inhom|=(k|[11  -  {  T-|r)(l|  }]
        k=mf.RENORMBLOCKHAMGMRES(B,B,np.eye(self.D),l,inhom,x0=None,tolerance=1e-10,maxiteration=2000,direction=-1)
        self.assertTrue(np.linalg.norm(k-herm(k))/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(np.reshape(k,self.D**2)-y)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(B,B,np.eye(self.D),l,direction=-1,momentum=0.0,vector=np.reshape(k,self.D**2))-inhom)/self.D**2<self.eps)
        self.assertTrue(np.linalg.norm(mf.OneMinusPseudoTransferOperator(B,B,np.eye(self.D),l,direction=-1,momentum=0.0,vector=y)-inhom)/self.D**2<self.eps)

        
class UCMPSRegaugingTests(unittest.TestCase):
    def setUp(self):
        self.D=20
        self.N=10
        self.eps=1E-12
        N1=random.randint(1,(self.N-(self.N%2))/2)
        self.d=[random.randint(2,4)]*N1+[random.randint(2,4)]*(self.N-N1)
        random.shuffle(self.d)

    def testRegaugeLeftFloat(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=float)
        mf.regaugeIMPS(self.mps,'left',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeLeftComplex(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=complex)
        mf.regaugeIMPS(self.mps,'left',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)

    def testRegaugeRightFloat(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=float)
        mf.regaugeIMPS(self.mps,'right',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([1,2],[1,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeRightComplex(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=complex)
        mf.regaugeIMPS(self.mps,'right',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([1,2],[1,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
    def testRegaugeSymmetricFloat(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=float)
        lam=mf.regaugeIMPS(self.mps,'symmetric',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeSymmetricComplex(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=1.0,dtype=complex)
        lam=mf.regaugeIMPS(self.mps,'symmetric',D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)


class UCMPSTruncationTests(unittest.TestCase):
    def setUp(self):
        self.D=50
        self.d=2
        self.N=10
        self.eps=1E-12
    def testRegaugeTruncateFloat(self):
        #create a random MPS
        dtype=float
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=0.2,dtype=dtype,shift=0.2)
        lam=mf.regaugeIMPS(self.mps,'symmetric',truncate=1E-16,D=10,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)

        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))

        assert(np.linalg.norm(lam-np.diag(np.diag(lam)))<self.eps)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(lam**2,2)
        Sz=np.diag([1,-1])
        
        m1=mf.measureLocal(self.mps,operators=[Sz]*len(self.mps),lb=lb,rb=rb,ortho='left')
        shapes=list(map(np.shape,self.mps))
        lam=mf.regaugeIMPS(self.mps,'symmetric',truncate=1E-5,D=40,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)

        #print (np.diag(lam))
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(lam**2,2)
        
        self.assertTrue(shapes!=list(map(np.shape,self.mps)))
        m2=mf.measureLocal(self.mps,operators=[Sz]*len(self.mps),lb=lb,rb=rb,ortho='left')
        self.assertTrue(np.linalg.norm(m2-m1)<1E-5)

    def testRegaugeTruncateComplex(self):
        #create a random MPS
        dtype=complex
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=0.1,dtype=dtype,shift=0.2)
        lam=mf.regaugeIMPS(self.mps,'symmetric',truncate=1E-16,D=None,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        assert(np.linalg.norm(lam-np.diag(np.diag(lam)))<self.eps)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(lam**2,2)
        Sz=np.diag([1,-1])        

        m1=mf.measureLocal(self.mps,operators=[Sz]*len(self.mps),lb=lb,rb=rb,ortho='left')
        shapes=list(map(np.shape,self.mps))
        lam=mf.regaugeIMPS(self.mps,'symmetric',truncate=1E-6,D=None,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)

        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(lam**2,2)
        self.assertTrue(shapes!=list(map(np.shape,self.mps)))

        m2=mf.measureLocal(self.mps,operators=[Sz]*len(self.mps),lb=lb,rb=rb,ortho='left')
        self.assertTrue(np.linalg.norm(m2-m1)<1E-5)


class MPSArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.D=20
        self.N=10
        self.d=[2]*self.N
        random.shuffle(self.d)
        self.eps=1E-10


    def testiAdd(self):
        for it in range(100):

            self.mps1=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps2=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps1[-1]*=self.mps1._mat#mps1._mat could contain a minus sign
            self.mps2[-1]*=self.mps2._mat#mps2._mat could contain a minus sign
            self.mps1._mat=np.ones((1,1))
            self.mps2._mat=np.ones((1,1))            
            a1=np.random.rand(1)[0]
            a2=np.random.rand(1)[0]
            mps=self.mps1*a1
            mps+=(self.mps2*a2)
            sz=[np.diag([1,-1]) for n in range(self.N)]
            ov12=self.mps1.__dot__(self.mps2)
            ov21=np.conj(ov12)
            ob11=(np.asarray(self.mps1.__measureList__(sz)))
            ob12=(np.asarray(self.mps1.__measureMatrixElementList__(self.mps2,sz)))
            ob21=np.conj(ob12)#(np.asarray(self.mps2.__measureMatrixElementList__(self.mps1,sz)))
            ob22=(np.asarray(self.mps2.__measureList__(sz)))
            ob33=(np.asarray(mps.__measureList__(sz)))
            ov33=mps.__dot__(mps)
            Z=a1**2+a2**2+ov12*a1*a2+ov21*a1*a2
            M=ob11*a1**2+ob22*a2**2+ob21*a2*a1+ob12*a1*a2
            self.assertTrue(np.linalg.norm(ob33/mps._Z**2-M/Z)<1E-10)
            self.assertTrue(np.linalg.norm(ob33/ov33-M/Z)<1E-10)            
        
    def testAdd(self):
        for it in range(100):

            self.mps1=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps2=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps1[-1]*=self.mps1._mat#mps1._mat could contain a minus sign
            self.mps2[-1]*=self.mps2._mat#mps2._mat could contain a minus sign
            self.mps1._mat=np.ones((1,1))
            self.mps2._mat=np.ones((1,1))            
            a1=np.random.rand(1)[0]
            a2=np.random.rand(1)[0]
            mps=(self.mps1*a1+self.mps2*a2)
            sz=[np.diag([1,-1]) for n in range(self.N)]
            ov12=self.mps1.__dot__(self.mps2)
            ov21=np.conj(ov12)
            ob11=(np.asarray(self.mps1.__measureList__(sz)))
            ob12=(np.asarray(self.mps1.__measureMatrixElementList__(self.mps2,sz)))
            ob21=np.conj(ob12)#(np.asarray(self.mps2.__measureMatrixElementList__(self.mps1,sz)))
            ob22=(np.asarray(self.mps2.__measureList__(sz)))
            ob33=(np.asarray(mps.__measureList__(sz)))
            ov33=mps.__dot__(mps)
            Z=a1**2+a2**2+ov12*a1*a2+ov21*a1*a2
            M=ob11*a1**2+ob22*a2**2+ob21*a2*a1+ob12*a1*a2
            self.assertTrue(np.linalg.norm(ob33/mps._Z**2-M/Z)<1E-10)
            self.assertTrue(np.linalg.norm(ob33/ov33-M/Z)<1E-10)            

    def testSub(self):
        for it in range(100):
            self.mps1=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps2=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps1[-1]*=self.mps1._mat
            self.mps2[-1]*=self.mps2._mat            
            self.mps1._mat=np.ones((1,1))
            self.mps2._mat=np.ones((1,1))            
            a1=np.random.rand(1)[0]
            a2=np.random.rand(1)[0]
            mps=(self.mps1*a1-self.mps2*a2)
            sz=[np.diag([1,-1]) for n in range(self.N)]
            ov12=self.mps1.__dot__(self.mps2)
            ov21=np.conj(ov12)
            ob11=(np.asarray(self.mps1.__measureList__(sz)))
            ob12=(np.asarray(self.mps1.__measureMatrixElementList__(self.mps2,sz)))
            ob21=np.conj(ob12)#(np.asarray(self.mps2.__measureMatrixElementList__(self.mps1,sz)))
            ob22=(np.asarray(self.mps2.__measureList__(sz)))
            ob33=(np.asarray(mps.__measureList__(sz)))
            ov33=mps.__dot__(mps)
            Z=a1**2+a2**2-ov12*a1*a2-ov21*a1*a2
            M=ob11*a1**2+ob22*a2**2-ob21*a2*a1-ob12*a1*a2
            self.assertTrue(np.linalg.norm(ob33/mps._Z**2-M/Z)<1E-10)
            self.assertTrue(np.linalg.norm(ob33/ov33-M/Z)<1E-10)
            
    def testISub(self):
        for it in range(100):
            self.mps1=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps2=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
            self.mps1[-1]*=self.mps1._mat
            self.mps2[-1]*=self.mps2._mat            
            self.mps1._mat=np.ones((1,1))
            self.mps2._mat=np.ones((1,1))            
            a1=np.random.rand(1)[0]
            a2=np.random.rand(1)[0]
            mps=self.mps1*a1
            mps-=(self.mps2*a2)
            sz=[np.diag([1,-1]) for n in range(self.N)]
            ov12=self.mps1.__dot__(self.mps2)
            ov21=np.conj(ov12)
            ob11=(np.asarray(self.mps1.__measureList__(sz)))
            ob12=(np.asarray(self.mps1.__measureMatrixElementList__(self.mps2,sz)))
            ob21=np.conj(ob12)#(np.asarray(self.mps2.__measureMatrixElementList__(self.mps1,sz)))
            ob22=(np.asarray(self.mps2.__measureList__(sz)))
            ob33=(np.asarray(mps.__measureList__(sz)))
            ov33=mps.__dot__(mps)
            Z=a1**2+a2**2-ov12*a1*a2-ov21*a1*a2
            M=ob11*a1**2+ob22*a2**2-ob21*a2*a1-ob12*a1*a2
            self.assertTrue(np.linalg.norm(ob33/mps._Z**2-M/Z)<1E-10)
            self.assertTrue(np.linalg.norm(ob33/ov33-M/Z)<1E-10)            
            
            
    def testSub2(self):
        self.mps1=mpslib.MPS.random(self.N,self.D,self.d,obc=True,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16,scaling=0.5,shift=0.2)
        self.mps1[-1]*=self.mps1._mat
        self.mps1._mat=np.ones((1,1))
        self.mps2=self.mps1.copy()
        mps=(self.mps1-self.mps2)
        mps.position(0)
        mps.position(len(mps))
        sz=[np.diag([1,-1]) for n in range(self.N)]
        ob33=(np.asarray(mps.__measureList__(sz)))
        self.assertTrue(np.linalg.norm(ob33)<1E-10)
            

class MPSTests(unittest.TestCase):
    def setUp(self):
        self.D=20
        self.N=20
        N1=random.randint(1,(self.N-(self.N%2))/2)
        self.d=[random.randint(2,4)]*N1+[random.randint(2,4)]*(self.N-N1)
        random.shuffle(self.d)

        self.eps=1E-10
        
    def testRegaugeLeftFloat(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.4,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='left',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeLeftComplex(self):
        #create a random MPS
    
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=complex,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='left',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==complex)            
    
    def testRegaugeRightFloat(self):
        #create a random MPS
    
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='right',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)
        
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([1,2],[1,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeRightComplex(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=complex,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='right',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)
    
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([1,2],[1,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==complex)            
    
    def testRegaugeSymmetricFloat(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='symmetric',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)
        
        for n in range(len(self.mps)):
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==float)
            
    def testRegaugeSymmetricComplex(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=complex,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.__regauge__(gauge='symmetric',nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12)        
        
        for n in range(len(self.mps)):
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            self.assertTrue(self.mps[n].dtype==complex)


    def testRegaugeTruncateFloat(self):
        #create a random MPS
        self.D=50
        dtype=float

        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=complex,shift=0.4,schmidt_thresh=1E-14,r_thresh=1E-16)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))

        assert(np.linalg.norm(self.mps._mat-np.diag(np.diag(self.mps._mat)))<self.eps)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(self.mps._mat**2,2)
        Ops=[]
        for n in range(len(self.mps)):
            Ops.append(np.diag(np.random.rand(self.mps._d[n])))
        m1=mf.measureLocal(self.mps,operators=Ops,lb=lb,rb=rb,ortho='left')
        shapes=list(map(np.shape,self.mps._tensors))
        #print(shapes)
        self.mps.__truncate__(schmidt_thresh=1E-4,D=None)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(self.mps._mat**2,2)
        #print(list(map(np.shape,self.mps._tensors)))
        self.assertTrue(shapes!=list(map(np.shape,self.mps._tensors)))

        m2=mf.measureLocal(self.mps,operators=Ops,lb=lb,rb=rb,ortho='left')
        self.assertTrue(np.linalg.norm(m2-m1)<1E-4)

    def testRegaugeTruncateComplex(self):
        #create a random MPS
        self.D=50
        dtype=complex

        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.1,dtype=complex,shift=0.35,schmidt_thresh=1E-14,r_thresh=1E-16)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        for n in range(len(self.mps)):
            self.assertTrue(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1]))<self.eps)
            #print(np.linalg.norm(np.tensordot(self.mps[n],np.conj(self.mps[n]),([0,2],[0,2]))-np.eye(self.mps[n].shape[1])))

        assert(np.linalg.norm(self.mps._mat-np.diag(np.diag(self.mps._mat)))<self.eps)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(self.mps._mat**2,2)
        Ops=[]
        for n in range(len(self.mps)):
            Ops.append(np.diag(np.random.rand(self.mps._d[n])))
        m1=mf.measureLocal(self.mps,operators=Ops,lb=lb,rb=rb,ortho='left')
        shapes=list(map(np.shape,self.mps._tensors))
        #print(shapes)
        self.mps.__truncate__(schmidt_thresh=1E-5,D=None)
        lb=np.expand_dims(np.eye(self.mps[0].shape[0]),2)
        #note that lam is a diagonal matrix
        rb=np.expand_dims(self.mps._mat**2,2)
        #print(list(map(np.shape,self.mps._tensors)))
        self.assertTrue(shapes!=list(map(np.shape,self.mps._tensors)))

        m2=mf.measureLocal(self.mps,operators=Ops,lb=lb,rb=rb,ortho='left')
        self.assertTrue(np.linalg.norm(m2-m1)<1E-5)


    def testtruncateMPS(self):
        return
        Jz=np.ones(self.N)
        Jxy=np.ones(self.N)
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=True)
        self.mps.__position__(self.N-1)
        self.mps.__position__(0)
        mpoobc=H.XXZ(Jz,Jxy,np.zeros(self.N),True)
        lb=np.ones((1,1,1))
        rb=np.ones((1,1,1))
        dmrg=en.DMRGengine(self.mps,mpoobc,'blabla',lb,rb)
        dmrg.__simulate__(20,1e-13,1e-10,50,verbose=0,cp=10)
        dmrg._mps.__position__(self.N)
        dmrg._mps.__position__(0)
        
        Sz=np.diag([0.5,-0.5])
        
        
        meanSzSz=[]
        meanSz=[]

        for n in range(self.N):
            meanSz=dmrg._mps.__measureLocal__([Sz]*self.N)
        for n in range(self.N):
            meanSzSz.append(dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n])))        
            
        Dt=18
        dmrg._mps.__truncate__(schmidt_thresh=1E-8,D=Dt,r_thresh=1E-14)
        dmrg._mps.__position__(self.N)
        dmrg._mps.__position__(0)

        self.assertTrue(dmrg._mps.__D__()<=[Dt]*len(dmrg._mps))
        meanSzSztrunc=[]
        meanSztrunc=[]    
        for n in range(self.N):
            meanSztrunc.append(dmrg._mps.__measure__(Sz,n))
        for n in range(self.N):
            meanSzSztrunc.append(dmrg._mps.__measure__([Sz,Sz],sorted([int(self.N/2),n])))        

        #plt.figure(1)
        #plt.plot(range(len(meanSz)),meanSz,range(len(meanSztrunc)),meanSztrunc,'--')
        #plt.ylabel(r'$\langle S^z_i\rangle$')
        #plt.xlabel(r'$i$')    
        #plt.legend(['before truncation (D={0})'.format(self.D),'after truncation (D={0})'.format(Dt)])
        #
        #plt.figure(2)
        #plt.plot(range(len(meanSzSz)),meanSzSz,range(len(meanSzSztrunc)),meanSzSztrunc,'--')
        #plt.ylabel(r'$\langle S^z_{N/2} S^z_{i}\rangle$')
        #plt.xlabel(r'$i$')        
        #plt.legend(['before truncation (D={0})'.format(self.D),'after truncation (D={0})'.format(Dt)])    
        #
        #plt.draw()
        #plt.show()
        #input()


        self.assertTrue(np.max(np.abs(np.array(meanSz)-np.array(meanSztrunc)))<1E-6)
        self.assertTrue(np.max(np.abs(np.array(meanSzSz)-np.array(meanSzSztrunc)))<1E-3)        

class CanonizeTests(unittest.TestCase):
    def setUp(self):
        self.D=20
        self.N=20
        self.eps=1E-12
        N1=random.randint(1,(self.N-(self.N%2))/2)
        self.d=[random.randint(2,4)]*N1+[random.randint(2,4)]*(self.N-N1)
        random.shuffle(self.d)

    def testCanonizeOBCMPSFloat(self):
        #create a random MPS
    
        self.mps=mpslib.MPS.random(self.N,D=10,d=2,obc=True,scaling=0.1,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16)
    
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
    
    def testCanonizeOBCMPSComplex(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,D=10,d=2,obc=True,scaling=0.1,dtype=complex,schmidt_thresh=1E-16,r_thresh=1E-16)
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)

    def testCanonizePBCMPSFloatMPS(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.4,dtype=float)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
            
    def testCanonizePBCMPSComplexMPS(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.4,dtype=complex)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
            
    def testCanonizePBCMPSFloatList(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=0.4,dtype=float)
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)


    def testCanonizePBCMPSComplexList(self):
        #create a random MPS
        self.mps=mf.MPSinit(self.N,self.D,self.d,obc=False,scale=0.4,dtype=complex)
        Gamma,Lambda=mf.canonizeMPS(self.mps)
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)



class CanonizeTestsMPS(unittest.TestCase):
    def setUp(self):
        self.D=20
        self.N=20
        self.eps=1E-12
        N1=random.randint(1,(self.N-(self.N%2))/2)
        self.d=[random.randint(2,4)]*N1+[random.randint(2,4)]*(self.N-N1)
        random.shuffle(self.d)

    def testCanonizeOBCMPSFloat(self):
        #create a random MPS
    
        self.mps=mpslib.MPS.random(self.N,D=10,d=2,obc=True,scaling=0.1,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.canonize()
        Gamma=self.mps._gamma
        Lambda=self.mps._lambda
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
    
    def testCanonizeOBCMPSComplex(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,D=10,d=2,obc=True,scaling=0.1,dtype=complex,schmidt_thresh=1E-16,r_thresh=1E-16)
        self.mps.canonize()
        Gamma=self.mps._gamma
        Lambda=self.mps._lambda
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)

    def testCanonizePBCMPSFloatMPS(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.4,dtype=float)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        self.mps.canonize()
        Gamma=self.mps._gamma
        Lambda=self.mps._lambda
        
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
            
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)

    def testCanonizePBCMPSComplexMPS(self):
        #create a random MPS
        self.mps=mpslib.MPS.random(self.N,self.D,self.d,obc=False,scaling=0.4,dtype=complex)
        self.mps.__regauge__(gauge='symmetric',nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-12)
        self.mps.canonize()
        Gamma=self.mps._gamma
        Lambda=self.mps._lambda
        
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)
            
        for n in range(len(Gamma)):
            A=np.tensordot(np.diag(Lambda[n]),Gamma[n],([1],[0]))
            self.assertTrue(np.linalg.norm(np.tensordot(A,np.conj(A),([0,2],[0,2]))-np.eye(A.shape[1]))<self.eps)
        for n in range(len(Gamma)):
            B=np.transpose(np.tensordot(Gamma[n],np.diag(Lambda[n+1]),([1],[0])),(0,2,1))
            self.assertTrue(np.linalg.norm(np.tensordot(B,np.conj(B),([1,2],[1,2]))-np.eye(B.shape[0]))<self.eps)




if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTMeigs)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestRegauge)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestRENORMBLOCKHAMGMRES)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(UCMPSRegaugingTests)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(UCMPSTruncationTests)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(MPSTests)    
    suite7 = unittest.TestLoader().loadTestsFromTestCase(CanonizeTests)
    suite8 = unittest.TestLoader().loadTestsFromTestCase(MPSArithmeticTests)
    suite9 = unittest.TestLoader().loadTestsFromTestCase(CanonizeTestsMPS)    
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)
    unittest.TextTestRunner(verbosity=2).run(suite3)
    unittest.TextTestRunner(verbosity=2).run(suite4)
    unittest.TextTestRunner(verbosity=2).run(suite5)
    unittest.TextTestRunner(verbosity=2).run(suite6)
    unittest.TextTestRunner(verbosity=2).run(suite7)
    unittest.TextTestRunner(verbosity=2).run(suite8)
    unittest.TextTestRunner(verbosity=2).run(suite9) 
