#!/usr/bin/env python
import numpy as np
import scipy as sp
import math
import datetime as dt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import eigs
from scipy.linalg import expm
from scipy.interpolate import interp1d

import functools as fct
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.Hamiltonians as H


comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))



class LanczosEngine:

    def __init__(self,Ndiag,nmax,numeig,delta,deltaEta,dtype):

        self._Ndiag=Ndiag
        self._nmax=nmax
        self._numeig=numeig
        self._dtype=dtype
        self._delta=delta
        self._deltaEta=deltaEta
        assert(Ndiag>0)
        

    def __simulate__(self,L,mpo,R,mps,verbose=False):
        #initialization:
        D1,D2,d=np.shape(mps)
        xn=np.copy(mps)
        xn_minus_1=np.zeros((D1,D2,d),dtype=self._dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            kn.append(np.linalg.norm(xn))
            if kn[-1]<self._delta:
                converged=True
            xn=xn/kn[-1]
            vecs.append(np.copy(xn))
            
            Hxn=mf.HAproductSingleSiteMPS(L,mpo,R,xn)
            epsn.append(np.tensordot(Hxn,np.conj(xn),([0,1,2],[0,1,2])))

            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])
            xn_plus_1=Hxn-epsn[-1]*xn-kn[-1]*xn_minus_1
            xn_minus_1=np.copy(xn)
            xn=np.copy(xn_plus_1)
            
            
            it=it+1
            if it>=self._nmax:
                break

        #now get the eigenvectors with the lowest eigenvalues:
        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(Heff)

        states=[]
        for n2 in range(self._numeig):
            state=np.zeros((D1,D2,d))
            for n1 in range(len(vecs)):
                state=state+vecs[n1]*u[n1,n2]

            states.append(np.copy(state))
        #print eta
        #print self._numeig

        return eta[0:self._numeig],states






    def __simulatecmps__(self,cmps,leftsite,centersite,rightsite,fl,mpo,fr,mps):
        #initialization:

        D1,D2,d=np.shape(mps)
        xn=np.copy(mps)
        xn_minus_1=np.zeros((D1,D2,d),dtype=self._dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            kn.append(np.linalg.norm(xn))
            if kn<self._delta:
                converged=True
            xn=xn/kn[-1]

            #store the Lanczos vector for later
            vecs.append(np.copy(xn))
            Hxn=cmfd.cmpsHAproduct(cmps,leftsite,centersite,rightsite,fl,mpo,fr,xn)            
            #Hxn=cmfd.cmpsHAproduct(cmps,leftsite,centersite,rightsite,fl,mpo,fr,xn)
            epsn.append(np.tensordot(Hxn,np.conj(xn),([0,1,2],[0,1,2])))
            #print epsn

            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                #print '===================='
                #print Heff
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])
            #raw_input('it={2}, k={0}, eps={1}'.format(kn[-1],epsn[-1],it))            

            #this one is NOT YET NORMALIZED
            xn_plus_1=Hxn-epsn[-1]*xn-kn[-1]*xn_minus_1
            xn_minus_1=np.copy(xn)
            xn=np.copy(xn_plus_1)
            
            
            it=it+1
            if it>self._nmax:
                break

        
        #now get the eigenvectors with the lowest eigenvalues:

        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(Heff)
        #if eta[0]<-0.2:
        #    print eta
        #    raw_input()

        states=[]
        for n2 in range(self._numeig):
            state=np.zeros((D1,D2,d))
            for n1 in range(len(vecs)):
                state=state+vecs[n1]*u[n1,n2]
            states.append(np.copy(state))
        #print eta
        #print self._numeig
        return eta[0:self._numeig],states

            
    def __evolve__(self,L,mpo,R,mps,dt):
        #initialization:

        D1,D2,d=np.shape(mps)
        xn=np.copy(mps)
        xn_minus_1=np.zeros((D1,D2,d),dtype=self._dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            kn.append(np.linalg.norm(xn))
            if kn<self._delta:
                converged=True
            xn=xn/kn[-1]

            #store the Lanczos vector for later
            vecs.append(np.copy(xn))
            
            Hxn=mf.HAproductSingleSiteMPS(L,mpo,R,xn)
            epsn.append(np.tensordot(Hxn,np.conj(xn),([0,1,2],[0,1,2])))

            
            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])
            
            #this one is NOT YET NORMALIZED
            xn_plus_1=Hxn-epsn[-1]*xn-kn[-1]*xn_minus_1
            xn_minus_1=np.copy(xn)
            xn=np.copy(xn_plus_1)
            
            
            it=it+1
            if it>self._nmax:
                break

        
        #now get the eigenvectors with the lowest eigenvalues:

        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        prop=expm(-dt*1j*Heff)
        state=np.zeros((D1,D2,d))
        for n1 in range(len(vecs)):
            state=state+vecs[n1]*prop[n1,0]

        return eta[0],state



            
    def __simulateBond__(self,L,mpo,R,mps,dens0,position):
        #initialization:

        D1,D2,d=np.shape(mps)
        xn=np.copy(dens0)
        xn_minus_1=np.zeros((D1,D2),dtype=self._dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            kn.append(np.linalg.norm(xn))
            if kn<self._delta:
                converged=True
            xn=xn/kn[-1]

            #store the Lanczos vector for later
            vecs.append(np.copy(xn))
            
            Hxn=mf.HAproductZeroSiteMat(L,mpo,mps,R,position,xn)
            epsn.append(np.tensordot(Hxn,np.conj(xn),([0,1],[0,1])))

            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])
            
            #this one is NOT YET NORMALIZED
            xn_plus_1=Hxn-epsn[-1]*xn-kn[-1]*xn_minus_1
            xn_minus_1=np.copy(xn)
            xn=np.copy(xn_plus_1)
            
            it=it+1
            if it>self._nmax:
                break

        
        #now get the eigenvectors with the lowest eigenvalues:

        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(Heff)

        states=[]
        for n2 in range(self._numeig):
            state=np.zeros((D1,D2))
            for n1 in range(len(vecs)):
                state=state+vecs[n1]*u[n1,n2]
            states.append(np.copy(state))

        return eta[0:self._numeig],states
            
    def __evolveBond__(self,L,mpo,R,mps,dt,dens0,position):
        #initialization:

        D1,D2,d=np.shape(mps)
        xn=np.copy(dens0)
        xn_minus_1=np.zeros((D1,D2),dtype=self._dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            kn.append(np.linalg.norm(xn))
            if kn<self._delta:
                converged=True
            xn=xn/kn[-1]

            #store the Lanczos vector for later
            vecs.append(np.copy(xn))
            
            Hxn=mf.HAproductZeroSiteMat(L,mpo,mps,R,position,xn)
            epsn.append(np.tensordot(Hxn,np.conj(xn),([0,1],[0,1])))
            
            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])
            
            #this one is NOT YET NORMALIZED
            xn_plus_1=Hxn-epsn[-1]*xn-kn[-1]*xn_minus_1
            xn_minus_1=np.copy(xn)
            xn=np.copy(xn_plus_1)
            
            
            it=it+1
            if it>self._nmax:
                break

        
        #now get the eigenvectors with the lowest eigenvalues:

        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        prop=expm(-dt*1j*Heff)
        state=np.zeros((D1,D2))
        for n1 in range(len(vecs)):
            state=state+vecs[n1]*prop[n1,0]

        return eta[0],state
            







        
                
                              
                
                
        
