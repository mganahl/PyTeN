#!/usr/bin/env python

import numpy as np
import scipy as sp
import math
import datetime as dt
import lib.mpslib.qmpsfunctions as qmf
import lib.sparsetensor.sparsenumpy as snp


comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

class qLanczosEngine:

    def __init__(self,Ndiag,nmax,numeig,delta,deltaEta,dtype):

        self._Ndiag=Ndiag
        self._nmax=nmax
        self._numeig=numeig
        self._dtype=dtype
        self._delta=delta
        self._deltaEta=deltaEta
        assert(Ndiag>0)
        

    def __simulate__(self,L,mps,mpo,R):
        #initialization:
        #print mpo._qflow
        #raw_input()
        xn=mps.__copy__()
        xn.__normalize__()

        xn_minus_1=snp.zeros(xn)
        #xn_minus_1.__squeeze__()
        converged=False
        it=0
        kn=[]
        epsn=[]
        vecs=[]
        first=True

        while converged==False:
            #normalize the current vector:
            knval=xn.__norm__()
            if knval<self._delta:
                converged=True
                break
            kn.append(knval)            
            xn/=kn[-1]
            vecs.append(xn.__copy__())

            Hxn=qmf.HAproduct(L,xn,mpo,R)

            #print snp.tensordot(Hxn,snp.conj(xn),([0,1,2],[0,1,2]))._tensor.values()
            epsn.append(list(snp.tensordot(Hxn,snp.conj(xn),([0,1,2],[0,1,2]))._tensor.values())[0])
            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                #print it,eta[0]
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=np.copy(eta[0:self._numeig])

            if it>0:
                Hxn-=(vecs[-1]*epsn[-1])
                Hxn-=(vecs[-2]*kn[-1])
                #print Hxn._qflow,vecs[-1]._qflow,vecs[-2]._qflow
            else:
                Hxn-=(vecs[-1]*epsn[-1])
                #print Hxn._qflow,vecs[-1]._qflow
            xn=Hxn.__copy__() 
            it=it+1
            if it>=self._nmax:
                break

        #now get the eigenvectors with the lowest eigenvalues:
        Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(Heff)
        states=[]
        for n2 in range(self._numeig):
            state=snp.zeros(mps.__copy__()).__squeeze__()
            #print '===== state._qflow:'
            #print state._qflow
            #print 'vecs._qflow,state._qflow'
            for n1 in range(len(vecs)):
                state+=(vecs[n1]*u[n1,n2])
            #    print vecs[n1]._qflow,state._qflow                
            #print '========done ===='                
            states.append(state.__copy__())


        return eta[0:self._numeig],states
    
                
                
        
