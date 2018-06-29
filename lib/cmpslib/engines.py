#!/usr/bin/env python
from sys import stdout
import numpy as np
import os,sys
import time
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
def checkWick(lam,Rl):
    Rc=Rl.dot(np.diag(lam))
    wick=(np.trace(herm(Rc).dot(herm(Rc)))*np.trace(Rc.dot(Rc))+np.trace(herm(Rc).dot(Rc))*np.trace(herm(Rc).dot(Rc)))
    #wick=np.trace(herm(Rc).dot(Rc))*np.trace(herm(Rc).dot(Rc))
    full=np.trace(herm(Rc).dot(herm(Rc)).dot(Rc).dot(Rc))
    return wick,full

try:
    LZ=sys.modules['lib.mpslib.Lanczos']
except KeyError:
    import lib.mpslib.Lanczos as LZ
try:
    mf=sys.modules['lib.mpslib.mpsfunctions']
except KeyError:
    import lib.mpslib.mpsfunctions as mf
try:
    cmf=sys.modules['lib.cmpslib.cmpsfunctions']
except KeyError:
    import lib.cmpslib.cmpsfunctions as cmf

try:
    H=sys.modules['lib.mpslib.Hamiltonians']
except KeyError:
    import lib.mpslib.Hamiltonians as H
try:
    dcmps=sys.modules['lib.cmpslib.discreteCMPS']
except KeyError:
    import lib.cmpslib.discreteCMPS as dcmps
try:
    utils=sys.modules['lib.utils.utilities']
except KeyError:
    import lib.utils.utilities as utils

try:
    cqr=sys.modules['lib.cmpslib.cQR']
except KeyError:
    import lib.cmpslib.cQR as cqr

#import lib.mpslib.Lanczos as LZ
#import lib.mpslib.Hamiltonians as H
#import lib.mpslib.mpsfunctions as mf
#import lib.utils.utilities as utils
#import lib.cmpslib.cmpsfunctions as cmf
#import lib.cmpslib.discreteCMPS as dcmps
#import lib.cmpslib.cQR as cqr
import shutil


from scipy.sparse.linalg import ArpackNoConvergence
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


#this is an optimization engine for the DiscreteCMPS class (single species of Bosons).
class IcDMRGengine:

    def __init__(self,cmps,mpo,filename):
        self._cmps=cmps
        self._mpo=mpo
        self._filename=filename
        self._N=self._cmps._N
        self._L=[]
        self._R=[]
        self._verbose=False

        
    #a subroutine which optimizes site 'n' of a DiscreteCMPS
    def __optimize__(self,n,direction,tol,ncv,mode='AR',Ndiag=4,landelta=1E-10,dt=0.0,verbosity=1,optimize=True):
        if verbosity>0:
            stdout.write('\r site = %i'%(n))
            stdout.flush()
        if direction>0:
            self._cmps.__position__(n+1)
            init=self._cmps.__wavefnct__('l')
        if direction<0:
            self._cmps.__position__(n)
            init=self._cmps.__wavefnct__('r')
        if optimize==False:
            e=0.0
        if optimize==True:
            if n==0:
                if mode=='AR':
                    e,opt=mf.eigsh(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')
            
            if (n>0)&(n<self._cmps._N-1):
                if mode=='AR':
                    e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                    
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')
            
            if n==(self._cmps._N-1):
                if mode=='AR':            
                    e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._rb,init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._rb,init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._rb,init,dt)
        

        if direction>0:
            if optimize==True:
                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                #tensor,mat=mf.prepareTensor(opt,direction=1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                #print mat-self._cmps._mats[n+1]
                self._cmps._mats[n+1]=np.copy(mat)
                self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            if n==0:
                self._L[n]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            if n>0:
                self._L[n]=np.copy(mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            return e



        if direction<0:
            if optimize==True:
                tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
                #tensor,mat=mf.prepareTensor(opt,direction=-1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                self._cmps._mats[n]=np.copy(mat)
                self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            
            if n==(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            if n<(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._R[self._cmps._N-n-2],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            return e

    #a subroutine which optimizes matrices at position grid[index] of a DiscreteCMPS
    def __optimize_grid__(self,index,grid,direction,tol,ncv,mode='AR',Ndiag=4,landelta=1E-10,dt=0.0,verbosity=1,optimize=True):
        #make sure that the last grid point is one larger than the length of the cMPS; this ensures that the code below
        #works for the update of the boundary site;
        #assert(grid[-1]==(self._cmps._N-1))
        #assert(grid[0]==0)
        assert(self._cmps._N>1)
        assert(len(grid)<=self._cmps._N)
        n=grid[index]
        if verbosity>0:
            stdout.write('\r site = %i'%(n))
            stdout.flush()
        if direction>0:
            self._cmps.__position__(n+1)
            init=self._cmps.__wavefnct__('l')

        if direction<0:
            self._cmps.__position__(n)
            init=self._cmps.__wavefnct__('r')

        if optimize==False:
            e=0.0
        if optimize==True:
            if n==0:
                if mode=='AR':
                    e,opt=mf.eigsh(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')
            
            if (n>0)&(n<self._cmps._N-1):
                if mode=='AR':
                    e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                    
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')
            
            if n==(self._cmps._N-1):
                if mode=='AR':            
                    e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._rb,init,tol,1,ncv)
                if mode=='LAN':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._rb,init,self._verbose)
                    e=es[0]
                    opt=np.copy(opts[0])
                if mode=='LANTEV':
                    lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._rb,init,dt)
        

        if direction>0:
            if optimize==True:
                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                #tensor,mat=mf.prepareTensor(opt,direction=1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                #print mat-self._cmps._mats[n+1]
                self._cmps._mats[n+1]=np.copy(mat)
                self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
                self._cmps._position=n+1

            if n<(self._cmps._N-1):
                if index<(len(grid)-1):
                    np1=grid[index+1]
                if index==(len(grid)-1):
                    np1=self._cmps._N-1

                self._cmps.__position__(np1)
                for site in range(n,np1):
                    if site==0:
                        self._L[site]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),1))
                    if site>0:
                        self._L[site]=np.copy(mf.addLayer(self._L[site-1],self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),1))

            if n==(self._cmps._N-1):
                np1=self._cmps._N-1
                self._cmps.__position__(np1)
                for site in range(n,np1):
                    if site==0:
                        self._L[site]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),1))
                    if site>0:
                        self._L[site]=np.copy(mf.addLayer(self._L[site-1],self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),1))

            #if n==0:
            #    self._L[n]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            #    #if n<(self._cmps._N-1):
            #    if index<(len(grid)-1):
            #        self._cmps.__position__(grid[index+1])
            #        for m in range(n+1,grid[index+1]):
            #            self._L[m]=np.copy(mf.addLayer(self._L[m-1],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),1))
            #    if index==(len(grid)-1):
            #        self._cmps.__position__(self._cmps._N)
            #        #here, index<len(grid)-1 due to the above assertions at the begining of the function
            #        for m in range(n+1,self._cmps._N):
            #            self._L[m]=np.copy(mf.addLayer(self._L[m-1],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),1))
            #
            #if n>0:
            #    self._L[n]=np.copy(mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            #    #if n<(self._cmps._N-1):
            #    if index<(len(grid)-1):
            #        self._cmps.__position__(grid[index+1])
            #        #here, index<len(grid)-1 due to the above assertions at the begining of the function
            #        for m in range(n+1,grid[index+1]):
            #            self._L[m]=np.copy(mf.addLayer(self._L[m-1],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),1))
            #    if index==(len(grid)-1):
            #        self._cmps.__position__(self._cmps._N)
            #        #here, index<len(grid)-1 due to the above assertions at the begining of the function
            #        for m in range(n+1,self._cmps._N):
            #            self._L[m]=np.copy(mf.addLayer(self._L[m-1],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),1))
            #
            return e


        if direction<0:
            print ('not implemented')
            sys.exit()
            #if optimize==True:
            #    tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
            #    #tensor,mat=mf.prepareTensor(opt,direction=-1)
            #    Z=np.trace(mat.dot(herm(mat)))
            #    mat=mat/np.sqrt(Z)
            #    self._cmps._mats[n]=np.copy(mat)
            #    self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            #
            #if n==(self._cmps._N-1):
            #    self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            #    #if n>0:
            #    if index>0:
            #        self._cmps.__position__(grid[index-1])
            #        for m in range(n-1,grid[index-1],-1):
            #            self._R[self._cmps._N-1-m]=np.copy(mf.addLayer(self._R[self._cmps._N-m-2],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),-1))
            #    if index==0:
            #        self._cmps.__position__(0)
            #        for m in range(n-1,0,-1):
            #            self._R[self._cmps._N-1-m]=np.copy(mf.addLayer(self._R[self._cmps._N-m-2],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),-1))
            #
            #if n<(self._cmps._N-1):
            #    self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._R[self._cmps._N-n-2],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            #    #if n>0:
            #    if index>0:
            #        self._cmps.__position__(grid[index-1])
            #        for m in range(n-1,grid[index-1],-1):
            #            self._R[self._cmps._N-1-m]=np.copy(mf.addLayer(self._R[self._cmps._N-m-2],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),-1))
            #    if index==0:
            #        self._cmps.__position__(0)
            #        for m in range(n-1,0,-1):
            #            self._R[self._cmps._N-1-m]=np.copy(mf.addLayer(self._R[self._cmps._N-m-2],self._cmps.__tensor__(m),self._mpo[m],self._cmps.__tensor__(m),-1))
            #
            #



    def __evolvehomogeneous__(self,n,direction,dt,optimize=True):
        stdout.write('.')
        stdout.flush()
        if direction>0:
            self._cmps.__position__(n+1)
            init=self._cmps.__wavefnct__('l')
            lam=np.copy(self._cmps._mats[self._cmps._position])
            if n==0:
                Hx=mf.HAproductSingleSiteMPS(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init)
                opt=init-dt*Hx
                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                
                #Z=np.trace(mat.dot(herm(mat)))
                #mat=mat/np.sqrt(Z)
                
                Hmat=mf.HAproductZeroSiteMat(self._lb,self._mpo[n],tensor,self._R[self._cmps._N-2-n],'right',mat)
                
                mat=mat+dt*Hmat
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
            if (n>0)&(n<self._cmps._N-1):
                Hx=mf.HAproductSingleSiteMPS(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init)
                opt=init-dt*Hx
                
                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                #Z=np.trace(mat.dot(herm(mat)))
                #mat=mat/np.sqrt(Z)
                
                Hmat=mf.HAproductZeroSiteMat(self._L[n-1],self._mpo[n],tensor,self._R[self._cmps._N-2-n],'right',mat)
                
                mat=mat+dt*Hmat
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
            if n==(self._cmps._N-1):
                Hx=mf.HAproductSingleSiteMPS(self._L[n-1],self._mpo[n],self._rb,init)
                opt=init-dt*Hx
                
                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
                #Hmat=mf.HAproductZeroSiteMat(self._L[n-1],self._mpo[n],tensor,self._rb,'right',mat)
                
                #mat=mat+dt*Hmat
                #Z=np.trace(mat.dot(herm(mat)))
                #mat=mat/np.sqrt(Z)
            e=np.tensordot(Hx,np.conj(init),([0,1,2],[0,1,2]))                
            #print np.tensordot(Hx,np.conj(Hx),([0,1,2],[0,1,2])),np.tensordot(Hmat,np.conj(Hmat),([0,1],[0,1])),e
            #raw_input()

            self._cmps._mats[n+1]=np.copy(mat)
            self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            if n==0:
                self._L[n]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            if n>0:
                self._L[n]=np.copy(mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            return e


        if direction<0:
            self._cmps.__position__(n)
            init=self._cmps.__wavefnct__('r')
            lam=np.copy(self._cmps._mats[self._cmps._position])

            if n==0:
                Hx=mf.HAproductSingleSiteMPS(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init)
                opt=init-dt*Hx
                tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
                
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
                Hmat=mf.HAproductZeroSiteMat(self._lb,self._mpo[n],tensor,self._R[self._cmps._N-2-n],'left',mat)
                
                mat=mat+dt*Hmat
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
            if (n>0)&(n<self._cmps._N-1):
                Hx=mf.HAproductSingleSiteMPS(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init)
                opt=init-dt*Hx
                
                tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
                Hmat=mf.HAproductZeroSiteMat(self._L[n-1],self._mpo[n],tensor,self._R[self._cmps._N-2-n],'right',mat)
                
                mat=mat+dt*Hmat
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
            if n==(self._cmps._N-1):
                Hx=mf.HAproductSingleSiteMPS(self._L[n-1],self._mpo[n],self._rb,init)
                opt=init-dt*Hx
                
                tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                
                Hmat=mf.HAproductZeroSiteMat(self._L[n-1],self._mpo[n],tensor,self._rb,'right',mat)
                
                mat=mat+dt*Hmat
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)

            self._cmps._mats[n]=np.copy(mat)
            self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            if n==(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            if n<(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._R[self._cmps._N-n-2],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            return e



    def __optimizetest__(self,n,direction,tol,ncv,mode='AR',Ndiag=4,landelta=1E-10,dt=0.0):
        #stdout.write('.')
        #stdout.flush()
        if direction>0:
            self._cmps.__position__(n+1)
            init=self._cmps.__wavefnct__('l')

        if direction<0:
            self._cmps.__position__(n)
            init=self._cmps.__wavefnct__('r')
            
        if n==0:
            if mode=='AR':
                e,opt=mf.eigsh(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
            if mode=='LAN':
                lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                es,opts=lan.__simulate__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],self._cmps._Q[n],self._cmps._R[n],self._cmps._mats[self._cmps._position],self._cmps._dx[n],direction,self._verbose)
                e=es[0]
                opt=np.copy(opts[0])
            if mode=='LANTEV':
                lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                e,opt=lan.__evolve__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')

        
        if (n>0)&(n<self._cmps._N-1):
            if mode=='AR':
                e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,tol,1,ncv)
            if mode=='LAN':
                lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],self._cmps._Q[n],self._cmps._R[n],self._cmps._mats[self._cmps._position],self._cmps._dx[n],direction,self._verbose)
                e=es[0]
                opt=np.copy(opts[0])
            if mode=='LANTEV':
                lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')

        
        if n==(self._cmps._N-1):
            if mode=='AR':            
                e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._rb,init,tol,1,ncv)
            if mode=='LAN':
                lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._rb,self._cmps._Q[n],self._cmps._R[n],self._cmps._mats[self._cmps._position],self._cmps._dx[n],direction,self._verbose)
                e=es[0]
                opt=np.copy(opts[0])
            if mode=='LANTEV':
                lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=landelta,deltaEta=tol,dtype=self._cmps._dtype)
                e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._rb,init,dt)


        if direction>0:        
            tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
            #tensor,mat=mf.prepareTensor(opt,direction=1)
            Z=np.trace(mat.dot(herm(mat)))
            mat=mat/np.sqrt(Z)
            self._cmps._mats[n+1]=np.copy(mat)
            self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            
            if n==0:
                self._L[n]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            if n>0:
                self._L[n]=np.copy(mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            return e

        if direction<0:        
            tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
            #tensor,mat=mf.prepareTensor(opt,direction=-1)
            Z=np.trace(mat.dot(herm(mat)))
            mat=mat/np.sqrt(Z)

            self._cmps._mats[n]=np.copy(mat)
            self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
            
            if n==(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            if n<(self._cmps._N-1):
                self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._R[self._cmps._N-n-2],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
            return e


    def __simulate__(self,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond=True,regauge=False,save=100,verbosity=1\
                     ,plot=0):
        self._start=0
        self._end=self._cmps._N-1-self._start

        D=self._cmps._D
        dens=np.zeros(self._cmps._N)
        self._cmps.__position__(0)
        self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)
        self._verbose=False
        converged=False

        
        e=np.random.rand(1)*10000
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)

        UCenergyold=0.0
        UCenergy=0.0

        if os.path.isfile('arnoldienergies'+self._filename+'.npy'):
            itarnoldienergies=np.load('arnoldienergies'+self._filename+'.npy')
        elif not os.path.isfile('arnoldienergies'+self._filename+'.npy'):
            itarnoldienergies=np.zeros(0)


        if skipsecond==True:
            self._end=self._cmps._N

        while not converged:
            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos) and Nmaxlanczos>0:
                    mode='LAN'
                    tol=lanczostol
                    ncv=lanczosncv
                if (itlan>=Nmaxlanczos) and Nmaxlanczosimagtime>0:
                    mode='LANTEV'
                    tol=lanczostol
                    ncv=lanczosncv

            #sweep through the chain
            for n in range(self._cmps._N):
                e=self.__optimize__(n,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=dt,verbosity=0,optimize=True)
            self._cmps.__position__(self._cmps._N)

            if regauge==False:
                if par%2==0:
                    if it>0:
                        if verbosity>0:
                            stdout.write("\r              Running DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e-eold)/self._cmps._L,itar,\
                                                                                                                               itlan,self._cmps._dx[0]))
                            stdout.flush()
            if regauge==True:
                if par%2==0:
                    if verbosity>0:
                        stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan,self._cmps._dx[0]))
                        stdout.flush()
            if skipsecond==False:
                for n in range(self._end,self._start,-1):
                    e=self.__optimize__(n,direction=-1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,verbosity=verbosity)
                if regauge==False:
                    if par%2==0:
                        if it>0:
                            if verbosity>0:
                                stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e-eold)/self._cmps._L,itar,\
                                                                                                                                   itlan,self._cmps._dx[0]))
                                stdout.flush()
                if regauge==True:
                    if par%2==0:
                        if verbosity>0:
                            stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan,self._cmps._dx[0]))
                            stdout.flush()


            if it>0 and (par%2==0):
                itarnoldienergies=np.append(itarnoldienergies,[(e-eold)/self._cmps._L])

            
            if (plot!=0) and (it%plot==0):
                self._cmps.__position__(self._cmps._N)
                lams=[]
                for n in range(self._cmps._N):
                    U,lam,V=np.linalg.svd(self._cmps._mats[n])
                    lams.append(lam)
                    
                    

                dens=dcmps.getLiebLinigerDens(self._cmps)
                denssq=dcmps.getLiebLinigerDensDens(self._cmps)
                energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)

                i0=int(np.floor(np.random.rand(1)*self._cmps._D))
                i1=int(np.floor(np.random.rand(1)*self._cmps._D))
                #i1=int(np.floor(np.random.rand(1)*self._cmps._D))

                cmpstemp=self._cmps.__copy__()
                cmpstemp.__regauge__(True,tol=1E-12,ncv=30)
                plt.figure(55)
                plt.clf()
                plt.title('Re Q_c[{0},{1}]'.format(i0,i1))
                plt.plot(cmpstemp._xm-cmpstemp._xb[0],(cmpstemp.__data__(i0,i1)[0,:]),'.')
                plt.figure(551)
                plt.clf()
                plt.title('Im Q_c[{0},{1}]'.format(i0,i1))
                plt.plot(cmpstemp._xm-cmpstemp._xb[0],np.imag(cmpstemp.__data__(i0,i1)[0,:]),'.')

                plt.figure(66)
                plt.clf()
                plt.title('Re R_c[{0},{1}]'.format(i0,i1))
                plt.plot(cmpstemp._xm-cmpstemp._xb[0],(cmpstemp.__data__(i0,i1)[1,:]),'.')
                plt.figure(661)
                plt.clf()
                plt.title('Im R_c[{0},{1}]'.format(i0,i1))
                plt.plot(cmpstemp._xm-cmpstemp._xb[0],np.imag(cmpstemp.__data__(i0,i1)[1,:]),'.')


                
                plt.figure(2)
                if it%4==0:
                    plt.clf()
                plt.title('particle density')
                plt.plot(self._cmps._xm-self._cmps._xb[0],dens)
                print(np.sum(dens)*self._cmps._dx[0])
                plt.figure(3)
                if it%4==0:
                    plt.clf()
                #plt.title('double occupation')
                #plt.plot(range(self._cmps._N),denssq)
                #
                #plt.figure(4)
                #if it%4==0:
                #    plt.clf()
                #plt.title('energy density')
                #plt.plot(range(self._cmps._N),energy,'x',range(self._cmps._N,2*self._cmps._N),energy,'x')

                plt.figure(5)
                plt.clf()
                plt.title('Re Q[{0},{1}]'.format(i0,i1))
                plt.plot(self._cmps._xm-self._cmps._xb[0],(self._cmps.__data__(i0,i1)[0,:]),'.')
                #plt.figure(51)
                #plt.clf()
                #plt.title('Im Q[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],np.imag(self._cmps.__data__(i0,i1)[0,:]),'.')

                plt.figure(6)
                plt.clf()
                plt.title('Re R[{0},{1}]'.format(i0,i1))
                plt.plot(self._cmps._xm-self._cmps._xb[0],(self._cmps.__data__(i0,i1)[1,:]),'.')
                #plt.figure(61)
                #plt.clf()
                #plt.title('Im R[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],np.imag(self._cmps.__data__(i0,i1)[1,:]),'.')

                #plt.figure(7)
                #if it%4==0:
                #    plt.clf()
                #plt.title('energy density')
                #plt.plot(self._cmps._xm-self._cmps._xb[0],energy-energyold)

                plt.figure(8)
                plt.clf()
                plt.title('lams')
                plt.semilogy(range(len(lams)),lams,'x')
                
                energyold=np.copy(energy)
                plt.draw()
                plt.show()
                plt.pause(0.01)



    
                #plt.figure(2)
                #if it%4==0:
                #    plt.clf()
                #plt.title('particle density')
                #plt.plot(self._cmps._xm-self._cmps._xb[0],dens,'o')
                #
                #plt.figure(3)
                #if it%4==0:
                #    plt.clf()
                #
                #plt.title('double occupation')
                #plt.plot(self._cmps._xm-self._cmps._xb[0],denssq,'o')
                #
                #plt.figure(4)
                #if it%4==0:
                #    plt.clf()
                #
                #plt.title('energy density')
                #plt.plot(self._cmps._xm-self._cmps._xb[0],energy,'o')

                #plt.figure(5)
                #plt.clf()
                #plt.title('Q[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[0,:],'o')
                #
                #plt.figure(6)
                #plt.clf()
                #plt.title('R[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[1,:],'o')
                plt.draw()
                plt.show()


            par=par+1
            self._cmps.__position__(self._cmps._N)
            #get the new ansatz mps:
            Q=[]
            R=[]
            for n in range(int(self._cmps._N/2),self._cmps._N):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))
            #bring center site to center
            self._cmps.__position__(int(self._cmps._N/2))
            connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))


            #get the new boundaries:
            for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

            self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

            #get the other half of the ansatz mps
            self._cmps.__position__(0)
            for n in range(int(self._cmps._N/2)):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))

            for n in range(self._cmps._N):
                self._cmps._Q[n]=np.copy(Q[n])
                self._cmps._R[n]=np.copy(R[n])
            
            self._cmps._position=int(self._cmps._N/2)
            self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])

            self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
            for n in range(self._cmps._N):
                self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

            self._cmps._connector=np.copy(connector)
            self._cmps.__position__(0)
            #self._cmps.__position__(self._cmps._N)
            if regauge==True:
                self._cmps.__regauge__(True)

            self._cmps.__position__(0)
            unit=self._cmps._connector.dot(self._cmps._mats[0])
            unitdconv=np.linalg.norm(unit.dot(herm(unit))-np.eye(unit.shape[0]))
            mf.patchmpo(self._mpo,int(self._cmps._N/2))
            munew=np.zeros((len(mu)))
            for s in range(int(len(mu)/2)):
                munew[s]=mu[int(len(mu)/2)+s]
            for s in range(int(len(mu)/2),len(mu)):
                munew[s]=mu[s-int(len(mu)/2)]
            
            mu=np.copy(munew)

            if regauge==True:
                self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)                

            self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
            #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
            if (((it+1)%save)==0)&(par%2==0):
                self._cmps.__save__(self._filename)
                np.save('arnoldienergies'+self._filename,itarnoldienergies)
                #np.save('energydens'+self._filename,itenergydens)

            if it<Nmaxarnoldi:
                itar=itar+1
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos):
                    itlan=itlan+1
                if (itlan>=Nmaxlanczos):
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
            it=it+1
            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
                if np.abs(UCenergy-UCenergyold)<Econv:
                    if verbosity>0:
                        print ('Energy={0}, converged within {1}'.format(UCenergy,Econv))
                        converged=True
                UCenergyold=UCenergy
                eold=e
            if it>Nmax:
                if par%2==0:
                    converged=True


        self._cmps.__position__(0)
        self._cmps.__save__(self._filename)
        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        #np.save('energydens'+self._filename,itenergydens)

        return e




        #self._cmps.__regauge__(True)
        self._cmps.__grid_interpolate__(intgrid,which='cubic',tol=1E-10,ncv=100)
        print ('saving file {0}:'.format(self._filename))
        self._cmps.save(self._filename)
        dens=dcmps.getLiebLinigerDens(self._cmps)
        denssq=dcmps.getLiebLinigerDensDens(self._cmps)
        energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)
        plt.figure(10)
        plt.title('particle density')
        plt.plot(self._cmps._xm-self._cmps._xb[0],dens)
        
        plt.figure(11)
        plt.title('double occupation')
        plt.plot(self._cmps._xm[0:-1]-self._cmps._xb[0],denssq[0:-1])
        
        plt.figure(12)
        plt.title('energy density')
        plt.plot(self._cmps._xm-self._cmps._xb[0],energy)
        plt.draw()
        plt.show()
        #raw_input('waiting for input in engines')

        return e


    def __simulate_grid__(self,optgrid,fullopt,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond=True,regauge=False,save=100,\
                        verbosity=1,plot=0):
        #assert(grid[0]==0)
        #assert(grid[-1]==(self._cmps._N-1))
        grida=np.copy(optgrid)
        gridind=0
        assert(self._cmps._N%2==0)
        while(optgrid[gridind]<int(self._cmps._N/2)):
            gridind=gridind+1
        gridb=np.append(optgrid[gridind::]-int(self._cmps._N/2),optgrid[:gridind]+int(self._cmps._N/2))
        grid=grida

        D=self._cmps._D
        dens=np.zeros(self._cmps._N)
        self._cmps.__position__(0)

        self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)

        if grid[0]==0:
            self._cmps.__position__(grid[0])
        if grid[0]>0:
            self._cmps.__position__(grid[0]+1)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)
        self._verbose=False
        converged=False
        itplot=1
        
        e=np.random.rand(1)*10000
        recalc=1
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)

        UCenergyold=0.0
        UCenergy=0.0

        if os.path.isfile('arnoldienergies'+self._filename+'.npy'):
            itarnoldienergies=np.load('arnoldienergies'+self._filename+'.npy')
        elif not os.path.isfile('arnoldienergies'+self._filename+'.npy'):
            itarnoldienergies=np.zeros(0)


        gridopt=False
        while not converged:
            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos) and Nmaxlanczos>0:
                    mode='LAN'
                    tol=lanczostol
                    ncv=lanczosncv
                if (itlan>=Nmaxlanczos) and Nmaxlanczosimagtime>0:
                    mode='LANTEV'
                    tol=lanczostol
                    ncv=lanczosncv

            #sweep through the chain
            for index in range(len(grid)):
                e=self.__optimize_grid__(index,grid,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=dt,verbosity=0,optimize=True)

            self._cmps.__position__(self._cmps._N)

            if regauge==False:
                if par%2==0:
                    if it>0:
                        if verbosity>0:
                            #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,(e-eold)/self._cmps._L,itar,itlan,itimag,dt))
                            stdout.write("\r              Running DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e-eold)/self._cmps._L,itar,\
                                                                                                                               itlan,self._cmps._dx[0]))
                            stdout.flush()
                    if it==0:
                        if verbosity>0:
                            #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
                            stdout.write("\r              Running DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan,self._cmps._dx[0]))
                            stdout.flush()

            if regauge==True:
                if par%2==0:
                    if verbosity>0:
                        stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan,self._cmps._dx[0]))
                        stdout.flush()
                        #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
            
            if skipsecond==False:
                for index in range(len(grid),-1,-1):
                    e=self.__optimize_grid__(index,grid,direction=-1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,verbosity=verbosity)


                if regauge==False:
                    if par%2==0:
                        if it>0:
                            if verbosity>0:
                                stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e-eold)/self._cmps._L,itar,\
                                                                                                                                   itlan,self._cmps._dx[0]))
                                stdout.flush()
                                #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,(e-eold),itar,itlan,itimag,dt))
                        if it==0:
                            if verbosity>0:
                                stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan\
                                                                                                                                   ,self._cmps._dx[0]))
                                stdout.flush()
                                #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
                if regauge==True:
                    if par%2==0:
                        if verbosity>0:
                            stdout.write("\rRunning DMRG at iteration %i, E=%.8f; itar=%i, itlan=%i, dx=%f" %(it,np.real(e),itar,itlan,self._cmps._dx[0]))
                            stdout.flush()
                            #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))

            if it>0 and (par%2==0):
                itarnoldienergies=np.append(itarnoldienergies,[(e-eold)/self._cmps._L])
            #if it==0:
            #    itarnoldienergies=np.append(itarnoldienergies,[e])
            
            if (plot!=0) and (it%plot==0):

                self._cmps.__position__(self._cmps._N)
                dens=dcmps.getLiebLinigerDens(self._cmps)
                denssq=dcmps.getLiebLinigerDensDens(self._cmps)
                energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)
                np.random.seed(1)
                i0=int(np.floor(np.random.rand(1)*self._cmps._D)) 
                i1=int(np.floor(np.random.rand(1)*self._cmps._D))
    
                plt.figure(2)
                if it%4==0:
                    plt.clf()
                plt.title('particle density')
                plt.plot(self._cmps._xm-self._cmps._xb[0],dens,'o')

                plt.figure(3)
                if it%4==0:
                    plt.clf()

                plt.title('double occupation')
                plt.plot(self._cmps._xm-self._cmps._xb[0],denssq,'o')

                plt.figure(4)
                if it%4==0:
                    plt.clf()

                plt.title('energy density')
                plt.plot(self._cmps._xm-self._cmps._xb[0],energy,'o')

                #plt.figure(5)
                #plt.clf()
                #plt.title('Q[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[0,:],'o')
                #
                #plt.figure(6)
                #plt.clf()
                #plt.title('R[{0},{1}]'.format(i0,i1))
                #plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[1,:],'o')
                plt.draw()
                plt.show()

            par=par+1
            self._cmps.__position__(self._cmps._N)
            #get the new ansatz mps:
            Q=[]
            R=[]
            for n in range(int(self._cmps._N/2),self._cmps._N):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))
            #bring center site to center
            self._cmps.__position__(int(self._cmps._N/2))
            connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))


            #get the new boundaries:
            for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

            self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

            #get the other half of the ansatz mps
            self._cmps.__position__(0)
            for n in range(int(self._cmps._N/2)):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))

            for n in range(self._cmps._N):
                self._cmps._Q[n]=np.copy(Q[n])
                self._cmps._R[n]=np.copy(R[n])
            

            self._cmps._position=int(self._cmps._N/2)
            self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])


            self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
            for n in range(self._cmps._N):
                self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

            self._cmps._connector=np.copy(connector)
            self._cmps.__position__(0)
            self._cmps.__position__(self._cmps._N)
            if regauge==True:
                self._cmps.__regauge__(True)

            self._cmps.__position__(0)
            mf.patchmpo(self._mpo,int(self._cmps._N/2))
            munew=np.zeros((len(mu)))
            for s in range(int(len(mu)/2)):
                munew[s]=mu[int(len(mu)/2)+s]
            for s in range(int(len(mu)/2),len(mu)):
                munew[s]=mu[s-int(len(mu)/2)]
            
            mu=np.copy(munew)


            if ((it+1)%2==1):
                grid=np.copy(gridb)
            
            if ((it+1)%2==0):
                grid=np.copy(grida)

            if (it+1)%fullopt==0:
                grid=np.arange(self._cmps._N)

            
            self._cmps.__position__(0)
            if grid[0]==0:
                self._cmps.__position__(grid[0])
            if grid[0]>0:
                self._cmps.__position__(grid[0]+1)

            if regauge==True:
                self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)                

            self._R=mf.getR(self._cmps.__toMPS__(),self._mpo,self._rb)
            self._L[0]=mf.addLayer(self._lb,self._cmps.__tensor__(0),self._mpo[0],self._cmps.__tensor__(0),1)
            for n in range(1,grid[0]+1):
                self._L[n]=mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1)

            if (((it+1)%save)==0)&(par%2==0):
                self._cmps.__save__(self._filename)
                np.save('arnoldienergies'+self._filename,itarnoldienergies)
                #np.save('energydens'+self._filename,itenergydens)

            if it<Nmaxarnoldi:
                itar=itar+1
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos):
                    itlan=itlan+1
                if (itlan>=Nmaxlanczos):
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
            it=it+1
            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
                if np.abs(UCenergy-UCenergyold)<Econv:
                    if verbosity>0:
                        print ('Energy={0}, converged within {1}'.format(UCenergy,Econv))
                        converged=True
                UCenergyold=UCenergy
                eold=e
            if it>Nmax:
                if par%2==0:
                    converged=True

        self._cmps.__save__(self._filename)
        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        #np.save('energydens'+self._filename,itenergydens)

        return e



    def __simulatehomogeneous__(self,lamconv,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond,save=100,verbosity=0,optit=1,plotit=0,cqr=False):

        self._start=0
        self._end=self._cmps._N-1-self._start
        D=self._cmps._D
        dens=np.zeros(self._cmps._N)

        self._cmps.__position__(0)
        self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)
        self._verbose=False
        if Nmax>0:
            converged=False
        if Nmax==0:
            converged=True

        itplot=1
        
        e=np.random.rand(1)*10000
        recalc=1
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)
        lamold=np.zeros(self._cmps._D)
        UCenergyold=0.0
        UCenergy=0.0
        itarnoldienergies=np.zeros(0)
        itenergydens=np.zeros(0)

        if skipsecond==True:
            self._end=self._cmps._N


        while converged==False:
            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos):
                    mode='LAN'
                    tol=lanczostol
                    ncv=lanczosncv
                if (itlan>=Nmaxlanczos):
                    mode='LAN'
                    tol=lanczostol
                    ncv=lanczosncv

            #sweep through the chain
            if it%optit==0:
                opt=True
            elif (it%optit)!=0:
                opt=False

            for n in range(self._start,self._end):
                e=self.__optimize__(n,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=dt,verbosity=verbosity,optimize=opt)


            #U=self._cmps._mats[-1].dot(self._cmps._U).dot(self._cmps._connector)
            #print U.dot(herm(U))
            if it%10==0:
                U,lam,V=np.linalg.svd(self._cmps._mats[1])
                if np.abs((lam[-1]-lamold[-1])/lam[-1])<lamconv:
                    converged=True
                lamold=np.copy(lam)

            if plotit%200==0:
                #print np.abs((lam[-1]-lamold[-1])/lam[-1])
                plt.figure(100)
                if (plotit)%4000==0:
                    plt.clf()

                plt.semilogy(plotit/10.0,[lam],'o')
                plt.draw()
                plt.show()

            #if it==0:
            #    #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
            #    print ('iteration {0}, E[{1}]={2}; cmps._dx[0]={3}'.format(it,n,e,self._cmps._dx[0]))
            self._cmps.__position__(self._cmps._N,cqr=cqr)
            if par%2==0:
                if it>0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}, dx[0]={7}'.format(it,n,(e-eold)/self._cmps._L,itar,itlan,itimag,dt,self._cmps._dx[0]))
                if it==0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
            
            if it%1==0:
                par=par+1
                self._cmps.__position__(self._cmps._N,cqr=cqr)
                #get the new ansatz mps:
                Q=[]
                R=[]
                for n in range(int(self._cmps._N/2),self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(int(self._cmps._N/2),cqr=cqr)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))

                #get the new boundaries:
                for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

                self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

                #get the other half of the ansatz mps
                self._cmps.__position__(0,cqr=cqr)
                for n in range(int(self._cmps._N/2)):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))

                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=int(self._cmps._N/2)
                self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])

                self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(0,cqr=cqr)
                #self._cmps.__position__(self._cmps._N)
                mf.patchmpo(self._mpo,int(self._cmps._N/2))
                munew=np.zeros((len(mu)))
                for s in range(int(len(mu)/2)):
                    munew[s]=mu[int(len(mu)/2)+s]
                for s in range(int(len(mu)/2),len(mu)):
                    munew[s]=mu[s-int(len(mu)/2)]
                mu=np.copy(munew)


            self._cmps.__position__(0,cqr=cqr)
            self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
            #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
            if (((it+1)%save)==0)&(par%2==0):
                #print 'saving'
                #print self._filename
                self._cmps.__save__(self._filename)
                np.save('arnoldienergies'+self._filename,itarnoldienergies)
                np.save('energydens'+self._filename,itenergydens)

            if it<Nmaxarnoldi:
                itar=itar+1
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos):
                    itlan=itlan+1
                if (itlan>=Nmaxlanczos):
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
            it=it+1
            plotit=plotit+1
            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
            #    if np.abs(UCenergy-UCenergyold)<Econv:
            #        if verbosity>0:
            #            print 'Energy={0}, converged within {1}'.format(UCenergy,Econv)
            #        #if par%2==0:
            #        converged=True
                UCenergyold=UCenergy
                eold=e
            if it>Nmax:
                #if par%2==0:
                converged=True
        self._cmps.__save__(self._filename)
        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        np.save('energydens'+self._filename,itenergydens)

        return e,plotit





    def __simulatehomogeneous4__(self,lb,rb,lamconv,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol1,lanczostol2,change,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond,save=100,verbosity=0,optit=1,plotit=0,cqr=False,timeEvolve=False):

        self._start=0
        self._end=self._cmps._N-1-self._start
        D=self._cmps._D
        dens=np.zeros(self._cmps._N)
        self._cmps.__position__(self._cmps._N)
        energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)



        self._cmps.__position__(0)
        self._lb=np.copy(lb)
        self._rb=np.copy(rb)
        #self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)

        self._verbose=False
        if Nmax>0:
            converged=False
        if Nmax==0:
            converged=True

        itplot=1
        
        e=np.random.rand(1)*10000
        recalc=1
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)
        lamold=np.zeros(self._cmps._D)
        UCenergyold=0.0
        UCenergy=0.0
        itarnoldienergies=np.zeros(0)
        itenergydens=np.zeros(0)

        if skipsecond==True:
            self._end=self._cmps._N

        opt=True
        if Nmaxarnoldi>0:
            mode='AR'
            tol=arnolditol
            ncv=arnoldincv

        if (Nmaxarnoldi==0):
            if (Nmaxlanczos>0):
                mode='LAN'
                tol=lanczostol2
                ncv=lanczosncv
            if (Nmaxlanczos==0):
                if (Nmaxlanczosimagtime>0):
                    mode='LANTEV'
                    tol=lanczostol2
                    ncv=lanczosncv
                if (Nmaxlanczosimagtime==0):
                    return

        ev=timeEvolve
        while converged==False:
            ncv=lanczosncv
            if it<change:
                tol=lanczostol1
            if it>=change:
                tol=lanczostol2

            #sweep through the chain
            #self._end=0
            #if it<changeev:
            #    ev=timeEvolve
            #if it>=changeev:
            #    ev=False
            for n in range(self._start,self._end):
                if ev==False:
                    e=self.__optimize__(n,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=dt,verbosity=verbosity,optimize=opt)
                if ev==True:
                    e=self.__evolvehomogeneous__(n,direction=1,dt=dt,optimize=opt)

            #U=self._cmps._mats[-1].dot(self._cmps._U).dot(self._cmps._connector)
            #print U.dot(herm(U))
            if it%10==0:
                U,lam,V=np.linalg.svd(self._cmps._mats[1])
                if np.abs((lam[-1]-lamold[-1])/lam[-1])<lamconv:
                    converged=True
                lamold=np.copy(lam)

            if plotit%500==0:
                #print np.abs((lam[-1]-lamold[-1])/lam[-1])
                plt.figure(100)
                #if (plotit)%20000==0:
                #    plt.clf()

                plt.semilogy(plotit/10.0,[lam],'o')
                plt.draw()
                plt.show()

            #if it==0:
            #    #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
            #    print ('iteration {0}, E[{1}]={2}; cmps._dx[0]={3}'.format(it,n,e,self._cmps._dx[0]))
            #self._cmps.__positionSVD__(self._cmps._N,'u')
            self._cmps.__position__(self._cmps._N)
            energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)
            if par%2==0:
                if it>0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; <h[{1}]>={8}, itar={3}, itlan={4}, itimag={5}, dt={6}, dx[0]={7}'.format(it,n,(e-eold)/self._cmps._L,itar,itlan,itimag,dt,self._cmps._dx[0],energy))
                if it==0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; <h[{1}]>={8}, itar={3}, itlan={4}, itimag={5}, dt={6}, dx[0]={7}'.format(it,n,e,itar,itlan,itimag,dt,self._cmps._dx[0],energy))



            #if it>=(Nmax-2):
            #    self._cmps.__position__(0,cqr=cqr)
            #    self._cmps.__save__(self._filename)

            if ((it+1)%save)==0:
                self._cmps.__save__(self._filename)
                np.save('arnoldienergies'+self._filename,itarnoldienergies)
                np.save('energydens'+self._filename,itenergydens)
            
            if it%1==0:
                par=par+1
                self._cmps.__position__(self._cmps._N,cqr=cqr)
                #get the new ansatz mps:
                Q=[]
                R=[]
                for n in range(int(self._cmps._N/2),self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(int(self._cmps._N/2),cqr=cqr)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))

                #get the new boundaries:
                for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

                self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

                #get the other half of the ansatz mps
                self._cmps.__position__(0,cqr=cqr)
                for n in range(int(self._cmps._N/2)):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))

                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=int(self._cmps._N/2)
                self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])

                self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(0,cqr=cqr)
                #self._cmps.__position__(self._cmps._N)
                mf.patchmpo(self._mpo,int(self._cmps._N/2))
                munew=np.zeros((len(mu)))
                for s in range(int(len(mu)/2)):
                    munew[s]=mu[int(len(mu)/2)+s]
                for s in range(int(len(mu)/2),len(mu)):
                    munew[s]=mu[s-int(len(mu)/2)]
                
                mu=np.copy(munew)

                self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
                #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
                    # 

            it=it+1

            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
                itar=itar+1
                

            if it>=Nmaxarnoldi:
                if (itlan>=Nmaxlanczos):
                    if itimag==0:
                        mode='LANTEV'
                        tol=lanczostol2
                        ncv=lanczosncv
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1

                if (itlan<Nmaxlanczos):
                    if itlan==0:
                        mode='LAN'
                        tol=lanczostol2
                        ncv=lanczosncv

                    itlan=itlan+1


            plotit=plotit+1
            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
                if np.abs(UCenergy-UCenergyold)<Econv:
            #        if verbosity>0:
                    print ('Energy={0}, converged within {1}'.format(UCenergy,Econv))
            #        #if par%2==0:
                    converged=True
                UCenergyold=UCenergy
                eold=e
            if it>=Nmax:
                #if par%2==0:
                converged=True
        self._cmps.__save__(self._filename)

        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        np.save('energydens'+self._filename,itenergydens)

        return self._cmps,e,it



    #this routine is used for preconditioning the cMPS calculations in the continuum
    def __LiebLinigerPreconditioner__(self,lb,rb,lamconv,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,mu,mass,g,save=100,verbosity=0,cqr=False):

        self._start=0
        self._end=self._cmps._N-1-self._start
        D=self._cmps._D
        dens=np.zeros(self._cmps._N)
        self._cmps.__position__(self._cmps._N)
        
        energy=dcmps.getLiebLinigerEDens(self._cmps,np.zeros(2),mass,g)

        self._cmps.__position__(0)
        self._lb=np.copy(lb)
        self._rb=np.copy(rb)


        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)

        self._verbose=False
        if Nmax>0:
            converged=False
        if Nmax==0:
            converged=True

        e=np.random.rand(1)*10000
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)
        lamold=np.zeros(self._cmps._D)
        UCenergyold=0.0
        UCenergy=0.0

        if Nmaxarnoldi>0:
            mode='AR'
            tol=arnolditol
            ncv=arnoldincv

        if (Nmaxarnoldi==0):
            if (Nmaxlanczos>0):
                mode='LAN'
                tol=lanczostol
                ncv=lanczosncv
            if (Nmaxlanczos==0):
                return

        while converged==False:
            ncv=lanczosncv
            tol=lanczostol

            for n in range(self._start,self._end):
                e=self.__optimize__(n,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=0.0,verbosity=0,optimize=True)

            if it%10==0:
                U,lam,V=np.linalg.svd(self._cmps._mats[1])
                if np.abs((lam[-1]-lamold[-1])/lam[-1])<lamconv:
                    converged=True
                lamold=np.copy(lam)

            self._cmps.__position__(self._cmps._N)
            energy=dcmps.getLiebLinigerEDens(self._cmps,np.zeros(2),mass,g)
            if par%2==0:
                if it>0:
                    if verbosity>0:
                        stdout.write("\rRunning DMRG at iteration %i, E=%.8f; <h>=[%.8f,%.8f], itar=%i, itlan=%i, dx=%f" %(it,np.real(e-eold)/self._cmps._L,np.real(energy[0]),np.real(energy[1]),itar,\
                            itlan,self._cmps._dx[0]))
                        stdout.flush()
                if it==0:
                    if verbosity>0:
                        stdout.write("\rRunning DMRG at iteration %i, E=%.8f; <h>=[%.8f,%.8f], itar=%i, itlan=%i, dx=%f" %(it,np.real(e),np.real(energy[0]),np.real(energy[1]),itar,itlan,self._cmps._dx[0]))
                        stdout.flush()

            if ((it+1)%save)==0:
                self._cmps.__save__(self._filename)
            
            if it%1==0:
                par=par+1
                self._cmps.__position__(self._cmps._N,cqr=cqr)
                #get the new ansatz mps:
                Q=[]
                R=[]

                for n in range(int(int(self._cmps._N/2)),self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(int(self._cmps._N/2),cqr=cqr)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))

                #get the new boundaries:
                for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

                self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

                #get the other half of the ansatz mps
                self._cmps.__position__(0,cqr=cqr)
                for n in range(int(self._cmps._N/2)):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))

                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=int(self._cmps._N/2)
                self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])

                self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(0,cqr=cqr)

                mf.patchmpo(self._mpo,int(self._cmps._N/2))

                self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)

            it=it+1

            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
                itar=itar+1
                

            if it>=Nmaxarnoldi:
                if (itlan>=Nmaxlanczos):
                    print ('DMRG did not converge to desired accuracy of {0} after {1} Arnoldi and {2} Lanczos iterations'.format(Econv,itar,itlan))
                    converged=True
                if (itlan<Nmaxlanczos):
                    if itlan==0:
                        mode='LAN'
                        tol=lanczostol
                        ncv=lanczosncv

                    itlan=itlan+1


            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
                if np.abs(UCenergy-UCenergyold)<Econv:
                    if verbosity>0:
                        print ('Energy={0}, converged within {1}'.format(UCenergy,Econv))
                    converged=True
                UCenergyold=UCenergy
                eold=e
            if it>=Nmax:
                converged=True
        self._cmps.__save__(self._filename)

        return self._cmps,e,it
        

        
    def __simulatehomogeneous_bond__(self,lb,rb,Econv,Nmax,tol,ncv,mu,mass,g,shift=True,save=100,verbosity=0,cqr=False):

        self._cmps.__position__(1)
        self._lb=np.copy(lb)
        self._rb=np.copy(rb)
        #self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)
        self._R=[]
        self._L=[]
        
        for n in range(self._cmps._N):
            self._R.append(None)
            self._L.append(None)
        self._L[0]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(0),self._mpo[0],self._cmps.__tensor__(0),1))
        self._R[0]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(1),self._mpo[1],self._cmps.__tensor__(1),-1))

        self._verbose=False
        if Nmax>0:
            converged=False
        if Nmax==0:
            converged=True

        e=np.random.rand(1)*10000
        eold=np.random.rand(1)*10000
        it=0
        UCenergyold=0.0
        UCenergy=0.0

        while converged==False:
            self._cmps.__position__(self._cmps._N,cqr=cqr)
            energy1=dcmps.getLiebLinigerEDens(self._cmps,0.0*np.ones(2),mass,g)
            #print 'energy before opt: ',energy
            self._cmps.__position__(1,cqr=cqr)
            init=np.copy(self._cmps._mats[self._cmps._position]*herm(self._cmps._mats[self._cmps._position]))
            #u,lam,v=np.linalg.svd(init)
            
            lan=LZ.LanczosEngine(Ndiag=4,nmax=ncv,numeig=1,delta=1E-14,deltaEta=tol,dtype=self._cmps._dtype)
            es,opts=lan.__simulateBondSimpleCMPSLiebLiniger__(self._L[0][:,:,0],self._cmps._Q[0],self._cmps._R[0],self._R[0][:,:,-1],self._cmps._Q[1],self._cmps._R[1],mass,mu,g,self._cmps._dx[0],init)
            #es,opts=lan.__simulateBondSimple__(self._L[0],self._R[0],init)
            e=es[0]
            opt=opts[0]
            #e,opt=mf.cmpseigshbondsimple(self._L[0][:,:,0],self._cmps._Q[0],self._cmps._R[0],self._R[0][:,:,-1],self._cmps._Q[1],self._cmps._R[1],init,mass,mu,g,self._cmps._dx[0],\
            #                                  tolerance=tol,numvecs=1,numcv=ncv) 
            #e,opt=mf.eigshbondsimple(self._L[0],self._R[0],init,tolerance=tol,numvecs=1,numcv=ncv)
            self._cmps._mats[n]=np.copy(opt)
            #self._cmps.__position__(self._cmps._N,cqr=cqr)
            self._L[1]=np.copy(mf.addLayer(self._L[0],self._cmps.__tensor__(1),self._mpo[1],self._cmps.__tensor__(1),1))
            self._cmps.__position__(self._cmps._N)
            #self._cmps.__positionSVD__(self._cmps._N,'u')
            energy=dcmps.getLiebLinigerEDens(self._cmps,0.0*np.ones(2),mass,g)
            #print 'energy after opt: ',energy
            #raw_input()
            if it>0:
                print ('E[{0}]={1}; before: <h[{0}]>={4}, after: <h[{0}]>={2},dx[0]={3}'.format(it,(e-eold)/np.sum(self._cmps._dx),energy,self._cmps._dx[0],energy1))
            if it==0:
                print ('E[{0}]={1}; before: <h[{0}]>={4}, after: <h[{0}]>={2},dx[0]={3}'.format(it,e,energy,self._cmps._dx[0],energy1))


            if shift==True:
                Q=[]
                R=[]
                for n in range(int(self._cmps._N/2),self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(int(self._cmps._N/2),cqr=cqr)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))
                
                #get the new boundaries:
                for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)
                
                self._lb=np.copy(self._L[int(self._cmps._N/2)-1])
                
                #get the other half of the ansatz mps
                self._cmps.__position__(0,cqr=cqr)
                for n in range(int(self._cmps._N/2)):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                
                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=int(self._cmps._N/2)
                self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])
                
                self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0
                
                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(1,cqr=cqr)
                
                self._L[0]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(0),self._mpo[0],self._cmps.__tensor__(0),1))
                self._R[0]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(1),self._mpo[1],self._cmps.__tensor__(1),-1))


            it=it+1
            UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
            if np.abs(UCenergy-UCenergyold)<Econv:
                print ('Energy={0}, converged within {1}'.format(UCenergy,Econv))
                converged=True
            UCenergyold=UCenergy
            eold=e
            if it>=Nmax:
                converged=True
        self._cmps.__save__(self._filename)
        return self._cmps,e,it




    def __simulatehomogeneous8__(self,lb,rb,lamconv,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond,save=100,verbosity=0,\
                                 optit=1,plotit=0,cqr=False):

        self._start=0
        self._end=self._cmps._N-1-self._start
        D=self._cmps._D
        dens=np.zeros(self._cmps._N)

        self._cmps.__position__(0)
        self._lb=np.copy(lb)
        self._rb=np.copy(rb)
        #self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)
        self._verbose=False
        if Nmax>0:
            converged=False
        if Nmax==0:
            converged=True

        itplot=1
        
        e=np.random.rand(1)*10000
        recalc=1
        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)
        lamold=np.zeros(self._cmps._D)
        UCenergyold=0.0
        UCenergy=0.0
        itarnoldienergies=np.zeros(0)
        itenergydens=np.zeros(0)

        if skipsecond==True:
            self._end=self._cmps._N

        opt=True
        if Nmaxarnoldi>0:
            mode='AR'
            tol=arnolditol
            ncv=arnoldincv

        if (Nmaxarnoldi==0):
            if (Nmaxlanczos>0):
                mode='LAN'
                tol=lanczostol
                ncv=lanczosncv
            if (Nmaxlanczos==0):
                if (Nmaxlanczosimagtime>0):
                    mode='LANTEV'
                    tol=lanczostol
                    ncv=lanczosncv
                if (Nmaxlanczosimagtime==0):
                    return 

        while converged==False:
            ncv=lanczosncv

            #sweep through the chain
            for n in range(self._start,self._end):
                e=self.__optimize__(n,direction=1,tol=tol,ncv=ncv,mode=mode,Ndiag=Ndiag,landelta=lanczosdelta,dt=dt,verbosity=verbosity,optimize=opt)


            #U=self._cmps._mats[-1].dot(self._cmps._U).dot(self._cmps._connector)
            #print U.dot(herm(U))
            if it%10==0:
                U,lam,V=np.linalg.svd(self._cmps._mats[1])
                if np.abs((lam[-1]-lamold[-1])/lam[-1])<lamconv:
                    converged=True
                lamold=np.copy(lam)

            if plotit%500==0:
                #print np.abs((lam[-1]-lamold[-1])/lam[-1])
                plt.figure(100)
                if (plotit)%20000==0:
                    plt.clf()

                plt.semilogy(plotit/10.0,[lam],'o')
                plt.draw()
                plt.show()

            #if it==0:
            #    #print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))
            #    print ('iteration {0}, E[{1}]={2}; cmps._dx[0]={3}'.format(it,n,e,self._cmps._dx[0]))
            self._cmps.__position__(self._cmps._N,cqr=cqr)
            energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)
            if par%2==0:
                if it>0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; <h[{1}]>={8}, itar={3}, itlan={4}, itimag={5}, dt={6}, dx[0]={7}'.format(it,n,(e-eold)/self._cmps._L,itar,itlan,itimag,dt,self._cmps._dx[0],energy[0]))
                if it==0:
                    if verbosity>0:
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))


            #if it>=(Nmax-2):
            #    self._cmps.__position__(0,cqr=cqr)
            #    self._cmps.__save__(self._filename)

            if ((it+1)%save)==0:
                np.save('arnoldienergies'+self._filename,itarnoldienergies)
                np.save('energydens'+self._filename,itenergydens)
            
            if it%1==0:
                par=par+1
                self._cmps.__position__(self._cmps._N,cqr=cqr)
                #get the new ansatz mps:
                Q=[]
                R=[]
                for n in range(int(self._cmps._N/2),self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(int(self._cmps._N/2),cqr=cqr)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[int(self._cmps._N/2)]))

                #get the new boundaries:
                for site in range(self._cmps._N-1,int(self._cmps._N/2)-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

                self._lb=np.copy(self._L[int(self._cmps._N/2)-1])

                #get the other half of the ansatz mps
                self._cmps.__position__(0,cqr=cqr)
                for n in range(int(self._cmps._N/2)):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))

                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=int(self._cmps._N/2)
                self._cmps._mats[int(self._cmps._N/2)]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])

                self._cmps._xb=np.copy(np.append(self._cmps._xb[int(self._cmps._N/2)::],self._cmps._L+self._cmps._xb[1:int(self._cmps._N/2)+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(0,cqr=cqr)
                #self._cmps.__position__(self._cmps._N)
                mf.patchmpo(self._mpo,int(self._cmps._N/2))
                munew=np.zeros((len(mu)))
                for s in range(int(len(mu)/2)):
                    munew[s]=mu[int(len(mu)/2)+s]
                for s in range(int(len(mu)/2),len(mu)):
                    munew[s]=mu[s-int(len(mu)/2)]
                
                mu=np.copy(munew)

                self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
                #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
                    # 

            it=it+1

            if it<Nmaxarnoldi:
                mode='AR'
                tol=arnolditol
                ncv=arnoldincv
                itar=itar+1
                

            if it>=Nmaxarnoldi:
                if (itlan>=Nmaxlanczos):
                    if itimag==0:
                        mode='LANTEV'
                        tol=lanczostol
                        ncv=lanczosncv
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1

                if (itlan<Nmaxlanczos):
                    if itlan==0:
                        mode='LAN'
                        tol=lanczostol
                        ncv=lanczosncv

                    itlan=itlan+1


            plotit=plotit+1
            if par%2==0:
                UCenergy=(e-eold)/self._cmps._L#self._cmps._dx[0]
            #    if np.abs(UCenergy-UCenergyold)<Econv:
            #        if verbosity>0:
            #            print 'Energy={0}, converged within {1}'.format(UCenergy,Econv)
            #        #if par%2==0:
            #        converged=True
                UCenergyold=UCenergy
                eold=e
            if it>=Nmax:
                #if par%2==0:
                converged=True
        self._cmps.__save__(self._filename)

        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        np.save('energydens'+self._filename,itenergydens)



        return e,plotit
    #this is a deprecated rouinte (is working though)
    def __simulateinfinitesequential__(self,Econv,Nmax,Nmaxarnoldi,arnolditol,arnoldincv,Nmaxlanczos,lanczosdelta,lanczosncv,lanczostol,Nmaxlanczosimagtime,dts,mu,mass,g,skipsecond,regauge=True,\
                                       save=100,proj=True,storeenergies=True):
        

        self._start=0
        self._end=self._cmps._N-1-self._start
        D=self._cmps._D
        dens=np.zeros(self._cmps._N)

        #getBoundaryHams regauges the cmps!!
        #cdl.plots(self._cmps,3,0)

        self._cmps.__position__(0,False)
        self._lb,self._rb,lbound,rbound=dcmps.getBoundaryHams(self._cmps,self._mpo)
        #cdl.plots(self._cmps,6,0)

        self._R=mf.getR(self._cmps.__toMPS__(connect=None),self._mpo,self._rb)
        self._L=mf.getL(self._cmps.__toMPS__(connect=None),self._mpo,self._lb)
        #self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
        #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)

        converged=False
        e0=100000.0
        dmrgenergy=[]

        itplot=1
        rec=False
        mod=10

        
        e=np.random.rand(1)*10000
        nplot=0
        recalc=1

        eold=np.random.rand(1)*10000

        it=0
        itar=0
        itlan=0
        itimag=0
        dtindex=0
        dt=dts[dtindex]
        Ndiag=4
        par=0
        energy=np.zeros(self._cmps._N)
        energyold=np.zeros(self._cmps._N)
        #itregauge=100000000
        UCenergyold=0.0
        UCenergy=0.0
        unitconv=None
        Econv=None
        itarnoldienergies=np.zeros(0)
        itenergydens=np.zeros(0)
        if skipsecond==True:
            self._end=self._cmps._N

        while not converged:
            #if it>200:
            #    regauge=True
            if it%10==0:
                calcObs=True
            elif it%10!=0:
                calcObs=False
            for n in range(self._start,self._end):
                self._cmps.__position__(n+1)
                stdout.write('.')
                stdout.flush()


                init=self._cmps.__wavefnct__('l')
                #init=np.copy(self._cmps.__tensor__(n))
                #init[:,:,0]=init[:,:,0].dot(self._cmps._mats[n+1])
                #init[:,:,1]=init[:,:,1].dot(self._cmps._mats[n+1])
                
                if n==0:
                    if it<Nmaxarnoldi:
                        e,opt=mf.eigsh(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,arnolditol,1,arnoldincv)
                    if (it>=Nmaxarnoldi):
                        lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                        if (itlan<Nmaxlanczos):
                            es,opts=lan.__simulate__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init)
                            e=es[0]
                            opt=np.copy(opts[0])
                        #if (itlan>=Nmaxlanczos):
                        #    e,opt=lan.__evolve__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                        #    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')


                if (n>0)&(n<self._N-1):
                    if it<Nmaxarnoldi:
                        e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,arnolditol,1,arnoldincv)
                    if (it>=Nmaxarnoldi):
                        lan=LZ.LanczosEngine(Ndiag=4,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                        if (itlan<Nmaxlanczos):
                            es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init)
                            e=es[0]
                            opt=np.copy(opts[0])
                        #if (itlan>=Nmaxlanczos):
                        #    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                        #    #ebondr,optbondr=lan.__evolveBond__(A,dt,initmat,'right')

                if n==(self._N-1):
                    if it<Nmaxarnoldi:

                        e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._rb,init,arnolditol,1,arnoldincv)
                    if (it>=Nmaxarnoldi):
                        lan=LZ.LanczosEngine(Ndiag=4,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                        if (itlan<Nmaxlanczos):
                            es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._rb,init)
                            e=es[0]
                            opt=np.copy(opts[0])
                        #if (itlan>=Nmaxlanczos):
                        #    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._rb,init,dt)

                tensor,mat=mf.prepareTensorfixA0(opt,direction=1)
                #tensor,mat=mf.prepareTensor(opt,direction=1)
                Z=np.trace(mat.dot(herm(mat)))
                mat=mat/np.sqrt(Z)
                self._cmps._mats[n+1]=np.copy(mat)
                self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
                            
                if n==0:
                    self._L[n]=np.copy(mf.addLayer(self._lb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
                if n>0:
                    self._L[n]=np.copy(mf.addLayer(self._L[n-1],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),1))
            
            self._cmps.__position__(self._cmps._N)

            if regauge==False:
                if par%2==0:
                    #print ('iteration {0}, E[{1}]={2}; ,<e>[{1}]={3}, sum <e>={8}, <e>/<n>={9}, itar={4}, itlan={5}, itimag={6}, dt={7}'.format(it,n,(e-eold),energy[n-1],itar,itlan,itimag,dt,\
                    #                                                                                                                                    np.sum(energy)/len(energy)/self._cmps._L,\
                    #                                                                                                                                np.sum(energy)/np.sum(dens)))
                    if it>0:
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,(e-eold),itar,itlan,itimag,dt))
                    if it==0:
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))

            if regauge==True:
                if par%2==0:
                    print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}'.format(it,n,e,itar,itlan,itimag,dt))


            if storeenergies==True:
                if it>0:
                    itarnoldienergies=np.append(itarnoldienergies,[(e-eold)])
                if it==0:
                    itarnoldienergies=np.append(itarnoldienergies,[e])

                    

            if skipsecond==False:
                for n in range(self._end,self._start,-1):
                    self._cmps.__position__(n)
                    stdout.write('.')
                    stdout.flush()

                    

                    init=self._cmps.__wavefnct__('r')
                    #init=np.copy(self._cmps.__tensor__(n))
                    #init[:,:,0]=self._cmps._mats[n].dot(init[:,:,0])
                    #init[:,:,1]=self._cmps._mats[n].dot(init[:,:,1])

                    if n==0:
                        if it<Nmaxarnoldi:
                            e,opt=mf.eigsh(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,arnolditol,1,arnoldincv)
                        if (it>=Nmaxarnoldi):
                            lan=LZ.LanczosEngine(Ndiag=Ndiag,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                            if (itlan<Nmaxlanczos):
                                es,opts=lan.__simulate__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init)
                                e=es[0]
                                opt=np.copy(opts[0])
                            #if (itlan>=Nmaxlanczos):
                            #    e,opt=lan.__evolve__(self._lb,self._mpo[n],self._R[self._cmps._N-2-n],init,dt)

                    if (n>0)&(n<self._N-1):
                        if it<Nmaxarnoldi:
                            e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,arnolditol,1,arnoldincv)
                        if (it>=Nmaxarnoldi):
                            lan=LZ.LanczosEngine(Ndiag=4,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                            if (itlan<Nmaxlanczos):
                                es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init)
                                e=es[0]
                                opt=np.copy(opts[0])
                            #if (itlan>=Nmaxlanczos):
                            #    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._R[self._cmps._N-2-n],init,dt)
                            
                    if n==(self._N-1):
                        if it<Nmaxarnoldi:
                            e,opt=mf.eigsh(self._L[n-1],self._mpo[n],self._rb,init,arnolditol,1,arnoldincv)
                        if (it>=Nmaxarnoldi):
                            lan=LZ.LanczosEngine(Ndiag=4,nmax=lanczosncv,numeig=1,delta=lanczosdelta,deltaEta=lanczostol,dtype=self._cmps._dtype)
                            if (itlan<Nmaxlanczos):
                                es,opts=lan.__simulate__(self._L[n-1],self._mpo[n],self._rb,init)
                                e=es[0]
                                opt=np.copy(opts[0])
                            #if (itlan>=Nmaxlanczos):
                            #    e,opt=lan.__evolve__(self._L[n-1],self._mpo[n],self._rb,init,dt)

                    tensor,mat=mf.prepareTensorfixA0(opt,direction=-1)
                    #tensor,mat=mf.prepareTensor(opt,direction=-1)
                    Z=np.trace(mat.dot(herm(mat)))
                    mat=mat/np.sqrt(Z)

                    self._cmps._mats[n]=np.copy(mat)
                    self._cmps._Q[n],self._cmps._R[n]=dcmps.fromMPSmat(tensor,self._cmps._dx[n])
                    
                    if n==(self._cmps._N-1):
                        self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._rb,self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
                    if n<(self._cmps._N-1):
                        self._R[self._cmps._N-1-n]=np.copy(mf.addLayer(self._R[self._cmps._N-n-2],self._cmps.__tensor__(n),self._mpo[n],self._cmps.__tensor__(n),-1))
                if regauge==False:
                    
                    if par%2==0:
                        if it>0:
                            print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}, |U herm(U)-1|={7}, Econv={8}'.format(it,n,(e-eold),itar,itlan,itimag,dt,unitconv,Econv))
                        if it==0:
                            print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}, |U herm(U)-1|={7}, Econv={8}'.format(it,n,e,itar,itlan,itimag,dt,unitconv,Econv))

                if regauge==True:
                    if par%2==0:
                        unitconv=None
                        Econv=None
                        print ('iteration {0}, E[{1}]={2}; itar={3}, itlan={4}, itimag={5}, dt={6}, |U herm(U)-1|={7}, Econv={8}'.format(it,n,e,itar,itlan,itimag,dt,unitconv,Econv))


            #if calcObs==True:
            #    dens=cdl.getLiebLinigerDens(self._cmps)
            #    denssq=cdl.getLiebLinigerDensDens(self._cmps)
            #    energy=cdl.getLiebLinigerEDens(self._cmps,mu,mass,g)
            #    if storeenergies==True:
            #        itenergydens=np.append(itenergydens,[np.sum(energy)/len(energy)/self._cmps._L])
            #    if par%2==0:
            #        print ('sum <e>={0}'.format(np.sum(energy)/len(energy)/self._cmps._L))
           
            if it%10==0:
                #self._cmps.__position__(0)
                self._cmps.__position__(self._cmps._N)
                #cmps=mf.copycmps(self._cmps)
                #cmps.__position__(0)
                #cmps.__position__(cmps._N)
                #cmps.__regauge__(True)
                #nplot=cdl.plots(self._cmps,0,nplot,'bla',self._filename)
                #if calcObs==False:
                dens=dcmps.getLiebLinigerDens(self._cmps)
                denssq=dcmps.getLiebLinigerDensDens(self._cmps)
                energy=dcmps.getLiebLinigerEDens(self._cmps,mu*0.0,mass,g)


                i0=int(np.floor(np.random.rand(1)*self._cmps._D))
                #i1=int(np.floor(np.random.rand(1)*self._cmps._D))
                i1=i0
                #i0=int(np.floor(np.random.rand(1)*6)+20) 
                #i1=int(np.floor(np.random.rand(1)*6)+20) 


                plt.figure(2)
                if it%4==0:
                    plt.clf()
                plt.title('particle density')
                plt.plot(range(self._cmps._N),dens)

                plt.figure(3)
                if it%4==0:
                    plt.clf()
                plt.title('double occupation')
                plt.plot(range(self._cmps._N),denssq)

                plt.figure(4)
                if it%4==0:
                    plt.clf()
                plt.title('energy density')
                plt.plot(range(self._cmps._N),energy,'x',range(self._cmps._N,2*self._cmps._N),energy,'x')

                plt.figure(5)
                plt.clf()
                plt.title('Q[{0},{1}]'.format(i0,i1))
                plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[0,:],'o')

                plt.figure(6)
                plt.clf()
                plt.title('R[{0},{1}]'.format(i0,i1))
                plt.plot(self._cmps._xm-self._cmps._xb[0],self._cmps.__data__(i0,i1)[1,:],'o')


                plt.figure(7)
                if it%4==0:
                    plt.clf()
                plt.title('energy density')
                plt.plot(self._cmps._xm-self._cmps._xb[0],energy-energyold)
                energyold=np.copy(energy)
                plt.draw()
                plt.show()
                plt.pause(0.01)

            if it%recalc==0:
                par=par+1
                self._cmps.__position__(self._cmps._N)
                #cmpspatched=cdl.patchcmps(self._cmps,self._cmps._N/2)
                #get the new ansatz mps:
                Q=[]
                R=[]
                for n in range(self._cmps._N/2,self._cmps._N):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))
                #bring center site to center
                self._cmps.__position__(self._cmps._N/2)
                connector=np.copy(np.linalg.pinv(self._cmps._mats[self._cmps._N/2]))

                #get the new boundaries:
                #self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
                for site in range(self._cmps._N-1,self._cmps._N/2-1,-1):
                    self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

                #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
                self._lb=np.copy(self._L[self._cmps._N/2-1])
                #self._rb=np.copy(self._R[self._cmps._N/2-1])
                #get the other half of the ansatz mps
                self._cmps.__position__(0)
                for n in range(self._cmps._N/2):
                    Q.append(np.copy(self._cmps._Q[n]))
                    R.append(np.copy(self._cmps._R[n]))

                for n in range(self._cmps._N):
                    self._cmps._Q[n]=np.copy(Q[n])
                    self._cmps._R[n]=np.copy(R[n])
                
                self._cmps._position=self._cmps._N/2
                self._cmps._mats[self._cmps._N/2]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])
                #self._cmps._mats[self._cmps._N/2]=self._cmps._mats[self._cmps._N].dot(self._cmps._connector).dot(self._cmps._mats[0])

                self._cmps._xb=np.copy(np.append(self._cmps._xb[self._cmps._N/2::],self._cmps._L+self._cmps._xb[1:self._cmps._N/2+1]))
                for n in range(self._cmps._N):
                    self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                    self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0

                self._cmps._connector=np.copy(connector)
                self._cmps.__position__(0)
                #self._cmps.__position__(self._cmps._N)
                if regauge==True:
                    #cdl.regaugecmps(self._cmps,delta=1E-8,itmax=100)
                    self._cmps.__regauge__()
                    #cdl.cmpsGauging(self._cmps,gauge='left',initial=None,nmaxit=100000,tol=1E-10,it=it)

                self._cmps.__position__(0)
                unit=self._cmps._connector.dot(self._cmps._mats[0])
                unitconv=np.linalg.norm(unit.dot(herm(unit))-np.eye(unit.shape[0]))
                mf.patchmpo(self._mpo,self._cmps._N/2)
                munew=np.zeros((len(mu)))
                for s in range(int(len(mu)/2)):
                    munew[s]=mu[int(len(mu)/2)+s]
                for s in range(int(len(mu)/2),len(mu)):
                    munew[s]=mu[s-int(len(mu)/2)]

                mu=np.copy(munew)

                if regauge==True:
                    self._lb,self._rb,lbound,rbound=dcmpsgetBoundaryHams(self._cmps,self._mpo)                
                #regauge=False
                self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
                #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)

                if (((it+1)%save)==0)&(par%2==0):
                    #print 'saving'
                    #print self._filename
                    self._cmps.__save__(self._filename)
                    #pos=self._cmps._position
                    #self._cmps.__position__(self._cmps._N)
                    #dens=cdl.getLiebLinigerDens(self._cmps)
                    #self._cmps.__position__(pos)
                    #plt.figure(100)
                    #plt.clf()
                    #plt.title('saving at iteration {0}'.format(it))
                    #plt.plot(dens)
                    #plt.draw()
                    #plt.show()

                    np.save('arnoldienergies'+self._filename,itarnoldienergies)
                    np.save('energydens'+self._filename,itenergydens)
                    
            if it<Nmaxarnoldi:
                itar=itar+1
            if it>=Nmaxarnoldi:
                if (itlan<Nmaxlanczos):
                    itlan=itlan+1
                if (itlan>=Nmaxlanczos):
                    if itimag<Nmaxlanczosimagtime:
                        itimag=itimag+1
                    if itimag>=Nmaxlanczosimagtime:
                        if dtindex<(len(dts)-1):
                            dtindex=dtindex+1
                            itimag=0
                        dt=dts[dtindex]
            it=it+1
            if par%2==0:
                UCenergy=e-eold
                Econv=np.abs(UCenergy-UCenergyold)
                #if np.abs(UCenergy-UCenergyold)<Econv:
                #    print 'Energy={0}, convergerd within {1}'.format(UCenergy,Econv)
                #    #if par%2==0:
                #    converged=True
                UCenergyold=UCenergy
                eold=e
            if it>Nmax:
                #if par%2==0:
                converged=True
        if par%2==0:
            self._cmps.__position__(0)
            self._cmps.__save__(self._filename)
        else:
            self._cmps.__position__(self._cmps._N)
            #cmpspatched=cdl.patchcmps(self._cmps,self._cmps._N/2)
            #get the new ansatz mps:
            Q=[]
            R=[]
            for n in range(self._cmps._N/2,self._cmps._N):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))
            #bring center site to center
            self._cmps.__position__(self._cmps._N/2)
            connector=np.copy(np.linalg.pinv(self._cmps._mats[self._cmps._N/2]))

            #get the new boundaries:
            #self._R=mf.getR(self._cmps.__topureMPS__(),self._mpo,self._rb)
            for site in range(self._cmps._N-1,self._cmps._N/2-1,-1):
                self._rb=mf.addLayer(self._rb,self._cmps.__tensor__(site),self._mpo[site],self._cmps.__tensor__(site),-1)

            #self._L=mf.getL(self._cmps.__topureMPS__(),self._mpo,self._lb)
            self._lb=np.copy(self._L[self._cmps._N/2-1])
            #self._rb=np.copy(self._R[self._cmps._N/2-1])
            #get the other half of the ansatz mps
            self._cmps.__position__(0)
            for n in range(self._cmps._N/2):
                Q.append(np.copy(self._cmps._Q[n]))
                R.append(np.copy(self._cmps._R[n]))

            for n in range(self._cmps._N):
                self._cmps._Q[n]=np.copy(Q[n])
                self._cmps._R[n]=np.copy(R[n])
            
            self._cmps._position=self._cmps._N/2
            self._cmps._mats[self._cmps._N/2]=self._cmps._mats[self._cmps._N].dot(self._cmps.__connection__(True)).dot(self._cmps._mats[0])
            #self._cmps._mats[self._cmps._N/2]=self._cmps._mats[self._cmps._N].dot(self._cmps._connector).dot(self._cmps._mats[0])

            self._cmps._xb=np.copy(np.append(self._cmps._xb[self._cmps._N/2::],self._cmps._L+self._cmps._xb[1:self._cmps._N/2+1]))
            for n in range(self._cmps._N):
                self._cmps._dx[n]=self._cmps._xb[n+1]-self._cmps._xb[n]
                self._cmps._xm[n]=self._cmps._xb[n]+self._cmps._dx[n]/2.0
            self._cmps._connector=np.copy(connector)
            self._cmps.__position__(0)
            self._cmps.__save__(self._filename)

        np.save('arnoldienergies'+self._filename,itarnoldienergies)
        np.save('energydens'+self._filename,itenergydens)

        return e


#regular tdvp for homogeneous cMPS; deprecated, used HomogeneouscMPSEngine instead
class TDVPcMPSEngine:
    def __init__(self, filename):
        self._filename=filename
        
    def __simulate__(self,Q,R,dx,dt,dtype,mu0,inter,mass,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100):

        D=np.shape(Q)[0]
        rold=np.random.rand(D*D)*1.0
        kold=np.random.rand(D*D)*1.0
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)

        it=0
        converged=False

        norms=np.zeros(0,dtype=dtype)
        kinpluspot=np.zeros(0,dtype=dtype)
        meanDenss=np.zeros(0,dtype=dtype)
        warmup=True
        while converged==False:
            it=it+1
            lam,Ql,Rl,Qr,Rr=cmf.regauge_old(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),nmaxit=10000,tol=regaugetol)
            #if it%100000==0:
            #    plt.figure(101)
            #    plt.semilogy(it+it0,[lam],'o')
            #
            #    plt.figure(102)
            #    plt.plot(it+it0,[np.diag(Ql)],'o')
            #
            #    plt.figure(103)
            #    plt.plot(it+it0,[np.diag(Rl)],'o')
            #    plt.draw()
            #    plt.show()
            lnorm=np.linalg.norm(Ql+herm(Ql)+herm(Rl).dot(Rl)+dx*herm(Ql).dot(Ql))/D
            rnorm=np.linalg.norm(Qr+herm(Qr)+Rr.dot(herm(Rr))+dx*Qr.dot(herm(Qr)))/D
            
            lamold=np.copy(lam)
        
            f=np.zeros((D,D),dtype=dtype)
            l=np.eye(D)
            r=np.diag(lam)
            ih=cmf.homogeneousdfdxLiebLiniger(Ql,Rl,Ql,Rl,dx,f,mu0,mass,inter,direction=1)

            ihprojected=-(ih-np.tensordot(ih,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            kleft=cmf.inverseTransferOperator(Ql,Rl,dx,l,r,ihprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            
            meanh=1.0/(2.0*mass)*np.trace(comm(Ql,Rl).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl))))+inter*np.trace(Rl.dot(Rl).dot(np.diag(lam**2)).dot(herm(Rl)).dot(herm(Rl)))
            meandensity=np.trace(Rl.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(Rl)))

            xopt=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl,Qr,Rr,kleft,np.diag(lam),mass,mu0,inter,dx,direction=1)
            xdot=xopt.dot(np.diag(1.0/lam))
            normxopt=np.linalg.norm(xopt)

            if dx>1E-6:
                Q=Ql+dt*np.linalg.inv(np.eye(D)+dx*herm(Ql)).dot(herm(Rl)).dot(xdot)
            if dx<=1E-6:
                Q=Ql+dt*herm(Rl).dot(xdot)

            R=Rl-dt*xdot

            norms=np.append(norms,normxopt)
            kinpluspot=np.append(kinpluspot,meanh)
            meanDenss=np.append(meanDenss,meandensity)

            print ('at iteration step {0}: norm(x) = {1},cmps h = {2},dx={3}, dt={4},lnorm={5},rnorm={6}'.format(it,normxopt,meanh,dx,dt,lnorm,rnorm))

            if it>itmax:
                converged=True
            if it%itsave==0:
                its=range(len(norms))
                np.save('norms'+self._filename,norms)
                np.save('kinpluspot'+self._filename,kinpluspot)
                np.save('meanDenss'+self._filename,meanDenss)
                #plt.figure(1)
                #plt.clf()
                #plt.semilogy(its,norms,'o')
                #
                #
                #plt.figure(2)
                #plt.clf()
                #plt.semilogy(its,kinpluspot,'o')
                #
                #plt.figure(3)
                #plt.clf()
                #plt.semilogy(its,meanDenss,'o')
                #plt.draw()
                #plt.show()

        np.save('norms'+self._filename,norms)
        np.save('kinpluspot'+self._filename,kinpluspot)
        np.save('meanDenss'+self._filename,meanDenss)

        lam,Ql,Rl,Qr,Rr=cmf.regauge_old(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),nmaxit=10000,tol=regaugetol)
        return lam,Ql,Rl,Qr,Rr,it,normxopt,it+it0


#for homogeneous states
class HomogeneousLiebLinigercMPSEngine:

    def __init__(self,filename,Ql,Rl,nxdots,dt,dts,dx,dtype,mu,inter,Delta,mass,itreset=10,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-6,acc=1E-4,itsave=100,verbosity=1,warmuptol=1E-6\
                 ,rescalingfactor=2.0,normtolerance=0.1,initnormtolerance=0.1,initthresh=0.01,numeig=5,ncv=100,Nmaxlgmres=100,outerklgmres=20,innermlgmres=30,nlcgupperthresh=1E-16,\
                 nlcglowerthresh=1E-100,nlcgreset=10,stdereset=3,nlcgnormtol=0.0,single_layer=False,pinv=1E-14):
        self._pinv=pinv
        self._eigenaccuracy=acc
        self._D=np.shape(Ql)[0]
        self._filename=filename
        self._Ql=np.copy(Ql)
        self._Rl=np.copy(Rl)
        self._Qr=np.copy(Ql)
        self._Rr=np.copy(Rl)

        if nxdots!=None:
            self._nxdots=np.copy(nxdots)
        elif nxdots==None:
            self._nxdots=nxdots

        self._dt=dt
        self._dt_=dt
        self._dts=np.copy(dts)

        self._dx=dx
        self._dtype=dtype
        self._mu=mu
        self._Delta=Delta
        self._inter=inter
        self._mass=mass
        self._itmax=itmax
        self._regaugetol=regaugetol
        self._lgmrestol=lgmrestol
        self._epsilon=epsilon
        self._itsave=itsave
        self._itreset=itreset
        self._single_layer_regauge=single_layer
        if self._nxdots!=None:
            assert((len(self._dts))==len(self._nxdots))
        self._verbosity=verbosity
        self._warmuptol=warmuptol
        self._factor=rescalingfactor
        self._normtol=normtolerance
        self._initnormtol=initnormtolerance
        self._initthresh=initthresh
        self._numeig=numeig
        self._ncv=ncv
        self._Nmaxlgmres=Nmaxlgmres
        self._outerklgmres=outerklgmres
        self._innermlgmres=innermlgmres
        self._nlcgupperthresh=nlcgupperthresh
        self._nlcglowerthresh=nlcglowerthresh
        self._nlcgreset=nlcgreset
        self._stdereset=stdereset
        self._nlcgnormtol=nlcgnormtol

        self._it=0
        self._warmup=True
        self._lamold=np.ones(self._D)/np.sqrt(self._D)
        self._Hl=np.eye(self._D)
        self._Hr=np.eye(self._D)
        self._reset=True
        self._it2=0
        self._Qlbackup=np.copy(self._Ql)
        self._Rlbackup=np.copy(self._Rl)
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    #eset all iterators and flags, so one can __load__() a file, and start a fresh simulation; don't forget to 
    #set the new parameters after rest, if you don't want to use the ones from the __load__()ed file
    def __reset__(self):
        self._it=0
        self._warmup=True
        #self._lamold=np.ones(self._D)/np.sqrt(self._D)
        #self._Hl=np.eye(self._D)
        #self._Hr=np.eye(self._D)
        self._reset=True
        self._it2=0
        #self._Qlbackup=np.copy(self._Ql)
        #self._Rlbackup=np.copy(self._Rl)
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        #self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        #self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    def __cleanup__(self):
        cwd=os.getcwd()
        if not os.path.exists('CHECKPOINT_'+self._filename):
            return
        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)


    #dump the simulation into a folder for later retrieval with ___load__()
    def __dump__(self):
        cwd=os.getcwd()
        #raw_input(not os.path.exists('CHECKPOINT_'+self._filename))
        if not os.path.exists('CHECKPOINT_'+self._filename):
            os.mkdir('CHECKPOINT_'+self._filename)

        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)
            os.mkdir('CHECKPOINT_'+self._filename)

        os.chdir('CHECKPOINT_'+self._filename)


        intparams=np.zeros(0,dtype=int)
        floatparams=np.zeros(0,dtype=float)
        intparams=np.append(intparams,self._D)
        intparams=np.append(intparams,self._itmax)
        intparams=np.append(intparams,self._itsave)
        intparams=np.append(intparams,self._it)
        intparams=np.append(intparams,self._it2)
        intparams=np.append(intparams,self._itreset)
        intparams=np.append(intparams,self._rescaledepth)
        intparams=np.append(intparams,self._verbosity)
        intparams=np.append(intparams,self._numeig)
        intparams=np.append(intparams,self._ncv)
        intparams=np.append(intparams,self._nlcgreset)
        intparams=np.append(intparams,self._stdereset)
        intparams=np.append(intparams,self._Nmaxlgmres)
        intparams=np.append(intparams,self._outerklgmres)
        intparams=np.append(intparams,self._innermlgmres)



        floatparams=np.append(floatparams,self._dt)
        floatparams=np.append(floatparams,self._dt_)
        floatparams=np.append(floatparams,self._dx)
        floatparams=np.append(floatparams,self._mu)
        floatparams=np.append(floatparams,self._inter)
        floatparams=np.append(floatparams,self._mass)
        floatparams=np.append(floatparams,self._regaugetol)
        floatparams=np.append(floatparams,self._lgmrestol)
        floatparams=np.append(floatparams,self._epsilon)
        floatparams=np.append(floatparams,self._normxopt)
        floatparams=np.append(floatparams,self._normxoptold)
        floatparams=np.append(floatparams,self._eigenaccuracy)
        floatparams=np.append(floatparams,self._warmuptol)
        floatparams=np.append(floatparams,self._factor)
        floatparams=np.append(floatparams,self._normtol)
        floatparams=np.append(floatparams,self._nlcgupperthresh)
        floatparams=np.append(floatparams,self._nlcglowerthresh)
        floatparams=np.append(floatparams,self._nlcgnormtol)
        floatparams=np.append(floatparams,self._initnormtol)
        floatparams=np.append(floatparams,self._initthresh)
        floatparams=np.append(floatparams,self._Delta)
        floatparams=np.append(floatparams,self._pinv)

        boolparams=np.empty((0),dtype=bool)
        boolparams=np.append(boolparams,self._warmup)        
        boolparams=np.append(boolparams,self._reset)        


        np.save('intparams',intparams)
        np.save('floatparams',floatparams)
        np.save('boolparams',boolparams)
        np.save('Ql',self._Ql)
        np.save('Rl',self._Rl)
        np.save('Qr',self._Qr)
        np.save('Rr',self._Rr)
        if self._nxdots!=None:
            np.save('nxdots',self._nxdots)
        if self._nxdots==None:
            np.save('nxdots',np.zeros(0))

        np.save('dts',self._dts)
        np.save('lamold',self._lamold)
        np.save('kleft',self._Hl)
        np.save('kright',self._Hr)
        np.save('Qlbackup',self._Qlbackup)
        np.save('Rlbackup',self._Rlbackup)
        np.save('norms',self._norms)
        np.save('totalEnergy',self._totalEnergy)
        np.save('kinpluspot',self._kinpluspot)
        np.save('meanDenss',self._meanDenss)
        np.save('Vlbackup',self._Vlbackup)
        np.save('Wlbackup',self._Wlbackup)
        os.chdir(cwd)
        
    #load a simulation from a folder named dirname
    def __load__(self,dirname):
        root=os.getcwd()
        os.chdir(dirname)
        
        intparams=np.load('intparams.npy')
        floatparams=np.load('floatparams.npy')
        boolparams=np.load('boolparams.npy')

        self._D=int(intparams[0])
        self._itmax=int(intparams[1])
        self._itsave=int(intparams[2])
        self._it=int(intparams[3])
        self._it2=int(intparams[4])
        self._itreset=int(intparams[5])
        self._rescaledepth=int(intparams[6])
        self._verbosity=int(intparams[7])
        self._numeig=int(intparams[8])
        self._ncv=int(intparams[9])
        self._nlcgreset=int(intparams[10])
        self._stdereset=int(intparams[11])
        self._Nmaxlgmres=int(intparams[12])
        self._outerklgmres=int(intparams[13])
        self._innermlgmres=int(intparams[14])


        self._dt=np.real(floatparams[0])
        self._dt_=np.real(floatparams[1])
        self._dx=np.real(floatparams[2])
        self._mu=np.real(floatparams[3])
        self._inter=np.real(floatparams[4])
        self._mass=np.real(floatparams[5])
        self._regaugetol=np.real(floatparams[6])
        self._lgmrestol=np.real(floatparams[7])
        self._epsilon=np.real(floatparams[8])
        self._normxopt=np.real(floatparams[9])
        self._normxoptold=np.real(floatparams[10])
        self._eigenaccuracy=np.real(floatparams[11])
        self._warmuptol=np.real(floatparams[12])
        self._factor=np.real(floatparams[13])
        self._normtol=np.real(floatparams[14])
        self._nlcgupperthresh=np.real(floatparams[15])
        self._nlcglowerthresh=np.real(floatparams[16])
        self._nlcgnormtol=np.real(floatparams[17])
        self._initnormtol=np.real(floatparams[18])
        self._initthresh=np.real(floatparams[19])
        self._Delta=np.real(floatparams[20])
        self._pinv=np.real(floatparams[21])
        self._warmup=boolparams[0]
        self._reset=boolparams[1]

        self._Ql=np.load('Ql.npy')
        self._Rl=np.load('Rl.npy')
        self._Qr=np.load('Qr.npy')
        self._Rr=np.load('Rr.npy')
        nxdots=np.load('nxdots.npy')

        if len(nxdots==0):
            self._nxdots=None

        if len(nxdots!=0):
            self._nxdots=np.copy(nxdots)

        self._dts=np.load('dts.npy')
        self._lamold=np.load('lamold.npy')
        self._Hl=np.load('kleft.npy')
        self._Hr=np.load('kright.npy')
        self._Qlbackup=np.load('Qlbackup.npy')
        self._Rlbackup=np.load('Rlbackup.npy')
        self._norms=np.load('norms.npy')
        self._totalEnergy=np.load('totalEnergy.npy')
        self._kinpluspot=np.load('kinpluspot.npy')
        self._meanDenss=np.load('meanDenss.npy')
        self._Vlbackup=np.load('Vlbackup.npy')
        self._Wlbackup=np.load('Wlbackup.npy')
        os.chdir(root)



    #simulates the regular tdvp for cMPS
    def __simulatetdvp__(self):
        converged=False
        f=np.zeros((self._D,self._D),dtype=self._dtype)
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))
        t1=time.time()
        while converged==False:
            self._it=self._it+1
            if self._it%100==0:
                t2=time.time()
                print('time for 100 iterations: {0}'.format(t2-t1))
                t1=t2

            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            self._lam,self._Ql,_Rl,self._Qr,_Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge(self._Ql,[self._Rl],self._dx,gauge='symmetric',\
                                                                                                       linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                       rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                       nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,pinv=self._pinv)
            self._Rl=_Rl[0]
            self._Rr=_Rr[0]

            #self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',\
            #                                                                                  initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
            #                                                                                  nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
            if self._verbosity==3:
                rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D
            self._lamold=np.copy(self._lam)


            ihl=cmf.homogeneousdfdxLiebLiniger(self._Ql,self._Rl,self._Ql,self._Rl,self._dx,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            
            self._Hl,numnum=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(self._Hl,self._D*self._D),\
                                                        tolerance=lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
        

            meanh=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                   self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
                   self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))

            meantotal=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                       self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
                       self._mu*np.trace(self._Rl.dot(np.diag(self._lam**2)).dot(herm(self._Rl)))+\
                       self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))

            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))


            Y=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(self._Ql,self._Rl,self._Qr,self._Rr,self._Hl,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta,self._dx,direction=1)
            Wl=Y.dot(np.diag(1.0/self._lam))
            self._normxopt=np.linalg.norm(Y)


            if self._dx>1E-6:
                Vl=-np.linalg.inv(np.eye(self._D)+self._dx*herm(self._Ql)).dot(herm(self._Rl)).dot(Wl)
            if self._dx<=1E-6:
                Vl=-herm(self._Rl).dot(Wl)

            if (self._normxopt-self._normxoptold)/self._normxopt>normtol and self._it>1:
                self._rescaledepth+=1
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)

                self._dt_/=self._factor
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)
                self._it2=0
                self._reset=False
                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}! '.format(self._normxopt,self._normxoptold,normtol))

            if (self._normxopt-self._normxoptold)/self._normxopt<=normtol or self._it==1:
                self._Vlbackup=np.copy(Vl)
                self._Wlbackup=np.copy(Wl)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)

                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)
                
                if self._reset==True:
                    self._it2=0
                    if self._rescaledepth>0:
                        self._rescaledepth-=1
                    if self._rescaledepth>0:
                        self._reset=False
                    if self._warmup==True:
                        self._dt_=self._dt/(self._factor**(self._rescaledepth))
                        if self._nxdots!=None:
                            if self._normxopt<self._nxdots[0]:
                                self._warmup=False
                    if self._warmup==False:
                        if self._nxdots!=None:
                            ind=0
                            while (self._normxopt<self._nxdots[ind]):
                                if ind==(len(self._nxdots)-1):
                                    self._dt_=self._dts[-1]/(self._factor**(self._rescaledepth))
                                    break
                                if self._nxdots[ind+1]<=self._normxopt:
                                    self._dt_=self._dts[ind]/(self._factor**(self._rescaledepth))
                                    break
                                if self._nxdots[ind+1]>self._normxopt:
                                    ind+=1
                        

                if self._reset==False:
                    self._it2+=1
                    if self._it2%self._itreset==0:
                        self._reset=True

                self._normxoptold=self._normxopt
                self._norms=np.append(self._norms,self._normxopt)
                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._kinpluspot=np.append(self._kinpluspot,meanh)
                self._meanDenss=np.append(self._meanDenss,meandensity)
                if self._verbosity==1:
                    #stdout.write("\rRuning DMRG at iteration %i, E=%f; <h>=[%f,%f], itar=%i, itlan=%i, dx=%f" %(it,e,energy[0],energy[1],itar,itlan,self._cmps._dx[0]))
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f " %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_))

                    #print ('{0}: norm(x) = {1}, <h> = {2}, <E>={3}, <n>={4}, dt={5}'.format(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_))
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, <h>/<n>**3=%.8f " %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_,
                                                                                                            np.real(meanh/meandensity**3)))

                    #print ('{0}: norm(x) = {1}, <h> = {2}, <E>={3}, <n>={4}, dt={5}, <h>/<n>**3={6}'.format(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_,\
                                                                                                            #np.real(meanh/meandensity**3)))
                if self._verbosity==3:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, <h>/<n>**3=%.8f, lnorm=%.8f, rnorm = %.8f " \
                                 %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_,np.real(meanh/meandensity**3),lnorm,rnorm))
                    

            stdout.flush()
            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True
            if self._it%self._itsave==0:
                self.__dump__()
                if self._verbosity>0:
                    print('   (checkpointing)   ')
                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('kinpluspot'+self._filename,self._kinpluspot)
                np.save('meanDenss'+self._filename,self._meanDenss)

                #plt.figure(1)
                #plt.clf()
                #plt.semilogy(its,norms,'o')
                #
                ##plt.figure()
                ##plt.semilogy(totalEnergy,'o')
                #
                #plt.figure(2)
                #plt.clf()
                #plt.semilogy(its,kinpluspot,'o')
                #
                #plt.figure(3)
                #plt.clf()
                #plt.semilogy(its,meanDenss,'o')
                #plt.draw()
                #plt.show()


        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('kinpluspot'+self._filename,self._kinpluspot)
        np.save('meanDenss'+self._filename,self._meanDenss)
        #self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
        #                                                          nmaxit=10000,tol=self._regauge;tol,numeig=self._numeig,ncv=self._ncv)
        self._lam,self._Ql,_Rl,self._Qr,_Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge(self._Ql,[self._Rl],self._dx,gauge='symmetric',\
                                                                                                   linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                   rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                   nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,pinv=self._pinv)
        self._Rl=_Rl[0]
        self._Rr=_Rr[0]

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt


    #new optimization for cMPS
    def __simulate__(self,singlelayernoise=1E-12):
        plt.ion()
        
        printnlcgmessage=True
        printstdemessage=True
        itbeta=0
        itstde=0
        dostde=False
        converged=False
        normstdvp=np.zeros(0)
        Vl=np.zeros((self._D,self._D),dtype=self._dtype)
        Wl=np.zeros((self._D,self._D),dtype=self._dtype)
        
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))

        f=np.zeros((self._D,self._D),dtype=self._dtype)
        t1=time.time()
        while converged==False:
            self._it=self._it+1
            if self._it%100==0:
                t2=time.time()
                print('time for 100 iterations: {0}'.format(t2-t1))
                t1=t2
            #print self._normxopt,self._eigenaccuracy
            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            if self._single_layer_regauge==False:

                #self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge_return_basis(self._Ql,self._Rl,self._dx,gauge='symmetric',\
                #                                                                                                   linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                #                                                                                                   rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                #                                                                                                   nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
                self._lam,self._Ql,_Rl,self._Qr,_Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge(self._Ql,[self._Rl],self._dx,gauge='symmetric',\
                                                                                               linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                               rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                               nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,pinv=self._pinv)
                self._Rl=_Rl[0]
                self._Rr=_Rr[0]

                
            if self._single_layer_regauge==True:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv=cmf.singleLayerRegauge_return_basis(self._Ql,self._Rl,self._dx,deltax=0.01,\
                                                                                                                    initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                    nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,noise=singlelayernoise)
                
                lNit=None
                rNit=None

            if self._it%self._itsave==0:
                self.__dump__()

                if self._verbosity>0:
                    #stdout.write('\r   (checkpointing)   ')
                    print('   (checkpointing)   ')

                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('normstdvp'+self._filename,normstdvp)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('kinpluspot'+self._filename,self._kinpluspot)
                np.save('meanDenss'+self._filename,self._meanDenss)


            if self._verbosity==3:
                if self._dx>=1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D
                elif self._dx<1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl))/self._D

            #ihl=cmf.homogeneousdfdxLiebLiniger(self._Ql,self._Rl,self._Ql,self._Rl,self._dx,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            #ihr=cmf.homogeneousdfdxLiebLiniger(self._Qr,self._Rr,self._Qr,self._Rr,self._dx,f,self._mu,self._mass,self._inter,self._Delta,direction=-1)
            mat=np.copy(self._Rl)

            ihl=cmf.homogeneousdfdxLiebLinigernodx(self._Ql,self._Rl,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            ihr=cmf.homogeneousdfdxLiebLinigernodx(self._Qr,self._Rr,f,self._mu,self._mass,self._inter,self._Delta,direction=-1)

            meantotal=np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))
            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            ihrprojected=-(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
            Hlregauged=np.transpose(herm(Glinv).dot(np.transpose(self._Hl)).dot(Glinv))

            self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(Hlregauged,self._D*self._D),\
                                                      tolerance=lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
            self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=np.reshape(self._Hr,self._D*self._D),\
                                                       tolerance=lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            meanh=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                   self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
                   self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))



            meanpsipsi=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsi=np.trace(self._Rl.dot(np.diag(self._lam**2)))

            meanpsidagpsidagpsipsi=np.trace(herm(self._Rl.dot(self._Rl)).dot(self._Rl).dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsidagpsidag=np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl)))

            
            meanK=np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))
            meaninter=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meantotal=1.0/(2.0*self._mass)*meanK+self._mu*meandensity+self._Delta*(meanpsipsi+meanpsidagpsidag)+self._inter*meaninter
            #print (meanpsidagpsidagpsipsi,(2*meandensity*meandensity+meanpsipsi*meanpsidagpsidag))            
            #meantotal=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
            #           self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
            #           self._mu*np.trace(self._Rl.dot(np.diag(self._lam**2)).dot(herm(self._Rl)))+\
            #           self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))
            #meantotal=cmf.HomogeneousLiebLinigerCMPS_epsilon_energy(self._Hl,self._Ql,self._Rl,self._Hr,self._Qr,self._Rr,np.diag(self._lam),self._mass,self._mu,self._inter)
            

            Vlam,Wlam=cmf.HomogeneousLiebLinigerGradient(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
                                                           self._Hl,self._Hr,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta)
            

            invlam=np.copy(1.0/self._lam)
            invlam[self._lam<self._pinv]=0.0
            Vl_=Vlam.dot(np.diag(invlam))
            Wl_=Wlam.dot(np.diag(invlam))


            Wlamtdvp=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(self._Ql,self._Rl,self._Qr,self._Rr,self._Hl,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta,self._dx,direction=1)
            normxopttdvp=np.linalg.norm(Wlamtdvp)
            
            self._normxopt=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(Wlam.dot(herm(Wlam))))
            self._dt_,self._rescaledepth,self._it2,self._reset,self._reject,self._warmup=utils.determineNewStepsize(self._dt_,self._dt,self._dts,self._nxdots,self._normxopt,\
                                                                                                                    self._normxoptold,normtol,self._warmup,self._it,self._rescaledepth,self._factor,\
                                                                                                                    self._it2,self._itreset,self._reset)

            if self._reject==True:
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                self._lam=np.copy(self._lamold)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normxopt,self._normxoptold,normtol))

            if self._reject==False:
                betanew,itstde,itbeta,dostde,printnlcgmessage,printstdemessage=utils.determineNonLinearCGBeta(self._nlcgupperthresh,self._nlcglowerthresh,self._nlcgnormtol,self._nlcgreset,self._normxopt,\
                                                                                                              self._normxoptold,self._it,itstde,self._stdereset,dostde,itbeta,printnlcgmessage,printstdemessage)

                    
                Vl=Vl_+betanew*Gl.dot(Vl).dot(Glinv)
                Wl=Wl_+betanew*Gl.dot(Wl).dot(Glinv)
                self._Vlbackup=np.copy(Vl_)
                self._Wlbackup=np.copy(Wl_)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                self._normxoptold=self._normxopt
                self._lamold=np.copy(self._lam)

                self._norms=np.append(self._norms,self._normxopt)
                normstdvp=np.append(normstdvp,normxopttdvp)
                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._kinpluspot=np.append(self._kinpluspot,meanh)
                self._meanDenss=np.append(self._meanDenss,meandensity)
                if self._verbosity==1:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, <h>\<n>**3 = %.8f, dt=% f, beta=%.3f" %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),np.real(meanh)/np.real(meandensity)**3,self._dt_,np.real(betanew)))
                
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f,<psi>=%.5f+i%.5f,  <psipsi>=%.5f+i%.5f, |<psipsi>|=%.5f, <psi*psi*>=%.5f+i%.5f,|<psi*psi*>|=%.5f, <psipsi>+<psi*psi*>=%.5f, <K>=%.5f, <I>=%.5f, dt=%.8f, beta=%.3f, <h>/<n>**3=%.8f " %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),np.real(meanpsi),np.imag(meanpsi),np.real(meanpsipsi),np.imag(meanpsipsi),np.abs(meanpsipsi),np.real(meanpsidagpsidag),np.imag(meanpsidagpsidag),np.abs(meanpsidagpsidag),np.real(meanpsipsi+meanpsidagpsidag),np.real(meanK),np.real(meaninter),self._dt_,np.real(betanew),np.real(meanh/meandensity**3)))

                
                if self._verbosity==3:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, beta=%.3f, <h>/<n>**3=%.8f, lnorm=%.8f, rnorm=%.8f, l Nlgmres=%i, r Nlgmres=%i, l regIt=%i, r regIt=%i" \
                                 %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),\
                                   self._dt_,np.real(betanew),np.real(meanh/meandensity**3),lnorm,rnorm,l_lgmresNit,r_lgmresNit,lNit,rNit))

                if self._verbosity==4:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <n>=%.8f, dt=%.8f, beta=%.3f, gamma=%.8f,  <h>/<n>**3=%.8f" \
                                 %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meandensity),\
                                   self._dt_,np.real(betanew),np.real(self._inter/meandensity),np.real(meanh/meandensity**3)))

                if self._verbosity==5:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <n>=%.8f, dt=%.8f, beta=%.3f, gamma=%.8f,  <h>/<n>**3=%.8f, regauetol=%.8f, lgmrestol=%.8f" \
                                 %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meandensity),\
                                   self._dt_,np.real(betanew),np.real(self._inter/meandensity),np.real(meanh/meandensity**3),rtol,lgmrestol))

                if self._verbosity==6:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <n>=%.8f, <psi*psi*psipsi>=%.8f, gamma=%.8f, <h>/<n>**3=%.8f" \
                                 %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meandensity),np.real(meanpsidagpsidagpsipsi),np.real(self._inter/meandensity),np.real(meanh/meandensity**3)))




            stdout.flush()
            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True

        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('kinpluspot'+self._filename,self._kinpluspot)
        np.save('meanDenss'+self._filename,self._meanDenss)
        if self._single_layer_regauge==False:
            #self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold),self._D*self._D),nmaxit=10000,\
            #                                                          tol=self._regaugetol,numeig=self._numeig,ncv=self._ncv)


            self._lam,self._Ql,_Rl,self._Qr,_Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge(self._Ql,[self._Rl],self._dx,gauge='symmetric',\
                                                                                                       linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                       rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                       nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,pinv=self._pinv)
            self._Rl=_Rl[0]
            self._Rr=_Rr[0]

        if self._single_layer_regauge==True:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.singleLayerRegauge(self._Ql,self._Rl,self._dx,deltax=0.01,initial=np.reshape(np.diag(self._lamold),self._D*self._D),\
                                                                                 nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt




    #new optimization for cMPS
    def __simulatediag__(self):
        single_layer_regauge=False
        printnlcgmessage=True
        printstdemessage=True
        itbeta=0
        itstde=0
        dostde=False
        converged=False
        normstdvp=np.zeros(0)
        Vl=np.zeros((self._D,self._D),dtype=self._dtype)
        Wl=np.zeros((self._D,self._D),dtype=self._dtype)
        
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))

        f=np.zeros((self._D,self._D),dtype=self._dtype)
        while converged==False:
            self._it=self._it+1
            #print self._normxopt,self._eigenaccuracy
            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            if single_layer_regauge==False:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge_return_basis(self._Ql,self._Rl,self._dx,gauge='symmetric',\
                                                                                                                   linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                                   rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                   nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
                
            if single_layer_regauge==True:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv=cmf.singleLayerRegauge_return_basis(self._Ql,self._Rl,self._dx,deltax=0.01,\
                                                                                                                    initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                    nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
                lNit=None
                rNit=None
                
            if self._verbosity==3:
                rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D


            ihl=cmf.homogeneousdfdxLiebLinigernodx(self._Ql,self._Rl,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            ihr=cmf.homogeneousdfdxLiebLinigernodx(self._Qr,self._Rr,f,self._mu,self._mass,self._inter,self._Delta,direction=-1)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            ihrprojected=-(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
            Hlregauged=np.transpose(herm(Glinv).dot(np.transpose(self._Hl)).dot(Glinv))

            self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(Hlregauged,self._D*self._D),\
                                                      tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
            self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=np.reshape(self._Hr,self._D*self._D),\
                                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            meanh=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                   self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
                   self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))



            meanpsipsi=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsidagpsidag=np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meanK=np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))
            meaninter=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meantotal=1.0/(2.0*self._mass)*meanK+self._mu*meandensity+self._Delta*(meanpsipsi+meanpsidagpsidag)+self._inter*meaninter

            
            optgradQ,optgradR=cmf.HomogeneousLiebLinigerGradientdiag(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),Gl,Glinv,Gr,Grinv,\
                                                                     self._Hl,self._Hr,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta)
            
            #optgradQ,optgradR=cmf.HomogeneousLiebLinigerGradient(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
            #                                                     self._Hl,self._Hr,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta)


            Wlamtdvp=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(self._Ql,self._Rl,self._Qr,self._Rr,self._Hl,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta,self._dx,direction=1)
            normxopttdvp=np.linalg.norm(Wlamtdvp)
            Vl_=Gl.dot(optgradQ).dot(Glinv)
            Wl_=Gl.dot(np.diag(optgradR)).dot(Glinv)
            Vlam=Vl_.dot(np.diag(self._lam))
            Wlam=Wl_.dot(np.diag(self._lam))
            #print (np.trace(Vlam.dot(herm(Vlam))),np.trace(Wlam.dot(herm(Wlam))))
            self._normxopt=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(Wlam.dot(herm(Wlam))))
            self._dt_,self._rescaledepth,self._it2,self._reset,self._reject,self._warmup=utils.determineNewStepsize(self._dt_,self._dt,self._dts,self._nxdots,self._normxopt,\
                                                                                                                    self._normxoptold,normtol,self._warmup,self._it,self._rescaledepth,self._factor,\
                                                                                                                    self._it2,self._itreset,self._reset)

            if self._reject==True:
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                self._lam=np.copy(self._lamold)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)


                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normxopt,self._normxoptold,normtol))

            if self._reject==False:
                if self._normxopt<1E-2:
                    betanew=0.9
                elif self._normxopt>=1E-2:
                    betanew=0.0
                Vl=Vl_+betanew*Gl.dot(Vl).dot(Glinv)
                Wl=Wl_+betanew*Gl.dot(Wl).dot(Glinv)
                #print betanew
                self._Vlbackup=np.copy(Vl_)
                self._Wlbackup=np.copy(Wl_)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                self._normxoptold=self._normxopt
                self._lamold=np.copy(self._lam)

                self._norms=np.append(self._norms,self._normxopt)
                normstdvp=np.append(normstdvp,normxopttdvp)
                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._kinpluspot=np.append(self._kinpluspot,meanh)
                self._meanDenss=np.append(self._meanDenss,meandensity)
                if self._verbosity==1:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=% f" %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_))
                
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, <psipsi>=%.5f+i%.5f, |<psipsi>|=%.5f, <psi*psi*>=%.5f+i%.5f,|<psi*psi*>|=%.5f, <psipsi>+<psi*psi*>=%.5f, <K>=%.5f, <I>=%.5f, dt=%.8f, <h>/<n>**3=%.8f " %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),np.real(meanpsipsi),np.imag(meanpsipsi),np.abs(meanpsipsi),np.real(meanpsidagpsidag),np.imag(meanpsidagpsidag),np.abs(meanpsidagpsidag),np.real(meanpsipsi+meanpsidagpsidag),np.real(meanK),np.real(meaninter),self._dt_,np.real(meanh/meandensity**3)))

                
                if self._verbosity==3:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, <h>/<n>**3=%.8f, lnorm=%.8f, rnorm=%.8f, l Nlgmres=%i, r Nlgmres=%i, l regIt=%i, r regIt=%i" \
                                 %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),\
                                   self._dt_,np.real(meanh/meandensity**3),lnorm,rnorm,l_lgmresNit,r_lgmresNit,lNit,rNit))



            stdout.flush()

            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True
            if self._it%self._itsave==0:
                self.__dump__()
                if self._verbosity>0:
                    #stdout.write('\r   (checkpointing)   ')
                    print('   (checkpointing)   ')

                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('normstdvp'+self._filename,normstdvp)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('kinpluspot'+self._filename,self._kinpluspot)
                np.save('meanDenss'+self._filename,self._meanDenss)


        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('kinpluspot'+self._filename,self._kinpluspot)
        np.save('meanDenss'+self._filename,self._meanDenss)
        if single_layer_regauge==False:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold),self._D*self._D),nmaxit=10000,\
                                                                      tol=self._regaugetol,numeig=self._numeig,ncv=self._ncv)
        if single_layer_regauge==True:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.singleLayerRegauge(self._Ql,self._Rl,self._dx,deltax=0.01,initial=np.reshape(np.diag(self._lamold),self._D*self._D),\
                                                                                 nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt





    #new optimization for cMPS
    def __simulatediag2__(self):
        single_layer_regauge=False
        printnlcgmessage=True
        printstdemessage=True
        itbeta=0
        itstde=0
        dostde=False
        converged=False
        normstdvp=np.zeros(0)
        Vl=np.zeros((self._D,self._D),dtype=self._dtype)
        Wl=np.zeros((self._D,self._D),dtype=self._dtype)
        
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))

        f=np.zeros((self._D,self._D),dtype=self._dtype)
        while converged==False:
            self._it=self._it+1
            #print self._normxopt,self._eigenaccuracy
            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            if single_layer_regauge==False:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge_return_basis(self._Ql,self._Rl,self._dx,gauge='symmetric',\
                                                                                                                   linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                                   rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                   nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

                
            if single_layer_regauge==True:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv=cmf.singleLayerRegauge_return_basis(self._Ql,self._Rl,self._dx,deltax=0.01,\
                                                                                                                    initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                    nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
                lNit=None
                rNit=None
            #print self._lam
            #print np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
            #print np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D
            #raw_input()
            if self._verbosity==3:
                rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D


            #ihl=cmf.homogeneousdfdxLiebLiniger(self._Ql,self._Rl,self._Ql,self._Rl,self._dx,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            #ihr=cmf.homogeneousdfdxLiebLiniger(self._Qr,self._Rr,self._Qr,self._Rr,self._dx,f,self._mu,self._mass,self._inter,self._Delta,direction=-1)
            ihl=cmf.homogeneousdfdxLiebLinigernodx(self._Ql,self._Rl,f,self._mu,self._mass,self._inter,self._Delta,direction=1)
            ihr=cmf.homogeneousdfdxLiebLinigernodx(self._Qr,self._Rr,f,self._mu,self._mass,self._inter,self._Delta,direction=-1)

            #print Gl.dot(herm(Gl))
            #print
            #print self._lam
            
            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            ihrprojected=-(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
            Hlregauged=np.transpose(herm(Glinv).dot(np.transpose(self._Hl)).dot(Glinv))
            Hrregauged=Gr.dot(self._Hr).dot(herm(Gr))

            self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(Hlregauged,self._D*self._D),\
                                                      tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
            self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=np.reshape(self._Hr,self._D*self._D),\
                                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            meanh=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                   self._inter*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
                   self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))



            meanpsipsi=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsidagpsidag=np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meanK=np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))
            meaninter=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meantotal=1.0/(2.0*self._mass)*meanK+self._mu*meandensity+self._Delta*(meanpsipsi+meanpsidagpsidag)+self._inter*meaninter

            
            Vl_,Wl_=cmf.HomogeneousLiebLinigerGradientdiag2(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
                                                           self._Hl,self._Hr,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta)
            
            
            
            
            Wlamtdvp=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(self._Ql,self._Rl,self._Qr,self._Rr,self._Hl,np.diag(self._lam),self._mass,self._mu,self._inter,self._Delta,self._dx,direction=1)
            normxopttdvp=np.linalg.norm(Wlamtdvp)
            
            self._normxopt=np.sqrt(np.trace(Vl_.dot(herm(Vl_)))+np.trace(Wl_.dot(herm(Wl_))))
            self._dt_,self._rescaledepth,self._it2,self._reset,self._reject,self._warmup=utils.determineNewStepsize(self._dt_,self._dt,self._dts,self._nxdots,self._normxopt,\
                                                                                                                    self._normxoptold,normtol,self._warmup,self._it,self._rescaledepth,self._factor,\
                                                                                                                    self._it2,self._itreset,self._reset)

            if self._reject==True:
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                self._lam=np.copy(self._lamold)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normxopt,self._normxoptold,normtol))

            if self._reject==False:
                Vl=Vl_
                Wl=Wl_
                self._Vlbackup=np.copy(Vl_)
                self._Wlbackup=np.copy(Wl_)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                self._normxoptold=self._normxopt
                self._lamold=np.copy(self._lam)

                self._norms=np.append(self._norms,self._normxopt)
                normstdvp=np.append(normstdvp,normxopttdvp)
                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._kinpluspot=np.append(self._kinpluspot,meanh)
                self._meanDenss=np.append(self._meanDenss,meandensity)
                if self._verbosity==1:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=% f" %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_))
                
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, <psipsi>=%.5f+i%.5f, |<psipsi>|=%.5f, <psi*psi*>=%.5f+i%.5f,|<psi*psi*>|=%.5f, <psipsi>+<psi*psi*>=%.5f, <K>=%.5f, <I>=%.5f, dt=%.8f, <h>/<n>**3=%.8f " %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),np.real(meanpsipsi),np.imag(meanpsipsi),np.abs(meanpsipsi),np.real(meanpsidagpsidag),np.imag(meanpsidagpsidag),np.abs(meanpsidagpsidag),np.real(meanpsipsi+meanpsidagpsidag),np.real(meanK),np.real(meaninter),self._dt_,np.real(meanh/meandensity**3)))

                
                if self._verbosity==3:
                    stdout.write("\rit %i: ||x|| = %.8f, ||x||-tdvp = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, <h>/<n>**3=%.8f, lnorm=%.8f, rnorm=%.8f, l Nlgmres=%i, r Nlgmres=%i, l regIt=%i, r regIt=%i" \
                                 %(self._it,np.real(self._normxopt),np.real(normxopttdvp),np.real(meanh),np.real(meantotal),np.real(meandensity),\
                                   self._dt_,np.real(meanh/meandensity**3),lnorm,rnorm,l_lgmresNit,r_lgmresNit,lNit,rNit))



            stdout.flush()
            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True
            if self._it%self._itsave==0:
                self.__dump__()
                if self._verbosity>0:
                    #stdout.write('\r   (checkpointing)   ')
                    print('   (checkpointing)   ')

                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('normstdvp'+self._filename,normstdvp)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('kinpluspot'+self._filename,self._kinpluspot)
                np.save('meanDenss'+self._filename,self._meanDenss)


        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('kinpluspot'+self._filename,self._kinpluspot)
        np.save('meanDenss'+self._filename,self._meanDenss)
        if single_layer_regauge==False:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold),self._D*self._D),nmaxit=10000,\
                                                                      tol=self._regaugetol,numeig=self._numeig,ncv=self._ncv)
        if single_layer_regauge==True:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.singleLayerRegauge(self._Ql,self._Rl,self._dx,deltax=0.01,initial=np.reshape(np.diag(self._lamold),self._D*self._D),\
                                                                                 nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt


#for homogeneous states
class HomogeneousExtLiebLinigercMPSEngine:

    def __init__(self, filename,Ql,Rl,nxdots,dt,dts,dx,dtype,mu,g1,g2,eta,mass,itreset=10,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-6,acc=1E-4,itsave=100,verbosity=1,warmuptol=1E-6\
                 ,rescalingfactor=2.0,normtolerance=0.1,initnormtolerance=0.1,initthresh=0.01,numeig=5,ncv=100,Nmaxlgmres=100,outerklgmres=20,innermlgmres=30,nlcgupperthresh=1E-16,\
                 nlcglowerthresh=1E-100,nlcgreset=10,stdereset=3,nlcgnormtol=0.0):
        self._eigenaccuracy=acc
        self._D=np.shape(Ql)[0]
        self._filename=filename
        self._Ql=np.copy(Ql)
        self._Rl=np.copy(Rl)
        self._Qr=np.copy(Ql)
        self._Rr=np.copy(Rl)

        if nxdots!=None:
            self._nxdots=np.copy(nxdots)
        elif nxdots==None:
            self._nxdots=nxdots

        self._dt=dt
        self._dt_=dt
        self._dts=np.copy(dts)

        self._dx=dx
        self._dtype=dtype
        self._mu=mu
        self._g1=g1
        self._g2=g2
        self._eta=eta        
        self._mass=mass
        self._itmax=itmax
        self._regaugetol=regaugetol
        self._lgmrestol=lgmrestol
        self._epsilon=epsilon
        self._itsave=itsave
        self._itreset=itreset
        if self._nxdots!=None:
            assert((len(self._dts))==len(self._nxdots))
        self._verbosity=verbosity
        self._warmuptol=warmuptol
        self._factor=rescalingfactor
        self._normtol=normtolerance
        self._initnormtol=initnormtolerance
        self._initthresh=initthresh
        self._numeig=numeig
        self._ncv=ncv
        self._Nmaxlgmres=Nmaxlgmres
        self._outerklgmres=outerklgmres
        self._innermlgmres=innermlgmres
        self._nlcgupperthresh=nlcgupperthresh
        self._nlcglowerthresh=nlcglowerthresh
        self._nlcgreset=nlcgreset
        self._stdereset=stdereset
        self._nlcgnormtol=nlcgnormtol

        self._it=0
        self._warmup=True
        self._lamold=np.ones(self._D)/np.sqrt(self._D)
        self._Hl=np.eye(self._D)
        self._Hr=np.eye(self._D)
        self._phileft=np.eye(self._D)
        self._phiright=np.eye(self._D)

        self._reset=True
        self._it2=0
        self._Qlbackup=np.copy(self._Ql)
        self._Rlbackup=np.copy(self._Rl)
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    #eset all iterators and flags, so one can __load__() a file, and start a fresh simulation; don't forget to 
    #set the new parameters after rest, if you don't want to use the ones from the __load__()ed file
    def __reset__(self):
        self._it=0
        self._warmup=True
        #self._lamold=np.ones(self._D)/np.sqrt(self._D)
        #self._Hl=np.eye(self._D)
        #self._Hr=np.eye(self._D)
        self._reset=True
        self._it2=0
        #self._Qlbackup=np.copy(self._Ql)
        #self._Rlbackup=np.copy(self._Rl)
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        #self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        #self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    def __cleanup__(self):
        cwd=os.getcwd()
        if not os.path.exists('CHECKPOINT_'+self._filename):
            return
        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)


    #dump the simulation into a folder for later retrieval with ___load__()
    def __dump__(self):
        cwd=os.getcwd()
        #raw_input(not os.path.exists('CHECKPOINT_'+self._filename))
        if not os.path.exists('CHECKPOINT_'+self._filename):
            os.mkdir('CHECKPOINT_'+self._filename)

        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)
            os.mkdir('CHECKPOINT_'+self._filename)

        os.chdir('CHECKPOINT_'+self._filename)


        intparams=np.zeros(0,dtype=int)
        floatparams=np.zeros(0,dtype=float)
        intparams=np.append(intparams,self._D)
        intparams=np.append(intparams,self._itmax)
        intparams=np.append(intparams,self._itsave)
        intparams=np.append(intparams,self._it)
        intparams=np.append(intparams,self._it2)
        intparams=np.append(intparams,self._itreset)
        intparams=np.append(intparams,self._rescaledepth)
        intparams=np.append(intparams,self._verbosity)
        intparams=np.append(intparams,self._numeig)
        intparams=np.append(intparams,self._ncv)
        intparams=np.append(intparams,self._nlcgreset)
        intparams=np.append(intparams,self._stdereset)
        intparams=np.append(intparams,self._Nmaxlgmres)
        intparams=np.append(intparams,self._outerklgmres)
        intparams=np.append(intparams,self._innermlgmres)



        floatparams=np.append(floatparams,self._dt)
        floatparams=np.append(floatparams,self._dt_)
        floatparams=np.append(floatparams,self._dx)
        floatparams=np.append(floatparams,self._mu)
        floatparams=np.append(floatparams,self._g1)
        floatparams=np.append(floatparams,self._mass)
        floatparams=np.append(floatparams,self._regaugetol)
        floatparams=np.append(floatparams,self._lgmrestol)
        floatparams=np.append(floatparams,self._epsilon)
        floatparams=np.append(floatparams,self._normxopt)
        floatparams=np.append(floatparams,self._normxoptold)
        floatparams=np.append(floatparams,self._eigenaccuracy)
        floatparams=np.append(floatparams,self._warmuptol)
        floatparams=np.append(floatparams,self._factor)
        floatparams=np.append(floatparams,self._normtol)
        floatparams=np.append(floatparams,self._nlcgupperthresh)
        floatparams=np.append(floatparams,self._nlcglowerthresh)
        floatparams=np.append(floatparams,self._nlcgnormtol)
        floatparams=np.append(floatparams,self._initnormtol)
        floatparams=np.append(floatparams,self._initthresh)
        floatparams=np.append(floatparams,self._eta)
        floatparams=np.append(floatparams,self._g2)

        boolparams=np.empty((0),dtype=bool)
        boolparams=np.append(boolparams,self._warmup)        
        boolparams=np.append(boolparams,self._reset)        


        np.save('intparams',intparams)
        np.save('floatparams',floatparams)
        np.save('boolparams',boolparams)
        np.save('Ql',self._Ql)
        np.save('Rl',self._Rl)
        np.save('Qr',self._Qr)
        np.save('Rr',self._Rr)
        if self._nxdots!=None:
            np.save('nxdots',self._nxdots)
        if self._nxdots==None:
            np.save('nxdots',np.zeros(0))

        np.save('dts',self._dts)
        np.save('lamold',self._lamold)
        np.save('kleft',self._Hl)
        np.save('kright',self._Hr)
        np.save('Qlbackup',self._Qlbackup)
        np.save('Rlbackup',self._Rlbackup)
        np.save('norms',self._norms)
        np.save('totalEnergy',self._totalEnergy)
        np.save('kinpluspot',self._kinpluspot)
        np.save('meanDenss',self._meanDenss)
        np.save('Vlbackup',self._Vlbackup)
        np.save('Wlbackup',self._Wlbackup)
        os.chdir(cwd)
        
    #load a simulation from a folder named filename
    def __load__(self,filename):
        os.chdir(filename)
        
        intparams=np.load('intparams.npy')
        floatparams=np.load('floatparams.npy')
        boolparams=np.load('boolparams.npy')

        self._D=int(intparams[0])
        self._itmax=int(intparams[1])
        self._itsave=int(intparams[2])
        self._it=int(intparams[3])
        self._it2=int(intparams[4])
        self._itreset=int(intparams[5])
        self._rescaledepth=int(intparams[6])
        self._verbosity=int(intparams[7])
        self._numeig=int(intparams[8])
        self._ncv=int(intparams[9])
        self._nlcgreset=int(intparams[10])
        self._stdereset=int(intparams[11])
        self._Nmaxlgmres=int(intparams[12])
        self._outerklgmres=int(intparams[13])
        self._innermlgmres=int(intparams[14])


        self._dt=np.real(floatparams[0])
        self._dt_=np.real(floatparams[1])
        self._dx=np.real(floatparams[2])
        self._mu=np.real(floatparams[3])
        self._g1=np.real(floatparams[4])
        self._mass=np.real(floatparams[5])
        self._regaugetol=np.real(floatparams[6])
        self._lgmrestol=np.real(floatparams[7])
        self._epsilon=np.real(floatparams[8])
        self._normxopt=np.real(floatparams[9])
        self._normxoptold=np.real(floatparams[10])
        self._eigenaccuracy=np.real(floatparams[11])
        self._warmuptol=np.real(floatparams[12])
        self._factor=np.real(floatparams[13])
        self._normtol=np.real(floatparams[14])
        self._nlcgupperthresh=np.real(floatparams[15])
        self._nlcglowerthresh=np.real(floatparams[16])
        self._nlcgnormtol=np.real(floatparams[17])
        self._initnormtol=np.real(floatparams[18])
        self._initthresh=np.real(floatparams[19])
        self._eta=np.real(floatparams[20])
        self._g2=np.real(floatparams[21])

        self._warmup=boolparams[0]
        self._reset=boolparams[1]

        self._Ql=np.load('Ql.npy')
        self._Rl=np.load('Rl.npy')
        self._Qr=np.load('Qr.npy')
        self._Rr=np.load('Rr.npy')
        nxdots=np.load('nxdots.npy')

        if len(nxdots==0):
            self._nxdots=None

        if len(nxdots!=0):
            self._nxdots=np.copy(nxdots)

        self._dts=np.load('dts.npy')
        self._lamold=np.load('lamold.npy')
        self._Hl=np.load('kleft.npy')
        self._Hr=np.load('kright.npy')
        self._Qlbackup=np.load('Qlbackup.npy')
        self._Rlbackup=np.load('Rlbackup.npy')
        self._norms=np.load('norms.npy')
        self._totalEnergy=np.load('totalEnergy.npy')
        self._kinpluspot=np.load('kinpluspot.npy')
        self._meanDenss=np.load('meanDenss.npy')
        self._Vlbackup=np.load('Vlbackup.npy')
        self._Wlbackup=np.load('Wlbackup.npy')
        os.chdir('../')

    #new optimization for cMPS
    def __simulate__(self):
        single_layer_regauge=False
        printnlcgmessage=True
        printstdemessage=True
        itbeta=0
        itstde=0
        dostde=False
        converged=False
        normstdvp=np.zeros(0)
        phiold=np.ones(self._D)
        Vl=np.zeros((self._D,self._D),dtype=self._dtype)
        Wl=np.zeros((self._D,self._D),dtype=self._dtype)
        
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))

        f=np.zeros((self._D,self._D),dtype=self._dtype)
        while converged==False:
            self._it=self._it+1

            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            if single_layer_regauge==False:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge_return_basis(self._Ql,self._Rl,self._dx,gauge='symmetric',\
                                                                                                                   linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                                   rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                   nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

                
            if single_layer_regauge==True:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv=cmf.singleLayerRegauge_return_basis(self._Ql,self._Rl,self._dx,deltax=0.01,\
                                                                                                                    initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                    nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

                lNit=None
                rNit=None


            if self._verbosity==3:
                if self._dx>=1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D
                elif self._dx<1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl))/self._D



            self._phileft,nl=cmf.geometricSumExtLiebLinigerInteraction(self._Ql,self._Rl,np.eye(self._D),np.diag(self._lam**2),self._eta,direction=1,x0=np.reshape(self._phileft,self._D**2))
            self._phiright,nr=cmf.geometricSumExtLiebLinigerInteraction(self._Qr,self._Rr,np.diag(self._lam**2),np.eye(self._D),self._eta,direction=-1,x0=np.reshape(self._phiright,self._D**2))

            
            #print (np.trace(herm(self._Rl).dot(np.transpose(self._phileft)).dot(self._Rl).dot(np.diag(self._lam**2))))
            #print (np.trace(np.diag(self._lam**2).dot(self._Rr).dot(self._phiright).dot(herm(self._Rr))))
            phileftprojected=self._phileft-np.tensordot(self._phileft,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D)
            phirightprojected=self._phiright-np.tensordot(np.diag(self._lam**2),self._phiright,([0,1],[0,1]))*np.eye(self._D)


            

            #ihl=cmf.homogeneousdfdxLiebLinigernodx(self._Ql,self._Rl,f,self._mu,self._mass,self._g1,0.0,direction=1)
            #ihr=cmf.homogeneousdfdxLiebLinigernodx(self._Qr,self._Rr,f,self._mu,self._mass,self._g1,0.0,direction=-1)

            ihl=cmf.homogeneousdfdxExtLiebLinigernodx(self._Ql,self._Rl,self._phileft,f,self._mu,self._mass,self._g1,self._g2,direction=1)
            ihr=cmf.homogeneousdfdxExtLiebLinigernodx(self._Qr,self._Rr,self._phiright,f,self._mu,self._mass,self._g1,self._g2,direction=-1)
            #ihl=cmf.homogeneousdfdxExtLiebLinigernodx(self._Ql,self._Rl,phileftprojected,f,self._mu,self._mass,self._g1,self._g2,direction=1)
            #ihr=cmf.homogeneousdfdxExtLiebLinigernodx(self._Qr,self._Rr,phirightprojected,f,self._mu,self._mass,self._g1,self._g2,direction=-1)
            
            
            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            ihrprojected=-(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
            #ihlprojected+=self._g2*np.transpose(herm(self._Rl).dot(np.transpose(self._phileft)).dot(self._Rl))
            #ihrprojected+=self._g2*self._Rr.dot(self._phiright).dot(herm(self._Rr))
            #Hlregauged=np.transpose(herm(Glinv).dot(np.transpose(self._Hl)).dot(Glinv))
            Hlregauged=self._Hl
            self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(Hlregauged,self._D*self._D),\
                                                      tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
            self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=np.reshape(self._Hr,self._D*self._D),\
                                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            
            #meanpsipsi=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))
            #meanpsidagpsidag=np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meanK=np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))
            meantotal=1.0/(2.0*self._mass)*meanK+self._mu*meandensity+self._g2*np.trace(herm(self._Rl).dot(np.transpose(self._phileft)).dot(self._Rl).dot(np.diag(self._lam**2)))
            meanh=np.tensordot(self._Hl,np.diag(self._lam**2),([0,1],[0,1]))+np.tensordot(self._Hr,np.diag(self._lam**2),([0,1],[0,1]))

            #meantotal=1.0/(2.0*self._mass)*np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
            #           self._g1*np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)).dot(herm(self._Rl)).dot(herm(self._Rl)))+\
            #           self._mu*np.trace(self._Rl.dot(np.diag(self._lam**2)).dot(herm(self._Rl)))+\
            #           self._Delta*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2))+np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))
            #meantotal=cmf.HomogeneousLiebLinigerCMPS_epsilon_energy(self._Hl,self._Ql,self._Rl,self._Hr,self._Qr,self._Rr,np.diag(self._lam),self._mass,self._mu,self._g1)

            Vlam,Wlam=cmf.HomogeneousExtendedLiebLinigerGradient(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
                                                                 self._Hl,self._Hr,self._phileft,self._phiright,np.diag(self._lam),self._mass,self._mu,self._g1,self._g2)
            #Vlam,Wlam=cmf.HomogeneousExtendedLiebLinigerGradient(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
            #                                                     self._Hl,self._Hr,phileftprojected,phirightprojected,np.diag(self._lam),self._mass,self._mu,self._g1,self._g2)


            Vl_=Vlam.dot(np.diag(1.0/self._lam))
            Wl_=Wlam.dot(np.diag(1.0/self._lam))


            #self._normxopt=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(Wlam.dot(herm(Wlam))))
            Z=np.linalg.norm(self._lam)
            phinew=self._lam/Z

            #self._normxopt=np.sqrt(np.abs(1.0-phinew.dot(phiold)))
            self._normxopt=np.linalg.norm(self._lam/Z-phiold)/self._dt_
            phiold=np.copy(phinew)
            #self._normxopt=np.sqrt(np.trace(Vl_.dot(herm(Vl_)))+np.trace(Wl_.dot(herm(Wl_))))
            self._dt_,self._rescaledepth,self._it2,self._reset,self._reject,self._warmup=utils.determineNewStepsize(self._dt_,self._dt,self._dts,self._nxdots,self._normxopt,\
                                                                                                                    self._normxoptold,normtol,self._warmup,self._it,self._rescaledepth,self._factor,\
                                                                                                                    self._it2,self._itreset,self._reset)

            if self._reject==True:
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                self._lam=np.copy(self._lamold)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normxopt,self._normxoptold,normtol))

            if self._reject==False:
                betanew,itstde,itbeta,dostde,printnlcgmessage,printstdemessage=utils.determineNonLinearCGBeta(self._nlcgupperthresh,self._nlcglowerthresh,self._nlcgnormtol,self._nlcgreset,self._normxopt,\
                                                                                                       self._normxoptold,self._it,itstde,self._stdereset,dostde,itbeta,printnlcgmessage,printstdemessage)

                betanew=0.0
                Vl=Vl_+betanew*Gl.dot(Vl).dot(Glinv)
                Wl=Wl_+betanew*Gl.dot(Wl).dot(Glinv)
                
                self._Vlbackup=np.copy(Vl_)
                self._Wlbackup=np.copy(Wl_)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)
                #print(np.linalg.norm(self._Ql.dot(np.diag(self._lam))-np.diag(self._lam).dot(self._Qr)))
                #r=np.diag((self._lam**2))
                #print('dd')
                #print(np.linalg.norm(self._phileft-herm(self._phileft)))
                #print(np.linalg.norm(self._phiright-herm(self._phiright)))
                #print ('before: ',np.linalg.norm((self._Ql.dot(r)+r.dot(herm(self._Ql))+self._Rl.dot(r).dot(herm(self._Rl)))))
                #Smat,Umat=np.linalg.eig(self._Ql)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)
                #print ('after: ',np.linalg.norm((self._Ql.dot(r)+r.dot(herm(self._Ql))+self._Rl.dot(r).dot(herm(self._Rl)))))
                #input()

                #Qc=self._Ql.dot(np.diag(self._lam))
                #Rc=self._Rl.dot(np.diag(self._lam))
                #Qnew=(self._Ql-self._dt_*Vl)
                #Rnew=(self._Rl-self._dt_*Wl)
                #Qcnew=Qnew.dot(np.diag(self._lam))
                #Rcnew=Rnew.dot(np.diag(self._lam))
                

                #print('')
                #phi1=(np.tensordot(Qc,np.diag(self._lam),([1,0],[1,0]))+np.tensordot(np.diag(self._lam),np.conj(Qcnew),([1,0],[1,0]))+np.tensordot(Rc,np.conj(Rcnew),([1,0],[1,0])))
                #phi2=(np.tensordot(Qcnew,np.diag(self._lam),([1,0],[1,0]))+np.tensordot(np.diag(self._lam),np.conj(Qcnew),([1,0],[1,0]))+np.tensordot(Rcnew,np.conj(Rcnew),([1,0],[1,0])))
                #self._normxopt=np.sqrt(np.abs(phi1-phi2/2.0))#,phi1,phi2/2.0)
                #print(np.tensordot(Qc,np.diag(self._lam),([1,0],[1,0]))+np.tensordot(np.diag(self._lam),np.conj(Qc),([1,0],[1,0]))+np.tensordot(Rc,np.conj(Rc),([1,0],[1,0])))
                #Smat2,Umat2=np.linalg.eig(self._Ql)
                #self._Ql=Qnew
                #self._Rl=Rnew
                self._normxoptold=self._normxopt

                self._lamold=np.copy(self._lam)

                self._norms=np.append(self._norms,self._normxopt)

                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._kinpluspot=np.append(self._kinpluspot,meanh)
                self._meanDenss=np.append(self._meanDenss,meandensity)

                if self._verbosity==1:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=% f, beta=%.3f" %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),self._dt_,np.real(betanew)))
                
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, <psipsi>=%.5f+i%.5f, |<psipsi>|=%.5f, <psi*psi*>=%.5f+i%.5f,|<psi*psi*>|=%.5f, <psipsi>+<psi*psi*>=%.5f, <K>=%.5f, <I>=%.5f, dt=%.8f, beta=%.3f, <h>/<n>**3=%.8f " %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),np.real(meanpsipsi),np.imag(meanpsipsi),np.abs(meanpsipsi),np.real(meanpsidagpsidag),np.imag(meanpsidagpsidag),np.abs(meanpsidagpsidag),np.real(meanpsipsi+meanpsidagpsidag),np.real(meanK),np.real(meaninter),self._dt_,np.real(betanew),np.real(meanh/meandensity**3)))

                
                if self._verbosity==3:
                    stdout.write("\rit %i: ||x|| = %.8f, <h> = %.8f, <E>=%.8f, <n>=%.8f, dt=%.8f, beta=%.3f, <h>/<n>**3=%.8f, lnorm=%.8f, rnorm=%.8f, l Nlgmres=%i, r Nlgmres=%i, l regIt=%i, r regIt=%i" \
                                 %(self._it,np.real(self._normxopt),np.real(meanh),np.real(meantotal),np.real(meandensity),\
                                   self._dt_,np.real(betanew),np.real(meanh/meandensity**3),lnorm,rnorm,l_lgmresNit,r_lgmresNit,lNit,rNit))



            stdout.flush()
            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True
            if self._it%self._itsave==0:
                self.__dump__()
                if self._verbosity>0:
                    #stdout.write('\r   (checkpointing)   ')
                    print('   (checkpointing)   ')

                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('kinpluspot'+self._filename,self._kinpluspot)
                np.save('meanDenss'+self._filename,self._meanDenss)


        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('kinpluspot'+self._filename,self._kinpluspot)
        np.save('meanDenss'+self._filename,self._meanDenss)
        if single_layer_regauge==False:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold),self._D*self._D),nmaxit=10000,\
                                                                      tol=self._regaugetol,numeig=self._numeig,ncv=self._ncv)
        if single_layer_regauge==True:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.singleLayerRegauge(self._Ql,self._Rl,self._dx,deltax=0.01,initial=np.reshape(np.diag(self._lamold),self._D*self._D),\
                                                                                 nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt


        
#for homogeneous states
class HomogeneouscMPSEngineMultiSpecies:
    #optimization for multi-species bosonic cMPS using penalty term
    def __simulateTwoBosons__(self,Ql,Rl,alpha,dtype,mu,intrag,interg,mass,filename,penalty,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,ncv=40,numeig=6):
        D=Ql.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                        tol=regaugetol,ncv=ncv,numeig=numeig)
            
            
            if it%100==0:
                Rl0diag=[]
                eigvals0,U0=np.linalg.eig(Rl[0])
                U0inv=np.linalg.inv(U0)
                Rl0diag.append(np.diag(eigvals0))
                Rl0diag.append(np.diag(np.diag(U0inv.dot(Rl[1]).dot(U0))))
                Ql0diag=U0inv.dot(Ql).dot(U0)

                Rl1diag=[]
                eigvals1,U1=np.linalg.eig(Rl[1])
                U1inv=np.linalg.inv(U1)
                Rl1diag.append(np.diag(np.diag(U1inv.dot(Rl[0]).dot(U1))))
                Rl1diag.append(np.diag(eigvals1))
                Ql1diag=U1inv.dot(Ql).dot(U1)
                np.save('Ql'+filename,Ql)
                np.save('Rl'+filename,Rl)
                np.save('lam'+filename,lam)
                np.save('Ql0diag'+filename,Ql0diag)
                np.save('Rl0diag'+filename,Rl0diag)
                np.save('Ql1diag'+filename,Ql1diag)
                np.save('Rl1diag'+filename,Rl1diag)


            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=penalty)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=penalty)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=(Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])).dot(np.diag(lam))
            
            
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(herm(commR1R2))))
            
            Vlam,Wlam=cmf.HomogeneousLiebLinigerTwoBosonSpeciesGradientPenalty(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,interg,penalty)
            Vl=Vlam.dot(np.diag(1.0/lam))
            Wl0=Wlam[0].dot(np.diag(1.0/lam))
            Wl1=Wlam[1].dot(np.diag(1.0/lam))

            T0=np.trace(Vlam.dot(herm(Vlam)))
            for n in range(2):
                T0+=np.trace(Wlam[n].dot(herm(Wlam[n])))
            normxopt=np.sqrt(T0)
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)
            if reject==True:
                Ql=np.copy(Qlbackup)
                Rl=np.copy(Rlbackup)
                lam=np.copy(lamold)
                Vl=np.copy(Vlbackup)
                Wl0=np.copy(Wl0backup)
                Wl1=np.copy(Wl1backup)
                Ql=(Ql-alpha_*Vl)
                Rl[0]=(Rl[0]-alpha_*Wl0)
                Rl[1]=(Rl[1]-alpha_*Wl1)
                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vlbackup=np.copy(Vl)
                Wl0backup=np.copy(Wl0)
                Wl1backup=np.copy(Wl1)
                Qlbackup=np.copy(Ql)
                Rlbackup=np.copy(Rl)
                Ql=(Ql-alpha_*Vl)
                Rl[0]=(Rl[0]-alpha_*Wl0)
                Rl[1]=(Rl[1]-alpha_*Wl1)
                normxoptold=normxopt
                lamold=np.copy(lam)

            stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            #stdout.write("\rit %i: ||x||= %.8f, <k> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, penalty=%4.1f, alpha=% f" %(it,np.real(normxopt),np.real(meank[0]),np.real(meank[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),penalty,alpha_))
            stdout.flush()

            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True

        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                                             tol=regaugetol,ncv=ncv,numeig=numeig)

        
        return lam,Ql,Rl,Qr,Rr,Gl,Glinv,it,normxopt


    #simulates a multi-species cMPS ground-state using a tensor product structure of the cMPS matrices R_1, R_2
    def __simulatetensorproduct__(self,Q,R,D,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,ncv=40,numeig=6):

        Rkron=[]
        Rkron.append(np.kron(R[0],np.eye(D[1])))
        Rkron.append(np.kron(np.eye(D[0]),R[1]))
        
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D[0]*D[1],D[0]*D[1]),dtype=dtype)
        lamold=np.ones(D[0]*D[1])/np.sqrt(D[0]*D[1])
        kleft=np.eye(D[0]*D[1])
        kright=np.eye(D[0]*D[1])
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,Rkron,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D[0]*D[1]*D[0]*D[1]),datatype=dtype,\
            #                                                                                     nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)

            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,Rkron,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D[0]*D[1]*D[0]*D[1]),rinitial=np.reshape(np.eye(D[0]*D[1]),D[0]**2*D[1]**2),\
                                                                        nmaxit=50000,tol=regaugetol,ncv=ncv,numeig=numeig)

            #normalization
            #Q-=eta/2.0*np.eye(D[0]*D[1])
            if it%100==0:
                np.save('QTP'+filename,Q)
                np.save('RTP'+filename,R)

                eta0,U0=np.linalg.eig(R[0])
                eta1,U1=np.linalg.eig(R[1])
                U=np.kron(U0,U1)
                Uinv=np.kron(np.linalg.inv(U0),np.linalg.inv(U1))
                Qdiag=Uinv.dot(Q).dot(U)
                Rdiag=[]
                Rdiag.append(np.kron(np.diag(eta0),np.eye(D[1])))
                Rdiag.append(np.kron(np.eye(D[0]),np.diag(eta1)))
                np.save('QTPdiag'+filename,Qdiag)
                np.save('RTPdiag'+filename,Rdiag)


            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D[0]*D[1]))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D[0]*D[1]))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D[0]*D[1]),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D[0]*D[1]*D[0]*D[1]),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D[0]*D[1]),ihrprojected,direction=-1,x0=np.reshape(kright,D[0]*D[1]*D[0]*D[1]),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl))
            meandensity=np.zeros(len(Rl))
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))
            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]

            
            V,W0,W1=cmf.TwoSpeciesBosonGradient(Ql,Rl,Qr,Rr,D[0],D[1],kleft,kright,Gl,Glinv,Gr,Grinv,lam,mass,mu,intrag,interg,0.0)



            Vlam=Gl.dot(V).dot(Grinv)
            W1lam=Gl.dot(np.kron(W0,np.eye(D[1]))).dot(Grinv)
            W2lam=Gl.dot(np.kron(np.eye(D[1]),W1)).dot(Grinv)
            normxopt=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(W1lam.dot(herm(W1lam)))+np.trace(W2lam.dot(herm(W2lam))))
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)


            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                Rkron[0]=np.kron(R[0],np.eye(D[1]))
                Rkron[1]=np.kron(np.eye(D[0]),R[1])

                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                Rkron[0]=np.kron(R[0],np.eye(D[1]))
                Rkron[1]=np.kron(np.eye(D[0]),R[1])

                normxoptold=normxopt
                lamold=np.copy(lam)

            #print np.linalg.norm(V), np.linalg.norm(W0),np.linalg.norm(W1)
            #normxopt=np.sqrt(np.real(np.trace(herm(V).dot(V))+np.trace(herm(W0).dot(W0))+np.trace(herm(W1).dot(W1))))
            #normxopt=np.sqrt(np.real(np.trace(herm(V).dot(V))+np.trace(herm(W0).dot(W0))+np.trace(herm(W1).dot(W1))))




            #R[0]=(R[0]-dt*np.kron(W0,np.eye(D[1])))
            #R[1]=(R[1]-dt*np.kron(np.eye(D[0]),W1))
            #print('it={0}, ||x||={1}, <h>={2}, <n>={3}, dt={4}'.format(it,np.real(normxopt),np.real(meanh),np.real(meandensity),dt))
            stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <e>= %.8f, dt=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),
                                                                                                         np.real(meandensity[1]),np.real(meanE),alpha_))
            stdout.flush()
            if it%100==0:
                print

            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy=cmf.regauge(Q,Rkron,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D[0]*D[1]*D[0]*D[1]),datatype=dtype,nmaxit=50000,\
        #                                                                                 tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,Rkron,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D[0]*D[1]*D[0]*D[1]),rinitial=np.reshape(np.eye(D[0]*D[1]),D[0]**2*D[1]**2),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)

        return lam,Ql,Rl,Qr,Rr,it,normxopt



    #simulates a multi-species cMPS ground-state using a codiagonal structure of the cMPS matrices R_1, R_2
    def __simulateTwoBosonsDiag__(self,Q,R,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,\
                                  ncv=40,numeig=6):
        D=Q.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D**2),nmaxit=50000,\
                                                                        tol=regaugetol,ncv=ncv,numeig=numeig)


            #Q-=eta/2.0*np.eye(D).astype(complex)            
            if it%100==0:
                np.save('Q'+filename,Q)
                np.save('R'+filename,R)
                np.save('lam'+filename,lam)

            #normxopt=np.linalg.norm((lam-lamold)/alpha_)
            #normxopt=np.linalg.norm((lam**2-lamold**2)/alpha_)
            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(commR1R2))))

            V,W0,W1=cmf.HomogeneousLiebLinigerTwoBosonsGradientDiag(Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,kleft,kright,np.diag(lam),mass,mu,intrag,interg)
            Vlam=Gl.dot(V).dot(Glinv).dot(np.diag(lam))
            W0lam=Gl.dot(W0).dot(Glinv).dot(np.diag(lam))
            W1lam=Gl.dot(W1).dot(Glinv).dot(np.diag(lam))
            T0=np.trace(Vlam.dot(herm(Vlam)))+np.trace(W0lam.dot(herm(W0lam)))+np.trace(W1lam.dot(herm(W1lam)))
            normxopt=np.sqrt(T0)
            normV=np.sqrt(np.trace(Vlam.dot(herm(Vlam))))
            normW0=np.sqrt(np.trace(W0lam.dot(herm(W0lam))))
            normW1=np.sqrt(np.trace(W1lam.dot(herm(W1lam))))
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)

            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                normxoptold=normxopt
                lamold=np.copy(lam)

            #stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            stdout.write("\rit %i: ||V||= %.8f, ||W0||= %.8f, ||W0||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normV),np.real(normW0),np.real(normW1),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))

            stdout.flush()
            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D**2),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt

    def __simulateTwoBosonsDiagtest__(self,Q,R,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,\
                                  ncv=40,numeig=6):
        D=Q[0].shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        Ql=[]
        Qr=[]
        Ql.append(None)
        Ql.append(None)
        Qr.append(None)
        Qr.append(None)
        while converged==False:
            it=it+1
            #use two Q matrices
            Q_=Q[0]+Q[1]
            #for the reduced eingenmatrices the separation into two doesn't matter
            #lam,Ql_,Rl,Qr_,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q_,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql_,Rl,Qr_,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q_,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                          tol=regaugetol,ncv=ncv,numeig=numeig)

            #print Ql_+herm(Ql_)+herm(Rl[0]).dot(Rl[0])+herm(Rl[1]).dot(Rl[1])
            #raw_input()
            #the normalization has to be done on both
            Q[0]-=eta/4.0*np.eye(D).astype(complex)            
            Q[1]-=eta/4.0*np.eye(D).astype(complex)  
            Q_=Q[0]+Q[1]
            #lam,Ql_,Rl,Qr_,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q_,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                       tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql_,Rl,Qr_,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q_,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                          tol=regaugetol,ncv=ncv,numeig=numeig)

            
            Ql[0]=Gl.dot(Q[0]).dot(Glinv)
            Ql[1]=Gl.dot(Q[1]).dot(Glinv)
            
            Qr[0]=Gr.dot(Q[0]).dot(Grinv)
            Qr[1]=Gr.dot(Q[1]).dot(Grinv)
            

            
            #check
            #print Ql[0]+herm(Ql[0])+Ql[1]+herm(Ql[1])+herm(Rl[0]).dot(Rl[0])+herm(Rl[1]).dot(Rl[1])

            if it%100==0:
                np.save('Q'+filename,Q)
                np.save('R'+filename,R)
                np.save('lam'+filename,lam)

            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))
            #the reduced hamiltonians are insensitive to the splitting into Q1 and Q2

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)
            
            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql/2.0,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql/2.0,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql/2.0,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql/2.0,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(commR1R2))))

            V,W0,W1=cmf.HomogeneousLiebLinigerTwoBosonsGradientDiag(Ql_,Rl,Qr_,Rr,Gl,Glinv,Gr,Grinv,kleft,kright,np.diag(lam),mass,mu,intrag,interg)
            Vlam=Gl.dot(V).dot(Glinv).dot(np.diag(lam))
            W0lam=Gl.dot(W0).dot(Glinv).dot(np.diag(lam))
            W1lam=Gl.dot(W1).dot(Glinv).dot(np.diag(lam))
            T0=np.trace(Vlam.dot(herm(Vlam)))+np.trace(W0lam.dot(herm(W0lam)))+np.trace(W1lam.dot(herm(W1lam)))
            normxopt=np.sqrt(T0)
            normV=np.sqrt(np.trace(Vlam.dot(herm(Vlam))))
            normW0=np.sqrt(np.trace(W0lam.dot(herm(W0lam))))
            normW1=np.sqrt(np.trace(W1lam.dot(herm(W1lam))))
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)

            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)

                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))

            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                normxoptold=normxopt
                lamold=np.copy(lam)

            #stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            stdout.write("\rit %i: ||V||= %.8f, ||W0||= %.8f, ||W0||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normV),np.real(normW0),np.real(normW1),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))

            stdout.flush()
            raw_input()
            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)

        
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt



    ###########################################################################################################################################################################################################################################################################################################
    ################################f everything below is development stage ####################################################################################################################################################################################################################################
    ###########################################################################################################################################################################################################################################################################################################
    ###########################################################################################################################################################################################################################################################################################################

    def __simulateTwoBosonsHalfDiag__(self,Q,R,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,\
                                  ncv=40,numeig=6):

        D=Q.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regaugeMltiSpecies(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regaugeMltiSpecies(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),datatype=dtype,nmaxit=50000,\
                                                                       tol=regaugetol,ncv=ncv,numeig=numeig)
            
            #Q-=eta/2.0*np.eye(D).astype(complex)            
            if it%100==0:
                np.save('Q'+filename,Q)
                np.save('R'+filename,R)
                np.save('lam'+filename,lam)

            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(commR1R2))))

            Vlam,Wlams=cmf.HomogeneousLiebLinigerTwoBosonsGradient(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,interg)
            W0lam=Wlams[0]
            W1lam=Wlams[1]
            V=Glinv.dot(Vlam).dot(Gr)
            W0=Glinv.dot(W0lam).dot(Gr)
            W1=Glinv.dot(W1lam).dot(Gr)
            #W0lam=Gl.dot(W0).dot(Glinv).dot(np.diag(lam))
            #W1lam=Gl.dot(W1).dot(Glinv).dot(np.diag(lam))
            #V=V
            #W0=W0
            #W1=W1

            T0=np.trace(Vlam.dot(herm(Vlam)))+np.trace(W0lam.dot(herm(W0lam)))+np.trace(W1lam.dot(herm(W1lam)))
            normxopt=np.sqrt(T0)
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)
            #print
            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)
                #if it%20<10:
                R0=(R[0]-alpha_*W0)
                R1=(R[1]-alpha_*W1)
                eta0,U0=np.linalg.eig(R0)
                eta1,U1=np.linalg.eig(R1)
                U=(U0+U1)/2.0
                Uinv=np.linalg.pinv(U)
                R[0]=np.diag(np.diag(Uinv.dot(R0).dot(U)))
                R[1]=np.diag(np.diag(Uinv.dot(R1).dot(U)))
                #Q=Q-alpha_*V
                #R[0]=U.dot(np.diag(eta0)).dot(Uinv)
                #R[1]=U.dot(np.diag(eta1)).dot(Uinv)
                #eta,U=np.linalg.eig(R[0])
                #Uinv=np.linalg.inv(U)
                #R[0]=np.diag(eta)
                
                #R[1]=np.diag(np.diag(Uinv.dot(R[1]-alpha_*W1).dot(U)))
                #print 'updating R0'
                #R[1]=np.diag(np.diag(Uinv.dot(R[1]).dot(U)))
                #if it%20>=10:
                #    R[1]=(R[1]-alpha_*W1)
                #    eta,U=np.linalg.eig(R[1])
                #    Uinv=np.linalg.inv(U)
                #    #R[0]=np.diag(np.diag(Uinv.dot(R[0]-alpha_*W0).dot(U)))
                #    R[0]=np.diag(np.diag(Uinv.dot(R[0]).dot(U)))
                #    print 'updating R1'
                #    R[1]=np.diag(eta)
                Q=Uinv.dot(Q-alpha_*V).dot(U)

                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                R0=(R[0]-alpha_*W0)
                R1=(R[1]-alpha_*W1)
                eta0,U0=np.linalg.eig(R0)
                eta1,U1=np.linalg.eig(R1)
                U=(U0+U1)/2.0
                Uinv=np.linalg.pinv(U)
                R[0]=np.diag(np.diag(Uinv.dot(R0).dot(U)))
                R[1]=np.diag(np.diag(Uinv.dot(R1).dot(U)))
                #Q=Q-alpha_*V
                #if it%20<10:
                #    R[0]=(R[0]-alpha_*W0)
                #    eta,U=np.linalg.eig(R[0])
                #    Uinv=np.linalg.inv(U)
                #    R[0]=np.diag(eta)                
                #    #R[1]=np.diag(np.diag(Uinv.dot(R[1]-alpha_*W1).dot(U)))
                #    R[1]=np.diag(np.diag(Uinv.dot(R[1]).dot(U)))
                #    print 'updating R0'
                #if it%20>=10:
                #    R[1]=(R[1]-alpha_*W1)
                #    eta,U=np.linalg.eig(R[1])
                #    Uinv=np.linalg.inv(U)
                #    #R[0]=np.diag(np.diag(Uinv.dot(R[0]-alpha_*W0).dot(U)))
                #    R[0]=np.diag(np.diag(Uinv.dot(R[0]).dot(U)))
                #    R[1]=np.diag(eta)
                #    print 
                Q=Uinv.dot(Q-alpha_*V).dot(U)
                normxoptold=normxopt
                lamold=np.copy(lam)

            stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            stdout.flush()

            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt

    def __simulateTwoBosonsDiagMixed__(self,Q,R,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,\
                                  ncv=40,numeig=6):

        D=Q.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                        tol=regaugetol,ncv=ncv,numeig=numeig)
            
            Q-=eta/2.0*np.eye(D).astype(complex)            
            if it%100==0:
                np.save('Q'+filename,Q)
                np.save('R'+filename,R)
                np.save('lam'+filename,lam)

            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(commR1R2))))
            

            #first upate Q and one of the Rs with the full update:
            gradQc,gradRc=cmf.HomogeneousLiebLinigerTwoBosonsGradient(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,interg)
            V=Glinv.dot(gradQc).dot(Gr)            
            W0=Glinv.dot(gradRc[0]).dot(Gr)
            R0_=(R[0]-alpha_*W0)
            Q_=Uinv.dot(Q-alpha_*V).dot(U)
            
            gradQ,gradR0,gradR1=cmf.HomogeneousLiebLinigerTwoBosonsGradientDiag(Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,kleft,kright,np.diag(lam),mass,mu,intrag,interg)
            W0lam=Wlams[0]
            W1lam=Wlams[1]
            

            W1=Glinv.dot(W1lam).dot(Gr)
            #W0lam=Gl.dot(W0).dot(Glinv).dot(np.diag(lam))
            #W1lam=Gl.dot(W1).dot(Glinv).dot(np.diag(lam))
            #V=V
            #W0=W0
            #W1=W1
            T0=np.trace(Vlam.dot(herm(Vlam)))+np.trace(W0lam.dot(herm(W0lam)))+np.trace(W1lam.dot(herm(W1lam)))
            normxopt=np.sqrt(T0)
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)
            print
            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)
                
                if it%20<10:
                    R[0]=(R[0]-alpha_*W0)
                    eta,U=np.linalg.eig(R[0])
                    Uinv=np.linalg.inv(U)
                    R[0]=np.diag(eta)
                    #R[1]=np.diag(np.diag(Uinv.dot(R[1]-alpha_*W1).dot(U)))
                    print ('updating R0')
                    R[1]=np.diag(np.diag(Uinv.dot(R[1]).dot(U)))
                if it%20>=10:
                    R[1]=(R[1]-alpha_*W1)
                    eta,U=np.linalg.eig(R[1])
                    Uinv=np.linalg.inv(U)
                    #R[0]=np.diag(np.diag(Uinv.dot(R[0]-alpha_*W0).dot(U)))
                    R[0]=np.diag(np.diag(Uinv.dot(R[0]).dot(U)))
                    print ('updating R1')
                    R[1]=np.diag(eta)
                Q=Uinv.dot(Q-alpha_*V).dot(U)

                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                if it%20<10:
                    R[0]=(R[0]-alpha_*W0)
                    eta,U=np.linalg.eig(R[0])
                    Uinv=np.linalg.inv(U)
                    R[0]=np.diag(eta)                
                    #R[1]=np.diag(np.diag(Uinv.dot(R[1]-alpha_*W1).dot(U)))
                    R[1]=np.diag(np.diag(Uinv.dot(R[1]).dot(U)))
                    print ('updating R0')
                if it%20>=10:
                    R[1]=(R[1]-alpha_*W1)
                    eta,U=np.linalg.eig(R[1])
                    Uinv=np.linalg.inv(U)
                    #R[0]=np.diag(np.diag(Uinv.dot(R[0]-alpha_*W0).dot(U)))
                    R[0]=np.diag(np.diag(Uinv.dot(R[0]).dot(U)))
                    R[1]=np.diag(eta)
                    print 
                Q=Uinv.dot(Q-alpha_*V).dot(U)
                normxoptold=normxopt
                lamold=np.copy(lam)

            stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            stdout.flush()

            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt

        
    def __simulateTwoBosonstest__(self,Q,R,D1,D2,alpha,dtype,mu,intrag,interg,mass,filename,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,ncv=40,numeig=6):
        D=Q.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)

        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Q,R,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                        tol=regaugetol,ncv=ncv,numeig=numeig)

            Q-=eta/2.0*np.eye(D).astype(complex)
            #print
            #print 'before: ', np.linalg.norm(comm(Rl[0],Rl[1]))            
            if it%100==0:
                np.save('Ql'+filename,Ql)
                np.save('Rl'+filename,Rl)
                np.save('lam'+filename,lam)

            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Ql,Rl,0.0,f,mu,mass,intrag,interg,direction=1,penalty=0.0)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoBosonSpecies(Qr,Rr,0.0,f,mu,mass,intrag,interg,direction=-1,penalty=0.0)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl)).astype(complex)
            meank=np.zeros(len(Rl)).astype(complex)
            meanI=np.zeros(len(Rl)).astype(complex)
            meandensity=np.zeros(len(Rl)).astype(complex)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meanI[n]=intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meank[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam**2)).dot(herm(Rl[n])))

            meanE=np.sum(meanh)+interg/4.0*np.trace((Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(np.diag(lam**2)).dot(herm(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0]))))+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            commR1R2=Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])
            trcommR1R2=np.sqrt(np.trace(commR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(commR1R2))))
            
            #the optimal gradient, without energy penalty

            
            Vlam,Wlam=cmf.HomogeneousLiebLinigerTwoBosonsGradient(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,interg)

            #Vlam,Wlam=cmf.HomogeneousLiebLinigerTwoBosonSpeciesGradientPenalty(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,interg,penalty)
            V=Glinv.dot(Vlam).dot(np.diag(1.0/lam)).dot(Gl)
            W0=Glinv.dot(Wlam[0].dot(np.diag(1.0/lam))).dot(Gl)
            W1=Glinv.dot(Wlam[1].dot(np.diag(1.0/lam))).dot(Gl)
            #W0=np.zeros((D,D)).astype(complex)
            #W0[0:D1,0:D1]=W0_[0:D1,0:D1]
            #W1=np.zeros((D,D)).astype(complex)
            #W1[D1:D1+D2,D1:D1+D2]=W1_[D1:D1+D2,D1:D1+D2]
            
            #Wl0=Gl.dot(Wl0__).dot(Glinv)
            #Wl1=Gl.dot(Wl1__).dot(Glinv)
            #print
            #print np.abs(W0)
            #print
            #print np.abs(W1)
            #print
            #print Glinv.dot(Rl[0]).dot(Gl)
            T0=np.trace(Vlam.dot(herm(Vlam)))
            for n in range(2):
                T0+=np.trace(Wlam[n].dot(herm(Wlam[n])))
            normxopt=np.sqrt(T0)
            
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)
            if reject==True:
                Q=np.copy(Qbackup)
                R=np.copy(Rbackup)
                V=np.copy(Vbackup)
                W0=np.copy(W0backup)
                W1=np.copy(W1backup)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vbackup=np.copy(V)
                W0backup=np.copy(W0)
                W1backup=np.copy(W1)
                Qbackup=np.copy(Q)
                Rbackup=np.copy(R)
                Q=(Q-alpha_*V)
                R[0]=(R[0]-alpha_*W0)
                R[1]=(R[1]-alpha_*W1)
                normxoptold=normxopt
                lamold=np.copy(lam)
            #print
            #print 'after: ', np.linalg.norm(comm(Rl[0],Rl[1]))
            #raw_input(it)
            #Ql=Ql-dt*Vlam.dot(np.diag(1.0/lam))
            #Rl[0]=Rl[0]-dt*Wlam[0].dot(np.diag(1.0/lam))
            #Rl[1]=Rl[1]-dt*Wlam[1].dot(np.diag(1.0/lam))
            #print('it={0}, ||x||={1}, <h>={2}, <n>={3}, dt={4}'.format(it,np.real(normxopt),np.real(meanh),np.real(meandensity),dt))
            #stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),\
            #                                                                                                                                                                         np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),np.reafilename,l(meanE),trcommR1R2,\
            #                                                                                                                                                                         np.linalg.norm(commR1R2),alpha_))

            stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <k> =[%.8f, %.8f], <n>=[%.8f,%.8f], <i>=[%.8f,%.8f], <e>= %.8f, sqrt(tr([R1,R2]*herm([R1,R2])))=%.8f, ||[R1,R2]||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),
                                                                                                                                                                                                        np.real(meank[0]),np.real(meank[1]),np.real(meandensity[0]),\
                                                                                                                                                                                                        np.real(meandensity[1]),np.real(meanI[0]),np.real(meanI[1]),\
                                                                                                                                                                                                        np.real(meanE),trcommR1R2,np.linalg.norm(commR1R2),alpha_))
            stdout.flush()

            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt





    #new optimization for cMPS
    def __simulateTwoFermions__(self,Ql,Rl,alpha,dtype,mu,intrag,mass,filename,penalty,factor,itreset,normtol,alphas,nxdots,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,ncv=40,numeig=6):
        D=Ql.shape[0]
        converged=False
        it=0
        if itmax==it:
            converged=True
            print ('found itmax=it={0}, leaving simulation'.format(itmax))

        f=np.zeros((D,D),dtype=dtype)
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)
        kright=np.eye(D)
        #penalty=100.0
        alpha_=alpha
        rescaledepth=0
        normxoptold=1E10
        warmup=True
        it2=0
        reset=True
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
            #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                        tol=regaugetol,ncv=ncv,numeig=numeig)


            if it%100==0:
                np.save('Ql'+filename,Ql)
                np.save('Rl'+filename,Rl)
                np.save('lam'+filename,lam)

            lamold=np.copy(lam)
            ihl=cmf.homogeneousdfdxLiebLinigerTwoFermionSpecies(Ql,Rl,0.0,f,mu,mass,intrag,direction=1,penalty=penalty)
            ihr=cmf.homogeneousdfdxLiebLinigerTwoFermionSpecies(Qr,Rr,0.0,f,mu,mass,intrag,direction=-1,penalty=penalty)

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam**2),([0,1],[0,1]))*np.eye(D))
            ihrprojected=-(ihr-np.tensordot(np.diag(lam**2),ihr,([0,1],[0,1]))*np.eye(D))

            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,0.0,np.eye(D),np.diag(lam**2),ihlprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            kright=cmf.inverseTransferOperatorMultiSpecies(Qr,Rr,0.0,np.diag(lam**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright,D*D),tolerance=lgmrestol,maxiteration=4000)

            meanh=np.zeros(len(Rl))
            meandensity=np.zeros(len(Rl))
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+intrag[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
                meandensity[n]=np.trace(Rl[n].dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(Rl[n])))
            meanE=np.sum(meanh)+mu[0]*meandensity[0]+mu[1]*meandensity[1]
            anticommR1R2=Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])
            tranticommR1R2=np.trace(anticommR1R2.dot(np.diag(lam)).dot(np.diag(lam)).dot(herm(anticommR1R2)))
            
            Vlam,Wlam=cmf.HomogeneousLiebLinigerTwoFermionSpeciesGradientPenalty(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag,penalty)
            #Vlam,Wlam=cmf.HomogeneousLiebLinigerTwoFermionSpeciesGradient(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,intrag)
            Vl=Vlam.dot(np.diag(1.0/lam))
            Wl0=Wlam[0].dot(np.diag(1.0/lam))
            Wl1=Wlam[1].dot(np.diag(1.0/lam))
            T0=np.trace(Vlam.dot(herm(Vlam)))
            for n in range(2):
                T0+=np.trace(Wlam[n].dot(herm(Wlam[n])))
            normxopt=np.sqrt(T0)
            
            alpha_,rescaledepth,it2,reset,reject,warmup=utils.determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,it2,itreset,reset)
            if reject==True:
                Ql=np.copy(Qlbackup)
                Rl=np.copy(Rlbackup)
                lam=np.copy(lamold)
                Vl=np.copy(Vlbackup)
                Wl0=np.copy(Wl0backup)
                Wl1=np.copy(Wl1backup)
                Ql=(Ql-alpha_*Vl)
                Rl[0]=(Rl[0]-alpha_*Wl0)
                Rl[1]=(Rl[1]-alpha_*Wl1)
                print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(normxopt,normxoptold,normtol))
            if reject==False:
                Vlbackup=np.copy(Vl)
                Wl0backup=np.copy(Wl0)
                Wl1backup=np.copy(Wl1)
                Qlbackup=np.copy(Ql)
                Rlbackup=np.copy(Rl)
                Ql=(Ql-alpha_*Vl)
                Rl[0]=(Rl[0]-alpha_*Wl0)
                Rl[1]=(Rl[1]-alpha_*Wl1)
                normxoptold=normxopt
                lamold=np.copy(lam)

            #Ql=Ql-dt*Vlam.dot(np.diag(1.0/lam))
            #Rl[0]=Rl[0]-dt*Wlam[0].dot(np.diag(1.0/lam))
            #Rl[1]=Rl[1]-dt*Wlam[1].dot(np.diag(1.0/lam))
            #print('it={0}, ||x||={1}, <h>={2}, <n>={3}, dt={4}'.format(it,np.real(normxopt),np.real(meanh),np.real(meandensity),dt))
            #stdout.write("\rit %i: ||x||= %.8f, <h> =[%.8f, %.8f], <n>=[%.8f,%.8f], <e>= %.8f, tr({R1,R2}*herm({R1,R2}))=%.8f, ||{R1,R2}||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meanh[0]),np.real(meanh[1]),np.real(meandensity[0]),np.real(meandensity[1]),\
            #                                                                                                                                               np.real(meanE),tranticommR1R2,np.linalg.norm(anticommR1R2),alpha_))
            stdout.write("\rit %i: ||x||= %.8f, <k> =[%.8f, %.8f], <n>=[%.8f,%.8f], <e>= %.8f, tr({R1,R2}*herm({R1,R2}))=%.8f, ||{R1,R2}||=%.8f, alpha=% f" %(it,np.real(normxopt),np.real(meank[0]),np.real(meank[1]),np.real(meandensity[0]),np.real(meandensity[1]),\
                                                                                                                                                           np.real(meanE),tranticommR1R2,np.linalg.norm(anticommR1R2),alpha_))

            stdout.flush()

            if it%100==0:
                print
            if normxopt<epsilon:
                converged=True

            if it>=itmax:
                converged=True
        #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,Vdag,x,invx,y,invy,eta=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',initial=np.reshape(np.diag(lamold**2),D*D),datatype=dtype,nmaxit=50000,\
        #                                                                                     tol=regaugetol,ncv=ncv,numeig=numeig)
        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Ql,Rl,0.0,gauge='symmetric',linitial=np.reshape(np.diag(lamold**2),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,\
                                                                    tol=regaugetol,ncv=ncv,numeig=numeig)
        
        return lam,Ql,Rl,Qr,Rr,it,normxopt



    def __simulatetdvp__(self,Q,R,dx,dt,dtype,mu0,inter,mass,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-10,itsave=100,it0=0):

        D=np.shape(Q)[0]
        rold=np.random.rand(D*D)*1.0
        kold=np.random.rand(D*D)*1.0
        lamold=np.ones(D)/np.sqrt(D)
        kleft=np.eye(D)

        it=0
        converged=False
        #plt.figure(101)
        #plt.clf()
        init=np.eye(D)*1.0/D
        while converged==False:
            it=it+1
            #lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv=cmf.regauge(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lamold),D*D),datatype=dtype,nmaxit=10000,tol=regaugetol)
            rnorm=1.0
            lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,dx,gauge='symmetric',linitial=np.reshape(init,D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=50000,tol=regaugetol)
            left=Ql+herm(Ql)+dx*herm(Ql).dot(Ql)
            right=Qr+herm(Qr)+dx*Qr.dot(herm(Qr))
            for n in range(len(Rl)):
                left=left+herm(Rl[n]).dot(Rl[n])
                right=right+Rr[n].dot(herm(Rr[n]))
            lnorm=np.linalg.norm(left)
            rnorm=np.linalg.norm(right)
            print ('right normalization: ',np.linalg.norm(rnorm))
            print ('left normalization:',np.linalg.norm(lnorm))
            
            if rnorm>1E-4:
                print ('rnorm > {0}; reiterating regauging procedure'.format(rnorm))
                init=np.diag(np.random.rand(D))
                init=init/np.sqrt(np.trace(init.dot(herm(init))))
                
            lamold=np.copy(lam)
            f=np.zeros((D,D),dtype=dtype)
            l=np.eye(D)
            r=np.diag(lam)
            ih=cmf.homogeneousdfdxLiebLinigerMultiSpecies(Ql,Rl,Ql,Rl,dx,f,mu0,mass,inter,direction=1)
            #ih=cmf.homogeneousdfdxLiebLiniger(Ql,Rl[0],Ql,Rl[0],dx,f,mu0,mass,inter,direction=1)
            ihprojected=-(ih-np.trace(np.transpose(ih).dot(r))*l)
            kleft=cmf.inverseTransferOperatorMultiSpecies(Ql,Rl,dx,l,r,ihprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            #kleft=cmf.inverseTransferOperator(Ql,Rl[0],dx,l,r,ihprojected,direction=1,x0=np.reshape(kleft,D*D),tolerance=lgmrestol,maxiteration=4000)
            meanh=np.zeros(2)
            for n in range(len(Rl)):
                meanh[n]=1.0/(2.0*mass[n])*np.trace(comm(Ql,Rl[n]).dot(np.diag(lam**2)).dot(herm(comm(Ql,Rl[n]))))+inter[n]*np.trace(Rl[n].dot(Rl[n]).dot(np.diag(lam**2)).dot(herm(Rl[n])).dot(herm(Rl[n])))
            #Wlam=[cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl[0],Qr,Rr[0],kleft,np.diag(lam),mass,mu0,inter,dx,direction=1)]      
            Wlam=cmf.HomogeneousLiebLinigerCMPSTDVPHAproductMultiSpecies(Ql,Rl,Qr,Rr,kleft,np.diag(lam),mass,mu0,inter,dx,direction=1)          

            #xopt=cmf.HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl[0],Qr,Rr[0],kleft,np.diag(lam),mass,mu0,inter,dx,direction=1)
            #Wlam=[np.copy(xopt)]
            #W=[Wlam[0].dot(np.diag(1.0/lam))]
            
            #normxopt=np.linalg.norm(Wlam[0])
            #if dx>1E-6:
            #    Q=Ql+dt*np.linalg.inv(np.eye(D)+dx*herm(Ql)).dot(herm(Rl[0])).dot(W[0])
            #if dx<=1E-6:
            #    Q=Ql+dt*herm(Rl[0]).dot(W[0])

            #R[0]=Rl[0]-dt*xdot

            #W=[Wlam[0].dot(np.diag(1.0/lam))]
            #Wlam=cmf.HomogeneousLiebLinigerCMPSTDVPHAproductMultiSpecies(Ql,Rl,Qr,Rr,kleft,np.diag(lam),mass,mu0,inter,dx,direction=1)
            W=[]
            W2=[]
            P=[]
            for n in range(len(Wlam)):
                P.append(np.diag(np.diag(Glinv.dot(Wlam[n].dot(np.diag(1.0/lam))).dot(Gl))))
                #P.append(Glinv.dot(Wlam[n].dot(np.diag(1.0/lam))).dot(Gl))
                #W.append(Gl.dot(P[-1]).dot(Glinv))
                W.append(Wlam[n].dot(np.diag(1.0/lam)))

            normxopt=0
            for n in range(len(Wlam)):
                normxopt=normxopt+np.linalg.norm(W[n].dot(np.diag(lam)))

            if dx>1E-6:
                Q=np.copy(Ql)
                for n in range(len(W)):
                    Q=Q+dt*np.linalg.inv(np.eye(D)+dx*herm(Ql)).dot(herm(Rl[n])).dot(W[n])
                
            if dx<=1E-6:
                Q=np.copy(Ql)
                for n in range(len(W)):
                    Q=Q+dt*herm(Rl[n]).dot(W[n])

            for n in range(len(W)):
                #R[n]=Glinv.dot(Rl[n]).dot(Gl)-dt*P[n]
                R[n]=Rl[n]-dt*W[n]

            #eta,X=np.linalg.eig(R[0])
            #print X.dot(np.diag(eta)).dot(np.linalg.inv(X))-R[0]
            #raw_input()
            #print Glinv.dot(Gl)
            #print Gl.dot(Glinv)
            #now make R again explicitly diagonal
            #for n in range(len(W)):
            #    #R[n]=Glinv.dot(R[n]).dot(Gl)
            #    R[n]=np.diag(eta)
            #    Q=np.linalg.inv(X).dot(Q).dot(X)
            #for n in range(len(Rl)):
            #    print Glinv.dot(Rl[n]).dot(Gl)
            #raw_input()
            print ('at iteration step {0}: norm(x) = {1},cmps h = {2},dx={3}, dt={4},lnorm={5},rnorm={6}'.format(it,normxopt,meanh,dx,dt,lnorm,rnorm))
            if normxopt<epsilon:
                converged=True
            if it>itmax:
                converged=True

        lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=cmf.regauge(Q,R,dx,gauge='symmetric',linitial=np.reshape(np.diag(lamold),D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=10000,tol=regaugetol)
        return lam,Ql,Rl,Qr,Rr,it,normxopt,it+it0






#for homogeneous states
class HomogeneouscMPSEnginePhiFour:
    def __init__(self, filename,Ql,Rl,nxdots,dt,dts,dtype,mass,inter,cutoff,itreset=10,itmax=10000,regaugetol=1E-10,lgmrestol=1E-10,epsilon=1E-6,acc=1E-4,itsave=100,verbosity=1,warmuptol=1E-6,\
                 rescalingfactor=2.0,normtolerance=0.1,initnormtolerance=0.1,initthresh=0.01,numeig=5,ncv=100,Nmaxlgmres=100,outerklgmres=20,innermlgmres=30,nlcgupperthresh=1E-16,\
                 nlcglowerthresh=1E-100,nlcgreset=10,stdereset=3,nlcgnormtol=0.0,single_layer=False,pinv=1E-10):
        self._pinv=pinv
        self._eigenaccuracy=acc
        self._D=np.shape(Ql)[0]
        self._filename=filename
        self._Ql=np.copy(Ql)
        self._Rl=np.copy(Rl)
        self._Qr=np.copy(Ql)
        self._Rr=np.copy(Rl)

        if nxdots!=None:
            self._nxdots=np.copy(nxdots)
        elif nxdots==None:
            self._nxdots=nxdots

        self._dt=dt
        self._dt_=dt
        self._dts=np.copy(dts)

        self._dx=0.0
        self._dtype=dtype
        self._mass=mass
        self._cutoff=cutoff
        self._inter=inter
        self._itmax=itmax
        self._regaugetol=regaugetol
        self._lgmrestol=lgmrestol
        self._epsilon=epsilon
        self._itsave=itsave
        self._itreset=itreset
        if self._nxdots!=None:
            assert((len(self._dts))==len(self._nxdots))
        self._verbosity=verbosity
        self._warmuptol=warmuptol
        self._factor=rescalingfactor
        self._normtol=normtolerance
        self._initnormtol=initnormtolerance
        self._initthresh=initthresh
        self._numeig=numeig
        self._ncv=ncv
        self._Nmaxlgmres=Nmaxlgmres
        self._outerklgmres=outerklgmres
        self._innermlgmres=innermlgmres
        self._nlcgupperthresh=nlcgupperthresh
        self._nlcglowerthresh=nlcglowerthresh
        self._nlcgreset=nlcgreset
        self._stdereset=stdereset
        self._nlcgnormtol=nlcgnormtol
        self._single_layer_regauge=single_layer
        
        self._it=0
        self._warmup=True
        self._lamold=np.ones(self._D)/np.sqrt(self._D)
        self._Hl=np.eye(self._D)
        self._Hr=np.eye(self._D)
        self._reset=True
        self._it2=0
        self._Qlbackup=np.copy(self._Ql)
        self._Rlbackup=np.copy(self._Rl)
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    #eset all iterators and flags, so one can __load__() a file, and start a fresh simulation; don't forget to 
    #set the new parameters after rest, if you don't want to use the ones from the __load__()ed file
    def __reset__(self):
        self._it=0
        self._warmup=True
        self._reset=True
        self._it2=0
        self._norms=np.zeros(0,dtype=self._dtype)
        self._totalEnergy=np.zeros(0,dtype=self._dtype)
        self._kinpluspot=np.zeros(0,dtype=self._dtype)
        self._meanDenss=np.zeros(0,dtype=self._dtype)
        self._rescaledepth=0
        self._normxoptold=1E10
        self._normxopt=1E10
        #self._Vlbackup=np.zeros((self._D,self._D),dtype=self._dtype)
        #self._Wlbackup=np.zeros((self._D,self._D),dtype=self._dtype)

    def __cleanup__(self):
        cwd=os.getcwd()
        if not os.path.exists('CHECKPOINT_'+self._filename):
            return
        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)


    #dump the simulation into a folder for later retrieval with ___load__()
    def __dump__(self):
        cwd=os.getcwd()
        #raw_input(not os.path.exists('CHECKPOINT_'+self._filename))
        if not os.path.exists('CHECKPOINT_'+self._filename):
            os.mkdir('CHECKPOINT_'+self._filename)

        elif os.path.exists('CHECKPOINT_'+self._filename):
            shutil.rmtree('CHECKPOINT_'+self._filename)
            os.mkdir('CHECKPOINT_'+self._filename)

        os.chdir('CHECKPOINT_'+self._filename)


        intparams=np.zeros(0,dtype=int)
        floatparams=np.zeros(0,dtype=float)
        intparams=np.append(intparams,self._D)
        intparams=np.append(intparams,self._itmax)
        intparams=np.append(intparams,self._itsave)
        intparams=np.append(intparams,self._it)
        intparams=np.append(intparams,self._it2)
        intparams=np.append(intparams,self._itreset)
        intparams=np.append(intparams,self._rescaledepth)
        intparams=np.append(intparams,self._verbosity)
        intparams=np.append(intparams,self._numeig)
        intparams=np.append(intparams,self._ncv)
        intparams=np.append(intparams,self._nlcgreset)
        intparams=np.append(intparams,self._stdereset)
        intparams=np.append(intparams,self._Nmaxlgmres)
        intparams=np.append(intparams,self._outerklgmres)
        intparams=np.append(intparams,self._innermlgmres)




        floatparams=np.append(floatparams,self._dt)
        floatparams=np.append(floatparams,self._dt_)
        floatparams=np.append(floatparams,self._dx)
        floatparams=np.append(floatparams,np.real(self._mass))
        floatparams=np.append(floatparams,self._inter)
        floatparams=np.append(floatparams,self._regaugetol)
        floatparams=np.append(floatparams,self._lgmrestol)
        floatparams=np.append(floatparams,self._epsilon)
        floatparams=np.append(floatparams,self._normxopt)
        floatparams=np.append(floatparams,self._normxoptold)
        floatparams=np.append(floatparams,self._eigenaccuracy)
        floatparams=np.append(floatparams,self._warmuptol)
        floatparams=np.append(floatparams,self._factor)
        floatparams=np.append(floatparams,self._normtol)
        floatparams=np.append(floatparams,self._nlcgupperthresh)
        floatparams=np.append(floatparams,self._nlcglowerthresh)
        floatparams=np.append(floatparams,self._nlcgnormtol)
        floatparams=np.append(floatparams,self._initnormtol)
        floatparams=np.append(floatparams,self._initthresh)
        floatparams=np.append(floatparams,self._cutoff)
        floatparams=np.append(floatparams,np.imag(self._mass))
        #floatparams=np.append(floatparams,self._pinv)

        boolparams=np.empty((0),dtype=bool)
        boolparams=np.append(boolparams,self._warmup)        
        boolparams=np.append(boolparams,self._reset)        


        np.save('intparams',intparams)
        np.save('floatparams',floatparams)
        np.save('boolparams',boolparams)
        np.save('Ql',self._Ql)
        np.save('Rl',self._Rl)
        np.save('Qr',self._Qr)
        np.save('Rr',self._Rr)
        if self._nxdots!=None:
            np.save('nxdots',self._nxdots)
        if self._nxdots==None:
            np.save('nxdots',np.zeros(0))

        np.save('dts',self._dts)
        np.save('lamold',self._lamold)
        np.save('kleft',self._Hl)
        np.save('kright',self._Hr)
        np.save('Qlbackup',self._Qlbackup)
        np.save('Rlbackup',self._Rlbackup)
        np.save('norms',self._norms)
        np.save('totalEnergy',self._totalEnergy)
        np.save('kinpluspot',self._kinpluspot)
        np.save('meanDenss',self._meanDenss)
        np.save('Vlbackup',self._Vlbackup)
        np.save('Wlbackup',self._Wlbackup)
        os.chdir(cwd)
        
    #load a simulation from a folder named filename
    def __load__(self,filename):
        os.chdir(filename)
        
        intparams=np.load('intparams.npy')
        floatparams=np.load('floatparams.npy')
        boolparams=np.load('boolparams.npy')

        self._D=int(intparams[0])
        self._itmax=int(intparams[1])
        self._itsave=int(intparams[2])
        self._it=int(intparams[3])
        self._it2=int(intparams[4])
        self._itreset=int(intparams[5])
        self._rescaledepth=int(intparams[6])
        self._verbosity=int(intparams[7])
        self._numeig=int(intparams[8])
        self._ncv=int(intparams[9])
        self._nlcgreset=int(intparams[10])
        self._stdereset=int(intparams[11])
        self._Nmaxlgmres=int(intparams[12])
        self._outerklgmres=int(intparams[13])
        self._innermlgmres=int(intparams[14])


        self._dt=np.real(floatparams[0])
        self._dt_=np.real(floatparams[1])
        self._dx=np.real(floatparams[2])
        self._mass=np.real(floatparams[3])+1j*np.real(floatparams[20])
        self._inter=np.real(floatparams[4])
        self._regaugetol=np.real(floatparams[5])
        self._lgmrestol=np.real(floatparams[6])
        self._epsilon=np.real(floatparams[7])
        self._normxopt=np.real(floatparams[8])
        self._normxoptold=np.real(floatparams[9])
        self._eigenaccuracy=np.real(floatparams[10])
        self._warmuptol=np.real(floatparams[11])
        self._factor=np.real(floatparams[12])
        self._normtol=np.real(floatparams[13])
        self._nlcgupperthresh=np.real(floatparams[14])
        self._nlcglowerthresh=np.real(floatparams[15])
        self._nlcgnormtol=np.real(floatparams[16])
        self._initnormtol=np.real(floatparams[17])
        self._initthresh=np.real(floatparams[18])
        self._cutoff=np.real(floatparams[19])
        #self._pinv=np.real(floatparams[20])

        self._warmup=boolparams[0]
        self._reset=boolparams[1]

        self._Ql=np.load('Ql.npy')
        self._Rl=np.load('Rl.npy')
        self._Qr=np.load('Qr.npy')
        self._Rr=np.load('Rr.npy')
        nxdots=np.load('nxdots.npy')

        if len(nxdots==0):
            self._nxdots=None

        if len(nxdots!=0):
            self._nxdots=np.copy(nxdots)

        self._dts=np.load('dts.npy')
        self._lamold=np.load('lamold.npy')
        self._Hl=np.load('kleft.npy')
        self._Hr=np.load('kright.npy')
        self._Qlbackup=np.load('Qlbackup.npy')
        self._Rlbackup=np.load('Rlbackup.npy')
        self._norms=np.load('norms.npy')
        self._totalEnergy=np.load('totalEnergy.npy')
        self._kinpluspot=np.load('kinpluspot.npy')
        self._meanDenss=np.load('meanDenss.npy')
        self._Vlbackup=np.load('Vlbackup.npy')
        self._Wlbackup=np.load('Wlbackup.npy')
        os.chdir('../')



    def __simulate__(self):
        plt.ion()

        printnlcgmessage=True
        printstdemessage=True
        itbeta=0
        itstde=0
        dostde=False
        converged=False
        phiold=np.ones(self._D)
        Vl=np.zeros((self._D,self._D),dtype=self._dtype)
        Wl=np.zeros((self._D,self._D),dtype=self._dtype)
        
        if self._itmax==self._it:
            converged=True
            print ('found self._itmax=self._it={0}, leaving simulation'.format(self._itmax))


        t1=time.time()
        while converged==False:

            self._it=self._it+1
            if self._it%100==0:
                t2=time.time()
                print('time for 100 iterations: {0}'.format(t2-t1))
                t1=t2
            #print self._normxopt,self._eigenaccuracy
            if self._normxopt<self._eigenaccuracy:
                rtol=self._regaugetol
                lgmrestol=self._lgmrestol
            if self._normxopt>=self._eigenaccuracy:
                rtol=self._warmuptol
                lgmrestol=self._warmuptol
            if self._normxopt<self._initthresh:
                normtol=self._normtol
            if self._normxopt>=self._initthresh:
                normtol=self._initnormtol
            if self._single_layer_regauge==False:

                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge_return_basis(self._Ql,self._Rl,self._dx,gauge='symmetric',\
                                                                                                                      linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                                      rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                      nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv,trunc=1E-16,pinv=self._pinv)
            if len(self._lam)!=self._D:
                print(self._D,len(self._lam))
                self._D=len(self._lam)
                self._Hr=np.eye(self._D)
                self._Hl=np.eye(self._D)                
                #print(self._lam)
                #input('asdf')
            if self._single_layer_regauge==True:
                self._lam,self._Ql,self._Rl,self._Qr,self._Rr,Gl,Glinv,Gr,Grinv=cmf.singleLayerRegauge_return_basis(self._Ql,self._Rl,self._dx,deltax=0.01,\
                                                                                                                    initial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                                    nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)
            
                lNit=None
                rNit=None
            #wick,full=checkWick(self._lam,self._Rl)
            #print()
            #print(wick)
            #print(full)
            #print((wick-full))
            if self._it%self._itsave==0:
                self.__dump__()

                if self._verbosity>0:
                    print('   (checkpointing)   ')

                its=range(len(self._norms))
                np.save('norms'+self._filename,self._norms)
                np.save('totalEnergy'+self._filename,self._totalEnergy)
                np.save('meanDenss'+self._filename,self._meanDenss)

            if self._verbosity==3:
                if dx>=1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr))+self._dx*self._Qr.dot(herm(self._Qr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl)+self._dx*herm(self._Ql).dot(self._Ql))/self._D
                elif df<1E-8:
                    rnorm=np.linalg.norm(self._Qr+herm(self._Qr)+self._Rr.dot(herm(self._Rr)))/self._D
                    lnorm=np.linalg.norm(self._Ql+herm(self._Ql)+herm(self._Rl).dot(self._Rl))/self._D


            #ihl=cmf.homogeneousdfdxPhiFour(self._Ql,self._Rl,f,self._mass,self._inter,self._cutoff,direction=1)
            #ihr=cmf.homogeneousdfdxPhiFour(self._Qr,self._Rr,f,self._mass,self._inter,self._cutoff,direction=-1)
            f=np.zeros((self._Ql.shape),dtype=self._dtype)            
            ihl=cmf.homogeneousdfdxPhiFour1(self._Ql,self._Rl,f,self._mass,self._inter,self._cutoff,direction=1)
            ihr=cmf.homogeneousdfdxPhiFour1(self._Qr,self._Rr,f,self._mass,self._inter,self._cutoff,direction=-1)
            #ihl=cmf.homogeneousdfdxPhiFourPenalty(self._Ql,self._Rl,f,self._mass,self._inter,self._cutoff,direction=1,penalty=0.0)
            #ihr=cmf.homogeneousdfdxPhiFourPenalty(self._Qr,self._Rr,f,self._mass,self._inter,self._cutoff,direction=-1,penalty=0.0)

            meantotal=np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))

            ihlprojected=-(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
            ihrprojected=-(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
            Hlregauged=np.transpose(herm(Glinv).dot(np.transpose(self._Hl)).dot(Glinv))
            #randominit=np.random.rand((self._D*self._D))
            self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=np.reshape(self._Hl,self._D*self._D),\
                                                   tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)


            self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=np.reshape(self._Hr,self._D*self._D),\
                                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            #self._Hl,l_lgmresNit=cmf.inverseTransferOperator(self._Ql,self._Rl,self._dx,np.eye(self._D),np.diag(self._lam**2),ihlprojected,direction=1,x0=randominit,\
            #                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)
            #self._Hr,r_lgmresNit=cmf.inverseTransferOperator(self._Qr,self._Rr,self._dx,np.diag(self._lam**2),np.eye(self._D),ihrprojected,direction=-1,x0=randominit,\
            #                                       tolerance=self._lgmrestol,maxiteration=self._Nmaxlgmres,outer_k=self._outerklgmres,inner_m=self._innermlgmres)

            I1=np.eye(self._D)
            I1bar=self._Rl.dot(self._Rl).dot(self._Rl).dot(self._Rl)
            I2=self._Rl
            I2bar=self._Rl.dot(self._Rl).dot(self._Rl)
            I3=self._Rl.dot(self._Rl)
            I3bar=self._Rl.dot(self._Rl)
            I4=self._Rl.dot(self._Rl).dot(self._Rl)
            I4bar=self._Rl
            I5=self._Rl.dot(self._Rl).dot(self._Rl).dot(self._Rl)
            I5bar=np.eye(self._D)
            meantotal=np.trace(comm(self._Ql,self._Rl).dot(np.diag(self._lam**2)).dot(herm(comm(self._Ql,self._Rl))))+\
                       (self._mass**2+self._cutoff**2)/(2.0)*np.trace(self._Rl.dot(np.diag(self._lam**2)).dot(herm(self._Rl)))+\
                       (self._mass**2-self._cutoff**2)/(4.0)*(np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))+np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl))))+\
                       np.trace(6*self._inter/(96*self._cutoff)*np.transpose(I3.dot(np.diag(self._lam**2)).dot(herm(I3bar))))+\
                       np.trace(4*self._inter/(96*self._cutoff)*np.transpose(I2.dot(np.diag(self._lam**2)).dot(herm(I2bar))))+\
                       np.trace(4*self._inter/(96*self._cutoff)*np.transpose(I4.dot(np.diag(self._lam**2)).dot(herm(I4bar))))+\
                       np.trace(self._inter/(96*self._cutoff)*np.transpose(I5.dot(np.diag(self._lam**2)).dot(herm(I5bar))))+\
                       np.trace(self._inter/(96*self._cutoff)*np.transpose(I1.dot(np.diag(self._lam**2)).dot(herm(I1bar))))

            pd_pd_p_p=np.trace(6*self._inter/(96*self._cutoff)*np.transpose(I3.dot(np.diag(self._lam**2)).dot(herm(I3bar))))
            pd_p_p_p=np.trace(4*self._inter/(96*self._cutoff)*np.transpose(I2.dot(np.diag(self._lam**2)).dot(herm(I2bar))))
            pd_pd_pd_p=np.trace(4*self._inter/(96*self._cutoff)*np.transpose(I4.dot(np.diag(self._lam**2)).dot(herm(I4bar))))
            p_p_p_p=np.trace(self._inter/(96*self._cutoff)*np.transpose(I5.dot(np.diag(self._lam**2)).dot(herm(I5bar))))
            pd_pd_pd_pd=np.trace(self._inter/(96*self._cutoff)*np.transpose(I1.dot(np.diag(self._lam**2)).dot(herm(I1bar))))            

            mat=np.copy(self._Rl)

            #np.trace(self._inter/(96*self._cutoff)*np.transpose(I1.dot(np.diag(self._lam**2)).dot(herm(I1bar)))+\
                #4*self._inter/(96*self._cutoff)*np.transpose(I2.dot(np.diag(self._lam**2)).dot(herm(I2bar)))+\
                #4*self._inter/(96*self._cutoff)*np.transpose(I4.dot(np.diag(self._lam**2)).dot(herm(I4bar)))+\
                #self._inter/(96*self._cutoff)*np.transpose(I5.dot(np.diag(self._lam**2)).dot(herm(I5bar))))
            
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))
            meanpsi=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)))
            meanpsipsi=np.trace(self._Rl.dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsidagpsidagpsipsi=np.trace(herm(self._Rl.dot(self._Rl)).dot(self._Rl).dot(self._Rl).dot(np.diag(self._lam**2)))
            meanpsidagpsidag=np.trace(np.diag(self._lam**2).dot(herm(self._Rl)).dot(herm(self._Rl)))
            meandensity=np.trace(self._Rl.dot(np.diag(self._lam)).dot(np.diag(self._lam)).dot(herm(self._Rl)))

            #print (meanpsidagpsidagpsipsi,2*meandensity*meandensity+meanpsipsi*meanpsidagpsidag)
            
            #Vlam,Wlam=cmf.HomogeneousPhiFourGradient(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
            #                                         self._Hl,self._Hr,np.diag(self._lam),self._mass,self._inter,self._cutoff)
            #Vlam,Wlam=cmf.HomogeneousPhiFourGradient2(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
            #                                          self._Hl,self._Hr,np.diag(self._lam),self._mass,self._inter,self._cutoff,penalty=0.0)
            Vlam,Wlam=cmf.HomogeneousPhiFourGradient1(self._Ql,self._Rl,self._Qr,self._Rr,self._Ql.dot(np.diag(self._lam)),self._Rl.dot(np.diag(self._lam)),\
                                                      self._Hl,self._Hr,np.diag(self._lam),self._mass,self._inter,self._cutoff)

            
            
            invlam=np.copy(1.0/self._lam)
            invlam[self._lam<self._pinv]=0.0
            Vl_=Vlam.dot(np.diag(invlam))
            Wl_=Wlam.dot(np.diag(invlam))

            self._normxopt=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(Wlam.dot(herm(Wlam))))
            self._normV=np.sqrt(np.trace(Vlam.dot(herm(Vlam))))
            self._normW=np.sqrt(np.trace(Wlam.dot(herm(Wlam))))


            self._dt_,self._rescaledepth,self._it2,self._reset,self._reject,self._warmup=utils.determineNewStepsize(self._dt_,self._dt,self._dts,self._nxdots,self._normxopt,\
                                                                                                                    self._normxoptold,normtol,self._warmup,self._it,self._rescaledepth,self._factor,\
                                                                                                                    self._it2,self._itreset,self._reset)

            if self._reject==True:
                self._Ql=np.copy(self._Qlbackup)
                self._Rl=np.copy(self._Rlbackup)
                self._lam=np.copy(self._lamold)
                Vl=np.copy(self._Vlbackup)
                Wl=np.copy(self._Wlbackup)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)

                if self._verbosity>0:
                    print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normxopt,self._normxoptold,normtol))

            if self._reject==False:
                betanew,itstde,itbeta,dostde,printnlcgmessage,printstdemessage=utils.determineNonLinearCGBeta(self._nlcgupperthresh,self._nlcglowerthresh,self._nlcgnormtol,self._nlcgreset,self._normxopt,\
                                                                                                              self._normxoptold,self._it,itstde,self._stdereset,dostde,itbeta,printnlcgmessage,printstdemessage)

                    
                Vl=Vl_+betanew*Gl.dot(Vl).dot(Glinv)
                Wl=Wl_+betanew*Gl.dot(Wl).dot(Glinv)
                self._Vlbackup=np.copy(Vl_)
                self._Wlbackup=np.copy(Wl_)
                self._Qlbackup=np.copy(self._Ql)
                self._Rlbackup=np.copy(self._Rl)
                self._Ql=(self._Ql-self._dt_*Vl)
                self._Rl=(self._Rl-self._dt_*Wl)
                
                self._normxoptold=self._normxopt
                #print (np.linalg.norm(self._lam-self._lamold)/self._dt_)
                self._lamold=np.copy(self._lam)

                self._norms=np.append(self._norms,self._normxopt)
                self._totalEnergy=np.append(self._totalEnergy,meantotal)
                self._meanDenss=np.append(self._meanDenss,meandensity)




                if self._verbosity==1:
                    stdout.write("\rit %i: ||x|| = %.8f, ||V||= %.8f, ||W|| = %.8f, <E>=%.8f, <n>=%.8f, <psi>=%.8f+i%.8f, |<psi>|=%.8f, dt=% f, beta=%.3f" %(self._it,np.real(self._normxopt),np.real(self._normV),np.real(self._normW),np.real(meantotal),np.real(meandensity),\
                                                                                                                    np.real(meanpsi),np.imag(meanpsi),np.linalg.norm(meanpsi),self._dt_,np.real(betanew)))
                
                if self._verbosity==2:
                    stdout.write("\rit %i: ||x|| = %.8f, ||V||= %.8f, ||W|| = %.8f, <E>=%.8f, <n>=%.8f, <psi>=%.8f+i%.8f, |<psi>|=%.8f, dt=%.8f, beta=%.3f, lnorm=%.8f, rnorm=%.8f, l Nlgmres=%i, r Nlgmres=%i, l regIt=%i, r regIt=%i" \
                                 %(self._it,np.real(self._normxopt),np.real(self._normV),np.real(self._normW),np.realm(eantotal),np.real(meandensity),np.real(meanpsi),np.imag(meanpsi),np.linalg.norm(meanpsi),\
                                   self._dt_,np.real(betanew),lnorm,rnorm,l_lgmresNit,r_lgmresNit,lNit,rNit))



            stdout.flush()
            if self._normxopt<self._epsilon:
                converged=True

            if self._it>=self._itmax:
                converged=True

        np.save('norms'+self._filename,self._norms)
        np.save('totalEnergy'+self._filename,self._totalEnergy)
        np.save('meanDenss'+self._filename,self._meanDenss)
        if self._single_layer_regauge==False:
            #self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.regauge_old(self._Ql,self._Rl,self._dx,gauge='symmetric',initial=np.reshape(np.diag(self._lamold),self._D*self._D),nmaxit=10000,\
            #                                                          tol=self._regaugetol,numeig=self._numeig,ncv=self._ncv)

            self._lam,self._Ql,_Rl,self._Qr,_Rr,Gl,Glinv,Gr,Grinv,Zl,lNit,rNit=cmf.regauge(self._Ql,[self._Rl],self._dx,gauge='symmetric',\
                                                                                                       linitial=np.reshape(np.eye(self._D),self._D*self._D),\
                                                                                                       rinitial=np.reshape(np.diag(self._lamold**2),self._D*self._D),\
                                                                                                       nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

            self._Rl=_Rl[0]
            self._Rr=_Rr[0]

        if self._single_layer_regauge==True:
            self._lam,self._Ql,self._Rl,self._Qr,self._Rr=cmf.singleLayerRegauge(self._Ql,self._Rl,self._dx,deltax=0.01,initial=np.reshape(np.diag(self._lamold),self._D*self._D),\
                                                                                 datatype=self._dtype,nmaxit=10000,tol=rtol,numeig=self._numeig,ncv=self._ncv)

        return self._lam,self._Ql,self._Rl,self._Qr,self._Rr,self._it,self._normxopt
