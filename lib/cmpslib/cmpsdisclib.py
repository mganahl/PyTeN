#!/usr/bin/env python
import numpy as np
import scipy as sp
import math,copy,sys
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.interpolate import PchipInterpolator
from sys import stdout
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import warnings 
#from scipy.signal import savgol_filter
import functools as fct
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import cmpsfunctions as cmf
#import cmpsfunctions_new as cmfnew
import Hamiltonians as H
from scipy.interpolate import griddata

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)


herm=lambda x:np.conj(np.transpose(x))



def isDiagUnitary(unitary):
    assert(np.shape(unitary)[0]==np.shape(unitary)[1])
    D=np.shape(unitary)[0]
    diff=np.linalg.norm(np.abs(unitary)-np.eye(D))
    if diff<1E-10:
        return True,diff
    if diff>=1E-10:
        return False,diff
#N is the lenght of the fine grid on which grid is defined
def bisection(grid,position):
    g=np.copy(grid)
    while  True:
        N=len(g)
        i0=(N-N%2)/2
        lgrid=np.copy(g[0:i0])
        rgrid=np.copy(g[i0::])
        if (position>=lgrid[-1])&(position<=rgrid[0]):
            closestleft=np.nonzero(np.array(grid)==lgrid[-1])[0][0]
            closestright=np.nonzero(np.array(grid)==rgrid[0])[0][0]
            break
        if position<lgrid[-1]:
            g=np.copy(lgrid)
            
        if position>rgrid[0]:
            g=np.copy(rgrid)
    return closestleft,closestright

def calculateRDM(cmps,n1,n2,eps=1E-12):
    #bring the center site of the cmps to the bond left to site n1
    cmps.__position__(n1)
    ltensor=np.tensordot(cmps._mats[n1],cmps.__tensor__(n1),([1],[0]))
    #ltensor=np.random.rand(cmps._D,cmps._D,10)
    etas=[]
    S=[]
    cmps.__position__(0)
    for n in range(n1,n2):
        n_=n%cmps._N
        print n_
        if (n%cmps._N==0)&(n1>0):
            rtensor=np.tensordot(cmps.__connection__(reset_unitaries=False).dot(cmps._mats[0]),cmps.__tensor__(n_),([1],[0]))
        if n%cmps._N<>0:
            rtensor=cmps.__tensor__(n_)

        [D1r,D2r,dr]=np.shape(rtensor)
        [D1l,D2l,dl]=np.shape(ltensor)

        mpsadd1=np.tensordot(ltensor,rtensor,([1],[0])) #index ordering  0 T T 2
                                                        #                  1 3
        #compute densityt matrix
       #bla=np.tensordot(mpsadd1,np.conj(mpsadd1),([0,2],[0,2]))
        rho=np.reshape(np.tensordot(mpsadd1,np.conj(mpsadd1),([0,2],[0,2])),(dl*dr,dl*dr))   #  0  1
                                                                                             #  2  3
                                                                                             

        #
        #test=np.reshape(rho,(dl,dr,dl,dr))
        #print test - bla
        #raw_input()
        #diagonalize rho
        ind=0
        eta,u=np.linalg.eigh(rho)
        while eta[ind]<eps:
            ind=ind+1
        eta=np.copy(eta[ind::])
        etas.append(eta)
        S.append(-np.sum(eta*np.log(eta)))
        u_=u[:,ind::]

        #print np.shape(u)

        #print eta, np.sum(eta)

        #contract u with mpsadd1
        utens=np.reshape(u_,(dl,dr,len(eta)))
        ltensor=np.tensordot(mpsadd1,np.conj(utens),([1,3],[0,1]))

    return etas,S


#takes a cmps defined on a grid cmps._N and extends it to another grid Ndense
#returned state is left normalized 
def cmpsinterpolation(cmps,xbdense,order):

    assert(cmps._position==0)
    #xbdense=np.linspace(0,cmps._L,Ndense+1)
    cmpsl=copy.deepcopy(cmps)
    cmpsc=copy.deepcopy(cmps)
    cmpsr=copy.deepcopy(cmps)
    cmpsd=CMPS('homogeneous','blabla',cmps._D,cmps._L,xbdense,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    

    cmpsl.__position__(0)
    cmpsc.__position__(0)
    cmpsr.__position__(0)

    unit=cmpsl._connector.dot(cmpsl._mats[0])
    x0=cmpsl._xm-cmpsl._xb[0]
    xm=np.append(np.append(x0-cmpsl._L,x0),x0+cmpsr._L)

    cmpsl.__position__(cmpsl._N)
    Q=np.copy(np.transpose(cmpsl._Q,(1,2,0)))
    R=np.copy(np.transpose(cmpsl._R,(1,2,0)))

    cmpsc._mats[0]=cmpsl._mats[cmpsl._N].dot(cmpsl._connector).dot(cmpsc._mats[0])
    cmpsc.__position__(cmpsc._N)

    #for later
    matnew=np.copy(cmpsc._mats[cmpsc._N])
    connectornew=np.copy(cmpsc._connector)
    Unew=np.copy(cmpsc._U)
    Vnew=np.copy(cmpsc._V)
    Q=np.append(Q,np.copy(np.transpose(cmpsc._Q,(1,2,0))),axis=2)
    R=np.append(R,np.copy(np.transpose(cmpsc._R,(1,2,0))),axis=2)

    cmpsr._mats[0]=cmpsc._mats[cmpsc._N].dot(cmpsc._connector).dot(cmpsr._mats[0])
    cmpsr.__position__(cmpsr._N)
    Q=np.append(Q,np.copy(np.transpose(cmpsr._Q,(1,2,0))),axis=2)
    R=np.append(R,np.copy(np.transpose(cmpsr._R,(1,2,0))),axis=2)
    
    Qs=[]
    Rs=[]
    for n in range(Q.shape[2]):
        Qs.append(Q[:,:,n])
        Rs.append(R[:,:,n])

    Rd,Qd=cmpsinterp(Qs,Rs,xm,cmpsd._xm,k=order)
    for n in range(len(cmpsd._xm)):
        cmpsd._Q[n]=np.copy(Qd[n])
        cmpsd._R[n]=np.copy(Rd[n])

    #Qd_=np.transpose(Qd,(1,2,0))
    #Rd_=np.transpose(Rd,(1,2,0))
    #print Qd_.shape
    #D=cmps._D
    #for d1 in range(D):
    #    for d2 in range(D):
    #        plt.figure(1)
    #        plt.clf()
    #        plt.subplot(2,1,1)
    #        plt.plot(cmpsd._xm,Qd_[d1,d2,:],'x',xm,Q[d1,d2,:])
     #        plt.subplot(2,1,2)
    #        plt.plot(cmpsd._xm,np.imag(Qd_[d1,d2,:]),'x',xm,np.imag(Q[d1,d2,:]))
    #        plt.figure(2)
    #        plt.clf()
    #        plt.subplot(2,1,1)
    #        plt.plot(cmpsd._xm,Rd_[d1,d2,:],'x',xm,R[d1,d2,:])
    #        plt.subplot(2,1,2)
    #        plt.plot(cmpsd._xm,np.imag(Rd_[d1,d2,:]),'x',xm,np.imag(R[d1,d2,:]))
    #
    #
    #        plt.show()
    #        
    #        raw_input()
    cmpsd._position=cmpsd._N
    cmpsd._connector=np.copy(connectornew)
    cmpsd._U=np.copy(Unew)
    cmpsd._V=np.copy(Vnew)
    cmpsd._mats[-1]=np.copy(matnew)
    cmpsd.__position__(0)
    cmpsd._mats[0]=np.linalg.pinv(cmpsl._mats[cmpsl._N].dot(cmpsl._connector)).dot(cmpsd._mats[0])

    return cmpsd





#splits and patches a cmps at bond position index; assumes a left orthonormal mps
def patchcmps(cmps,index,gauge):
    if gauge=='left':
        assert(index>=0)
        assert(index<=cmps._N)
        cmpspatched=CMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
        cmps.__position__(cmps._N)
        #get the new ansatz mps:
        for n in range(index,cmps._N):
            cmpspatched._Q[n-index]=np.copy(cmps._Q[n])
            cmpspatched._R[n-index]=np.copy(cmps._R[n])
        #bring center site to center
        cmps.__position__(index)
        connector=np.copy(np.linalg.inv(cmps._mats[index]))
        #get the other half of the ansatz mps
        cmps.__position__(0)
        for n in range(index):
            cmpspatched._Q[cmps._N-index+n]=np.copy(cmps._Q[n])
            cmpspatched._R[cmps._N-index+n]=np.copy(cmps._R[n])
        
        cmpspatched._position=cmps._N-index
        cmpspatched._mats[cmps._N-index]=cmps._mats[cmps._N].dot(cmps.__connection__(True)).dot(cmps._mats[0])
        cmpspatched._connector=np.copy(connector)
        cmpspatched._U=np.eye(cmps._D)
        cmpspatched._V=np.eye(cmps._D)
        cmpspatched.__position__(0)
        cmpspatched.__position__(cmpspatched._N)
        cmps.__position__(cmps._N)
        return cmpspatched

    if gauge=='right':
        assert(index>=0)
        assert(index<=cmps._N)

        cmpspatched=CMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
        cmps.__position__(0)
        #get the new ansatz mps:
        for n in range(index):
            cmpspatched._Q[cmps._N-index+n]=np.copy(cmps._Q[n])
            cmpspatched._R[cmps._N-index+n]=np.copy(cmps._R[n])

        #bring center site to center
        cmps.__position__(index)
        connector=np.copy(np.linalg.inv(cmps._mats[index]))
        #get the other half of the ansatz mps
        cmps.__position__(cmps._N)


        for n in range(index,cmps._N):
            cmpspatched._Q[n-index]=np.copy(cmps._Q[n])
            cmpspatched._R[n-index]=np.copy(cmps._R[n])
        
        cmpspatched._position=cmps._N-index
        cmpspatched._mats[cmps._N-index]=cmps._mats[cmps._N].dot(cmps.__connection__(True)).dot(cmps._mats[0])
        cmpspatched._connector=np.copy(connector)
        cmpspatched._U=np.eye(cmps._D)
        cmpspatched._V=np.eye(cmps._D)
        cmpspatched.__position__(0)
        cmpspatched.__position__(cmpspatched._N)

        cmps.__position__(0)
        return cmpspatched



#splits and patches a cmps at bond position index; assumes a left orthonormal mps
def patchcmpsbasic(cmps,index):
    assert(index>=0)
    assert(index<=cmps._N)
    cmpspatched=CMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
    #get the new ansatz mps:
    for n in range(index,cmps._N):
        cmpspatched._Q[n-index]=np.copy(cmps._Q[n])
        cmpspatched._R[n-index]=np.copy(cmps._R[n])
    #get the other half of the ansatz mps
    for n in range(index):
        cmpspatched._Q[cmps._N-index+n]=np.copy(cmps._Q[n])
        cmpspatched._R[cmps._N-index+n]=np.copy(cmps._R[n])
        
    cmpspatched._position=cmps._N-index
    cmpspatched._mats[cmps._N-index]=cmps._mats[cmps._position]
    cmpspatched._connector=np.copy(cmps._connector)
    cmpspatched._U=np.copy(cmps._U)
    cmpspatched._V=np.copy(cmps._V)
    return cmpspatched


    
#cuts an mpo at "index", flips the parts and repatches them
def patchmpo(mpo,index):
    temp=[]
    for n in range(index,len(mpo)):
        temp.append(np.copy(mpo[n]))

    for n in range(0,index):
        temp.append(np.copy(mpo[n]))

    for n in range(len(mpo)):
        mpo[n]=np.copy(temp[n])



def getLiebLinigerEDens(cmps,mu,mass,inter):
    energy=np.zeros(cmps._N)
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if cmps._position==cmps._N:
        for n in range(cmps._N):
            if n<(cmps._N-1):
                mpol,mpor=getLocalLiebLinigerMPO(mu,inter,mass,n,cmps._dx,dtype=cmps._dtype,proj=True)
                ltensor=np.copy(cmps.__tensor__(n))
                rtensor=np.copy(cmps.__tensor__(n+1))
                rtensor=np.transpose(np.tensordot(rtensor,cmps._mats[n+2],([1],[0])),(0,2,1))
                left=cmf.initializeLayer(ltensor,np.eye(cmps._D),ltensor,mpor[0],direction=1)
                right=cmf.addLayer(left,rtensor,mpor[1],rtensor,1)
                energy[n]=np.trace(right[:,:,0])/((cmps._dx[n]+cmps._dx[n+1])/2)
            if n==(cmps._N-1):
                mpol,mpor=getLocalLiebLinigerMPO(mu,inter,mass,n,cmps._dx,dtype=cmps._dtype,proj=True)
                ltensor=np.copy(cmps.__tensor__(cmps._N-1))
                ltensor=np.transpose(np.tensordot(ltensor,cmps._mats[cmps._N].dot(cmps.__connection__(False)),([1],[0])),(0,2,1))
                #cmps.__position__(0)
                rtensor=np.copy(cmps.__tensor__(0))
                rtensor=np.transpose(np.tensordot(rtensor,cmps._mats[1],([1],[0])),(0,2,1))

                #rtensor=np.tensordot(cmps._mats[0],rtensor,([1],[0]))
                left=cmf.initializeLayer(ltensor,np.eye(cmps._D),ltensor,mpor[0],direction=1)
                right=cmf.addLayer(left,rtensor,mpol[1],rtensor,1)
                energy[n]=np.trace(right[:,:,0])/((cmps._dx[n]+cmps._dx[0])/2)
                #cmps.__position__(cmps._N)

    if cmps._position==0:
        for n in range(cmps._N):
            if n>0:
                mpol,mpor=getLocalLiebLinigerMPO(mu,inter,mass,n,cmps._dx,dtype=cmps._dtype,proj=True)
                rtensor=cmps.__tensor__(n)
                ltensor=np.copy(cmps.__tensor__(n-1))
                ltensor=np.tensordot(cmps._mats[n-1],ltensor,([1],[0]))
                right=cmf.initializeLayer(rtensor,np.eye(cmps._D),rtensor,mpol[1],-1)
                left=cmf.addLayer(right,ltensor,mpol[0],ltensor,-1)
                energy[n]=np.trace(left[:,:,0])/((cmps._dx[n-1]+cmps._dx[n])/2)
            if n==0:
                mpol,mpor=getLocalLiebLinigerMPO(mu,inter,mass,n,cmps._dx,dtype=cmps._dtype,proj=True)

                rtensor=cmps.__tensor__(0)
                rtensor=np.tensordot(cmps.__connection__(False).dot(cmps._mats[0]),rtensor,([1],[0]))
                ltensor=np.copy(cmps.__tensor__(cmps._N-1))
                ltensor=np.tensordot(cmps._mats[cmps._N-1],ltensor,([1],[0]))

                right=cmf.initializeLayer(rtensor,np.eye(cmps._D),rtensor,mpol[1],-1)
                left=cmf.addLayer(right,ltensor,mpol[0],ltensor,-1)
                energy[n]=np.trace(left[:,:,0])/((cmps._dx[cmps._N-1]+cmps._dx[n])/2)

    return energy

def getLiebLinigerDens(cmps):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    NUC=cmps._N
    dens=np.zeros(NUC)
    for si in range(NUC):
        if cmps._position==0:
            dens[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(cmps._R[si]).dot(herm(cmps._R[si])))
        if cmps._position==cmps._N:
            dens[si]=np.trace(herm(cmps._R[si]).dot(cmps._R[si]).dot(cmps._mats[si+1]).dot(herm(cmps._mats[si+1])))
    return dens


def getLiebLinigerDensDens(cmps):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    NUC=cmps._N
    denssq=np.zeros(NUC)
    for si in range(NUC):
        if cmps._position==0:
            if si<NUC-1:
                denssq[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(cmps._R[si]).dot(cmps._R[si+1]).dot(herm(cmps._R[si+1])).dot(herm(cmps._R[si])))
            if si==NUC-1:
                RR=cmps._R[cmps._N-1].dot(cmps.__connection__(False)).dot(cmps._mats[0]).dot(cmps._R[0])
                denssq[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(RR).dot(herm(RR)))

        if cmps._position==cmps._N:
            if si>0:
                denssq[si]=np.trace(herm(cmps._R[si]).dot(herm(cmps._R[si-1])).dot(cmps._R[si-1]).dot(cmps._R[si]).dot(cmps._mats[si+1]).dot(herm(cmps._mats[si+1])))
            if si==0:
                RR=cmps._R[cmps._N-1].dot(cmps._mats[cmps._N]).dot(cmps.__connection__(False)).dot(cmps._R[0])
                denssq[si]=np.trace(herm(RR).dot(RR).dot(cmps._mats[1].dot(herm(cmps._mats[1]))))

    return denssq
def getLiebLinigerDensDensDens(cmps):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    NUC=cmps._N
    dens3=np.zeros(NUC)
    for si in range(NUC):
        if cmps._position==0:
            if si<NUC-2:
                dens3[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(cmps._R[si]).dot(cmps._R[si+1]).dot(cmps._R[si+2]).dot(herm(cmps._R[si+2])).dot(herm(cmps._R[si+1])).dot(herm(cmps._R[si])))
            if si==NUC-2:
                RRR=cmps._R[cmps._N-2].dot(cmps._R[cmps._N-1]).dot(cmps.__connection__(False)).dot(cmps._mats[0]).dot(cmps._R[0])
                dens3[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(RRR).dot(herm(RRR)))
            if si==NUC-1:
                RRR=cmps._R[cmps._N-1].dot(cmps.__connection__(False)).dot(cmps._mats[0]).dot(cmps._R[0]).dot(cmps._R[1])
                dens3[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(RRR).dot(herm(RRR)))

        if cmps._position==cmps._N:
            if si>1:
                dens3[si]=np.trace(herm(cmps._R[si]).dot(herm(cmps._R[si-1])).dot(herm(cmps._R[si-2])).dot(cmps._R[si-2]).dot(cmps._R[si-1]).dot(cmps._R[si]).dot(cmps._mats[si+1]).dot(herm(cmps._mats[si+1])))
            if si==1:
                RRR=cmps._R[cmps._N-1].dot(cmps._mats[cmps._N]).dot(cmps.__connection__(False)).dot(cmps._R[0]).dot(cmps._R[1])
                dens3[si]=np.trace(herm(RRR).dot(RRR).dot(cmps._mats[2].dot(herm(cmps._mats[2]))))
            if si==0:
                RRR=cmps._R[cmps._N-2].dot(cmps._R[cmps._N-1]).dot(cmps._mats[cmps._N]).dot(cmps.__connection__(False)).dot(cmps._R[0])
                dens3[si]=np.trace(herm(RRR).dot(RRR).dot(cmps._mats[1].dot(herm(cmps._mats[1]))))

    return dens3

#NOTES:=======================================================================================
#r and l matrices are stored according to the following notation:
# l_i, Q_i, R_i and r_i belong together in the sense that l_i is left to Q_i, R_i and r_i is on the 
#right bond of Q_i, R_i. There is no ambiguity for either for obc or for pbv. For pbc, simply use 
#that l_{i+N}=l_i. That also means that l_i has been evolved by Q_{i-1}, R_{i-1} from site i-1 to i.


class CMPS:
    def __copy__(self):
        cmps=CMPS('homogeneous','bla',self._D,self._L,self._xb,self._dtype,scaling=0.8,epstrunc=self._epstrunc,obc=self._obc)
        for n in range(self._N):
            cmps._Q[n]=np.copy(self._Q[n])
            cmps._R[n]=np.copy(self._R[n])
            cmps._mats[n]=np.copy(self._mats[n])

        cmps._mats[self._N]=np.copy(self._mats[self._N])
        cmps._U=np.copy(self._U)
        cmps._V=np.copy(self._V)
        cmps._connector=np.copy(self._connector)
        cmps._rcond=self._rcond
        cmps._position=self._position
        cmps._dims=np.ones(self._N)*self._D        
        cmps._invmats=[]
        cmps._lams=[]
        for n in range(self._N):
            cmps._invmats.append(np.copy(self._invmats[n]))
            cmps._lams.append(np.copy(self._lams[n]))
        return cmps
        
    def __init__(self,which,gauge,D,L,x_bonds,dtype,scaling=0.5,epstrunc=1E-10,obc=False):
        self._obc=obc
        self._L=L
        self._xb=np.copy(x_bonds)
        self._xbraw=np.copy(x_bonds)
        self._N=len(self._xb)-1
        self._xm=np.zeros(self._N)
        self._dx=np.zeros(self._N)
        self._eps=epstrunc
        self._dtype=dtype
        self._D=D
        self._rcond=1E-13
        self._dims=np.ones(self._N)*self._D
        for n in range(self._N):
            self._dx[n]=self._xb[n+1]-self._xb[n]
            self._xm[n]=(self._xb[n+1]+self._xb[n])/2.0
        
        if which=='homogeneous':
            self._Q=[]
            self._R=[]
            if dtype==float:
                m1=(np.random.rand(D,D)-0.5)*scaling
                m2=(np.random.rand(D,D)-0.5)*scaling
            if dtype==complex:
                m1=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling
                m2=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling

            for n in range(self._N):
                self._Q.append(m1-herm(m1)-0.5*herm(m2).dot(m2))
                self._R.append(m2)


        if which=='random':
            self._Q=[]
            self._R=[]

            for n in range(N):
                if dtype==float:
                    m1=(np.random.rand(D,D)-0.5)*scaling
                    m2=(np.random.rand(D,D)-0.5)*scaling
                if dtype==complex:
                    m1=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling
                    m2=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling
                self._Q.append(m1)
                self._R.append(m2)


        if gauge=='left':
            #self._Q,self._R,self._left,self._right,temp=cmpsUnitcellGauging(self._Q,self._R,self._dx,gauge='symmetric',returntype=gauge,initial=np.eye(D),datatype=dtype,nmaxit=100000,tol=1E-12)
            #self._connector=np.linalg.pinv(self._left)
            #self._boundarymatrix=np.eye(self._D)
            self._mats=[]
            self._invmats=[]
            self._lams=[]
            for n in range(self._N+1):
                self._lams.append(None)
                self._mats.append(None)
                self._invmats.append(None)
 

            self._position=self._N
            self._mats[-1]=np.eye(self._D)
            self._connector=np.eye(self._D)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            self.__regauge__()

        if gauge=='right':
            #self._Q,self._R,self._left,self._right,temp=cmpsUnitcellGauging(self._Q,self._R,self._dx,gauge='symmetric',returntype=gauge,initial=np.eye(D),datatype=dtype,nmaxit=100000,tol=1E-12)
            #self._connector=np.linalg.pinv(self._right)
            #self._boundarymatrix=np.eye(self._D)
            
            self._mats=[]
            self._invmats=[]
            self._lams=[]
            for n in range(self._N+1):
                self._lams.append(None)
                self._mats.append(None)
                self._invmats.append(None)
 
            self._position=self._N
            self._mats[-1]=np.eye(self._D)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            self._connector=np.eye(self._D)
            self.__regauge__()
            self.__position__(0)

        if (gauge!='left')&(gauge!='right'):
            self._position=0

            print 'creating ungauged cmps'

            self._mats=[]
            self._invmats=[]
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            self._connector=np.eye(self._D)
            self._lams=[]
            for n in range(self._N+1):
                self._lams.append(None)
                self._mats.append(None)
                self._invmats.append(None)

            
        if self._obc==True:
            m=np.random.rand(self._D,self._D)
            u,lam,v=np.linalg.svd(m)
            lam=lam*0.0
            lam[0]=1.0
            self._mats[0]=np.copy(np.diag(lam))#u.dot(np.diag(lam)).dot(v)
            self._mats[-1]=np.copy(np.diag(lam))#u.dot(lam).dot(v)


    def __getBondDimension__(self):
        vec=np.zeros((self._N+1))
        for n in range(self._N):
            vec[n]=np.shape(self._Q[n])[0]

        vec[self._N]=np.shape(self._Q[self._N-1])[1]
        return vec


    #position is a bond-like index; if position=0, all matrices are right orthonormal, and if position = N, all matrices are left orthonormal
    def __position__bla(self,position):

        if self._obc==True:
            if position==0:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')
            if position==self._N:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')

        if position>self._N:
            print 'CMPS.__position__bla(position): position index cannot be larger than N=',self._N
            return
        if position<0:
            print 'CMPS.__position__bla(position): position index cannot be negative!'
            return
        if position==self._position:
            print 'CMPS.__position__bla(position): position index already at ',self._position
            return

        if self._position<position:
            for site in range(self._position,position):
                if site==0:
                    u,lam,v=np.linalg.svd(self._mats[site])
                    lam=lam/np.linalg.norm(lam)
                    invlam=np.ones(len(lam))
                    invlam=1.0/lam
                    self._mats[site]=np.copy(u.dot(np.diag(lam)).dot(v))
                    self._invmats[site]=np.copy(herm(v).dot(np.diag(invlam)).dot(herm(u)))
                    self._lams[site]=np.copy(lam)

                    self._Q[site],self._R[site],self._mats[site+1]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site],self._invmats[site],self._dx[site],direction=1)


                if site>0:
                    u,lam,v=np.linalg.svd(self._mats[site])
                    lam=lam/np.linalg.norm(lam)
                    invlam=np.ones(len(lam))
                    invlam=1.0/lam

                    self._mats[site]=np.copy(u.dot(np.diag(lam)).dot(v))
                    self._invmats[site]=np.copy(herm(v).dot(np.diag(invlam)).dot(herm(u)))
                    self._lams[site]=np.copy(lam)
                    self._Q[site],self._R[site],self._mats[site+1]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site],self._invmats[site],self._dx[site],direction=1)
                    

            self._position=position


        if self._position>position:
            #site can at most be N-1, self._position at most N
            for site in range(self._position-1,position-1,-1):

                if site==(self._N-1):

                    u,lam,v=np.linalg.svd(self._mats[site+1])
                    lam=lam/np.linalg.norm(lam)
                    invlam=np.ones(len(lam))
                    invlam=1.0/lam

                    self._mats[site+1]=np.copy(u.dot(np.diag(lam)).dot(v))
                    self._invmats[site+1]=np.copy(herm(v).dot(np.diag(invlam)).dot(herm(u)))
                    self._lams[site+1]=np.copy(lam)

                    self._Q[site],self._R[site],self._mats[site]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site+1],self._invmats[site+1],self._dx[site],direction=-1)
                    


                if site<(self._N-1):

                    u,lam,v=np.linalg.svd(self._mats[site+1])
                    lam=lam/np.linalg.norm(lam)
                    invlam=np.ones(len(lam))
                    invlam=1.0/lam

                    self._mats[site+1]=np.copy(u.dot(np.diag(lam)).dot(v))
                    self._invmats[site+1]=np.copy(herm(v).dot(np.diag(invlam)).dot(herm(u)))
                    self._lams[site+1]=np.copy(lam)
                    self._Q[site],self._R[site],self._mats[site]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site+1],self._invmats[site+1],self._dx[site],direction=-1)
                    

            self._position=position


    #position is a bond-like index; if position=0, all matrices are right orthonormal, and if position = N, all matrices are left orthonormal
    #if position=0, it checks that for self._mats[0]=U*D*V U is diagonal. If False, it pushes U to self._U->self._U*U
    #if position=self._N, it checks that for self._mats[self._N]=U*D*V V is diagonal. If False, it pushes V to self._V->V*self._V
    #method=0 uses prepareTensorfixA0-like orthonornmalization
    #method=1 inverts mat and puts it on the next bond prior to qr 
    def __position__(self,position,method=0,pushunitaries=False):
        if self._obc==True:
            if position==0:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')
            if position==self._N:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')

        if position>self._N:
            return
        if position<0:
            return
        if position==self._position:
            return

        if self._position<position:
            if pushunitaries==True:
                if self._position==0:
                    u,l,v=np.linalg.svd(self._mats[0])
                    l=l/np.linalg.norm(l)
                    diff=np.linalg.norm(np.abs(u)-np.eye(self._D))
                    if diff>1E-8:
                        warnings.warn('WARNING: CMPS__position__(self): in self._mats[0]=U*D*V, D is not a diagonal unitary (diff>{0}); state is probably not correctly gauged! pushing U to self._U'\
                                      .format(diff))
                        self._U=self._U.dot(u)
                        self._mats[0]=np.diag(l).dot(v)

            for site in range(self._position,position):
                self._Q[site],self._R[site],self._mats[site+1]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site],self._dx[site],direction=1,method=method)
                #if site>0:
                #    print np.abs(self._Q[site]-self._Q[site-1])
                #    print np.abs(self._R[site])-np.abs(-self._R[site-1])
                #    #print np.imag(self._mats[site])-np.imag(self._mats[site-1])
                #    raw_input()


            if position==self._N:
                if pushunitaries==True:
                    u,l,v=np.linalg.svd(self._mats[self._N])
                    l=l/np.linalg.norm(l)
                    diff=np.linalg.norm(np.abs(v)-np.eye(self._D))
                    if diff>1E-8:
                        warnings.warn('WARNING: CMPS__position__(self): in self._mats[-1]=U*D*V, V is not a diagonal unitary (diff>{0}); state is probably not correctly gauged! pushing V to self._V'\
                                      .format(diff))
                        self._V=v.dot(self._V)
                        self._mats[self._N]=u.dot(np.diag(l))                    
                        
            self._position=position
            #self.__distribute__()


        if self._position>position:
            if pushunitaries==True:
                if self._position==self._N:
                    u,l,v=np.linalg.svd(self._mats[self._N])
                    l=l/np.linalg.norm(l)
                    diff=np.linalg.norm(np.abs(v)-np.eye(self._D))
                    if diff>1E-8:
                        warnings.warn('WARNING: CMPS__position__(self): in self._mats[-1]=U*D*V, V is not a diagonal unitary (diff>{0}); state is probably not correctly gauged! pushing V to self._V'\
                                      .format(diff))
                        self._V=v.dot(self._V)
                        self._mats[self._N]=u.dot(np.diag(l))                    

            for site in range(self._position-1,position-1,-1):
                self._Q[site],self._R[site],self._mats[site]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site+1],self._dx[site],direction=-1,method=method)
            if pushunitaries==True:
                if position==0:
                    u,l,v=np.linalg.svd(self._mats[0])
                    l=l/np.linalg.norm(l)
                    diff=np.linalg.norm(np.abs(u)-np.eye(self._D))
                    if diff>1E-8:
                        warnings.warn('WARNING: CMPS__position__(self): in self._mats[0]=U*D*V, D is not a diagonal unitary (diff>{0}); state is probably not correctly gauged! pushing U to self._U'\
                                      .format(diff))
                        self._U=self._U.dot(u)
                        self._mats[0]=np.diag(l).dot(v)

            self._position=position
            #self.__distribute__()

    #position is a bond-like index; if position=0, all matrices are right orthonormal, and if position = N, all matrices are left orthonormal
    #if position=0, it checks that for self._mats[0]=U*D*V U is diagonal. If False, it pushes U to self._U->self._U*U
    #if position=self._N, it checks that for self._mats[self._N]=U*D*V V is diagonal. If False, it pushes V to self._U->V*self._U

    def __positionSVD__(self,position,fix,mode=None):
        if self._obc==True:
            if position==0:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')
            if position==self._N:
                print 'warning: you are setting position of an obc cmps to ',position,'!'
                #raw_input('do you really want to continue?')

        if position>self._N:
            #print 'CMPS.__position__(position): position index cannot be larger than N=',self._N
            return
        if position<0:
            #print 'CMPS.__position__(position): position index cannot be negative!'
            return
        if position==self._position:
            return

        if self._position<position:
            for site in range(self._position,position):
                self.__prepareTensor__(site,1,fix)

           
            self._position=position
            if mode=='diagonal':
                self.__distribute__(fix)



        if self._position>position:
            for site in range(self._position-1,position-1,-1):
                self.__prepareTensor__(site,-1,fix)
            self._position=position
            if mode=='diagonal':
                self.__distribute__(fix)


    def __wavefnct__(self,side,reset_unitaries=False):
        if side=='l':
            assert(self._position>0)
            if (self._position>1)&(self._position<self._N):
                return np.transpose(np.tensordot(self.__tensor__(self._position-1),self._mats[self._position],([1],[0])),(0,2,1))
            if (self._position==1):
                tensor=np.transpose(np.tensordot(self.__tensor__(self._position-1),self._mats[self._position],([1],[0])),(0,2,1))
                tensor=np.tensordot(self._U,tensor,([1],[0]))
                if reset_unitaries==True:
                    self._U=np.eye(self._D)
                return tensor
                
            if (self._position==self._N):
                tensor=np.transpose(np.tensordot(self.__tensor__(self._position-1),self._mats[self._position].dot(self._V),([1],[0])),(0,2,1))
                if reset_unitaries==True:
                    self._V=np.eye(self._D)
                return tensor

        if side=='r':
            assert(self._position<(self._N))
            if (self._position>0)&(self._position<(self._N-1)):
                return np.tensordot(self._mats[self._position],self.__tensor__(self._position),([1],[0]))
            if (self._position==(self._N-1)):
                tensor=np.tensordot(self._mats[self._position],self.__tensor__(self._position),([1],[0]))
                tensor=np.transpose(np.tensordot(tensor,self._V,([1],[0])),(0,2,1))
                if reset_unitaries==True:
                    self._V=np.eye(self._D)
                return tensor

            if (self._position==0):
                tensor=np.tensordot(self._U.dot(self._mats[self._position]),self.__tensor__(self._position),([1],[0]))
                if reset_unitaries==True:
                    self._U=np.eye(self._D)
                return tensor

    def __tensor__(self,site,contract_unitaries=False,reset_unitaries=False):
        assert(site>=0)
        assert(site<self._N)
                
        if (site<>0)&(site<>(self._N-1)):
            return toMPSmat(self._Q[site],self._R[site],self._dx[site])

        if (site==0):
            if contract_unitaries==False:
                return toMPSmat(self._Q[site],self._R[site],self._dx[site])
            if contract_unitaries==True:
                tensor=toMPSmat(self._Q[site],self._R[site],self._dx[site])
                tensor=np.tensordot(self._U,tensor,([1],[0]))
                if reset_unitaries==True:
                    self._U=np.eye(self._D)
                return tensor

        if (site==(self._N-1)):
            if contract_unitaries==False:
                return toMPSmat(self._Q[site],self._R[site],self._dx[site])
            if contract_unitaries==True:
                tensor=toMPSmat(self._Q[site],self._R[site],self._dx[site])
                tensor=np.transpose(np.tensordot(tensor,self._V,([1],[0])),(0,2,1))
                if reset_unitaries==True:
                    self._V=np.eye(self._D)
                return tensor


    def __connection__(self,reset_unitaries=False):
        
        mat=self._V.dot(self._connector).dot(self._U)
        if reset_unitaries==True:
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
        return mat

    def __toMPS__(self,connect=None):
        mps=[]
        if (self._position>0)&(self._position<self._N):
            for site in range(self._position):
                mps.append(toMPSmat(self._Q[site],self._R[site],self._dx[site]))
            
            mps[self._position-1]=np.transpose(np.tensordot(mps[self._position-1],self._mats[self._position],([1],[0])),(0,2,1))

            for site in range(self._position,self._N):
                mps.append(toMPSmat(self._Q[site],self._R[site],self._dx[site]))

            mps[0]=np.tensordot(self._U,mps[0],([1],[0]))
            mps[self._position-1]=np.transpose(np.tensordot(mps[self._position-1],self._V,([1],[0])),(0,2,1))
            self._V=np.eye(self._D)
            self._U=np.eye(self._D)


        if self._position==0:
            for site in range(self._N):
                mps.append(toMPSmat(self._Q[site],self._R[site],self._dx[site]))
            mps[0]=np.tensordot(self._U.dot(self._mats[0]),mps[0],([1],[0]))
            mps[self._position-1]=np.transpose(np.tensordot(mps[self._position-1],self._V,([1],[0])),(0,2,1))
            self._V=np.eye(self._D)
            self._U=np.eye(self._D)
        if self._position==self._N:
            for site in range(self._N):
                mps.append(toMPSmat(self._Q[site],self._R[site],self._dx[site]))

            mps[0]=np.tensordot(self._U,mps[0],([1],[0]))
            mps[self._position-1]=np.transpose(np.tensordot(mps[self._position-1],self._mats[-1].dot(self._V),([1],[0])),(0,2,1))
            self._V=np.eye(self._D)
            self._U=np.eye(self._D)
        if connect=='left':
            mps[0]=np.tensordot(self._connector,mps[0],([1],[0]))

        if connect=='right':
            mps[self._position-1]=np.transpose(np.tensordot(mps[self._position-1],self._connector,([1],[0])),(0,2,1))

        return mps

    def __topureMPS__(self,reset_unitaries=True):
        assert((self._position==0)|(self._position==self._N))
        mps=[]
        for site in range(self._N):
            mps.append(np.copy(self.__tensor__(site,contract_unitaries=True,reset_unitaries=reset_unitaries)))
        return mps


    def __data__(self,ind1,ind2):
        data=np.zeros((2,self._N),dtype=self._dtype)
        #for si in range(self._N):
        for si in range(self._position):
            data[0,si]=self._Q[si][ind1,ind2]
            data[1,si]=self._R[si][ind1,ind2]

        for si in range(self._position,self._N):
            #data[0,si]=np.conj(self._Q[si][ind2,ind1])
            #data[1,si]=np.conj(self._R[si][ind2,ind1])
            data[0,si]=self._Q[si][ind1,ind2]
            data[1,si]=self._R[si][ind1,ind2]

        return data

    def __mats__(self,ind1,ind2):
        data=np.zeros((self._N+1),dtype=self._dtype)
        for si in range(self._N+1):
            data[si]=self._mats[si][ind1,ind2]
        return data


    def save(self,filename):
        np.save('Q_'+filename,self._Q)
        np.save('R_'+filename,self._R)
        np.save('mats_'+filename,self._mats)
        np.save('lams_'+filename,self._lams)
        np.save('invmats_'+filename,self._invmats)
        np.save('connector_'+filename,self._connector)
        np.save('U_'+filename,self._U)
        np.save('V_'+filename,self._V)
        np.save('xm_'+filename,self._xm)
        np.save('xb_'+filename,self._xb)
        np.save('dx_'+filename,self._dx)

        params=[]
        params.append(self._obc)
        params.append(self._L)
        params.append(self._N)
        params.append(self._eps)
        params.append(self._dtype)
        params.append(self._D)
        params.append(self._rcond)
        params.append(self._position)
        np.save('params_'+filename,params)

    def load(self,filename):
        self._Q=np.load('Q_'+filename)
        self._R=np.load('R_'+filename)
        self._mats=np.load('mats_'+filename)
        self._invmats=np.load('invmats_'+filename)
        self._lams=np.load('lams_'+filename)
        self._connector=np.load('connector_'+filename)
        self._U=np.load('U_'+filename)
        self._V=np.load('V_'+filename)
        params=np.load('params_'+filename)
        self._xm=np.load('xm_'+filename)
        self._xb=np.load('xb_'+filename)
        self._dx=np.load('dx_'+filename)


        self._obc=bool(params[0])
        self._L=float(params[1])
        self._N=int(params[2])
        self._eps=float(params[3])
        self._dtype=str(params[4])
        self._D=int(params[5])
        self._rcond=float(params[6])
        self._position=int(params[7])
        self._dims=np.ones(self._N)*self._D

    def __squeeze__(self,grid,thresh):
        for i0 in range(self._D):
            for i1 in range(self._D):
                if np.std(self.__data__(i0,i1)[0,grid])<thresh:
                    meanQ=np.mean(self.__data__(i0,i1)[0,grid])
                    for s in range(self._N):
                        self._Q[s][i0,i1]=meanQ

                if np.std(self.__data__(i0,i1)[1,grid])<thresh:
                    meanR=np.mean(self.__data__(i0,i1)[1,grid])
                    for s in range(self._N):
                        self._R[s][i0,i1]=meanR

        for site in range(self._N):
            self._Q[site][np.nonzero(np.abs(self._Q[site])<thresh)]=0.0
            self._R[site][np.nonzero(np.abs(self._R[site])<thresh)]=0.0

    def __polyfit__(self,grid,deg):
        self._polyparams_Q=[]
        self._polyparams_R=[]
        R=np.zeros((self._D,self._D,self._N),dtype=self._dtype)
        Q=np.zeros((self._D,self._D,self._N),dtype=self._dtype)

        for n1 in range(self._D):
            params_Q=[]
            params_R=[]
            for n2 in range(self._D):
                params_Q.append(np.polyfit(self._xm[grid],self.__data__(n1,n2)[0,grid],deg))
                params_R.append(np.polyfit(self._xm[grid],self.__data__(n1,n2)[1,grid],deg))
                R[n1,n2,:]=np.polyval(params_R[n2],self._xm)
                Q[n1,n2,:]=np.polyval(params_Q[n2],self._xm)
            self._polyparams_Q.append(params_Q)
            self._polyparams_R.append(params_R)
        for site in range(self._N):
            self._Q[site]=np.copy(Q[:,:,site])
            self._R[site]=np.copy(R[:,:,site])
        

    #interpolate the cmps in place
    #self,xbdense,order=5,tol=1E-10,ncv=100,regauge=True
    def __interpolate__(self,N,order):
        bpoints=order
        #N=len(xdense)-1
        xbdense=np.linspace(0,self._xb[-1]-self._xb[0],N+1)
        dxdense=np.zeros(N)
        xmdense=np.zeros(N)
        for n in range(N):
            dxdense[n]=xbdense[n+1]-xbdense[n]
            xmdense[n]=(xbdense[n+1]+xbdense[n])/2.0

        #xbdense=np.linspace(0,cmps._L,Ndense+1)
        self.__position__(cmps._N)
        #cmps.__regauge__()
        R=[]
        Q=[]
        x=self._xm-self._xb[0]
        xint=np.append(np.append(x[self._N-bpoints:]-self._L,x),x[0:bpoints]+self._L)
        for si in range(self._N-bpoints,self._N):
            Q.append(self._Q[si])
            R.append(self._R[si])
        
        for si in range(self._N):
            Q.append(self._Q[si])
            R.append(self._R[si])
        
        
        for si in range(bpoints):
            Q.append(self._Q[si])
            R.append(self._R[si])
        
        #n2=Ndense/1100 #n2 can be any integer fraction of NUCdense
        n2=N
        n1=n2*self._N/N #=self._N
        
        Nfine=2*n2*(self._N+(2*bpoints-1))
        inds=range((2*bpoints-1)*n2+n1,Nfine-(2*(bpoints-1)*n2+n1),2*n1)
        #the very fine grid
        xmdenseint=np.linspace(xint[0],xint[-1],Nfine+1)
        
        #before interpolation, normalize matrices to the new dx
        Rd,Qd=cmpsinterp(Q,R,xint,xmdenseint[inds],k=order)

        mat=np.copy(self._mats[self._position])
        self._Q=[]
        self._R=[]
        self._mats=[]

        for n in range(N):
            self._Q.append(np.copy(Qd[n]))
            self._R.append(np.copy(Rd[n]))
            self._mats.append(None)
        self._mats.append(None)

        if self._position==self._N:
            self._position=N
        self._N=N            
        self._mats[self._position]=mat
        self._dx=np.copy(dxdense)
        self._xm=np.copy(xmdense)
        self._xb=np.copy(xbdense)


    def __grid_interpolate__(self,grid,order,tol=1E-10,ncv=100):
        self.__regauge__(True)
        bpoints=order
        Ndense=self._N
        cmpsd=CMPS('homogeneous','blabla',self._D,self._L,self._xb,self._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
        R=[]
        Q=[]
        x=self._xm-self._xb[0]
        xint=np.append(np.append(x[grid[len(grid)-bpoints:]]-self._L,x[grid]),x[grid[0:bpoints]]+self._L)
        for s in range(len(grid)-bpoints,len(grid)):
            site=grid[s]
            Q.append(self._Q[site])
            R.append(self._R[site])
        for site in grid:
            Q.append(self._Q[site])
            R.append(self._R[site])
        for s in range(bpoints):
            site=grid[s]
            Q.append(self._Q[site])
            R.append(self._R[site])

        n1=self._N
        xdense=np.append(np.append(x[grid[len(grid)-bpoints:]]-self._L,x),x[grid[0:bpoints]]+self._L)
        Rd,Qd=cmpsinterp(Q,R,xint,xdense,k=order)
            
        #i0=int(np.floor(np.random.rand(1)*self._D)) 
        #i1=int(np.floor(np.random.rand(1)*self._D))
        #for i0 in range(self._D):
        #    for i1 in range(self._D):
        #        d=np.zeros(len(Rd))
        #        for n in range(len(d)):
        #            d[n]=Rd[n][i0,i1]
        #
        #
        #        plt.figure(11)
        #        plt.plot(xdense,d,self._xm-self._xb[0],self.__data__(i0,i1)[1,:])
        #        raw_input()                

        for n in range(Ndense):
            self._Q[n]=np.copy(Qd[n+bpoints])
            self._R[n]=np.copy(Rd[n+bpoints])

        self.__regauge__(True)


    #returns a new cmps on the grid xdense
    #assumes that the cmps is regauged to be periodic
    
    def __interpolate_cmps__(self,xbdense,order=5,tol=1E-10,ncv=100,regauge=True):

        bpoints=order
        Ndense=len(xbdense)-1
        L=xbdense[-1]-xbdense[0]
        cmpsd=CMPS('homogeneous','blabla',self._D,L,xbdense,self._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
        self.__position__(self._N)
        self.__regauge__(regauge,tol=tol,ncv=ncv)

        R=[]
        Q=[]
        x=self._xm-self._xb[0]
        xint=np.append(np.append(x[self._N-bpoints:]-self._L,x),x[0:bpoints]+self._L)
        for si in range(self._N-bpoints,self._N):
            Q.append(self._Q[si])
            R.append(self._R[si])

        for si in range(self._N):
            Q.append(self._Q[si])
            R.append(self._R[si])


        for si in range(bpoints):
            Q.append(self._Q[si])
            R.append(self._R[si])


        #n2=Ndense/1100 #n2 can be any integer fraction of NUCdense
        #n2=Ndense
        #n1=n2*self._N/Ndense #=cmps._N
        #
        #Nfine=2*n2*(self._N+(2*bpoints-1))
        #inds=range((2*bpoints-1)*n2+n1,Nfine-(2*(bpoints-1)*n2+n1),2*n1)
        #xmdenseint=np.linspace(xint[0],xint[-1],Nfine+1)
        #before interpolation, normalize matrices to the new dx
        #plt.plot(xmdenseint[inds],np.zeros(xmdenseint[inds].shape),'o',xint,np.zeros(xint.shape),'d',xbdense,np.zeros(xbdense.shape),'x')
        #plt.show()
        #print xbdense
        #raw_input()
        #Rd,Qd=cmpsinterp(Q,R,xint,xmdenseint[inds],k=order)
        Rd,Qd=cmpsinterp(Q,R,xint,cmpsd._xm,k=order)
        for n in range(Ndense):
            cmpsd._Q[n]=np.copy(Qd[n])
            cmpsd._R[n]=np.copy(Rd[n])
        cmpsd._position=cmpsd._N

        cmpsd._connector=np.eye(self._D)
        cmpsd._U=np.copy(self._U)
        cmpsd._V=np.copy(self._V)
        cmpsd._mats[-1]=np.eye(self._D)

        
        cmpsd.__regauge__(regauge,tol=tol,ncv=ncv)
        return cmpsd






    #does an svd of self._mats[self._position]=U*D*V, and distributes the unitary U to the left, and V to the right matrices
    #if self._mats[self._position] is diagonal (with a possible phase) it takes its phase and distributes it equally onto all 
    #matrices
    #if either U or V is a diagonal unitary, its phase is fixed to 1
    #if neither U nor V are diagonal, then "which" determines which of the two gets its phase fixed before they are distributed.
    #if at the left boundary, it pushes U into self._U; if at the right boundary, it pushes V into self._V
    def __distribute__(self):
        if self._position==self._N:
            
            #first make sure that the self._mats[self._N] is diagonal:
            diag=np.diag(self._mats[self._N])
            mat=self._mats[self._N]-np.diag(diag)
            if np.linalg.norm(mat)>1E-10:
                sys.exit('cmpsdisclib.py: CMPS.__distribute__(): cmps._mats[cmps._N] is not diagonal!')

            angle=np.angle(diag)
            self._mats[self._N]=np.diag(diag*np.exp(-1j*angle))
            phi=angle/self._N
            for n in range(self._N-1,-1,-1):
                A=toMPSmat(self._Q[n],self._R[n],self._dx[n])
                Ur=np.diag(np.exp(1j*(n+1)*phi))
                Ul_dag=np.diag(np.exp(-1j*n*phi))
                A[:,:,0]=Ul_dag.dot(A[:,:,0]).dot(Ur)
                A[:,:,1]=Ul_dag.dot(A[:,:,1]).dot(Ur)
                self._Q[n],self._R[n]=fromMPSmat(A,self._dx[n])


        if self._position==0:
            
            #first make sure that the self._mats[self._N] is diagonal:
            diag=np.diag(self._mats[0])
            mat=self._mats[0]-np.diag(diag)
            if np.linalg.norm(mat)>1E-10:
                sys.exit('cmpsdisclib.py: CMPS.__distribute__(): cmps._mats[cmps._N] is not diagonal!')

            angle=np.angle(diag)
            self._mats[0]=np.diag(diag*np.exp(-1j*angle))
            phi=angle/self._N
            for n in range(self._N):
                A=toMPSmat(self._Q[n],self._R[n],self._dx[n])
                Ul=np.diag(np.exp(1j*(self._N-n)*phi))
                Ur_dag=np.diag(np.exp(-1j*(self._N-1-n)*phi))
                A[:,:,0]=Ul.dot(A[:,:,0]).dot(Ur_dag)
                A[:,:,1]=Ul.dot(A[:,:,1]).dot(Ur_dag)
                self._Q[n],self._R[n]=fromMPSmat(A,self._dx[n])



    #does an svd of self._mats[self._position]=U*D*V, and distributes the unitary U to the left, and V to the right matrices
    #if self._mats[self._position] is diagonal (with a possible phase) it takes its phase and distributes it equally onto all 
    #matrices
    #if either U or V is a diagonal unitary, its phase is fixed to 1
    #if neither U nor V are diagonal, then "which" determines which of the two gets its phase fixed before they are distributed.
    #if at the left boundary, it pushes U into self._U; if at the right boundary, it pushes V into self._V
    def __distributeold__(self,which):
        U,l,V=np.linalg.svd(self._mats[self._position])
        #raw_input('in distribute')
        #check if U and V are both diagonal, in which case l was diagonal before hand.
        #in that case, don't do anything
        if (isDiagUnitary(U)[0]==True)&(isDiagUnitary(V)[0]==True):
            diff=np.linalg.norm(U.dot(V)-np.eye(self._D))
            if diff<1E-10:
                print 'CMPS.__distribute__(): self._mats[{0}] is diagonal'.format(self._position)
                return 
            if diff>=1E-10:
                Nl=self._position
                Nr=self._N-self._position
                unit=U.dot(V)
                phase=np.angle(np.diag(unit))
                U=np.diag(np.exp(1j*phase/self._N*Nl))
                V=np.diag(np.exp(1j*phase/self._N*Nr))


        #if one of the two unitaries is diagonal, then fix its phase to be 1
        if (isDiagUnitary(U)[0]==True)&(isDiagUnitary(V)[0]==False):
            phase=np.angle(np.diag(U))
            unit=np.diag(np.exp(-1j*phase))
            U=U.dot(unit)
            V=herm(unit).dot(V)

        if (isDiagUnitary(U)[0]==False)&(isDiagUnitary(V)[0]==True):
            phase=np.angle(np.diag(V))
            unit=np.diag(np.exp(-1j*phase))
            U=U.dot(herm(unit))
            V=unit.dot(V)

        #if both U and V are not diagonal, then the ask the user which should be phase fixed
        elif (isDiagUnitary(U)[0]==False)&(isDiagUnitary(V)[0]==False):
            if which=='v':
                phase=np.angle(np.diag(V))
                unit=np.diag(np.exp(-1j*phase))
                U.dot(herm(unit))
                V=unit.dot(V)
            if which=='u':
                phase=np.angle(np.diag(U))
                unit=np.diag(np.exp(-1j*phase))
                U=U.dot(unit)
                V=herm(unit).dot(V)


        l=l/np.linalg.norm(l)
        if self._position==(self._N):
            print ('CMPS__distribute__(self): pushing v to self._V')
            self._V=V.dot(self._V)

        if self._position==0:
            print ('CMPS__distribute__(self): pushing u to self._U')
            self._U=self._U.dot(U)
        
        dx=self._L/self._N

        if self._position>0:
            etaU,matU=np.linalg.eig(U)
            invmatU=np.linalg.inv(matU)
            gaugematU=matU.dot(np.diag(etaU**(dx))).dot(invmatU)
            HU=(gaugematU-np.eye(self._D))/dx
            
            
            #now redistribute U to theleft matrices
            for n in range(self._position-1,0,-1):
                x=n*dx
                Un=matU.dot(np.diag(etaU**x)).dot(invmatU)
                invUn=matU.dot(np.diag(1./(etaU**x))).dot(invmatU)
                self._Q[n]=invUn.dot(self._Q[n]).dot(Un)
                self._R[n]=invUn.dot(self._R[n]).dot(Un)
            for n in range(self._position-1,-1,-1):
                self._Q[n]=self._Q[n]+HU+dx*self._Q[n].dot(HU)
                self._R[n]=self._R[n]+dx*self._R[n].dot(HU)


        if self._position<self._N:
            etaV,matV=np.linalg.eig(V)
            invmatV=np.linalg.inv(matV)
            gaugematV=matV.dot(np.diag(etaV**(dx))).dot(np.linalg.inv(matV))
            HV=(gaugematV-np.eye(self._D))/dx
            #redistribute V to the right matrices
            for n in range(self._position,self._N-1):
                x=(self._N-1-n)*dx
                Vn=matV.dot(np.diag(etaV**x)).dot(invmatV)
                invVn=matV.dot(np.diag(1./(etaV**x))).dot(invmatV)
                #print n,np.linalg.norm(Vn-np.eye(self._D))
                #Qold=np.copy(self._Q[n])
                #Rold=np.copy(self._R[n])
                self._Q[n]=Vn.dot(self._Q[n]).dot(invVn)
                self._R[n]=Vn.dot(self._R[n]).dot(invVn)
                #print np.linalg.norm(Qold-self._Q[n])
                #print np.linalg.norm(Rold-self._R[n])


                
            for n in range(self._position,self._N):
                #Qold=np.copy(self._Q[n])
                #Rold=np.copy(self._R[n])
                self._Q[n]=(self._Q[n]+HV+dx*HV.dot(self._Q[n]))
                self._R[n]=(self._R[n]+dx*HV.dot(self._R[n]))
                #print np.linalg.norm(Qold-self._Q[n])
                #print np.linalg.norm(Rold-self._R[n])
                #raw_input(n)            
        self._mats[self._position]=np.diag(l)


    #brings a cmps into canonized form by successive SVD; pushes non-diagonal boundary unitaries to self._connector
    def __canonize__(self):
        if self._position<(self._N-(self._N%2))/2:
            #self.__position__(0)
            self.__positionSVD__(0,'v')
            self.__positionSVD__(self._N,'u')
            #self.__distribute__('v')
            #self.__positionSVD__(self._0,'accumulate',pushunitaries=True)

        if self._position>=(self._N-(self._N%2))/2:
            #self.__position__(self._N)
            
            self.__positionSVD__(self._N,'u')
            self.__positionSVD__(0,'v')
            #self.__distribute__('u')
                                 

    #boundarymat has to be on the left end!
    #Q and R have to be left orthonormal!
    def __regauge__(self,canonize=False,initial=None,nmaxit=100000,tol=1E-10,ncv=100):
        #shift position to N, state doesn't connect back now
        #self.__position__(self._N)
        if self._position<(self._N-(self._N%2))/2:
            self.__position__(0)

        if self._position>=(self._N-(self._N%2))/2:
            self.__position__(self._N)

        if self._position==self._N:
            boundarymatrix=self._mats[self._N].dot(self.__connection__(reset_unitaries=True))
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the right boundarymatrix into mps2
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            #find the right eigenvector of the transfer operator
            initial=herm(self._mats[self._N]).dot(self._mats[self._N])
            [eta,vr,numeig]=cmf.UnitcellTMeigs(mps2,-1,1,initial,datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            mps2=self.__topureMPS__()
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            r=np.reshape(vr,(self._D,self._D))
            #fix phase of l and restore the proper normalization of l
            r=r/np.trace(r)
            if self._dtype==float:
                r=np.real(r+herm(r))/2.0
            if self._dtype==complex:
                r=(r+herm(r))/2.0
            eigvals,u=np.linalg.eigh(r)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            x=u.dot(np.diag(np.sqrt(eigvals)))
            invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
            
            [eta,vl,numeig]=cmf.UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print 'found large abs(eta)={0'.format(np.abs(eta))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            l=np.reshape(vl,(self._D,self._D))
            l=l/np.trace(l)
            
            if self._dtype==float:
                l=np.real(l+herm(l))/2.0
            if self._dtype==complex:
                l=(l+herm(l))/2.0
            
            eigvals,u=np.linalg.eigh(l)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))))
            invy=np.transpose(np.diag(np.sqrt(inveigvals)).dot(herm(u)))
            
            [U,lam,Vdag]=np.linalg.svd(y.dot(x))        
            left=np.tensordot(Vdag,invx,([1],[0]))
            leftlam=np.diag(lam).dot(left)
            
            right=np.tensordot(invy,U,([1],[0]))
            rightlam=right.dot(np.diag(lam))
            
            temp=boundarymatrix.dot(rightlam)
            Zr=np.trace(temp.dot(herm(temp)))

            self._mats[-1]=np.copy(temp/np.sqrt(Zr))
            #self.__distribute__('v')
            self.__positionSVD__(0,'v') 
            #use SVD to shift position; there is still non-trivial matrix at the unitcell boundary
            
            temp=leftlam.dot(self._mats[0])
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[0]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)

            #this removes a possible diagonal phase from u for tempmat=u l v
            #self.__distribute__('u')
            if canonize==True:

                self.__positionSVD__(self._N,'u')
                #remove a possible phase from the left most bond matrix self._mats[-1]=u l v
                self.__distribute__()
                #self.__distributeold__('v')
                #self._mats[self._N]=np.copy(np.diag(lam))

                #unitary=self._mats[-1].dot(self._connector)
                ##to get the unitaries that regauge the cmps, i assume that the individual 
                ##unitaries that generate the gauge change over a length dx[n] at site n are
                ##the same for every site:
                #eta,mat=np.linalg.eig(unitary)
                #dx=self._L/self._N
                #gaugemat=mat.dot(np.diag(eta**(dx))).dot(np.linalg.inv(mat))
                #H=(gaugemat-np.eye(self._D))/dx
                ##now redistribute the gaugechange to all matrices
                #for n in range(self._N-1,0,-1):
                #    x=n*dx
                #    U=mat.dot(np.diag(eta**x)).dot(np.linalg.inv(mat))
                #    self._Q[n]=herm(U).dot(self._Q[n]).dot(U)
                #    self._R[n]=herm(U).dot(self._R[n]).dot(U)
                #    
                #for n in range(self._N-1,-1,-1):
                #    self._Q[n]=self._Q[n]+H+dx*self._Q[n].dot(H)
                #    self._R[n]=self._R[n]+dx*self._R[n].dot(H)


        if self._position==0:
            boundarymatrix=self.__connection__(reset_unitaries=True).dot(self._mats[0])
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the left boundarymatrix into mps2
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            #find the right eigenvector of the transfer operator
            initial=self._mats[0].dot(self._mats[0])
            [eta,vr,numeig]=cmf.UnitcellTMeigs(mps2,-1,1,initial,datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            mps2=self.__topureMPS__()
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            r=np.reshape(vr,(self._D,self._D))
            #fix phase of l and restore the proper normalization of l
            r=r/np.trace(r)
            if self._dtype==float:
                r=np.real(r+herm(r))/2.0
            if self._dtype==complex:
                r=(r+herm(r))/2.0
            eigvals,u=np.linalg.eigh(r)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            x=u.dot(np.diag(np.sqrt(eigvals)))
            invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
            
            [eta,vl,numeig]=cmf.UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print 'found large abs(eta)={0'.format(np.abs(eta))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            l=np.reshape(vl,(self._D,self._D))
            l=l/np.trace(l)
            
            if self._dtype==float:
                l=np.real(l+herm(l))/2.0
            if self._dtype==complex:
                l=(l+herm(l))/2.0
            
            eigvals,u=np.linalg.eigh(l)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))))
            invy=np.transpose(np.diag(np.sqrt(inveigvals)).dot(herm(u)))
            
            [U,lam,Vdag]=np.linalg.svd(y.dot(x))        
            left=np.tensordot(Vdag,invx,([1],[0]))
            leftlam=np.diag(lam).dot(left)
            
            right=np.tensordot(invy,U,([1],[0]))
            rightlam=right.dot(np.diag(lam))
            
            
            temp=leftlam.dot(boundarymatrix)
            Zr=np.trace(temp.dot(herm(temp)))

            self._mats[0]=np.copy(temp/np.sqrt(Zr))
            #use SVD to shift position; there is still non-trivial matrix at the unitcell boundary
            self.__positionSVD__(self._N,'u')
            temp=self._mats[-1].dot(rightlam)
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[-1]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            if canonize==True:
                self.__positionSVD__(0,'v')
                self.__distribute__()
                #self.__distributeold__('u')
                


    #boundarymat has to be on the left end!
    #Q and R have to be left orthonormal!
    def __gridregauge__(self,grid,canonize=False,initial=None,nmaxit=100000,tol=1E-10,ncv=100):
        if self._position<(self._N-(self._N%2))/2:
            self.__position__(0)

        if self._position>=(self._N-(self._N%2))/2:
            self.__position__(self._N)

        if self._position==self._N:
            boundarymatrix=self._mats[self._N].dot(self.__connection__(reset_unitaries=True))
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the right boundarymatrix into mps2
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            #find the right eigenvector of the transfer operator
            initial=herm(self._mats[self._N]).dot(self._mats[self._N])
            [eta,vr,numeig]=cmf.UnitcellTMeigs(mps2,-1,1,initial,datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            mps2=self.__topureMPS__()
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            r=np.reshape(vr,(self._D,self._D))
            #fix phase of l and restore the proper normalization of l
            r=r/np.trace(r)
            if self._dtype==float:
                r=np.real(r+herm(r))/2.0
            if self._dtype==complex:
                r=(r+herm(r))/2.0
            eigvals,u=np.linalg.eigh(r)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            x=u.dot(np.diag(np.sqrt(eigvals)))
            invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
            
            [eta,vl,numeig]=cmf.UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print 'found large abs(eta)={0'.format(np.abs(eta))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            l=np.reshape(vl,(self._D,self._D))
            l=l/np.trace(l)
            
            if self._dtype==float:
                l=np.real(l+herm(l))/2.0
            if self._dtype==complex:
                l=(l+herm(l))/2.0
            
            eigvals,u=np.linalg.eigh(l)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))))
            invy=np.transpose(np.diag(np.sqrt(inveigvals)).dot(herm(u)))
            
            [U,lam,Vdag]=np.linalg.svd(y.dot(x))        
            left=np.tensordot(Vdag,invx,([1],[0]))
            leftlam=np.diag(lam).dot(left)
            
            right=np.tensordot(invy,U,([1],[0]))
            rightlam=right.dot(np.diag(lam))
            
            temp=boundarymatrix.dot(rightlam)
            Zr=np.trace(temp.dot(herm(temp)))

            #u,l,v=np.linalg.svd(temp)
            self._mats[-1]=np.copy(temp/np.sqrt(Zr))
            #self.__distribute__('v')
            self.__positionSVD__(0,'v') 
            #use SVD to shift position; there is still non-trivial matrix at the unitcell boundary
            
            temp=leftlam.dot(self._mats[0])
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[0]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)

            #this removes a possible diagonal phase from u for tempmat=u l v
            #self.__distribute__('u')
            if canonize==True:

                self.__positionSVD__(self._N,'u')
                #remove a possible phase from the left most bond matrix self._mats[-1]=u l v
                self.__distribute__('v')
                #self._mats[self._N]=np.copy(np.diag(lam))

                #unitary=self._mats[-1].dot(self._connector)
                ##to get the unitaries that regauge the cmps, i assume that the individual 
                ##unitaries that generate the gauge change over a length dx[n] at site n are
                ##the same for every site:
                #eta,mat=np.linalg.eig(unitary)
                #dx=self._L/self._N
                #gaugemat=mat.dot(np.diag(eta**(dx))).dot(np.linalg.inv(mat))
                #H=(gaugemat-np.eye(self._D))/dx
                ##now redistribute the gaugechange to all matrices
                #for n in range(self._N-1,0,-1):
                #    x=n*dx
                #    U=mat.dot(np.diag(eta**x)).dot(np.linalg.inv(mat))
                #    self._Q[n]=herm(U).dot(self._Q[n]).dot(U)
                #    self._R[n]=herm(U).dot(self._R[n]).dot(U)
                #    
                #for n in range(self._N-1,-1,-1):
                #    self._Q[n]=self._Q[n]+H+dx*self._Q[n].dot(H)
                #    self._R[n]=self._R[n]+dx*self._R[n].dot(H)


        if self._position==0:
            boundarymatrix=self.__connection__(reset_unitaries=True).dot(self._mats[0])
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the left boundarymatrix into mps2
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            #find the right eigenvector of the transfer operator
            initial=self._mats[0].dot(self._mats[0])
            [eta,vr,numeig]=cmf.UnitcellTMeigs(mps2,-1,1,initial,datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            mps2=self.__topureMPS__()
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            r=np.reshape(vr,(self._D,self._D))
            #fix phase of l and restore the proper normalization of l
            r=r/np.trace(r)
            if self._dtype==float:
                r=np.real(r+herm(r))/2.0
            if self._dtype==complex:
                r=(r+herm(r))/2.0
            eigvals,u=np.linalg.eigh(r)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            x=u.dot(np.diag(np.sqrt(eigvals)))
            invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
            
            [eta,vl,numeig]=cmf.UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),datatype=self._dtype,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print 'found large abs(eta)={0'.format(np.abs(eta))
            if np.abs(np.imag(eta))>1E-10:
                print 'in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta
            l=np.reshape(vl,(self._D,self._D))
            l=l/np.trace(l)
            
            if self._dtype==float:
                l=np.real(l+herm(l))/2.0
            if self._dtype==complex:
                l=(l+herm(l))/2.0
            
            eigvals,u=np.linalg.eigh(l)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            inveigvals=1.0/eigvals
            y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))))
            invy=np.transpose(np.diag(np.sqrt(inveigvals)).dot(herm(u)))
            
            [U,lam,Vdag]=np.linalg.svd(y.dot(x))        
            left=np.tensordot(Vdag,invx,([1],[0]))
            leftlam=np.diag(lam).dot(left)
            
            right=np.tensordot(invy,U,([1],[0]))
            rightlam=right.dot(np.diag(lam))
            
            
            temp=leftlam.dot(boundarymatrix)
            Zr=np.trace(temp.dot(herm(temp)))

            self._mats[0]=np.copy(temp/np.sqrt(Zr))
            #use SVD to shift position; there is still non-trivial matrix at the unitcell boundary
            self.__positionSVD__(self._N,'u')
            temp=self._mats[-1].dot(rightlam)
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[-1]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            if canonize==True:
                self.__positionSVD__(0,'v')
                self.__distribute__('u')









                
    #calculates the derivative of the bond matrix "mat", at bond "bond".
    #for direction>0: __dC_dx SV decomposes mat,  and pushes U onto the left tensor (returned in Q,R)
    #                 dQ is the difference between Q[bond] (obtained after orthonormalization) and Q[bond-1]=Q (similar for dR)
    #                 dV (dU) is the difference between U1 l1 V1=mat[bond] and U2 l2 V2 = mat[bond+1] (and similar for dl)
    #for direction<0: __dC_dx SV decomposes mat,  and pushes V onto the right tensor (returned in Q,R)
    #                 dQ is the difference between Q[bond]=Q and Q[bond-1] (obtained after orthonormalization) (similar for dR)
    #                 dV (dU) is the difference between U1 l1 V1=mat[bond] and U2 l2 V2 = mat[bond-1] (and similar for dl)
    def __dC_dx__(self,mat,bond,direction):
        if direction>0:
            U,l,V=np.linalg.svd(mat)
            phase=np.angle(np.diag(U))
            unitary=np.diag(np.exp(-1.0j*phase))
            U=U.dot(unitary)
            V=herm(unitary).dot(V)

            tensor=np.transpose(np.tensordot(self.__tensor__(bond-1,False,False),U,([1],[0])),(0,2,1))

            Ql,Rl=fromMPSmat(tensor,self._dx[bond-1])
            Q,R,matout=prepareCMPSTensorSVD(self._Q[bond],self._R[bond],np.diag(l).dot(V),fix='u',dx=self._dx[bond],direction=1)
            
            dQ=(Q-Ql)/(self._xm[bond]-self._xm[bond-1])
            dR=(R-Rl)/(self._xm[bond]-self._xm[bond-1])
            Uout,lout,Vout=np.linalg.svd(matout)
            phase=np.angle(np.diag(Uout))
            unitary=np.diag(np.exp(-1.0j*phase))
            Uout=Uout.dot(unitary)
            Vout=herm(unitary).dot(Vout)

            dV=(Vout-V)/(self._xm[bond]-self._xm[bond-1])
            dl=(lout-l)/(self._xm[bond]-self._xm[bond-1])
            return Ql,dQ,Rl,dR,V,dV,l,dl

        if direction<0:
            U,l,V=np.linalg.svd(mat)
            phase=np.angle(np.diag(V))
            unitary=np.diag(np.exp(-1.0j*phase))
            U=U.dot(herm(unitary))
            V=unitary.dot(V)
            tensor=np.tensordot(V,self.__tensor__(bond,False,False),([1],[0]))
            Qr,Rr=fromMPSmat(tensor,self._dx[bond])
            Q,R,matout=prepareCMPSTensorSVD(self._Q[bond-1],self._R[bond-1],U.dot(np.diag(l)),fix='v',dx=self._dx[bond-1],direction=-1)
            
            dQ=(Qr-Q)/(self._xm[bond]-self._xm[bond-1])
            dR=(Rr-R)/(self._xm[bond]-self._xm[bond-1])
            Uout,lout,Vout=np.linalg.svd(matout)
            phase=np.angle(np.diag(Vout))
            unitary=np.diag(np.exp(-1.0j*phase))
            Uout=Uout.dot(herm(unitary))
            Vout=unitary.dot(Vout)

            dU=(U-Uout)/(self._xm[bond]-self._xm[bond-1])
            dl=(l-lout)/(self._xm[bond]-self._xm[bond-1])
            return Qr,dQ,Rr,dR,U,dU,l,dl


    def __d_dx__(self,bond,direction):
        if direction>0:
            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((self._D,self._D),dtype=self._dtype)
            A0=np.eye(self._D)+self._dx[site]*mat.dot(self._Q[site]).dot(invmat)
            Rout=mat.dot(self._R[site]).dot(invmat).dot(np.linalg.pinv(A0))
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),1)
            if fixphase==False:
                r2=r.dot(A0)
                Qout,Rout=fromMPSmat(tensor,self._dx[site])
                return (r2-np.eye(self._D))/self._dx[site],Qout,Rout
            if fixphase==True:
                tempmat=r.dot(A0)
                Qt,Rt=fromMPSmat(tensor,self._dx[site])
                q,r2=cmf.qr(tempmat,'q')
                H=(q-np.eye(self._D))/self._dx[site]
                
                Qout=Qt+H+self._dx[site]*Qt.dot(H)
                Rout=Rt+self._dx[site]*Rt.dot(H)

                u_,l,v=np.linalg.svd(tempmat)
                l=l/np.linalg.norm(l)
                phase=np.angle(np.diag(u_))
                unit=np.diag(np.exp(-1j*phase))
                u=u_.dot(unit)
                
                H=(u-np.eye(D))/dx
                Qout=Qt+H+dx*Qt.dot(H)
                Rout=Rt+dx*Rt.dot(H)
                matout=np.diag(l).dot(herm(unit)).dot(v)

                return (r2-np.eye(self._D))/self._dx[site],Qout,Rout

        if direction<0:
            assert(self._position>site)
            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((self._D,self._D),dtype=self._dtype)
            A0=np.eye(self._D)+self._dx[site]*invmat.dot(self._Q[site]).dot(mat)
            Rout=np.linalg.pinv(A0).dot(invmat).dot(self._R[site]).dot(mat)
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),-1)
            if fixphase==False:
                r2=A0.dot(r)
                Qout,Rout=fromMPSmat(tensor,self._dx[site])
                return (r2-np.eye(self._D))/self._dx[site],Qout,Rout
            if fixphase==True:
                tempmat=A0.dot(r)
                Qt,Rt=fromMPSmat(tensor,self._dx[site])
                q,r2=cmf.qr(herm(tempmat),'q')
                H=(herm(q)-np.eye(self._D))/self._dx[site]
                Qout=Qt+H+self._dx[site]*H.dot(Qt)
                Rout=Rt+self._dx[site]*H.dot(Rt)
                return (herm(r2)-np.eye(self._D))/self._dx[site],Qout,Rout



    def __calculateC__(self,sites,direction):
        if direction>0:
            self.__positionSVD__(0,'accumulate')
            mats=np.zeros((self._D,self._D,len(sites)),dtype=self._dtype)
            mats[:,:,0]=np.copy(self._mats[0])
            for n in range(len(sites)-1):
                dx=np.sum(self._dx[sites[n]:sites[n+1]])
                ddx,Q,R=self.__d_dx__(mats[:,:,n],sites[n],1,fixphase=True)
                mats[:,:,n+1]=mats[:,:,n]+dx*ddx.dot(mats[:,:,n])
            return mats

        if direction<0:
            self.__positionSVD__(self._N,'accumulate')
            mats=np.zeros((self._D,self._D,len(sites)),dtype=self._dtype)
            mats[:,:,-1]=np.copy(self._mats[sites[-1]+1])
            for n in range(len(sites)-1,0,-1):
                dx=np.sum(self._dx[sites[n-1]:sites[n]])
                ddx,Q,R=self.__d_dx__(mats[:,:,n],sites[n],-1,fixphase=True)
                mats[:,:,n-1]=mats[:,:,n]+dx*mats[:,:,n].dot(ddx)
            return mats


    #shifts the center site using numerical derivatives;
    def __jump__(self,position):

        if self._position==position:
            return
        if self._position<position:
            assert(self._position>0)
            if position==self._N:
                if self._position==(self._N-1):
                    print 'shifting from site {0} to site {1}'.format(self._position,self._N)
                    self.__positionSVD__(self._N,fix='u')
                if self._position<(self._N-1):
                    self.__jump__(self._N-1)
                    print 'shifting from site {0} to site {1}'.format(self._position,self._N)
                    self.__positionSVD__(self._N,fix='u')

            if position<self._N:
                print 'jumping from site {0} to site {1}'.format(self._position,position)
                mat=self._mats[self._position]
                dx=np.sum(self._dx[self._position:position])
                #__dC_dx SV decomposes mat,  and pushes U onto the left tensor (returned in Q,R)
                #dQ is the difference between Q[bond] and Q[bond-1] (similar for dR)
                Q,dQ,R,dR,V,dV,l,dl=self.__dC_dx__(mat,self._position,1)
                self._Q[self._position-1]=np.copy(Q)
                self._R[self._position-1]=np.copy(R)
                self._Q[self._position]=np.copy(Q+(self._xm[self._position]-self._xm[self._position-1])*dQ)
                self._R[self._position]=np.copy(R+(self._xm[self._position]-self._xm[self._position-1])*dR)

                print np.linalg.norm(np.eye(self._D)-self.__checkOrtho__(self._position-1,1)),np.linalg.norm(np.eye(self._D)-self.__checkOrtho__(self._position,1))
                self._mats[self._position]=np.diag(l).dot(V)

                #dmdx,Q,R=self.__dC_dx__(mat,self._position,1)
                #ddx,Q,R=self.__d_dx__(mat,self._position,1,fixphase)
                #self._mats[position]=mat+dx*ddx.dot(mat)
                Q2=Q+dx*dQ
                R2=R+dx*dR
                V2=V+dx*dV
                l2=l+dx*dl
                l2=l2/np.linalg.norm(l2)


                #for position=self._position+1, this should be the same as above
                self._mats[position]=np.diag(l2).dot(V2)
                self._Q[position-1]=np.copy(Q2)
                self._R[position-1]=np.copy(R2)
                self._position=position

        if self._position>position:
            assert(self._position<self._N)
            if position==0:
                if self._position==1:
                    print 'shifting from site {0} to site {1}'.format(self._position,0)
                    self.__positionSVD__(0,fix='v')
                if self._position>1:
                    self.__jump__(1)
                    print 'shifting from site {0} to site {1}'.format(self._position,0)
                    self.__positionSVD__(0,fix='v')

            if position>0:
                print 'jumping from site {0} to site {1}'.format(self._position,position)
                mat=self._mats[self._position]
                dx=np.sum(self._dx[position:self._position])
                Q,dQ,R,dR,U,dU,l,dl=self.__dC_dx__(mat,self._position,-1)
                self._Q[self._position]=np.copy(Q)
                self._R[self._position]=np.copy(R)
                self._Q[self._position-1]=np.copy(Q-(self._xm[self._position]-self._xm[self._position-1])*dQ)
                self._R[self._position-1]=np.copy(R-(self._xm[self._position]-self._xm[self._position-1])*dR)
                self._mats[self._position]=U.dot(np.diag(l))
                #ddx,Q,R=self.__d_dx__(mat,self._position-1,-1,fixphase)
                #dmdx,Q,R=self.__dC_dx__(mat,self._position-1,-1)
                #self._mats[position]=mat+dx*mat.dot(ddx)
                #mat2=mat-dx*dmdx
                Q2=Q-dx*dQ
                R2=R-dx*dR
                U2=U-dx*dU
                l2=l-dx*dl
                l2=l2/np.linalg.norm(l2)

                self._mats[position]=U2.dot(np.diag(l2))
                self._Q[self._position]=np.copy(Q2)
                self._R[self._position]=np.copy(R2)
                self._position=position


    #shifts center site moving points on "grid" from self._position to position; "grid" is a SITE grid, not a BOND grid;
    #if self._position or position are
    #not on "grid", it moves them to the point closest by. Only matrices on "grid" are modified by __shiftPosition__ (and 
    #those needed to shift self._position if it initially was not on "grid")
    #if the shift is going from left to right, then self._position stays to the left of the last site of "grid"
    #if the shift is going from right to left, then self._position stays to the right of the first site of "grid"
    def __shiftPosition__(self,position,grid,fixphase,method_simple):
        #now move the center site on "grid" to the new "position"
        if position==self._position:
            return 
        
        if position>self._position:

            #find the closest point on "grid" to self._position
            if self._position>=grid[0]:
                closestleft,closestright=bisection(grid,self._position)
         
                distl=self._position-grid[closestleft]
                distr=grid[closestright]-self._position

                oldposition=self._position
                if distl<distr:
                    self.__position__(grid[closestleft],fixphase,method_simple)
                    initial=closestleft
                if distl>=distr:
                    self.__position__(grid[closestright],fixphase,method_simple)
                    initial=closestright
                if (distl<>0)&(distr<>0):
                    warnings.warn('CMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))

            if self._position<grid[0]:
                warnings.warn('CMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(self._position,grid[0]))
                self.__position__(grid[0],fixphase,method_simple)
                initial=0

            #print 'self._position={0}'.format(self._position)
            if position<=grid[-1]:
                closestleft,closestright=bisection(grid,position)
                distl=position-grid[closestleft]
                distr=grid[closestright]-position
                if distl<distr:
                    newposition=grid[closestleft]
                    final=closestleft
                if distl>=distr:
                    newposition=grid[closestright]
                    final=closestright

                if (distl<>0)&(distr<>0):
                    warnings.warn('CMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                position=newposition
            if position>grid[-1]:
                warnings.warn('CMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,grid[-1]))
                position=grid[-1]
                final=len(grid)-1

            #now move from self._position to position using __jump__
            for index in range(initial+1,final+1):
                print 'at self._position={2}; jumping from {0} to {1}'.format(grid[index-1],grid[index],self._position)
                self.__jump__(grid[index],fixphase,method_simple)
        
        if position<self._position:
            if self._position<=grid[-1]:
                #find the closest point on "grid" to self._position
                
                closestleft,closestright=bisection(grid,self._position)
                #print 'self._position={2} between {0} and {1}'.format(grid[closestleft],grid[closestright],self._position)
                distl=self._position-grid[closestleft]
                distr=grid[closestright]-self._position
                
                oldposition=self._position
                if distl<distr:
                    self.__position__(grid[closestleft]+1,fixphase,method_simple)
                    initial=closestleft
                if distl>=distr:
                    self.__position__(grid[closestright]+1,fixphase,method_simple)
                    initial=closestright
                if (distl<>0)&(distr<>0):
                    warnings.warn('CMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))
                    print
            if self._position>grid[-1]:
                self.__position__(grid[-1]+1,fixphase,method_simple)
                initial=len(grid)-1

            if position>=grid[0]:
                closestleft,closestright=bisection(grid,position)
                distl=position-grid[closestleft]
                distr=grid[closestright]-position
                if distl<distr:
                    newposition=grid[closestleft]+1
                    final=closestleft
                if distl>=distr:
                    newposition=grid[closestright]+1
                    final=closestright
                
                if (distl<>0)&(distr<>0):
                    warnings.warn('CMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                    print
                position=newposition

            if position<grid[0]:
                position=grid[0]+1
                final=0
        
            #now move from self._position to position using __jump__
            for index in range(initial-1,final-1,-1):
                print 'at self._position={2}; jumping from {0} to {1}'.format(grid[index+1]+1,grid[index]+1,self._position)
                self.__jump__(grid[index]+1,fixphase,method_simple)

        
    def __diagonalize__(self,which):
        u,l,v=np.linalg.svd(self._mats[self._position])
        l=l/np.linalg.norm(l)
        if which=='v':
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)
        if which=='u':
            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
        
        if (self._position>0)&(self._position<self._N):
            tensorl=toMPSmat(self._Q[self._position-1],self._R[self._position-1],self._dx[self._position-1])
            tensorr=toMPSmat(self._Q[self._position],self._R[self._position],self._dx[self._position])
            tensorl=np.transpose(np.tensordot(tensorl,u,([1],[0])),(0,2,1))
            tensorr=np.tensordot(v,tensorr,([1],[0]))
            self._Q[self._position-1],self._R[self._position-1]=fromMPSmat(tensorl,self._dx[self._position-1])
            self._Q[self._position],self._R[self._position]=fromMPSmat(tensorr,self._dx[self._position])
            self._mats[self._position]=np.diag(l)
        
        if self._position==0:
            #diff=np.linalg.norm(np.abs(u)-np.eye(self._D))
            #if diff>1E-10:
            #print 'CMPS.__diagonalize__(): at position={0}: found non-diagonal u; pushing it to self._U'.format(self._position)
            print 'CMPS.__diagonalize__(): at position={0}: pushing u to self._U'.format(self._position)
            self._U=self._U.dot(u)
            #if diff<=1E-10:
            #    v=u.dot(v)

            tensorr=toMPSmat(self._Q[self._position],self._R[self._position],self._dx[self._position])
            tensorr=np.tensordot(v,tensorr,([1],[0]))

            self._Q[self._position],self._R[self._position]=fromMPSmat(tensorr,self._dx[self._position])
            self._mats[self._position]=np.diag(l)
        if self._position==self._N:
            #diff=np.linalg.norm(np.abs(v)-np.eye(self._D))
            #if diff>1E-10:
            #print 'CMPS.__diagonalize__(): at position={0}: found non-diagonal v; pushing it to self._V'.format(self._position)
            print 'CMPS.__diagonalize__(): at position={0}: pushing v to self._V'.format(self._position)
            self._V=v.dot(self._V)
            print v
            print self._V
            raw_input('in diagonalize')

            #if diff<=1E-10:
            #    u=u.dot(v)

            tensorl=toMPSmat(self._Q[self._position-1],self._R[self._position-1],self._dx[self._position-1])
            tensorl=np.transpose(np.tensordot(tensorl,u,([1],[0])),(0,2,1))

            self._Q[self._position-1],self._R[self._position-1]=fromMPSmat(tensorl,self._dx[self._position-1])
            self._mats[self._position]=np.diag(l)

    def __checkOrtho__(self,site,direction):
        assert(site<self._N)
        assert(site>=0)

        if direction>0:
            return np.tensordot(self.__tensor__(site),np.conj(self.__tensor__(site)),([0,2],[0,2]))
        if direction<0:
            return np.tensordot(self.__tensor__(site),np.conj(self.__tensor__(site)),([1,2],[1,2]))

    def __prepareTensor__(self,site,direction,fix):
        if direction>0:
            Qout=np.zeros((self._D,self._D))
            A0=self._mats[site].dot(np.eye(self._D)+self._dx[site]*self._Q[site])
            Rout=self._mats[site].dot(self._R[site]).dot(np.linalg.pinv(A0))
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),1)
            tempmat=r.dot(A0)
            vd,l,ud=np.linalg.svd(herm(tempmat))
            u=herm(ud)
            v=herm(vd)
            l=l/np.linalg.norm(l)
            if fix=='u':
                phase=np.angle(np.diag(u))
                unit=np.diag(np.exp(-1j*phase))
                u=u.dot(unit)
                v=herm(unit).dot(v)
            if fix=='v':
                warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
                phase=np.angle(np.diag(v))
                unit=np.diag(np.exp(-1j*phase))
                u=u.dot(herm(unit))
                v=unit.dot(v)

            tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            self._mats[site+1]=np.diag(l).dot(v)
                
                #if site==(self._N-1):
                #    print 'CMPS.__prepareTensor__(mode=accumulte): at position={0}: pushing v to self._V'.format(site+1)
                #    self._V=v.dot(self._V)


            #if mode=='diagonal':
            #    if fix=='u':
            #        phase=np.angle(np.diag(u))
            #        unit=np.diag(np.exp(-1j*phase))
            #        u=u.dot(unit)
            #        v=herm(unit).dot(v)
            #    if fix=='v':
            #        warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1},mode={2}): can produce jumps in Q and R! '.format(direction,fix,mode))
            #        phase=np.angle(np.diag(v))
            #        unit=np.diag(np.exp(-1j*phase))
            #        u=u.dot(herm(unit))
            #        v=unit.dot(v)
            #
            #    if site<(self._N-1):
            #        temptens=toMPSmat(self._Q[site+1],self._R[site+1],self._dx[site+1],self._dtype)                    
            #        temptens=np.tensordot(v,temptens,([1],[0]))
            #        self._Q[site+1],self._R[site+1]=fromMPSmat(temptens,self._dx[site])
            #
            #        tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            #        self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            #        self._mats[site+1]=np.diag(l)
            #
            #
            #    if site==(self._N-1):
            #        tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            #        self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            #        self._mats[site+1]=np.diag(l).dot(v)

            
        if direction<0:
            Qout=np.zeros((self._D,self._D))
            A0=(np.eye(self._D)+self._dx[site]*self._Q[site]).dot(self._mats[site+1])
            Rout=np.linalg.pinv(A0).dot(self._R[site]).dot(self._mats[site+1])
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),-1)
            
            tempmat=A0.dot(r)
            u,l,v=np.linalg.svd(tempmat)
            l=l/np.linalg.norm(l)
            #if mode=='accumulate':
            if fix=='u':
                warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
                phase=np.angle(np.diag(u))
                unit=np.diag(np.exp(-1j*phase))
                u=u.dot(unit)
                v=herm(unit).dot(v)
            if fix=='v':
                phase=np.angle(np.diag(v))
                unit=np.diag(np.exp(-1j*phase))
                u=u.dot(herm(unit))
                v=unit.dot(v)
                    
            tensor=np.tensordot(v,tensor,([1],[0]))
            self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            self._mats[site]=u.dot(np.diag(l))
                

            #if mode=='diagonal':
            #    if fix=='u':
            #        warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1},mode={2}): can produce jumps in Q and R! '.format(direction,fix,mode))
            #        phase=np.angle(np.diag(u))
            #        unit=np.diag(np.exp(-1j*phase))
            #        u=u.dot(unit)
            #        v=herm(unit).dot(v)
            #    if fix=='v':
            #        phase=np.angle(np.diag(v))
            #        unit=np.diag(np.exp(-1j*phase))
            #        u=u.dot(herm(unit))
            #        v=unit.dot(v)
            #
            #    if site>0:
            #        temptens=toMPSmat(self._Q[site-1],self._R[site-1],self._dx[site-1],self._dtype)
            #        temptens=np.transpose(np.tensordot(temptens,u,([1],[0])),(0,2,1))                    
            #        self._Q[site-1],self._R[site-1]=fromMPSmat(temptens,self._dx[site-1])
            #        tensor=np.tensordot(v,tensor,([1],[0]))
            #        self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            #        self._mats[site]=np.diag(l)
            #
            #    if site==0:
            #        tensor=np.tensordot(v,tensor,([1],[0]))
            #        self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            #        self._mats[site]=u.dot(np.diag(l))
                


def toMPSmat(Q,R,dx):
    assert(np.shape(Q)[0]==np.shape(Q)[1])
    D=np.shape(Q)[0]
    matrix=np.zeros((D,D,2)).astype(R.dtype)
    matrix[:,:,0]=(np.eye(D)+dx*Q)
    matrix[:,:,1]=np.copy(np.sqrt(dx)*R)
    return matrix


def fromMPSmat(mat,dx):
    D=np.shape(mat)[0]
    Q=np.copy((mat[:,:,0]-np.eye(D))/dx)
    R=np.copy(mat[:,:,1]/np.sqrt(dx))
    return Q,R



##mat is a matrix living on the links, containing the schmidt-values, but not neccessarily diagonal. To prepare Q,R in left orthonormalized form, mat has to live to the left of Q and R
##for right orthonormalization (direction<0), mat has to live to the right of Q and R.
#def prepareCMPSTensor(Q,R,mat,invmat,dx,direction,dtype=float):
#    if direction>0:
#        Qout=mat.dot(Q).dot(invmat)
#        Rout=mat.dot(R).dot(invmat)
#        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx,dtype),1)
#        Qout,Rout=fromMPSmat(tensor,dx)
#        matout=r.dot(mat)
#        return Qout,Rout,matout
#        
#    if direction<0:
#        Qout=invmat.dot(Q).dot(mat)
#        Rout=invmat.dot(R).dot(mat)
#        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx,dtype),-1)
#        Qout,Rout=fromMPSmat(tensor,dx)
#        matout=mat.dot(r)
#        return Qout,Rout,matout


def prepareCMPSTensorSVD(Q,R,mat,fix,dx,direction):
    D=np.shape(Q)[0]
    if direction>0:
        Qout=np.zeros((D,D)).astype(Q.dtype)
        A0=mat.dot(np.eye(D)+dx*Q)
        Rout=mat.dot(R).dot(np.linalg.pinv(A0))
        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),1)
        tempmat=r.dot(A0)
        vd,l,ud=np.linalg.svd(herm(tempmat))
        u=herm(ud)
        v=herm(vd)
        l=l/np.linalg.norm(l)
        if fix=='u':
            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
        if fix=='v':
            warnings.warn('prepareCMPSTensorfixPhase(): can produce jumps in Q and R! '.format(direction,fix))
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)

        tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
        Qout,Rout=fromMPSmat(tensor,dx)
        matout=np.diag(l).dot(v)
        return Qout,Rout,matout

            
        
    if direction<0:
        Qout=np.zeros((D,D)).astype(Q.dtype)
        A0=(np.eye(D)+dx*Q).dot(mat)
        Rout=np.linalg.pinv(A0).dot(R).dot(mat)
        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),-1)
        tempmat=A0.dot(r)
        u,l,v=np.linalg.svd(tempmat)
        l=l/np.linalg.norm(l)
        if fix=='u':
            warnings.warn('prepareCMPSTensorfixPhase(): can produce jumps in Q and R! '.format(direction,fix))
            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
        if fix=='v':
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)
                
        tensor=np.tensordot(v,tensor,([1],[0]))
        Qout,Rout=fromMPSmat(tensor,dx)
        matout=u.dot(np.diag(l))
        return Qout,Rout,matout



def prepareCMPSTensorfixPhase(Q,R,mat,dx,direction,dtype=float,method='accumulate'):
    D=np.shape(Q)[0]
    if direction>0:
        Qout=np.zeros((D,D))
        A0=mat.dot(np.eye(D)+dx*Q)
        Rout=mat.dot(R).dot(np.linalg.pinv(A0))
        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),1)
        tempmat=r.dot(A0)
        vd,l,ud=np.linalg.svd(herm(tempmat))
        u=herm(ud)
        v=herm(vd)
        l=l/np.linalg.norm(l)

        if method=='phasefree':
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)

            matrix=u.dot(np.diag(l)).dot(v).dot(np.diag(1./l))
            tensor=np.transpose(np.tensordot(tensor,matrix,([1],[0])),(0,2,1))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=np.diag(l)
            return Qout,Rout,matout

        if method=='accumulate':
            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
            tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=np.diag(l).dot(v)
            return Qout,Rout,matout
            
    if direction<0:
        Qout=np.zeros((D,D))
        A0=(np.eye(D)+dx*Q).dot(mat)
        Rout=np.linalg.pinv(A0).dot(R).dot(mat)
        tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),-1)
        
        tempmat=A0.dot(r)
        u,l,v=np.linalg.svd(tempmat)
        l=l/np.linalg.norm(l)

        if method=='phasefree':
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)

            matrix=np.diag(1./l).dot(u).dot(np.diag(l)).dot(v)
            tensor=np.tensordot(matrix,tensor,([1],[0]))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=np.diag(l)
            return Qout,Rout,matout

        if method=='accumulate':
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)

            tensor=np.tensordot(v,tensor,([1],[0]))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=u.dot(np.diag(l))

def prepareCMPSTensor(Q,R,mat,dx,direction,method=0):
    D=np.shape(Q)[0]
    if direction>0:
        if method==0:
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=mat.dot(np.eye(D)+dx*Q)
            Rout=mat.dot(R).dot(np.linalg.pinv(A0))
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),1)
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=r.dot(A0)
            matout=matout/np.sqrt(np.trace(matout.dot(herm(matout))))

        if method==1:
            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=np.eye(D)+dx*mat.dot(Q).dot(invmat)
            Rout=mat.dot(R).dot(invmat).dot(np.linalg.pinv(A0))
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),1)
            tempmat=r.dot(A0)
            q1,r1=cmf.qr(tempmat,'q')
            tensor=np.transpose(np.tensordot(tensor,q1,([1],[0])),(0,2,1))
            Qout,Rout=fromMPSmat(tensor,dx)

            matout=r1.dot(mat)

        return Qout,Rout,matout
    if direction<0:
        if method==0:
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=(np.eye(D)+dx*Q).dot(mat)
            Rout=np.linalg.pinv(A0).dot(R).dot(mat)
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),-1)
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=A0.dot(r)
            matout=matout/np.sqrt(np.trace(matout.dot(herm(matout))))


        if method==1:

            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=np.eye(D)+dx*invmat.dot(Q).dot(mat)
            Rout=np.linalg.pinv(A0).dot(invmat).dot(R).dot(mat)
            tensor,r=cmf.prepareTensor(toMPSmat(Qout,Rout,dx),-1)
            tempmat=A0.dot(r)
            q1,r1=cmf.qr(herm(tempmat),'q')
            tensor=np.tensordot(herm(q1),tensor,([1],[0]))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=mat.dot(r1)
            

        return Qout,Rout,matout




def transferOperator(Q,R,sigma,direction,vector):
    D=np.shape(Q)[0]
    if direction>0:
        x=np.conj(np.reshape(vector,(D,D)))
        if abs(sigma)<1E-14:
            return np.conj(np.reshape(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R),(D*D)))
        elif abs(sigma)>=1E-14:
            return np.conj(np.reshape(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R)-sigma*x,(D*D)))
    if direction<=0:
        x=np.reshape(vector,(D,D))
        if abs(sigma)<1E-14:
            return np.reshape(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R))-sigma*x,(D*D))



def transferOperator2ndOrder(Q,R,dx,sigma,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))

    if direction>0:
        #if abs(sigma)<1E-15:
        #    return np.reshape(np.transpose(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R)),(D*D))
        #elif abs(sigma)>=1E-15:
        #    return np.reshape(np.transpose(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R)-sigma*x),D*D)
        if abs(sigma)<1E-15:
            return np.reshape(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R)+dx*herm(Q).dot(x).dot(Q),(D*D))
            #return np.reshape(herm(Q).dot(np.transpose(x))+np.transpose(x).dot(Q)+herm(R).dot(np.transpose(x)).dot(R),(D*D))
        elif abs(sigma)>=1E-15:
            #return np.reshape(herm(Q).dot(np.transpose(x))+np.transpose(x).dot(Q)+herm(R).dot(np.transpose(x)).dot(R)-sigma*x,D*D)
            return np.reshape(herm(Q).dot(x)+x.dot(Q)+herm(R).dot(x).dot(R)+dx*herm(Q).dot(x).dot(Q)-sigma*x,D*D)
    if direction<=0:
        #if abs(sigma)<1E-15:
        #    return np.reshape(np.transpose(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R))),(D*D))
        #elif abs(sigma)>=1E-15:
        #    return (np.reshape(np.transpose(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R))-sigma*x),D*D))
        if abs(sigma)<1E-15:
            #print np.reshape(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R)),(D*D))
            return np.reshape(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q)),(D*D))
        elif abs(sigma)>=1E-15:
            return np.reshape(x.dot(herm(Q))+Q.dot(x)+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q))-sigma*x,D*D)



#dens0 is in matrix form
def UCtransferOperator(Q,R,dx,direction,dens0):
    D=np.shape(Q[0])[0]
    N=len(Q)
    dens=np.reshape(dens0,D*D)
    if direction>0:
        for n in range(N):
            dens=dostep(dens,Q,R,n,dx,direction)
    elif direction<0:
        for n in range(N-1,-1,-1):
            dens=dostep(dens,Q,R,n,dx,direction)
    return np.reshape(dens,(D,D))


#takes a density matrix "vector" in vector form; l and r are in matrix form
def pseudoOneMinusUCtransferOperator(Q,R,dx,l,r,direction,vector):
    D=np.shape(Q[0])[0]
    x=np.reshape(vector,(D,D))
    if direction >0:
        return np.reshape(x+np.trace(x.dot(r[-1]))*l[-1]-UCtransferOperator(Q,R,dx,direction,x),D*D)
    if direction <0:
        return np.reshape(x+np.trace(x.dot(l[0]))*r[0]-UCtransferOperator(Q,R,dx,direction,x),D*D)


def TDVPGMRESUC(Q,R,dx,ldens,rdens,inhom,x0,tolerance=1e-10,maxiteration=2000,datatype=float,direction=1):
    if direction>0:
        D=np.shape(Q[0])[0]
        mv=fct.partial(pseudoOneMinusUCtransferOperator,*[Q,R,dx,ldens,rdens,direction])
        LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=datatype)
        return sp.sparse.linalg.lgmres(LOP,inhom,x0,tol=tolerance,maxiter=maxiteration)
    if direction<0:
        D=np.shape(Q[0])[0]
        mv=fct.partial(pseudoOneMinusUCtransferOperator,*[Q,R,dx,ldens,rdens,direction])
        LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=datatype)
        return sp.sparse.linalg.lgmres(LOP,inhom,x0,tol=tolerance,maxiter=maxiteration)



#everything is passed in matrix form
def computeUCsteadyStateHamiltonianGMRES(Q,R,dx,mu,inter,mass,boundary,ldens,rdens,direction,thresh,imax,dtype=float):
    N=len(Q)
    D=np.shape(Q[0])[0]
    f0=np.zeros((D,D),dtype=dtype)
    if direction>0:
        for n in range(N):
            f0=np.copy(dofxstep(f0,ldens,rdens,Q,R,Q[-1],R[-1],ldens[-1],rdens[-1],n,dx,mass,mu,inter,direction))
        h=np.trace(f0.dot(rdens[-1]))
        inhom=np.reshape(f0-h*np.transpose(np.eye(D)),D*D) 
        [k2,info]=TDVPGMRESUC(Q,R,dx,ldens,rdens,inhom,np.reshape(boundary,(D*D)),thresh,imax,datatype=dtype,direction=1)
        f0=np.reshape(k2,(D,D))
        return f0

    if direction<0:
        mpo=np.zeros((B1,1,d1,d2),dtype=dtype)
        mpo[:,0,:,:]=mpopbc[0][:,0,:,:]
        R=initializeLayer(mps[0],np.eye(D),mps[0],mpo,-1)
        for n in range(len(mps)-1,-1,-1):
            R=addLayer(R,mps[n],mpopbc[n],mps[n],-1)    
        h=np.trace(R[:,:,-1].dot(ldens[0]))
        inhom=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D)),D2r*D2r) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D*D)),thresh,imax,datatype=dtype,direction=-1)
        R2=np.copy(R)
        R[:,:,-1]=np.reshape(k2,(D,D))

        return [np.copy(R),R2]


#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs2ndOrder(Q,R,dx,direction,numeig,init=None,datatype=float,nmax=10000000,tolerance=1e-8,ncv=10,which='SR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    mv=fct.partial(transferOperator2ndOrder,*[Q,R,dx,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)

    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print 'found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig)
        print eta
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))

    return eta[m],np.reshape(vec[:,m],D*D),numeig
    #return eigs(LOP,k=numeig, which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)


def UCTMeigs(Q,R,dx,direction,numeig,init=None,datatype=float,nmax=10000,tolerance=1e-8,ncv=10,which='LR'):
    D=np.shape(Q[0])[0]
    mv=fct.partial(UCtransferOperator,*[Q,R,dx,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)
    [eta,v]=eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print 'found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig)
        print eta
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))

    return eta[m],np.reshape(vec[:,m],D*D),numeig







#evolves blockham living on bond n1 to bond n2, in either left or right direction, using numerical derivatives
def evolveF(cmps,blockham,n1,n2,mpo,direction):
    if direction>0:
        assert(cmps._position>n1)
        assert(cmps._position>(n2-1))
        assert(n1<cmps._N)
        assert(n2<cmps._N)
        assert(n2>n1)
        f0=blockham[:,:,0]
        f1=np.copy(cmf.addLayer(blockham,cmps.__tensor__(n1),mpo[n1],cmps.__tensor__(n1),1))[:,:,0]
        dfdx=(f1-f0)/cmps._dx[n1]
        
        #dx=np.sum(cmps._dx[n1:n2])
        dx=cmps._xb[n2]-cmps._xb[n1]
        f=f0+dx*dfdx
        D1,D2,d1,d2=np.shape(mpo[n2-1])
        locmpo=np.zeros((1,D2,d1,d2),dtype=cmps._dtype)
        locmpo[:,:,:,:]=mpo[n2-1][-1,:,:,:]
        left=cmf.initializeLayer(cmps.__tensor__(n2-1),np.eye(cmps._D),cmps.__tensor__(n2-1),locmpo,direction=1)
        left[:,:,0]=np.copy(f)
        return left

    if direction<0:
        assert(cmps._position<n1)
        assert(cmps._position<n2)
        assert(n1>=0)
        assert(n2>=0)
        assert(n2>n1)
        f1=blockham[:,:,-1]
        f0=np.copy(cmf.addLayer(blockham,cmps.__tensor__(n2-1),mpo[n2-1],cmps.__tensor__(n2-1),-1))[:,:,-1]
        dfdx=(f1-f0)/cmps._dx[n2-1]
        dx=np.sum(cmps._dx[n1:n2])

        f=f1-dx*dfdx
        D1,D2,d1,d2=np.shape(mpo[n1])
        locmpo=np.zeros((D1,1,d1,d2),dtype=cmps._dtype)
        
        locmpo[:,0,:,:]=mpo[n1][:,0,:,:]
        right=cmf.initializeLayer(cmps.__tensor__(n1),np.eye(cmps._D),cmps.__tensor__(n1),locmpo,direction=-1)
        right[:,:,-1]=np.copy(f)
        return right



#calculates the left block hamiltonians using numerical derivatives, on a grid "grid", lb is initial cxondition at left end; grid is a BOND grid, not a site grid
def getL(cmps,mpo,lb,grid):
    #assert(cmps._position==cmps._N)
    #assert(grid[0]>0)

    L=[]
    for n in range(cmps._N+1):
        L.append(None)
    L[0]=np.copy(lb)
    for n in range(len(grid)):
        if n==0:
            L[grid[n]]=evolveF(cmps,L[0],0,grid[n],mpo,direction=1)
        if n>0:
            L[grid[n]]=evolveF(cmps,L[grid[n-1]],grid[n-1],grid[n],mpo,direction=1)
    return L


#calculates the right block hamiltonians using numerical derivatives, on a grid "grid", rb is initial condition on right end; grid is a BOND grid, not a site grid
def getR(cmps,mpo,rb,grid):
    #assert(cmps._position==0)
    #assert(grid[-1]<cmps._N)
    R=[]
    for n in range(cmps._N,-1,-1):
        R.append(None)
    R[cmps._N]=np.copy(rb)
    for n in range(len(grid)-1,-1,-1):
        if n==(len(grid)-1):
            R[grid[n]]=evolveF(cmps,R[cmps._N],grid[n],cmps._N,mpo,direction=-1)
            #R[cmps._N-grid[n]]=evolveF(cmps,R[cmps._N],grid[n],cmps._N,mpo,direction=-1)
        if n<(len(grid)-1):
            R[grid[n]]=evolveF(cmps,R[grid[n+1]],grid[n],grid[n+1],mpo,direction=-1)
    return R


#def dfdxLiebLiniger(cmps,n,mass,mu,interact,direction,fx):
#    if direction>0:
#        assert(cmps._position>n+1)
#        assert(n<cmps._N-1)
#        if n<(cmps._N-1):
#            D=cmps._D
#            kinetic=cmps._Q[n].dot(cmps._R[n+1])-cmps._R[n].dot(cmps._Q[n+1])+(cmps._R[n+1]-cmps._R[n])/(cmps._xm[n+1]-cmps._xm[n])
#            temp=1.0/(2.0*mass)*herm(kinetic).dot(kinetic)+mu[n]/2.0*herm(cmps._R[n]).dot(cmps._R[n])+mu[n+1]/2.0*herm(cmps._R[n+1]).dot(cmps._R[n+1])+\
#                  interact*herm(cmps._R[n].dot(cmps._R[n+1])).dot(cmps._R[n]).dot(cmps._R[n+1])
#            return np.conj(temp)+np.reshape(transferOperator2ndOrder(cmps._Q[n],cmps._R[n],cmps._dx[n],0.0,1,np.reshape(fx,(D*D))),(D,D))
#
#    if direction<0:
#        if n>0:
#            D=cmps._D
#            kinetic=cmps._Q[n-1].dot(cmps._R[n])-cmps._R[n-1].dot(cmps._Q[n])+(cmps._R[n]-cmps._R[n-1])/(cmps._xm[n]-cmps._xm[n-1])
#            temp=1.0/(2.0*mass)*kinetic.dot(herm(kinetic))+mu[n]*cmps._R[n].dot(herm(cmps._R[n]))+mu[n-1]*cmps._R[n-1].dot(herm(cmps._R[n-1]))+\
#                  interact*cmps._R[n-1].dot(cmps._R[n]).dot(herm(cmps._R[n-1].dot(cmps._R[n])))
#
#            return temp+np.reshape(transferOperator2ndOrder(cmps._Q[n],cmps._R[n],cmps._dx[n],0.0,-1,np.reshape(fx,D*D)),(D,D))

#def dfdxLiebLiniger(ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction,fx):
#    N=len(Q)
#    if direction>0:
#        if n>0:
#            D=np.shape(Q[n])[0]
#            ld0=ldens[n-1]
#            ld1=ldens[n]
#            term1=Q[n-1].dot(R[n])-R[n-1].dot(Q[n])+(R[n]-R[n-1])/dx
#            #term1=0.0*(Q[n-1].dot(R[n])-R[n-1].dot(Q[n])+(R[n]-R[n-1])/dx)
#            #temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+mu[n-1]/2.0*herm(R[n-1]).dot(ld1).dot(R[n-1])+mu[n]/2.0*herm(R[n]).dot(ld1).dot(R[n])+\
#            #    interact*herm(R[n-1].dot(R[n])).dot(ld0).dot(R[n-1]).dot(R[n])
#
#            temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+mu[n-1]/2.0*herm(R[n-1]).dot(ld1).dot(R[n-1])+mu[n]/2.0*herm(R[n]).dot(ld1).dot(R[n])+\
#                  interact*herm(R[n-1].dot(R[n])).dot(ld0).dot(R[n-1]).dot(R[n])
#                
#            return np.conj(temp)+np.reshape(transferOperator(Q[n],R[n],0.0,1,np.reshape(fx,D*D)),(D,D))
#        if n==0:#F contains ONLY THE LOCAL POTENTIAL
#            if Qbound==None:
#                D=np.shape(Q[n])[0]
#                ld=ldens[0]
#                temp=mu[0]*herm(R[0]).dot(ld).dot(R[0])+interact*herm(R[0].dot(R[0])).dot(ld).dot(R[0]).dot(R[0])
#                return np.conj(temp)+np.reshape(transferOperator(Q[n],R[n],0.0,1,np.reshape(fx,D*D)),(D,D))
#            if Qbound!=None:
#                D=np.shape(Q[n])[0]
#                ld0=lbound
#                ld1=ldens[n]
#                term1=Qbound.dot(R[n])-Rbound.dot(Q[n])+(R[n]-Rbound)/dx
#                #temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+mu[N-1]/2.0*herm(R[N-1]).dot(ld0).dot(R[N-1])+mu[n]/2.0*herm(R[n]).dot(ld1).dot(R[n])+\
#                #      interact*herm(Rbound.dot(R[n])).dot(ld0).dot(Rbound).dot(R[n])
#                temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+mu[N-1]/2.0*herm(R[N-1]).dot(ld0).dot(R[N-1])+mu[n]/2.0*herm(R[n]).dot(ld1).dot(R[n])+\
#                      interact*herm(R[n].dot(R[n])).dot(ld0).dot(R[n]).dot(R[n])
#
#                return np.conj(temp)+np.reshape(transferOperator(Q[n],R[n],0.0,1,np.reshape(fx,D*D)),(D,D))
#
#
#    if direction<0:
#        if n<(N-1):
#            D=np.shape(Q[n])[0]
#            rd0=rdens[n+1]
#            rd1=rdens[n+2]
#            term1=Q[n].dot(R[n+1])-R[n].dot(Q[n+1])+(R[n+1]-R[n])/dx
#            #temp=1.0/(2.0*mass)*term1.dot(rd1).dot(herm(term1))+mu[n]*R[n].dot(rd0).dot(herm(R[n]))+\
#            #    interact*R[n].dot(R[n]).dot(rd1).dot(herm(R[n].dot(R[n])))
#
#            temp=1.0/(2.0*mass)*term1.dot(rd1).dot(herm(term1))+mu[n]*R[n].dot(rd0).dot(herm(R[n]))+\
#                  interact*R[n].dot(R[n+1]).dot(rd1).dot(herm(R[n].dot(R[n+1])))
#            return temp+np.reshape(transferOperator(Q[n],R[n],0.0,-1,np.reshape(fx,D*D)),(D,D))
#        if n==(N-1):#F contains ONLY THE LOCAL POTENTIAL
#            if Qbound==None:
#                D=np.shape(Q[n])[0]
#                rd=rdens[N]
#                temp=mu[N-1]*R[N-1].dot(rd).dot(herm(R[N-1]))+interact*R[N-1].dot(R[N-1]).dot(rd).dot(herm(R[N-1].dot(R[N-1])))
#                return temp+np.reshape(transferOperator(Q[n],R[n],0.0,-1,np.reshape(fx,D*D)),(D,D))
#            if Qbound!=None:
#                sdf

def getBoundaryHams(cmps,mpo):
    #regauge the cmps:
    pos=cmps._position
    NUC=cmps._N
    D=cmps._D
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if pos==NUC:
        cmps.__positionSVD__(0,'v') 

        mps=cmps.__toMPS__(connect='left')
        eta,lss,numeig=cmf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')

        lbound=np.reshape(lss,(D,D))
        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z

    
        ldens=cmf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=cmf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)

        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 9'
        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        
        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')
        #print ' =========================================================================    at position 10'
        eta,rss,numeig=cmf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        
        rdens=cmf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=cmf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 11'
        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if pos==0:

        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')
        eta,rss,numeig=cmf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        
        rdens=cmf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=cmf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)

        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 13'
        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
        lb=np.copy(f0l)
        
        cmps.__positionSVD__(0,'v') 
        mps=cmps.__toMPS__(connect='left')
        eta,lss,numeig=cmf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        lbound=np.reshape(lss,(D,D))
        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
    
        ldens=cmf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=cmf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
            
        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 15'
        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)

        return lb,rb,lbound,rbound



#def getGridBoundaryHams(cmps,mpo,grid):
#    #regauge the cmps:
#    pos=cmps._position
#    NUC=cmps._N
#    D=cmps._D
#    assert((cmps._position==0)|(cmps._position==cmps._N))
#    if pos==NUC:
#        cmps.__positionSVD__(0,'v') 
#        #mps=cmps.__toMPS__(connect='left')
#        #eta,lss,numeig=cmf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
#        eta,lss,numeig=cmfnew.UCTMeigs(cmps,grid,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps.dtype,nmax=10000,tolerance=1E-12,ncv=100,which='LR')
#
#        lbound=np.reshape(lss,(D,D))
#        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
#        lbound=(lbound+herm(lbound))/2.0
#        Z=np.trace(np.eye(D).dot(lbound))
#        lbound=lbound/Z
#
#    
#        ldens=cmf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
#        rdens=cmf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
#
#        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
#        #print ' =========================================================================    at position 9'
#        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
#
#
#        
#        cmps.__positionSVD__(NUC,'u')
#        mps=cmps.__toMPS__(connect='right')
#        #print ' =========================================================================    at position 10'
#        eta,rss,numeig=cmf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
#        rbound=np.reshape(rss,(D,D))
#        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
#        rbound=(rbound+herm(rbound))/2.0
#        Z=np.trace(np.eye(D).dot(rbound))
#        rbound=rbound/Z
#        
#        rdens=cmf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
#        ldens=cmf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
#    
#        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
#        #print ' =========================================================================    at position 11'
#        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
#
#
#        lb=np.copy(f0l)
#        rb=np.copy(f0r)
#        return lb,rb,lbound,rbound
#
#    if pos==0:
#
#        cmps.__positionSVD__(NUC,'u')
#        mps=cmps.__toMPS__(connect='right')
#        eta,rss,numeig=cmf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
#        rbound=np.reshape(rss,(D,D))
#        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
#        rbound=(rbound+herm(rbound))/2.0
#        Z=np.trace(np.eye(D).dot(rbound))
#        rbound=rbound/Z
#        
#        rdens=cmf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
#        ldens=cmf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
#
#        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
#        #print ' =========================================================================    at position 13'
#        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
#        lb=np.copy(f0l)
#        
#        cmps.__positionSVD__(0,'v') 
#        mps=cmps.__toMPS__(connect='left')
#        eta,lss,numeig=cmf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
#        lbound=np.reshape(lss,(D,D))
#        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
#        lbound=(lbound+herm(lbound))/2.0
#        Z=np.trace(np.eye(D).dot(lbound))
#        lbound=lbound/Z
#    
#        ldens=cmf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
#        rdens=cmf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
#            
#        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
#        #print ' =========================================================================    at position 15'
#        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
#
#
#        lb=np.copy(f0l)
#        rb=np.copy(f0r)
#
#        return lb,rb,lbound,rbound


def getBoundaryHams2(cmps,mpo):
    #regauge the cmps:
    pos=cmps._position
    NUC=cmps._N
    D=cmps._D
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if pos==NUC:

        cmps.__position__bla(0) 
        eta,lss,numeig=cmf.UnitcellTMeigs(cmps.__topureMPS__(),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')

        lbound=np.reshape(lss,(D,D))
        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
    
        ldens=cmf.computeDensity(lbound,cmps.__topureMPS__(),direction=1,dtype=cmps._dtype)
        rdens=cmf.computeDensity(np.eye(D)/(1.0),cmps.__topureMPS__(),direction=-1,dtype=cmps._dtype)
        
        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 9'
        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(cmps.__topureMPS__(),mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        
        cmps.__position__bla(NUC)
        #print ' =========================================================================    at position 10'
        eta,rss,numeig=cmf.UnitcellTMeigs(cmps.__topureMPS__(),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        
        rdens=cmf.computeDensity(rbound,cmps.__topureMPS__(),direction=-1,dtype=cmps._dtype)
        ldens=cmf.computeDensity(np.eye(D),cmps.__topureMPS__(),direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 11'
        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(cmps.__topureMPS__(),mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if pos==0:

        cmps.__position__bla(NUC)
        eta,rss,numeig=cmf.UnitcellTMeigs(cmps.__topureMPS__(),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')


        rbound=np.reshape(rss,(D,D))
        rbound=cmf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        rdens=cmf.computeDensity(rbound,cmps.__topureMPS__(),direction=-1,dtype=cmps._dtype)
        ldens=cmf.computeDensity(np.eye(D),cmps.__topureMPS__(),direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 13'
        f0l=cmf.computeUCsteadyStateHamiltonianGMRES(cmps.__topureMPS__(),mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
        lb=np.copy(f0l)
        
        cmps.__position__bla(0) 
        #print ' =========================================================================    at position 14'

        #u,l,v=np.linalg.svd(cmps._mats[0])
        #
        #print np.imag(np.diag(u))/np.real(np.diag(u))
        ##print np.diag(u)
        #print
        #print l
        #print 
        #print v
        #print 
        #print 1.0/np.diag(cmps._connector)
        #cmps._connector=cmps._connector.dot(u)
        #cmps._mats[0]=np.diag(l).dot(v)
        eta,lss,numeig=cmf.UnitcellTMeigs(cmps.__topureMPS__(),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        lbound=np.reshape(lss,(D,D))
        lbound=cmf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
    
        ldens=cmf.computeDensity(lbound,cmps.__topureMPS__(),direction=1,dtype=cmps._dtype)
        rdens=cmf.computeDensity(np.eye(D)/(1.0),cmps.__topureMPS__(),direction=-1,dtype=cmps._dtype)
            
        f0r=np.zeros((D*D))#+np.random.rand(D*D)*1j
        #print ' =========================================================================    at position 15'
        f0r=cmf.computeUCsteadyStateHamiltonianGMRES(cmps.__topureMPS__(),mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)

        return lb,rb,lbound,rbound


def getBoundaryHamsHomo(A,B,rA,lB,mpo,kold=None,tol=1E-8,dtype=float):
    #D=cmps._D
    D=np.shape(A)[0]
    L0=cmf.initializeLayer(A,np.eye(D),A,mpo[0],1)
    L=cmf.addLayer(L0,A,mpo[1],A,1)    
    h=np.trace(L[:,:,0].dot(rA))
    inhoml=np.reshape(L[:,:,0]-h*np.transpose(np.eye(D)),D*D) 
    
    [kl,info] = cmf.TDVPGMRES(A,rA,np.eye(D),inhoml,kold,tol,1000,datatype=dtype,direction=1,momentum=0.0)

    L0[:,:,0]=np.reshape(kl,(D,D))


    R0=cmf.initializeLayer(B,np.eye(D),B,mpo[1],-1)
    R=cmf.addLayer(R0,B,mpo[0],B,-1)    
    h=np.trace(R[:,:,-1].dot(lB))
    inhomr=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D)),D*D) 
    [kr,info] = cmf.TDVPGMRES(B,np.eye(D),lB,inhomr,kold,tol,1000,datatype=dtype,direction=-1,momentum=0.0)
    R0[:,:,-1]=np.reshape(kr,(D,D))
    return L0,R0



#computes the action of the pseudo transfer operator T^P=T-|r)(l|:
#
#   (x| [ T-|r)(l| ]
#
#for the (UNSHIFTED) transfer operator T, with (l| and |r) the left and right eigenvectors (in matrix form) of T to eigenvalue 0
def pseudotransferOperator2ndOrder(Q,R,dx,l,r,direction,vector):
    D=np.shape(Q)[1]
    x=np.reshape(vector,(D,D))
    if direction >0:
        return transferOperator2ndOrder(Q,R,dx,0.0,direction,vector)-np.trace(x.dot(r))*np.reshape(l,(D*D))
    if direction <0:
        return transferOperator2ndOrder(Q,R,dx,0.0,direction,vector)-np.trace(x.dot(l))*np.reshape(r,(D*D))

#solves the equation system 
#
#   direction >0: (K| [ T-|r)(l| ]=(i|
#                     ------------
#or
#   direction <=0: [ T-|r)(l| ] |K)=|i)
#                  ------------
#
#where the underlined expressions are associated with the pseudo inverse of T (|r) and (l| are left and right eigenvectors of T to eigenvalue 0)
#for the vector (K| or |K), given the inhomogeinity vector (i| or |i) on the right side:
#
#        (i|=-(l| { 1/(2m) ([Q,R] x [\bar Q,\bar R]) + v R x \bar R + c (R x \bar R)*(R x \bar R)}
#        |i)=-{ 1/(2m) ([Q,R] x [\bar Q,\bar R]) + v R x \bar R + c (R x \bar R)*(R x \bar R)}|r)
def doInversion(Q,R,dx,l,r,ih,direction,x0=None,tolerance=1e-12,maxiteration=4000,datatype=float):
    D=np.shape(Q)[1]
    mv=fct.partial(pseudotransferOperator2ndOrder,*[Q,R,dx,l,r,direction])
    #mv=fct.partial(transferOperator,*[Q,R,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=datatype)
    #[x,info]=bicgstab(LOP,np.reshape(ih,D*D),x0,tol=tolerance,maxiter=maxiteration)
    [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0,tol=tolerance,maxiter=maxiteration,outer_k=6)
    
    while info<0:
        #print x
        #raw_input("doInversion: bicgstab reports breakdown with info={0}: restarting with random initial state".format(info))
        #[x,info]=bicgstab(LOP,np.reshape(ih,D*D),x0=np.random.rand(D*D),tol=tolerance,maxiter=maxiteration)        
        [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0=np.random.rand(D*D),tol=tolerance,maxiter=maxiteration,outer_k=6)
        
        #print x
        #raw_input("doInversion: exiting with info={0}".format(info))
    return np.reshape(x,(D,D))
    #return out-np.trace(r.dot(out))*l


def getBoundaryHamsHomotest(A,B,rA,lB,dx,mpo,kold=None,tol=1E-8,dtype=float):
    #D=cmps._D
    Ql,Rl=fromMPSmat(A,dx)
    D=np.shape(A)[0]

    L0=cmf.initializeLayer(A,np.eye(D),A,mpo[0],1)
    L=cmf.addLayer(L0,A,mpo[1],A,1)    
    h=np.trace(L[:,:,0].dot(rA))
    inhoml=np.reshape(L[:,:,0]-h*np.transpose(np.eye(D)),D*D) 
    
    [kl,info] = cmf.TDVPGMRES(A,rA,np.eye(D),inhoml,kold,tol,1000,datatype=dtype,direction=1,momentum=0.0)




    Atest=np.zeros((D,D,2),dtype=dtype)
    Atest[:,:,0]=np.copy(Ql)
    Atest[:,:,1]=np.copy(Rl)
    #Atest=np.copy(A)
    mu=-0.1
    inter=1.0
    mass=0.5
    mpotest=H.HomogeneousLiebLinigermpo(inter,mass,dtype=dtype)

    L0test=cmf.initializeLayer(Atest,np.eye(D),Atest,mpotest[0],1)
    Ltest=cmf.addLayer(L0test,Atest,mpotest[2],Atest,1)    
    #print np.shape(Ltest)
    Ltest=np.reshape(Ltest[:,:,0],(D,D))
    Ltest=Ltest+herm(Rl).dot(Rl)*mu
    htest=np.trace(Ltest.dot(rA))

    print htest*dx
    print h
    inhomltest=np.reshape(Ltest-htest*np.transpose(np.eye(D)),D*D) 

    [kltest,info] = cmf.TDVPGMRES(A,rA,np.eye(D),inhomltest,kold,tol,1000,datatype=dtype,direction=1,momentum=0.0)

    print kl
    print kltest
    print 
    raw_input()
    #kl2=doInversion(Ql,Rl,dx,np.eye(D),rA,inhoml,direction=1,x0=None,tolerance=1e-12,maxiteration=4000,datatype=float)

    #print Ql
    #print np.real(np.reshape(kl,(D,D)))-np.real(np.reshape(-kl2/dx,(D,D)))

    L0[:,:,0]=np.reshape(kl,(D,D))
    R0=cmf.initializeLayer(B,np.eye(D),B,mpo[1],-1)
    R=cmf.addLayer(R0,B,mpo[0],B,-1)    
    h=np.trace(R[:,:,-1].dot(lB))
    inhomr=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D)),D*D) 
    [kr,info] = cmf.TDVPGMRES(B,np.eye(D),lB,inhomr,kold,tol,1000,datatype=dtype,direction=-1,momentum=0.0)
    R0[:,:,-1]=np.reshape(kr,(D,D))
    return L0,R0




##assumes that Q and R are second order let orthonormal
#def dfdxLiebLiniger(ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction,fx):
#    N=len(Q)
#    if direction>0:
#        if n>0:
#            D=np.shape(Q[n])[0]
#            ld0=ldens[n-1]
#            ld1=ldens[n]
#
#            term1=Q[n-1].dot(R[n])-R[n-1].dot(Q[n])+(R[n]-R[n-1])/dx
#
#            term2=R[n-1].dot(R[n-1]).dot(np.eye(D)+dx*Q[n])
#            term3=R[n-1].dot(R[n-1]).dot(np.sqrt(dx)*R[n])
#
#            term4=R[n].dot(R[n])
#
#            term6=R[n-1].dot(np.eye(D)+dx*Q[n])
#            term7=R[n-1].dot(np.sqrt(dx)*R[n])
#
#
#            temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+\
#                  interact/2.0*(herm(term2).dot(ld0).dot(term2)+herm(term3).dot(ld0).dot(term3))+\
#                  interact/2.0*herm(term4).dot(ld1).dot(term4)+\
#                  mu[n-1]/2.0*(herm(term6).dot(ld0).dot(term6)+herm(term7).dot(ld1).dot(term7))+\
#                  mu[n]/2.0*(herm(R[n]).dot(ld0).dot(R[n]))
#            
#                
#            return np.conj(temp)+np.reshape(transferOperator(Q[n],R[n],dx,0.0,1,np.reshape(fx,D*D)),(D,D))
#        if n==0:#F contains ONLY THE LOCAL POTENTIAL
#            D=np.shape(Q[n])[0]
#            ld0=ldens[N-1]
#            ld1=ldens[n]
#            term1=Q[N-1].dot(R[n])-R[N-1].dot(Q[n])+(R[n]-R[N-1])/dx
#            
#            term2=R[N-1].dot(R[N-1]).dot(np.eye(D)+dx*Q[n])
#            term3=R[N-1].dot(R[N-1]).dot(np.sqrt(dx)*R[n])
#            
#            term4=R[n].dot(R[n])
#            
#            term6=R[N-1].dot(np.eye(D)+dx*Q[n])
#            term7=R[N-1].dot(np.sqrt(dx)*R[n])
#
#
#            temp=1.0/(2.0*mass)*herm(term1).dot(ld0).dot(term1)+\
#                  interact/2.0*(herm(term2).dot(ld0).dot(term2)+herm(term3).dot(ld0).dot(term3))+\
#                  interact/2.0*herm(term4).dot(ld1).dot(term4)+\
#                  mu[N-1]/2.0*(herm(term6).dot(ld0).dot(term6)+herm(term7).dot(ld1).dot(term7))+\
#                  mu[n]/2.0*(herm(R[n]).dot(ld0).dot(R[n]))
#                
#
#            return np.conj(temp)+np.reshape(transferOperator(Q[n],R[n],dx,0.0,1,np.reshape(fx,D*D)),(D,D))
#
#    if direction<0:
#        print 'dfdxLiebLiniger direction<0 not implemented'
#        sldkf
#



#n is always the index of the guy i want to evolve
#for pbc=False, l(N-1) and r(0) are obtained by EULER EXPLICIT EVOLUTION of l(N-2) and r(1)
def dostep(vector,Q,R,n,dx,direction):
    N=len(Q)
    D=np.shape(Q[n])[0]
    assert(n<N)
    if direction > 0:
        return vector + dx[n]*transferOperator(Q[n],R[n],0.0,direction,vector)
    if direction < 0:
        return vector + dx[n]*transferOperator(Q[n],R[n],0.0,direction,vector)

#fx are located on the links. f_i contains contributions to the left of site i, that is
#Q_i and R_i are NOT contained in f_i; in other words, it is the energy content to the left
#of site i

def dofxstep(fx,ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction):
    D=np.shape(Q[0])[0]
    N=len(Q)
    if direction>0:
        return fx+dx*dfdxLiebLiniger(ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction,fx)
    if direction<0:
        return fx+dx*dfdxLiebLiniger(ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction,fx)



def computefx(f0,ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,dx,mass,mu,interact,direction):
    D=np.shape(Q[0])[0]
    N=len(Q)
    f=[]
    f1=np.copy(f0)
    for site in range(N+1):
        f.append(site)
    if direction>0:
        f[0]=np.copy(f1)
        for n in range(N):
            f1=np.copy(dofxstep(f1,ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction))
            f[n+1]=np.copy(f1)
        return f
    if direction<0:
        f[len(x)-1]=np.copy(f1)
        for n in range(N-1,-1,-1):
            f1=np.copy(dofxstep(f1,ldens,rdens,Q,R,Qbound,Rbound,lbound,rbound,n,dx,mass,mu,interact,direction))
            f[n]=np.copy(f1)
        return f


#evolves an initial f0 over a unitcell of Q,R matrices to compute the f(L) at the other end of the unit cell
def UCcomputefx(ldens,rdens,Q,R,dx,mass,mu,interact,pbc,direction,datatype,finit):
    D=np.shape(Q[0])[0]
    N=len(Q)
    f=np.copy(finit)
    
    if direction>0:
        for n in range(N):
            f=np.copy(dofxstep(f,ldens,rdens,Q,R,None,None,None,None,n,dx,mass,mu,interact,direction))
            #print f
        return np.copy(f)#-f.dot(rdens[-1])*ldens[-1])

    if direction<0:
        for n in range(N-1,-1,-1):
            f=np.copy(dofxstep(f,ldens,rdens,Q,R,None,None,None,None,n,dx,mass,mu,interact,direction))
        return np.copy(f)#-f.dot(ldens[-1])*rdens[-1])


def getSteadyStatef0(finit,ldens,rdens,Q,R,xm,x,mass,mu,intstrength,pbc,direction,pseudo,implicit,conv,maxit,tolerance=1E-8,maxiteration=10000,datatype=float,outerk=40):
    converged=False
    if maxit==0:
        converged=True
    print('computing steady state f(x):')
    it=0
    while not converged:
        it=it+1
        stdout.write('.')
        stdout.flush()
        f1=UCcomputefx(ldens,rdens,Q,R,xm,x,mass,mu,intstrength,pbc,direction,pseudo,implicit,tolerance,maxiteration,datatype,outerk,finit)
        f1b=np.copy(f1-f1.dot(rdens[-1])*ldens[-1])
        if np.linalg.norm(finit-f1b)<conv:
            converged=True
        if it>=maxit:
            converged=True
        finit=np.copy(f1b)
        #print f0
    return finit-finit.dot(rdens[0])*ldens[0]




#first and last density matrices are the same
def computedensity(dens0,Q,R,dx,direction):
    D=np.shape(Q[0])[0]
    N=len(Q)
    dens=[]
    for n in range(N+1):
        dens.append(n)

    if direction>0:
        dens[0]=dens0
        dm=np.copy(np.reshape(dens0,(D*D)))
        for n in range(N):
            dm=dostep(dm,Q,R,n,dx,direction)
            dens[n+1]=np.reshape(dm,(D,D))

    elif direction<0:
        dens[-1]=dens0
        dm=np.copy(np.reshape(dens0,(D*D)))
        for n in range(N-1,-1,-1):
            dm=dostep(dm,Q,R,n,dx,direction)
            dens[n]=np.reshape(dm,(D,D))
    return dens

def UCpseudotransferOperator(Q,R,x,xm,l,r,direction,pbc,implicit,tolerance,maxiteration,datatype,dens0):
    D=np.shape(Q._mat)[0]
    dens=np.copy(dens0)
    if direction>0:
        for n in range(0,len(xm)):
            dx=x[n+1]-x[n]
            dens=dostep(dens,Q,R,n,dx,direction)
        return dens0-dens+dens0.dot(r)*l

    elif direction<0:
        for n in range(len(xm)-1,-1,-1):
            dx=x[n+1]-x[n]
            dens=dostep(dens,Q,R,n,dx,direction)
        return dens0-dens+l.dot(dens0)*r

#uses a left orthonormalization in the tangent plane:
#this update is a bulkupdate!!!
#it assumes that l=11, hence it doesn't appear in the routine.
#"n" is the updated site; when updating site "n", note that the r-matrix to the right of site "n" 
#is at rdens[n+1]
def tdvpUpdateKinetic(Q,R,mat,Qlbound,Rlbound,Qrbound,Rrbound,rdens,rbound,n,dx,mass,rcond=1E-10):
    D=np.shape(Q[0])[0]
    N=len(Q)
    if (n<(N-1))&(n>0):
        r1=np.reshape(rdens[n+1],(D,D))
        r2=np.reshape(rdens[n+2],(D,D))
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        left=1.0/dx*(R[n]-R[n-1])+Q[n-1].dot(R[n])-R[n-1].dot(Q[n])
        right=1.0/dx*(R[n+1]-R[n])+Q[n].dot(R[n+1])-R[n].dot(Q[n+1])
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))

        #NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        #would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.
        
        #this is second order update idenctical to the cmpsTDVPupdate
        return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
                               -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
                               -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
        
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(herm(R[n-1])).dot(left).dot(sqrtr1)-right.dot(r2).dot(isqrtr1)/dx-\
        #                           right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))

    if n == (N-1):
        #note that r[0] and r[-1] are the same for pbc
        r1=np.reshape(rdens[N],(D,D))
        r2=rbound
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        left=1.0/dx*(R[n]-R[n-1])+Q[n-1].dot(R[n])-R[n-1].dot(Q[n])
        right=1.0/dx*(Rrbound-R[n])+Q[n].dot(Rrbound)-R[n].dot(Qrbound)
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        #NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        #would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.

        #this is second order update idenctical to the cmdpTDVPupdate
        return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
                                   -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)\
                                   -R[n].dot(mat).dot(right).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))


        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(herm(R[n-1])).dot(left).dot(sqrtr1)-right.dot(r2).dot(isqrtr1)/dx-\
        #                           right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(Rbound)).dot(isqrtr1))


    if n == 0:
        r1=np.reshape(rdens[n+1],(D,D))
        r2=np.reshape(rdens[n+2],(D,D))
        sqrtr1=sqrtm(r1)
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        left=1.0/dx*(R[n]-Rlbound)+Qlbound.dot(R[n])-Rlbound.dot(Q[n])
        right=1.0/dx*(R[n+1]-R[n])+Q[n].dot(R[n+1])-R[n].dot(Q[n+1])

        #NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        #would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.


        #this is second order update idenctical to the cmdpTDVPupdate
        return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
                                   -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
                                   -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))


        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
        #                           -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))

#uses a left orthonormalization in the tangent plane:
#this update is a bulkupdate!!!
#it assumes that l=11, hence it doesn't appear in the routine.
#the updated site is the site where Q1 and R1 live
def tdvpUpdateInteraction(R,Rlbound,Rrbound,rdens,rbound,n,g,rcond=1E-10):
    D=np.shape(R[0])[0]
    N=len(R)
    if (n<(N-1))&(n>0):
        r1=np.reshape(rdens[n+1],(D,D))
        r2=np.reshape(rdens[n+2],(D,D))
        sqrtr1=sqrtm(r1)
        sqrtr2=sqrtm(r2)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        #return g*(herm(R[n]).dot(R[n]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n]).dot(r2).dot(herm(R[n])).dot(isqrtr1))
        return g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
    
    if n == (N-1):
        r1=np.reshape(rdens[N],(D,D))
        r2=rbound
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        #return g*(herm(R[n]).dot(R[n]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n]).dot(r2).dot(herm(R[n])).dot(isqrtr1))
        return g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(Rrbound).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))
    if n == 0:
        r1=np.reshape(rdens[1],(D,D))
        r2=np.reshape(rdens[2],(D,D))
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        #return g*(herm(R[n]).dot(R[n]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n]).dot(r2).dot(herm(R[n])).dot(isqrtr1))
        return g*(herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))

#uses a left orthonormalization in the tangent plane:
#this update is a bulkupdate!!!
#it assumes that l=11, hence it doesn't appear in the routine.
#the updated site is the site where Q1 and R1 live
#pbc has no effect
def tdvpUpdatePotential(Q,R,mat,Rlbound,rdens,n,mu,dx):
    D=np.shape(R[0])[0]
    N=len(R)

    #return mu[n]*R[n].dot(sqrtr1)+mu[n]*R[n].dot(sqrtr1)
    if (n>0):
        r1=np.reshape(rdens[n+1],(D,D))
        sqrtr1=sqrtm(r1)
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))

        #this is second order update idenctical to the cmdpTDVPupdate
        return mu[n-1]/2*(-dx*R[n].dot(mat).dot(herm(R[n-1])).dot(R[n-1]).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+\
                              dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)


        #return mu[n-1]/2*(-dx*R[n].dot(herm(R[n-1])).dot(R[n-1]).dot(sqrtr1)+dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)

    if n==0:
        r1=np.reshape(rdens[n+1],(D,D))
        sqrtr1=sqrtm(r1)


        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        #this is second order update idenctical to the cmdpTDVPupdate
        return mu[N-1]/2*(-dx*R[n].dot(mat).dot(herm(Rlbound)).dot(Rlbound).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+\
                                 dx*herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)

        #return mu[N-1]/2*(-dx*R[n].dot(herm(R[N-1])).dot(R[N-1]).dot(sqrtr1)+dx*herm(R[N-1]).dot(R[N-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)

        
#"n" is the updated site; when updating site "n", note that the r-matrix to the right of site "n" 
#is at rdens[n+1]
def tdvpupdateF(fx,Q,R,mat,rdens,n,dx,dtype=float):
    D=np.shape(Q[0])[0]
    N=len(Q)
    #f=np.reshape(fx[n],(D,D))
    #if n<(N-1):
    #    r1=np.reshape(rdens[n+1],(D,D))
    #    sqrtr1=sqrtm(r1)
    #    mps=np.zeros((D,D,2),dtype=dtype)
    #    VL=np.zeros((D,D,2),dtype=dtype)
    #    mps[:,:,0]=np.eye(D)+dx*Q[n]
    #    mps[:,:,1]=np.sqrt(dx)*R[n]
    #    VL[:,:,0]=-dx*herm(R[n])
    #    VL[:,:,1]=np.sqrt(dx)*np.eye(D)
    #
    #    L=np.reshape(cmf.GeneralizedMatrixVectorProduct(1,VL,mps,f),(D,D))
    #    term3=np.transpose(L).dot(sqrtr1)
    #    return np.transpose(L).dot(sqrtr1)
    #
    #if n == (N-1):
    #    r1=np.reshape(rdens[0],(D,D))
    #    sqrtr1=sqrtm(r1)
    #    mps=np.zeros((D,D,2),dtype=dtype)
    #    VL=np.zeros((D,D,2),dtype=dtype)
    #    mps[:,:,0]=np.eye(D)+dx*Q[n]
    #    mps[:,:,1]=np.sqrt(dx)*R[n]
    #    VL[:,:,0]=-dx*herm(R[n])
    #    VL[:,:,1]=np.sqrt(dx)*np.eye(D)
    #    L=np.reshape(cmf.GeneralizedMatrixVectorProduct(1,VL,mps,f),(D,D))
    #    return np.transpose(L).dot(sqrtr1)



    f=fx[n]
    if n<(N-1):
        r1=np.reshape(rdens[n+1],(D,D))
        sqrtr1=sqrtm(r1)
        
        #this is second order update idenctical to the cmdpTDVPupdate
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        return -R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1)
        
        #return -R[n].dot(np.conj(f)).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1)
    if n == (N-1):
        r1=np.reshape(rdens[N],(D,D))
        sqrtr1=sqrtm(r1)
        #this is second order update idenctical to the cmdpTDVPupdate
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        return -R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1)

        #return -R[n].dot(np.conj(f)).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1)










#uses a left orthonormalization in the tangent plane:
#this update is a bulkupdate!!!
#it assumes that l=11, hence it doesn't appear in the routine.
#"n" is the updated site; when updating site "n", note that the r-matrix to the right of site "n" 
#is at rdens[n+1]



def tdvpUpdate(Q,R,mat,Qlbound,Rlbound,Qrbound,Rrbound,rdens,rbound,n,dx,mass,g,mu,fx,rcond=1E-10):
    D=np.shape(Q[0])[0]
    N=len(Q)
    f=fx[n]

    if (n<(N-1))&(n>0):
        r1=np.reshape(rdens[n+1],(D,D))
        r2=np.reshape(rdens[n+2],(D,D))
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        left=1.0/dx*(R[n]-R[n-1])+Q[n-1].dot(R[n])-R[n-1].dot(Q[n])
        right=1.0/dx*(R[n+1]-R[n])+Q[n].dot(R[n+1])-R[n].dot(Q[n+1])
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))

        #NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        #would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.
        
        #this is second order update idenctical to the cmpsTDVPupdate
        K=1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
                               -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
                               -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
        In=g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
        
        Pot=mu[n-1]/2*(-dx*R[n].dot(mat).dot(herm(R[n-1])).dot(R[n-1]).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)

        Fu=-R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
        

        #if np.max(np.abs(K))>10.0:
        #    print 'K: large value found!!!', np.max(np.abs(K))
        #if np.max(np.abs(In))>10.0:
        #    print 'In: large value found!!!', np.max(np.abs(In))
        #if np.max(np.abs(Pot))>10.0:
        #    print 'Pot: large value found!!! ', np.max(np.abs(Pot))
        #if np.max(np.abs(Fu))>10.0:
        #    print 'Fu: large value found!!!', np.max(np.abs(Fu))
        #    print np.max(np.abs(Fu))

        
        return K+In+Pot+Fu
        
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
        #                       -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
        #                       -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))\
        #    +g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))\
        #    +mu[n-1]/2*(-dx*R[n].dot(mat).dot(herm(R[n-1])).dot(R[n-1]).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)\
        #    -R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
            
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(herm(R[n-1])).dot(left).dot(sqrtr1)-right.dot(r2).dot(isqrtr1)/dx-\
        #                           right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))



    if n == 0:
        if N>1:
            r1=np.reshape(rdens[n+1],(D,D))
            r2=np.reshape(rdens[n+2],(D,D))
        if N==1:
            r1=np.reshape(rdens[n+1],(D,D))
            r2=r1
        sqrtr1=sqrtm(r1)
        ##mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        if N>1:
            left=1.0/dx*(R[n]-Rlbound)+Qlbound.dot(R[n])-Rlbound.dot(Q[n])
            right=1.0/dx*(R[n+1]-R[n])+Q[n].dot(R[n+1])-R[n].dot(Q[n+1])
        if N==1:
            left=1.0/dx*(R[n]-Rlbound)+Qlbound.dot(R[n])-Rlbound.dot(Q[n])
            right=1.0/dx*(Rrbound-R[n])+Q[n].dot(Rrbound)-R[n].dot(Qrbound)
        #
        ##NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        ##would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.
        if N>1:
            K=1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
                          -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
                          -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
            In=g*(herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))
            Pot=mu[N-1]/2*(-dx*R[n].dot(mat).dot(herm(Rlbound)).dot(Rlbound).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)
            Fu=-R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
        
        if N==1:
            K=1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
                              -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)\
                              -R[n].dot(mat).dot(right).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))
            In=g*(herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1)+R[n].dot(Rrbound).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))
            Pot=mu[N-1]/2*(-dx*R[n].dot(mat).dot(herm(Rlbound)).dot(Rlbound).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)
            Fu=-R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
        
        #if np.max(np.abs(K))>10.0:
        #    print 'K: large value found!!!',np.max(np.abs(K))
        #if np.max(np.abs(In))>10.0:
        #    print 'In: large value found!!!',np.max(np.abs(In))
        #if np.max(np.abs(Pot))>10.0:
        #    print 'Pot: large value found!!! ', np.max(np.abs(Pot))
        #if np.max(np.abs(Fu))>10.0:
        #    print 'Fu: large value found!!!', np.max(np.abs(Fu))
        #    print np.max(np.abs(Fu))
        
        
        return K+In+Pot+Fu
        #this is second order update idenctical to the cmdpTDVPupdate
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
        #                               -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)\
        #                               -R[n].dot(mat).dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))\
        #        +g*(herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1)+R[n].dot(R[n+1]).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))\
        #        +mu[N-1]/2*(-dx*R[n].dot(mat).dot(herm(Rlbound)).dot(Rlbound).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(Rlbound).dot(Rlbound).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)\
        #        -R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
        
        
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Qlbound).dot(left).dot(sqrtr1)+R[n].dot(herm(Rlbound)).dot(left).dot(sqrtr1)\
        #                           -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Q[n+1])).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(R[n+1])).dot(isqrtr1))


    if n == (N-1):
        #note that r[0] and r[-1] are the same for pbc
        r1=np.reshape(rdens[N],(D,D))
        r2=rbound
        sqrtr1=sqrtm(r1)
        isqrtr1=np.linalg.pinv(sqrtr1,rcond=rcond)
        left=1.0/dx*(R[n]-R[n-1])+Q[n-1].dot(R[n])-R[n-1].dot(Q[n])
        right=1.0/dx*(Rrbound-R[n])+Q[n].dot(Rrbound)-R[n].dot(Qrbound)
        #mat=herm(np.linalg.pinv(np.eye(D)+dx*herm(Q[n])))
        #NOTE: the two terms width /dx will not exactly cancel, even for true TI, due to the inverse. A possible workaround
        #would be to subtract them from each other and set the result to zero manually if its smaller than some threshold.
        
        K=1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
                                   -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)\
                                   -R[n].dot(mat).dot(right).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))

        In=g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(Rrbound).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))
        
        Pot=mu[n-1]/2*(-dx*R[n].dot(mat).dot(herm(R[n-1])).dot(R[n-1]).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)
        
        Fu=-R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1
        #if np.max(np.abs(K))>10.0:
        #    print 'K: large value found!!!',np.max(np.abs(K))
        #if np.max(np.abs(In))>10.0:
        #    print 'In: large value found!!!', np.max(np.abs(In))
        #if np.max(np.abs(Pot))>10.0:
        #    print 'Pot: large value found!!! ', np.max(np.abs(Pot))
        #if np.max(np.abs(Fu))>10.0:
        #    print 'Fu: large value found!!!', np.max(np.abs(Fu))
        #    print np.max(np.abs(Fu))



        return K+In+Pot+Fu
        #this is second order update idenctical to the cmdpTDVPupdate
        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(mat).dot(herm(R[n-1])).dot(left).dot(sqrtr1)\
        #                           -right.dot(r2).dot(isqrtr1)/dx-right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)\
        #                           -R[n].dot(mat).dot(right).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))\
        #    +g*(herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1)+R[n].dot(Rrbound).dot(r2).dot(herm(Rrbound)).dot(isqrtr1))\
        #    +mu[n-1]/2*(-dx*R[n].dot(mat).dot(herm(R[n-1])).dot(R[n-1]).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+dx*herm(R[n-1]).dot(R[n-1]).dot(R[n]).dot(sqrtr1))+mu[n]*R[n].dot(sqrtr1)\
        #    -R[n].dot(mat).dot(np.conj(f)).dot(np.eye(D)+dx*Q[n]).dot(sqrtr1)+np.conj(f).dot(R[n]).dot(sqrtr1),isqrtr1


        #return 1.0/(2.0*mass)*(left.dot(sqrtr1)/dx+herm(Q[n-1]).dot(left).dot(sqrtr1)+R[n].dot(herm(R[n-1])).dot(left).dot(sqrtr1)-right.dot(r2).dot(isqrtr1)/dx-\
        #                           right.dot(r2).dot(herm(Qrbound)).dot(isqrtr1)-R[n].dot(right).dot(r2).dot(herm(Rbound)).dot(isqrtr1))
        




#def cmpspartialSVDorthonormalize(mps,direction,site1,site2):
#    assert(direction!=0),'do NOT use direction=0!'
#    N=len(mps)
#    if direction>0:
#        for site in range(site1,site2+1):
#            tensor,lam,v=cmf.prepareTensorSVD(mps[site],direction)
#            mps[site]=np.transpose(np.tensordot(tensor,v,([1],[0])),(0,2,1))
#
#    if direction<0:
#        for site in range(site2,site1-1,-1):
#            tensor,lam,v=cmf.prepareTensorSVD(mps[site],direction)
#            mps[site]=np.tensordot(v,tensor,([1],[0]))
#
#    return lam
#
#
#def cmpspartialQRorthonormalize(mps,direction,site1,site2):
#    assert(direction!=0),'do NOT use direction=0!'
#    N=len(mps)
#    if direction>0:
#        for site in range(site1,site2+1):
#            tensor,r=cmf.prepareTensor(mps[site],direction)
#            try :
#                u,lam,v=np.linalg.svd(r)
#            except np.linalg.LinAlgError:
#                u,lam,v=np.linalg.svd(r)
#            mat=u.dot(v)
#            mps[site]=np.transpose(np.tensordot(tensor,mat,([1],[0])),(0,2,1))
#
#    if direction<0:
#        for site in range(site2,site1-1,-1):
#            tensor,r=cmf.prepareTensor(mps[site],direction)
#            try :
#                u,lam,v=np.linalg.svd(r)
#            except np.linalg.LinAlgError:
#                u,lam,v=np.linalg.svd(r)
#
#            mat=u.dot(v)
#            mps[site]=np.tensordot(mat,tensor,([1],[0]))
#
#    return lam
#
#
#def mpsfromcmpssimple(Q,R,dx,direction,dtype=float):
#    D=np.shape(Q[0])[0]
#    N=len(Q)
#    mps=[]
#    for n in range(N):
#        matrix=np.zeros((D,D,2),dtype=dtype)
#        matrix[:,:,0]=np.copy(np.eye(D)+dx*Q[n])
#        matrix[:,:,1]=np.copy(np.sqrt(dx)*R[n])
#        mps.append(np.copy(matrix))
#    if direction>0:
#        rest=cmpspartialSVDorthonormalize(mps,1,0,N-1)
#        #rest=cmpspartialQRorthonormalize(mps,1,0,N-1)
#    if direction<0:
#        rest=cmpspartialSVDorthonormalize(mps,-1,0,N-1)
#        #rest=cmpspartialQRorthonormalize(mps,-1,0,N-1)
#
#    return mps


def cmpspartialSVDorthonormalize(mps,direction,site1,site2):
    assert(direction!=0),'do NOT use direction=0!'
    N=len(mps)
    if direction>0:
        for site in range(site1,site2+1):
            tensor,lam,v=cmf.prepareTensorSVD(mps[site],direction)
            v=np.diag(lam).dot(v)
            mps[site]=np.transpose(np.tensordot(tensor,v,([1],[0])),(0,2,1))

    if direction<0:
        for site in range(site2,site1-1,-1):
            tensor,lam,v=cmf.prepareTensorSVD(mps[site],direction)
            v=v.dot(np.diag(lam))
            print np.sum(lam)
            mps[site]=np.tensordot(v,tensor,([1],[0]))

    return lam


def cmpspartialQRorthonormalize(mps,direction,site1,site2):
    assert(direction!=0),'do NOT use direction=0!'
    N=len(mps)
    [D1,D2,d]=np.shape(mps[0])
    if direction>0:
        for site in range(site1,site2+1):
            tensor,r=cmf.prepareTensor(mps[site],direction)
            try :
                u,lam,v=np.linalg.svd(r)
            except np.linalg.LinAlgError:
                u,lam,v=np.linalg.svd(r)
            mat=u.dot(v)
            #mat=r/np.trace(r)*D1
            mps[site]=np.transpose(np.tensordot(tensor,mat,([1],[0])),(0,2,1))

    if direction<0:
        for site in range(site2,site1-1,-1):
            tensor,r=cmf.prepareTensor(mps[site],direction)
            try :
                u,lam,v=np.linalg.svd(r)
            except np.linalg.LinAlgError:
                u,lam,v=np.linalg.svd(r)
            
            mat=u.dot(v)
            #mat=r/np.trace(r)*D1

            mps[site]=np.tensordot(mat,tensor,([1],[0]))

    return r



def mpsfromcmpssimple(Q,R,dx,direction,dtype=float):
    D=np.shape(Q[0])[0]
    N=len(Q)
    mps=[]
    for n in range(N):
        matrix=np.zeros((D,D,2),dtype=dtype)
        matrix[:,:,0]=np.copy(np.eye(D)+dx[n]*Q[n])
        matrix[:,:,1]=np.copy(np.sqrt(dx[n])*R[n])
        mps.append(np.copy(matrix))
    rest=np.zeros((D,D))
    if direction>0:
        rest=cmf.partialQRorthonormalize(mps,1,0,N-1)
    if direction<0:
        rest=cmf.partialQRorthonormalize(mps,-1,0,N-1)


    return mps,rest


def mpsfromcmpssimpleQR(Q,R,dx,direction,dtype=float):
    D=np.shape(Q[0])[0]
    N=len(Q)
    mps=[]
    for n in range(N):
        matrix=np.zeros((D,D,2),dtype=dtype)
        matrix[:,:,0]=np.copy(np.eye(D)+dx*Q[n])
        matrix[:,:,1]=np.copy(np.sqrt(dx)*R[n])
        mps.append(np.copy(matrix))
    if direction>0:
        rest=cmpspartialQRorthonormalize(mps,1,0,N-1)
    if direction<0:
        rest=cmpspartialQRorthonormalize(mps,-1,0,N-1)

    return mps,rest

def mpsfromcmpssimpleSVD(Q,R,dx,direction,dtype=float):
    D=np.shape(Q[0])[0]
    N=len(Q)
    mps=[]
    for n in range(N):
        matrix=np.zeros((D,D,2),dtype=dtype)
        matrix[:,:,0]=np.copy(np.eye(D)+dx*Q[n])
        matrix[:,:,1]=np.copy(np.sqrt(dx)*R[n])
        mps.append(np.copy(matrix))
    if direction>0:
        rest=cmpspartialSVDorthonormalize(mps,1,0,N-1)
    if direction<0:
        rest=cmpspartialSVDorthonormalize(mps,-1,0,N-1)


    return mps,rest

def cmpsfrommpssimple(mps,dx):
    D=np.shape(mps[0])[0]
    N=len(mps)
    Q=[]
    R=[]
    for n in range(N):
        Q.append((mps[n][:,:,0]-np.eye(D))/dx[n])
        R.append(mps[n][:,:,1]/np.sqrt(dx[n]))
    return [R,Q]



#left gauge for gauge>0
#right gauge for gauge<0
def cmpsintergriddata(R,K,x,xdense,gauge,interpmethod,dtype=float):
    assert(x[0]<=xdense[0])
    assert(x[-1]>=xdense[-1])
    assert(len(x)==len(R))
    N=len(x)
    
    D=np.shape(R[0])[0]
    Rtemp=np.zeros((D,D,N),dtype=dtype)
    Ktemp=np.zeros((D,D,N),dtype=dtype)

    Rtempdense=np.zeros((D,D,len(xdense)),dtype=dtype)
    Ktempdense=np.zeros((D,D,len(xdense)),dtype=dtype)

    for n in range(N):
        Rtemp[:,:,n]=np.copy(R[n])
        Ktemp[:,:,n]=np.copy(K[n])

    for n1 in range(0,D):
        for n2 in range(0,D):
            Rtempdense[n1,n2,:] = griddata(x,Rtemp[n1,n2,:], (xdense), method=interpmethod)
            Ktempdense[n1,n2,:] = griddata(x,Ktemp[n1,n2,:], (xdense), method=interpmethod)


    Rnew=[]
    Knew=[]
    Qnew=[]
    #left gauge for gauge>0
    if gauge>0:
        for n in range(len(xdense)):
            Rnew.append(Rtempdense[:,:,n])
            Knew.append(Ktempdense[:,:,n])
            Qnew.append(Ktempdense[:,:,n]-0.5*herm(Rtempdense[:,:,n]).dot(Rtempdense[:,:,n]))
        return [Rnew,Knew,Qnew]
    #right gauge for gauge<0
    if gauge<0:
        for n in range(len(xdense)):
            Rnew.append(Rtempdense[:,:,n])
            Knew.append(Ktempdense[:,:,n])
            Qnew.append(Ktempdense[:,:,n]-0.5*Rtempdense[:,:,n].dot(herm(Rtempdense[:,:,n])))
        return [Rnew,Knew,Qnew]



#left gauge for gauge>0
#right gauge for gauge<0
#x is the current grid
#xdense is the new grid
def cmpsinterp(Q,R,x,xdense,k):
    assert(x[0]<=xdense[0])
    assert(x[-1]>=xdense[-1])
    assert(len(x)==len(R))
    N=len(x)
    D=np.shape(R[0])[0]
    Rtemp=np.zeros((D,D,N)).astype(R[0].dtype)
    Qtemp=np.zeros((D,D,N)).astype(R[0].dtype)
    
    Rtempdense=np.zeros((D,D,len(xdense))).astype(R[0].dtype)
    Qtempdense=np.zeros((D,D,len(xdense))).astype(R[0].dtype)
    for n in range(N):
        Rtemp[:,:,n]=np.copy(R[n])
        Qtemp[:,:,n]=np.copy(Q[n])

    for n1 in range(0,D):
        for n2 in range(0,D):
            splRreal=splrep(x,np.real(Rtemp[n1,n2,:]),k=k)
            splQreal=splrep(x,np.real(Qtemp[n1,n2,:]),k=k)
            splRimag=splrep(x,np.imag(Rtemp[n1,n2,:]),k=k)
            splQimag=splrep(x,np.imag(Qtemp[n1,n2,:]),k=k)

            Rtempdense[n1,n2,:] = splev(xdense,splRreal)+1j*splev(xdense,splRimag)
            Qtempdense[n1,n2,:] = splev(xdense,splQreal)+1j*splev(xdense,splQimag)

    Rnew=[]
    Qnew=[]
    for n in range(len(xdense)):
        Rnew.append(Rtempdense[:,:,n])
        Qnew.append(Qtempdense[:,:,n])
    return [Rnew,Qnew]

def cmpsUnivariateSpline(R,Q,x,xdense,interpmethod,dtype=float):
    assert(x[0]<=xdense[0])
    assert(x[-1]>=xdense[-1])
    assert(len(x)==len(R))
    N=len(x)
    
    D=np.shape(R[0])[0]
    Rtemp=np.zeros((D,D,N),dtype=dtype)
    Qtemp=np.zeros((D,D,N),dtype=dtype)

    Rtempdense=np.zeros((D,D,len(xdense)),dtype=dtype)
    Qtempdense=np.zeros((D,D,len(xdense)),dtype=dtype)

    for n in range(N):
        Rtemp[:,:,n]=np.copy(R[n])
        Qtemp[:,:,n]=np.copy(Q[n])
    Rs=[]
    Qs=[]
    for n1 in range(D):
        for n2 in range(D):
            Rs.append(Rtemp[n1,n2,:])
            Qs.append(Qtemp[n1,n2,:])

    for n1 in range(0,D):
        for n2 in range(0,D):
            splR = UnivariateSpline(x,Rtemp[n1,n2,:], k=3)
            splQ = UnivariateSpline(x,Qtemp[n1,n2,:], k=3)
            Rtempdense[n1,n2,:] = splR(xdense)
            Qtempdense[n1,n2,:] = splQ(xdense)

    Rnew=[]
    Qnew=[]
    for n in range(len(xdense)):
        Rnew.append(Rtempdense[:,:,n])
        Qnew.append(Qtempdense[:,:,n])
    return [Rnew,Qnew]


#mats is a list of matrices
def matrixinterp(mats,x,xdense,interpmethod,dtype):
    D=np.shape(mats[0])[0]
    assert(x[0]<=xdense[0])
    assert(x[-1]>=xdense[-1])
    assert(len(x)==len(mats))
    N=len(x)
    temp=np.zeros((D,D,N),dtype=dtype)
    tempdense=np.zeros((D,D,len(xdense)),dtype=dtype)
    for n in range(N):
        temp[:,:,n]=np.copy(mats[n])

    for n1 in range(0,D):
        for n2 in range(0,D):
            tempdense[n1,n2,:] = griddata(x,temp[n1,n2,:], (xdense), method=interpmethod)

    matsnew=[]
    for n in range(len(xdense)):
        matsnew.append(tempdense[:,:,n])

    return matsnew


def getLocalLiebLinigerMPO(mu,inter,mass,site,dx,dtype,proj):
    N=len(mu)
    mul=np.zeros(2,dtype=dtype)
    mul[0]=mu[site-1]
    mul[1]=mu[site]

    dx_=np.zeros(2)
    dx_[0]=dx[site-1]
    dx_[1]=dx[site]
    mpol=H.projectedLiebLinigermpo3(mul,inter,mass,dx_,True,dtype,proj=proj)
    mur=np.zeros(2,dtype=dtype)
    if site<(N-1):
        mur[0]=mu[site]
        mur[1]=mu[site+1]

        dx_[0]=dx[site]
        dx_[1]=dx[site+1]

    if site==(N-1):
        mur[0]=mu[site]
        mur[1]=mu[0]

        dx_[0]=dx[site]
        dx_[1]=dx[0]

    mpor=H.projectedLiebLinigermpo3(mur,inter,mass,dx_,True,dtype,proj=proj)
    return [mpol,mpor]


def connectcMPS(cmps,Nconnect):
    cmps.__position__(0)

    cmpsc=copy.deepcopy(cmps)
    cmpsc.__position__(cmpsc._N)
    Q=np.copy(np.transpose(cmpsc._Q,(1,2,0)))
    R=np.copy(np.transpose(cmpsc._R,(1,2,0)))
    mat=cmpsc._mats[cmpsc._N].dot(cmpsc._connector)
    x0=np.copy(cmpsc._xm-cmpsc._xb[0])
    x=np.copy(x0)
    for n in range(Nconnect-1):
        cmpsc=copy.deepcopy(cmps)
        cmpsc._mats[0]=mat.dot(cmpsc._mats[0])
        cmpsc.__position__(cmpsc._N)
        mat=cmpsc._mats[cmpsc._N].dot(cmpsc._connector)
        Q=np.append(Q,np.copy(np.transpose(cmpsc._Q,(1,2,0))),axis=2)
        R=np.append(R,np.copy(np.transpose(cmpsc._R,(1,2,0))),axis=2)
        x=np.append(x,x0+(n+1)*cmpsc._L)
    return x,Q,R
