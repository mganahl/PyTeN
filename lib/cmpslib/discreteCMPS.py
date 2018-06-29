#!/usr/bin/env python
import numpy as np
import scipy as sp
import math,sys
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
import functools as fct
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.integrate import ode

#import cannot be called with cmpslib.cmpsfunctions because cmpsfunctions imports discreteCMPS as well. 
#this will lead to import errors
#import lib.cmpslib.cmpsfunctions as cmf
#import lib.mpslib.mpsfunctions as mf
#import lib.mpslib.Hamiltonians as H
#import lib.utils.utilities as utils
try:
    cmf=sys.modules['lib.cmpslib.cmpsfunctions']
except KeyError:
    import lib.cmpslib.cmpsfunctions as cmf
try:
    mf=sys.modules['lib.mpslib.mpsfunctions']
except KeyError:
    import lib.mpslib.mpsfunctions as mf
try:
    H=sys.modules['lib.mpslib.Hamiltonians']
except KeyError:
    import lib.mpslib.Hamiltonians as H
try:
    utils=sys.modules['lib.utils.utilities']
except KeyError:
    import lib.utils.utilities as utils





comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


#This library contais routines for the discrete cMPS calculations. 
#It contains a class definition DiscreteCMPS for a single species of bosons, and routines needed to do dmrg, interpolation, measuring operators (mostly for Lieb Liniger type models)

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

#dens0 is in matrix form;
#start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
#it returns a list of densities at position of grid; the length of the list is len(grid)+1
def computeDensityGrid(dens0,cmps,grid,direction):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    assert(grid[0]==0)
    assert(grid[-1]==(cmps._N-1))
    densities=[]
    for n in range(len(grid)+1):
        densities.append(None)
    if direction>0:
        #assert(cmps._position==0)
        #start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
        #the last step is evolving dens[grid[-1]] to dens[grid[-1]+1]; the returned dens is then living on 
        #the equivalent bond as the initial densitymatrix, just evolved by a single unit cell
        if(cmps._position==0):
            densities[0]=np.copy(dens0)
            connector=cmps._connector.dot(cmps._U).dot(cmps._mats[0])
            dens=np.transpose(connector).dot(dens0).dot(np.conj(connector))
            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                dens=evolveDensityMatrixMatrix(dens,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
                densities[n+1]=np.copy(dens)

            dens=evolveDensityMatrixMatrix(dens,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._V).dot(dens).dot(np.conj(cmps._V))
            densities[len(grid)]=np.copy(dens)
            return densities

        if(cmps._position==cmps._N):
            densities[0]=np.copy(dens0)
            connector=cmps._U
            dens=np.transpose(connector).dot(dens0).dot(np.conj(connector))

            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                dens=evolveDensityMatrixMatrix(dens,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
                densities[n+1]=np.copy(dens)

            dens=evolveDensityMatrixMatrix(dens,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)).dot(dens).dot(np.conj(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)))
            densities[len(grid)]=np.copy(dens)
            return densities


    if direction<0:
        if cmps._position==cmps._N:
            densities[len(grid)]=np.copy(dens0)
            connector=cmps._mats[-1].dot(cmps._V).dot(cmps._connector)
            dens=connector.dot(dens0).dot(herm(connector))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                dens=evolveDensityMatrixMatrix(dens,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
                densities[n]=np.copy(dens)

            dens=evolveDensityMatrixMatrix(dens,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._U.dot(dens).dot(herm(cmps._U))
            densities[0]=np.copy(dens)
            return densities

        if cmps._position==0:
            densities[len(grid)]=np.copy(dens0)
            connector=cmps._V
            dens=connector.dot(dens0).dot(herm(connector))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                dens=evolveDensityMatrixMatrix(dens,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
                densities[n]=np.copy(dens)

            dens=evolveDensityMatrixMatrix(dens,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._connector.dot(cmps._U).dot(cmps._mats[0]).dot(dens).dot(herm(cmps._connector.dot(cmps._U).dot(cmps._mats[0])))
            densities[0]=np.copy(dens)
            #now absorb the connector and the boundary unitaries into dens:
            return densities



def toMPSmat(Q,R,dx):
    assert(np.shape(Q)[0]==np.shape(Q)[1])
    D=np.shape(Q)[0]
    matrix=np.zeros((D,D,2)).astype(R.dtype)
    matrix[:,:,0]=(np.eye(D)+dx*Q)
    matrix[:,:,1]=np.copy(np.sqrt(dx)*R)
    return matrix

def toMPOmat(Gamma,dx):
    D=np.shape(Gamma[0][0])[0]
    matrix=np.zeros((D,D,2,2)).astype(Gamma[0][0].dtype)
    matrix[:,:,0,0]=(np.eye(D)+dx*Gamma[0][0])
    matrix[:,:,0,1]=(np.sqrt(dx)*Gamma[0][1])
    matrix[:,:,1,0]=(np.sqrt(dx)*Gamma[1][0])    
    matrix[:,:,1,1]=(np.eye(D)+dx*Gamma[1][1])
    #matrix[:,:,1,1]=(Gamma[1][1])        

    return matrix


def fromMPSmat(mat,dx):
    D=np.shape(mat)[0]
    Q=np.copy((mat[:,:,0]-np.eye(D))/dx)
    R=np.copy(mat[:,:,1]/np.sqrt(dx))
    return Q,R

def toMPS(Q,R,dx):
    D=Q.shape[0]
    dtype=Q[0,0,0].dtype

    mps=[]
    for n in range(Q.shape[2]):
        m=np.zeros((D,D,2)).astype(dtype)
        m[:,:,0]=np.eye(D)+dx[n]*Q[:,:,n]
        m[:,:,1]=np.sqrt(dx[n])*R[:,:,n]
        mps.append(np.copy(m))
    return mps

def fromMPS(mps,dx):
    D=mps[0].shape[0]
    dtype=mps[0][0,0,0].dtype

    Q=np.zeros((D,D,len(mps))).astype(dtype)
    R=np.zeros((D,D,len(mps))).astype(dtype)
    for n in range(Q.shape[2]):
        Q[:,:,n]=(mps[n][:,:,0]-np.eye(D))/dx[n]
        R[:,:,n]=mps[n][:,:,1]/np.sqrt(dx[n])
    return Q,R



def getLiebLinigerEDens(cmps,mu,mass,inter):
    energy=np.zeros(cmps._N,dtype=cmps._dtype)
    assert(cmps._position==cmps._N)
    if cmps._position==cmps._N:
        for n in range(cmps._N):
            if n<(cmps._N-1):
                K=(cmps._Q[n].dot(cmps._R[n+1])-cmps._R[n].dot(cmps._Q[n+1])+(cmps._R[n+1]-cmps._R[n])/(cmps._xm[n+1]-cmps._xm[n])).dot(cmps._mats[n+2])
                I=cmps._R[n].dot(cmps._R[n+1]).dot(cmps._mats[n+2])
                V1=cmps._R[n].dot(cmps._mats[n+1])
                V2=cmps._R[n+1].dot(cmps._mats[n+2])
                energy[n]=np.real(1.0/(2.0*mass)*(np.trace(herm(K).dot(K)))+inter*np.trace(herm(I).dot(I))+mu[n]/2.0*np.trace(herm(V1).dot(V1))+mu[n+1]/2.0*np.trace(herm(V2).dot(V2)))

            if n==(cmps._N-1):
                K=(cmps._Q[n].dot(cmps._mats[n+1]).dot(cmps.__connection__(False)).dot(cmps._R[0])-cmps._R[n].dot(cmps._mats[n+1]).dot(cmps.__connection__(False)).dot(cmps._Q[0])+\
                   (cmps._mats[n+1].dot(cmps.__connection__(False)).dot(cmps._R[0])-cmps._R[n].dot(cmps._mats[n+1]).dot(cmps.__connection__(False)))/((cmps._xb[-1]-cmps._xm[n])+(cmps._xm[0]-cmps._xb[0]))).dot(cmps._mats[1])
                I=cmps._R[n].dot(cmps._mats[n+1]).dot(cmps.__connection__(False)).dot(cmps._R[0]).dot(cmps._mats[1])
                V1=cmps._R[n].dot(cmps._mats[n+1])
                V2=cmps._R[0].dot(cmps._mats[1])
                energy[n]=np.real(1.0/(2.0*mass)*(np.trace(herm(K).dot(K)))+inter*np.trace(herm(I).dot(I))+mu[n]/2.0*np.trace(herm(V1).dot(V1))+mu[0]/2.0*np.trace(herm(V2).dot(V2)))
    return energy


def getLiebLinigerDens(cmps):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    NUC=cmps._N
    dens=np.zeros(NUC,dtype=cmps._dtype)
    for si in range(NUC):
        if cmps._position==0:
            dens[si]=np.real(np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(cmps._R[si]).dot(herm(cmps._R[si]))))
        if cmps._position==cmps._N:
            dens[si]=np.real(np.trace(herm(cmps._R[si]).dot(cmps._R[si]).dot(cmps._mats[si+1]).dot(herm(cmps._mats[si+1]))))
    return dens


def getLiebLinigerDensDens(cmps):
    assert((cmps._position==0)|(cmps._position==cmps._N))
    NUC=cmps._N
    denssq=np.zeros(NUC,dtype=cmps._dtype)
    for si in range(NUC):
        if cmps._position==0:
            if si<NUC-1:
                denssq[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(cmps._R[si]).dot(cmps._R[si+1]).dot(herm(cmps._R[si+1])).dot(herm(cmps._R[si])))
            if si==NUC-1:
                RR=cmps._R[cmps._N-1].dot(cmps.__connection__(False)).dot(cmps._mats[0]).dot(cmps._R[0])
                denssq[si]=np.trace(herm(cmps._mats[si]).dot(cmps._mats[si]).dot(RR).dot(herm(RR)))

        if cmps._position==cmps._N:
            if si>0:
                denssq[si]=np.real(np.trace(herm(cmps._R[si]).dot(herm(cmps._R[si-1])).dot(cmps._R[si-1]).dot(cmps._R[si]).dot(cmps._mats[si+1]).dot(herm(cmps._mats[si+1]))))
            if si==0:
                RR=cmps._R[cmps._N-1].dot(cmps._mats[cmps._N]).dot(cmps.__connection__(False)).dot(cmps._R[0])
                denssq[si]=np.real(np.trace(herm(RR).dot(RR).dot(cmps._mats[1].dot(herm(cmps._mats[1])))))

    return denssq


#compute entanglement entropy of a finite region
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
        print (n_)
        if (n%cmps._N==0)&(n1>0):
            rtensor=np.tensordot(cmps.__connection__(reset_unitaries=False).dot(cmps._mats[0]),cmps.__tensor__(n_),([1],[0]))
        if n%cmps._N!=0:
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





def getGridBoundaryHams(cmps,grid,mpo):
    #regauge the cmps:
    pos=cmps._position
    NUC=cmps._N
    D=cmps._D
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if pos==NUC:
        cmps.__positionSVD__(0,'v') 

        mps=cmps.__toMPS__(connect='left')
        #eta,lss,numeig=UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        eta,lss,numeig=UCTMeigs(cmps,grid,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
    
        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)

        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)

        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')

        #eta,rss,numeig=UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')        
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        
        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if pos==0:

        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')
        #eta,rss,numeig=UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        
        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)

        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
        lb=np.copy(f0l)
        
        cmps.__positionSVD__(0,'v') 
        
        
        mps=cmps.__toMPS__(connect='left')
        #eta,lss,numeig=UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1E-16,which='LR')
        eta,lss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
    
        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
            
        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)

        return lb,rb,lbound,rbound


def getBoundaryHams(cmps,mpo):
    #regauge the cmps:
    pos=cmps._position
    NUC=cmps._N
    D=cmps._D
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if pos==NUC:
        #USING __position__ INSTEAD OF __positionSVD__ SEEMS TO BE OK SO FAR
        #cmps.__positionSVD__(0,'v') 
        cmps.__position__(0) 
        mps=cmps.__toMPS__(connect='left')
        eta,lss,numeig=mf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')
        #eta,lss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
        if cmps._dtype==float:
            lbound=np.real((lbound+herm(lbound))/2.0)
        if cmps._dtype==complex:
            lbound=(lbound+herm(lbound))/2.0

    
        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)

        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=mf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        #USING __position__ INSTEAD OF __positionSVD__ SEEMS TO BE OK SO FAR
        #cmps.__positionSVD__(NUC,'u')
        cmps.__position__(NUC)
        mps=cmps.__toMPS__(connect='right')
        eta,rss,numeig=UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),nmax=10000,tolerance=1E-16,which='LR')
        #eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')        
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        if cmps._dtype==float:
            rbound=np.real((rbound+herm(rbound))/2.0)
        if cmps._dtype==complex:
            rbound=(rbound+herm(rbound))/2.0

        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=mf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if pos==0:
        #USING __position__ INSTEAD OF __positionSVD__ SEEMS TO BE OK SO FAR
        #cmps.__positionSVD__(NUC,'u')
        cmps.__position__(NUC)

        mps=cmps.__toMPS__(connect='right')
        eta,rss,numeig=mf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),nmax=10000,tolerance=1E-16,which='LR')
        #eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(np.eye(D).dot(rbound))
        rbound=rbound/Z
        if cmps._dtype==float:
            rbound=np.real((rbound+herm(rbound))/2.0)
        if cmps._dtype==complex:
            rbound=(rbound+herm(rbound))/2.0
        
        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)

        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=mf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
        lb=np.copy(f0l)
        
        #USING __position__ INSTEAD OF __positionSVD__ SEEMS TO BE OK SO FAR
        #cmps.__positionSVD__(0,'v') 
        cmps.__position__(0) 
        
        
        mps=cmps.__toMPS__(connect='left')
        eta,lss,numeig=mf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')
        #eta,lss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(np.eye(D).dot(lbound))
        lbound=lbound/Z
        if cmps._dtype==float:
            lbound=np.real((lbound+herm(lbound))/2.0)
        if cmps._dtype==complex:
            lbound=(lbound+herm(lbound))/2.0

        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=mf.computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)

        return lb,rb,lbound,rbound




def getBoundaryHamsCMPS(cmps,mpo):
    #regauge the cmps:
    pos=cmps._position
    NUC=cmps._N
    D=cmps._D
    assert((cmps._position==0)|(cmps._position==cmps._N))
    if pos==NUC:
        cmps.__position__(0,cqr=True) 

        mps=cmps.__toMPS__(connect='left')
        #eta,lss,numeig=mf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')
        eta,lss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(lbound)
        lbound=lbound/Z

    
        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)

        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        
        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')
        #eta,rss,numeig=mf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),nmax=10000,tolerance=1E-16,which='LR')
        eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')        
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(rbound)
        rbound=rbound/Z
        
        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)
    
        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if pos==0:

        cmps.__positionSVD__(NUC,'u')
        mps=cmps.__toMPS__(connect='right')
        #eta,rss,numeig=mf.UnitcellTMeigs(mps,direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D))/(D*1.0),nmax=10000,tolerance=1E-16,which='LR')
        eta,rss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=-1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        rbound=np.reshape(rss,(D,D))
        rbound=mf.fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(rbound)
        rbound=rbound/Z
        
        rdens=mf.computeDensity(rbound,mps,direction=-1,dtype=cmps._dtype)
        ldens=mf.computeDensity(np.eye(D),mps,direction=1,dtype=cmps._dtype)

        f0l=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0l=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000,dtype=cmps._dtype)
        lb=np.copy(f0l)
        
        cmps.__positionSVD__(0,'v') 
        
        
        mps=cmps.__toMPS__(connect='left')
        #eta,lss,numeig=mf.UnitcellTMeigs(mps,direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')
        eta,lss,numeig=UCTMeigs(cmps,grid=range(cmps._N),direction=1,numeig=1,init=np.reshape(np.eye(D),(D*D)),datatype=cmps._dtype,nmax=10000,tolerance=1e-10,ncv=100,which='LR')
        
        lbound=np.reshape(lss,(D,D))
        lbound=mf.fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(lbound)
        lbound=lbound/Z
    
        ldens=mf.computeDensity(lbound,mps,direction=1,dtype=cmps._dtype)
        rdens=mf.computeDensity(np.eye(D)/(1.0),mps,direction=-1,dtype=cmps._dtype)
            
        f0r=np.zeros((D*D),cmps._dtype)#+np.random.rand(D*D)*1j
        f0r=computeUCsteadyStateHamiltonianGMRES(mps,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000,dtype=cmps._dtype)


        lb=np.copy(f0l)
        rb=np.copy(f0r)

        return lb,rb,lbound,rbound


#calculates the left block hamiltonians using numerical derivatives, on a grid "grid", lb is initial cxondition at left end; grid is a BOND grid, not a site grid
def getLfromEv(cmps,mpo,lb,grid):
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
def getRfromEv(cmps,mpo,rb,grid):
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




#if direction>0: evolve f from ind1 to ind2 
#if direction<0: evolve f from ind2 to ind1
#this function doesn't care about a grid, or wether the step size is reasonable. It 
#just does what you ask it for
#f is a bond-like matrix that lives in between the cmps tensors. 
def evolveFLiebLiniger(cmps,f,mu,mass,g,ind1,ind2,direction):
    assert(ind1<ind2)
    if direction>0:
        assert(ind1<cmps._N)
        return f+(cmps._xb[ind2]-cmps._xb[ind1])*dfdxLiebLiniger(cmps,f,mu,mass,g,direction,ind1)

    if direction<0:
        assert(ind2>0)
        return f+(cmps._xb[ind2]-cmps._xb[ind1])*dfdxLiebLiniger(cmps,f,mu,mass,g,direction,ind2)


#the routine evolves f living on bond n1 (surrounded by tensors Q[n1-1],R[n1-1] and Q[n1], R[n1]) 
#to bond n2 using the intermediate points "m" in "grid", and tensors  Q[m-1],R[m-1] and Q[m], R[m].
#this routine can be used instead of addLayer to get the L[:,:,0] or R[:,:,-1] expression
#of the Block hamiltonians. To get L[m+n] from L[m], n1=m+1 and n2=n+1 (for my conventions in the DMRG code).
#This is because i store L[m] such that it has tensor A[m] contracted into it, and hence L[m][:,:,0] lives 
#on bond m+1.the resulting f lives on n2; it reduces to regular addLayer function if grid hs stride 1 (i.e. contains all fine lattice points).
#similar for R: R[m] has tensor A[m] contracted, so R[m][:,:,-1] lives on bond m. to get R[n<m], use n2=m, n1=n.
#If grid has stride 1, it again reduces to addLayer.
def gridevolveFLiebLiniger(cmps,f,mu,mass,inter,grid,n1,n2,direction):
    #find the index in grid at which grid[ind1]=n1

    l1=np.nonzero(np.array(grid)==n1)
    if not l1:
        print ('in addcmpsLayerLiebLiniger: could not find index "n1"={0} in "grid"!'.format(n1))
        return
    ind1=l1[0][0]

    #find the index in grid at which grid[ind2]=n2
    l2=np.nonzero(np.array(grid)==n2)
    if not l2:
        print ('in addcmpsLayerLiebLiniger: could not find index "n2"={0} in "grid"!'.format(n2))
        return
    ind2=l2[0][0]

    if direction>0:
        tgrid=grid[ind1:ind2+1]
        for n in range(len(tgrid)-1):
            #print ('going from {0} to {1}, using Q[{0}],R[{0}]'.format(tgrid[n],tgrid[n+1]))
            f=evolveFLiebLiniger(cmps,f,mu,mass,inter,tgrid[n],tgrid[n+1],direction)
        return f
    
    if direction<0:
        tgrid=grid[ind1:ind2+1]
        for n in range(len(tgrid)-1,0,-1):
            f=evolveFLiebLiniger(cmps,f,mu,mass,inter,tgrid[n-1],tgrid[n],direction)
        return f


#pass "f" as a matrix
#note that the index "0" of the density-like matrix lives on the UPPER bond (hence the multiple transpose operations below)
#"cmps" is an object of the CMPS class, defined in cmpsdisclib or cmpslib.
#"f" is a bond-like matrix living between matrices Q and R. 
#"mu" is a vector of length cmps._N that contains the values of the chemical potential. The routine assumes unit-cell translational invariance, that is mu[cmps._N]=mu[0].
#"mass" and "g" are mass and interaction strenght of the bosons
#"direction" can be >0 or <0. use direction>0 when going from left to right, and direction<0 if going from right to left (i.e. when calculating left or right block 
#hamiltonians).
#"f" is living on bond "ind", and the routine gives back the derivative of "f" calculated from adjacent matrices Q[ind-1], R[ind-1] and Q[ind], R[ind].
def dfdxLiebLiniger(cmps,f,mu,mass,g,direction,ind):
    assert(ind<cmps._N)
    assert(ind>0)
    if direction>0:
        assert(cmps._position>ind)
        if (ind>0) and (ind<cmps._N):
            K=cmps._Q[ind-1].dot(cmps._R[ind])-cmps._R[ind-1].dot(cmps._Q[ind])+(cmps._R[ind]-cmps._R[ind-1])/(cmps._xm[ind]-cmps._xm[ind-1])
            I=cmps._R[ind-1].dot(cmps._R[ind])
            return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,f)\
                +1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
                g*np.transpose(herm(I).dot(I))+\
                mu[ind-1]/2.0*np.transpose(herm(np.eye(cmps._D)+cmps._dx[ind]*cmps._Q[ind]).dot(herm(cmps._R[ind-1])).dot(cmps._R[ind-1]).dot(np.eye(cmps._D)+cmps._dx[ind]*cmps._Q[ind]))+\
                mu[ind-1]/2.0*np.transpose(herm(np.sqrt(cmps._dx[ind])*cmps._R[ind]).dot(herm(cmps._R[ind-1])).dot(cmps._R[ind-1]).dot(np.sqrt(cmps._dx[ind])*cmps._R[ind]))+\
                mu[ind]/2.0*np.transpose(herm(cmps._R[ind]).dot(cmps._R[ind]))
            #mu[ind]*np.transpose(herm(cmps._R[ind]).dot(cmps._R[ind]))
        #if ind==0:
        #    K=Q.dot(cmps._R[ind])-R.dot(cmps._Q[ind])+(cmps._R[ind]-R)/(cmps._xm[ind]-xm)
        #    I=R.dot(cmps._R[ind])
        #    return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,f)\
        #        +1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
        #        mu[ind]*np.transpose(herm(cmps._R[ind]).dot(cmps._R[ind]))+\
        #        g*np.transpose(herm(I).dot(I))
        #
        #if ind==(cmps._N):
        #    K=cmps._Q[ind-1].dot(R)-cmps._R[ind-1].dot(Q)+(R-cmps._R[ind-1])/(xm-cmps._xm[ind-1])
        #    I=cmps._R[ind-1].dot(R)
        #    return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,f)\
        #        +1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
        #        mu[0]*np.transpose(herm(R).dot(R))+\
       #        g*np.transpose(herm(I).dot(I))

    if direction<0:
        assert(cmps._position<ind)
        if (ind>0) and (ind<cmps._N):
            K=cmps._Q[ind-1].dot(cmps._R[ind])-cmps._R[ind-1].dot(cmps._Q[ind])+(cmps._R[ind]-cmps._R[ind-1])/(cmps._xm[ind]-cmps._xm[ind-1])
            I=cmps._R[ind-1].dot(cmps._R[ind])
            return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,f)+1.0/(2.0*mass)*K.dot(herm(K))\
                +mu[ind]/2.0*(np.eye(cmps._D)+cmps._dx[ind-1]*cmps._Q[ind-1]).dot(cmps._R[ind]).dot(herm(cmps._R[ind])).dot(herm(np.eye(cmps._D)+cmps._dx[ind-1]*cmps._Q[ind-1]))+\
                +mu[ind]/2.0*(np.sqrt(cmps._dx[ind-1])*cmps._R[ind-1]).dot(cmps._R[ind]).dot(herm(cmps._R[ind])).dot(np.sqrt(cmps._dx[ind-1])*herm(cmps._R[ind-1]))+\
                mu[ind-1]/2.0*cmps._R[ind-1].dot(herm(cmps._R[ind-1]))+g*I.dot(herm(I))


        #if ind==(cmps._N):
        #    K=cmps._Q[ind-1].dot(R)-cmps._R[ind-1].dot(Q)+(R-cmps._R[ind-1])/(xm-cmps._xm[ind-1])
        #    I=cmps._R[ind-1].dot(R)
        #    return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,vec)+1.0/(2.0*mass)*K.dot(herm(K))\
        #        +mu[ind-1]*cmps._R[ind-1].dot(herm(cmps._R[ind-1]))+g*I.dot(herm(I))
        #
        #
        #if (ind==0):
        #    K=Q.dot(cmps._R[ind])-R.dot(cmps._Q[ind])+(cmps._R[ind]-R)/(cmps._xm[ind]-xm)
        #    I=R.dot(cmps._R[ind])
        #    return matrixtransferOperator2ndOrder(cmps._Q[ind],cmps._R[ind],cmps._dx[ind],0.0,direction,vec)+1.0/(2.0*mass)*K.dot(herm(K))\
        #        +mu[-1]*R.dot(herm(R))+g*I.dot(herm(I))



#pass "f" as a matrix
#note that the index "0" of the density-like matrix lives on the UPPER bond (hence the multiple transpose operations below)
#"cmps" is an object of the CMPS class, defined in cmpsdisclib or cmpslib.
#"f" is a bond-like matrix living between matrices Q and R. 
#"mu" is a vector of length cmps._N that contains the values of the chemical potential. The routine assumes unit-cell translational invariance, that is mu[cmps._N]=mu[0].
#"mass" and "g" are mass and interaction strenght of the bosons
#"direction" can be >0 or <0. use direction>0 when going from left to right, and direction<0 if going from right to left (i.e. when calculating left or right block 
#hamiltonians).
#"f" is living on bond "ind", and the routine gives back the derivative of "f" calculated from adjacent matrices Q[ind-1], R1 and Q[ind], R2.




#dens0 is in matrix form;
#start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
def UCtransferOperatorcQR(cmps,grid,direction,vec):
    #the boundary has to treated with special care
    assert((cmps._position==0)|(cmps._position==cmps._N))
    assert(grid[0]==0)
    assert(grid[-1]==(cmps._N-1))
    if direction>0:
        #assert(cmps._position==0)
        #start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
        #the last step is evolving dens[grid[-1]] to dens[grid[-1]+1]; the returned dens is then living on 
        #the equivalent bond as the initial densitymatrix, just evolved by a single unit cell
        if(cmps._position==0):
            connector=cmps._connector.dot(cmps._U).dot(cmps._mats[0])
            densvec=np.reshape(np.transpose(connector).dot(np.reshape(vec,(cmps._D,cmps._D))).dot(np.conj(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                densvec=evolveDensityMatrixcQR(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrixcQR(densvec,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._V).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(np.conj(cmps._V))
            return np.reshape(dens,(cmps._D*cmps._D))
        if(cmps._position==cmps._N):
            connector=cmps._U
            densvec=np.reshape(np.transpose(connector).dot(np.reshape(vec,(cmps._D,cmps._D))).dot(np.conj(connector)),(cmps._D*cmps._D))

            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                densvec=evolveDensityMatrixcQR(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrixcQR(densvec,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(np.conj(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)))
            return np.reshape(dens,(cmps._D*cmps._D))

    if direction<0:
        if cmps._position==cmps._N:
            connector=cmps._mats[-1].dot(cmps._V).dot(cmps._connector)
            densvec=np.reshape(connector.dot(np.reshape(vec,(cmps._D,cmps._D))).dot(herm(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                densvec=evolveDensityMatrixcQR(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrixcQR(densvec,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._U.dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(herm(cmps._U))
            #now absorb the connector and the boundary unitaries into dens:
            return np.reshape(dens,(cmps._D*cmps._D))

        if cmps._position==0:
            connector=cmps._V
            densvec=np.reshape(connector.dot(np.reshape(vec,(cmps._D,cmps._D))).dot(herm(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                densvec=evolveDensityMatrixcQR(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrixcQR(densvec,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._connector.dot(cmps._U).dot(cmps._mats[0]).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(herm(cmps._connector.dot(cmps._U).dot(cmps._mats[0])))
            #now absorb the connector and the boundary unitaries into dens:
            return np.reshape(dens,(cmps._D*cmps._D))





def UCTMeigscQR(cmps,grid,direction,numeig,init,datatype=float,nmax=10000,tolerance=1e-8,ncv=10,which='LR'):
    D=cmps._D
    mv=fct.partial(UCtransferOperatorcQR,*[cmps,grid,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)
    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0} and SM'.format(numeig))
        print (eta)
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))

    return eta[m],np.reshape(vec[:,m],D*D),numeig





def UCTMeigs(cmps,grid,direction,numeig,init=None,datatype=float,nmax=10000,tolerance=1e-8,ncv=10,which='LR'):
    D=cmps._D
    mv=fct.partial(UCtransferOperator,*[cmps,grid,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)
    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0} and SM'.format(numeig))
        print (eta)
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))

    return eta[m],np.reshape(vec[:,m],D*D),numeig


#dens0 is in matrix form;
#start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
def UCtransferOperator(cmps,grid,direction,vec):
    #the boundary has to treated with special care
    assert((cmps._position==0)|(cmps._position==cmps._N))
    assert(grid[0]==0)
    assert(grid[-1]==(cmps._N-1))
    if direction>0:
        #assert(cmps._position==0)
        #start at the left boundary, and evolve dens to the right boundary. make sure that grid[0]==0, and grid[-1]==cmps._N-1
        #the last step is evolving dens[grid[-1]] to dens[grid[-1]+1]; the returned dens is then living on 
        #the equivalent bond as the initial densitymatrix, just evolved by a single unit cell
        if(cmps._position==0):
            connector=cmps._connector.dot(cmps._U).dot(cmps._mats[0])
            densvec=np.reshape(np.transpose(connector).dot(np.reshape(vec,(cmps._D,cmps._D))).dot(np.conj(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                densvec=evolveDensityMatrix(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrix(densvec,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._V).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(np.conj(cmps._V))
            return np.reshape(dens,(cmps._D*cmps._D))
        if(cmps._position==cmps._N):
            connector=cmps._U
            densvec=np.reshape(np.transpose(connector).dot(np.reshape(vec,(cmps._D,cmps._D))).dot(np.conj(connector)),(cmps._D*cmps._D))

            for n in range(len(grid)-1):
                deltax=cmps._xm[grid[n+1]]-cmps._xm[grid[n]]
                densvec=evolveDensityMatrix(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrix(densvec,cmps._Q[-1],cmps._R[-1],cmps._dx[-1],cmps._dx[-1],direction)
            #now absorb the connector and the boundary unitaries into dens:
            dens=np.transpose(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(np.conj(cmps._mats[-1].dot(cmps._V).dot(cmps._connector)))
            return np.reshape(dens,(cmps._D*cmps._D))

    if direction<0:
        if cmps._position==cmps._N:
            connector=cmps._mats[-1].dot(cmps._V).dot(cmps._connector)
            densvec=np.reshape(connector.dot(np.reshape(vec,(cmps._D,cmps._D))).dot(herm(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                densvec=evolveDensityMatrix(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrix(densvec,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._U.dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(herm(cmps._U))
            #now absorb the connector and the boundary unitaries into dens:
            return np.reshape(dens,(cmps._D*cmps._D))

        if cmps._position==0:
            connector=cmps._V
            densvec=np.reshape(connector.dot(np.reshape(vec,(cmps._D,cmps._D))).dot(herm(connector)),(cmps._D*cmps._D))
            for n in range(len(grid)-1,0,-1):
                deltax=-cmps._xm[grid[n-1]]+cmps._xm[grid[n]]
                densvec=evolveDensityMatrix(densvec,cmps._Q[grid[n]],cmps._R[grid[n]],deltax,cmps._dx[grid[n]],direction)
            densvec=evolveDensityMatrix(densvec,cmps._Q[0],cmps._R[0],cmps._dx[0],cmps._dx[0],direction)
            dens=cmps._connector.dot(cmps._U).dot(cmps._mats[0]).dot(np.reshape(densvec,(cmps._D,cmps._D))).dot(herm(cmps._connector.dot(cmps._U).dot(cmps._mats[0])))
            #now absorb the connector and the boundary unitaries into dens:
            return np.reshape(dens,(cmps._D*cmps._D))




#calculates the <C*(x) C(y)> correlation function
#takes a vector, returns a vector
def LiebLinigerCdagC(cmps,n1,n2):
    cmps.__position__(0)
    cmps.__position__(cmps._N)
    cmps.__regauge__(True)
    initial=None
    nmaxit=100000
    tol=1E-10
    ncv=100
    D=cmps._D
    boundarymatrix=cmps._mats[cmps._N].dot(cmps.__connection__(reset_unitaries=False))
    mps=cmps.__topureMPS__()
    mps[cmps._N-1]=np.transpose(np.tensordot(mps[cmps._N-1],boundarymatrix,([1],[0])),(0,2,1))

    [eta,vl,numeig]=mf.UnitcellTMeigs(mps,direction=1,numeig=4,init=initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
    l=np.reshape(vl,(D,D))
    l=l/np.trace(l)

    [eta,vr,numeig]=mf.UnitcellTMeigs(mps,direction=-1,numeig=4,init=initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
    r=np.reshape(vr,(D,D))
    r=r/np.trace(r)

    rdens=mf.computeDensity(r,mps,direction=-1,dtype=cmps._dtype)
    ldens=mf.computeDensity(l,mps,direction=1,dtype=cmps._dtype)
    corr=[]
    pos=[]
    #evolve l to n1
    x=np.reshape(herm(cmps._R[n1]).dot(ldens[n1]),D*D)
    n1p1=(n1+1)%cmps._N
    Z=np.trace(ldens[n1].dot(rdens[n1p1]))
    corr.append(np.trace(np.reshape(x,(D,D)).dot(cmps._R[n1]).dot(rdens[n1p1]))/Z)
    pos.append(cmps._xm[n1]-cmps._xm[n1])
    for n in range(n1+1,n2+1):
        n_=n%cmps._N
        x=TransferOperator(1,mps[n_],x)
        np1=(n+1)%cmps._N
        Z=np.trace(ldens[n_].dot(rdens[n1p1]))
        corr.append(np.trace(np.reshape(x,(D,D)).dot(cmps._R[n_]).dot(rdens[np1]))/Z)
        pos.append(cmps._xm[n_]+(n-n%cmps._N)/cmps._N*cmps._L-cmps._xm[n1])

    return pos,corr


#left gauge for gauge>0
#right gauge for gauge<0
#x is the current grid
#xdense is the new grid
def matrixinterp(mats,x,xdense,k):
    assert(x[0]<=xdense[0])
    assert(x[-1]>=xdense[-1])
    assert(len(x)==len(mats))
    N=len(x)
    D1,D2=mats[0].shape
    temp=np.zeros((D1,D2,N)).astype(mats[0].dtype)
    tempdense=np.zeros((D1,D2,len(xdense))).astype(mats[0].dtype)

    for n in range(N):
        temp[:,:,n]=np.copy(mats[n])

    for n1 in range(0,D1):
        for n2 in range(0,D2):
            splreal=splrep(x,np.real(temp[n1,n2,:]),k=k)
            splimag=splrep(x,np.imag(temp[n1,n2,:]),k=k)
            tempdense[n1,n2,:] = splev(xdense,splreal)+1j*splev(xdense,splimag)
    new=[]
    for n in range(len(xdense)):
        new.append(tempdense[:,:,n])
    return new


#left gauge for gauge>0
#right gauge for gauge<0
#x is the current grid
#xdense is the new grid
def splineinterp(Q,R,x,xdense,k):
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


#takes a cmps defined on a grid cmps._N and extends it to another grid with gridpoints xbdense
#using an order="order" spline interpolation; the iterpolation is done by patching 3 cmps-states together,
#interpolating the central one, and returning a denser cmps. The patching reduces boundary artifacts due to interpolation.
def cmpsinterpolation(cmps,xbdense,order):
    assert(cmps._position==0)
    #xbdense=np.linspace(0,cmps._L,Ndense+1)
    cmpsl=copy.deepcopy(cmps)
    cmpsc=copy.deepcopy(cmps)
    cmpsr=copy.deepcopy(cmps)
    cmpsd=DiscreteCMPS('homogeneous','blabla',cmps._D,cmps._L,xbdense,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    

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

    Rd,Qd=splineinterp(Qs,Rs,xm,cmpsd._xm,k=order)
    for n in range(len(cmpsd._xm)):
        cmpsd._Q[n]=np.copy(Qd[n])
        cmpsd._R[n]=np.copy(Rd[n])

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
        cmpspatched=DiscreteCMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
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

        cmpspatched=DiscreteCMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
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
    cmpspatched=DiscreteCMPS('homogeneous','bla',cmps._D,cmps._L,cmps._xb,cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
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




def copycmps(cmps):
    cmpsout=dcmps.DiscreteCMPS('homogeneous','bla',D=cmps._D,L=cmps._L,x_bonds=cmps._xb,dtype=cmps._dtype,scaling=0.8,epstrunc=1E-16,obc=cmps._obc)    
    for s in range(cmps._N):
        cmpsout._Q[s]=np.copy(cmps._Q[s])
        cmpsout._R[s]=np.copy(cmps._R[s])
    for s in range(cmps._N+1):
        cmpsout._mats[s]=np.copy(cmps._mats[s])
    cmpsout._position=cmps._position
    cmpsout._U=np.copy(cmps._U)
    cmpsout._V=np.copy(cmps._V)

    cmpsout._connector=np.copy(cmps._connector)

    return cmpsout



#This is a regular MPS class for infinite systems with a finite unit-cell.
#Instead of storing the mps tensors, it stores the cMPS content of it.
#The class only works for bosonic single species models. 
#which = 'homogeneous' or 'random': determines initialization of the MPS
#gauge = 'left' or 'right': determines which gauge the initial MPS should have
#D: bond dimension
#L: unit-cell length
#x_bonds: position of the bonds of the MPS
#dtype =float or complex; the data type of the matrices
#self._xb, self._xm: store the locations of the bonds and tensors, respectively. 
#dx: the lattice spacing.
#scaling: initial scaling of the MPS tensors
#epstrunc: truncation of the MPS
#obc= True or False: boundary condition of the MPS (code might break for obc=True)

class DiscreteCMPS:
    def __copy__(self):
        cmps=DiscreteCMPS('homogeneous','bla',self._D,self._L,self._xb,self._dtype,scaling=0.8,epstrunc=self._eps,obc=self._obc)
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
        #cmps._invmats=[]
        #cmps._lams=[]
        #for n in range(self._N):
        #    cmps._invmats.append(np.copy(self._invmats[n]))
        #    cmps._lams.append(np.copy(self._lams[n]))
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
            elif dtype==complex:
                m1=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling
                m2=(np.random.rand(D,D)-0.5+(np.random.rand(D,D)-0.5)*1j)*scaling
            else:
                sys.exit('DiscreteCMPS.__init__(): unrecognized type {0}'.format(dtype))
            if gauge=='left':
                for n in range(self._N):
                    self._Q.append(m1-herm(m1)-0.5*herm(m2).dot(m2))
                    self._R.append(m2)
                self._mats=[]
                for n in range(self._N+1):
                    self._mats.append(None)
                self._position=self._N
                self._mats[-1]=np.eye(self._D)
                self._connector=np.eye(self._D)
                self._U=np.eye(self._D)
                self._V=np.eye(self._D)
                self.__regauge__(True)

            if gauge=='right':
                for n in range(self._N):
                    self._Q.append(m1-herm(m1)-0.5*m2.dot(herm(m2)))
                    self._R.append(m2)
                self._mats=[]
                for n in range(self._N+1):
                    self._mats.append(None)
                self._position=0
                self._mats[0]=np.eye(self._D)
                self._U=np.eye(self._D)
                self._V=np.eye(self._D)
                self._connector=np.eye(self._D)
                self.__regauge__(True)


            if (gauge!='left')&(gauge!='right'):
                for n in range(self._N):
                    self._Q.append(m1)
                    self._R.append(m2)
                self._position=self._N

                self._mats=[]
                self._U=np.eye(self._D)
                self._V=np.eye(self._D)
                self._connector=np.eye(self._D)
                for n in range(self._N+1):
                    self._mats.append(None)
                self._mats[self._position]=np.eye(self._D)


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
            self._mats=[]
            for n in range(self._N+1):
                self._mats.append(None)
            self._position=0
            self._mats[0]=np.eye(self._D)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            self._connector=np.eye(self._D)
            self.__regauge__(True)
        

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


    #this is a position-shifting function which does not perform any checks. It's just a regular MPS-type 
    #routine
    def __positionMPS__(self,position):
        if self._obc==True:
            if position==0:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
                #raw_input('do you really want to continue?')
            if position==self._N:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
                #raw_input('do you really want to continue?')

        if position>self._N:
            return
        if position<0:
            return
        if position==self._position:
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
    def __position__(self,position,method=0,pushunitaries=False,cqr=False):
        if self._obc==True:
            if position==0:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
                #raw_input('do you really want to continue?')
            if position==self._N:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
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
                if cqr==False:
                    self._Q[site],self._R[site],self._mats[site+1]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site],self._dx[site],direction=1,method=method)
                if cqr==True:
                    self._Q[site],self._R[site],dmdx=prepareCMPSQR(self._Q[site],self._R[site],self._mats[site],self._dx[site],direction=1)
                    mat=self._mats[site]+self._dx[site]*dmdx.dot(self._mats[site])
                    Z=np.trace(mat.dot(herm(mat)))
                    self._mats[site+1]=np.copy(mat)/np.sqrt(Z)

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
                if cqr==False:
                    self._Q[site],self._R[site],self._mats[site]=prepareCMPSTensor(self._Q[site],self._R[site],self._mats[site+1],self._dx[site],direction=-1,method=method)
                if cqr==True:
                    self._Q[site],self._R[site],dmdx=prepareCMPSQR(self._Q[site],self._R[site],self._mats[site+1],self._dx[site],direction=-1)
                    mat=self._mats[site+1]+self._dx[site]*self._mats[site+1].dot(dmdx)
                    Z=np.trace(mat.dot(herm(mat)))
                    self._mats[site]=np.copy(mat)/np.sqrt(Z)

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

    def __positionSVD__(self,position,fix,mode=None,verbose=False):
        if self._obc==True:
            if position==0:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
                #raw_input('do you really want to continue?')
            if position==self._N:
                print ('warning: you are setting position of an obc cmps to ',position,'!')
                #raw_input('do you really want to continue?')

        if position>self._N:
            #print ('CMPS.__position__(position): position index cannot be larger than N=',self._N)
            return
        if position<0:
            #print ('CMPS.__position__(position): position index cannot be negative!')
            return
        if position==self._position:
            return

        if self._position<position:
            for site in range(self._position,position):
                if verbose==True:
                    print ('CMPS.__positionSVD__() at site {0}'.format(site))
                self.__prepareTensor__(site,1,fix)

           
            self._position=position
            if mode=='diagonal':
                self.__distribute__(fix)



        if self._position>position:
            for site in range(self._position-1,position-1,-1):
                if verbose==True:
                    print ('CMPS.__positionSVD__() at site {0}'.format(site))
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
                
        if (site!=0)&(site!=(self._N-1)):
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



    def __QRtensor__(self,site):
        assert(site>=0)
        assert(site<self._N)
                
        tensor=np.zeros((self._D,self._D,2),dtype=self._dtype)
        tensor[:,:,0]=self._dx[site]*self._Q[site]
        tensor[:,:,1]=np.sqrt(self._dx[site])*self._R[site]
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


    def __save__(self,filename):
        np.save('Q_'+filename,self._Q)
        np.save('R_'+filename,self._R)
        np.save('mats_'+filename,self._mats)
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

    def __load__(self,filename):
        self._Q=np.load('Q_'+filename)
        self._R=np.load('R_'+filename)
        self._mats=np.load('mats_'+filename)
        self._connector=np.load('connector_'+filename)
        self._U=np.load('U_'+filename)
        self._V=np.load('V_'+filename)
        params=np.load('params_'+filename)
        self._xm=np.load('xm_'+filename)
        self._xb=np.load('xb_'+filename)
        self._dx=np.load('dx_'+filename)

        self._obc=params[0]
        self._L=float(params[1])
        self._N=int(params[2])
        self._eps=float(params[3])
        self._dtype=params[4]
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
    def __interpolate_inplace__(self,N,order=5,tol=1E-10,ncv=100):
        bpoints=order
        #N=len(xdense)-1
        xbdense=np.linspace(0,self._xb[-1]-self._xb[0],N+1)
        dxdense=np.zeros(N)
        xmdense=np.zeros(N)
        for n in range(N):
            dxdense[n]=xbdense[n+1]-xbdense[n]
            xmdense[n]=(xbdense[n+1]+xbdense[n])/2.0

        #xbdense=np.linspace(0,cmps._L,Ndense+1)
        self.__position__(self._N)
        self.__regauge__(True,tol=tol,ncv=ncv)

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
        Rd,Qd=splineinterp(Q,R,xint,xmdenseint[inds],k=order)

        mat=np.copy(self._mats[self._position])
        self._Q=[]
        self._R=[]
        self._mats=[]

        for n in range(N):
            self._Q.append(np.copy(Qd[n]))
            self._R.append(np.copy(Rd[n]))
            self._mats.append(None)
        self._mats.append(None)

        self._position=N
        self._N=N            
        self._dx=np.copy(dxdense)
        self._xm=np.copy(xmdense)
        self._xb=np.copy(xbdense)
        self._connector=np.eye(self._D)
        self._U=np.copy(self._U)
        self._V=np.copy(self._V)
        self._mats[self._N]=np.eye(self._D)
        self.__regauge__(True,tol=tol,ncv=ncv)

    #returns a new cmps on the grid xdense
    #assumes that the cmps is regauged to be periodic
    def __interpolate__(self,xbdense,order=5,tol=1E-10,ncv=100):
        bpoints=order
        Ndense=len(xbdense)-1
        L=xbdense[-1]-xbdense[0]
        #print
        #print
        #print
        #print (type(self._dtype))
        cmpsd=DiscreteCMPS('homogeneous','blabla',self._D,L,xbdense,self._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
        self.__position__(self._N)
        self.__regauge__(True,tol=tol,ncv=ncv)

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

        Rd,Qd=splineinterp(Q,R,xint,cmpsd._xm,k=order)
        for n in range(Ndense):
            cmpsd._Q[n]=np.copy(Qd[n])
            cmpsd._R[n]=np.copy(Rd[n])
        cmpsd._position=cmpsd._N
        cmpsd._connector=np.eye(self._D).astype(cmpsd._dtype)
        cmpsd._U=np.copy(self._U)
        cmpsd._V=np.copy(self._V)
        cmpsd._mats[-1]=np.eye(self._D).astype(cmpsd._dtype)

        cmpsd.__regauge__(True,tol=tol,ncv=ncv)

        return cmpsd
        

    def __grid_interpolate__(self,grid,order,tol=1E-10,ncv=100):
        self.__regauge__(True)
        bpoints=order
        Ndense=self._N
        cmpsd=DiscreteCMPS('homogeneous','blabla',self._D,self._L,self._xb,self._dtype,scaling=0.8,epstrunc=1E-16,obc=False)    
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
        Rd,Qd=splineinterp(Q,R,xint,xdense,k=order)

        for n in range(Ndense):
            self._Q[n]=np.copy(Qd[n+bpoints])
            self._R[n]=np.copy(Rd[n+bpoints])

        self.__regauge__(True)




    #does an svd of self._mats[self._position]=U*D*V, and distributes the unitary U to the left, and V to the right matrices
    #if self._mats[self._position] is diagonal (with a possible phase) it takes its phase and distributes it equally onto all 
    #matrices
    #if either U or V is a diagonal unitary, its phase is fixed to 1
    #if neither U nor V are diagonal, then "which" determines which of the two gets its phase fixed before they are distributed.
    #if at the left boundary, it pushes U into self._U; if at the right boundary, it pushes V into self._V
    def __distribute_2__(self):
        if self._position==self._N:
            #first make sure that the self._mats[self._N] is diagonal:
            diag=np.diag(self._mats[self._N])
            mat=self._mats[self._N]-np.diag(diag)
            if np.linalg.norm(mat)>1E-10:
                warnings.warn('cmpsdisclib.py: DiscreteCMPS.__distribute__(): cmps._mats[cmps._N] is not diagonal!',stacklevel=2)

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
                warnings.warn('cmpsdisclib.py: DiscreteCMPS.__distribute__(): cmps._mats[cmps._N] is not diagonal!',stacklevel=2)

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



    ############################       deprecated routine ####################################################

    #does an svd of self._mats[self._position]=U*D*V, and distributes the unitary U to the left, and V to the right matrices
    #if self._mats[self._position] is diagonal (with a possible phase) it takes its phase and distributes it equally onto all 
    #matrices
    #if either U or V is a diagonal unitary, its phase is fixed to 1
    #if neither U nor V are diagonal, then "which" determines which of the two gets its phase fixed before they are distributed.
    #if at the left boundary, it pushes U into self._U; if at the right boundary, it pushes V into self._V
    def __distribute__(self,which):
        #fir real data type, check if mats is diagonal
        if self._dtype==float:
            #if self._mats[self._position] is diagonal, it probably has a random sign; remove the sign
            #I don't know if this is fully kosher
            if (utils.isDiag(self._mats[self._position])):
                self._mats[self._position]=np.abs(self._mats[self._position])
            return

        if self._dtype==complex:
            U,l,V=np.linalg.svd(self._mats[self._position])
            #check if U and V are both diagonal, in which case l was diagonal before hand.
            #in that case, don't do anything
            if (utils.isDiagUnitary(U)[0]==True)&(utils.isDiagUnitary(V)[0]==True):
                diff=np.linalg.norm(U.dot(V)-np.eye(self._D))
                if diff<1E-8:
                    return 
                if diff>=1E-8:
                    Nl=self._position
                    Nr=self._N-self._position
                    unit=U.dot(V)
            
                    phase=np.angle(np.diag(unit))
                    U=np.diag(np.exp(1j*phase/self._N*Nl))
                    V=np.diag(np.exp(1j*phase/self._N*Nr))

            #if one of the two unitaries is diagonal, then fix its phase to be 1
            if (utils.isDiagUnitary(U)[0]==True)&(utils.isDiagUnitary(V)[0]==False):
                phase=np.angle(np.diag(U))
                unit=np.diag(np.exp(-1j*phase))
                U=U.dot(unit)
                V=herm(unit).dot(V)
            
            if (utils.isDiagUnitary(U)[0]==False)&(utils.isDiagUnitary(V)[0]==True):
                phase=np.angle(np.diag(V))
                unit=np.diag(np.exp(-1j*phase))
                U=U.dot(herm(unit))
                V=unit.dot(V)
            
            #if both U and V are not diagonal, then the ask the user which should be phase fixed
            if (utils.isDiagUnitary(U)[0]==False)&(utils.isDiagUnitary(V)[0]==False):
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
                self._V=V.dot(self._V)
            
            if self._position==0:
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
                    self._Q[n]=Vn.dot(self._Q[n]).dot(invVn)
                    self._R[n]=Vn.dot(self._R[n]).dot(invVn)
                    
                for n in range(self._position,self._N):
                    self._Q[n]=(self._Q[n]+HV+dx*HV.dot(self._Q[n]))
                    self._R[n]=(self._R[n]+dx*HV.dot(self._R[n]))
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

#    #boundarymat has to be on the left end!
#    #Q and R have to be left orthonormal!
    def __regauge__(self,canonize=False,initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=4,cqr=False,pinv=1E-12):
        #shift position to N, state doesn't connect back now
        #self.__position__(self._N)
        if self._position<(self._N-(self._N%2))/2:
            self.__position__(0,cqr=cqr)

        if self._position>=(self._N-(self._N%2))/2:
            self.__position__(self._N,cqr=cqr)

        if self._position==self._N:
            boundarymatrix=self._mats[self._N].dot(self.__connection__(reset_unitaries=True))
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the right boundarymatrix into mps2
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            #find the right eigenvector of the transfer operator
            initial=herm(self._mats[self._N]).dot(self._mats[self._N])            
            [eta,vr,numeig]=mf.UnitcellTMeigs(mps2,-1,numeig,initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.__save__('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))

            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            mps2=self.__topureMPS__()
            mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
            r=np.reshape(vr,(self._D,self._D))
            #fix phase of l and restore the proper normalization of l
            r=r/np.trace(r)
            if self._dtype==float:
                r=np.real(r+herm(r))/2.0
            if self._dtype==complex:
                r=(r+herm(r))/2.0

            eigvals,u=np.linalg.eigh(r)
            eigvals=np.abs(eigvals)
            eigvals/=np.sum(eigvals)
            eigvals[np.nonzero(eigvals<pinv)]=0.0
            r=u.dot(np.diag(eigvals)).dot(herm(u))
            
            inveigvals=np.zeros(len(eigvals))
            inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
            inveigvals[np.nonzero(eigvals<=pinv)]=0.0
            
            r=u.dot(np.diag(eigvals)).dot(herm(u))
            x=u.dot(np.diag(np.sqrt(eigvals)))
            invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))

            [eta,vl,numeig]=mf.UnitcellTMeigs(mps2,1,numeig,np.reshape(np.eye(self._D),self._D*self._D),nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.__save__('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print ('found large abs(eta)={0}'.format(np.abs(eta)))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
            l=np.reshape(vl,(self._D,self._D))
            l=l/np.trace(l)
            
            if self._dtype==float:
                l=np.real(l+herm(l))/2.0
            if self._dtype==complex:
                l=(l+herm(l))/2.0

            eigvals,u=np.linalg.eigh(l)
            eigvals=np.abs(eigvals)
            eigvals=eigvals/np.sum(eigvals)
            eigvals[np.nonzero(eigvals<pinv)]=0.0
            eigvals=eigvals/np.sum(eigvals)            
            l=u.dot(np.diag(eigvals)).dot(herm(u))
            
            inveigvals=np.zeros(len(eigvals))
            inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
            inveigvals[np.nonzero(eigvals<=pinv)]=0.0

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
            if cqr==False:
                self.__positionSVD__(0,'v') 

            if cqr==True:
                self.__position__(0,cqr=True) 
            #use SVD to shift position; there is still non-trivial matrix at the unitcell boundary
            temp=leftlam.dot(self._mats[0])
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[0]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.copy(np.diag(1./lam))
            self._U=np.eye(self._D).astype(self._dtype)
            self._V=np.eye(self._D).astype(self._dtype)
            #this removes a possible diagonal phase from u for tempmat=u l v
            #self.__distribute__('u')

            if canonize==True:
                if cqr==False:
                    self.__positionSVD__(self._N,'u')
                    self.__distribute__('v')
                    
                if cqr==True:
                    self.__position__(self._N,cqr=True)
                    self.__distribute__('v')

                #remove a possible phase from the left most bond matrix self._mats[-1]=u l v
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
            return

        if self._position==0:
            boundarymatrix=self.__connection__(reset_unitaries=True).dot(self._mats[0])
            mps=self.__topureMPS__()
            mps2=self.__topureMPS__()
            #multiply the left boundarymatrix into mps2
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            #find the right eigenvector of the transfer operator
            initial=self._mats[0].dot(self._mats[0])            
            [eta,vr,numeig]=mf.UnitcellTMeigs(mps2,-1,numeig,initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)

            if np.abs(eta)>10.0:
                self.__save__('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            mps2=self.__topureMPS__()
            mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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

            
            [eta,vl,numeig]=mf.UnitcellTMeigs(mps2,1,numeig,np.reshape(np.eye(self._D),self._D*self._D),nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.__save__('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print ('found large abs(eta)={0}'.format(np.abs(eta)))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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
            if cqr==False:
                self.__positionSVD__(self._N,'u')
            if cqr==True:
                self.__position__(self._N,cqr=True)

            temp=self._mats[-1].dot(rightlam)
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[-1]=temp/np.sqrt(Zl)

            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D).astype(self._dtype)
            self._V=np.eye(self._D).astype(self._dtype)

            if canonize==True:
                if cqr==False:
                    self.__positionSVD__(0,'v')
                    self.__distribute__('u')
                if cqr==True:
                    self.__position__(0,cqr=True)
                    self.__distribute__('u')

            return                

    #boundarymat has to be on the left end!
    #Q and R have to be left orthonormal!
    def __gridregauge__(self,grid,canonize=False,initial=None,nmaxit=100000,tol=1E-10,ncv=100,cqr=False,verbose=False):
        #shift position to N, state doesn't connect back now
        #self.__position__(self._N)
        if self._position<(self._N-(self._N%2))/2:
            self.__position__(0,cqr=cqr)

        if self._position>=(self._N-(self._N%2))/2:
            self.__position__(self._N,cqr=cqr)

        if self._position==self._N:
            boundarymatrix=self._mats[self._N].dot(self.__connection__(reset_unitaries=True))
            mps=self.__topureMPS__()
            #mps2=self.__topureMPS__()
            #multiply the right boundarymatrix into mps2
            #mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            #find the right eigenvector of the transfer operator
            if verbose==True:
                print ('in cmps.__gridregauge__(): calculating right unit cell eigenvector for self._position==self._N')
            [eta,vr,numeig]=UCTMeigs(self,grid,direction=-1,numeig=1,init=None,datatype=self._dtype,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
            if verbose==True:
                print ('done')
            #[eta,vr,numeig]=UnitcellTMeigs(mps2,-1,1,initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            #mps2=self.__topureMPS__()
            #mps2[self._N-1]=np.transpose(np.tensordot(mps2[self._N-1],boundarymatrix,([1],[0])),(0,2,1))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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
            if verbose==True:
                print ('in cmps.__gridregauge__(): calculating left unit cell eigenvector for self._position==self._N')
            [eta,vl,numeig]=UCTMeigs(self,grid,direction=1,numeig=1,init=np.reshape(np.eye(self._D),self._D*self._D),datatype=self._dtype,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
            if verbose==True:
                print ('done')

            #[eta,vl,numeig]=UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print ('found large abs(eta)={0}'.format(np.abs(eta)))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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
            if verbose==True:
                print ('in cmps.__gridregauge__(): normalizing the cmps by sweeping from self._N to 0')
            if cqr==False:
                self.__positionSVD__(0,'v',verbose=verbose) 
            if cqr==True:
                self.__position__(0,cqr=True)
            if verbose==True:
                print ('done')
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
                if verbose==True:
                    print ('in cmps.__gridregauge__(): canonizing the cmps by sweeping from 0 to self._N')

                if cqr==False:
                    self.__positionSVD__(self._N,'u',verbose=verbose)
                if cqr==True:
                    self.__position__(self._N,cqr=True) 
                #remove a possible phase from the left most bond matrix self._mats[-1]=u l v
                self.__distribute__('v')
                if verbose==True:
                    print ('done')

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
            return

        if self._position==0:
            boundarymatrix=self.__connection__(reset_unitaries=True).dot(self._mats[0])
            mps=self.__topureMPS__()
            #mps2=self.__topureMPS__()
            #multiply the left boundarymatrix into mps2
            #mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            #find the right eigenvector of the transfer operator
            if verbose==True:
                print ('in __gridregauge__(): calculating right  unit cell eigenvector for self._position=0')
            [eta,vr,numeig]=UCTMeigs(self,grid,direction=-1,numeig=1,init=None,datatype=self._dtype,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
            if verbose==True:
                print ('done')
            #[eta,vr,numeig]=UnitcellTMeigs(mps2,-1,1,initial,nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
            sqrteta=np.real(eta)**(1./(2.*self._N))
            for site in range(self._N):
                mps[site]=mps[site]/sqrteta
                self._Q[site],self._R[site]=fromMPSmat(mps[site],self._dx[site])
            
            #mps2=self.__topureMPS__()
            #mps2[0]=np.tensordot(boundarymatrix,mps2[0],([1],[0]))

            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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
            if verbose==True:
                print ('in __gridregauge__(): calculating left unit cell eigenvector for self._position=0')
            [eta,vl,numeig]=UCTMeigs(self,grid,direction=1,numeig=1,init=None,datatype=self._dtype,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
            if verbose==True:
                print ('done')
            #[eta,vl,numeig]=UnitcellTMeigs(mps2,1,1,np.reshape(np.eye(self._D),self._D*self._D),nmax=nmaxit,tolerance=tol,which='LM',ncv=ncv)
            if np.abs(eta)>10.0:
                self.save('selfGaugingSave_eta{0}'.format(np.abs(eta)))
                print ('found large abs(eta)={0}'.format(np.abs(eta)))
            if np.abs(np.imag(eta))>1E-10:
                print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
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
            if verbose==True:
                print ('in cmps.__gridregauge__(): normalizing the cmps by sweeping from 0 to self._N')

            if cqr==False:
                self.__positionSVD__(self._N,'u',verbose=verbose)
            if cqr==True:
                self.__position__(self._N,cqr=True)
            if verbose==True:
                print ('done')

            temp=self._mats[-1].dot(rightlam)
            Zl=np.trace(temp.dot(herm(temp)))
            self._mats[-1]=temp/np.sqrt(Zl)
            lam=lam/np.linalg.norm(lam)
            self._connector=np.diag(1./lam)
            self._U=np.eye(self._D)
            self._V=np.eye(self._D)
            if canonize==True:
                if verbose==True:
                    print ('in cmps.__gridregauge__(): canonizing the cmps by sweeping from self._N to 0')

                if cqr==False:
                    self.__positionSVD__(0,'v',verbose=verbose)
                if cqr==True:
                    self.__position__(0,cqr=True)
                self.__distribute__('u')

                if verbose==True:
                    print ('done')
            return                

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
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),direction=1,fixphase='q')
            if fixphase==False:
                r2=r.dot(A0)
                Qout,Rout=fromMPSmat(tensor,self._dx[site])
                return (r2-np.eye(self._D))/self._dx[site],Qout,Rout
            if fixphase==True:
                tempmat=r.dot(A0)
                Qt,Rt=fromMPSmat(tensor,self._dx[site])
                q,r2=qr(tempmat,'q')
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
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),direction=-1,fixphase='q')
            if fixphase==False:
                r2=A0.dot(r)
                Qout,Rout=fromMPSmat(tensor,self._dx[site])
                return (r2-np.eye(self._D))/self._dx[site],Qout,Rout
            if fixphase==True:
                tempmat=A0.dot(r)
                Qt,Rt=fromMPSmat(tensor,self._dx[site])
                q,r2=qr(herm(tempmat),'q')
                H=(herm(q)-np.eye(self._D))/self._dx[site]
                Qout=Qt+H+self._dx[site]*H.dot(Qt)
                Rout=Rt+self._dx[site]*H.dot(Rt)
                return (herm(r2)-np.eye(self._D))/self._dx[site],Qout,Rout


    #shifts the center site using numerical derivatives
    #routine uses continuous QR decomposition to obtain the derivative of the bond matrix.
    def __jump__(self,position):
        if self._position==position:
            return
        assert(position>=0)
        if self._position<position:
            #print ('jumping from site {0} to site {1}'.format(self._position,position)
            #====================    get the derivative of the bond matrix, using the cQR routine: ====================================
            self._Q[self._position],self._R[self._position],lderiv=prepareCMPSQR(self._Q[self._position],self._R[self._position],self._mats[self._position],self._dx[self._position],direction=1)
            #we are moving from left to right
            deltax=self._xb[position]-self._xb[self._position]
            #print deltax
            mat=self._mats[self._position]+deltax*lderiv.dot(self._mats[self._position])
            Z=np.trace(mat.dot(herm(mat)))
            self._mats[position]=np.copy(mat)/np.sqrt(Z)
            self._position=position

            
        if self._position>position:
            #print ('jumping from site {0} to site {1}'.format(self._position,position)
            self._Q[self._position-1],self._R[self._position-1],rderiv=prepareCMPSQR(self._Q[self._position-1],self._R[self._position-1],self._mats[self._position],self._dx[self._position-1],direction=-1)
            #we are moving from right to left
            deltax=self._xb[self._position]-self._xb[position]

            mat=self._mats[self._position]+deltax*self._mats[self._position].dot(rderiv)
            Z=np.trace(mat.dot(herm(mat)))
            self._mats[position]=np.copy(mat)/np.sqrt(Z)
            self._position=position



    #shifts center site moving points on "grid" from self._position to position; "grid" is a SITE grid, not a BOND grid;
    #if self._position or position are
    #not on "grid", it moves them to the point closest by. Only matrices on "grid" are modified by __shiftPosition__ (and 
    #those needed to shift self._position if it initially was not on "grid")
    #if the shift is going from left to right, then self._position stays to the left of the last site of "grid"
    #if the shift is going from right to left, then self._position stays to the right of the first site of "grid"
    def __shift__(self,position,grid):
        #now move the center site on "grid" to the new "position"
        if position==self._position:
            return 
        if position>self._position:
            #find the closest point on "grid" to self._position, and shift the position to that point
            if self._position>=grid[0]:
                closestleft,closestright=utils.bisection(grid,self._position)
                distl=self._position-grid[closestleft]
                distr=grid[closestright]-self._position

                oldposition=self._position
                if distl<distr:
                    self.__position__(grid[closestleft],cqr=True)
                    initial=closestleft
                if distl>=distr:
                    self.__position__(grid[closestright],cqr=True)
                    initial=closestright
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))

            if self._position<grid[0]:
                warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(self._position,grid[0]))
                self.__position__(grid[0],cqr=True)
                initial=0


            if position<=grid[-1]:
                closestleft,closestright=utils.bisection(grid,position)
                distl=position-grid[closestleft]
                distr=grid[closestright]-position
                if distl<distr:
                    newposition=grid[closestleft]
                    final=closestleft
                if distl>=distr:
                    newposition=grid[closestright]
                    final=closestright

                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                position=newposition
            if position>grid[-1]:
                warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,grid[-1]))
                position=grid[-1]
                final=len(grid)-1

            #now move from self._position to position using __jump__
            for index in range(initial+1,final+1):
                print ('at self._position={2}; jumping from {0} to {1}'.format(grid[index-1],grid[index],self._position))
                self.__jump__(grid[index])
        
        if position<self._position:
            if self._position<=grid[-1]:
                #find the closest point on "grid" to self._position
                closestleft,closestright=utils.bisection(grid,self._position)
                #print ('self._position={2} between {0} and {1}'.format(grid[closestleft],grid[closestright],self._position))
                distl=self._position-grid[closestleft]
                distr=grid[closestright]-self._position
                
                oldposition=self._position
                if distl<distr:
                    self.__position__(grid[closestleft],cqr=True)
                    initial=closestleft
                if distl>=distr:
                    self.__position__(grid[closestright],cqr=True)
                    initial=closestright
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))
                    print
            if self._position>grid[-1]:
                self.__position__(grid[-1],cqr=True)
                initial=len(grid)-1

            if position>=grid[0]:
                closestleft,closestright=utils.bisection(grid,position)
                distl=position-grid[closestleft]
                distr=grid[closestright]-position
                if distl<distr:
                    newposition=grid[closestleft]
                    final=closestleft
                if distl>=distr:
                    newposition=grid[closestright]
                    final=closestright
                
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                    print
                position=newposition

            if position<grid[0]:
                position=grid[0]
                final=0
        
            #now move from self._position to position using __jump__
            for index in range(initial-1,final-1,-1):
                print ('at self._position={2}; jumping from {0} to {1}'.format(grid[index+1]+1,grid[index]+1,self._position))
                self.__jump__(grid[index])


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
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))

            if self._position<grid[0]:
                warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(self._position,grid[0]))
                self.__position__(grid[0],fixphase,method_simple)
                initial=0

            #print ('self._position={0}'.format(self._position))
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

                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                position=newposition
            if position>grid[-1]:
                warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,grid[-1]))
                position=grid[-1]
                final=len(grid)-1

            #now move from self._position to position using __jump__
            for index in range(initial+1,final+1):
                print ('at self._position={2}; jumping from {0} to {1}'.format(grid[index-1],grid[index],self._position))
                self.__jump__(grid[index],fixphase,method_simple)
        
        if position<self._position:
            if self._position<=grid[-1]:
                #find the closest point on "grid" to self._position
                
                closestleft,closestright=bisection(grid,self._position)
                #print ('self._position={2} between {0} and {1}'.format(grid[closestleft],grid[closestright],self._position))
                distl=self._position-grid[closestleft]
                distr=grid[closestright]-self._position
                
                oldposition=self._position
                if distl<distr:
                    self.__position__(grid[closestleft]+1,fixphase,method_simple)
                    initial=closestleft
                if distl>=distr:
                    self.__position__(grid[closestright]+1,fixphase,method_simple)
                    initial=closestright
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): self._position={0} was not on "grid"; shifted it to self._position={1}'.format(oldposition,self._position))
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
                
                if (distl!=0)&(distr!=0):
                    warnings.warn('DiscreteCMPS.__shiftPosition__(self,position,grid): "position={0}" was not on "grid"; shifted it to "position"={1}'.format(position,newposition))
                    print
                position=newposition

            if position<grid[0]:
                position=grid[0]+1
                final=0
        
            #now move from self._position to position using __jump__
            for index in range(initial-1,final-1,-1):
                print ('at self._position={2}; jumping from {0} to {1}'.format(grid[index+1]+1,grid[index]+1,self._position))
                self.__jump__(grid[index]+1,fixphase,method_simple)



    #does an SVD of self._mats[self._position], and fixes the phase of either u or v, depending in "which"
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
            #print ('DiscreteCMPS.__diagonalize__(): at position={0}: found non-diagonal u; pushing it to self._U'.format(self._position))
            #print ('DiscreteCMPS.__diagonalize__(): at position={0}: pushing u to self._U'.format(self._position))
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
            #print ('DiscreteCMPS.__diagonalize__(): at position={0}: found non-diagonal v; pushing it to self._V'.format(self._position))
            #print ('DiscreteCMPS.__diagonalize__(): at position={0}: pushing v to self._V'.format(self._position))
            self._V=v.dot(self._V)
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
            #return np.tensordot(self.__tensor__(site),np.conj(self.__tensor__(site)),([0,2],[0,2]))
            return self._Q[site]+herm(self._Q[site])+herm(self._R[site]).dot(self._R[site])+self._dx[site]*herm(self._Q[site]).dot(self._Q[site])
        if direction<0:
            #return np.tensordot(self.__tensor__(site),np.conj(self.__tensor__(site)),([1,2],[1,2]))
            return self._Q[site]+herm(self._Q[site])+self._R[site].dot(herm(self._R[site]))+self._dx[site]*self._Q[site].dot(herm(self._Q[site]))

    def __prepareTensor__(self,site,direction,fix):

        if direction>0:
            Qout=np.zeros((self._D,self._D),self._dtype)
            A0=self._mats[site].dot(np.eye(self._D)+self._dx[site]*self._Q[site])
            Rout=self._mats[site].dot(self._R[site]).dot(np.linalg.pinv(A0))
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),direction=1,fixphase='q')
            tempmat=r.dot(A0)
            vd,l,ud=np.linalg.svd(herm(tempmat))
            u=herm(ud)
            v=herm(vd)

            l=l/np.linalg.norm(l)
            if fix=='u':
                if self._dtype==complex:
                    phase=np.angle(np.diag(u))
                    unit=np.diag(np.exp(-1j*phase))
                if self._dtype==float:
                    sign=np.sign(np.diag(u))
                    unit=np.diag(sign)

                u=u.dot(unit)
                v=herm(unit).dot(v)
            if fix=='v':
                warnings.warn('DiscreteCMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
                if self._dtype==complex:
                    phase=np.angle(np.diag(v))
                    unit=np.diag(np.exp(-1j*phase))
                
                if self._dtype==float:
                    sign=np.sign(np.diag(v))
                    unit=np.diag(sign)

                u=u.dot(herm(unit))
                v=unit.dot(v)

            tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            self._mats[site+1]=np.diag(l).dot(v)
            
        if direction<0:
            Qout=np.zeros((self._D,self._D),self._dtype)
            A0=(np.eye(self._D)+self._dx[site]*self._Q[site]).dot(self._mats[site+1])
            Rout=np.linalg.pinv(A0).dot(self._R[site]).dot(self._mats[site+1])
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,self._dx[site]),direction=-1,fixphase='q')
            
            tempmat=A0.dot(r)
            u,l,v=np.linalg.svd(tempmat)
            l=l/np.linalg.norm(l)
            #if mode=='accumulate':
            if fix=='u':
                warnings.warn('DiscreteCMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
                if self._dtype==complex:
                    phase=np.angle(np.diag(u))
                    unit=np.diag(np.exp(-1j*phase))

                if self._dtype==float:
                    sign=np.sign(np.diag(u))
                    unit=np.diag(sign)
                u=u.dot(unit)
                v=herm(unit).dot(v)

            if fix=='v':
                if self._dtype==complex:
                    phase=np.angle(np.diag(v))
                    unit=np.diag(np.exp(-1j*phase))

                
                if self._dtype==float:
                    sign=np.sign(np.diag(v))
                    unit=np.diag(sign)

                u=u.dot(herm(unit))
                v=unit.dot(v)

            tensor=np.tensordot(v,tensor,([1],[0]))
            self._Q[site],self._R[site]=fromMPSmat(tensor,self._dx[site])
            self._mats[site]=u.dot(np.diag(l))


#preares Q,R in left or right orthonormal form using cQR decomposition
#input: Q,R: cMPS tensors
#       mat: bond matrix on the left or right bond of the Q,R matrices
#       direction: left or right orthonormalite (>0, <0)
#       dx: normalization epsilon of Q,R (the fine grid)
def prepareCMPSQR(Q,R,mat,dx,direction,rcond=1E-14):
    if direction>0:
        Qtemp=mat.dot(Q).dot(np.linalg.pinv(mat,rcond=rcond))
        Rtemp=mat.dot(R).dot(np.linalg.pinv(mat,rcond=rcond))
        return cqr.cQR(Qtemp,Rtemp,dx)
    if direction<0:
        Qtemp=np.linalg.pinv(mat,rcond=rcond).dot(Q).dot(mat)
        Rtemp=np.linalg.pinv(mat,rcond=rcond).dot(R).dot(mat)
        Qt,Rt,UTt=cqr.cQR(herm(Qtemp),herm(Rtemp),dx)
        return herm(Qt),herm(Rt),herm(UTt)


def prepareCMPSTensorSVD(Q,R,mat,fix,dx,direction):
    D=np.shape(Q)[0]
    if direction>0:
        Qout=np.zeros((D,D)).astype(Q.dtype)
        A0=mat.dot(np.eye(D)+dx*Q)
        Rout=mat.dot(R).dot(np.linalg.pinv(A0))
        tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=1,fixphase='q')
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
            warnings.warn('prepareCMPSTensorSVD(): can produce jumps in Q and R! '.format(direction,fix))
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
        tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=-1,fixphase='q')
        tempmat=A0.dot(r)
        u,l,v=np.linalg.svd(tempmat)
        l=l/np.linalg.norm(l)
        if fix=='u':
            warnings.warn('prepareCMPSTensorSVD(): can produce jumps in Q and R! '.format(direction,fix))
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




def prepareCMPSTensor(Q,R,mat,dx,direction,method=0):
    D=np.shape(Q)[0]
    if direction>0:
        if method==0:
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=mat.dot(np.eye(D)+dx*Q)
            Rout=mat.dot(R).dot(np.linalg.pinv(A0))
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=1,fixphase='q')
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=r.dot(A0)
            matout=matout/np.sqrt(np.trace(matout.dot(herm(matout))))

        if method==1:
            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=np.eye(D)+dx*mat.dot(Q).dot(invmat)
            Rout=mat.dot(R).dot(invmat).dot(np.linalg.pinv(A0))
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=1,fixphase='q')
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
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=-1,fixphase='q')
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=A0.dot(r)
            matout=matout/np.sqrt(np.trace(matout.dot(herm(matout))))


        if method==1:

            invmat=np.linalg.pinv(mat)
            Qout=np.zeros((D,D)).astype(Q.dtype)
            A0=np.eye(D)+dx*invmat.dot(Q).dot(mat)
            Rout=np.linalg.pinv(A0).dot(invmat).dot(R).dot(mat)
            tensor,r=mf.prepareTensor(toMPSmat(Qout,Rout,dx),direction=-1,fixphase='q')
            tempmat=A0.dot(r)
            q1,r1=cmf.qr(herm(tempmat),'q')
            tensor=np.tensordot(herm(q1),tensor,([1],[0]))
            Qout,Rout=fromMPSmat(tensor,dx)
            matout=mat.dot(r1)
            

        return Qout,Rout,matout



