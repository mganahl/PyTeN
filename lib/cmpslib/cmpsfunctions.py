#!/usr/bin/env python
from sys import stdout
import numpy as np
import time,sys
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.interpolate import griddata
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigs
from scipy.optimize import fmin_cg
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.sparse.linalg import lgmres
import warnings


#import lib.mpslib.Hamiltonians as H
#import lib.mpslib.mpsfunctions as mf
#import lib.cmpslib.cQR as cqr
#import lib.cmpslib.discreteCMPS as dcmps
try:
    ncon=sys.modules['lib.ncon']
except KeyError:
    import lib.ncon as ncon

try:
    cqr=sys.modules['lib.cmpslib.cQR']
except KeyError:
    import lib.cmpslib.cQR as cqr
try:
    mf=sys.modules['lib.mpslib.mpsfunctions']
except KeyError:
    import lib.mpslib.mpsfunctions as mf
try:
    H=sys.modules['lib.mpslib.Hamiltonians']
except KeyError:
    import lib.mpslib.Hamiltonians as H
try:
    dcmps=sys.modules['lib.cmpslib.discreteCMPS']
except KeyError:
    import lib.cmpslib.discreteCMPS as dcmps


##TODO LIST:
"""
replace tensordot and tranpose approach with ncon
"""


comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


"""
embeds a bosonic cMPS
 """
def embed(Q,R,Dnew,diag=-10.0,default=0.0,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-12,thresh=1E-10):

    D=Q.shape[0]
    if Dnew<=D:
        warnings.warn('cmpsfunctions.py:embed(Q,R,Dnew): Dnew<=Q.shape[0]; cannot embed cMPS')
        return


    lam,Ql,Rl,Qr,Rr=regauge_old(Q,R,dx=0.0,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-12,thresh=1E-10)

    #embed:
    Dnew=2*D
    Q_=-np.eye(Dnew)*diag+(np.random.rand(Dnew,Dnew)-0.5+1j*(np.random.rand(Dnew,Dnew)-0.5))*default

    R_=(np.random.rand(Dnew,Dnew)-0.5+1j*(np.random.rand(Dnew,Dnew)-0.5))*np.sqrt(default)
    Q_[0:D,0:D]=np.copy(Q)
    R_[0:D,0:D]=np.copy(R)
    lam_,Ql_,Rl_,Qr_,Rr_=regauge_old(Q_,R_,dx=0.0,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-12,thresh=1E-10)

        
    #Qnew=-np.eye(Dnew).astype(Q.dtype)*np.abs(diag)+np.random.rand(Dnew,Dnew)*default
    #Qnew[0:D,0:D]=np.copy(Q)
    #Rnew=[]
    #for n in range(len(R)):
    #    temp=np.random.rand(Dnew,Dnew).astype(R.dtype)*default
    #    temp[0:D,0:D]=np.copy(R[n])
    #    Rnew.append(np.copy(temp))
    #    
    #    lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=regauge(Qnew,Rnew,dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=nmaxit,tol=tol,ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh)    
        
    return lam_,Ql_,Rl_,Qr_,Rr_

#this routine is needed for calculating the matrix elements of the polaron-Lieb-Liniger interaction
def calculateMatrixElement(V_n_k,W_n_k,V_np_kp,W_np_kp,Q1,R1,Q2,R2,k_a,k_ap,k_b,k_bp,ldens,rdens,\
                           minusT22_minus_kbminuskbp_inv_R2_r_R2bar,\
                           minusT11_minus_kaminuskap_inv_R1bar_l_R1,thresh=1E-12,tolerance=1E-10,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    if abs(k_a+k_b-k_ap-k_bp)>1E-8:
        return 0.0
    else:
        #TERM 1
        temp1=np.transpose(herm(W_np_kp).dot(ldens.transpose()).dot(W_n_k))
        T1=np.tensordot(temp1,minusT22_minus_kbminuskbp_inv_R2_r_R2bar,([0,1],[0,1]))\
        #TERM 2
        temp2=W_n_k.dot(rdens).dot(herm(W_np_kp))
        T2=np.tensordot(minusT11_minus_kaminuskap_inv_R1bar_l_R1,temp2,([0,1],[0,1]))
        #TERM 3
        T3=np.tensordot(l.transpose().dot(W_n_k).dot(r),np.conj(W_np_kp),([0,1],[0,1]))
        
        #TERM 4
        temporary=np.transpose(minusT11_minus_kaminuskap_inv_R1bar_l_R1.transpose().dot(V_n_k))+\
            np.transpose(herm(R1).dot(minusT11_minus_kaminuskap_inv_R1bar_l_R1.transpose()).dot(W_n_k))
        minusT21_plus_kap_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,\
                                                                             sigma=-1j*k_ap,direction=1,l=ldens,r=rdens,\
                                                                             thresh=thresh,x0=None,tolerance=tolerance,\
                                                                             maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        T4W=minusT21_plus_kap_inv_temporary.transpose().dot(R2).dot(rdens)
        T4V=minusT21_plus_kap_inv_temporary.transpose().dot(rdens)
        T4=np.tensordot(T4W,np.conj(W_np_kp),([0,1],[0,1]))+np.tensordot(T4V,np.conj(V_np_kp),([0,1],[0,1]))
        
        
        #TERM 5 
        temporary=V_n_k.dot(rdens)+W_n_k.dot(rdens).dot(herm(R2))
        minusT12_minus_ka_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,\
                                                                           sigma=1j*k_a,direction=-1,l=ldens,r=rdens,\
                                                                           thresh=thresh,x0=None,tolerance=tolerance,\
                                                                           maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        T5V=minusT11_minus_kaminuskap_inv_R1bar_l_R1.transpose().dot(minusT12_minus_ka_inv_temporary)
        T5W=minusT11_minus_kaminuskap_inv_R1bar_l_R1.transpose().dot(R1).dot(minusT12_minus_ka_inv_temporary)
        T5=np.tensordot(T5W,np.conj(W_np_kp),([0,1],[0,1]))+np.tensordot(T5V,np.conj(V_np_kp),([0,1],[0,1]))
        
        #Term 6
        temporary=np.transpose(herm(R1).dot(ldens.transpose()).dot(W_n_k))
        minusT21_plus_kap_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,\
                                                                             sigma=-1j*k_ap,direction=1,l=ldens,r=rdens,\
                                                                             thresh=thresh,x0=None,tolerance=tolerance,\
                                                                             maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
                                                                             
        T6W=minusT21_plus_kap_inv_temporary.transpose().dot(R2).dot(rdens)
        T6V=minusT21_plus_kap_inv_temporary.transpose().dot(rdens)
        T6=np.tensordot(T6W,np.conj(W_np_kp),([0,1],[0,1]))+np.tensordot(T6V,np.conj(V_np_kp),([0,1],[0,1]))  
        
        #Term 7
        temporary=V_n_k.dot(rdens)+W_n_k.dot(rdens).dot(herm(R2))
        minusT12_minus_ka_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,\
                                                                            sigma=1j*k_a,direction=-1,l=ldens,r=rdens,\
                                                                            thresh=thresh,x0=None,tolerance=tolerance,\
                                                                            maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        T7W=ldens.transpose().dot(R1).dot(minusT12_minus_ka_inv_temporary)
        T7=np.tensordot(T7W,np.conj(W_np_kp),([0,1],[0,1]))
        return T1+T2+T3+T4+T5+T6+T7


def PiPhiCorr(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=-1j/2.0*(np.reshape(np.transpose(Rl)-np.conj(Rl),D*D))
    rdens=np.tensordot(Rl,r,([1],[0]))+np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr

def PiPiCorr(Ql,Rl,r,dx,N,cutoff,initial=None):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    if initial==None:
        vec=-cutoff/2.0*(np.reshape(np.transpose(Rl)-np.conj(Rl),D*D))
    else:
        vec=initial
    rdens=np.tensordot(Rl,r,([1],[0]))-np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr,vec

def PiPiCorrMoronOrdered(Ql,Rl,r,dx,N,cutoff,psiinitial=None,psidaginitial=None):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    if psiinitial==None:
        vecpsi=-cutoff/2.0*(np.reshape(np.transpose(Rl),D*D))        
    else:
        vecpsi=psiinitial
    if psidaginitial==None:
        vecpsidag=-cutoff/2.0*(np.reshape(np.conj(Rl),D*D))
    else:
        vecpsidag=psidaginitial


    rdenspsi=np.tensordot(Rl,r,([1],[0]))
    rdenspsidag=np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vecpsi=vecpsi+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vecpsi)
        vecpsidag=vecpsidag+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vecpsidag)
        corr[n]=np.tensordot(np.reshape(vecpsi,(D,D)),rdenspsi,([0,1],[0,1]))+\
                 np.tensordot(np.reshape(vecpsidag,(D,D)),rdenspsidag,([0,1],[0,1]))-\
                 2.0*np.tensordot(np.reshape(vecpsidag,(D,D)),rdenspsi,([0,1],[0,1]))
    return corr,vecpsi,vecpsidag

def PhiPhiCorr(Ql,Rl,r,dx,N,cutoff):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=1.0/(2.0*cutoff)*(np.reshape(np.transpose(Rl)+np.conj(Rl),D*D))
    rdens=np.tensordot(Rl,r,([1],[0]))+np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def HomogeneousLiebLinigerCdagC(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=np.reshape(np.conj(Rl),D*D)
    rdens=np.tensordot(Rl,r,([1],[0]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def HomogeneousLiebLinigerCCdag(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=np.reshape(np.transpose(Rl),D*D)
    rdens=np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def HomogeneousLiebLinigerCdagCdag(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=np.reshape(np.conj(Rl),D*D)
    rdens=np.tensordot(r,np.conj(Rl),([1],[1]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def HomogeneousLiebLinigerCC(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=np.reshape(np.transpose(Rl),D*D)
    rdens=np.tensordot(Rl,r,([1],[0]))
    for n in range(N):
        if n%100==0:
            stdout.write("\r %i" % n)
            stdout.flush()
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr


def HomogeneousLiebLinigerNN(Ql,Rl,r,dx,N):
    D=np.shape(Ql)[0]
    corr=np.zeros(N,dtype=type(Ql[0,0]))
    vec=np.reshape(np.transpose(herm(Rl).dot(Rl)),D*D)
    rdens=np.tensordot(np.tensordot(Rl,r,([1],[0])),np.conj(Rl),([1],[1]))
    for n in range(N):
        vec=vec+dx*transferOperator2ndOrder(Ql,Rl,0.0,0.0,1,vec)
        corr[n]=np.tensordot(np.reshape(vec,(D,D)),rdens,([0,1],[0,1]))
    return corr



def computeSteadyStateFHomogeneousLiebLiniger_l_r(cmps,l,r,mu,g,mass,direction,initdensity=None,initx0=None):
    if direction<0:
        assert(cmps._position==0)
        r=np.eye(cmps._D)
        f=np.zeros((cmps._D,cmps._D),cmps._dtype)
        ih=dfdxLiebLiniger(cmps,f,mu,mass,g,direction=-1,ind=1)
        ihprojected=-(ih-np.trace(np.transpose(l).dot(ih))*r)
        f1,nit=inverseTransferOperator(cmps._Q[0],cmps._R[0],cmps._dx[0],l,r,ihprojected,direction=-1,x0=initx0,tolerance=1e-12,maxiteration=4000)
        return f1

    if direction>0:
        assert(cmps._position==cmps._N)
        l=np.eye(cmps._D)
        f=np.zeros((cmps._D,cmps._D),cmps._dtype)

        ih=dfdxLiebLiniger(cmps,f,mu,mass,g,direction=1,ind=1)
        ihprojected=-(ih-np.trace(np.transpose(ih).dot(r))*l)

        f1,nit=inverseTransferOperator(cmps._Q[0],cmps._R[0],cmps._dx[0],l,r,ihprojected,direction=1,x0=initx0,tolerance=1e-12,maxiteration=4000)
        return f1



def computeSteadyStateFHomogeneousLiebLiniger(cmps,mu,g,mass,direction,initdensity=None,initx0=None,tol=1E-10,nmax=4000):
    if direction<0:
        assert(cmps._position==0)
        r=np.eye(cmps._D)
        [eta,v,numeig]=TMeigs2ndOrder(cmps._Q[0],cmps._R[0],cmps._dx[0],1,numeig=1,init=initdensity,nmax=nmax,tolerance=nmax,ncv=100,which='SM')
        l=np.reshape(v,(cmps._D,cmps._D))
        l=l/np.trace(l)
        if cmps._dtype==float:
            l=np.real(l+herm(l))/2.0
        if cmps._dtype==complex:
            l=(l+herm(l))/2.0

        f=np.zeros((cmps._D,cmps._D),cmps._dtype)
        ih=dfdxLiebLiniger(cmps,f,mu,mass,g,direction=-1,ind=1)
        ihprojected=-(ih-np.trace(np.transpose(l).dot(ih))*r)
        f1,nit=inverseTransferOperator(cmps._Q[0],cmps._R[0],cmps._dx[0],l,r,ihprojected,direction=-1,x0=initx0,tolerance=tol,maxiteration=nmax)
        return f1

    if direction>0:
        assert(cmps._position==cmps._N)
        l=np.eye(cmps._D)
        [eta,v,numeig]=TMeigs2ndOrder(cmps._Q[0],cmps._R[0],cmps._dx[0],-1,numeig=1,init=initdensity,nmax=nmax,tolerance=tol,ncv=100,which='SM')
        r=np.reshape(v,(cmps._D,cmps._D))
        r=r/np.trace(r)
        if cmps._dtype==float:
            r=np.real(r+herm(r))/2.0
        if cmps._dtype==complex:
            r=(r+herm(r))/2.0


        f=np.zeros((cmps._D,cmps._D),cmps._dtype)

        ih=dfdxLiebLiniger(cmps,f,mu,mass,g,direction=1,ind=1)
        ihprojected=-(ih-np.trace(np.transpose(ih).dot(r))*l)

        f1,nit=inverseTransferOperator(cmps._Q[0],cmps._R[0],cmps._dx[0],l,r,ihprojected,direction=1,x0=initx0,tolerance=tol,maxiteration=nmax)
        return f1





def LiebLinigerCMPSSteadyStateHamiltonianGMRES(cmps,mpopbc,mu,inter,mass,boundary,ldens,rdens,direction,thresh,imax):

    NUC=cmps._N
    D=cmps._D
    mps=cmps.__toMPS__(connect='right')
    [B1,B2,d1,d2]=np.shape(mpopbc[NUC-1])
    if direction>0:
        mpo=np.zeros((1,B2,d1,d2),dtype=cmps._dtype)
        mpo[0,:,:,:]=mpopbc[NUC-1][-1,:,:,:]
        L=mf.initializeLayer(mps[NUC-1],np.eye(D),mps[NUC-1],mpo,1)
        f=np.zeros((cmps._D,cmps._D),cmps._dtype)
        for n in range(0,cmps._N-1):
            f=evolveFLiebLiniger(cmps,f,mu,mass,inter,n+1,n+2,direction=1)
            L=addLayer(L,mps[n],mpopbc[n],mps[n],1)    
            print (L[:,:,0]-f)
            raw_input()

        h=np.trace(L[:,:,0].dot(rdens[-1]))
        inhom=np.reshape(L[:,:,0]-h*np.transpose(np.eye(D)),D*D) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D*D)),thresh,imax,datatype=dtype,direction=1)
        L[:,:,0]=np.reshape(k2,(D,D))
        return np.copy(L)

    if direction<0:
        mpo=np.zeros((B1,1,d1,d2),dtype=dtype)
        mpo[:,0,:,:]=mpopbc[0][:,0,:,:]
        R=mf.initializeLayer(mps[0],np.eye(D),mps[0],mpo,-1)
        for n in range(len(mps)-1,-1,-1):
            R=addLayer(R,mps[n],mpopbc[n],mps[n],-1)    
        h=np.trace(R[:,:,-1].dot(ldens[0]))

        inhom=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D)),D*D) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D*D)),thresh,imax,datatype=dtype,direction=-1)
        R[:,:,-1]=np.reshape(k2,(D,D))
        return np.copy(R)


def computeGridUCsteadyStateHamiltonianGMRES(cmps,grid,mpopbc,boundary,ldens,rdens,direction,thresh,imax):
    NUC=len(mps)
    [D1r,D2r,d]=np.shape(mps[NUC-1])
    [D1l,D2l,d]=np.shape(mps[NUC-1])
    [B1,B2,d1,d2]=np.shape(mpopbc[NUC-1])
    if direction>0:
        mpo=np.zeros((1,B2,d1,d2),dtype=cmps._dtype)
        mpo[0,:,:,:]=mpopbc[NUC-1][-1,:,:,:]
        L=mf.initializeLayer(mps[NUC-1],np.eye(D1r),mps[NUC-1],mpo,1)

        #L=np.zeros((D2r,D2r,B2))
        #L[:,:,-1]=np.eye(D1l)
        #L=addLayer(L,mps[NUC-1],mpopbc[-1],mps[NUC-1],1)    
        for n in range(0,len(mps)):
            L=addLayer(L,mps[n],mpopbc[n],mps[n],1)    

        h=np.trace(L[:,:,0].dot(rdens[-1]))
        inhom=np.reshape(L[:,:,0]-h*np.transpose(np.eye(D2r)),D2r*D2r) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D1l*D1l)),thresh,imax,datatype=cmps._dtype,direction=1)
        L[:,:,0]=np.reshape(k2,(D2r,D2r))
        return np.copy(L)

    if direction<0:
        mpo=np.zeros((B1,1,d1,d2),dtype=cmps._dtype)
        mpo[:,0,:,:]=mpopbc[0][:,0,:,:]
        R=mf.initializeLayer(mps[0],np.eye(D1l),mps[0],mpo,-1)
        for n in range(len(mps)-1,-1,-1):
            R=addLayer(R,mps[n],mpopbc[n],mps[n],-1)    
        h=np.trace(R[:,:,-1].dot(ldens[0]))

        inhom=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D1l)),D2r*D2r) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D1l*D1l)),thresh,imax,datatype=cmps._dtype,direction=-1)
        R[:,:,-1]=np.reshape(k2,(D1l,D1l))
        return np.copy(R)

def svd(mat,full_matrices=False,r_thresh=1E-14):
    try: 
        [u,s,v]=np.linalg.svd(mat,full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        [q,r]=np.linalg.qr(mat)
        r[np.abs(r)<r_thresh]=0.0
        u_,s,v=np.linalg.svd(r)
        u=q.dot(u_)
        print('caught a LinAlgError with dir>0')
    return u,s,v

def qr(mat,signfix):
    dtype=type(mat[0,0])
    q,r=np.linalg.qr(mat)
    if signfix=='q':
        sign=np.sign(np.diag(q))
        unit=np.diag(sign)
        return q.dot(unit),herm(unit).dot(r)
    if signfix=='r':
        sign=np.sign(np.diag(r))
        unit=np.diag(sign)
        return q.dot(herm(unit)),unit.dot(r)




#returns the derivative of vector; Q and R should be normalized according to dx
#to be consistent with the other MPS code, vector is obtained from a bond matrix, where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def transferOperator2ndOrder(Q,R,dx,sigma,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    if direction>0:
        if abs(sigma)<1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))-sigma*x,(D*D))

    if direction<0:
        if abs(sigma)<1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q))-sigma*x,(D*D))




""" 
Q1,R1 is the cMPS that has to be matched to Q2,R2 using a cMPO operator.
pass lam as a diagonal matrix; pass left orthogonal cMPS matrices Q1l and R1l and Q1ltensored and R1ltensored for the 
tensored and untensored state; G1ltensored is the gauge transformation that brings Qltensored and R1tensored into
left canonical gauge, 
"""

def calculateCMPOGradient_1(R1ltensored,lam1tensored,G1ltensored,G1linvtensored,R1l):
    #reshape the matrices into tensors:
    D=R1l.shape[0]
    M=int(R1ltensored.shape[0]/D)
    if np.linalg.norm(M-R1ltensored.shape[0]/D)>1E-10:
        sys.exit("calculateCMPOGradient: shape[0] of R1ltensored is not an integer multiple of shape[0] of R1l")
    Rctensored=R1ltensored.dot(lam1tensored)
    G1l=np.reshape(G1ltensored,(D*M,D,M))
    G1linvlam=np.reshape(G1linvtensored.dot(lam1tensored),(D,M,D*M))
    #this is the contribution of the square-cMPO expression:
    #the gradient of Gammas[0][1]  and Gammas[1][0] are contracted correctly, this has been checked with leftnorm (see below)
    gG00=ncon.ncon([np.conj(G1l),lam1tensored,np.conj(G1linvlam)],[[1,2,-1],[1,3],[2,-2,3]])
    gG01=ncon.ncon([np.conj(G1l),Rctensored,np.conj(G1linvlam)],[[1,2,-1],[1,3],[2,-2,3]])
    gG10=ncon.ncon([np.conj(G1l),lam1tensored,np.conj(R1l),np.conj(G1linvlam)],[[1,2,-1],[1,3],[2,4],[4,-2,3]])
    return [[gG00,gG01],[gG10,np.zeros((M,M)).astype(R1l.dtype)]]

def calculateCMPOGradient_2(R2l,lam2,R1l,lam1tensored,G1ltensored,G1linvtensored,L21,R21):
    #reshape the matrices into tensors:
    D=R1l.shape[0]
    M=int(G1ltensored.shape[0]/D)
    if np.linalg.norm(M-lam1tensored.shape[0]/D)>1E-10:
        sys.exit("calculateCMPOGradient: shape[0] of Rltensored is not an integer multiple of shape[0] of Rl")
    R2c=R2l.dot(lam2)
    G1l=np.reshape(G1ltensored,(D*M,D,M))
    G1linvlam=np.reshape(G1linvtensored.dot(lam1tensored),(D,M,D*M))
    #this is the contribution of the square-cMPO expression:
    #the gradient of Gammas[0][1]  and Gammas[1][0] are contracted correctly, this has been checked with leftnorm (see below)
    gG00=ncon.ncon([L21,np.conj(G1l),lam2,np.conj(G1linvlam),R21],[[1,2],[2,3,-1],[1,4],[3,-2,5],[4,5]])
    gG10=ncon.ncon([L21,np.conj(G1l),lam2,R1l,np.conj(G1linvlam),R21],[[1,2],[2,3,-1],[1,4],[3,6],[6,-2,5],[4,5]])
    gG01=ncon.ncon([L21,np.conj(G1l),R2c,np.conj(G1linvlam),R21],[[1,2],[2,3,-1],[1,4],[3,-2,5],[4,5]])

    return [[gG00,gG01],[gG10,np.zeros((M,M)).astype(R1l.dtype)]]

"""
for now use only D1=D2!!!
"""
def cMPOoptimization(Q1,R1,Q2,R2,M,alpha=1E-3,Gamma_init=None,IMAX=10000):
    plt.ion()
    dtype=Q1.dtype
    D=Q1.shape[0]
    if Gamma_init==None:
        scaling=0.1
        Gamma00=(np.random.rand(M,M)-0.5-1j*(np.random.rand(M,M)-0.5))*scaling
        #Gamma00=(Gamma00+herm(Gamma00))/2.0
        
        Gamma01=(np.random.rand(M,M)-0.5-1j*(np.random.rand(M,M)-0.5))*scaling
        #Gamma01=(Gamma01+herm(Gamma01))/2.0
        
        #Gamma10=np.copy((Gamma01))
        Gamma10=(np.random.rand(M,M)-0.5-1j*(np.random.rand(M,M)-0.5))*scaling        
        Gamma11=(np.random.rand(M,M)-0.5-1j*(np.random.rand(M,M)-0.5))*scaling
        #Gamma11=(Gamma11+herm(Gamma11))/2.0

        Gammas=[[Gamma00,Gamma01],[Gamma10,Gamma11]]    
    else:
        Gammas=Gamma_init
    assert(M==Gammas[0][0].shape[0])

    """ 
    first normalize the cMPS matrices
    """
    D1=Q1.shape[0]
    D2=Q2.shape[0]
    lam1,Q1l,R1l,Q1r,R1r=regauge_old(Q1,R1,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-12,thresh=1E-10)
    lam2,Q2l,R2l,Q2r,R2r=regauge_old(Q2,R2,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-12,thresh=1E-10)    


    dx=0.01
    N=2000
    nn1=HomogeneousLiebLinigerNN(Q1l,R1l,np.diag(lam1**2),dx,N)
    nn2=HomogeneousLiebLinigerNN(Q2l,R2l,np.diag(lam2**2),dx,N)
    x=np.linspace(0,N*dx,len(nn1))
    #plt.semilogx(x,nn1,x,nn2)
    #plt.legend(['state 1','state 2'])
    #plt.draw()
    #plt.show()

    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)

    converged=False
    #grad_Gammas=cMPOLocalGradient(np.diag(lam1),R1c,np.diag(lam2),R2c,Gammas,L11,R11,L12,R12,L21,R21,L22,R22)
    it=0
    gradnorm=[[0.,0.],[0.,0.]]
    D=Q1.shape[0]
    dtype=Q1.dtype

    while not converged:
        #(D,M,D,M)
        """
        the normalized cMPS matrices Q1l, R1l and Q1r, R1r are tensored with the current Gammas:
        """
        Q1ltens=np.copy(np.kron(np.eye(D),Gammas[0][0])+np.kron(Q1l,diag)+np.kron(R1l,Gammas[1][0]))
        R1ltens=np.copy(np.kron(np.eye(D),Gammas[0][1])+np.kron(R1l,diag))
        Q1rtens=np.copy(np.kron(np.eye(D),Gammas[0][0])+np.kron(Q1r,diag)+np.kron(R1r,Gammas[1][0]))
        R1rtens=np.copy(np.kron(np.eye(D),Gammas[0][1])+np.kron(R1r,diag))

        """
        The  tensored matrices Q1ltens, ... are now normalized:
        """
        #lamtens,Qltens,Rltens,Qrtens,Rrtens,Gltens,Glinvtens,Grtens,Grinvtens,Z,eta=regaugeSymmetricKron(Q1ltens,R1ltens,D,M,initial=None,nmaxit=100000,tol=1E-12,ncv=100,numeig=5,pinv=1E-12,thresh=1E-12,trunc=1E-8)
        lamtens,Qltens,Rltens,Qrtens,Rrtens,Gltens,Glinvtens,Grtens,Grinvtens,Z,nl,nr=regauge_return_basis(Q1ltens,R1ltens,dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-12,ncv=100,numeig=5,pinv=1E-12,thresh=1E-12,trunc=1E-200)
        
        nn3=HomogeneousLiebLinigerNN(Qltens,Rltens,np.diag(lamtens**2),dx,N)
        plt.clf()
        plt.semilogx(x,nn1,x,nn2,x,nn3)
        plt.legend(['state 1','state 2','cMPO applied'],loc='best')
        plt.draw()
        plt.show()
        input()
        #Dtens=Qltens.shape[0]        
        """
        calculate the mixed transfer operators using Q1l, R1l, Q2l,R2l and  Q1r, R1r, Q2r ,R2r
        """
        etal21tens,vltens=mixedTMeigs1stOrder(np.copy(Q2l),np.copy(R2l),np.copy(Qltens),np.copy(Rltens),direction=1,numeig=1,init=None,nmax=10000000,tolerance=1e-10,ncv=100,which='LR')
        etar21tens,vrtens=mixedTMeigs1stOrder(np.copy(Q2r),np.copy(R2r),np.copy(Qrtens),np.copy(Rrtens),direction=-1,numeig=1,init=None,nmax=10000000,tolerance=1e-10,ncv=100,which='LR')    
        L21tens=np.reshape(vltens,(D,D,M))
        L21tens/=np.trace(L21tens[:,:])
        L21tens=np.reshape(L21tens,(D,D*M))
        R21tens=np.reshape(vrtens,(D,D,M))
        R21tens/=np.trace(R21tens[:,:])
        R21tens=np.reshape(R21tens,(D,D*M))

        overlap=np.tensordot(L21tens,R21tens,([0,1],[0,1]))

        g1=calculateCMPOGradient_1(Rltens,np.diag(lamtens),Gltens,Glinvtens,R1l)
        g2=calculateCMPOGradient_2(R2l,np.diag(lam2),R1l,np.diag(lamtens),Gltens,Glinvtens,L21tens,R21tens)
        for m in range(len(Gammas)):
            for n in range(len(Gammas[m])):
                if not(m==1 and n==1):
                    gradnorm[m][n]=np.linalg.norm(g1[m][n])+np.linalg.norm(g2[m][n])
                    #print(m,n,np.linalg.norm(g1[m][n]),np.linalg.norm(g2[m][n]))
                    Gammas[m][n]-=alpha*(g1[m][n]-g2[m][n])
                    #Gammas[m][n]-=alpha*g2[m][n]
        #if it%10==0:
            #print (Gammas[0][0])
            #print()
            #print (Gammas[0][1])
            #print()
            #print (Gammas[1][0])
            #print('g1')
            #print (g1[0][0])
            #print()
            #print (g1[0][1])
            #print()
            #print (g1[1][0])
            #print()
            #print('g2')
            #print (g2[0][0])
            #print()
            #print (g2[0][1])
            #print()
            #print (g2[1][0])

        it+=1
                
        if it>IMAX:
            print('cMPOoptimization did not converge within {0} steps'.format(IMAX))
            break
        print (it,np.sum(gradnorm),np.abs(overlap))




"""
 ########################################                               THE CMPO ROUTINES ARE FOR THE MOST PART DEPRECATED; IN PRACTICE EVERYTHING CAN BE DONE USING             ############################################
 ########################################                           A SINGLE LAYER PICTURE BY TENSORING THE CMPO WITH TNE CMPS AND KEEPING TRACK OF WHERE THE INDICES GO         ############################################
"""

"""
calculates the derivative operator for a cMPS-cMPO-cMPS transfer-operation (similar to addLayer in mpsfunctions.py)
Gammas is a double-list of cMPO matrices [[Gamma00,Gamma01],[Gamma10, Gamma11]]; the local Hilbert-space
is restricted to d=2;
"""
def cMPOtransferOperator(Qupper,Rupper,Qlower,Rlower,Gammas,direction,vector):
    Du=np.shape(Qupper)[0]
    Dl=np.shape(Qlower)[0]
    M=Gammas[0][0].shape[0]
    dtype=Qupper.dtype
    assert(Rupper.dtype==dtype)
    assert(Qlower.dtype==dtype)
    assert(Rlower.dtype==dtype)
    assert(vector.dtype==dtype)
    x=np.reshape(vector,(Du,Dl,M))

    """diag below incorporates a special parametrization of the cMPO of the from of 

    -----------------------------------------------------------------
    |delta_{0,m} +eps Gammas[0][0]  |     sqrt(eps)Gammas[0][1]      |
    |---------------------------------------------------------------|
    |      sqrt(eps)Gammas[1][0]     | delta_{0,m} + eps Gammas[1][1] |
    -----------------------------------------------------------------
      
    where delta_{0,m}=diag; the use of only a single non-zero element on diag ensures that
    after application of the cMPO O to the cMPS |psi>, O|psi>, and in the limit of Gammas[i][j]=0, 
    the resulting state reduces to the initial one, i.e O|psi>=|psi>; if a full diagonal was used,
    the application O|psi> would result in a superposition O|psi>=(\sum_i |psi>) of M identical states |psi>.
    Algebraicly, this is not a problem, since it is merely a rescaling of the state. Numerically, such 
    superposition can be problematic due to the non-injectivity of the state.
    """

    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)
    if direction>0:
        return np.reshape(ncon.ncon([x,Qupper,diag],[[1,-2,2],[1,-1],[2,-3]])+\
                          ncon.ncon([x,diag,np.conj(Qlower)],[[-1,2,1],[1,-3],[2,-2]])+\
                          ncon.ncon([x,Gammas[0][0]],[[-1,-2,1],[1,-3]])+\
                          ncon.ncon([x,Rupper,Gammas[0][1]],[[1,-2,2],[1,-1],[2,-3]])+\
                          ncon.ncon([x,np.conj(Rlower),Gammas[1][0]],[[-1,2,1],[2,-2],[1,-3]])+\
                          ncon.ncon([x,Rupper,np.conj(Rlower),diag],[[1,2,3],[1,-1],[2,-2],[3,-3]]),(Du*Dl*M))
    if direction<0:
        return np.reshape(ncon.ncon([x,Qupper,diag],[[1,-2,2],[-1,1],[-3,2]])+\
                          ncon.ncon([x,diag,np.conj(Qlower)],[[-1,2,1],[-3,1],[-2,2]])+\
                          ncon.ncon([x,Gammas[0][0]],[[-1,-2,1],[-3,1]])+\
                          ncon.ncon([x,Rupper,Gammas[0][1]],[[1,-2,2],[-1,1],[-3,2]])+\
                          ncon.ncon([x,np.conj(Rlower),Gammas[1][0]],[[-1,2,1],[-2,2],[-3,1]])+\
                          ncon.ncon([x,Rupper,np.conj(Rlower),diag],[[1,2,3],[-1,1],[-2,2],[-3,3]]),(Du*Dl*M))
        

"""
calculates the derivative operator for a cMPS-cMPO-cMPS transfer-operation (similar to addLayer in mpsfunctions.py)
Gammas is a double-list of cMPO matrices [[Gamma00,Gamma01],[Gamma10, Gamma11]]; the local Hilbert-space
is restricted to d=2;
"""
def cMPOsquaredTransferOperator(Qupper,Rupper,Qlower,Rlower,Gammas,direction,vector):
    Du=np.shape(Qupper)[0]
    Dl=np.shape(Qlower)[0]
    M=Gammas[0][0].shape[0]
    dtype=Qupper.dtype
    assert(Rupper.dtype==dtype)
    assert(Qlower.dtype==dtype)
    assert(Rlower.dtype==dtype)
    assert(vector.dtype==dtype)
    x=np.reshape(vector,(Du,Dl,M,M))

    """diag below incorporates a special parametrization of the cMPO of the from of 

    -----------------------------------------------------------------
    |delta_{0,m} +eps Gammas[0][0]  |     sqrt(eps)Gammas[0][1]      |
    |---------------------------------------------------------------|
    |      sqrt(eps)Gammas[1][0]     | delta_{0,m} + eps Gammas[1][1] |
    -----------------------------------------------------------------
      
    where delta_{0,m}=diag; the use of only a single non-zero element on diag ensures that
    after application of the cMPO O to the cMPS |psi>, O|psi>, and in the limit of Gammas[i][j]=0, 
    the resulting state reduces to the initial one, i.e O|psi>=|psi>; if a full diagonal was used,
    the application O|psi> would result in a superposition O|psi>=(\sum_i |psi>) of M identical states |psi>.
    Algebraicly, this is not a problem, since it is merely a rescaling of the state. Numerically, such 
    superposition can be problematic due to the non-injectivity of the state.
    """
    #ncon([x,A,B,C,D],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)
    eyeu=np.eye(Du).astype(dtype)
    eyel=np.eye(Dl).astype(dtype)
    v=np.reshape(np.transpose(x,(0,2,1,3)),(Du*M,Dl*M))
    D=Du
    if direction>0:
        ##1
        #T1=np.kron(Qupper,diag)
        #T2=np.kron(eyel,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,Qupper,diag,diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,Qupper,diag,diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        ##2
        #T1=np.kron(eyeu,Gammas[0][0])
        #T2=np.kron(eyel,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,Gammas[0][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,Gammas[0][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        ##3
        #T1=np.kron(eyeu,diag)
        #T2=np.kron(eyel,Gammas[0][0])
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,diag,np.conj(Gammas[0][0]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,diag,np.conj(Gammas[0][0]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        ##4
        #T1=np.kron(eyeu,diag)
        #T2=np.kron(Qlower,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,diag,diag,np.conj(Qlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,diag,diag,np.conj(Qlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        ##5
        #T1=np.kron(eyeu,Gammas[0][1])
        #T2=np.kron(eyel,Gammas[0][1])
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,Gammas[0][1],np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,Gammas[0][1],np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        #
        ##6
        #T1=np.kron(eyeu,Gammas[0][1])
        #T2=np.kron(Rlower,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,Gammas[0][1],diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,Gammas[0][1],diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        #
        #
        ##7
        #T1=np.kron(eyeu,diag)
        #T2=np.kron(Rlower,Gammas[1][0])
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,eyeu,diag,np.conj(Gammas[1][0]),np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,eyeu,diag,np.conj(Gammas[1][0]),np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        #
        ##8
        #T1=np.kron(Rupper,Gammas[1][0])
        #T2=np.kron(eyel,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,Rupper,Gammas[1][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,Rupper,Gammas[1][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        #
        ##9
        #T1=np.kron(Rupper,diag)
        #T2=np.kron(eyel,Gammas[0][1])
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,Rupper,diag,np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,Rupper,diag,np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))
        #
        ##10
        #T1=np.kron(Rupper,diag)
        #T2=np.kron(Rlower,diag)
        #bla=np.transpose(herm(T2).dot(np.transpose(v)).dot(T1))
        #term2=np.transpose(np.reshape(bla,(D,M,D,M)),(0,2,1,3))
        #term1=ncon.ncon([x,Rupper,diag,diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #term1=ncon.ncon([x,Rupper,diag,diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])
        #print (np.linalg.norm(term2-term1))

        return np.reshape(ncon.ncon([x,Qupper,diag,diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,diag,np.conj(Gammas[0][0]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,diag,diag,np.conj(Qlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][1],np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][1],diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,eyeu,diag,np.conj(Gammas[1][0]),np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,Rupper,Gammas[1][0],diag,eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,Rupper,diag,np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]])+\
                          ncon.ncon([x,Rupper,diag,diag,np.conj(Rlower)],[[1,2,3,4],[1,-1],[3,-3],[4,-4],[2,-2]]),(Du*Dl*M*M))
        
    if direction<0:
        return np.reshape(ncon.ncon([x,Qupper,diag,diag,eyel],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][0],diag,eyel],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,diag,np.conj(Gammas[0][0]),eyeu],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,diag,diag,np.conj(Qlower)],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][1],np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,Gammas[0][1],diag,np.conj(Rlower)],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,eyeu,diag,np.conj(Gammas[1][0]),np.conj(Rlower)],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,Rupper,Gammas[1][0],diag,eyel],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,Rupper,diag,np.conj(Gammas[0][1]),eyel],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]])+\
                          ncon.ncon([x,Rupper,diag,diag,np.conj(Rlower)],[[1,2,3,4],[-1,1],[-3,3],[-4,4],[-2,2]]),(Du*Dl*M*M))


#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def cMPOTMeigs(Qupper,Rupper,Qlower,Rlower,Gammas,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    Du=np.shape(Qupper)[0]
    Dl=np.shape(Qupper)[0]
    M=Gammas[0][0].shape[0]

    mv=fct.partial(cMPOtransferOperator,*[Qupper,Rupper,Qlower,Rlower,Gammas,direction])
    LOP=LinearOperator((Du*Dl*M,Du*Dl*M),matvec=mv,rmatvec=None,matmat=None,dtype=Qupper.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        #while np.abs(np.imag(eta[m]))>1E-6:
        #    #numeig=numeig+1
        #    print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
        #    numeig=7
        #    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D*M),maxiter=nmax,tol=tolerance,ncv=ncv)
        #    m=np.argmax(np.real(eta))
        return eta[m],vec[:,m],numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return cMPOTMeigs(Qupper,Rupper,Qlower,Rlower,Gammas,numeig,np.random.rand(Du*Dl*M),nmax,tolerance,ncv,which)



#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def cMPOsquaredTMeigs(Q,R,Gammas,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    Du=np.shape(Q)[0]
    Dl=np.shape(Q)[0]
    M=Gammas[0][0].shape[0]

    mv=fct.partial(cMPOsquaredTransferOperator,*[Q,R,Q,R,Gammas,direction])
    LOP=LinearOperator((Du*Dl*M*M,Du*Dl*M*M),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        #while np.abs(np.imag(eta[m]))>1E-6:
        #    #numeig=numeig+1
        #    print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
        #    numeig=7
        #    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D*M),maxiter=nmax,tol=tolerance,ncv=ncv)
        #    m=np.argmax(np.real(eta))
        return eta[m],vec[:,m],numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return cMPOTMeigs(Q,R,Gammas,numeig,np.random.rand(Du*Dl*M),nmax,tolerance,ncv,which)

#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def cMPOsquaredTMeig(Q,R,Gammas,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial

    D=Q.shape[0]
    dtype=Q.dtype
    Rcon=np.conj(R)
    Qcon=np.conj(Q)
    Gamma00=Gammas[0][0]
    Gamma10=Gammas[1][0]
    Gamma01=Gammas[0][1]
    Gamma11=Gammas[1][1]

    Gamma00con=np.conj(Gammas[0][0])
    Gamma10con=np.conj(Gammas[1][0])
    Gamma01con=np.conj(Gammas[0][1])
    Gamma11con=np.conj(Gammas[1][1])
    M=Gammas[0][0].shape[0]
    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)
    eye=np.eye(D).astype(dtype)
    dim=D**2*M**2
    OP=np.zeros((dim,dim)).astype(dtype)
    for m in range(dim):
        stdout.write("\r%i"%(m))
        stdout.flush()
        x=np.zeros(dim).astype(dtype)
        x[m]=1.0
        for n in range(dim):
            y=np.zeros(dim).astype(dtype)
            y[n]=1.0
            OP[m,n]=y.dot(cMPOsquaredTransferOperator(Q,R,Q,R,Gammas,direction,x))
            
    eta,U=np.linalg.eig(OP)
    return eta,U
"""
pass lami and lamj as diagonal matrices
"""
def cMPScMPOcMPSterm_21_Oeps(lam1,Q1c,R1c,lam2,Q2c,R2c,Gammas,L21,R21):
    d=np.zeros(L21.shape[2]).astype(L21.dtype)
    d[:]=1.0
    diag=np.diag(d)
    
    return ncon.ncon([L21,Q2c,lam1,R21,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L21,lam2,np.conj(Q1c),R21,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L21,lam2,lam1,R21,np.conj(Gammas[0][0])],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L21,lam2,np.conj(R1c),R21,np.conj(Gammas[0][1])],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L21,R2c,lam1,R21,np.conj(Gammas[1][0])],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L21,R2c,np.conj(R1c),R21,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])

"""
pass lam2 and lamj as diagonal matrices
"""
def cMPScMPOcMPSterm_21_O1(lam2,lam1,L21,R21):
    d=np.zeros(L21.shape[2]).astype(L21.dtype)
    d[:]=1.0
    diag=np.diag(d)
    return ncon.ncon([L21,lam2,lam1,R21,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])    


"""
pass lami and lamj as diagonal matrices
"""
def cMPScMPOcMPSterm_12_Oeps(lam1,Q1c,R1c,lam2,Q2c,R2c,Gammas,L12,R12):
    d=np.zeros(L12.shape[2]).astype(L12.dtype)
    d[:]=1.0
    diag=np.diag(d)

    return ncon.ncon([L12,Q1c,lam2,R12,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L12,lam1,np.conj(Q2c),R12,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L12,lam1,lam2,R12,Gammas[0][0]],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L12,lam1,np.conj(R2c),R12,Gammas[0][1]],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L12,R1c,lam2,R12,Gammas[1][0]],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])+\
        ncon.ncon([L12,R1c,np.conj(R2c),R12,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])

"""
pass lam2 and lamj as diagonal matrices
"""
def cMPScMPOcMPSterm_12_O1(lam1,lam2,L12,R12):
    d=np.zeros(L12.shape[2]).astype(L12.dtype)
    d[:]=1.0
    diag=np.diag(d)
    return ncon.ncon([L12,lam1,lam2,R12,diag],[[1,3,5],[1,2],[3,4],[2,4,6],[5,6]])    


def cMPScMPOcMPOcMPSterm_Oeps(lam1,Q1c,R1c,Gammas,L11sq,R11sq):
    d=np.zeros(L11sq.shape[2]).astype(L11sq.dtype)
    d[:]=1.0
    diag=np.diag(d)
    return ncon.ncon([L11sq,Q1c,diag,lam1,R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,Gammas[0][0],lam1,R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,diag,np.conj(Q1c),R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,diag,lam1,R11sq,np.conj(Gammas[0][0])],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,Gammas[0][1],lam1,R11sq,np.conj(Gammas[1][0])],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,Gammas[0][1],np.conj(R1c),R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,lam1,diag,np.conj(R1c),R11sq,np.conj(Gammas[0][1])],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,R1c,Gammas[1][0],lam1,R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,R1c,diag,lam1,R11sq,np.conj(Gammas[1][0])],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])+\
        ncon.ncon([L11sq,R1c,diag,np.conj(R1c),R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])

def cMPScMPOcMPOcMPSterm_O1(lam1,L11sq,R11sq):
    d=np.zeros(L11sq.shape[2]).astype(L11sq.dtype)
    d[:]=1.0
    diag=np.diag(d)
    return ncon.ncon([L11sq,lam1,diag,lam1,R11sq,diag],[[1,2,3,7],[1,4],[3,6],[2,5],[4,5,6,8],[7,8]])
    
"""
L11 is the L-cMPO expression with Q1,R1 on both the upper and lower leg, ie.e the steady state of the cMPS-cMPO-cMPS sandwich expression.
L21 is the L-cMPO expression with Q2,R2 on the upper and Q1,R1 on the lower leg (the rest follows from that)
R11,R21,R12,R22 are the R-cMPO expressions
Note that the local update of the Gammas does not depend on the cMPS matrices Q1, Q2. Q1 and Q2 enter only via L11,L12,... 
as these depend implicitly on Q1, Q2.
the update of the Gammas does not depend on the cMPS matrices Qi, Qj!
pass lami and lamj as diagonal matrices, NOT AS VECTORS!!
"""
def cMPOLocalGradientMisc(lam1,R1c,lam2,R2c,Gammas,L21,R21,L11sq,R11sq):
    dtype=L21.dtype
    assert(R21.dtype==dtype)
    assert(R2c.dtype==dtype)
    assert(R1c.dtype==dtype)
    M=L21.shape[2]
    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)

    """
    This is the contribution to lowest order in eps from the cMPS-cMPO-cMPS part:
    """
    grad_Gamma_1=[[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)],[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)]]
    grad_Gamma_1[0][0]=ncon.ncon([L21,lam2,lam1,R21],[[1,3,-1],[1,2],[3,4],[2,4,-2]])
    grad_Gamma_1[1][0]=ncon.ncon([L21,R2c,lam1,R21],[[1,3,-1],[1,2],[3,4],[2,4,-2]])
    grad_Gamma_1[0][1]=ncon.ncon([L21,lam2,np.conj(R1c),R21],[[1,3,-1],[1,2],[3,4],[2,4,-2]])

    """ 
    These are the contributions from the cMPS-cMPO-cMPO-cMPS term.
    Gammas[1][0] gets two contributions; it's correct as far as I can tell ...
    """
    
    grad_Gamma_2=[[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)],[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)]]
    grad_Gamma_2[0][0]=ncon.ncon([L11sq,lam1,diag,lam1,R11sq],[[1,2,3,-1],[1,4],[3,6],[2,5],[4,5,6,-2]])
    grad_Gamma_2[1][0]=ncon.ncon([L11sq,lam1,Gammas[0][1],lam1,R11sq],[[1,2,3,-1],[1,4],[3,6],[2,5],[4,5,6,-2]])+ncon.ncon([L11sq,R1c,diag,lam1,R11sq],[[1,2,3,-1],[1,4],[3,6],[2,5],[4,5,6,-2]])
    grad_Gamma_2[0][1]=ncon.ncon([L11sq,lam1,diag,np.conj(R1c),R11sq],[[1,2,3,-1],[1,4],[3,6],[2,5],[4,5,6,-2]])

    grad_Gamma=[[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)],[np.zeros((M,M)).astype(dtype),np.zeros((M,M)).astype(dtype)]]
    
    for m in range(len(grad_Gamma)):
        for n in range(len(grad_Gamma[m])):
            grad_Gamma[m][n]=-grad_Gamma_1[m][n]+grad_Gamma_2[m][n]

    return grad_Gamma





def norm_ij(Lij,Rij,lami,lamj):
    d=np.zeros(Lij.shape[2]).astype(Lij.dtype)
    d[:]=1.0
    return ncon.ncon([Lij,lami,np.diag(d),lamj,Rij],[[1,5,3],[1,2],[3,4],[5,6],[2,6,4]])

def norm_O1(L11sq,R11sq,L12,R12,L21,R21,lam1,lam2):
    d=np.zeros(L12.shape[2]).astype(L12.dtype)
    d[:]=1.0
    diag=np.diag(d)
    #print (np.linalg.norm(lam2))
    print (L11sq[:,:,0,0])
    #print(ncon.ncon([L12,np.diag(lam1),diag,np.diag(lam2),R12],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]]))
    #print(ncon.ncon([L21,np.diag(lam2),diag,np.diag(lam1),R21],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]]))
    #input()
    return ncon.ncon([L11sq,lam1,diag,diag,lam1,R11sq],[[1,2,3,4],[1,5],[3,7],[4,8],[2,6],[5,6,7,8]])+np.linalg.norm(lam2)-\
        ncon.ncon([L12,lam1,diag,lam2,R12],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]])-\
        ncon.ncon([L21,lam2,diag,lam1,R21],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]])

def norm_Oeps(L11sq,R11sq,L12,R12,L21,R21,lam1,lam2):
    d=np.zeros(L12.shape[2]).astype(L12.dtype)
    d[:]=1.0
    diag=np.diag(d)
    #print (np.linalg.norm(lam2))
    print (L11sq[:,:,0,0])
    #print(ncon.ncon([L12,np.diag(lam1),diag,np.diag(lam2),R12],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]]))
    #print(ncon.ncon([L21,np.diag(lam2),diag,np.diag(lam1),R21],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]]))
    #input()
    return ncon.ncon([L11sq,lam1,diag,diag,lam1,R11sq],[[1,2,3,4],[1,5],[3,7],[4,8],[2,6],[5,6,7,8]])+np.linalg.norm(lam2)-\
        ncon.ncon([L12,lam1,diag,lam2,R12],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]])-\
        ncon.ncon([L21,lam2,diag,lam1,R21],[[1,2,3],[1,4],[3,6],[2,5],[4,5,6]])



    



#returns the derivative of vector; Q and R should be normalized according to dx
#to be consistent with the other MPS code, vector is obtained from a bond matrix, where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def transferOperator1stOrder(Q,R,sigma,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    if direction>0:
        if abs(sigma)<1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))-sigma*x,(D*D))

    if direction<0:
        if abs(sigma)<1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))-sigma*x,(D*D))
"""
The routine returns the cMPS transfer operator for matrices Q and R that have been obtained from tensoring a cMPS Q_,R_ 
with a cMPO with matrices Gammas[i][j] in the following way:
Q=np.copy(np.kron(np.eye(D),Gammas[0][0])+np.kron(Q_,diag)+np.kron(R_,Gammas[1][0]))
R=np.copy(np.kron(np.eye(D),Gammas[0][1])+np.kron(R_,diag))
diag is a projector onto the state 0 of dimension M, i.e. it is a diagonal matrix of the form np.diag([1,0,...,0]), with M-1 zeros on the diagonal.
using np.reshape(Q,(D,M,D,M) gives a tensor of the form (similar for R)
         0 --     11     -- 2       0 --  Q_ -- 2       0 --     R_     -- 2
Q=                 x            +         x         +            x                                   
         1 --Gammas[0][0]-- 3       1 -- diag-- 3       1 --Gammas[1][0]-- 3
vector is a D**2*M**2 vector. 
To obtain results that are consistent with cMPOsquaredTransferOperator, vector has to be reshaped and transposed, i.e.
tensor=np.transpose(np.reshape(vector,(D,M,D,M)),(0,2,1,3)) can be reshaped into a D**2*M**2 vector fed 
result=cMPOsquaredTransferOperator(...,np.reshape(tensor,D**2*M**2))
np.transpose(np.reshape(result,(D,D,M,M)),(0,2,1,3)) will then be identical to np.reshape(transferOperatorKronecker(...,vector),(D,M,D,M))
"""     
def transferOperatorKronecker(Q,R,D,M,direction,vector):
    dtype=Q.dtype
    d=np.zeros(M).astype(dtype)
    d[:]=1.0
    diag=np.diag(d).astype(dtype)
    
    x=np.reshape(vector,(D*M,D*M))        
    if direction>0:
        return np.reshape(np.transpose(Q).dot(x).dot(np.kron(np.eye(D),diag))+np.transpose(np.kron(np.eye(D),diag)).dot(x).dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R)),(D*M*D*M))
        
    if direction<0:
        return np.reshape(Q.dot(x).dot(np.transpose(np.kron(np.eye(D),diag)))+np.kron(np.eye(D),diag).dot(x).dot(herm(Q))+R.dot(x).dot(herm(R)),(D*M*D*M))        

"""
The routine returns the eigenvector to largest eigenvalue of the cMPS transfer operator for matrices Q and R that have been obtained from tensoring a cMPS Q_,R_ 
with a cMPO with matrices Gammas[i][j] in the following way:
Q=np.copy(np.kron(np.eye(D),Gammas[0][0])+np.kron(Q_,diag)+np.kron(R_,Gammas[1][0]))
R=np.copy(np.kron(np.eye(D),Gammas[0][1])+np.kron(R_,diag))
diag is a projector onto the state 0 of dimension M, i.e. it is a diagonal matrix of the form np.diag([1,0,...,0]), with M-1 zeros on the diagonal.
returns eigenvalue and eigenvector as a D**2*M**2 vector. 
See transferOperatorKronecker(Q,R,D,M,direction,vector) for more details on index convention
D: bond dimension of Q_ and R_, 
M: bond dimension of the cMPO
"""
def TMeigsKronecker(Q,R,D,M,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    mv=fct.partial(transferOperatorKronecker,*[Q,R,D,M,direction])
    LOP=LinearOperator((D**2*M**2,D**2*M**2),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-6:
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state'.format(eta))
            numeig=5
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='LR',v0=np.random.rand(D**2*M**2),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],vec[:,m]

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigsKronecker(Q,R,D,M,direction,numeig,np.random.rand(D**2*M**2),nmax,tolerance,ncv,which)



#returns the derivative of vector; Q and R should be normalized according to dx
#to be consistent with the other MPS code, vector is obtained from a bond matrix, where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def transferOperator2ndOrderCNT(n,Q,R,dx,sigma,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    n[0]+=1
    #print np.diag(x)[0:4]
    if direction>0:
        if abs(sigma)<1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))-sigma*x,(D*D))

    if direction<0:
        if abs(sigma)<1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q))-sigma*x,(D*D))

#returns the derivative of vector; Q and R should be normalized according to dx
#to be consistent with the other MPS code, vector is obtained from a bond matrix, where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def transferOperator1stOrderCNT(n,Q,R,sigma,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    n[0]+=1
    #print np.diag(x)[0:4]
    if direction>0:
        if abs(sigma)<1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))-sigma*x,(D*D))

    if direction<0:
        if abs(sigma)<1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R)),(D*D))
        elif abs(sigma)>=1E-14:
            return np.reshape(Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))-sigma*x,(D*D))

#returns the derivative of vector; Q and R should be normalized according to dx
#to be consistent with the other MPS code, vector is obtained from a bond matrix, where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def transferOperator2ndOrderMultiSpecies(Q,R,dx,direction,vector):
    D=np.shape(Q)[0]
    x=np.reshape(vector,(D,D))
    if direction>0:
        out=np.transpose(Q).dot(x)+x.dot(np.conj(Q))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))
        for n in range(len(R)):
            out=out+np.transpose(R[n]).dot(x).dot(np.conj(R[n]))
        return np.reshape(out,(D*D))

    if direction<0:
        out=Q.dot(x)+x.dot(herm(Q))+dx*Q.dot(x).dot(herm(Q))
        for n in range(len(R)):
            out=out+R[n].dot(x).dot(herm(R[n]))
        return np.reshape(out,(D*D))


#def twoSiteTransferOperator2ndOrder(cmps,direction,vector):
#    assert(cmps._N==2)
#    if direction>0:
#        assert(cmps._position==cmps._N)
#        v0=transferOperator2ndOrder(cmps._Q[0],cmps._R[0],cmps._dx[0],0.0,direction,vector)
#        v1=transferOperator2ndOrder(cmps._Q[1],cmps._R[1],cmps._dx[1],0.0,direction,vector)
#        v01=transferOperator2ndOrder(cmps._Q[1],cmps._R[1],cmps._dx[1],0.0,direction,v0)
#        mat=cmps._mats[-1].dot(cmps._V).dot(cmps._connector)
#
#    return v1+v2+cmps._dx[0]*v12
#

#returns the derivative of x (matrix); Q and R should be normalized according to dx
#to be consistent with the other MPS code, matrix is a bond matrix where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below; R is a list containing the 
#R-matrices for the different species
def matrixtransferOperator2ndOrderMultiSpecies(Q,R,dx,direction,x):
    if direction>0:
        out=np.transpose(Q).dot(x)+x.dot(np.conj(Q))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))
        for n in range(len(R)):
            out=out+np.transpose(R[n]).dot(x).dot(np.conj(R[n]))
        return out

    if direction<0:
        out=Q.dot(x)+x.dot(herm(Q))+dx*Q.dot(x).dot(herm(Q))
        for n in range(len(R)):
            out=out+R[n].dot(x).dot(herm(R[n]))
        return out



#returns the derivative of x (matrix); Q and R should be normalized according to dx
#to be consistent with the other MPS code, matrix is a bond matrix where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def matrixtransferOperator2ndOrder(Q,R,dx,sigma,direction,x):
    if direction>0:
        if abs(sigma)<1E-14:
            return np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))
        elif abs(sigma)>=1E-14:
            return np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))+dx*np.transpose(Q).dot(x).dot(np.conj(Q))-sigma*x

    if direction<0:
        if abs(sigma)<1E-14:
            return Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q))
        elif abs(sigma)>=1E-14:
            return Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))+dx*Q.dot(x).dot(herm(Q))-sigma*x


#returns the derivative of x (matrix); Q and R should be normalized according to dx
#to be consistent with the other MPS code, matrix is a bond matrix where the index 0 lives on the UPPER bond
#and 1 on the lower, hence the very weird tensor multiplications below
def matrixtransferOperator(Q,R,sigma,direction,x):
    if direction>0:
        if abs(sigma)<1E-14:
            return np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))
        elif abs(sigma)>=1E-14:
            return np.transpose(Q).dot(x)+x.dot(np.conj(Q))+np.transpose(R).dot(x).dot(np.conj(R))-sigma*x

    if direction<0:
        if abs(sigma)<1E-14:
            return Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))
        elif abs(sigma)>=1E-14:
            return Q.dot(x)+x.dot(herm(Q))+R.dot(x).dot(herm(R))-sigma*x



#takes a density matrix and evolves it by deltax; the cMPS matrices should be normalized according to dx
def evolveDiagDensityMatrix(matrix,Q,R,deltax,dx,direction):
    N=len(Q)
    D=np.shape(Q)[0]
    vector=np.reshape(matrix,D*D)
    if direction > 0:
        deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        deriv=np.diag(np.diag(deriv))
        return np.reshape(vector,(D,D)) + deltax*deriv
    if direction < 0:
        deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        deriv=np.diag(np.diag(deriv))
        #deriv[np.abs(deriv)<1E-8]=0.0
        return np.reshape(vector,(D,D)) + deltax*deriv


#pass density matrix as a vector
#this routine uses 2nd order transfer operation to evolve the density matrix; it is
#not compatible with the single layer evolution picture
def evolveDensityMatrix(vector,Q,R,deltax,dx,direction):
    N=len(Q)
    D=np.shape(Q)[0]
    #vector=np.reshape(matrix,D*D)
    if direction > 0:
        #deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        #return np.reshape(vector,(D,D)) + deltax*deriv
        deriv=transferOperator2ndOrder(Q,R,dx,0.0,direction,vector)
        return vector + deltax*deriv

    if direction < 0:
        #deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        #return np.reshape(vector,(D,D)) + deltax*deriv
        deriv=transferOperator2ndOrder(Q,R,dx,0.0,direction,vector)
        return vector + deltax*deriv


#pass density matrix as a matrix
#this routine uses 2nd order transfer operation to evolve the density matrix; it is
#not compatible with the single layer evolution picture
def evolveDensityMatrixMatrix(matrix,Q,R,deltax,dx,direction):
    N=len(Q)
    D=np.shape(Q)[0]
    #vector=np.reshape(matrix,D*D)
    if direction > 0:
        #deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        #return np.reshape(vector,(D,D)) + deltax*deriv
        deriv=matrixtransferOperator2ndOrder(Q,R,dx,0.0,direction,matrix)
        return matrix + deltax*deriv

    if direction < 0:
        #deriv=np.reshape(transferOperator2ndOrder(Q,R,dx,0.0,direction,vector),(D,D))
        #return np.reshape(vector,(D,D)) + deltax*deriv
        deriv=matrixtransferOperator2ndOrder(Q,R,dx,0.0,direction,matrix)
        return matrix + deltax*deriv



#pass density matrix as a vector
#this routine uses a cQR decomposition to evolve the density matrix in the double layer picture. it is compatible
#with the single layer evolution
def evolveDensityMatrixcQR(vector,Q,R,deltax,dx,direction):
    N=len(Q)
    D=np.shape(Q)[0]
    mat=np.reshape(vector,(D,D))
    if direction > 0:
        
        #note the tranpose before eigh: it's due to the fact
        #I have my 0-index of the reduced density matrix on the upper bond

        eta,U=np.linalg.eigh(np.transpose(mat))
        C=herm(U.dot(np.diag(np.sqrt(eta+0.0j))))

        #conjugate Q and R by C
        Qt=C.dot(Q).dot(np.linalg.inv(C))
        Rt=C.dot(R).dot(np.linalg.inv(C))

        #normalize the matrices Qt and Rt:
        Qn,Rn,deriv,conv=cqr.cQR(Qt,Rt,dx)
        temp=(np.eye(D)+deltax*deriv).dot(C)
        #derivative=herm(C).dot(deriv).dot(C)+herm(C).dot(herm(deriv)).dot(C)+dx*herm(C).dot(herm(deriv)).dot(deriv).dot(C)
        
        out=np.transpose(herm(temp).dot(temp))
        #derivative=(out-mat)/dx
        #return derivative
        Z=np.trace(out)
        #return out/np.sqrt(Z)
        return np.reshape(out/Z,D*D)

    if direction < 0:
        eta,U=np.linalg.eigh(mat)
        C=U.dot(np.diag(np.sqrt(eta+0.0j)))

        #conjugate Q and R by C
        Qt=np.linalg.inv(C).dot(Q).dot(C)
        Rt=np.linalg.inv(C).dot(R).dot(C)


        #normalize the matprices Qt and Rt:
        Qt,Rt,deriv,conv=cqr.cQR(herm(Qt),herm(Rt),dx)
        temp=C.dot(np.eye(D)+deltax*herm(deriv))
        out=temp.dot(herm(temp))
        Z=np.trace(out)
        #return out/np.sqrt(Z)
        return np.reshape(out/Z,D*D)


#This routine does a single-layer search for the dominant eigenvector of the transfer matrix
def singleLayerTMeigs(Q,R,dx,deltax,direction,nmax,tol,lam0,verbosity=0):
    D=np.shape(Q)[0]
    converged=False
    lam=np.copy(lam0)
    it=0
    if direction>0:
        while not converged:
            Qt=lam.dot(Q).dot(np.linalg.inv(lam))
            Rt=lam.dot(R).dot(np.linalg.inv(lam))            
            Qn,Rn,deriv,conv=cqr.cQR(Qt,Rt,dx)
            if conv==False:
                print('rejected deltax in singleLayerTMeigs')
                return None,None,False,False
            lamnew=lam+deltax*deriv.dot(lam)
            Z=np.linalg.norm(lamnew)
            lamnew=lamnew/Z
            if verbosity>0:
                stdout.write("\r %.10f" % np.linalg.norm(lamnew-lam))
                stdout.flush()
            if np.linalg.norm(lamnew-lam)<tol:
                converged=True
            elif np.linalg.norm(lamnew-lam)>=tol:
                lam=np.copy(lamnew)
            if it>nmax:
                break
            it=it+1
        
        return Z,lamnew,converged,True
    if direction<0:
        while not converged:
            Qt=np.linalg.inv(lam).dot(Q).dot(lam)
            Rt=np.linalg.inv(lam).dot(R).dot(lam)
            Qn,Rn,deriv,conv=cqr.cQR(herm(Qt),herm(Rt),dx)
            if conv==False:
                print('rejected deltax in singleLayerTMeigs')
                return None,None,False,False
            
            lamnew=lam+deltax*lam.dot(herm(deriv))
            Z=np.linalg.norm(lamnew)
            
            lamnew=lamnew/Z
            if verbosity>0:
                stdout.write("\r %.10f" % np.linalg.norm(lamnew-lam))
                stdout.flush()
                
            if np.linalg.norm(lamnew-lam)<tol:
                converged=True                
            elif np.linalg.norm(lamnew-lam)>=tol:
                lam=np.copy(lamnew)
            if it>nmax:
                break                
            it=it+1
        return Z,lamnew,converged,True
            
    

#regauging procedure for homogeneous cMPS as given by Q,R matrices, using single-layer regauging (potentially more accurate than double layer)
def singleLayerRegauge(Q,R,dx,deltax,linitial=None,rinitial=None,nmaxit=100000,tol=1E-10,numeig=4,ncv=50,trunc=1E-16,noise=1E-10,verbosity=0,skipinit=False):
    deltax_=deltax
    D=Q.shape[0]
    dtype=Q.dtype    
    if not skipinit:
        
        [chi ,chi2]=np.shape(Q)
        [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,1,numeig=numeig,init=np.reshape(linitial,D*D),nmax=nmaxit,tolerance=1E-6,ncv=ncv,which='LR')
        l=np.reshape(v,(chi,chi))
        l=l/np.trace(l)
        l=(l+herm(l))/2.0
        lam,u=np.linalg.eigh(l)
        lam=np.abs(lam)
        lam=lam/np.sum(lam)
        if dtype==complex:
            yinit=np.transpose(u.dot(np.diag(np.sqrt(lam))))+(np.random.random_sample((len(lam),len(lam)))-0.5+1j*(np.random.random_sample((len(lam),len(lam)))-0.5))*noise
        if dtype==float:
            yinit=np.transpose(u.dot(np.diag(np.sqrt(lam))))+(np.random.random_sample((len(lam),len(lam)))-0.5)*noise
    else:
        if dtype==complex:
            yinit=linitial+(np.random.random_sample((D,D))-0.5+1j*(np.random.random_sample((D,D))-0.5))*noise
        if dtype==float:
            yinit=linitial+(np.random.random_sample((D,D))-0.5)*noise

    etay,y,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=1,nmax=nmaxit,tol=tol,lam0=yinit,verbosity=verbosity)
    while info==False:
        deltax_=deltax_/2.0
        etay,y,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=1,nmax=nmaxit,tol=tol,lam0=yinit,verbosity=verbosity)        
    if not skipinit:
        [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,-1,numeig=numeig,init=np.reshape(rinitial,D*D),nmax=nmaxit,tolerance=1E-6,ncv=ncv,which='LR')
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        r=(r+herm(r))/2.0
        lam,u=np.linalg.eigh(r)
        lam=np.abs(lam)
        lam=lam/np.sum(lam)
        if dtype==complex:    
            xinit=np.dot(u,np.diag(np.sqrt(lam)))+(np.random.random_sample((len(lam),len(lam)))-0.5+1j*(np.random.random_sample((len(lam),len(lam)))-0.5))*noise
        if dtype==float:    
            xinit=np.dot(u,np.diag(np.sqrt(lam)))+(np.random.random_sample((len(lam),len(lam)))-0.5)*noise
    else:
        if dtype==complex:    
            xinit=rinitial+(np.random.random_sample((D,D))-0.5+1j*(np.random.random_sample((D,D))-0.5))*noise
        if dtype==float:    
            xinit=rinitial+(np.random.random_sample((D,D))-0.5)*noise
        
        

    deltax_=deltax
    etax,x,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=-1,nmax=nmaxit,tol=tol,lam0=xinit,verbosity=verbosity)
    while info==False:
        deltax_=deltax_/2.0
        etax,x,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=-1,nmax=nmaxit,tol=tol,lam0=xinit,verbosity=verbosity)        

    
    invy=np.linalg.inv(y)
    invx=np.linalg.inv(x)


    [U,lam,V]=svd(y.dot(x))        
    Z=np.linalg.norm(lam)
    lam=lam/Z
    if trunc>1E-15:
        lam=lam[lam>=trunc]
        U=U[:,0:len(lam)]
        V=V[0:len(lam),:]
        Z=np.linalg.norm(lam)
        chi=len(lam)
        lam=lam/Z

    #lam=lam[lam>trunc]
    #U=U[:,0:len(lam)]
    #V=V[0:len(lam),:]
    #Z=np.linalg.norm(lam)

    Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
    Rl=Z*np.diag(lam).dot(V).dot(invx).dot(R).dot(invy).dot(U)

    Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
    Rr=Z*V.dot(invx).dot(R).dot(invy).dot(U).dot(np.diag(lam))

    #normalize the state:
    eta=(herm(Ql)+Ql+herm(Rl).dot(Rl)+dx*herm(Ql).dot(Ql))[0,0]

    if dx>=1E-8:
        phi_=1.0/np.sqrt(1+dx*eta)-1.0
        Ql=Ql+phi_/dx*np.eye(chi)+phi_*Ql
        Rl=Rl*(1.0+phi_)

        Qr=Qr+phi_/dx*np.eye(chi)+phi_*Qr
        Rr=Rr*(1.0+phi_)
            
    if dx<1E-8 and dx>1E-12:
        Ql=Ql+phiovereps(dx,eta)*np.eye(chi)+phi(dx,eta)*Ql
        Rl=Rl*(1.0+phi(dx,eta))

        Qr=Qr+phiovereps(dx,eta)*np.eye(chi)+phi(dx,eta)*Qr
        Rr=Rr*(1.0+phi(dx,eta))
    if dx<=1E-12:
        Ql=Ql-eta/2.0*np.eye(chi)
        Qr=Qr-eta/2.0*np.eye(chi)

    return lam,Ql,Rl,Qr,Rr
    

#regauging procedure for homogeneous cMPS as given by Q,R matrices, using single-layer regauging (potentially more accurate than double layer)
def singleLayerRegauge_return_basis(Q,R,dx,deltax=0.01,initial=None,nmaxit=100000,tol=1E-10,numeig=4,ncv=50,trunc=1E-10,noise=1E-2):
    dtype=Q.dtype
    deltax_=deltax
    [chi ,chi2]=np.shape(Q)
    [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
    l=np.reshape(v,(chi,chi))
    l=l/np.trace(l)
    l=(l+herm(l))/2.0
    lam,u=np.linalg.eigh(l)
    lam=np.abs(lam)
    lam=lam/np.sum(lam)
    if dtype==complex:
        yinit=np.transpose(u.dot(np.diag(np.sqrt(lam))))+(np.random.random_sample((len(lam),len(lam)))-0.5+1j*(np.random.random_sample((len(lam),len(lam)))-0.5))*noise
    if dtype==float:
        yinit=np.transpose(u.dot(np.diag(np.sqrt(lam))))+(np.random.random_sample((len(lam),len(lam)))-0.5)*noise
        
    etay,y,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=1,nmax=nmaxit,tol=tol,lam0=yinit)
    while info==False:
        deltax_=deltax_/2.0
        etay,y,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=1,nmax=nmaxit,tol=tol,lam0=yinit)        




    [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
    r=np.reshape(v,(chi,chi))
    r=r/np.trace(r)
    r=(r+herm(r))/2.0
    lam,u=np.linalg.eigh(r)
    lam=np.abs(lam)
    lam=lam/np.sum(lam)

    if dtype==complex:    
        xinit=np.dot(u,np.diag(np.sqrt(lam)))+(np.random.random_sample((len(lam),len(lam)))-0.5+1j*(np.random.random_sample((len(lam),len(lam)))-0.5))*noise
    if dtype==float:    
        xinit=np.dot(u,np.diag(np.sqrt(lam)))+(np.random.random_sample((len(lam),len(lam)))-0.5)*noise
    deltax_=deltax
    etax,x,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=-1,nmax=nmaxit,tol=tol,lam0=xinit)
    while info==False:
        deltax_=deltax_/2.0
        etax,x,conv,info=singleLayerTMeigs(np.copy(Q),np.copy(R),dx,deltax_,direction=-1,nmax=nmaxit,tol=tol,lam0=xinit)        



    invy=np.linalg.pinv(y)
    invx=np.linalg.pinv(x)
    
    [U,lam,V]=svd(y.dot(x))        
    Z=np.linalg.norm(lam)
    lam=lam/Z
    if trunc>1E-15:
        lam=lam[lam>=trunc]
        U=U[:,0:len(lam)]
        V=V[0:len(lam),:]
        Z=np.linalg.norm(lam)
        chi=len(lam)
        lam=lam/Z
    
    #lam=lam[lam>trunc]
    #U=U[:,0:len(lam)]
    #V=V[0:len(lam),:]
    #Z=np.linalg.norm(lam)
    #lam=lam/Z
    
    Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
    Rl=Z*np.diag(lam).dot(V).dot(invx).dot(R).dot(invy).dot(U)

    Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
    Rr=Z*V.dot(invx).dot(R).dot(invy).dot(U).dot(np.diag(lam))



    #normalize the state:
    eta=(herm(Ql)+Ql+herm(Rl).dot(Rl)+dx*herm(Ql).dot(Ql))[0,0]

    if dx>=1E-8:
        phi_=1.0/np.sqrt(1+dx*eta)-1.0
        Ql=Ql+phi_/dx*np.eye(chi)+phi_*Ql
        Rl=Rl*(1.0+phi_)

        Qr=Qr+phi_/dx*np.eye(chi)+phi_*Qr
        Rr=Rr*(1.0+phi_)
            
    if dx<1E-8 and dx>1E-12:
        Ql=Ql+phiovereps(dx,eta)*np.eye(chi)+phi(dx,eta)*Ql
        Rl=Rl*(1.0+phi(dx,eta))

        Qr=Qr+phiovereps(dx,eta)*np.eye(chi)+phi(dx,eta)*Qr
        Rr=Rr*(1.0+phi(dx,eta))
    if dx<=1E-12:
        Ql=Ql-eta/2.0*np.eye(chi)
        Qr=Qr-eta/2.0*np.eye(chi)

    Gl=Z*np.diag(lam).dot(V).dot(invx)
    Glinv=invy.dot(U)
    
    Gr=Z*V.dot(invx)
    Grinv=invy.dot(U).dot(np.diag(lam))
    
    return lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv
    


#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs2ndOrder(Q,R,dx,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    n=[0]
    mv=fct.partial(transferOperator2ndOrder,*[Q,R,dx,0.0,direction])
    #mv=partial.partial(transferOperator2ndOrder,*[Q,R,dx,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        #m=np.argmin(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-6:
            #numeig=numeig+1
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
            numeig=7
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
            
        return eta[m],vec[:,m],numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs2ndOrder(Q,R,dx,direction,numeig,np.random.rand(D*D),nmax,tolerance,ncv,which)




#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs1stOrder(Q,R,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    mv=fct.partial(transferOperator1stOrder,*[Q,R,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-6:
            #numeig=numeig+1
            numeig=7
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],vec[:,m],numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs1stOrder(Q,R,direction,numeig,np.random.rand(D*D),nmax,tolerance,ncv,which)



def TMeigstest(Q,R,direction,numeig,sigma,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    mv=fct.partial(transferOperator1stOrder,*[Q,R,sigma,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-6:
            #numeig=numeig+1
            numeig=7
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],np.reshape(vec[:,m],D*D),numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigstest(Q,R,direction,numeig,sigma,np.random.rand(D*D),nmax,tolerance,ncv,which)



#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs2ndOrderCNT(Q,R,dx,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    n=[0]
    #mv=partial.partial(transferOperator2ndOrder,*[Q,R,dx,0.0,direction])
    mv=fct.partial(transferOperator2ndOrderCNT,*[n,Q,R,dx,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-6:
            #numeig=numeig+1
            #print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
            numeig=7
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],np.reshape(vec[:,m],D*D),numeig,n[0]

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs2ndOrderCNT(Q,R,dx,direction,numeig,np.random.rand(D*D),nmax,tolerance,ncv,which)


    #return eigs(LOP,k=numeig, which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)


#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs1stOrderCNT(Q,R,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    n=[0]
    mv=fct.partial(transferOperator1stOrderCNT,*[n,Q,R,0.0,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:

        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        #print np.linalg.norm(mv(vec[:,m])-eta[m]*vec[:,m]),

        while np.abs(np.imag(eta[m]))>1E-6:
            #numeig=numeig+1
            #print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
            
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and LR'.format(eta))
            numeig=10
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='LR',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],np.reshape(vec[:,m],D*D),numeig,n[0]

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs1stOrderCNT(Q,R,direction,numeig,np.random.rand(D*D),nmax,tolerance,ncv,which)


    #return eigs(LOP,k=numeig, which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)

#computes the smallest magnitude left or right eigenvalue-eigenvector pairs of the (UNSHIFTED sigma=0.0) transfer matrix T using Arpack arnoldi method
def TMeigs2ndOrderMultiSpecies(Q,R,dx,direction,numeig,init=None,nmax=100000,tolerance=1e-10,ncv=100,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    D=np.shape(Q)[0]
    mv=fct.partial(transferOperator2ndOrderMultiSpecies,*[Q,R,dx,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,rmatvec=None,matmat=None,dtype=Q.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))>1E-4:
            #numeig=numeig+1
            #print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state and SM'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='SM',v0=np.random.rand(D*D),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))
        return eta[m],np.reshape(vec[:,m],D*D),numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs2ndOrderMultiSpecies(Q,R,dx,direction,numeig,np.random.rand(D*D),nmax,tolerance,ncv,which)




def mixedTMeigs1stOrder(Qu,Ru,Ql,Rl,direction,numeig,init=None,nmax=10000000,tolerance=1e-8,ncv=40,which='LR'):
    #define the matrix vector product mv(v) using functools.partial
    Du=np.shape(Qu)[0]
    Dl=np.shape(Ql)[0]
    mv=fct.partial(mixedTransferOperator,*[Qu,Ru,Ql,Rl,direction,0.0])
    LOP=LinearOperator((Du*Dl,Du*Dl),matvec=mv,rmatvec=None,matmat=None,dtype=Qu.dtype)
    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        return eta[m],vec[:,m]
    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return mixedTMeigs1stOrder(Qu,Ru,Ql,Rl,direction,numeig,init=np.random.rand(Du*Dl).astype(Qu.dtype),\
                                   nmax=nmax,tolerance=tolerance,ncv=ncv,which=which)
    #return eta,vec
    #return eigs(LOP,k=numeig, which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)




#computes the action of the pseudo transfer operator T^P=T-|r)(l|:
#
#   (x| [ T-|r)(l| ]
#
#for the (UNSHIFTED) transfer operator T, with (l| and |r) the left and right eigenvectors of T to eigenvalue 0
def pseudotransferOperator2ndOrderMultiSpecies(Q,R,dx,l,r,direction,vector):
    D=np.shape(Q)[1]
    x=np.reshape(vector,(D,D))
    if direction >0:
        return transferOperator2ndOrderMultiSpecies(Q,R,dx,direction,vector)-np.trace(np.transpose(x).dot(r))*np.reshape(l,(D*D))
    if direction <0:
        return transferOperator2ndOrderMultiSpecies(Q,R,dx,direction,vector)-np.trace(np.transpose(l).dot(x))*np.reshape(r,(D*D))


def inverseTransferOperatorMultiSpecies(Q,R,dx,l,r,ih,direction,x0=None,tolerance=1e-12,maxiteration=4000):
    D=np.shape(Q)[1]
    mv=fct.partial(pseudotransferOperator2ndOrderMultiSpecies,*[Q,R,dx,l,r,direction])
    LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=Q.dtype)
    [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0,tol=tolerance,maxiter=maxiteration,outer_k=6)
    while info<0:
        [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0=np.random.rand(D*D),tol=tolerance,maxiter=maxiteration,outer_k=6)
    return np.reshape(x,(D,D))



#computes the action of the pseudo transfer operator T^P=T-|r)(l|:
#
#   (x| [ T-|r)(l| ]
#
#for the (UNSHIFTED) transfer operator T, with (l| and |r) the left and right eigenvectors of T to eigenvalue 0
def pseudotransferOperator2ndOrderCNT(n,Q,R,dx,l,r,direction,sigma,vector):
    D=np.shape(Q)[1]
    x=np.reshape(vector,(D,D))
    n[0]+=1
    if direction >0:
        return transferOperator2ndOrder(Q,R,dx,sigma,direction,vector)-np.trace(np.transpose(x).dot(r))*np.reshape(l,(D*D))
    if direction <0:
        return transferOperator2ndOrder(Q,R,dx,sigma,direction,vector)-np.trace(np.transpose(l).dot(x))*np.reshape(r,(D*D))



#computes the action of the pseudo transfer operator T^P=T-|r)(l|:
#
#   (x| [ T-|r)(l| ]
#
#for the (UNSHIFTED) transfer operator T, with (l| and |r) the left and right eigenvectors of T to eigenvalue 0
def pseudotransferOperator1stOrderCNT(n,Q,R,l,r,direction,sigma,vector):
    D=np.shape(Q)[1]
    x=np.reshape(vector,(D,D))
    n[0]+=1
    if direction >0:
        return transferOperator1stOrder(Q,R,sigma,direction,vector)-np.trace(np.transpose(x).dot(r))*np.reshape(l,(D*D))
    if direction <0:
        return transferOperator1stOrder(Q,R,sigma,direction,vector)-np.trace(np.transpose(l).dot(x))*np.reshape(r,(D*D))




"""
computes :
direction>0: (x| := (ih|1/(T-eta) 
direction<0: |x) := 1/(T-eta)|ih)
for T=Q\otimes 1+1\otimes Qbar+R\otimes Rbar
uses lgmres to invert the equation
if sigma<1E-8, uses the pseudo-inverse 1/(Tp) with eta=0.0
"""

def inverseTransferOperator(Q,R,dx,l,r,ih,direction,x0=None,tolerance=1e-12,maxiteration=4000,inner_m=30,outer_k=20,sigma=0.0):
    D=np.shape(Q)[1]
    n=[0]
    if dx>=1E-8:
        if np.abs(sigma)<1E-2:
            mv=fct.partial(pseudotransferOperator2ndOrderCNT,*[n,Q,R,dx,l,r,direction,0.0])
        else:
            mv=fct.partial(transferOperator2ndOrderCNT,*[n,Q,R,dx,sigma,direction])
    elif dx<1E-8:
        if np.abs(sigma)<1E-2:
            mv=fct.partial(pseudotransferOperator1stOrderCNT,*[n,Q,R,l,r,direction,0.0])
        else:
            mv=fct.partial(transferOperator1stOrderCNT,*[n,Q,R,sigma,direction])

    LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=Q.dtype)
    [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0,tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)
    while info<0:
        [x,info]=lgmres(LOP,np.reshape(ih,D*D),x0=np.random.rand(D*D),tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)
    #print ('did {0} steps in lgmres with dir={1}'.format(n[0],direction))
    return np.reshape(x,(D,D)),n[0]


def mixedTransferOperator(Qu,Ru,Ql,Rl,direction,sigma,vector):
    D1=np.shape(Qu)[0]
    D2=np.shape(Ql)[0]
    x=np.reshape(vector,(D1,D2))
    if direction>0:
        if abs(sigma)<1E-10:
            return np.reshape(np.transpose(Qu).dot(x)+x.dot(np.conj(Ql))+np.transpose(Ru).dot(x).dot(np.conj(Rl)),(D1*D2))
        elif abs(sigma)>=1E-10:
            return np.reshape(np.transpose(Qu).dot(x)+x.dot(np.conj(Ql))+np.transpose(Ru).dot(x).dot(np.conj(Rl))+sigma*x,(D1*D2))

    if direction<0:
        if abs(sigma)<1E-10:
            return np.reshape(Qu.dot(x)+x.dot(herm(Ql))+Ru.dot(x).dot(herm(Rl)),(D1*D2))
        elif abs(sigma)>=1E-10:
            return np.reshape(Qu.dot(x)+x.dot(herm(Ql))+Ru.dot(x).dot(herm(Rl))+sigma*x,(D1*D2))
        
#computes |x) = (T+sigma*1)^(-1,P)|ih) or (x| = (ih|(T+sigma*1)^(-1,P) for a given vector ih (in matrix form). The 
#"P" in the notation refers to the use of a pseudo-inverse. the only case where one needs to be careful when taking 
#the inverse is for sigma=0.0 and an un-mixed transfer operator, i.e. where Q1=Q2,R1=R2. In all other cases 
#there is no problem inverting it.
def inverseMixedTransferOperator(ih,Qu,Ru,Ql,Rl,sigma,direction,l=None,r=None,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    D1=np.shape(Qu)[1]
    D2=np.shape(Ql)[1]
    assert(Qu.dtype==Ql.dtype)
    assert(Qu.dtype==Ru.dtype)
    assert(Qu.dtype==Rl.dtype)
    if np.linalg.norm(Qu-Ql)<thresh and np.linalg.norm(Ru-Rl)<thresh and abs(sigma)<thresh:
        #print 'using pseudoinverse'
        #pseudoinv=True
        #warnings.warn("using pseudo-inversion of the transfer operator; make certain that Qu==Ql, Ru==Rl")
        if l==None or r==None:
            sys.exit('inverseMixedTransferOperator: no l and r are provided for pseudo-inversion')
        mv=fct.partial(pseudotransferOperator1stOrderCNT,*[[0],Qu,Ru,l,r,direction,0.0])
        if direction>0:
            if abs(np.tensordot(ih,r,([0,1],[0,1])))>thresh:
                warnings.warn('in inverseMixedTransferOperator: the inhomogeinity vector has support in the kernel of the transfer operator T; reducing it to the orthogonal part of the kernel of T')
                inhom=ih-np.tensordot(ih,r,([0,1],[0,1]))*l
                ih-=np.tensordot(ih,r,([0,1],[0,1]))*l
            else:
                inhom=ih
        if direction<0:
            if abs(np.tensordot(l,ih,([0,1],[0,1])))>thresh:
                warnings.warn('in inverseMixedTransferOperator: the inhomogeinity vector has support in the kernel of the transfer operator T; reducing it to the orthogonal part of the kernel of T')
                inhom=ih-np.tensordot(l,ih,([0,1],[0,1]))*r
                ih-=np.tensordot(l,ih,([0,1],[0,1]))*r
            else:
                inhom=ih 

#    elif np.linalg.norm(Qu-Ql)<thresh and np.linalg.norm(Ru-Rl)<thresh and abs(sigma)>thresh:
#        print'hello2'
#        #print 'using pseudoinverse'
#        #pseudoinv=True
#        #warnings.warn("using pseudo-inversion of the transfer operator; make certain that Qu==Ql, Ru==Rl")
#        if l==None or r==None:
#            sys.exit('inverseMixedTransferOperator: no l and r are provided for pseudo-inversion')
#        mv=fct.partial(pseudotransferOperator1stOrderCNT,*[[0],Qu,Ru,l,r,direction,sigma])
#        if direction>0:
#            if abs(np.tensordot(ih,r,([0,1],[0,1])))>thresh:
#                print 'aha'
#                warnings.warn('in inverseMixedTransferOperator: the inhomogeinity vector has support in the kernel of the transfer operator T; reducing it to the orthogonal part of the kernel of T')
#                inhom=ih-np.tensordot(ih,r,([0,1],[0,1]))*l
#                ih-=np.tensordot(ih,r,([0,1],[0,1]))*l
#                
#            else:
#                inhom=ih
#        if direction<0:
#            if abs(np.tensordot(l,ih,([0,1],[0,1])))>thresh:
#                warnings.warn('in inverseMixedTransferOperator: the inhomogeinity vector has support in the kernel of the transfer operator T; reducing it to the orthogonal part of the kernel of T')
#                inhom=ih-np.tensordot(l,ih,([0,1],[0,1]))*r
#                ih-=np.tensordot(l,ih,([0,1],[0,1]))*r
#                
#            else:
#                inhom=ih 
#
    else:

        mv=fct.partial(mixedTransferOperator,*[Qu,Ru,Ql,Rl,direction,sigma])
        inhom=ih

    LOP=LinearOperator((D1*D2,D1*D2),matvec=mv,dtype=Qu.dtype)
    [x,info]=lgmres(LOP,np.reshape(inhom,D1*D2),x0,tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)
    while info<0:
        [x,info]=lgmres(LOP,np.reshape(inhom,D1*D2),x0=np.random.rand(D1*D2),tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)

    return np.reshape(x,(D1,D2))




#takes an mps tensor 'tensor' and fixes its gauge; 'gauge' can be 'left','right' or 'symetric';
#returns a mps matrix and a matrix: [out,lam]; for gauge = 'left' or 'right', out is either left or right orthonormal and lam is the identity; 
#for gauge='symetric', out is the "gamma" matrix and lam contains the schmidt values

#taylor expansion of 1/sqrt(1+dx*eta)-1 and (1/sqrt(1+dx*eta)-1)/dx
phi=lambda dx,eta: -1./2. *(dx*eta) + 3./8. *(dx*eta)**2 - 5./16. *(dx*eta)**3 + 35./128. *(dx*eta)**4 - 63./256. *(dx*eta)**5 + 231./1024. *(dx*eta)**6 - 429./2048. *(dx*eta)**7 + 6435./32768. *(dx*eta)**8 - 12155./65536. *(dx*eta)**9 + 46189./262144. *(dx*eta)**10 \
                    - 88179./524288. *(dx*eta)**11 + 676039./4194304. *(dx*eta)**12
phiovereps=lambda dx,eta:  -1./2.*eta + 3./8. *dx*eta**2 - 5./16. *dx**2*eta**3 + 35./128. *dx**3*eta**4 - 63./256. *dx**4*eta**5 + 231./1024. *dx**5*eta**6 - 429./2048. *dx**6*eta**7 + 6435./32768. *dx**7*eta**8 - 12155./65536. *dx**8*eta**9 + \
            46189./262144. *dx**9*eta**10 - 88179./524288. *dx**10*eta**11 + 676039./4194304. *dx**11*eta**12

        
#def fixPhase(matrix):
#    bla=matrix/np.trace(matrix);
#    Z=np.sqrt(np.trace(bla.dot(np.conj(np.transpose(bla)))));
#    return bla/Z

#takes two mps matrices A,B, a density matrix x (can be either left or
#right eigenmatrix of the Transfer Matrix), a boundary mpo of either left
#or right type and a direction argument dir > or < 0; if dir > 0, x is
#assumed to be a left eigenvector of the TM, if dir<0, then it's assumed to
#be a right eigenvector. returns the contraction of the objects, which is a
#three index object (l,l',b), where l and l' are upper and lower A and B
#matrix indices, and b is the left or right auxiliary index of the mpo


def homogeneousdfdxLiebLinigerTwoBosonSpecies(Q,R,dx,f,mu,mass,intrag,interg,direction,penalty):
    D=np.shape(Q)[0]
    dtype=type(Q[0,0])
    if direction>0:
        K=[]
        I=[]
        for n in range(len(R)):
            K.append(Q.dot(R[n])-R[n].dot(Q))
            I.append(R[n].dot(R[n]))
        II=(R[0].dot(R[1])+R[1].dot(R[0]))/2.0
        commR1R2=R[0].dot(R[1])-R[1].dot(R[0])
        out=np.zeros((D,D),dtype=dtype)
        for n in range(len(R)):
            out=out+1.0/(2.0*mass[n])*np.transpose(herm(K[n]).dot(K[n]))+intrag[n]*np.transpose(herm(I[n]).dot(I[n]))+\
                 mu[n]*np.transpose(herm(R[n]).dot(R[n]))
        out+=interg*np.transpose(herm(II).dot(II))
        out+=penalty*np.transpose(herm(commR1R2).dot(commR1R2))
        return matrixtransferOperator2ndOrderMultiSpecies(Q,R,dx,direction,f)+out
        
    if direction<0:
        K=[]
        I=[]
        for n in range(len(R)):
            K.append(Q.dot(R[n])-R[n].dot(Q))
            I.append(R[n].dot(R[n]))
        II=(R[0].dot(R[1])+R[1].dot(R[0]))/2.0
        commR1R2=R[0].dot(R[1])-R[1].dot(R[0])
        out=np.zeros((D,D),dtype=dtype)
        for n in range(len(R)):
            out=out+1.0/(2.0*mass[n])*K[n].dot(herm(K[n]))+intrag[n]*I[n].dot(herm(I[n]))+\
                 mu[n]*R[n].dot(herm(R[n]))
        out+=interg*II.dot(herm(II))
        out+=penalty*commR1R2.dot(herm(commR1R2))
        return matrixtransferOperator2ndOrderMultiSpecies(Q,R,dx,direction,f)+out



#pass "f" as a matrix
#note that the index "0" of the density-like matrix lives on the UPPER bond (hence the multiple transpose operations below)
#R is a list containing the R-matrices of the fermionic cMPS
def homogeneousdfdxLiebLinigerTwoFermionSpecies(Q,R,dx,f,mu,mass,intrag,direction,penalty):
    D=np.shape(Q)[0]
    dtype=type(Q[0,0])
    if direction>0:
        K=[]
        I=[]
        for n in range(len(R)):
            K.append(Q.dot(R[n])-R[n].dot(Q))
            I.append(R[n].dot(R[n]))
        anticommR1R2=R[0].dot(R[1])+R[1].dot(R[0])
        out=np.zeros((D,D),dtype=dtype)
        for n in range(len(R)):
            out=out+1.0/(2.0*mass[n])*np.transpose(herm(K[n]).dot(K[n]))+intrag[n]*np.transpose(herm(I[n]).dot(I[n]))+\
                 mu[n]*np.transpose(herm(R[n]).dot(R[n]))
        out+=penalty*np.transpose(herm(anticommR1R2).dot(anticommR1R2))
        return matrixtransferOperator2ndOrderMultiSpecies(Q,R,dx,direction,f)+out
        
    if direction<0:
        K=[]
        I=[]
        for n in range(len(R)):
            K.append(Q.dot(R[n])-R[n].dot(Q))
            I.append(R[n].dot(R[n]))
        anticommR1R2=R[0].dot(R[1])+R[1].dot(R[0])
        out=np.zeros((D,D),dtype=dtype)
        for n in range(len(R)):
            out=out+1.0/(2.0*mass[n])*K[n].dot(herm(K[n]))+intrag[n]*I[n].dot(herm(I[n]))+\
                 mu[n]*R[n].dot(herm(R[n]))
        out+=penalty*anticommR1R2.dot(herm(anticommR1R2))
        return matrixtransferOperator2ndOrderMultiSpecies(Q,R,dx,direction,f)+out
        
#this is the same as homogeneousdfdxLiebLiniger, just that Q and R are assumed translational invariant, i.e. Q1=Q2,R1=R2
#calculates the derivative of f-like expressions 
def homogeneousdfdxLiebLinigernodx(Q,R,f,mu,mass,g,Delta,direction):
    D=np.shape(Q)[0]
    if direction>0:
        K=Q.dot(R)-R.dot(Q)
        I=R.dot(R)
        return matrixtransferOperator(Q,R,0.0,direction,f)+\
            1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
            g*np.transpose(herm(I).dot(I))+Delta*(np.transpose(herm(R).dot(herm(R))+R.dot(R)))+\
            mu*np.transpose(herm(R).dot(R))

    if direction<0:
        K=Q.dot(R)-R.dot(Q)
        I=R.dot(R)
        return matrixtransferOperator(Q,R,0.0,direction,f)+\
            1.0/(2.0*mass)*K.dot(herm(K))+\
            g*I.dot(herm(I))+Delta*(R.dot(R)+herm(R).dot(herm(R)))+\
            mu*R.dot(herm(R))


#here, g is the strength of the extended interaction
#there is no local interaction, thus it's not really Lieb-Liniger type
#phi contains the geometric series (l|R\otimes Rbar (1/T-eta) (possibly with a minus sign)
def homogeneousdfdxExtLiebLinigernodx(Q,R,phi,f,mu,mass,g1,g2,direction):
    D=np.shape(Q)[0]
    if direction>0:
        K=Q.dot(R)-R.dot(Q)
        I=R.dot(R)
        return matrixtransferOperator(Q,R,0.0,direction,f)+\
            1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
            mu*np.transpose(herm(R).dot(R))+\
            g1*np.transpose(herm(I).dot(I))+\
            g2*np.transpose(herm(R).dot(np.transpose(phi)).dot(R))

    if direction<0:
        K=Q.dot(R)-R.dot(Q)
        I=R.dot(R)
        return matrixtransferOperator(Q,R,0.0,direction,f)+\
            1.0/(2.0*mass)*K.dot(herm(K))+\
            mu*R.dot(herm(R))+\
            g1*I.dot(herm(I))+\
            g2*R.dot(phi).dot(herm(R))

#pass "f" as a matrix
#note that the index "0" of the density-like matrix lives on the UPPER bond (hence the multiple transpose operations below)
#"cmps" is an object of the CMPS class, defined in cmpsdisclib or cmpslib.
#"f" is a bond-like matrix living between matrices Q and R. 
#"mu" is a vector of length cmps._N that contains the values of the chemical potential. The routine assumes unit-cell translational invariance, that is mu[cmps._N]=mu[0].
#"mass" and "g" are mass and interaction strenght of the bosons
#"direction" can be >0 or <0. use direction>0 when going from left to right, and direction<0 if going from right to left (i.e. when calculating left or right block 
#hamiltonians).
#"f" is living on bond "ind", and the routine gives back the derivative of "f" calculated from adjacent matrices Q[ind-1], R1 and Q[ind], R2
def homogeneousdfdxLiebLiniger(Q1,R1,Q2,R2,dx,f,mu,mass,g,Delta,direction):
    D=np.shape(Q1)[0]
    if direction>0:
        K=Q1.dot(R2)-R1.dot(Q2)
        I=R1.dot(R2)
        return matrixtransferOperator2ndOrder(Q2,R2,dx,0.0,direction,f)+\
            1.0/(2.0*mass)*np.transpose(herm(K).dot(K))+\
            g*np.transpose(herm(I).dot(I))+Delta*(np.transpose(herm(R2).dot(herm(R1))+R1.dot(R2)))+\
            mu/2.0*np.transpose(herm(np.eye(D)+dx*Q2).dot(herm(R1)).dot(R1).dot(np.eye(D)+dx*Q2))+\
            mu/2.0*np.transpose(herm(np.sqrt(dx)*R2).dot(herm(R1)).dot(R1).dot(np.sqrt(dx)*R2))+\
            mu/2.0*np.transpose(herm(R2).dot(R2))

    if direction<0:
        K=Q1.dot(R2)-R1.dot(Q2)
        I=R1.dot(R2)
        return matrixtransferOperator2ndOrder(Q2,R2,dx,0.0,direction,f)+\
            1.0/(2.0*mass)*K.dot(herm(K))+\
            g*I.dot(herm(I))+Delta*(R1.dot(R2)+herm(R2).dot(herm(R1)))+\
            mu/2.0*(np.eye(D)+dx*Q1).dot(R2).dot(herm(R2)).dot(herm(np.eye(D)+dx*Q1))+\
            mu/2.0*(np.sqrt(dx)*R1).dot(R2).dot(herm(R2)).dot(np.sqrt(dx)*herm(R1))+\
            mu/2.0*R1.dot(herm(R1))




"""
computes the geometries series from the interaction part of the extended Lieb Liniger model, i.e. herm(Rl)*Rl*1/(eta-T_lP) (or the one with R_r), where eta is the length scale of the 
exponential interaction, and T_lP is the pseudo-transferOperator in left form (there's a similar one for the right form)
"""
def geometricSumExtLiebLinigerInteraction(Q,R,l,r,eta,direction,x0=None,tolerance=1e-12,maxiteration=4000,inner_m=30,outer_k=20,debug=False,use_pseudo=False):
    
    #first get the inhomogeneity
    D=np.shape(R)[1]
    dtype=type(R[0,0])
    #n stores the number of iterations that were needed
    n=[0]
    if direction>0:
        #the minus sign is correct, it is needed due to a minus sign in the definition of
        #transferOperator1stOrder=T-sigma eye, when using a non-zero sigma there (see function)
        inhom=(-1.0)*np.transpose(herm(R).dot(R)) 
        #inhomproj=inhom-np.trace(inhom.dot(r))*l

    if direction<0:
        D=np.shape(R)[1]
        #the minus sign is correct, it is needed due to a minus sign in the definition of 
        #transferOperator1stOrder=T-sigma eye, when using a non-zero sigma there (see function)
        inhom=(-1.0)*R.dot(herm(R)) 
        #inhomproj=inhom-np.trace(np.transpose(l).dot(inhom))*r



    if use_pseudo==False:
        mv=fct.partial(transferOperator1stOrderCNT,*[n,Q,R,eta,direction]) 
        #mv=fct.partial(transferOperator2ndOrderCNT,*[n,Q,R,0.0,eta,direction]) 
    elif use_pseudo==True:
        #this is (T_P-eta), where the _P denotes the pseudo-operator        
        mv=fct.partial(pseudotransferOperator1stOrderCNT,*[n,Q,R,l,r,direction,eta])
    else: 
        sys.exit("geometricSumExtLiebLinigerInteraction: use_pseudo: value unrecognized; should be bool")
    LOP=LinearOperator((D*D,D*D),matvec=mv,dtype=dtype)
    [x,info]=lgmres(LOP,np.reshape(inhom,D*D),x0,tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)
    while info<0:
        [x,info]=lgmres(LOP,np.reshape(inhom,D*D),x0=np.random.rand(D*D),tol=tolerance,maxiter=maxiteration,outer_k=outer_k,inner_m=inner_m)

    if debug:
        print (np.linalg.norm(mv(x)-np.reshape(inhom,D*D)))
        input()
    return np.reshape(x,(D,D)),n[0] #n[0] contains the number of iterations of the solver

    #print ('did {0} steps in lgmres with dir={1}'.format(n[0],direction))



#calculates derivative of f-like expressions (reduced Hamiltonian expressions) for the Phi-Four model. Cutoff is the UV cutoff parameter (see notes)
#this routine has an overall factor 1/cutoff as compared to homogeneousdfdxPhiFour1 below
def homogeneousdfdxPhiFour(Q,R,f,mass,inter,cutoff,direction):
    D=np.shape(Q)[0]
    g=inter/(96.0*cutoff**2)
    #Delta=0.0
    if direction>0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            1.0/cutoff*np.transpose(herm(K).dot(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0*cutoff)*(np.transpose(herm(R.dot(R))+R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0*cutoff)*np.transpose(herm(R).dot(R))+\
            1.0*g*np.transpose(herm(I1bar).dot(I1))+\
            4.0*g*np.transpose(herm(I2bar).dot(I2))+\
            6.0*g*np.transpose(herm(I3bar).dot(I3))+\
            4.0*g*np.transpose(herm(I4bar).dot(I4))+\
            1.0*g*np.transpose(herm(I5bar).dot(I5))
            #Delta*(np.transpose(herm(R))+np.transpose(R))
        
    if direction<0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            1.0/cutoff*K.dot(herm(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0*cutoff)*(R.dot(R)+herm(R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0*cutoff)*(R.dot(herm(R)))+\
            1.0*g*I1.dot(herm(I1bar))+\
            4.0*g*I2.dot(herm(I2bar))+\
            6.0*g*I3.dot(herm(I3bar))+\
            4.0*g*I4.dot(herm(I4bar))+\
            1.0*g*I5.dot(herm(I5bar))
            #Delta*(herm(R)+R)





#calculates derivative of f-like expressions (reduced Hamiltonian expressions) for the Phi-Four model. Cutoff is the UV cutoff parameter (see notes)
def homogeneousdfdxPhiFour1(Q,R,f,mass,inter,cutoff,direction):
    D=np.shape(Q)[0]
    g=inter/(96.0*cutoff)

    if direction>0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            np.transpose(herm(K).dot(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0)*(np.transpose(herm(R.dot(R))+R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0)*np.transpose(herm(R).dot(R))+\
            6.0*g*np.transpose(herm(I3bar).dot(I3))+\
            4.0*g*np.transpose(herm(I2bar).dot(I2))+\
            4.0*g*np.transpose(herm(I4bar).dot(I4))+\
            1.0*g*np.transpose(herm(I1bar).dot(I1))+\
            1.0*g*np.transpose(herm(I5bar).dot(I5))
            #Delta*(np.transpose(herm(R))+np.transpose(R))

        
    if direction<0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            K.dot(herm(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0)*(R.dot(R)+herm(R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0)*(R.dot(herm(R)))+\
            6.0*g*I3.dot(herm(I3bar))+\
            4.0*g*I2.dot(herm(I2bar))+\
            4.0*g*I4.dot(herm(I4bar))+\
            1.0*g*I1.dot(herm(I1bar))+\
            1.0*g*I5.dot(herm(I5bar))

            #Delta*(herm(R)+R)

#calculates derivative of f-like expressions (reduced Hamiltonian expressions) for the Phi-Four model. Cutoff is the UV cutoff parameter (see notes)
#It uses a penalty to supress quadrupular occupations 
def homogeneousdfdxPhiFourPenalty(Q,R,f,mass,inter,cutoff,direction,penalty=0.0):
    D=np.shape(Q)[0]
    g=inter/(96.0*cutoff)

    if direction>0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)
        mat=np.copy(R)
        for b in range(3):
            mat=mat.dot(R)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            np.transpose(herm(K).dot(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0)*(np.transpose(herm(R.dot(R))+R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0)*np.transpose(herm(R).dot(R))+\
            6.0*g*np.transpose(herm(I3bar).dot(I3))+\
            1.0*g*np.transpose(herm(I1bar).dot(I1))+\
            1.0*g*np.transpose(herm(I5bar).dot(I5))+\
            4.0*g*np.transpose(herm(I2bar).dot(I2))+\
            4.0*g*np.transpose(herm(I4bar).dot(I4))+\
            penalty*np.transpose(herm(mat).dot(mat))
            #Delta*(np.transpose(herm(R))+np.transpose(R))

        
    if direction<0:
        K=Q.dot(R)-R.dot(Q)
        I1=np.eye(D)
        I1bar=R.dot(R).dot(R).dot(R)
        I2=R
        I2bar=R.dot(R).dot(R)
        I3=R.dot(R)
        I3bar=R.dot(R)
        I4=R.dot(R).dot(R)
        I4bar=R
        I5=R.dot(R).dot(R).dot(R)
        I5bar=np.eye(D)
        mat=np.copy(R)
        for b in range(3):
            mat=mat.dot(R)

        return matrixtransferOperator2ndOrder(Q,R,0.0,0.0,direction,f)+\
            K.dot(herm(K))+\
            (np.real(mass**2)-cutoff**2)/(4.0)*(R.dot(R)+herm(R.dot(R)))+\
            (np.real(mass**2)+cutoff**2)/(2.0)*(R.dot(herm(R)))+\
            6.0*g*I3.dot(herm(I3bar))+\
            1.0*g*I1.dot(herm(I1bar))+\
            1.0*g*I5.dot(herm(I5bar))+\
            4.0*g*I2.dot(herm(I2bar))+\
            4.0*g*I4.dot(herm(I4bar))+\
            penalty*mat.dot(herm(mat))
            #Delta*(herm(R)+R)






#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousPhiFourGradient(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,inter,cutoff):
    #left tangent gauge
    D=np.shape(Ql)[0]
    g=inter/(96.0*cutoff**2)
    #first the kinetic energy term
    #Kin=Ql.dot(R)-Rl.dot(Q)
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    I0=lam
    I1=R
    I2=Rl.dot(R)
    I3=Rl.dot(Rl).dot(R)
    Delta=0.0
    V=1.0/cutoff*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
        np.tensordot(kleft,lam,([0],[0]))+\
        np.tensordot(lam,kright,([1],[0]))

    W=1.0/cutoff*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       (np.real(mass**2)+cutoff**2)/(2.0*cutoff)*R+\
       (np.real(mass**2)-cutoff**2)/(4.0*cutoff)*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
       np.tensordot(kleft,R,([0],[0]))+\
       np.tensordot(R,kright,([1],[0]))+\
       1.0*g*(herm(Rl.dot(Rl).dot(Rl)).dot(I0)+herm(Rl.dot(Rl)).dot(I0).dot(herm(Rr))+herm(Rl).dot(I0).dot(herm(Rr.dot(Rr)))+I0.dot(herm(Rr.dot(Rr).dot(Rr))))+\
       4.0*g*(herm(Rl.dot(Rl)).dot(I1)+herm(Rl).dot(I1).dot(herm(Rr))+I1.dot(herm(Rr.dot(Rr))))+\
       6.0*g*(herm(Rl).dot(I2)+I2.dot(herm(Rr)))+\
       4.0*g*I3
       #Delta*lam
       #the 1.0*g part is missing here because it doesn't contain any herm(Rc)

    return V,W



#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousPhiFourGradient1(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,inter,cutoff):
    #left tangent gauge
    D=np.shape(Ql)[0]
    g=inter/(96.0*cutoff)
    #first the kinetic energy term
    #Kin=Ql.dot(R)-Rl.dot(Q)
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    I0=lam
    I1=R
    I2=Rl.dot(R)
    I3=Rl.dot(Rl).dot(R)
    Delta=0.0
    V=(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
        np.tensordot(kleft,lam,([0],[0]))+\
        np.tensordot(lam,kright,([1],[0]))

    W=(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
        (np.real(mass**2)+cutoff**2)/(2.0)*R+\
        (np.real(mass**2)-cutoff**2)/(4.0)*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
        np.tensordot(kleft,R,([0],[0]))+\
        np.tensordot(R,kright,([1],[0]))+\
        6.0*g*(herm(Rl).dot(I2)+I2.dot(herm(Rr)))+\
        4.0*g*(herm(Rl.dot(Rl)).dot(I1)+herm(Rl).dot(I1).dot(herm(Rr))+I1.dot(herm(Rr.dot(Rr))))+\
        4.0*g*I3+\
        1.0*g*(herm(Rl.dot(Rl).dot(Rl)).dot(I0)+herm(Rl.dot(Rl)).dot(I0).dot(herm(Rr))+herm(Rl).dot(I0).dot(herm(Rr.dot(Rr)))+I0.dot(herm(Rr.dot(Rr).dot(Rr))))

        #the 1.0*g part is missing here because it doesn't contain any herm(Rc)

    return V,W

#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain a lambda 
def HomogeneousPhiFourGradient2(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,inter,cutoff,penalty=0.0):
    #left tangent gauge
    D=np.shape(Ql)[0]
    g=inter/(96.0*cutoff)

    #first the kinetic energy term
    #Kin=Ql.dot(R)-Rl.dot(Q)
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    I0=lam
    I1=R
    I2=Rl.dot(R)
    I3=Rl.dot(Rl).dot(R)
    R4=Rl.dot(Rl).dot(Rl).dot(R)
    Delta=0.0
    V=(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
        np.tensordot(kleft,lam,([0],[0]))+\
        np.tensordot(lam,kright,([1],[0]))

    W=(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
        (np.real(mass**2)+cutoff**2)/(2.0)*R+\
        (np.real(mass**2)-cutoff**2)/(4.0)*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
        np.tensordot(kleft,R,([0],[0]))+\
        np.tensordot(R,kright,([1],[0]))+\
        6.0*g*(herm(Rl).dot(I2)+I2.dot(herm(Rr)))+\
        1.0*g*(herm(Rl.dot(Rl).dot(Rl)).dot(I0)+herm(Rl.dot(Rl)).dot(I0).dot(herm(Rr))+herm(Rl).dot(I0).dot(herm(Rr.dot(Rr)))+I0.dot(herm(Rr.dot(Rr).dot(Rr))))+\
        4.0*g*(herm(Rl.dot(Rl)).dot(I1)+herm(Rl).dot(I1).dot(herm(Rr))+I1.dot(herm(Rr.dot(Rr))))+\
        4.0*g*I3+\
        penalty*(herm(Rl.dot(Rl).dot(Rl)).dot(R4)+herm(Rl.dot(Rl)).dot(R4).dot(herm(Rr))+herm(Rl).dot(R4).dot(herm(Rr.dot(Rr)))+R4.dot(herm(Rr.dot(Rr).dot(Rr))))

       #Delta*lam
       #the 1.0*g part is missing here because it doesn't contain any herm(Rc)

    return V,W


        
#pass "f" as a matrix
#note that the index "0" of the density-like matrix lives on the UPPER bond (hence the multiple transpose operations below)
#"cmps" is an object of the CMPS class, defined in cmpsdisclib or cmpslib.
#"f" is a bond-like matrix living between matrices Q and R. 
#"mu" is a vector of length N that contains the values of the chemical potential. The routine assumes unit-cell translational invariance, that is mu[cmps._N]=mu[0].
#"mass" and "g" are mass and interaction strenght of the bosons
#"direction" can be >0 or <0. use direction>0 when going from left to right, and direction<0 if going from right to left (i.e. when calculating left or right block 
#hamiltonians).
#"f" is living on bond "ind", and the routine gives back the derivative of "f" calculated from adjacent matrices Q[ind-1], R[ind-1] and Q[ind], R[ind].
def HomogeneousMixedDfdxLiebLiniger(Qu,Ru,Ql,Rl,dens,f,mu,mass,g,direction):
    Dl=np.shape(Ql)[0]
    Du=np.shape(Qu)[0]
    if direction>0:
        Ku=Qu.dot(Ru)-Ru.dot(Qu)
        Kl=Ql.dot(Rl)-Rl.dot(Ql)
        Iu=Ru.dot(Ru)
        Il=Rl.dot(Rl)
        return np.reshape(mixedTransferOperator(Qu,Ru,Ql,Rl,direction,sigma=0.0,vector=np.reshape(f,Dl*Du)),(Du,Dl))+\
            1.0/(2.0*mass)*np.transpose(herm(Kl).dot(np.transpose(dens)).dot(Ku))+\
            g*np.transpose(herm(Il).dot(np.transpose(dens)).dot(Iu))+\
            mu/2.0*np.transpose(herm(np.eye(Dl)).dot(herm(Rl)).dot(np.transpose(dens)).dot(Ru).dot(np.eye(Du)))+\
            mu/2.0*np.transpose(herm(Rl).dot(np.transpose(dens)).dot(Ru))+\
            mu/2.0*np.transpose(herm(Rl).dot(herm(np.eye(Dl))).dot(np.transpose(dens)).dot(np.eye(Du)).dot(Ru))+\
            mu/2.0*np.transpose(herm(Rl).dot(np.transpose(dens)).dot(Ru))


    if direction<0:
        Ku=Qu.dot(Ru)-Ru.dot(Qu)
        Kl=Ql.dot(Rl)-Rl.dot(Ql)
        Iu=Ru.dot(Ru)
        Il=Rl.dot(Rl)
        return np.reshape(mixedTransferOperator(Qu,Ru,Ql,Rl,direction,sigma=0.0,vector=np.reshape(f,Dl*Du)),(Du,Dl))+\
            +1.0/(2.0*mass)*Ku.dot(dens).dot(herm(Kl))\
            +g*Iu.dot(dens).dot(herm(Il))\
            +mu/2.0*(np.eye(Du)).dot(Ru).dot(dens).dot(herm(Rl)).dot(herm(np.eye(Dl)))+\
            +mu/2.0*Ru.dot(dens).dot(herm(Rl))+\
            +mu/2.0*(Ru).dot(np.eye(Du)).dot(dens).dot(herm(np.eye(Dl))).dot(herm(Rl))+\
            +mu/2.0*(Ru).dot(dens).dot(herm(Rl))




#calls a sparse eigensolver to find the lowest eigenvalue
#takes only L and R blocks, and finds the ground state
def cmpseigshbondsimple(fl,Ql,Rl,fr,Qr,Rr,lam0,mass,mu,inter,dx,tolerance=1e-6,numvecs=1,numcv=10,dtype=float):
    D=np.shape(Ql)[1]
    mv=fct.partial(HomogeneousLiebLinigerCMPSHAproductBond,*[fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx])
    LOP=LinearOperator((D**2,D**2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=np.reshape(lam0,D**2),ncv=numcv)
    return [e,np.reshape(v,(D,D))]


    


#fl0,fr0 and lam0 are vector-shaped matrices
def prepareSimulationHomogeneousLiebLinger(Q,R,dx,mu0,mass,inter,Delta=0.0,dtype=float,fl0=None,fr0=None,lam0=None,regaugetol=1E-10,lgmrestol=1E-10):
    Delta=0.0
    D=np.shape(Q)[0]

    #lam,Ql,Rl,Qr,Rr=regauge_old(Q,[R],dx,gauge='symmetric',linitial=lam0,rinitial=np.reshape(np.eye(D),D*D),nmaxit=100000,tol=regaugetol)
    lam,Ql,_Rl,Qr,_Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=regauge(Q,[R],dx,gauge='symmetric',linitial=lam0,rinitial=np.reshape(np.eye(D),D*D),nmaxit=100000,tol=regaugetol)
    Rl=_Rl[0]
    Rr=_Rr[0]
    mpopbc=H.projectedLiebLinigermpo3(mu0*np.ones(2),inter,mass,dx*np.ones(2),False,dtype=dtype)
    #mpopbc=H.projectedLiebLinigermpo3withDelta(mu0*np.ones(2),inter,mass,Delta,dx*np.ones(2),False,dtype=dtype)
    

    A=dcmps.toMPSmat(Ql,Rl,dx)
    B=dcmps.toMPSmat(Qr,Rr,dx)
    
    f=np.zeros((D,D),dtype=dtype)
    
    l=np.eye(D)
    r=np.diag(lam)
    ihl=homogeneousdfdxLiebLiniger(Ql,Rl,Ql,Rl,dx,f,mu0,mass,inter,Delta,direction=1)
    ihlprojected=-(ihl-np.trace(np.transpose(ihl).dot(r))*l)
    fl,nit=inverseTransferOperator(Ql,Rl,dx,l,r,ihlprojected,direction=1,x0=fl0,tolerance=lgmrestol,maxiteration=4000)
    
    
    l=np.diag(lam)
    r=np.eye(D)
    ihr=homogeneousdfdxLiebLiniger(Qr,Rr,Qr,Rr,dx,f,mu0,mass,inter,Delta,direction=-1)
    ihrprojected=-(ihr-np.trace(np.transpose(l).dot(ihr))*r)
    fr,nit=inverseTransferOperator(Qr,Rr,dx,l,r,ihrprojected,direction=-1,x0=fr0,tolerance=lgmrestol,maxiteration=4000)
    
    [B1,B2,d1,d2]=np.shape(mpopbc[1])
    mpo=np.zeros((1,B2,d1,d2),dtype=dtype)
    mpo[0,:,:,:]=mpopbc[1][-1,:,:,:]
    lb=mf.initializeLayer(A,np.eye(D),A,mpo,1)
    lb[:,:,0]=np.copy(fl)
    
    
    [B1,B2,d1,d2]=np.shape(mpopbc[0])
    mpo=np.zeros((B1,1,d1,d2),dtype=dtype)
    mpo[:,0,:,:]=mpopbc[0][:,0,:,:]
    rb=mf.initializeLayer(B,np.eye(D),B,mpo,-1)
    rb[:,:,-1]=np.copy(fr)

    scaling=0.2
    LUC=1.0
    NUC=2.0
    xb=np.sort(np.linspace(0,LUC,NUC+1))
    cmps=dcmps.DiscreteCMPS('homogeneous','bla',D,LUC,xb,dtype,scaling=scaling,epstrunc=1E-12,obc=False)            
    for n in range(cmps._N):
        cmps._Q[n]=np.copy(Ql)
        cmps._R[n]=np.copy(Rl)
        cmps._mats[n]=np.diag(lam)
    cmps._position=cmps._N
    cmps._mats[-1]=np.diag(lam)
    #for n in range(cmps._N):
    #    cmps._Q[n]=np.copy(Qr)
    #    cmps._R[n]=np.copy(Rr)
    #cmps._position=0
    #cmps._mats[0]=np.diag(lam)
    cmps._connector=np.diag(1./lam)
    cmps._dx=dx*np.ones(cmps._N)
    cmps._L=np.sum(cmps._dx)
    cmps._xb=np.linspace(0,cmps._L,cmps._N+1)
    for n in range(cmps._N):
        cmps._xm[n]=(cmps._xb[n+1]+cmps._xb[n])/2.0

    #cmps.__regauge__(True,tol=1E-16,ncv=100)
    return lb,fl,rb,fr,lam,Ql,Rl,Qr,Rr,cmps,mpopbc


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



"""
Regauging method for a cMPS that has been obtained from tensoring it with a cMPO
The cMPO matrices Q and R have to be obtained by tensoring a cMPS with matrices Q_ and R_ with a cMPO with matrices Gammas[i][j] in the following way:
Q=np.copy(np.kron(np.eye(D),Gammas[0][0])+np.kron(Q_,diag)+np.kron(R_,Gammas[1][0]))
R=np.copy(np.kron(np.eye(D),Gammas[0][1])+np.kron(R_,diag))
diag is a projector onto the state 0 of dimension M, i.e. it is a diagonal matrix of the form np.diag([1,0,...,0]), with M-1 zeros on the diagonal.
using np.reshape(Q,(D,M,D,M) gives a tensor of the form (similar for R)
         0 --     11     -- 2       0 --  Q_ -- 2       0 --     R_     -- 2
Q=                 x            +         x         +            x                                   
         1 --Gammas[0][0]-- 3       1 -- diag-- 3       1 --Gammas[1][0]-- 3                          
Q is modified by the routine: it is normalized
"""
def regaugeSymmetricKron(Q,R,D,M,initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-14,thresh=1E-10,trunc=1E-16):
    chi=D*M
    dtype=Q.dtype
    
    etal,vl=TMeigsKronecker(Q,R,D,M,direction=1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
    l=np.reshape(vl,(chi,chi))
    l=l/np.trace(l)
    if dtype==float:
        l=np.real((l+herm(l))/2.0)
    if dtype==complex:
        l=(l+herm(l))/2.0
    eigvals,u=np.linalg.eigh(l)
    eigvals[np.nonzero(eigvals<pinv)]=0.0
    eigvals=eigvals/np.sum(eigvals)

    l=u.dot(np.diag(eigvals)).dot(herm(u))

    inveigvals=np.zeros(len(eigvals))
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0

    y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
    invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

    etar,vr=TMeigsKronecker(Q,R,D,M,direction=-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
    Q-=etar/2.0*np.eye(chi)
    r=np.reshape(vr,(chi,chi))
    r=r/np.trace(r)

    if dtype==float:
        r=np.real((r+herm(r))/2.0)
    if dtype==complex:
        r=(r+herm(r))/2.0


    eigvals,u=np.linalg.eigh(r)
    eigvals[np.nonzero(eigvals<pinv)]=0.0
    eigvals/=np.sum(eigvals)

    r=u.dot(np.diag(eigvals)).dot(herm(u))

    inveigvals=np.zeros(len(eigvals))
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0


    r=u.dot(np.diag(eigvals)).dot(herm(u))
    x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
    invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

    [U,lam,V]=svd(y.dot(x))        
    Z=np.linalg.norm(lam)
    lam=lam/Z
    #truncate:
    if trunc>1E-15:
        lam=lam[lam>=trunc]
        U=U[:,0:len(lam)]
        V=V[0:len(lam),:]
        Z=np.linalg.norm(lam)
        lam=lam/Z


    Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
    Rl=Z*np.diag(lam).dot(V).dot(invx).dot(R).dot(invy).dot(U)
    
    
    Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
    Rr=Z*V.dot(invx).dot(R).dot(invy).dot(U).dot(np.diag(lam))
    
    Gl=np.diag(lam).dot(V).dot(invx)*np.sqrt(Z)
    Glinv=invy.dot(U)*np.sqrt(Z)
    Gr=V.dot(invx)*np.sqrt(Z)
    Grinv=invy.dot(U).dot(np.diag(lam))*np.sqrt(Z)

    return lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,etar
    #return lam,Ql,Rl,Qr,Rr


"""
regauging procedure for homogeneous cMPS as given by Q,R matrices.
returns the gauge-transformation that brings the cMPS into the diagonal gauge. The matrices Q and R 
are modified by the routine: they are normalized!

"""

def regauge_old(Q,R,dx=0.0,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-14,thresh=1E-10,trunc=1E-16):
    dtype=Q.dtype
    if gauge=='left':
        [chi,chi2]=np.shape(Q)
        if dx>=1E-8:
            [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig]=TMeigs1stOrder(Q,R,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        #normalization: this is a second order normalization
        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

        if np.abs(np.imag(eta))>thresh:
            print('in regauge: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))


        Ql=y.dot(Q).dot(invy)
        Rl=y.dot(R).dot(invy)
        return l,y,Ql,Rl

    if gauge=='right':
        [chi,chi2]=np.shape(Q)
        if dx>=1E-8:
            [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig]=TMeigs1stOrder(Q,R,-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

        if np.abs(np.imag(eta))>thresh:
            print('in regauge: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))


        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0




        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        Qr=invx.dot(Q).dot(x)
        Rr=invx.dot(R).dot(x)
        return r,x,Qr,Rr




    if gauge=="symmetric":
        [chi ,chi2]=np.shape(Q)
        if dx>=1E-8:
            [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig]=TMeigs1stOrder(Q,R,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')


        if np.abs(np.imag(eta))>thresh:
            print('in regauge: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))

        l=np.reshape(v,(chi,chi))
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0
        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)

        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))
        #y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))))
        #invy=np.transpose(np.diag(np.sqrt(1.0/eigvals)).dot(herm(u)))
        if dx>=1E-8:
            [eta,v,numeig]=TMeigs2ndOrder(Q,R,dx,-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig]=TMeigs1stOrder(Q,R,-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')

        if dtype==float:
            assert(np.abs(np.imag(eta))<1E-10)
            eta=np.real(eta)

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)+phi(dx,eta)*Q
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

        #print eta*dx
        #raw_input()
        if np.abs(np.imag(eta))>thresh:
            print('in regauge: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)

        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0


        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        r=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))
        
        #x=u.dot(np.diag(np.sqrt(eigvals)))
        #invx=np.diag(1.0/np.sqrt(eigvals)).dot(herm(u))

        [U,lam,V]=svd(y.dot(x))        
        Z=np.linalg.norm(lam)
        lam=lam/Z        
        if trunc>1E-15:
            lam=lam[lam>=trunc]
            U=U[:,0:len(lam)]
            V=V[0:len(lam),:]
            Z=np.linalg.norm(lam)
            lam=lam/Z


        Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
        Rl=Z*np.diag(lam).dot(V).dot(invx).dot(R).dot(invy).dot(U)


        Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
        Rr=Z*V.dot(invx).dot(R).dot(invy).dot(U).dot(np.diag(lam))

        return lam,Ql,Rl,Qr,Rr


"""
regauging procedure for homogeneous cMPS as given by Q,R matrices.
returns the gauge-transformation that brings the cMPS into the diagonal gauge. The matrices Q and R 
are modified by the routine: they are normalized!

"""

def regauge_return_basis(Q,R,dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-14,thresh=1E-10,trunc=1E-16):
    dtype=Q.dtype
    if gauge=='left':
        [chi,chi2]=np.shape(Q)
        if dx>=1E-8:
            [eta,v,numeig,nl]=TMeigs2ndOrderCNT(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig,nl]=TMeigs1stOrderCNT(Q,R,1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        
        #normalization: this is a second order normalization
        

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

            
            
        if np.abs(np.imag(eta))>thresh:
            print('in regauge_return_basis: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0


        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))


        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        Ql=y.dot(Q).dot(invy)
        Rl=y.dot(R).dot(invy)
        return l,y,Ql,Rl,nl

    if gauge=='right':
        [chi,chi2]=np.shape(Q)
        if dx>=1E-8:
            [eta,v,numeig,nr]=TMeigs2ndOrderCNT(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig,nr]=TMeigs1stOrderCNT(Q,R,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')



        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

        if np.abs(np.imag(eta))>thresh:
            print('in regauge_return_basis: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))


        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0


        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        #x=sqrtm(r)
        #invx=np.linalg.inv(x)
        #x=u.dot(np.diag(np.sqrt(eigvals)))
        #invx=np.diag(1.0/np.sqrt(eigvals)).dot(herm(u))

        Qr=invx.dot(Q).dot(x)
        Rr=invx.dot(R).dot(x)
        return r,x,Qr,Rr,nr


    if gauge=="symmetric":
        [chi ,chi2]=np.shape(Q)

        if dx>=1E-8:
            [eta,v,numeig,nl]=TMeigs2ndOrderCNT(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:

            [eta,v,numeig,nl]=TMeigs1stOrderCNT(Q,R,1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')



        if np.abs(np.imag(eta))>thresh:
            print('in regauge_return_basis: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))
        
        l=np.reshape(v,(chi,chi))
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))
        #y=np.transpose(sqrtm(l))
        #invy=np.linalg.pinv(y)
  
        if dtype==float:
            assert(np.abs(np.imag(eta))<1E-10)
            eta=np.real(eta)

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            R*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-14:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-14:
            Q-=eta/2.0*np.eye(chi)

        if dx>=1E-8:
            [eta,v,numeig,nr]=TMeigs2ndOrderCNT(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        elif dx<1E-8:
            [eta,v,numeig,nr]=TMeigs1stOrderCNT(Q,R,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')

        if np.abs(np.imag(eta))>thresh:
            print('in regauge_return_basis: found eigenvalue eta with imaginary part>{1}: eta={0}'.format(eta,thresh))
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0

            
        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))


        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))
        
        [U,lam,V]=svd(y.dot(x))
        Z=np.linalg.norm(lam)
        lam=lam/Z        
        if trunc>1E-15:
            lam=lam[lam>trunc]
            U=U[:,0:len(lam)]
            V=V[0:len(lam),:]
            Z=np.linalg.norm(lam)
            lam=lam/Z

        Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
        Rl=Z*np.diag(lam).dot(V).dot(invx).dot(R).dot(invy).dot(U)

        Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
        Rr=Z*V.dot(invx).dot(R).dot(invy).dot(U).dot(np.diag(lam))

        #Gl=np.diag(lam).dot(V).dot(invx)
        #Glinv=invy.dot(U)*Z

        #Gr=Z*V.dot(invx)
        #Grinv=invy.dot(U).dot(np.diag(lam))

        Gl=np.diag(lam).dot(V).dot(invx)*np.sqrt(Z)
        Glinv=invy.dot(U)*np.sqrt(Z)
        Gr=V.dot(invx)*np.sqrt(Z)
        Grinv=invy.dot(U).dot(np.diag(lam))*np.sqrt(Z)

        return lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr





"""
regauging procedure for homogeneous cMPS as given by Q,R matrices.
the routine takes a single matrix Q and a list of matrices R=[R1,R2,...], and returns an orthogonalized cMPS
returns the gauge-transformation that brings the cMPS into the diagonal gauge. The matrices Q and R 
are modified by the routine: they are normalized!
"""

def regauge(Q,R,dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-14,thresh=1E-10,trunc=1E-16,Dmax=100,verbosity=0):
    dtype=Q.dtype
    for n in range(len(R)):
        assert(Q.dtype==R[n].dtype)
    if gauge=='left':
        [chi,chi2]=np.shape(Q)
        [eta,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if np.abs(np.imag(eta))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

        #normalization: this is a second order normalization
        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-12:
            Q-=eta/2.0*np.eye(chi)

        #if np.abs(np.imag(eta))>1E-4:
        #    print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))
        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        Ql=y.dot(Q).dot(invy)
        Rl=[]
        for n in range(len(R)):
            Rl.append(y.dot(R[n]).dot(invy))
        return l,y,Ql,Rl

    if gauge=='right':
        [chi,chi2]=np.shape(Q)
        [eta,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if np.abs(np.imag(eta))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-12:
            Q-=eta/2.0*np.eye(chi)


        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0
        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))


        Rr=[]
        Qr=invx.dot(Q).dot(x)
        for n in range(len(R)):
            Rr.append(invx.dot(R[n]).dot(x))
        return r,x,Qr,Rr


    if gauge=="symmetric":
        [chi ,chi2]=np.shape(Q)
        [etal,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if verbosity>0:
            print('left eigenvalue=',etal)
        if np.abs(np.imag(etal))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue etal with large imaginary part: {0}'.format(etal))
            etal=np.real(etal)

        l=np.reshape(v,(chi,chi))
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0
        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*etal)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,etal)*np.eye(chi)
        elif dx<=1E-12:
            Q-=etal/2.0*np.eye(chi)

        [etar,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if verbosity>0:
            print('right eigenvalue=',etar)
        if np.abs(np.imag(etar))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(etar))
        
        #if np.abs(np.imag(etar))>1E-4:
        #    print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(etar))
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)

        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0

        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        r=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        D=Q.shape[0]
        [U,lam,V]=svd(y.dot(x))        
        Z=np.linalg.norm(lam)
        lam=lam/Z        
        rest=[0.0]
        if trunc>1E-15:
            rest=lam[lam<=trunc]
            lam=lam[lam>trunc]
            rest=np.append(lam[min(len(lam),Dmax)::],rest)
            lam=lam[0:min(len(lam),Dmax)]
            U=U[:,0:len(lam)]
            V=V[0:len(lam),:]
            Z1=np.linalg.norm(lam)
            lam=lam/Z1


        Rl=[]
        Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
        for n in range(len(R)):
            Rl.append(Z*np.diag(lam).dot(V).dot(invx).dot(R[n]).dot(invy).dot(U))
        
        Rr=[]
        Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
        for n in range(len(R)):
            Rr.append(Z*V.dot(invx).dot(R[n]).dot(invy).dot(U).dot(np.diag(lam)))

        Gl=np.diag(lam).dot(V).dot(invx)*np.sqrt(Z)
        Glinv=invy.dot(U)*np.sqrt(Z)
        Gr=V.dot(invx)*np.sqrt(Z)
        Grinv=invy.dot(U).dot(np.diag(lam))*np.sqrt(Z)
        
        #nl and nr are returned for consistency with reguage_return_basis, which returns the number of iterations it took the solver TMeigs to finish
        nl=None
        nr=None
        return lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr
        #return lam,Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,Z,U,V,x,invx,y,invy,etal

def regauge_with_trunc(Q,R,dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-10,ncv=100,numeig=5,pinv=1E-14,thresh=1E-10,trunc=1E-16,Dmax=100,verbosity=0):
    dtype=Q.dtype
    for n in range(len(R)):
        assert(Q.dtype==R[n].dtype)
    if gauge=='left':
        [chi,chi2]=np.shape(Q)
        [eta,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if np.abs(np.imag(eta))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

        #normalization: this is a second order normalization
        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-12:
            Q-=eta/2.0*np.eye(chi)

        #if np.abs(np.imag(eta))>1E-4:
        #    print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: ',eta)
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))
        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        Ql=y.dot(Q).dot(invy)
        Rl=[]
        for n in range(len(R)):
            Rl.append(y.dot(R[n]).dot(invy))
        return l,y,Ql,Rl

    if gauge=='right':
        [chi,chi2]=np.shape(Q)
        [eta,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if np.abs(np.imag(eta))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*eta)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,eta)*np.eye(chi)
        elif dx<=1E-12:
            Q-=eta/2.0*np.eye(chi)


        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0
        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))


        Rr=[]
        Qr=invx.dot(Q).dot(x)
        for n in range(len(R)):
            Rr.append(invx.dot(R[n]).dot(x))
        return r,x,Qr,Rr


    if gauge=="symmetric":
        [chi ,chi2]=np.shape(Q)
        [etal,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,1,numeig=numeig,init=linitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if verbosity>0:
            print('left eigenvalue=',etal)
        if np.abs(np.imag(etal))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue etal with large imaginary part: {0}'.format(etal))
            etal=np.real(etal)

        l=np.reshape(v,(chi,chi))
        l=l/np.trace(l)
        if dtype==float:
            l=np.real((l+herm(l))/2.0)
        if dtype==complex:
            l=(l+herm(l))/2.0
        eigvals,u=np.linalg.eigh(l)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        if dx>=1E-8:
            phi_=1.0/np.sqrt(1+dx*etal)-1.0
            Q+=phi_/dx*np.eye(chi)+phi_*Q
            for n in range(len(R)):
                R[n]*=(1.0+phi_)
        elif dx<1E-8 and dx>1E-12:
            Q+=phiovereps(dx,etal)*np.eye(chi)
        elif dx<=1E-12:
            Q-=etal/2.0*np.eye(chi)

        [etar,v,numeig]=TMeigs2ndOrderMultiSpecies(Q,R,dx,-1,numeig=numeig,init=rinitial,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LR')
        if verbosity>0:
            print('right eigenvalue=',etar)
        if np.abs(np.imag(etar))>thresh:
            print ('in regaugeMultiSpecies: warning: found eigenvalue eta with large imaginary part: {0}'.format(etar))
        
        #if np.abs(np.imag(etar))>1E-4:
        #    print ('in fixGauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(etar))
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)

        if dtype==float:
            r=np.real((r+herm(r))/2.0)
        if dtype==complex:
            r=(r+herm(r))/2.0

        eigvals,u=np.linalg.eigh(r)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)
        r=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals))
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0


        r=u.dot(np.diag(eigvals)).dot(herm(u))
        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        D=Q.shape[0]
        [U,lam,V]=svd(y.dot(x))        
        Z=np.linalg.norm(lam)
        lam=lam/Z        
        rest=[0.0]
        if trunc>1E-15:
            rest=lam[lam<=trunc]
            lam=lam[lam>trunc]
            rest=np.append(lam[min(len(lam),Dmax)::],rest)
            lam=lam[0:min(len(lam),Dmax)]
            U=U[:,0:len(lam)]
            V=V[0:len(lam),:]
            Z1=np.linalg.norm(lam)
            lam=lam/Z1


        Rl=[]
        Ql=Z*np.diag(lam).dot(V).dot(invx).dot(Q).dot(invy).dot(U)
        for n in range(len(R)):
            Rl.append(Z*np.diag(lam).dot(V).dot(invx).dot(R[n]).dot(invy).dot(U))
        
        Rr=[]
        Qr=Z*V.dot(invx).dot(Q).dot(invy).dot(U).dot(np.diag(lam))
        for n in range(len(R)):
            Rr.append(Z*V.dot(invx).dot(R[n]).dot(invy).dot(U).dot(np.diag(lam)))


        return lam,Ql,Rl,Qr,Rr,rest





#compute entanglement entropy of a finite region                                                                                          
def calculateRenyiEntropy(Q,R,lamold,N,dx,alpha,eps=1E-8,Dmax=50,regaugetol=1E-10):
    D=np.shape(Q)[0]
    #lam,Ql,Rl,Qr,Rr=regauge_old(Q,R,dx,gauge='symmetric',linitial=np.reshape(lamold,D*D),rinitial=np.reshape(np.eye(D),D*D),nmaxit=100000,tol=regaugetol)
    lam,Ql,_Rl,Qr,_Rr,Gl,Glinv,Gr,Grinv,Z,nl,nr=regauge(Q,[R],dx,gauge='symmetric',linitial=lamold,rinitial=np.reshape(np.eye(D),D*D),nmaxit=100000,tol=regaugetol)
    Rl=_Rl[0]
    Rr=_Rr[0]

    etas=[]
    R=[]
    B=dcmps.toMPSmat(Qr,Rr,dx)
    ltensor=np.tensordot(np.diag(lam),B,([1],[0]))
    [D1r,D2r,dr]=np.shape(B)
    reachedmax=False
    for n in range(N):
        #print n,N
        [D1l,D2l,dl]=np.shape(ltensor)
        mpsadd1=np.tensordot(ltensor,B,([1],[0])) #index ordering  0 T T 2
                                                  #                  1 3
        rho=np.reshape(np.tensordot(mpsadd1,np.conj(mpsadd1),([0,2],[0,2])),(dl*dr,dl*dr))   #  0  1
        eta,u=np.linalg.eigh(rho)
        inds=np.nonzero(eta>eps)
        indarray=np.array(inds[0])
        if len(indarray)<=Dmax:
            eta=eta[indarray]
        elif len(indarray)>Dmax:
            while len(indarray)>Dmax:
                indarray=np.copy(indarray[1::])
            eta=eta[indarray]
        etas.append(eta)
        R.append(1.0/(1.0-alpha)*np.log(np.sum(eta**alpha)))
        u_=u[:,indarray]
        utens=np.reshape(u_,(dl,dr,len(eta)))
        ltensor=np.tensordot(mpsadd1,np.conj(utens),([1,3],[0,1]))
    return etas,R



#takes two cMPS tensors (C1,Q1,R1),(C2,Q2,R2) and returns the scalar product
#cmpsl gets conjugated
def scalarProduct(cmpsl,cmpsr,Operator):
    temp=np.tensordot(cmpsr,Operator,([2],[0]))
    return np.tensordot(np.conj(cmpsl),temp,([0,1,2],[0,1,2]))



#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
#THIS ROUTINE RETURNS THE "Y" FROM THE TDVP
def HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl,Qr,Rr,k,lam,mass,mu,inter,Delta,dx,direction):
    #left tangent gauge
    D=np.shape(Ql)[0]
    if direction>0:
        #first the kinetic energy term
        upperkinetic=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
        upperinteraction=Rl.dot(lam).dot(Rr)
        if dx>1E-6:
            V0=(-1.0)*np.linalg.inv(np.eye(D)+dx*herm(Ql)).dot(herm(Rl))
        if dx<=1E-6:
            V0=(-1.0)*herm(Rl)

        #four terms in the kinetic part:
        kinetic=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-herm(V0).dot(herm(Rl)).dot(upperkinetic)+herm(V0).dot(upperkinetic).dot(herm(Rr))-upperkinetic.dot(herm(Qr)))
        interaction=inter*(herm(Rl).dot(upperinteraction)+upperinteraction.dot(herm(Rr)))
        potential=dx*mu/2.0*(herm(V0).dot(herm(Rl)).dot(Rl).dot(np.eye(D)+dx*Ql).dot(lam)+herm(Rl).dot(Rl).dot(Rl).dot(lam))+mu*Rl.dot(lam)
        pairing=Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))
        renormalizedleft=np.tensordot(k.dot(np.conj(V0)),(lam+dx*Ql.dot(lam)),([0],[0]))+np.tensordot(k,Rl.dot(lam),([0],[0]))
        return kinetic+interaction+potential+pairing+renormalizedleft

    if direction<0:
        print ('HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl,Qr,Rr,k,lam,mass,mu,inter,dx,direction):  direction<0 not implemented')
        return



#deprecated 
#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
def HomogeneousLiebLinigerCMPSTDVPSymmetricHAproduct(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,inter,dx):
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    upperkinetic=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
    upperinteraction=Rl.dot(lam).dot(Rr)
    
    
    V=(1.0/(2.0*mass)*(-herm(Rl).dot(upperkinetic)+upperkinetic.dot(herm(Rr)))+\
       np.tensordot(kleft,(lam+dx*Ql.dot(lam)),([0],[0]))+\
       np.tensordot((lam+dx*Ql.dot(lam)),kright,([1],[0])))
    

    W=(1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-upperkinetic.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(upperinteraction)+upperinteraction.dot(herm(Rr)))+\
       +mu*Rl.dot(lam)+\
       +np.tensordot(kleft,Rl.dot(lam),([0],[0]))+\
       +np.tensordot(Rl.dot(lam),kright,([1],[0])))

    return V,W



#local energy of the homogeneous Lieb Liniger model
def HomogeneousLiebLinigerEnergy(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,mu,inter):
    #left tangent gauge
    D=np.shape(Ql)[0]
    
    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    

    return 1.0/(4.0*mass)*np.tensordot(Kinleft,np.conj(Kinleft),([0,1],[0,1]))+\
        1.0/(4.0*mass)*np.tensordot(Kinright,np.conj(Kinright),([0,1],[0,1]))+\
        inter/2.0*np.tensordot(Ileft,np.conj(Ileft),([0,1],[0,1]))+\
        inter/2.0*np.tensordot(Iright,np.conj(Iright),([0,1],[0,1]))+\
        mu*np.tensordot(R,np.conj(R),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,lam,([0],[0])),np.conj(Q),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,Q,([0],[0])),np.conj(lam),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,R,([0],[0])),np.conj(R),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kright,lam,([0],[1])),np.conj(Q),([0,1],[1,0]))+\
        np.tensordot(np.tensordot(kright,Q,([0],[1])),np.conj(lam),([0,1],[1,0]))+\
        np.tensordot(np.tensordot(kright,R,([0],[1])),np.conj(R),([0,1],[1,0]))
    

#local energy of the homogeneous Lieb Liniger model
def HomogeneousLiebLinigerEnergyVector(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,inter,vec):
    #left tangent gauge
    D=np.shape(Ql)[0]
    mats=np.reshape(vec,(D,D,2))
    Q=np.copy(mats[:,:,0])
    R=np.copy(mats[:,:,1])

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    

    return 1.0/(4.0*mass)*np.tensordot(Kinleft,np.conj(Kinleft),([0,1],[0,1]))+\
        1.0/(4.0*mass)*np.tensordot(Kinright,np.conj(Kinright),([0,1],[0,1]))+\
        inter/2.0*np.tensordot(Ileft,np.conj(Ileft),([0,1],[0,1]))+\
        inter/2.0*np.tensordot(Iright,np.conj(Iright),([0,1],[0,1]))+\
        mu*np.tensordot(R,np.conj(R),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,lam,([0],[0])),np.conj(Q),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,Q,([0],[0])),np.conj(lam),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kleft,R,([0],[0])),np.conj(R),([0,1],[0,1]))+\
        np.tensordot(np.tensordot(kright,lam,([0],[1])),np.conj(Q),([0,1],[1,0]))+\
        np.tensordot(np.tensordot(kright,Q,([0],[1])),np.conj(lam),([0,1],[1,0]))+\
        np.tensordot(np.tensordot(kright,R,([0],[1])),np.conj(R),([0,1],[1,0]))

#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousLiebLinigerGradient(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,mu,inter,Delta=0):
    #left tangent gauge
    D=np.shape(Ql)[0]

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    V=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
       np.tensordot(kleft,lam,([0],[0]))+\
       np.tensordot(lam,kright,([1],[0]))

    W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
       mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
       np.tensordot(kleft,R,([0],[0]))+np.tensordot(R,kright,([1],[0]))
    return V,W


#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousLiebLinigerGradientRealtime(Ql,Rl,Qr,Rr,Q,R,lam,mass,mu,inter,Delta=0):
    #left tangent gauge
    D=np.shape(Ql)[0]

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    V=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))

    W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
       mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))
    return V,W




#computes the gradients of Ql = Gl Q Glinv = Gl(Q1 x 11 + 11 x Q2)Glinv, __R1__ and __R2__. Ql is a D**2 by D**2 matrix, and __R1__ and __R2__ are D by D matrices (the __ __ 
#is used to distinguish them from R1=__R1__ x 11, and R2=11 x __R2__).
#Gl and Glinv are the matrices that bring the state into left orthogonal form.
#For Ql, the upate is done in the left orthogonal gauge. The functions returns Vl, such that Ql -> Ql-alpha Vl.
#To obtain the update of Q in the original (non-normalized) gauge, transform Ql back: Q_=Glinv(Ql-alpha Vl)Gl; for a product state,
#Q_  will be of the form (Q1_ x 11+ 11 x Q2_), with two new D by D matrices Q1_, Q2_.
#__R1__ and __R2__ are updated in the (non-normalized) original gauge. The update is __R1__ -> __R1__-alpha __gradR1__, __R2__ -> __R2__-alpha __gradR2__.
#In the tensored basis, you get R1_=(__R1__-alpha __gradR1__) x 11,R2_=(11 x __R2__-alpha __gradR2__).
#After the update, use Q_, R1_, and R2_ as the new matrices for the next iteration.

#lam are the schmidt coefficients in vector form
def TwoSpeciesBosonGradient(Ql,Rl,Qr,Rr,D1,D2,kleft,kright,Gl,Glinv,Gr,Grinv,lam,mass,mu,intrag,interg,meanE):
    
    dtype=type(Ql[0,0])
    assert(len(Rl)==len(Rr))
    Gl_=np.reshape(Gl,(D1*D2,D1,D2))
    Glinv_=np.reshape(Glinv,(D1,D2,D1*D2))
    Gr_=np.reshape(Gr,(D1*D2,D1,D2))
    Grinv_=np.reshape(Grinv,(D1,D2,D1*D2))

    #THE FOLLOWING TERMS ARE THE GRADIENTS FOR Q,R1,R2 IN THE TENSORED BASIS. Q,R1,R2 ARE NOT IN ANY PARTICULAR GAUGE (EXCEPT THE STATE BEING NORMALZED TO 1).
    # gradient for R1 from Hl (front layer)
    temp=np.transpose(kleft).dot(Rl[0]).dot(np.diag(lam**2))
    HlgradR1=np.tensordot(np.tensordot(temp,np.conj(Gl_),([0],[0])),np.conj(Glinv_),([0,2],[2,1]))
    # gradient for R2 from Hl (back layer)
    temp=np.transpose(kleft).dot(Rl[1]).dot(np.diag(lam**2))
    HlgradR2=np.tensordot(np.tensordot(temp,np.conj(Gl_),([0],[0])),np.conj(Glinv_),([0,1],[2,0]))
    # gradient for R1 from Hr (front layer)
    temp=np.diag(lam**2).dot(Rr[0]).dot(kright)
    HrgradR1=np.tensordot(np.tensordot(np.conj(Gr_),temp,([0],[0])),np.conj(Grinv_),([1,2],[1,2]))
    # gradient for R2 from Hr (back layer)
    #checked
    temp=np.diag(lam**2).dot(Rr[1]).dot(kright)
    HrgradR2=np.tensordot(np.tensordot(np.conj(Gr_),temp,([0],[0])),np.conj(Grinv_),([0,2],[0,2]))

    # gradient for R1 from kinetic energy (front layer)
    #there is a magical cancellation going on; the full update for R1 consists of the difference of two parts A-B
    #only the difference of the two will give a product structure for an incoming product state, due to a cancellation
    #this has to be written down at some point
    temp1=herm(Ql).dot(Ql.dot(Rl[0])-Rl[0].dot(Ql)).dot(np.diag(lam**2))
    K1term1gradR1=1.0/(2.0*mass[0])*np.tensordot(np.tensordot(temp1,np.conj(Gl_),([0],[0])),np.conj(Glinv_),([0,2],[2,1]))
    temp2=np.diag(lam**2).dot(Qr.dot(Rr[0])-Rr[0].dot(Qr)).dot(herm(Qr))
    K1term2gradR1=-1.0/(2.0*mass[0])*np.tensordot(np.tensordot(temp2,np.conj(Gr_),([0],[0])),np.conj(Grinv_),([0,2],[2,1]))
    # gradient for R2 from kinetic energy (back layer)
    #there is a magical cancellation going on; the full update for R2 consists of the difference of two parts A-B
    #only the difference of the two will give a product structure for an incoming product state, due to a cancellation
    #this has to be written down at some point
    temp1=herm(Ql).dot(Ql.dot(Rl[1])-Rl[1].dot(Ql)).dot(np.diag(lam**2))
    K2term1gradR2=1.0/(2.0*mass[1])*np.tensordot(np.tensordot(temp1,np.conj(Gl_),([0],[0])),np.conj(Glinv_),([0,1],[2,0]))
    temp2=np.diag(lam**2).dot(Qr.dot(Rr[1])-Rr[1].dot(Qr)).dot(herm(Qr))
    K2term2gradR2=-1.0/(2.0*mass[1])*np.tensordot(np.tensordot(temp2,np.conj(Gr_),([0],[0])),np.conj(Grinv_),([0,1],[2,0]))

    # gradient for R1 from intra-species interaction energy (front layer)
    temp1=herm(Rl[0]).dot(Rl[0]).dot(Rl[0]).dot(np.diag(lam**2))
    Intra1term1gradR1=intrag[0]*np.tensordot(np.tensordot(np.conj(Gl_),temp1,([0],[0])),np.conj(Glinv_),([1,2],[1,2]))
    temp2=np.diag(lam**2).dot(Rr[0]).dot(Rr[0]).dot(herm(Rr[0]))
    Intra1term2gradR1=intrag[0]*np.tensordot(np.tensordot(np.conj(Gr_),temp2,([0],[0])),np.conj(Grinv_),([1,2],[1,2]))
    # gradient for R2 from intra-species interaction energy (back layer)
    temp1=herm(Rl[1]).dot(Rl[1]).dot(Rl[1]).dot(np.diag(lam**2))
    Intra2term1gradR2=intrag[1]*np.tensordot(np.tensordot(np.conj(Gl_),temp1,([0],[0])),np.conj(Glinv_),([0,2],[0,2]))
    temp2=np.diag(lam**2).dot(Rr[1]).dot(Rr[1]).dot(herm(Rr[1]))
    Intra2term2gradR2=intrag[1]*np.tensordot(np.tensordot(np.conj(Gr_),temp2,([0],[0])),np.conj(Grinv_),([0,2],[0,2]))


    # gradient for R1 from inter-species interaction energy (front layer)
    temp1=herm(Rl[1]).dot(Rl[0]).dot(Rl[1]).dot(np.diag(lam**2))
    Inter1term1gradR1=0.5*interg*np.tensordot(np.tensordot(np.conj(Gl_),temp1,([0],[0])),np.conj(Glinv_),([1,2],[1,2]))
    temp2=np.diag(lam**2).dot(Rr[0]).dot(Rr[1]).dot(herm(Rr[1]))
    Inter1term2gradR1=0.5*interg*np.tensordot(np.tensordot(np.conj(Gr_),temp2,([0],[0])),np.conj(Grinv_),([1,2],[1,2]))

    # gradient for R2 from inter-species interaction energy (back layer)
    temp1=herm(Rl[0]).dot(Rl[1]).dot(Rl[0]).dot(np.diag(lam**2))
    Inter2term1gradR2=0.5*interg*np.tensordot(np.tensordot(np.conj(Gl_),temp1,([0],[0])),np.conj(Glinv_),([0,2],[0,2]))
    temp2=np.diag(lam**2).dot(Rr[0]).dot(Rr[1]).dot(herm(Rr[0]))
    Inter2term2gradR2=0.5*interg*np.tensordot(np.tensordot(np.conj(Gr_),temp2,([0],[0])),np.conj(Grinv_),([0,2],[0,2]))


    # gradient for R1 from chemical potential energy (frontlayer)
    ChemgradR1=mu[0]*np.tensordot(np.tensordot(np.conj(Gl_),Rl[0].dot(np.diag(lam**2)),([0],[0])),np.conj(Glinv_),([1,2],[1,2]))
    # gradient for R2 from chemical potential energy (frontlayer)
    ChemgradR2=mu[1]*np.tensordot(np.tensordot(np.conj(Gl_),Rl[1].dot(np.diag(lam**2)),([0],[0])),np.conj(Glinv_),([0,2],[0,2]))

    #IdenR1=-meanE*np.tensordot(np.tensordot(np.conj(Gl_),Rl[0].dot(np.diag(lam**2)),([0],[0])),np.conj(Glinv_),([1,2],[1,2]))
    #IdenR2=-meanE*np.tensordot(np.tensordot(np.conj(Gl_),Rl[1].dot(np.diag(lam**2)),([0],[0])),np.conj(Glinv_),([0,2],[0,2]))

    gradR1=HlgradR1+HrgradR1+K1term1gradR1+K1term2gradR1+Inter1term1gradR1+Inter1term2gradR1+Intra1term1gradR1+Intra1term2gradR1+ChemgradR1#+IdenR1
    gradR2=HlgradR2+HrgradR2+K2term1gradR2+K2term2gradR2+Inter2term1gradR2+Inter2term2gradR2+Intra2term1gradR2+Intra2term2gradR2+ChemgradR2#+IdenR2

    # effective norms for R1 and R2, two layer vs single layer
    #effective norm for R1:
    left=np.tensordot(Gl_,np.conj(Gl_),([0],[0]))
    right=np.tensordot(Grinv_,np.conj(Grinv_),([2],[2]))
    NeffR1=np.transpose(np.reshape(np.transpose(np.tensordot(left,right,([1,3],[1,3])),(0,2,1,3)),(D1**2,D1**2)))
    NeffR2=np.transpose(np.reshape(np.transpose(np.tensordot(left,right,([0,2],[0,2])),(0,2,1,3)),(D2**2,D2**2)))

    optgradR1=np.reshape(np.linalg.pinv(NeffR1,rcond=1E-10).dot(np.reshape(gradR1,D1**2)),(D1,D1))
    optgradR2=np.reshape(np.linalg.pinv(NeffR2,rcond=1E-10).dot(np.reshape(gradR2,D2**2)),(D2,D2))
    #optgradR1=gradR1
    #optgradR2=gradR2
    ######################################                             Q gradient        #################################################
    K=[]
    for n in range(len(Rl)):
        K.append(Ql.dot(np.diag(lam)).dot(Rr[n])-Rl[n].dot(np.diag(lam)).dot(Qr))
    Vlam=np.zeros((D1*D2,D1*D2),dtype=dtype)
    for n in range(len(K)):
        Vlam=Vlam+1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))
    Vlam=Vlam+np.tensordot(kleft,np.diag(lam),([0],[0]))+np.tensordot(np.diag(lam),kright,([1],[0]))#-meanE*np.diag(lam)


    #use this gradient to update Q
    optgradQ=Glinv.dot(Vlam).dot(Gr)

    ## gradient for Q from Hl
    #HlgradQ=herm(Gl).dot(np.transpose(kleft)).dot(np.diag(lam**2)).dot(herm(Glinv))
    ## gradient for Q from Hr:
    #HrgradQ=herm(Gr).dot(np.diag(lam**2)).dot(kright).dot(herm(Grinv))
    ## gradient for Q from kinetic energy K1:
    #K1term1gradQ=-1.0/(2.0*mass[0])*herm(Gl).dot(herm(Rl[0])).dot(Ql.dot(Rl[0])-Rl[0].dot(Ql)).dot(np.diag(lam**2)).dot(herm(Glinv))
    #K1term2gradQ=1.0/(2.0*mass[0])*herm(Gr).dot(np.diag(lam**2)).dot(Qr.dot(Rr[0])-Rr[0].dot(Qr)).dot(herm(Rr[0])).dot(herm(Grinv))
    ## gradient for Q from kinetic energy K2:
    #K2term1gradQ=-1.0/(2.0*mass[1])*herm(Gl).dot(herm(Rl[1])).dot(Ql.dot(Rl[1])-Rl[1].dot(Ql)).dot(np.diag(lam**2)).dot(herm(Glinv))
    #K2term2gradQ=1.0/(2.0*mass[1])*herm(Gr).dot(np.diag(lam**2)).dot(Qr.dot(Rr[1])-Rr[1].dot(Qr)).dot(herm(Rr[1])).dot(herm(Grinv))

    #gradQ=HlgradQ+HrgradQ+K1term1gradQ+K1term2gradQ+K2term1gradQ+K2term2gradQ
    #optgradQ=herm(Glinv).dot(gradQ).dot(herm(Gr)).dot(np.diag(1.0/lam))

    return optgradQ,optgradR1,optgradR2

def HomogeneousLiebLinigerTwoBosonsGradient(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,g1,g2):
    #left tangent gauge
    D=np.shape(Ql)[0]
    dtype=type(Ql[0,0])
    assert(len(Rl)==2)
    assert(len(Rr)==2)
    #this is a check that the R-matrices are in fact commuting
    #for n1 in range(len(Rl)-1):
    #    for n2 in range(n1+1,len(Rl)):
    #        assert(np.linalg.norm(comm(Rl[n1],Rl[n2]))<1E-10)
    #        assert(np.linalg.norm(comm(Rr[n1],Rr[n2]))<1E-10)

    K=[]
    Iintra=[]
    Wlam=[]
    for n in range(len(Rl)):
        K.append(Ql.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr))
        Iintra.append(Rl[n].dot(lam).dot(Rr[n]))
        Wlam.append(None)
    Inter=(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(lam)

    #Vlam=(1.0/(2.0*mass[0])*(-herm(Rl[0]).dot(K[0])+K[0].dot(herm(Rr[0]))))+\
    #    np.tensordot(kleft,(lam+dx*Ql.dot(lam)),([0],[0]))+\
    #    np.tensordot(lam,kright,([1],[0]))
    Vlam=np.zeros((D,D),dtype=dtype)
    for n in range(len(K)):
        Vlam=Vlam+1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))

    Vlam=Vlam+np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))
    #the kinetic update is only slightly more messy here: each R[n] gets updated by a W[n], that has to be obtained form GlWGlinvLam[n]; each GlWGlinvLam has one contribution from the kinetic energy, one from inter-particle interaction one from 
    #intra-particle interaction, and one from the chemical potential. There is only one contribution in the kinetic part, as opposed to the TDVP, where due to the Vlam parametrization, there is also one for each part where there is a Vlam.
    #W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
    #   inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
    #   mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
    #   np.tensordot(kleft,R,([0],[0]))+np.tensordot(R,kright,([1],[0]))

    #for n in range(len(K)):
    Wlam[0]=1.0/(2.0*mass[0])*(herm(Ql).dot(K[0])-K[0].dot(herm(Qr)))+\
             g1[0]*(herm(Rl[0]).dot(Iintra[0])+Iintra[0].dot(herm(Rr[0])))+mu[0]*Rl[0].dot(lam)+0.25*g2*herm(Rl[1]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[1]))+\
             np.tensordot(kleft,Rl[0].dot(lam),([0],[0]))+np.tensordot(Rl[0].dot(lam),kright,([1],[0]))
    Wlam[1]=1.0/(2.0*mass[1])*(herm(Ql).dot(K[1])-K[1].dot(herm(Qr)))+\
             g1[1]*(herm(Rl[1]).dot(Iintra[1])+Iintra[1].dot(herm(Rr[1])))+mu[1]*Rl[1].dot(lam)+0.25*g2*herm(Rl[0]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[0]))+\
             np.tensordot(kleft,Rl[1].dot(lam),([0],[0]))+np.tensordot(Rl[1].dot(lam),kright,([1],[0]))

    return Vlam,Wlam




def HomogeneousLiebLinigerTwoBosonSpeciesGradientPenalty(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,g1,g2,penalty):
    D=np.shape(Ql)[0]
    dtype=type(Ql[0,0])
    assert(len(Rl)==len(Rr))
    K=[]
    Iintra=[]
    Wlam=[]
    for n in range(len(Rl)):
        K.append(Ql.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr))
        Iintra.append(Rl[n].dot(lam).dot(Rr[n]))
        Wlam.append(None)
        
    Inter=(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(lam)
    #the penalty from non-commuting Rs
    commR1R2=(Rl[0].dot(Rl[1])-Rl[1].dot(Rl[0])).dot(lam)
    Vlam=np.zeros((D,D),dtype=dtype)
    for n in range(len(K)):
        Vlam=Vlam+1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))
    Vlam=Vlam+np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))

    Wlam[0]=1.0/(2.0*mass[0])*(herm(Ql).dot(K[0])-K[0].dot(herm(Qr)))+\
             g1[0]*(herm(Rl[0]).dot(Iintra[0])+Iintra[0].dot(herm(Rr[0])))+mu[0]*Rl[0].dot(lam)+0.25*g2*herm(Rl[1]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[1]))+\
             penalty*(-herm(Rl[1]).dot(commR1R2)+commR1R2.dot(herm(Rr[1])))+\
             np.tensordot(kleft,Rl[0].dot(lam),([0],[0]))+np.tensordot(Rl[0].dot(lam),kright,([1],[0]))
    Wlam[1]=1.0/(2.0*mass[1])*(herm(Ql).dot(K[1])-K[1].dot(herm(Qr)))+\
             g1[1]*(herm(Rl[1]).dot(Iintra[1])+Iintra[1].dot(herm(Rr[1])))+mu[1]*Rl[1].dot(lam)+0.25*g2*herm(Rl[0]).dot(Inter)+\
             0.25*g2*Inter.dot(herm(Rr[0]))+penalty*(herm(Rl[0]).dot(commR1R2)-commR1R2.dot(herm(Rr[0])))+\
             np.tensordot(kleft,Rl[1].dot(lam),([0],[0]))+np.tensordot(Rl[1].dot(lam),kright,([1],[0]))
    

    return Vlam,Wlam

def HomogeneousLiebLinigerTwoBosonsGradientDiag(Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,kleft,kright,lam,mass,mu,g1,g2):
    #left tangent gauge
    D=np.shape(Ql)[0]
    dtype=type(Ql[0,0])
    assert(len(Rl)==2)
    assert(len(Rr)==2)

    K=[]
    Iintra=[]
    Wlam=[]
    for n in range(len(Rl)):
        K.append(Ql.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr))
        Iintra.append(Rl[n].dot(lam).dot(Rr[n]))
        Wlam.append(None)
    Inter=(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(lam)

    Vlam=np.zeros((D,D),dtype=dtype)
    for n in range(len(K)):
        Vlam+=1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))

    Vlam+=np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))
    optgradQ=Glinv.dot(Vlam).dot(Gr)
    
    Wlam[0]=1.0/(2.0*mass[0])*(herm(Ql).dot(K[0])-K[0].dot(herm(Qr)))+\
             g1[0]*(herm(Rl[0]).dot(Iintra[0])+Iintra[0].dot(herm(Rr[0])))+mu[0]*Rl[0].dot(lam)+0.25*g2*herm(Rl[1]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[1]))+\
             np.tensordot(kleft,Rl[0].dot(lam),([0],[0]))+np.tensordot(Rl[0].dot(lam),kright,([1],[0]))
    Wlam[1]=1.0/(2.0*mass[1])*(herm(Ql).dot(K[1])-K[1].dot(herm(Qr)))+\
             g1[1]*(herm(Rl[1]).dot(Iintra[1])+Iintra[1].dot(herm(Rr[1])))+mu[1]*Rl[1].dot(lam)+0.25*g2*herm(Rl[0]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[0]))+\
             np.tensordot(kleft,Rl[1].dot(lam),([0],[0]))+np.tensordot(Rl[1].dot(lam),kright,([1],[0]))

    gradR0=np.diag(herm(Gl).dot(Wlam[0]).dot(herm(Grinv)))
    gradR1=np.diag(herm(Gl).dot(Wlam[1]).dot(herm(Grinv)))
    Neff=(herm(Gl).dot(Gl))*(np.conj(Grinv.dot(herm(Grinv))))
    Neffinv=np.linalg.pinv(Neff)
    optgradR0=np.diag(Neffinv.dot(gradR0))
    optgradR1=np.diag(Neffinv.dot(gradR1))

    return optgradQ,optgradR0,optgradR1



#uses two Ql and Qr tensors
def HomogeneousLiebLinigerTwoBosonsGradientDiagtest(Ql,Rl,Qr,Rr,Gl,Glinv,Gr,Grinv,kleft,kright,lam,mass,mu,g1,g2):
    #left tangent gauge
    D=np.shape(Ql[0])[0]
    dtype=type(Ql[0][0,0])
    assert(len(Rl)==2)
    assert(len(Rr)==2)

    K=[]
    Iintra=[]
    Wlam=[]
    Ql_=Ql[0]+Ql[1]
    Qr_=Qr[0]+Qr[1]
    for n in range(len(Rl)):
        K.append(Ql_.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr_))
        Iintra.append(Rl[n].dot(lam).dot(Rr[n]))
        Wlam.append(None)

    Inter=(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(lam)
    Vlam=np.zeros((D,D),dtype=dtype)
    for n in range(len(K)):
        Vlam+=1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))

    Vlam+=np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))
    optgradQ=Glinv.dot(Vlam).dot(Gr)
    
    Wlam[0]=1.0/(2.0*mass[0])*(herm(Ql[0]).dot(K[0])-K[0].dot(herm(Qr[0])))+\
             g1[0]*(herm(Rl[0]).dot(Iintra[0])+Iintra[0].dot(herm(Rr[0])))+mu[0]*Rl[0].dot(lam)+0.25*g2*herm(Rl[1]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[1]))+\
             np.tensordot(kleft,Rl[0].dot(lam),([0],[0]))+np.tensordot(Rl[0].dot(lam),kright,([1],[0]))
    Wlam[1]=1.0/(2.0*mass[1])*(herm(Ql[1]).dot(K[1])-K[1].dot(herm(Qr[1])))+\
             g1[1]*(herm(Rl[1]).dot(Iintra[1])+Iintra[1].dot(herm(Rr[1])))+mu[1]*Rl[1].dot(lam)+0.25*g2*herm(Rl[0]).dot(Inter)+0.25*g2*Inter.dot(herm(Rr[0]))+\
             np.tensordot(kleft,Rl[1].dot(lam),([0],[0]))+np.tensordot(Rl[1].dot(lam),kright,([1],[0]))

    gradR0=np.diag(herm(Gl).dot(Wlam[0]).dot(herm(Grinv)))
    gradR1=np.diag(herm(Gl).dot(Wlam[1]).dot(herm(Grinv)))
    Neff=(herm(Gl).dot(Gl))*(np.conj(Grinv.dot(herm(Grinv))))
    Neffinv=np.linalg.pinv(Neff)
    optgradR0=np.diag(Neffinv.dot(gradR0))
    optgradR1=np.diag(Neffinv.dot(gradR1))

    return optgradQ,optgradR0,optgradR1


#this is not correct    
#def HomogeneousLiebLinigerTwoFermionSpeciesGradientPenalty(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,g1,penalty):
#    D=np.shape(Ql)[0]
#    dtype=type(Ql[0,0])
#    assert(len(Rl)==len(Rr))
#    K=[]
#    Iintra=[]
#    Wlam=[]
#    for n in range(len(Rl)):
#        K.append(Ql.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr))
#        Iintra.append(Rl[n].dot(lam).dot(Rr[n]))
#        Wlam.append(None)
#        
#
#    #the penalty from non-anti-commuting Rs
#    anticommR1R2=(Rl[0].dot(Rl[1])+Rl[1].dot(Rl[0])).dot(lam)
#    Vlam=np.zeros((D,D),dtype=dtype)
#    for n in range(len(K)):
#        Vlam=Vlam+1.0/(2.0*mass[n])*(-herm(Rl[n]).dot(K[n])+K[n].dot(herm(Rr[n])))
#    Vlam=Vlam+np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))
#
#    Wlam[0]=1.0/(2.0*mass[0])*(herm(Ql).dot(K[0])-K[0].dot(herm(Qr)))+\
#             g1[0]*(herm(Rl[0]).dot(Iintra[0])+Iintra[0].dot(herm(Rr[0])))+mu[0]*Rl[0].dot(lam)+penalty*(herm(Rl[1]).dot(anticommR1R2)+anticommR1R2.dot(herm(Rr[1])))+\
#             np.tensordot(kleft,Rl[0].dot(lam),([0],[0]))+np.tensordot(Rl[0].dot(lam),kright,([1],[0]))
#    Wlam[1]=1.0/(2.0*mass[1])*(herm(Ql).dot(K[1])-K[1].dot(herm(Qr)))+\
#             g1[1]*(herm(Rl[1]).dot(Iintra[1])+Iintra[1].dot(herm(Rr[1])))+mu[1]*Rl[1].dot(lam)+penalty*(herm(Rl[0]).dot(anticommR1R2)+anticommR1R2.dot(herm(Rr[0])))+\
#             np.tensordot(kleft,Rl[1].dot(lam),([0],[0]))+np.tensordot(Rl[1].dot(lam),kright,([1],[0]))
#
#    return Vlam,Wlam
#




#calculates the gradient of the homogeneous Lieb Liniger model with extended, exponentially decaying interactions locallt. It's not the true
#energy, because it omits spatial variations; lambda is hidden in Q,R, i.e. Q,R have to contain a lambda 
def HomogeneousExtendedLiebLinigerGradient(Ql,Rl,Qr,Rr,Q,R, Hl,Hr,phileft,phiright,lam,mass,mu,g1,g2):
    #left tangent gauge
    D=np.shape(Ql)[0]

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)
    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)

    V=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
       np.tensordot(Hl,lam,([0],[0]))+\
       np.tensordot(lam,Hr,([1],[0]))+\
       g2*np.transpose(phileft).dot(lam).dot(phiright)

    W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       mu*R+\
       np.tensordot(Hl,R,([0],[0]))+\
       np.tensordot(R,Hr,([1],[0]))+\
       g1*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
       g2*(np.transpose(phileft).dot(R)+R.dot(phiright))+\
       g2*(np.transpose(phileft).dot(R).dot(phiright))
       

    return V,W



def HomogeneousLiebLinigerGradientdiag(Ql,Rl,Qr,Rr,Q,R,Gl,Glinv,Gr,Grinv,kleft,kright,lam,mass,mu,inter,Delta=0):
    D=np.shape(Ql)[0]

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    
    Vlam=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
              np.tensordot(kleft,lam,([0],[0]))+\
              np.tensordot(lam,kright,([1],[0]))
    optgradQ=Glinv.dot(Vlam).dot(Gr)
    Wlam=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
          inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
          mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
          np.tensordot(kleft,R,([0],[0]))+\
          np.tensordot(R,kright,([1],[0]))
    
    gradR=np.diag(herm(Gl).dot(Wlam).dot(herm(Grinv)))
    Neff=(herm(Gl).dot(Gl))*(np.conj(Grinv.dot(herm(Grinv))))
    #print Neff.dot(vec)-np.diag(herm(Gl).dot(Gl).dot(np.diag(vec)).dot(Grinv).dot(herm(Grinv)))


    Neffinv=np.linalg.pinv(Neff)
    optgradR=Neffinv.dot(gradR)
    
    return optgradQ,optgradR



def HomogeneousLiebLinigerGradientdiag2(Ql,Rl,Qr,Rr,Q,U,D,kleft,kright,lam,mass,mu,inter,Delta=0):
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    
    #V=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft).dot(herm(lam))+herm(lam).dot(Kinright).dot(herm(Rr)))+\
        #   np.tensordot(kleft,lam.dot(herm(lam)),([0],[0]))+\
        #   np.tensordot(herm(lam).dot(lam),kright,([1],[0]))
    
    V=1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
       np.tensordot(kleft,lam,([0],[0]))+\
       np.tensordot(lam,kright,([1],[0]))
    
    
    dU=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft).dot(herm(lam)).dot(herm(Uinv)).dot(np.diag(np.conj(D)))+\
                       herm(Uinv).dot(np.diag(np.conj(D))).dot(herm(U)).dot(herm(Ql)).dot(Kinleft).dot(herm(lam)).dot(herm(Uinv))-\
                       Kinright.dot(herm(Qr)).dot(herm(lam)).dot(herm(Uinv)).dot(np.diag(np.conj(D)))-\
                       herm(Uinv).dot(np.diag(np.conj(D))).dot(herm(U)).dot(Kinright).dot(herm(Qr)).dot(herm(lam)).dot(herm(Uinv)))+\
        inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
        mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
        np.tensordot(kleft,R,([0],[0]))+\
        np.tensordot(R,kright,([1],[0]))
    
    dD=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
        inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
        mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
        np.tensordot(kleft,R,([0],[0]))+\
        np.tensordot(R,kright,([1],[0]))
    
    return V.dot(herm(lam)),W.dot(herm(lam))
    #return V,W




#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lamda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousLiebLinigerFullGradient(Ql,Rl,Qr,Rr,Q,R,kleft,kright,lam,mass,mu,inter,Delta=0):
    #left tangent gauge
    D=np.shape(Ql)[0]

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    
    V=(1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
       np.tensordot(kleft,lam,([0],[0]))+\
       np.tensordot(lam,kright,([1],[0]))+\
       np.tensordot(kleft,Q,([0],[0]))+\
       np.tensordot(Q,kright,([1],[0])))

    W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
       mu*R+Delta*(herm(Rl).dot(lam)+lam.dot(herm(Rr)))+\
       np.tensordot(kleft,R,([0],[0]))+\
       np.tensordot(R,kright,([1],[0]))

    lamdot=np.tensordot(kleft,lam,([0],[0]))+np.tensordot(lam,kright,([1],[0]))
    return lamdot,V,W


#calculates the gradient of the homogeneous Lieb Liniger energy locally (the routine above). It's not the true
#energy, because it omits spatial variations; lamda is hidden in Q,R, i.e. Q,R have to contain 
#a lambda 
def HomogeneousLiebLinigerGradientVector(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,inter,vec):
    D=np.shape(Ql)[0]
    mats=np.reshape(vec,(D,D,2))
    Q=np.copy(mats[:,:,0])
    R=np.copy(mats[:,:,1])

    #first the kinetic energy term
    Kinleft=Ql.dot(R)-Rl.dot(Q)
    Kinright=Q.dot(Rr)-R.dot(Qr)

    Ileft=Rl.dot(R)
    Iright=R.dot(Rr)
    
    V=(1.0/(2.0*mass)*(-herm(Rl).dot(Kinleft)+Kinright.dot(herm(Rr)))+\
       np.tensordot(kleft,lam,([0],[0]))+\
       np.tensordot(lam,kright,([1],[0])))


    W=1.0/(2.0*mass)*(herm(Ql).dot(Kinleft)-Kinright.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(Ileft)+Iright.dot(herm(Rr)))+\
       mu*R+\
       np.tensordot(kleft,R,([0],[0]))+\
       np.tensordot(R,kright,([1],[0]))

    mats[:,:,0]=np.copy(V)
    mats[:,:,1]=np.copy(W)
    return np.reshape(mats,D*D*2)




def LiebLinigerfmin(Ql,Rl,Qr,Rr,kleft,kright,lam,mass,mu,inter,dt,gradtol=0.001,nmaxit=3):
    dtype=type(Ql[0,0])
    D=np.shape(Ql)[0]
    initial=np.zeros((D,D,2),dtype=dtype)
    initial[:,:,0]=Ql.dot(np.diag(lam))
    initial[:,:,1]=Rl.dot(np.diag(lam))
    x0=np.reshape(initial,D*D*2)
    gradient=fct.partial(HomogeneousLiebLinigerGradientVector,*[Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,inter])
    Energy=fct.partial(HomogeneousLiebLinigerEnergyVector,*[Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,inter])

#    #delta=gradient(initial)
#    #grad=HomogeneousLiebLinigerGradientVector(Ql,Rl,Qr,Rr,kleft,kright,np.diag(lam),mass,mu,inter,x0)
#    grad=gradient(x0)
#    mats=np.reshape(grad,(D,D,2))
#    Vlam=mats[:,:,0]
#    Wlam=mats[:,:,1]
#    Vl=Vlam.dot(np.diag(1.0/lam))
#    Wl=Wlam.dot(np.diag(1.0/lam))
    
    #xopt=x0-0.001*gradient(x0)
    xopt=fmin_cg(f=Energy,x0=x0, fprime=gradient,gtol=gradtol, maxiter=nmaxit, full_output=0, disp=1, retall=0)
    mats=np.reshape(xopt,(D,D,2))
    Ql_new=np.copy(mats[:,:,0].dot(np.diag(1.0/lam)))
    Rl_new=np.copy(mats[:,:,1].dot(np.diag(1.0/lam)))

    #return Ql-dt*Vl,Rl-dt*Wl
    return Ql_new,Rl_new
#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
def HomogeneousLiebLinigerCMPSTDVPSymmetricDiagonalHAproduct(Ql,Rl,Qr,Rr,Gl,diagR,Glinv,kleft,kright,lam,mass,mu,inter,dx):
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    upperkinetic=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
    upperinteraction=Rl.dot(lam).dot(Rr)
    
    V=(1.0/(2.0*mass)*(-herm(Rl).dot(upperkinetic)+upperkinetic.dot(herm(Rr)))+\
       np.tensordot(kleft,(lam+dx*Ql.dot(lam)),([0],[0]))+\
       np.tensordot((lam+dx*Ql.dot(lam)),kright,([1],[0]))).dot(lam)
    
    dG=(1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-upperkinetic.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(upperinteraction)+upperinteraction.dot(herm(Rr)))+\
       +mu*Rl.dot(lam)+\
       +np.tensordot(kleft,Rl.dot(lam),([0],[0]))+\
       +np.tensordot(Rl.dot(lam),kright,([1],[0]))).dot(lam).dot(herm(Glinv)).dot(diagR)

    dDiag=np.diag(Gl.dot(1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-upperkinetic.dot(herm(Qr)))+\
       inter*(herm(Rl).dot(upperinteraction)+upperinteraction.dot(herm(Rr)))+\
       +mu*Rl.dot(lam)+\
       +np.tensordot(kleft,Rl.dot(lam),([0],[0]))+\
       +np.tensordot(Rl.dot(lam),kright,([1],[0]))).dot(lam).dot(herm(Glinv)))
    
    return V,dG,dDiag






#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
#no cross terms due to the vanishing commutator
def HomogeneousLiebLinigerCMPSTDVPHAproductMultiSpecies(Ql,Rl,Qr,Rr,k,lam,mass,mu,g,dx,direction):
    #left tangent gauge
    D=np.shape(Ql)[0]
    dtype=type(Ql[0,0])
    assert(len(Rl)==len(Rr))
    for n1 in range(len(Rl)-1):
        for n2 in range(n1+1,len(Rl)):
            assert(np.linalg.norm(comm(Rl[n1],Rl[n2]))<1E-10)
            assert(np.linalg.norm(comm(Rr[n1],Rr[n2]))<1E-10)

    if direction>0:
        #first the kinetic energy term
        K=[]
        I=[]
        Wlam=[]
        for n in range(len(Rl)):
            K.append(Ql.dot(lam).dot(Rr[n])-Rl[n].dot(lam).dot(Qr))
            I.append(Rl[n].dot(lam).dot(Rr[n]))
            Wlam.append(np.zeros((D,D),dtype=dtype))
        if dx>1E-6:
            V0=[]
            for n in range(len(Rl)):
                V0.append((-1.0)*np.linalg.inv(np.eye(D)+dx*herm(Ql)).dot(herm(Rl[n])))
        if dx<=1E-6:
            V0=[]
            for n in range(len(Rl)):
                V0.append((-1.0)*herm(Rl[n]))


        #the kinetic update is more messy here; each R[n] has a contribution from its own K[n], but also one from each other K[m!=n]
        for n1 in range(len(K)):
            for n2 in range(len(K)):
                if n1==n2:
                    Wlam[n1]=Wlam[n1]+1.0/(2.0*mass[n1])*(herm(Ql).dot(K[n1])-herm(V0[n1]).dot(herm(Rl[n1])).dot(K[n1])+herm(V0[n1]).dot(K[n1]).dot(herm(Rr[n1]))-K[n1].dot(herm(Qr)))
                if n1!=n2:
                    Wlam[n1]=Wlam[n1]+1.0/(2.0*mass[n2])*(-herm(V0[n1]).dot(herm(Rl[n2])).dot(K[n2])+herm(V0[n1]).dot(K[n2]).dot(herm(Rr[n2])))

        #the interaction update: fow now there is only inter-species interaction
        for n in range(len(I)):
            Wlam[n]=Wlam[n]+g[n]*(herm(Rl[n]).dot(I[n])+I[n].dot(herm(Rr[n])))

        #potential=mu*Rl.dot(lam)
        #the potentialupdate is more messy as well; each R[n] has a contribution from its own mu[n], but also one from each other mu[m!=n]
        sumRl=np.zeros((D,D),dtype=dtype)
        #note: there is no transpose here because later on we're connecting it with matrices on the lower branch
        for n in range(len(Rr)):
            sumRl=sumRl+mu[n]/2.0*herm(Rl[n]).dot(Rl[n])

        for n in range(len(mu)):
            Wlam[n]=Wlam[n]+dx*(herm(V0[n]).dot(sumRl).dot(np.eye(D)+dx*Ql).dot(lam)+sumRl.dot(Rl[n]).dot(lam))+mu[n]*Rl[n].dot(lam)



        for n in range(len(Rl)):
            Wlam[n]=Wlam[n]+np.tensordot(k.dot(np.conj(V0[n])),(lam+dx*Ql.dot(lam)),([0],[0]))+np.tensordot(k,Rl[n].dot(lam),([0],[0]))

        return Wlam

    if direction<0:
        print ('HomogeneousLiebLinigerCMPSTDVPHAproduct(Ql,Rl,Qr,Rr,k,lam,mass,mu,inter,dx,direction):  direction<0 not implemented')
        return


#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
def HomogeneousLiebLinigerCMPSHAproductBond(fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx,lamvec):
    D=np.shape(Ql)[0]
    lam=np.reshape(lamvec,(D,D))
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    upperkinetic1=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
    upperkinetic2=(-Rl.dot(lam)+lam.dot(Rr))
    upperinteraction=Rl.dot(lam).dot(Rr)

    #four terms in the kinetic part:
    kinetic11=dx*1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic1).dot(herm(Rr))-herm(Rl).dot(upperkinetic1).dot(herm(Qr)))
    kinetic21=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic2).dot(herm(Rr))-herm(Rl).dot(upperkinetic2).dot(herm(Qr)))
    #kinetic11=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic1).dot(herm(Rr))-herm(Rl).dot(upperkinetic1).dot(herm(Qr)))
    kinetic12=1.0/(2.0*mass)*((upperkinetic1.dot(herm(Rr))-herm(Rl).dot(upperkinetic1)))
    kinetic22=(1.0/dx)*1.0/(2.0*mass)*((upperkinetic2.dot(herm(Rr))-herm(Rl).dot(upperkinetic2)))
    interaction=inter*(herm(Rl).dot(upperinteraction).dot(herm(Rr)))
    potential=mu/2.0*(herm(Rl).dot(Rl).dot(lam)+lam.dot(Rr).dot(herm(Rr)))
    left=np.tensordot(fl,lam,([0],[0]))
    right=np.tensordot(lam,fr,([1],[0]))
    return np.reshape(kinetic11+dx*(interaction+potential)+left+right,D*D)
    #return np.reshape(kinetic11+kinetic12+kinetic21+kinetic22+dx*(interaction+potential)+left+right,D*D)
    #return np.reshape(kinetic11+kinetic12+kinetic21+kinetic22+dx*(interaction+potential),D*D)
    #return np.reshape(left+right,D*D)


#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
def HomogeneousLiebLinigerCMPSHAproductBondMatrix(fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx,lam):
    D=np.shape(Ql)[0]
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    upperkinetic1=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
    upperkinetic2=(-Rl.dot(lam)+lam.dot(Rr))
    upperinteraction=Rl.dot(lam).dot(Rr)

    #four terms in the kinetic part:
    kinetic11=dx*1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic1).dot(herm(Rr))-herm(Rl).dot(upperkinetic1).dot(herm(Qr)))
    kinetic21=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic2).dot(herm(Rr))-herm(Rl).dot(upperkinetic2).dot(herm(Qr)))
    #kinetic11=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic1).dot(herm(Rr))-herm(Rl).dot(upperkinetic1).dot(herm(Qr)))
    kinetic12=1.0/(2.0*mass)*((upperkinetic1.dot(herm(Rr))-herm(Rl).dot(upperkinetic1)))
    kinetic22=(1.0/dx)*1.0/(2.0*mass)*((upperkinetic2.dot(herm(Rr))-herm(Rl).dot(upperkinetic2)))
    interaction=inter*(herm(Rl).dot(upperinteraction).dot(herm(Rr)))
    potential=mu/2.0*(herm(Rl).dot(Rl).dot(lam)+lam.dot(Rr).dot(herm(Rr)))
    left=np.tensordot(fl,lam,([0],[0]))
    right=np.tensordot(lam,fr,([1],[0]))
    #return kinetic11+dx*(interaction+potential)+left+right
    return kinetic11+kinetic12+kinetic21+kinetic22+dx*(interaction+potential)+left+right
    #return np.reshape(kinetic11+kinetic12+kinetic21+kinetic22+dx*(interaction+potential),D*D)
    #return np.reshape(left+right,D*D)




def HomogeneousLiebLinigerCMPSVlamWlamHAproduct(fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx,vec):

    D=np.shape(Ql)[0]
    tensor=np.reshape(vec,(D,D,2))
    Vlam=tensor[:,:,0]
    Wlam=tensor[:,:,1]


    upperkinetic=Ql.dot(Wlam)-Wlam.dot(Qr)+Vlam.dot(Rr)-Rl.dot(Vlam)
    upperinteraction=Rl.dot(Wlam)-Wlam.dot(Rr)


    Wlam_out_kinetic=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-upperkinetic.dot(herm(Qr)))
    Vlam_out_kinetic=1.0/(2.0*mass)*(-herm(Rl).dot(upperkinetic)+upperkinetic.dot(herm(Rr)))

    Wlam_out_interaction=inter*(herm(Rl).dot(upperinteraction)-upperinteraction.dot(herm(Rr)))

    Wlam_out_potential=mu*Wlam


    Wlam_out_left_renormalized=np.tensordot(fl,Wlam,([0],[0]))
    Vlam_out_left_renormalized=dx*np.tensordot(fl,Vlam,([0],[0]))

    Wlam_out_right_renormalized=np.tensordot(Wlam,fr,([1],[0]))
    Vlam_out_right_renormalized=dx*np.tensordot(Vlam,fr,([1],[0]))    

    Wlam_out=Wlam_out_kinetic+Wlam_out_interaction+Wlam_out_potential+Wlam_out_left_renormalized+Wlam_out_right_renormalized
    Vlam_out=Vlam_out_kinetic+Vlam_out_left_renormalized+Vlam_out_right_renormalized

    out=np.zeros((D,D,2),dtype=type(Wlam_out[0,0]))
    out[:,:,0]=dx*Vlam_out
    out[:,:,1]=Wlam_out
    return np.reshape(out,D*D*2)
    
def HomogeneousLiebLinigerCMPSVlamWlamHAproductExplicit(fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx,Vlam,Wlam):

    D=np.shape(Ql)[0]

    upperkinetic=Ql.dot(Wlam)-Wlam.dot(Qr)+Vlam.dot(Rr)-Rl.dot(Vlam)
    upperinteraction=Rl.dot(Wlam)-Wlam.dot(Rr)


    Wlam_out_kinetic=1.0/(2.0*mass)*(herm(Ql).dot(upperkinetic)-upperkinetic.dot(herm(Qr)))
    Vlam_out_kinetic=1.0/(2.0*mass)*(-herm(Rl).dot(upperkinetic)+upperkinetic.dot(herm(Rr)))

    Wlam_out_interaction=inter*(herm(Rl).dot(upperinteraction)-upperinteraction.dot(herm(Rr)))

    Wlam_out_potential=mu*Wlam


    Wlam_out_left_renormalized=np.tensordot(fl,Wlam,([0],[0]))
    Vlam_out_left_renormalized=dx*np.tensordot(fl,Vlam,([0],[0]))

    Wlam_out_right_renormalized=np.tensordot(Wlam,fr,([1],[0]))
    Vlam_out_right_renormalized=dx*np.tensordot(Vlam,fr,([1],[0]))    

    Wlam_out=Wlam_out_kinetic+Wlam_out_interaction+Wlam_out_potential+Wlam_out_left_renormalized+Wlam_out_right_renormalized
    Vlam_out=Vlam_out_kinetic+Vlam_out_left_renormalized+Vlam_out_right_renormalized

    out=np.zeros((D,D,2),dtype=type(Wlam_out[0,0]))
    return Vlam_out,Wlam_out
    

def CMPSeigshHomogeneousLiebLiniger(fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx,numvecs=1,tolerance=1E-8,init=None,ncv=100):
    D=np.shape(Ql)[1]
    dtype=type(Ql[0,0])
    mv=fct.partial(HomogeneousLiebLinigerCMPSVlamWlamHAproduct,*[fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx])
    LOP=LinearOperator((2*D**2,2*D**2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=init,ncv=ncv)
    return [e,np.reshape(v,(D,D,2))]

#returns the new (W lam); lam has to be divided out from (W lam) to get the update W and V for Ql and Rl ((lam W), Qr,Rr for direction<0).
def HomogeneousLiebLinigerCMPS_epsilon_energy(kleft,Ql,Rl,kright,Qr,Rr,lam,mass,mu,inter):
    #left tangent gauge
    D=np.shape(Ql)[0]
    #first the kinetic energy term
    kinetic=Ql.dot(lam).dot(Rr)-Rl.dot(lam).dot(Qr)
    interaction=Rl.dot(lam).dot(Rr)
    energy=1.0/(2.0*mass)*np.trace(herm(kinetic).dot(kinetic))+\
            inter*np.trace(herm(interaction).dot(interaction))+\
            mu*np.trace(Rl.dot(lam).dot(lam).dot(herm(Rl)))+\
            np.trace(np.transpose(kleft).dot(Ql).dot(lam).dot(lam))+np.trace(np.transpose(kleft).dot(lam).dot(lam).dot(herm(Ql)))+np.trace(np.transpose(kleft).dot(Rl).dot(lam).dot(lam).dot(herm(Rl)))+\
            np.trace(lam.dot(Ql).dot(lam).dot(kright))+np.trace(herm(Ql.dot(lam)).dot(lam).dot(kright))+np.trace(herm(Rl.dot(lam)).dot(Rl).dot(lam).dot(kright))

    return energy



#Lower one is conjugated
def V_W_scalarProduct(VlamU,WlamU,VlamL,WlamL):
    return np.trace(VlamU.dot(herm(VlamL)))+np.trace(WlamU.dot(herm(WlamL)))



#def LiebLinigerBracketSearch(Ql,Rl,Qr,Rr,lam,Vl,Wl,kleft,kright,mass,mu,inter,initialenergy,initialnormgrad,dtype,dt0,rescalingfactor,normtol,regaugetol=1E-10,lgmrestol=1E-10,nmaxit=10000,numeig=4,ncv=40,steepestdesc=False):
#
#    dx=0.0
#    D=np.shape(Ql)[0]
#    f=np.zeros((D,D),dtype=dtype)
#    rtol=regaugetol
#    lgmrestol=lgmrestol
#    lam_=np.copy(lam)
#    dt=dt0
#    it=1
#    kleftnew=np.copy(kleft)
#    krightnew=np.copy(kright)
#    converged=False
#    leftnormgrad=initialnormgrad
#    currentenergy=initialenergy
#    tleft=0
#    tright=2.0*dt0
#    dt=(tright-tleft)/2.0
#
#    while not converged:
#        Q=Ql-dt*Vl
#        R=Rl-dt*Wl
#        lam_,Ql_,Rl_,Qr_,Rr_,Gl_,Glinv_,Gr_,Grinv_=regauge_return_basis(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lam_),D*D),datatype=dtype,nmaxit=10000,tol=rtol,numeig=numeig,ncv=ncv)
#
#        ihl=homogeneousdfdxLiebLiniger(Ql_,Rl_,Ql_,Rl_,dx,f,mu,mass,inter,direction=1)
#        ihr=homogeneousdfdxLiebLiniger(Qr_,Rr_,Qr_,Rr_,dx,f,mu,mass,inter,direction=-1)
#    
#        ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam_**2),([0,1],[0,1]))*np.eye(D))
#        ihrprojected=-(ihr-np.tensordot(np.diag(lam_**2),ihr,([0,1],[0,1]))*np.eye(D))
#    
#        kleft_=inverseTransferOperator(Ql_,Rl_,dx,np.eye(D),np.diag(lam_**2),ihlprojected,direction=1,x0=np.reshape(kleftnew,D*D),tolerance=lgmrestol,maxiteration=4000)
#        kright_=inverseTransferOperator(Qr_,Rr_,dx,np.diag(lam_**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(krightnew,D*D),tolerance=lgmrestol,maxiteration=4000)
#        Vlam,Wlam=HomogeneousLiebLinigerGradient(Ql_,Rl_,Qr_,Rr_,Ql_.dot(np.diag(lam_)), Rl_.dot(np.diag(lam_)),kleft_,kright_,np.diag(lam_),mass,mu,inter)
#        normgrad=np.sqrt(np.trace(Vlam.dot(herm(Vlam)))+np.trace(Wlam.dot(herm(Wlam))))
#
#        energy=1.0/(2.0*mass)*np.trace(comm(Ql_,Rl_).dot(np.diag(lam_**2)).dot(herm(comm(Ql_,Rl_))))+inter*np.trace(Rl_.dot(Rl_).dot(np.diag(lam_**2)).dot(herm(Rl_)).dot(herm(Rl_)))+\
#                mu*np.trace(Rl_.dot(np.diag(lam_**2)).dot(herm(Rl_)))
#
#
#        if (normgrad-currentnormgrad)/currentnormgrad>normtol:
#            if leftmoving:
#                tright=dt
#                dt=(tright-tleft)/2.0
#        #if (normgrad-currentnormgrad)/currentnormgrad<(-normtol:)
#        if (normgrad-currentnormgrad)/currentnormgrad<0.0:
#            tleft=dt
#            dt=(tright-tleft)/2.0
#        if ((normgrad-currentnormgrad)/currentnormgrad>=0.0) and ((normgrad-currentnormgrad)/currentnormgrad<normtol):
#        kleftnew=np.copy(kleft_)
#        krightnew=np.copy(kright_)
#        currentenergy=energy
#        currentnormgrad=normgrad
#        Qlnew=np.copy(Ql_)
#        Rlnew=np.copy(Rl_)
#        Qrnew=np.copy(Qr_)
#        Rrnew=np.copy(Rr_)
#        lamnew=np.copy(lam_)
#        Vlamnew=np.copy(Vlam)
#        Wlamnew=np.copy(Wlam)
#        Gl=np.copy(Gl_)
#        Glinv=np.copy(Glinv_)
#        Gr=np.copy(Gr_)
#        Grinv=np.copy(Grinv_)
#
#    Vlnew=Vlamnew.dot(np.diag(1.0/lam_))
#    Wlnew=Wlamnew.dot(np.diag(1.0/lam_))
#    return Qlnew,Rlnew,Qrnew,Rrnew,lamnew,Vlnew,Wlnew,Gl,Glinv,Gr,Grinv,kleftnew,krightnew,currentenergy,currentnormgrad,dt*(it-2)


def LiebLinigerLineSearch(Ql,Rl,Qr,Rr,lam,Vl,Wl,kleft,kright,mass,mu,inter,initialenergy,initialnormgrad,dtype,dt0,rescalingfactor,normtol,regaugetol=1E-10,lgmrestol=1E-10,nmaxit=10000,numeig=4,ncv=40,itmax=10):

    dx=0.0
    D=np.shape(Ql)[0]
    f=np.zeros((D,D),dtype=dtype)
    rtol=regaugetol
    lgmrestol=lgmrestol
    lamnew=np.copy(lam)
    dt=dt0
    it=1
    kleftnew=np.copy(kleft)
    krightnew=np.copy(kright)
    converged=False
    currentnormgrad=initialnormgrad
    currentenergy=initialenergy
    while not converged:
        Q=Ql-it*dt*Vl
        R=Rl-it*dt*Wl

        lam_,Ql_,Rl_,Qr_,Rr_,Gl_,Glinv_,Gr_,Grinv_=regauge_return_basis(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lamnew),D*D),datatype=dtype,nmaxit=10000,tol=rtol,numeig=numeig,ncv=ncv)

        ihl=homogeneousdfdxLiebLiniger(Ql_,Rl_,Ql_,Rl_,dx,f,mu,mass,inter,Delta=0.0,direction=1)
        ihr=homogeneousdfdxLiebLiniger(Qr_,Rr_,Qr_,Rr_,dx,f,mu,mass,inter,Delta=0.0,direction=-1)
    
        ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam_**2),([0,1],[0,1]))*np.eye(D))
        ihrprojected=-(ihr-np.tensordot(np.diag(lam_**2),ihr,([0,1],[0,1]))*np.eye(D))
    
        kleft_,nit=inverseTransferOperator(Ql_,Rl_,dx,np.eye(D),np.diag(lam_**2),ihlprojected,direction=1,x0=np.reshape(kleftnew,D*D),tolerance=lgmrestol,maxiteration=4000)
        kright_,nit=inverseTransferOperator(Qr_,Rr_,dx,np.diag(lam_**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(krightnew,D*D),tolerance=lgmrestol,maxiteration=4000)
        Vlam_,Wlam_=HomogeneousLiebLinigerGradient(Ql_,Rl_,Qr_,Rr_,Ql_.dot(np.diag(lam_)), Rl_.dot(np.diag(lam_)),kleft_,kright_,np.diag(lam_),mass,mu,inter)
        normgrad=np.sqrt(np.trace(Vlam_.dot(herm(Vlam_)))+np.trace(Wlam_.dot(herm(Wlam_))))

        energy=1.0/(2.0*mass)*np.trace(comm(Ql_,Rl_).dot(np.diag(lam_**2)).dot(herm(comm(Ql_,Rl_))))+inter*np.trace(Rl_.dot(Rl_).dot(np.diag(lam_**2)).dot(herm(Rl_)).dot(herm(Rl_)))+\
                mu*np.trace(Rl_.dot(np.diag(lam_**2)).dot(herm(Rl_)))

        it+=1
        if (it>2) and ((normgrad-currentnormgrad)/currentnormgrad>=0.0):
            break

        if (it==2) and ((normgrad-currentnormgrad)/currentnormgrad>0.0):
            dt=dt/rescalingfactor
            it=1

        if it>itmax:
            converged=True
        if (it>=2):
            kleftnew=np.copy(kleft_)
            krightnew=np.copy(kright_)
            currentenergy=energy
            currentnormgrad=normgrad
            Qlnew=np.copy(Ql_)
            Rlnew=np.copy(Rl_)
            Qrnew=np.copy(Qr_)
            Rrnew=np.copy(Rr_)
            lamnew=np.copy(lam_)
            Vlamnew=np.copy(Vlam_)
            Wlamnew=np.copy(Wlam_)
            Gl=np.copy(Gl_)
            Glinv=np.copy(Glinv_)
            Gr=np.copy(Gr_)
            Grinv=np.copy(Grinv_)

    Vlnew=Vlamnew.dot(np.diag(1.0/lamnew))
    Wlnew=Wlamnew.dot(np.diag(1.0/lamnew))
    return Qlnew,Rlnew,Qrnew,Rrnew,lamnew,Vlnew,Wlnew,Gl,Glinv,Gr,Grinv,kleftnew,krightnew,currentenergy,currentnormgrad,dt*(it-2)


def LiebLinigerSteepestDescent(Ql,Rl,Qr,Rr,lam,Vl,Wl,kleft,kright,mass,mu,inter,initialenergy,initialnormgrad,dtype,dt0,rescalingfactor,normtol,regaugetol=1E-10,lgmrestol=1E-10,nmaxit=10000,numeig=4,ncv=40):

    dx=0.0
    D=np.shape(Ql)[0]
    f=np.zeros((D,D),dtype=dtype)
    rtol=regaugetol
    lgmrestol=lgmrestol
    lam_=np.copy(lam)
    dt=dt0
    it=1
    kleft_=np.copy(kleft)
    kright_=np.copy(kright)
    converged=False
    while not converged:
        Q=Ql-dt*Vl
        R=Rl-dt*Wl
        lam_,Ql_,Rl_,Qr_,Rr_,Gl_,Glinv_,Gr_,Grinv_=regauge_return_basis(Q,R,dx,gauge='symmetric',initial=np.reshape(np.diag(lam_),D*D),datatype=dtype,nmaxit=10000,tol=rtol,numeig=numeig,ncv=ncv)

        ihl=homogeneousdfdxLiebLiniger(Ql_,Rl_,Ql_,Rl_,dx,f,mu,mass,inter,Delta=0.0,direction=1)
        ihr=homogeneousdfdxLiebLiniger(Qr_,Rr_,Qr_,Rr_,dx,f,mu,mass,inter,Delta=0.0,direction=-1)
    
        ihlprojected=-(ihl-np.tensordot(ihl,np.diag(lam_**2),([0,1],[0,1]))*np.eye(D))
        ihrprojected=-(ihr-np.tensordot(np.diag(lam_**2),ihr,([0,1],[0,1]))*np.eye(D))
    
        kleft_,nit=inverseTransferOperator(Ql_,Rl_,dx,np.eye(D),np.diag(lam_**2),ihlprojected,direction=1,x0=np.reshape(kleft_,D*D),tolerance=lgmrestol,maxiteration=4000)
        kright_,nit=inverseTransferOperator(Qr_,Rr_,dx,np.diag(lam_**2),np.eye(D),ihrprojected,direction=-1,x0=np.reshape(kright_,D*D),tolerance=lgmrestol,maxiteration=4000)
        Vlam_,Wlam_=HomogeneousLiebLinigerGradient(Ql_,Rl_,Qr_,Rr_,Ql_.dot(np.diag(lam_)), Rl_.dot(np.diag(lam_)),kleft_,kright_,np.diag(lam_),mass,mu,inter)
        normgrad=np.sqrt(np.trace(Vlam_.dot(herm(Vlam_)))+np.trace(Wlam_.dot(herm(Wlam_))))

        energy=1.0/(2.0*mass)*np.trace(comm(Ql_,Rl_).dot(np.diag(lam_**2)).dot(herm(comm(Ql_,Rl_))))+inter*np.trace(Rl_.dot(Rl_).dot(np.diag(lam_**2)).dot(herm(Rl_)).dot(herm(Rl_)))+\
                mu*np.trace(Rl_.dot(np.diag(lam_**2)).dot(herm(Rl_)))

        if ((normgrad-initialnormgrad)/initialnormgrad>normtol):
            dt=dt/rescalingfactor

        if ((normgrad-initialnormgrad)/initialnormgrad<=normtol):
            converged = True
    Vl_=Vlam_.dot(np.diag(1.0/lam_))
    Wl_=Wlam_.dot(np.diag(1.0/lam_))
    return Ql_,Rl_,Qr_,Rr_,lam_,Vl_,Wl_,Gl_,Glinv_,Gr_,Grinv_,kleft_,kright_,energy,normgrad,dt


def addCMPSLayer(k,Q,R,deltax,mu,mass,g,direction):
    return k+deltax*homogeneousdfdxLiebLiniger(Q,R,Q,R,0.0,k,mu,mass,g,0.0,direction)


def LiebLinigerNonLinearConGrad(kleft,Ql,Rl,kright,Qr,Rr,lamold,mass,mu,inter,V,W,dx,dtype,regaugetol=1E-10,lgmrestol=1E-10):
    D=np.shape(Ql)[1]
    dtype=type(Ql[0,0])
    mv=fct.partial(HomogeneousLiebLinigerCMPSVlamWlamHAproduct,*[fl,Ql,Rl,fr,Qr,Rr,mass,mu,inter,dx])
    LOP=LinearOperator((2*D**2,2*D**2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=init,ncv=ncv)
    return [e,np.reshape(v,(D,D,2))]


