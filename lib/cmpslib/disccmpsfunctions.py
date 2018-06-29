#!/usr/bin/env python
import sys
import Hamiltonians as Hams
from sys import stdout
import numpy as np
import utilities as utils
import time
import scipy as sp
import random
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
import Hamiltonians as H
import mpsfunctions as mf
import mpsfunctionsgradopt as mfgo
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.interpolate import griddata
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigs
import warnings
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))



#Q and R are cMPS matrices stored in an array  that will be normalized according to dx
#returns GammaQ, GammaR, Y, the Gamma-content of the cMPS matrices; i.e. Gamma[:,:,0]=Y+dx*GammaQ,Gamma[:,:,1]=sqrt(dx)*GammaR;
#Cvecs is a (len(x),D) array that contains the center matrices; note that Y^-1 and Cvecs are not the same:
#Y[n]=invsqrtr[N-1-n+1].dot(invsqrtl[n+1], i.e. invsrqrtr lives to the left and invsqrtl to the right of the tensor A
#C[n]=sqrtl[n].dot(sqrtl[N-1-n+1]) at the same site of A
def regaugecentralC(Q,R,dx,init=None,tol=1E-10,ncv=20,nmax=10000,which='LM',rcond=1E-16):
    D=Q.shape[0]
    N=Q.shape[2]
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    eta1,v1,numeig=UnitcellTMeigs(Q,R,dx,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)

    #normalize the state:
    
    sqeta=np.real(eta1**(-1.0/(2.0*N)))
    for n in range(N):
        phi=(sqeta-1.0)/dx[n]
        Q[:,:,n]=Q[:,:,n]+np.eye(D)*phi+dx[n]*Q[:,:,n]*phi
        R[:,:,n]=R[:,:,n]+dx[n]*R[:,:,n]*phi

    l=np.reshape(v1,(D,D))
    l/=np.trace(l)
    l=(l+herm(l))/2.0
    if dtype=='float64':
        l=np.real(l)
    ldens=UnitcellTransferOperator(direction=1,Q=Q,R=R,dx=dx,vector=np.reshape(l,D*D),returnfull=True)

    eta2,v2,numeig=UnitcellTMeigs(Q,R,dx,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)
    r=np.reshape(v2,(D,D))
    r/=np.trace(r)
    r=(r+herm(r))/2.0
    if dtype=='float64':
        r=np.real(r)

    rdens=UnitcellTransferOperator(direction=-1,Q=Q,R=R,dx=dx,vector=np.reshape(r,D*D),returnfull=True)
    sqrtldens=np.zeros(ldens.shape).astype(dtype)
    #remember that rdens is in reversed order, i.e. redens[0] lives at the right boundary
    sqrtrdens=np.zeros(rdens.shape).astype(dtype)
    Cvecs=np.zeros(rdens.shape).astype(dtype)

    #get the sqrt(l) and interpolate it; this is needed to get the derivative
    for n in range(ldens.shape[0]):

        l=np.reshape(ldens[n,:],(D,D))
        sqrtl=np.transpose(sqrtm(l))

        r=np.reshape(rdens[rdens.shape[0]-1-n,:],(D,D))
        sqrtr=sqrtm(r)
        
        C=sqrtl.dot(sqrtr)
        sqrtldens[n,:]=np.reshape(sqrtl,D*D)
        #invsqrtldens[n,:]=np.reshape(np.linalg.pinv(sqrtl,rcond=rcond),D*D)
        sqrtrdens[rdens.shape[0]-1-n,:]=np.reshape(sqrtr,D*D)
        #invsqrtrdens[rdens.shape[0]-1-n,:]=np.reshape(np.linalg.pinv(sqrtr,rcond=rcond),D*D)
        Cvecs[n,:]=np.reshape(C,D*D)

    
    GammaQ=np.zeros(Q.shape).astype(Q.dtype)
    GammaR=np.zeros(R.shape).astype(R.dtype)

    Y=np.zeros(R.shape).astype(R.dtype)    
    #now run through all cMPS matrices Q and R and transform them
    for n in range(N):
        sqrtl=np.reshape(sqrtldens[n+1,:],(D,D))
        invsqrtl=np.linalg.pinv(sqrtl,rcond=rcond)

        sqrtr=np.reshape(sqrtrdens[N-n,:],(D,D))
        invsqrtr=np.linalg.pinv(sqrtr,rcond=rcond)
        GammaQ[:,:,n]=invsqrtr.dot(Q[:,:,n]).dot(invsqrtl)
        GammaR[:,:,n]=invsqrtr.dot(R[:,:,n]).dot(invsqrtl)
        Y[:,:,n]=invsqrtr.dot(invsqrtl)

    Ql=np.zeros(Q.shape).astype(Q.dtype)
    Rl=np.zeros(R.shape).astype(R.dtype)
    Qr=np.zeros(Q.shape).astype(Q.dtype)
    Rr=np.zeros(R.shape).astype(R.dtype)

    for n in range(N):
        Cl=np.reshape(Cvecs[n,:],(D,D))
        Cr=np.reshape(Cvecs[n+1,:],(D,D))
        Pl=(Cl.dot(Y[:,:,n])-np.eye(D))/dx[n]
        Pr=(Y[:,:,n].dot(Cr)-np.eye(D))/dx[n]
        Ql[:,:,n]=Cl.dot(GammaQ[:,:,n])+Pl
        Rl[:,:,n]=Cl.dot(GammaR[:,:,n])
        Qr[:,:,n]=GammaQ[:,:,n].dot(Cr)+Pr
        Rr[:,:,n]=GammaR[:,:,n].dot(Cr)

    return Y,GammaQ,GammaR,Ql,Rl,Qr,Rr,Cvecs,ldens,rdens,sqrtldens,sqrtrdens,eta1,eta2


#Q and R are cMPS matrices stored in an array  that will be normalized according to dx
#returns GammaQ, GammaR, Y, the Gamma-content of the cMPS matrices; i.e. Gamma[:,:,0]=Y+dx*GammaQ,Gamma[:,:,1]=sqrt(dx)*GammaR;
#Cvecs is a (len(x),D) array that contains the center matrices; note that Y^-1 and Cvecs are not the same:
#Y[n]=invsqrtr[N-1-n+1].dot(invsqrtl[n+1], i.e. invsrqrtr lives to the left and invsqrtl to the right of the tensor A
#C[n]=sqrtl[n].dot(sqrtl[N-1-n+1]) at the same site of A
def regaugecentralLambda(Q,R,dx,init=None,tol=1E-10,ncv=20,nmax=10000,which='LM',rcond=1E-16):
    D=Q.shape[0]
    N=Q.shape[2]
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    eta1,v1,numeig=UnitcellTMeigs(Q,R,dx,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)

    #normalize the state:
    
    sqeta=np.real(eta1**(-1.0/(2.0*N)))
    for n in range(N):
        phi=(sqeta-1.0)/dx[n]
        Q[:,:,n]=Q[:,:,n]+np.eye(D)*phi+dx[n]*Q[:,:,n]*phi
        R[:,:,n]=R[:,:,n]+dx[n]*R[:,:,n]*phi

    l=np.reshape(v1,(D,D))
    l/=np.trace(l)
    l=(l+herm(l))/2.0
    if dtype=='float64':
        l=np.real(l)
    ldens=UnitcellTransferOperator(direction=1,Q=Q,R=R,dx=dx,vector=np.reshape(l,D*D),returnfull=True)

    eta2,v2,numeig=UnitcellTMeigs(Q,R,dx,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)
    r=np.reshape(v2,(D,D))
    r/=np.trace(r)
    r=(r+herm(r))/2.0
    if dtype=='float64':
        r=np.real(r)

    rdens=UnitcellTransferOperator(direction=-1,Q=Q,R=R,dx=dx,vector=np.reshape(r,D*D),returnfull=True)
    sqrtldens=np.zeros(ldens.shape).astype(dtype)
    #remember that rdens is in reversed order, i.e. redens[0] lives at the right boundary
    sqrtrdens=np.zeros(rdens.shape).astype(dtype)
    Umats=np.zeros((D,D,ldens.shape[0])).astype(dtype)
    Vmats=np.zeros((D,D,ldens.shape[0])).astype(dtype)
    Uvecs=np.zeros(rdens.shape).astype(dtype)
    Vvecs=np.zeros(rdens.shape).astype(dtype)
    lams=np.zeros((rdens.shape[0],D)).astype(float)
    diagU=np.zeros((rdens.shape[0],D)).astype(dtype)
    diagV=np.zeros((rdens.shape[0],D)).astype(dtype)
    diagU0=np.zeros((rdens.shape[0],D)).astype(dtype)
    diagV0=np.zeros((rdens.shape[0],D)).astype(dtype)

    for n in range(ldens.shape[0]):

        l=np.reshape(ldens[n,:],(D,D))
        sqrtl=np.transpose(sqrtm(l))

        r=np.reshape(rdens[rdens.shape[0]-1-n,:],(D,D))
        sqrtr=sqrtm(r)
        np.random.seed(10)
        U,lam,V=np.linalg.svd(sqrtl.dot(sqrtr))        
        #choose diagonal of U to be real and positive:
        sqrtldens[n,:]=np.reshape(sqrtl,D*D)
        #invsqrtldens[n,:]=np.reshape(np.linalg.pinv(sqrtl,rcond=rcond),D*D)
        sqrtrdens[rdens.shape[0]-1-n,:]=np.reshape(sqrtr,D*D)
        #invsqrtrdens[rdens.shape[0]-1-n,:]=np.reshape(np.linalg.pinv(sqrtr,rcond=rcond),D*D)
        lams[n,:]=np.copy(lam)
        Umats[:,:,n]=np.copy(U)
        Vmats[:,:,n]=np.copy(V)


    Uvecs[0,:]=np.reshape(Umats[:,:,0],D**2)
    Vvecs[0,:]=np.reshape(Vmats[:,:,0],D**2)
    diagU0[0,:]=np.diag(Umats[:,:,0])
    diagV0[0,:]=np.diag(Vmats[:,:,0])
    diagU[0,:]=np.diag(Umats[:,:,0])
    diagV[0,:]=np.diag(Vmats[:,:,0])

    dU0=np.diag(Umats[:,:,0])    
    dV0=np.diag(Vmats[:,:,0])
    for n in range(1,ldens.shape[0]):
        #Ov=herm(Umats[:,:,0]).dot(Umats[:,:,n])
        ##for each vectpr U0[:,n], find the state in U1[:,m] with largest overlap , and bring it to position n
        ##indy=np.argmax(Ov,1)
        #indx=np.argmax(Ov,0)
        ##indsy=tuple(np.arange(D),indy)
        ##indsx=tuple(indx,np.arange(D))
        #Uout=np.zeros((D,D)).astype(dtype)
        #Ulist=[]
        #lamnew=[]
        ##Ulist[0]=np.copy(Umats[:,0,n])
        ##lamnew[0]=np.copy(lams[n,0])
        #
        #for m in range(D):
        #    Ulist.append(None)
        #    lamnew.append(None)
        #for y in range(len(indx)):
        #    #redundancy:
        #    numy=np.nonzero(indx==indx[y])[0]
        #    
        #    #if Ulist[indx[y]]==None:
        #    if 
        #    Ulist[indx[y]]=np.copy(Umats[:,y,n])
        #    lamnew[indx[y]]=np.copy(lams[n,y])
        #    elif Ulist[indx[y]]!=None:
        #        bla=1
        #        while Ulist[indx[y]+bla]!=None:
        #            if lamnew[indx[y]]>=lams[n,y]:
        #                Ulist[indx[y]]=np.copy(Umats[:,y,n])
        #                lamnew[indx[y]]=np.copy(lams[n,y])
        #            elif lamnew[indx[y]]>lams[n,y]:

        phase=utils.findPhase2(Umats[:,:,0],Umats[:,:,n],Vmats[:,:,0],Vmats[:,:,n])
        diagU0[n,:]=np.diag(Umats[:,:,n])
        diagV0[n,:]=np.diag(Vmats[:,:,n])
        Umats[:,:,n]=Umats[:,:,n].dot(np.diag(np.exp(1j*phase)))
        Vmats[:,:,n]=np.diag(np.exp(-1j*phase)).dot(Vmats[:,:,n])
        diagU[n,:]=np.diag(Umats[:,:,n])
        diagV[n,:]=np.diag(Vmats[:,:,n])
        Uvecs[n,:]=np.reshape(Umats[:,:,n],D**2)
        Vvecs[n,:]=np.reshape(Vmats[:,:,n],D**2)



    for n in range(Q.shape[2]):
        sqrtl=np.reshape(sqrtldens[n,:],(D,D))
        invsqrtl=np.linalg.pinv(sqrtl,rcond=rcond)
        sqrtr=np.reshape(sqrtrdens[n,:],(D,D))
        invsqrtr=np.linalg.pinv(sqrtr,rcond=rcond)

        V=Vmats[:,:,n]
        U=Umats[:,:,n]
        GammaQ[:,:,n]=V.dot(invsqrtr).dot(Q[:,:,n]).dot(invsqrtl).dot(U)+\
                       V.dot(invsqrtr).dot(invsqrtl).dot(dUdx-phi.dot(U))
        GammaR[:,:,n]=V.dot(invsqrtr).dot(cmps.R[:,:,n]).dot(invsqrtl).dot(U)



    #return Y,GammaQ,GammaR,Ql,Rl,Qr,Rr,Cvecs,ldens,rdens,sqrtldens,sqrtrdens,eta1,eta2
    return Uvecs,Vvecs,lams,diagU,diagV,diagU0,diagV0


#Here is the ordering: Cvecs lives between the MPS tensors
#-------------------------------------------
#                      Q0  Q1  Q2  Q3  Q4  Q5 
#                      R0  R1  R2  R3  R4  R5 
#-------------------------------------------
#                    C0  C1  C2  C3  C4  C5  C6
def computeOrthogonalMPS(Q,R,dx,init=None,tol=1E-10,ncv=20,nmax=10000,rcond=1E-16):
    D=Q.shape[0]
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    Y,GammaQ,GammaR,Ql,Rl,Qr,Rr,Cvecs,ldens,rdens,sqrtldens,sqrtrdens,etal,etar=regaugecentralC(Q,R,dx,init=init,tol=tol,ncv=ncv,nmax=nmax,which='LM',rcond=rcond)
    Cs=np.zeros((D,D,Cvecs.shape[0])).astype(dtype)
    for n in range(Cvecs.shape[0]):
        Cs[:,:,n]=np.reshape(Cvecs[n,:],(D,D))
    return Ql,Rl,Qr,Rr,Cs,Cvecs,ldens,rdens,etal,etar




#mps, ldens rdens are lists of tensors and matrices, respectively
def computeUCsteadyStateHamiltonianGMRES(Q,R,x,LUC,g,mass,mu,init,ldens,rdens,direction,thresh,imax,dtype='float64'):
    D=Q.shape[0]
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'
    dx=utils.computedx(0,x,LUC)
    mps=utils.toMPS(Q,R,dx)
    if direction>0:
        f=np.zeros((D*D)).astype(dtype)
        #F,Flist,xF=FiniteHamiltonianEvolutionLiebLiniger(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        #F,Flist,xF=FiniteHamiltonianEvolutionEulerLiebLiniger(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        F,Flist,xF=FiniteHamiltonianEvolutionEulerLiebLinigerSym(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        h=np.trace(Flist[-1].dot(rdens[-1]))
        inhom=np.reshape(Flist[-1]-h*np.transpose(np.eye(D)),D*D) 
        [k,info]=mf.TDVPGMRESUC(mps,ldens,rdens,inhom,init,thresh,imax,datatype=dtype,direction=direction)
        HL=np.reshape(k,(D,D))
        #The GMRES routine sometimes mixes back some parts of the the discared space |r)(l| into the solution; manually remove it
        HL=HL-np.tensordot(HL,rdens[-1],([0,1],[0,1]))*ldens[-1]
        return HL

    if direction<0:
        f=np.zeros((D*D)).astype(dtype)
        #F,Flist,xF=FiniteHamiltonianEvolutionLiebLiniger(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        #F,Flist,xF=FiniteHamiltonianEvolutionEulerLiebLiniger(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        F,Flist,xF=FiniteHamiltonianEvolutionEulerLiebLinigerSym(Q=Q,R=R,x=x,LUC=LUC,g=g,mass=mass,mu=mu,direction=direction,vector=f)
        h=np.tensordot(ldens[0],Flist[-1],([0,1],[0,1]))
        inhom=np.reshape(Flist[-1]-h*np.transpose(np.eye(D)),D*D) 
        [k,info]=mf.TDVPGMRESUC(mps,ldens,rdens,inhom,init,thresh,imax,datatype=dtype,direction=direction)
        HR=np.reshape(k,(D,D))
        #The GMRES routine sometimes mixes back some parts of the the discared space |r)(l| into the solution; manually remove it
        HR=HR-np.tensordot(ldens[0],HR,([0,1],[0,1]))*rdens[0]
        return HR


#computes the fully left renormalized Hamiltonian H_l from my notes; user can provide rightode, the dominant right reduced density matrices of cmpsl
#on the grid x inside the unit cell; if rinit="None", then it is calculated from scratch.
#returns lists Hl, Hlrenorm containing the normalized H_l and unnormalized H_l
#also returns the same objects in an array format with the same convention as in the cmpsfunctions.py routins
#returns rightdens: 
#mu is a function 
def computeHLLiebLiniger(Ql,Rl,x,L,g,mass,mu,rinit=None,init=None,thresh=1E-8,imax=1000,nmax=1000,tol=1E-10,ncv=20):
    D=Ql.shape[0]
    if Ql[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Ql[0,0,0].dtype==np.complex128:
        dtype='complex128'
    dx=utils.computedx(0,x,L)
    #be sure that all dxs are identical
    if not all(np.abs(dx-dx[0])<1E-10):
        sys.exit('disccmpsfunctions.pu computeHLLiebLiniger: x is not equally spaced between 0 and L, i.e. the resulting dx are not uniform')

    mpo=Hams.projectedLiebLinigermpo3(mu(x),g,mass,dx,False,dtype=dtype)
    #todo: add a check that Ql,Rl is left-isometric
    leftdens=[]
    #rightdens=[]

    mpsl=utils.toMPS(Ql,Rl,dx)
    if rinit==None:
        etar,vr,numeig=UnitcellTMeigs(Ql,Rl,dx,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which='LM')
        #etar,vr,numeig=mfgo.UnitcellTMeigs(mpsl,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which='LM')
        r=np.reshape(vr,(D,D))
        r/=np.trace(r)
        r=(r+herm(r))/2.0
        if dtype=='float64':
            r=np.real(r)
    else:
        r=rinit
    rightdens=mf.computeDensity(r,mpsl,direction=-1,dtype=dtype)        
    #rightdens=UnitcellTransferOperator(direction=-1,Q=Q,R=R,dx=dx,returnfull=True,vector=np.reshape(r,D*D))
    for n in range(len(rightdens)):
        leftdens.append(np.eye(D))
        

    hlv=np.reshape(mf.computeUCsteadyStateHamiltonianGMRES(mpsl,mpo,init=init,ldens=leftdens,rdens=rightdens,direction=1,thresh=thresh,imax=imax,dtype=dtype)[:,:,0],(D*D))
    #HLarray,HL,x_hl=FiniteHamiltonianEvolutionLiebLiniger(Ql,Rl,x,L,g,mass,mu,direction=1,vector=hlv)

    #the following uses a local chemical potential instead of a two site split up
    #hlv=np.reshape(computeUCsteadyStateHamiltonianGMRES(Ql,Rl,x,L,g,mass,mu,init=init,ldens=leftdens,rdens=rightdens,direction=1,thresh=thresh,imax=imax,dtype=dtype),(D*D))
    HLarray,HL,x_hl=FiniteHamiltonianEvolutionEulerLiebLinigerSym(Ql,Rl,x,L,g,mass,mu,direction=1,vector=hlv)
    #HLarray,HL,x_hl=FiniteHamiltonianEvolutionEulerLiebLiniger(Ql,Rl,x,L,g,mass,mu,direction=1,vector=hlv)
    #LMPO0=mf.computeUCsteadyStateHamiltonianGMRES(mpsl,mpo,init=init,ldens=leftdens,rdens=rightdens,direction=1,thresh=thresh,imax=imax,dtype=dtype)
    #LMPO=[]
    #HL=[]
    #HLarray=np.zeros((len(mpsl)+1,D**2)).astype(dtype)
    #LMPO.append(np.copy(LMPO0))
    #HL.append(np.copy(LMPO0[:,:,0]))
    #
    #for n in range(len(mpsl)):
    #    LMPO.append(np.copy(mf.addLayer(LMPO[-1],mpsl[n],mpo[n],mpsl[n],1)))
    #    HL.append(np.copy(LMPO[-1][:,:,0]))
    #    HLarray[n,:]=np.reshape(HL[-1],D*D)

    HLrenormarray=np.zeros(HLarray.shape).astype(dtype)
    HLrenorm=[]
    for n in range(len(HL)):
        hlt=HL[n]
        HLrenorm.append(hlt-np.tensordot(hlt,rightdens[n],([0,1],[0,1]))*np.eye(D))
        HLrenormarray[n,:]=np.reshape(HLrenorm[-1],D*D)
    return HLrenorm,HL,HLrenormarray,HLarray,rightdens




#computes the fully right renormalized Hamiltonian H_r from my notes; user can provide linit, the dominant right reduced density matrices of cmpsl
#on the grid x inside the unit cell; if linit="None", then it is calculated from scratch.
#returns lists Hr, Hrrenorm containing the normalized H_r and unnormalized H_r
#also returns the same objects in an array format with the same convention as in the cmpsfunctions.py routins
#returns leftdens: 
#mu is a function 
def computeHRLiebLiniger(Qr,Rr,x,L,g,mass,mu,linit=None,init=None,thresh=1E-8,imax=1000,nmax=1000,tol=1E-10,ncv=20):
    D=Qr.shape[0]
    if Qr[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Qr[0,0,0].dtype==np.complex128:
        dtype='complex128'

    dx=utils.computedx(0,x,L)
    #be sure that all dxs are identical
    if not all(np.abs(dx-dx[0])<1E-10):
        sys.exit('disccmpsfunctions.pu computeHLRiebLiniger: x is not equally spaced between 0 and L, i.e. the resulting dx are not uniform')

    mpo=Hams.projectedLiebLinigermpo3(mu(x),g,mass,dx,False,dtype=dtype)
    mpsr=utils.toMPS(Qr,Rr,dx)

    rightdens=[]

    if linit==None:
        etal,vl,numeig=UnitcellTMeigs(Qr,Rr,dx,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which='LM')
        l=np.reshape(vl,(D,D))
        l/=np.trace(l)
        l=(l+herm(l))/2.0
        if dtype=='float64':
            l=np.real(l)
    else:
        l=linit

    leftdens=mf.computeDensity(l,mpsr,direction=1,dtype=dtype)        
    #leftdens2=UnitcellTransferOperator(direction=1,Q=Q,R=R,dx=dx,returnfull=True,vector=np.reshape(l,D*D))

    for n in range(len(leftdens)):
        rightdens.append(np.eye(D))
        

    hrv=np.reshape(mf.computeUCsteadyStateHamiltonianGMRES(mpsr,mpo,init=init,ldens=leftdens,rdens=rightdens,direction=-1,thresh=thresh,imax=imax,dtype=dtype)[:,:,-1],(D*D))
    #HRarray,HR,x_hr=FiniteHamiltonianEvolutionLiebLiniger(Qr,Rr,x,L,g,mass,mu,direction=-1,vector=hrv)

    #hrv=np.reshape(computeUCsteadyStateHamiltonianGMRES(Qr,Rr,x,L,g,mass,mu,init=init,ldens=leftdens,rdens=rightdens,direction=-1,thresh=thresh,imax=imax,dtype=dtype),(D*D))
    HRarray,HR,x_hr=FiniteHamiltonianEvolutionEulerLiebLinigerSym(Qr,Rr,x,L,g,mass,mu,direction=-1,vector=hrv)
    #HRarray,HR,x_hr=FiniteHamiltonianEvolutionEulerLiebLiniger(Qr,Rr,x,L,g,mass,mu,direction=-1,vector=hrv)

    #RMPO0=mf.computeUCsteadyStateHamiltonianGMRES(mpsr,mpo,init=init,ldens=leftdens,rdens=rightdens,direction=-1,thresh=thresh,imax=imax,dtype=dtype)
    #RMPO=[]
    #HR=[]
    HRrenorm=[]
    for n in range(len(mpsr)+1):
        HRrenorm.append(None)
    #    RMPO.append(None)
    #    HR.append(None)

    #RMPO[-1]=np.copy(RMPO0)
    #HRarray=np.zeros((len(mpsr)+1,D**2)).astype(dtype)
    #HR[-1]=np.copy(RMPO0[:,:,-1])
    #for n in range(len(mpsr)-1,-1,-1):
    #    RMPO[n]=mf.addLayer(RMPO[n+1],mpsr[n],mpo[n],mpsr[n],-1)
    #    HR[n]=np.copy(RMPO[n][:,:,-1])
    #    HRarray[n,:]=np.reshape(HR[n],D*D)

    HRrenormarray=np.zeros(HRarray.shape).astype(dtype)
    for n in range(len(HR)):
        HRrenorm[n]=HR[n]-np.tensordot(leftdens[len(HR)-1-n],HR[n],([0,1],[0,1]))*np.eye(D)
        HRrenormarray[n,:]=np.reshape(HRrenorm[n],D*D)
    return HRrenorm,HR,HRrenormarray,HRarray,leftdens


#computes the mixed transer matrix vector product vector*E_A^B or E_A^B*vector
#A and B are mps tensors of dimension (chi1 x chi2 x d), from which the transfer matrix can computed if A=B; 
#B is always the upper matrix, A is always the lower one
#direction > 0 does a left-side product; direction < 0 does a right side product;
#vector is a chi1 x chi1 (direction > 0) or chi2 x chi2 (direction < 0) matrix, given in VECTOR format!
#returns a vector 
def TransferOperator(direction,Q,R,dx,vector):
    D=Q.shape[0]
    mat=np.reshape(vector,(D,D))
    if direction>0:
        return np.reshape(mat+dx*(np.transpose(Q).dot(mat)+mat.dot(np.conj(Q))+np.transpose(R).dot(mat).dot(np.conj(R)))+dx**2*(np.transpose(Q).dot(mat).dot(np.conj(Q))),(D*D))
    if direction<0:
        return np.reshape(mat+dx*(Q.dot(mat)+mat.dot(herm(Q))+R.dot(mat).dot(herm(R)))+dx**2*(Q.dot(mat).dot(herm(Q))),(D*D))



#takes a vector, returns a vector
def UnitcellTransferOperator(direction,Q,R,dx,returnfull,vector):
    D=Q.shape[0]
    N=Q.shape[2]
    if returnfull==False:
        x=np.copy(vector)
        if direction>0:
            for n in range(N):
                x=TransferOperator(direction,Q[:,:,n],R[:,:,n],dx[n],x)
            return x
        if direction<0:
            for n in range(N-1,-1,-1):
                x=TransferOperator(direction,Q[:,:,n],R[:,:,n],dx[n],x)
            return x
    if returnfull==True:
        x=np.zeros((N+1,D**2)).astype(vector.dtype)
        x[0,:]=np.copy(vector)
        if direction>0:
            for n in range(N):
                x[n+1,:]=TransferOperator(direction,Q[:,:,n],R[:,:,n],dx[n],x[n,:])
            return x
        if direction<0:
            for n in range(N-1,-1,-1):
                x[N-1-n+1,:]=TransferOperator(direction,Q[:,:,n],R[:,:,n],dx[n],x[N-1-n,:])
            return x


#returns the unitcellTO eigenvector with 'LR'
def UnitcellTMeigs(Q,R,dx,direction,numeig,init=None,datatype='float64',nmax=800,tolerance=1e-12,ncv=10,which='LM'):
    #define the matrix vector product mv(v) using functools.partial and GeneralizedMatrixVectorProduct(direction,A,B,vector):
    D=Q.shape[0]

    mv=fct.partial(UnitcellTransferOperator,*[direction,Q,R,dx,False])
    LOP=LinearOperator((D**2,D**2),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)
    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print 'found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig)
        print eta
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
    return eta[m],np.reshape(vec[:,m],D*D),numeig



def FiniteHamiltonianEvolutionLiebLiniger(Q,R,x,LUC,g,mass,mu,direction,vector):
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    dx=utils.computedx(0,x,LUC)
    mps=utils.toMPS(Q,R,dx)

    mpo=Hams.projectedLiebLinigermpo3(mu(x),g,mass,dx,False,dtype=dtype)
    NUC=len(mps)
    
    D=Q.shape[0]
    [B1,B2,d1,d2]=np.shape(mpo[NUC-1])
    if direction>0:
        HL=np.zeros((len(x)+1,D**2)).astype(dtype)
        mpol=np.zeros((1,B2,d1,d2),dtype=dtype)
        mpol[0,:,:,:]=mpo[NUC-1][-1,:,:,:]
        L=mf.initializeLayer(mps[NUC-1],np.eye(D),mps[NUC-1],mpol,1)
        L[:,:,0]=np.reshape(vector,(D,D))
        HL[0,:]=np.reshape(L[:,:,0],D*D)
        HLlist=[]
        HLlist.append(np.reshape(HL[0,:],(D,D)))
        for n in range(len(mps)):
            L=mf.addLayer(L,mps[n],mpo[n],mps[n],1)    
            HL[n+1,:]=np.reshape(L[:,:,0],D*D)
            HLlist.append(np.reshape(HL[n+1,:],(D,D)))


        return HL,HLlist,np.linspace(0,LUC,len(x)+1)

    if direction<0:
        #HR=[]
        HR=np.zeros((len(x)+1,D**2)).astype(dtype)
        mpor=np.zeros((B1,1,d1,d2),dtype=dtype)
        mpor[:,0,:,:]=mpo[0][:,0,:,:]
        R=mf.initializeLayer(mps[0],np.eye(D),mps[0],mpor,-1)
        R[:,:,-1]=np.reshape(vector,(D,D))
        #HR[len(x),:]=np.reshape(R[:,:,-1],D*D)
        HRlist=[]
        HR[0,:]=np.reshape(R[:,:,-1],D*D)
        HRlist.append(np.reshape(HR[0,:],(D,D)))
        #HR.append(np.copy(R[:,:,-1]))

        for n in range(len(mps)-1,-1,-1):
            R=mf.addLayer(R,mps[n],mpo[n],mps[n],-1)    
            HR[len(mps)-1-n+1,:]=np.reshape(R[:,:,-1],D*D)
            HRlist.append(np.reshape(HR[len(mps)-1-n+1,:],(D,D)))
            #HR.append(np.copy(R[:,:,-1]))
        return HR,HRlist,np.linspace(0,LUC,len(x)+1)




#takes an initail vector and evolves it over a unit cell given by tensors Q,R at positions x. LUC is the length of the unit cell.
#g, mass, mu are Hamiltonian parameters (mu is a function); direction>0 evolves from left to right, direction<0 evolves from right to left
#returns the Hamiltonian environments in array form (len(x)+1,D*D) (consistent with the conventions in cmpsfunctions.py) and in a list, as well 
#as the points x where they are defined (note that these are different from input x)
#if direction<0, the return array HR and list HRlist contain the environment in reversed order, i.e. HL[0,:] lives at the right boundary, and same for HLlist[0]
def FiniteHamiltonianEvolutionEulerLiebLiniger(Q,R,x,LUC,g,mass,mu,direction,vector):
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    dx=utils.computedx(0,x,LUC)
    D=Q.shape[0]
    if direction>0:
        HL=np.zeros((len(x)+1,D**2)).astype(dtype)
        HL[0,:]=vector
        K=(np.eye(D)+dx[-1]*Q[:,:,-1]).dot(np.sqrt(dx[0])*R[:,:,0])-np.sqrt(dx[-1])*R[:,:,-1].dot(np.eye(D)+dx[0]*Q[:,:,0])

        Kin=1.0/(2.0*mass*(dx[-1]**2))*(herm(K).dot(K))
        Int=g*dx[-1]*herm(R[:,:,0]).dot(herm(R[:,:,-1])).dot(R[:,:,-1]).dot(R[:,:,0])
        Pot=dx[0]*mu(x[0])*herm(R[:,:,0]).dot(R[:,:,0])
        flocal=np.reshape(np.transpose(Kin+Int+Pot),(D*D))

        HL[1,:]=TransferOperator(direction=1,Q=Q[:,:,0],R=R[:,:,0],dx=dx[0],vector=HL[0,:])+flocal

        for n in range(1,Q.shape[2]):
            K=(np.eye(D)+dx[n-1]*Q[:,:,n-1]).dot(np.sqrt(dx[n])*R[:,:,n])-np.sqrt(dx[n-1])*R[:,:,n-1].dot(np.eye(D)+dx[n]*Q[:,:,n])
            Kin=1.0/(2.0*mass*(dx[n-1]**2))*(herm(K).dot(K))
            Int=g*dx[n-1]*herm(R[:,:,n]).dot(herm(R[:,:,n-1])).dot(R[:,:,n-1]).dot(R[:,:,n])
            Pot=dx[n]*mu(x[n])*herm(R[:,:,n]).dot(R[:,:,n])
            flocal=np.reshape(np.transpose(Kin+Int+Pot),(D*D))
            HL[n+1,:]=TransferOperator(direction=1,Q=Q[:,:,n],R=R[:,:,n],dx=dx[n],vector=HL[n,:])+flocal
            
        HLlist=[]
        for n in range(len(HL)):
            HLlist.append(np.reshape(HL[n,:],(D,D)))

        return HL,HLlist,np.linspace(0,LUC,len(x)+1)
 
    if direction<0:
        HR=np.zeros((len(x)+1,D**2)).astype(dtype)
        HR[0,:]=vector
        K=(np.eye(D)+dx[-1]*Q[:,:,-1]).dot(np.sqrt(dx[0])*R[:,:,0])-np.sqrt(dx[-1])*R[:,:,-1].dot(np.eye(D)+dx[0]*Q[:,:,0])

        Kin=1.0/(2.0*mass*(dx[0]**2))*(K.dot(herm(K)))
        Int=g*dx[0]*R[:,:,-1].dot(R[:,:,0]).dot(herm(R[:,:,-1].dot(R[:,:,0])))
        Pot=dx[-1]*mu(x[-1])*herm(R[:,:,-1]).dot(R[:,:,-1])
        flocal=np.reshape(Kin+Int+Pot,(D*D))

        HR[1,:]=TransferOperator(direction=-1,Q=Q[:,:,-1],R=R[:,:,-1],dx=dx[-1],vector=HR[0,:])+flocal

        for n in range(Q.shape[2]-1,0,-1):
            K=(np.eye(D)+dx[n-1]*Q[:,:,n-1]).dot(np.sqrt(dx[n])*R[:,:,n])-np.sqrt(dx[n-1])*R[:,:,n-1].dot(np.eye(D)+dx[n]*Q[:,:,n])
            Kin=1.0/(2.0*mass*(dx[n-1]**2))*(K.dot(herm(K)))
            Int=g*dx[n-1]*R[:,:,n-1].dot(R[:,:,n]).dot(herm(R[:,:,n-1].dot(R[:,:,n])))
            Pot=dx[n-1]*mu(x[n-1])*R[:,:,n-1].dot(herm(R[:,:,n-1]))
            flocal=np.reshape(Kin+Int+Pot,(D*D))
            HR[HR.shape[0]-1-n+1,:]=TransferOperator(direction=-1,Q=Q[:,:,n-1],R=R[:,:,n-1],dx=dx[n-1],vector=HR[HR.shape[0]-1-n,:])+flocal
            
        HRlist=[]
        for n in range(len(HR)):
            HRlist.append(None)
        for n in range(len(HR)):
            HRlist[n]=np.reshape(HR[n,:],(D,D))

        return HR,HRlist,np.linspace(0,LUC,len(x)+1)

def FiniteHamiltonianEvolutionEulerLiebLinigerSym(Q,R,x,LUC,g,mass,mu,direction,vector):
    if Q[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Q[0,0,0].dtype==np.complex128:
        dtype='complex128'

    dx=utils.computedx(0,x,LUC)
    D=Q.shape[0]
    if direction>0:
        HL=np.zeros((len(x)+1,D**2)).astype(dtype)
        HL[0,:]=vector
        K=(np.eye(D)+dx[-1]*Q[:,:,-1]).dot(np.sqrt(dx[0])*R[:,:,0])-np.sqrt(dx[-1])*R[:,:,-1].dot(np.eye(D)+dx[0]*Q[:,:,0])

        Kin=np.transpose(1.0/(2.0*mass*(dx[-1]**2))*(herm(K).dot(K)))
        Int=np.transpose(g*dx[-1]*herm(R[:,:,0]).dot(herm(R[:,:,-1])).dot(R[:,:,-1]).dot(R[:,:,0]))

        Pot1vec=np.reshape(np.transpose(dx[-1]*mu(x[-1])/2.0*herm(R[:,:,-1]).dot(R[:,:,-1])),D*D)
        Pot1=np.reshape(TransferOperator(direction=1,Q=Q[:,:,0],R=R[:,:,0],dx=dx[0],vector=Pot1vec),(D,D))
        Pot2=np.transpose(dx[0]*mu(x[0])/2.0*herm(R[:,:,0]).dot(R[:,:,0]))
        flocal=np.reshape(Kin+Int+Pot1+Pot2,(D*D))

        HL[1,:]=TransferOperator(direction=1,Q=Q[:,:,0],R=R[:,:,0],dx=dx[0],vector=HL[0,:])+flocal

        for n in range(1,Q.shape[2]):
            K=(np.eye(D)+dx[n-1]*Q[:,:,n-1]).dot(np.sqrt(dx[n])*R[:,:,n])-np.sqrt(dx[n-1])*R[:,:,n-1].dot(np.eye(D)+dx[n]*Q[:,:,n])
            Kin=np.transpose(1.0/(2.0*mass*(dx[n-1]**2))*(herm(K).dot(K)))
            Int=np.transpose(g*dx[n-1]*herm(R[:,:,n]).dot(herm(R[:,:,n-1])).dot(R[:,:,n-1]).dot(R[:,:,n]))

            Pot1vec=np.reshape(np.transpose(dx[n-1]*mu(x[n-1])/2.0*herm(R[:,:,n-1]).dot(R[:,:,n-1])),D*D)
            Pot1=np.reshape(TransferOperator(direction=1,Q=Q[:,:,n],R=R[:,:,n],dx=dx[n],vector=Pot1vec),(D,D))
            Pot2=np.transpose(dx[n]*mu(x[n])/2.0*herm(R[:,:,n]).dot(R[:,:,n]))
            flocal=np.reshape(Kin+Int+Pot1+Pot2,(D*D))
            HL[n+1,:]=TransferOperator(direction=1,Q=Q[:,:,n],R=R[:,:,n],dx=dx[n],vector=HL[n,:])+flocal
            
        HLlist=[]
        for n in range(len(HL)):
            HLlist.append(np.reshape(HL[n,:],(D,D)))

        return HL,HLlist,np.linspace(0,LUC,len(x)+1)
 
    if direction<0:
        HR=np.zeros((len(x)+1,D**2)).astype(dtype)
        HR[0,:]=vector
        K=(np.eye(D)+dx[-1]*Q[:,:,-1]).dot(np.sqrt(dx[0])*R[:,:,0])-np.sqrt(dx[-1])*R[:,:,-1].dot(np.eye(D)+dx[0]*Q[:,:,0])

        Kin=1.0/(2.0*mass*(dx[0]**2))*(K.dot(herm(K)))
        Int=g*dx[0]*R[:,:,-1].dot(R[:,:,0]).dot(herm(R[:,:,-1].dot(R[:,:,0])))
        Pot1vec=np.reshape(dx[0]*mu(x[0])/2.0*R[:,:,0].dot(herm(R[:,:,0])),D*D)
        Pot1=np.reshape(TransferOperator(direction=-1,Q=Q[:,:,-1],R=R[:,:,-1],dx=dx[-1],vector=Pot1vec),(D,D))
        Pot2=dx[-1]*mu(x[-1])/2.0*R[:,:,-1].dot(herm(R[:,:,-1]))
        flocal=np.reshape(Kin+Int+Pot1+Pot2,(D*D))

        HR[1,:]=TransferOperator(direction=-1,Q=Q[:,:,-1],R=R[:,:,-1],dx=dx[-1],vector=HR[0,:])+flocal

        for n in range(Q.shape[2]-1,0,-1):
            K=(np.eye(D)+dx[n-1]*Q[:,:,n-1]).dot(np.sqrt(dx[n])*R[:,:,n])-np.sqrt(dx[n-1])*R[:,:,n-1].dot(np.eye(D)+dx[n]*Q[:,:,n])
            Kin=1.0/(2.0*mass*(dx[n-1]**2))*(K.dot(herm(K)))
            Int=g*dx[n-1]*R[:,:,n-1].dot(R[:,:,n]).dot(herm(R[:,:,n-1].dot(R[:,:,n])))
            Pot1vec=np.reshape(dx[n]*mu(x[n])/2.0*R[:,:,n].dot(herm(R[:,:,n])),D*D)
            Pot1=np.reshape(TransferOperator(direction=-1,Q=Q[:,:,n-1],R=R[:,:,n-1],dx=dx[n-1],vector=Pot1vec),(D,D))
            Pot2=dx[n-1]*mu(x[n-1])/2.0*R[:,:,n-1].dot(herm(R[:,:,n-1]))
            #Pot=dx[n-1]*mu(x[n-1])*R[:,:,n-1].dot(herm(R[:,:,n-1]))
            flocal=np.reshape(Kin+Int+Pot1+Pot2,(D*D))
            HR[HR.shape[0]-1-n+1,:]=TransferOperator(direction=-1,Q=Q[:,:,n-1],R=R[:,:,n-1],dx=dx[n-1],vector=HR[HR.shape[0]-1-n,:])+flocal
            
        HRlist=[]
        for n in range(len(HR)):
            HRlist.append(None)
        for n in range(len(HR)):
            HRlist[n]=np.reshape(HR[n,:],(D,D))

        return HR,HRlist,np.linspace(0,LUC,len(x)+1)


#C is a VectorFunction of the center-matrix C=sqrtm(l(x))*sqrtm(r(x))
#Hl and Hr are just arrays containing the left and right renormalized Hamiltonians at the points cmpsl.x (=cmpsr.x).
#They have a Hl.shape=Hr.shape=(len(cmpsl.x),cmpsl.D)=(len(cmpsr.x),cmpsr.D).
#(For debugging reasons, it might be better to pass them as VectorFunctions)
#Note that Hr is in reversed order, i.e. Hr[0,:] lives at the right boundary, and Hl[0,:] ar the left one
#returns: (D,D,len(cmpsl.x))-tensors V,W which contain the gradient of Q_l and R_l. 
def UpdateLiebLinigerleft(Ql,Rl,Qr,Rr,C,x,L,Hl,Hr,g,mass,mu,rcond=1E-10):
    if Ql[0,0,0].dtype==np.float64:
        dtype='float64'
    elif Ql[0,0,0].dtype==np.complex128:
        dtype='complex128'

    D=Ql.shape[0]
    N=Ql.shape[2]
    dx=utils.computedx(0,x,L)
    #now calculate the update for R_l(x_i)C(x_i) for every x_i:
    V=np.zeros((D,D,len(x))).astype(dtype)
    W=np.zeros((D,D,len(x))).astype(dtype)

    gradnorm=np.zeros((len(x)))


    #left boundary
    Ql_=Ql[:,:,-1]
    Rl_=Rl[:,:,-1]
    Qc_=Ql[:,:,0].dot(C[:,:,1])
    Rc_=Rl[:,:,0].dot(C[:,:,1])
    Qr_=Qr[:,:,1]
    Rr_=Rr[:,:,1]
    Cinv_=np.linalg.pinv(C[:,:,1])
    Hl_=Hl[0]
    Hr_=Hr[len(Hr)-2] #the index here is different from the cmpsfunctions.py because Hl and Hr have one more entry than Ql,Rl,Qr and Rr here.
    #V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,1],Qr_,Rr_,x[0],dx[0],Hl_,Hr_,g,mass,mu(x[0]))
    V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,1],Qr_,Rr_,x[-1],x[0],x[1],dx[0],Hl_,Hr_,g,mass,mu)

    V[:,:,0]=V_.dot(Cinv_)
    W[:,:,0]=W_.dot(Cinv_)
    
    gradnorm[0]=np.real(np.sqrt(np.trace(herm(V_).dot(V_)+herm(W_).dot(W_))))

    #the bulk updates
    for n in range(1,len(x)-1):
        Ql_=Ql[:,:,n-1]
        Rl_=Rl[:,:,n-1]
        Qc_=Ql[:,:,n].dot(C[:,:,n+1])
        Rc_=Rl[:,:,n].dot(C[:,:,n+1])
        Qr_=Qr[:,:,n+1]
        Rr_=Rr[:,:,n+1]
        Cinv_=np.linalg.pinv(C[:,:,n+1],rcond=rcond)
        Hl_=Hl[n]
        Hr_=Hr[len(Hr)-1-n-1] #the index here is different from the cmpsfunctions.py because Hl and Hr have one more entry than Ql,Rl,Qr and Rr here.
        #V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,n+1],Qr_,Rr_,x[n],dx[n],Hl_,Hr_,g,mass,mu(x[n]))
        V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,n+1],Qr_,Rr_,x[n-1],x[n],x[n+1],dx[n],Hl_,Hr_,g,mass,mu)

        V[:,:,n]=V_.dot(Cinv_)
        W[:,:,n]=W_.dot(Cinv_)

        gradnorm[n]=np.real(np.sqrt(np.trace(herm(V_).dot(V_)+herm(W_).dot(W_))))


    #right boundary
    n=len(x)-1
    Ql_=Ql[:,:,n-1]
    Rl_=Rl[:,:,n-1]
    Qc_=Ql[:,:,n].dot(C[:,:,n+1])
    Rc_=Rl[:,:,n].dot(C[:,:,n+1])
    Qr_=Qr[:,:,0]
    Rr_=Rr[:,:,0]
    Cinv_=np.linalg.pinv(C[:,:,n+1])
    Hl_=Hl[n]
    Hr_=Hr[0] #the index here is different from the cmpsfunctions.py because Hl and Hr have one more entry than Ql,Rl,Qr and Rr here.
    #V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,n+1],Qr_,Rr_,x[n],dx[n],Hl_,Hr_,g,mass,mu(x[n]))
    V_,W_=LocalUpdateLiebLiniger(Ql_,Rl_,Qc_,Rc_,C[:,:,n+1],Qr_,Rr_,x[n-1],x[n],x[0],dx[n],Hl_,Hr_,g,mass,mu)

    V[:,:,n]=V_.dot(Cinv_)
    W[:,:,n]=W_.dot(Cinv_)
    
    gradnorm[n]=np.real(np.sqrt(np.trace(herm(V_).dot(V_)+herm(W_).dot(W_))))
    
    return V,W,gradnorm
    


#def LocalUpdateLiebLiniger(Ql,Rl,Qc,Rc,C,Qr,Rr,xl,xc,xr,dx,HLrenorm,HRrenorm,g,mass,mu):
#
#    WK1=1.0/(2.0*mass)*((2*Rc-Rl.dot(C)-C.dot(Rr))/dx**2+(Ql.dot(Rc)-Rl.dot(Qc)-Qc.dot(Rr)+Rc.dot(Qr))/dx)
#    WK2=1.0/(2.0*mass)*(  ( herm(Ql).dot(Rc-Rl.dot(C))-(C.dot(Rr)-Rc).dot(herm(Qr)) )/dx+\
#                          herm(Ql).dot(Ql.dot(Rc)-Rl.dot(Qc))-(Qc.dot(Rr)-Rc.dot(Qr)).dot(herm(Qr)))
#    VK=1.0/(2.0*mass)*(  (-herm(Rl).dot(Rc-Rl.dot(C))+(C.dot(Rr)-Rc).dot(herm(Rr))   )/dx -  herm(Rl).dot(Ql.dot(Rc)-Rl.dot(Qc))+(Qc.dot(Rr)-Rc.dot(Qr)).dot(herm(Rr)))
#    
#    WE=(np.tensordot(HLrenorm,Rc,([0],[0]))+np.tensordot(Rc,HRrenorm,([1],[0])))
#    VE=(np.tensordot(HLrenorm,C+dx*Qc,([0],[0]))+np.tensordot(C+dx*Qc,HRrenorm,([1],[0])))
#    
#
#    WI=g*(herm(Rl).dot(Rl).dot(Rc)+Rc.dot(Rr).dot(herm(Rr)))
#    #use a local chemical potential, i.e. no split up between two sites
#    #WP=mu*Rc
#    
#
#    WP=mu()
#    return VK+VE,WK1+WK2+WE+WI+WP
       

def LocalUpdateLiebLiniger(Ql,Rl,Qc,Rc,C,Qr,Rr,xl,xc,xr,dx,HLrenorm,HRrenorm,g,mass,mu):

    WK1=1.0/(2.0*mass)*((2*Rc-Rl.dot(C)-C.dot(Rr))/dx**2+(Ql.dot(Rc)-Rl.dot(Qc)-Qc.dot(Rr)+Rc.dot(Qr))/dx)
    WK2=1.0/(2.0*mass)*(  ( herm(Ql).dot(Rc-Rl.dot(C))-(C.dot(Rr)-Rc).dot(herm(Qr)) )/dx+\
                          herm(Ql).dot(Ql.dot(Rc)-Rl.dot(Qc))-(Qc.dot(Rr)-Rc.dot(Qr)).dot(herm(Qr)))
    VK=1.0/(2.0*mass)*(  (-herm(Rl).dot(Rc-Rl.dot(C))+(C.dot(Rr)-Rc).dot(herm(Rr))   )/dx -  herm(Rl).dot(Ql.dot(Rc)-Rl.dot(Qc))+(Qc.dot(Rr)-Rc.dot(Qr)).dot(herm(Rr)))
    
    WE=(np.tensordot(HLrenorm,Rc,([0],[0]))+np.tensordot(Rc,HRrenorm,([1],[0])))
    VE=(np.tensordot(HLrenorm,C+dx*Qc,([0],[0]))+np.tensordot(C+dx*Qc,HRrenorm,([1],[0])))
    

    WI=g*(herm(Rl).dot(Rl).dot(Rc)+Rc.dot(Rr).dot(herm(Rr)))

    #WP=mu(xc)*Rc
    WP=mu(xl)/2.0*dx*(herm(Rl).dot(Rl).dot(Rc))+mu(xc)*Rc+mu(xr)/2.0*dx*(Rc.dot(Rr).dot(herm(Rr)))
    VP=mu(xl)/2.0*dx*(herm(Rl).dot(Rl).dot(C+dx*Qc))+mu(xr)/2.0*dx*((C+dx*Qc).dot(Rr).dot(herm(Rr)))
    return VK+VE+VP,WK1+WK2+WE+WI+WP
       

