#!/usr/bin/env python
from sys import stdout
import numpy as np
import time
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
import Hamiltonians as H
import mpsfunctions as mf
import utils.utilities as utils
import warnings
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

#This module contains some special versions of mps-routines; the module largely parallels some
#splinemps module functions. Most of them are discrete versions of the continuous-limit functions;
#the regauge routines for unit-cell mps coded in this module have some special properties which
#are usually not needed in the lattice case.
def regaugecentralC(mps,init=None,tol=1E-10,ncv=20,nmax=10000,which='LM',rcond=1E-16):
    D=mps[0].shape[0]
    d=mps[0].shape[2]
    if mps[0][0,0,0].dtype==np.float64:
        dtype='float64'
    elif mps[0][0,0,0].dtype==np.complex128:
        dtype='complex128'

    eta1,v1,numeig=UnitcellTMeigs(mps,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)

    #normalize the state:
    phi=np.real(eta1**(-1.0/(2.0*len(mps))))
    for n in range(len(mps)):
        mps[n]*=phi

    l=np.reshape(v1,(D,D))
    l/=np.trace(l)
    l=(l+herm(l))/2.0
    if dtype=='float64':
        l=np.real(l)
    ldens=UnitcellTransferOperator(direction=1,mps=mps,vector=np.reshape(l,D*D),returnfull=True)

    eta2,v2,numeig=UnitcellTMeigs(mps,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)
    r=np.reshape(v2,(D,D))
    r/=np.trace(r)
    r=(r+herm(r))/2.0
    if dtype=='float64':
        r=np.real(r)

    rdens=UnitcellTransferOperator(direction=-1,mps=mps,vector=np.reshape(r,D*D),returnfull=True)
    sqrtldens=np.zeros(ldens.shape).astype(dtype)
    #remember that rdens is in reversed order, i.e. redens[0] lives at the right boundary
    sqrtrdens=np.zeros(rdens.shape).astype(dtype)
    Cvecs=np.zeros(rdens.shape).astype(dtype)


    for n in range(ldens.shape[0]):

        l=np.reshape(ldens[n,:],(D,D))
        sqrtl=np.transpose(sqrtm(l))

        r=np.reshape(rdens[rdens.shape[0]-1-n,:],(D,D))
        sqrtr=sqrtm(r)

        C=sqrtl.dot(sqrtr)
        sqrtldens[n,:]=np.reshape(sqrtl,D*D)
        sqrtrdens[rdens.shape[0]-1-n,:]=np.reshape(sqrtr,D*D)
        Cvecs[n,:]=np.reshape(C,D*D)
    
    Gamma=[]
    #now run through all cMPS matrices Q and R and transform them
    for n in range(len(mps)):
        sqrtl=np.reshape(sqrtldens[n+1,:],(D,D))
        invsqrtl=np.linalg.pinv(sqrtl,rcond=rcond)

        sqrtr=np.reshape(sqrtrdens[len(mps)-n,:],(D,D))
        invsqrtr=np.linalg.pinv(sqrtr,rcond=rcond)
        G=np.zeros((D,D,d)).astype(dtype)
        for m in range(d):
            G[:,:,m]=invsqrtr.dot(mps[n][:,:,m]).dot(invsqrtl)
        Gamma.append(np.copy(G))
    mpsl=[]
    mpsr=[]
    for n in range(len(mps)):
        Cl=np.reshape(Cvecs[n,:],(D,D))
        Cr=np.reshape(Cvecs[n+1,:],(D,D))
        ml=np.zeros((D,D,d)).astype(dtype)
        mr=np.zeros((D,D,d)).astype(dtype)
        for m in range(d):
            ml[:,:,m]=Cl.dot(Gamma[n][:,:,m])
            mr[:,:,m]=Gamma[n][:,:,m].dot(Cr)
        mpsl.append(np.copy(ml))
        mpsr.append(np.copy(mr))

    return Gamma,mpsl,mpsr,Cvecs,ldens,rdens,sqrtldens,sqrtrdens,eta1,eta2

def regauge(mps,init=None,tol=1E-10,ncv=20,nmax=10000,which='LM',rcond=1E-16):
    D=mps[0].shape[0]
    d=mps[0].shape[2]
    if mps[0][0,0,0].dtype==np.float64:
        dtype='float64'
    elif mps[0][0,0,0].dtype==np.complex128:
        dtype='complex128'

    eta1,v1,numeig=UnitcellTMeigs(mps,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)

    #normalize the state:
    phi=np.real(eta1**(-1.0/(2.0*len(mps))))
    for n in range(len(mps)):
        mps[n]*=phi

    l=np.reshape(v1,(D,D))
    l/=np.trace(l)
    l=(l+herm(l))/2.0
    if dtype=='float64':
        l=np.real(l)
    ldens=UnitcellTransferOperator(direction=1,mps=mps,vector=np.reshape(l,D*D),returnfull=True)

    eta2,v2,numeig=UnitcellTMeigs(mps,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which=which)
    r=np.reshape(v2,(D,D))
    r/=np.trace(r)
    r=(r+herm(r))/2.0
    if dtype=='float64':
        r=np.real(r)

    rdens=UnitcellTransferOperator(direction=-1,mps=mps,vector=np.reshape(r,D*D),returnfull=True)
    sqrtldens=np.zeros(ldens.shape).astype(dtype)
    #remember that rdens is in reversed order, i.e. redens[0] lives at the right boundary
    sqrtrdens=np.zeros(rdens.shape).astype(dtype)
    l=np.reshape(ldens[0,:],(D,D))
    sqrtl=np.transpose(sqrtm(l))
    invsqrtl=np.linalg.pinv(sqrtl)
    r=np.reshape(rdens[rdens.shape[0]-1,:],(D,D))
    sqrtr=sqrtm(r)
    invsqrtr=np.linalg.pinv(sqrtr)
    U,lam,V=np.linalg.svd(sqrtl.dot(sqrtr))        
    #Z=np.sqrt(np.sum(lam**2))
    #lam/=Z
    #print np.linalg.norm(lam)
    LB=np.diag(lam).dot(V).dot(invsqrtr)
    LBinv=np.linalg.pinv(LB)
    RB=invsqrtl.dot(U)
    RBinv=np.linalg.pinv(RB)

    #mps[0]=np.transpose(np.tensordot(np.tensordot(LB,mps[0],([1],[0])),LBinv,([1],[0])),(0,2,1))
    #tensor,r=prepareTensor(mps[0],1)
    mps[0]=np.tensordot(LB,mps[0],([1],[0]))
    mps[-1]=np.transpose(np.tensordot(mps[-1],RB,([1],[0])),(0,2,1))
    #tensor,r=prepareTensorfixA0(mps[n],1,dtype)
    #mps[0]=np.copy(tensor)
    #mps[1]=np.tensordot(r,mps[1],([1],[0]))
    for n in range(len(mps)-1):
        tensor,r=prepareTensorFixPhase(mps[n],1,'bla',dtype)
        #tensor,r=prepareTensorfixA0(mps[n],1,dtype)
        #print np.trace(r.dot(herm(r)))
        #tensor,r=prepareTensorFixPhase(mps[n],1,'bla',dtype)
        #tensor,r=prepareTensor(mps[n],1)
        #tensor,r=mf.prepareTensor(mps[n],1)
        #Z=np.sqrt(np.trace(r.dot(herm(r))))
        #r/=Z
        mps[n]=np.copy(tensor)
        mps[n+1]=np.tensordot(r,mps[n+1],([1],[0]))

    tensor,r=prepareTensorFixPhase(mps[-1],1,'bla',dtype)
    #tensor,r=prepareTensorfixA0(mps[-1],1,dtype)
    mps[-1]=np.copy(tensor)
    #for n in range(len(mps)-1,0,-1):
    #    tensor,r=prepareTensorFixPhase(mps[n],-1,'v',dtype)
    #    mps[n]=np.copy(tensor)
    #    mps[n-1]=np.transpose(np.tensordot(mps[n-1],r,([1],[0])),(0,2,1))

    return mps,r


def prepareTensorFixPhase(mps,direction,fix,dtype='float64'):
    [D1,D2,d]=np.shape(mps)
    dtype=type(mps[0,0,0])

    if direction>0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])
        mat[:,:,0]=np.eye(D1)
        mat[:,:,1]=mps[:,:,1].dot(np.linalg.pinv(A0))
        tensor,r=prepareTensor(mat,1)
        matout=r.dot(A0)


        #mat=np.zeros((D1,D2,d),dtype=dtype)
        #A0=np.copy(mps[:,:,0])
        #mat[:,:,0]=np.eye(D1)
        #mat[:,:,1]=mps[:,:,1].dot(np.linalg.pinv(A0))
        #tensor,r=prepareTensor(mat,1)
        #matout=r.dot(A0)


        if fix=='u':
            vd,l,ud=np.linalg.svd(herm(matout))
            u=herm(ud)
            v=herm(vd)
            l=l/np.linalg.norm(l)

            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
            tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            matout=np.diag(l).dot(v)
        if fix=='v':
            vd,l,ud=np.linalg.svd(herm(matout))
            u=herm(ud)
            v=herm(vd)
            l=l/np.linalg.norm(l)

            warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)

            tensor=np.transpose(np.tensordot(tensor,u,([1],[0])),(0,2,1))
            matout=np.diag(l).dot(v)
        
        return tensor,matout
        
    if direction<0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])
        mat[:,:,0]=np.eye(D1)
        mat[:,:,1]=np.linalg.pinv(A0).dot(mps[:,:,1])
        
        tensor,r=prepareTensor(mat,-1)
        #Z=np.sqrt(np.trace(r))
        #r/=Z

        matout=A0.dot(r)
        if fix=='u':
            u,l,v=np.linalg.svd(matout)
            l=l/np.linalg.norm(l)

            warnings.warn('CMPS.__prepareTensor__(site,direction={0},fix={1}): can produce jumps in Q and R! '.format(direction,fix))
            phase=np.angle(np.diag(u))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(unit)
            v=herm(unit).dot(v)
            tensor=np.tensordot(v,tensor,([1],[0]))
            matout=u.dot(np.diag(l))

        if fix=='v':
            u,l,v=np.linalg.svd(matout)
            l=l/np.linalg.norm(l)

            phase=np.angle(np.diag(v))
            unit=np.diag(np.exp(-1j*phase))
            u=u.dot(herm(unit))
            v=unit.dot(v)
            tensor=np.tensordot(v,tensor,([1],[0]))
            matout=u.dot(np.diag(l))
        
        return tensor,matout


def prepareTensorfixA0(mps,direction,dtype='float64'):
    [D1,D2,d]=np.shape(mps)
    dtype=type(mps[0,0,0])

    if direction>0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])
        mat[:,:,0]=np.eye(D1)
        mat[:,:,1]=mps[:,:,1].dot(np.linalg.pinv(A0))
        tensor,r=prepareTensor(mat,1)
        matout=r.dot(A0)

        return tensor,matout
        
    if direction<0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])

        mat[:,:,0]=np.eye(D1)
        mat[:,:,1]=np.linalg.pinv(A0).dot(mps[:,:,1])
        
        tensor,r=prepareTensor(mat,-1)
        matout=A0.dot(r)

        return tensor,matout



def prepareTensor(tensor,direction):
    assert(direction!=0),'do NOT use direction=0!'
    dtype=type(tensor[0,0,0])
    [l1,l2,d]=tensor.shape
    if direction>0:
        temp=np.reshape(np.transpose(tensor,(2,0,1)),(d*l1,l2))
        q,r=np.linalg.qr(temp)
        phase=np.angle(np.diag(q))
        unit=np.diag(np.exp(-1j*phase))
        q=q.dot(unit)
        r=herm(unit).dot(r)
        [size1,size2]=q.shape
        out=np.transpose(np.reshape(q,(d,l1,size2)),(1,2,0))
        
    if direction<0:
        temp=np.conjugate(np.transpose(np.reshape(tensor,(l1,d*l2),order='F'),(1,0)))
        q,r_=np.linalg.qr(temp)
        phase=np.angle(np.diag(q))
        unit=np.diag(np.exp(-1j*phase))
        q=q.dot(unit)
        r_=herm(unit).dot(r_)

        [size1,size2]=q.shape
        out=np.conjugate(np.transpose(np.reshape(q,(l2,d,size2),order='F'),(2,0,1)))
        r=np.conjugate(np.transpose(r_,(1,0)))
    return out,r

def prepareTensorSVD(tensor,direction):
    [l1,l2,d]=tensor.shape
    if direction>0:
        temp=np.reshape(np.transpose(tensor,(2,0,1)),(d*l1,l2))
        #[u,s,v]=svd(temp,full_matrices=False)
        [u,s,v]=np.linalg.svd(temp,full_matrices=False)
        for n in range(np.shape(u)[1]):
            if u[n,n]<0.0:
                u[:,n]=(-1.0)*u[:,n]
                v[n,:]=(-1.0)*v[n,:]

        [size1,size2]=u.shape
        out=np.transpose(np.reshape(u,(d,l1,size2)),(1,2,0))
    if direction<0:
        temp=np.reshape(tensor,(l1,d*l2),order='F')
        [v,s,u]=svd(temp,full_matrices=False)
        for n in range(np.shape(u)[0]):
            if u[n,n]<0.0:
                u[n,:]=(-1.0)*u[n,:]
                v[:,n]=(-1.0)*v[:,n]

        [size1,size2]=u.shape

        out=np.reshape(u,(l1,size1,d),order='F')
    return out,s,v

def distributeR(mps,x,L,r):
    D=mps[0].shape[0]
    dx=utils.computedx(0,x,L)    
    if not all(np.abs(dx-dx[0])<1E-10):
        sys.exit('mpsfunctionsgradopt.py distributeR: x is not equally spaced between 0 and L, i.e. the resulting dx are not uniform')

    eta,U=np.linalg.eig(r)
    #print U.dot(np.diag(eta)).dot(herm(U))-r
    temp=eta**(1.0/(len(mps)+1))
    #H=(U.dot(np.diag(temp)).dot(herm(U))-np.eye(D))/dx[0]
    for n in range(len(mps)):
        C=U.dot(np.diag(temp**(n+1))).dot(herm(U))
        #print np.linalg.norm(C.dot(herm(C))-np.eye(D))
        Cinv=U.dot(np.diag(temp**(-1.0*n))).dot(herm(U))
        mps[n]=np.transpose(np.tensordot(np.tensordot(Cinv,mps[n],([1],[0])),C,([1],[0])),(0,2,1))
    return mps

    

def computeOrthogonalMPS(mps,init=None,tol=1E-10,ncv=20,nmax=10000,rcond=1E-16):
    Gamma,mpsl,mpsr,Cs,ldens,rdens,sqrtldens,sqrtrdens,etal,etar=regaugecentralC(mps,init=init,tol=tol,ncv=ncv,nmax=nmax,which='LM',rcond=rcond)


    return mpsl,mpsr,Cs,ldens,rdens,etal,etar




#computes the fully left renormalized Hamiltonian H_l from my notes; user can provide rightode, the dominant right reduced density matrices of cmpsl
#on the grid cmpsl.x inside the unit cell; if rightode="None", then it is calculated from scratch.
#returns the normalized H_l, unnormalized H_l and rightode
def computeHLLiebLiniger(mpsl,x,L,g,mass,mu,rinit=None,init=None,thresh=1E-8,imax=1000,nmax=1000,tol=1E-10,ncv=20):
    D=mpsl[0].shape[0]

    d=mpsl[0][2]
    if mpsl[0][0,0,0].dtype==np.float64:
        dtype='float64'
    elif mpsl[0][0,0,0].dtype==np.complex128:
        dtype='complex128'
    dx=utils.computedx(0,x,L)
    #be sure that all dxs are identical
    if not all(np.abs(dx-dx[0])<1E-10):
        sys.exit('disccmpsfunctions.pu computeHLLiebLiniger: x is not equally spaced between 0 and L, i.e. the resulting dx are not uniform')

    mpo=Hams.projectedLiebLinigermpo3(mu(x),g,mass,dx,False,dtype=dtype)

    #todo: add a check that cmpsl is left-isometric
    leftdens=[]
    #rightdens=[]    
    if rinit==None:
        etar,vr,numeig=UnitcellTMeigs(mpsl,direction=-1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which='LM')
        r=np.reshape(vr,(D,D))
        r/=np.trace(r)
        r=(r+herm(r))/2.0
        if dtype=='float64':
            r=np.real(r)
        rightdens=mf.computeDensity(r,mpsl,direction=-1,dtype=dtype)        
        #rightdens=UnitcellTransferOperator(direction=-1,mps=mpsl,returnfull=True,vector=np.reshape(r,D*D))

    for n in range(len(rightdens)):
        leftdens.append(np.eye(D))
        

    HL0=mf.computeUCsteadyStateHamiltonianGMRES(mpsl,mpopbc,init=init,ldens=leftdens,rdens=rightdens,direction=1,thresh=thresh,imax=imax,dtype=dtype)
    HL=[]
    HLrenorm=[]
    HL.append(np.copy(HL0))
    HLrenorm.append(np.copy(HL0))
    for n in range(len(mpsl)):
        HL.append(mf.addLayer(HL[-1],mpsl[n],mpopbc[n],mpsl[n],1))
        HLrenorm.append(mf.addLayer(HLrenorm[-1],mpsl[n],mpopbc[n],mpsl[n],1))
    for n in range(len(HL)):
        hlt=HL[n][:,:,0]
        HLrenorm[n][:,:,0]=hlt-np.tensordot(hlt,rightdens[n],([0,1],[0,1]))*np.eye(D)

    return HLrenorm,HL,rightdens



#computes the fully left renormalized Hamiltonian H_l from my notes; user can provide rightode, the dominant right reduced density matrices of cmpsl
#on the grid cmpsl.x inside the unit cell; if rightode="None", then it is calculated from scratch.
#returns the normalized H_l, unnormalized H_l and rightode
def computeHRLiebLiniger(mpsr,x,L,g,mass,mu,linit=None,init=None,thresh=1E-8,imax=1000,nmax=1000,tol=1E-10,ncv=20):
    D=mpsr[0].shape[0]

    d=mpsr[0][2]
    if mpsr[0][0,0,0].dtype==np.float64:
        dtype='float64'
    elif mpsr[0][0,0,0].dtype==np.complex128:
        dtype='complex128'
    dx=utils.computedx(0,x,L)
    #be sure that all dxs are identical
    if not all(np.abs(dx-dx[0])<1E-10):
        sys.exit('disccmpsfunctions.pu computeHLLiebLiniger: x is not equally spaced between 0 and L, i.e. the resulting dx are not uniform')

    mpo=Hams.projectedLiebLinigermpo3(mu(x),g,mass,dx,False,dtype=dtype)

    #todo: add a check that cmpsr is left-isometric
    rightdens=[]
    #rightdens=[]    
    if linit==None:
        etal,vl,numeig=UnitcellTMeigs(mpsr,direction=1,numeig=6,init=None,datatype=dtype,nmax=nmax,tolerance=tol,ncv=ncv,which='LM')
        l=np.reshape(vl,(D,D))
        l/=np.trace(l)
        l=(l+herm(l))/2.0
        if dtype=='float64':
            l=np.real(l)
        leftdens=mf.computeDensity(l,mpsr,direction=1,dtype=dtype)        
        #leftdens=UnitcellTransferOperator(direction=1,mps=mpsr,returnfull=True,vector=np.reshape(l,D*D))

    for n in range(len(leftdens)):
        rightdens.append(np.eye(D))
        
    
    HR0=mf.computeUCsteadyStateHamiltonianGMRES(mpsr,mpopbc,init=init,ldens=leftdens,rdens=rightdens,direction=-1,thresh=thresh,imax=imax,dtype=dtype)
    HR=[]
    HRrenorm=[]
    for n in range(len(mpsr)+1):
        HR.append(None)
        HRrenorm.append(None)

    HR[-1]=np.copy(HR0)
    HRrenorm[-1]=np.copy(HR0)
    for n in range(len(mpsr)-1,-1,-1):
        HR[n]=mf.addLayer(HR[n+1],mpsr[n],mpopbc[n],mpsr[n],-1)
        HRrenorm[n]=mf.addLayer(HRrenorm[n+1],mpsr[n],mpopbc[n],mpsr[n],-1)

    for n in range(len(HR)):
        hrt=HR[n][:,:,-1]
        HRrenorm[n][:,:,-1]=hrt-np.tensordot(leftdens[n],hrt,([0,1],[0,1]))*np.eye(D)

    return HRrenorm,HR,leftdens


#computes the mixed transer matrix vector product vector*E_A^B or E_A^B*vector
#A and B are mps tensors of dimension (chi1 x chi2 x d), from which the transfer matrix can computed if A=B; 
#B is always the upper matrix, A is always the lower one
#direction > 0 does a left-side product; direction < 0 does a right side product;
#vector is a chi1 x chi1 (direction > 0) or chi2 x chi2 (direction < 0) matrix, given in VECTOR format!
#returns a vector 
def TransferOperator(direction,A,vector):
    [D1,D2,d]= A.shape
    if direction>0:
        x=np.reshape(vector,(D1,D1))
        return np.reshape(np.tensordot(np.tensordot(A,x,([0],[0])),np.conj(A),([1,2],[2,0])),D1*D2)
    if direction<0:
        x=np.reshape(vector,(D2,D2))
        return np.reshape(np.tensordot(np.tensordot(A,x,([1],[0])),np.conj(A),([1,2],[2,1])),D1*D2)


#computes the left or right eigenvector of the transfer matrix using the GeneralizedMatrixVectorProduct(direction,A,B,vector) function to
#to do the matrix-vector multiplication
def TMeigs(tensor,direction,numeig,init=None,datatype='float64',nmax=6000,tolerance=1e-10,ncv=10,which='LR'):
    #define the matrix vector product mv(v) using functools.partial and GeneralizedMatrixVectorProduct(direction,A,B,vector):
    [chi1,chi2,d]=np.shape(tensor)

    #either of the following two TransferOps works (for Real numbers at least)

    #mv=fct.partial(TransferOperator,*[direction,tensor])
    mv=fct.partial(GeneralizedMatrixVectorProduct,*[direction,tensor,tensor])


    LOP=LinearOperator((chi1*chi1,chi2*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)

    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))

    while np.abs(np.imag(eta[m]))>1E-10:
        numeig=numeig+1
        print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
        print (eta)
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))

    return eta[m],np.reshape(vec[:,m],chi1*chi1),numeig

#takes a vector, returns a vector
def UnitcellTransferOperator(direction,mps,returnfull,vector):
    D=mps[0].shape[0]
    if returnfull==False:
        x=np.copy(vector)
        if direction>0:
            for n in range(len(mps)):
                x=TransferOperator(direction,mps[n],x)
            return x
        if direction<0:
            for n in range(len(mps)-1,-1,-1):
                x=TransferOperator(direction,mps[n],x)
            return x
    if returnfull==True:
        x=np.zeros((len(mps)+1,D**2)).astype(vector.dtype)
        x[0,:]=np.copy(vector)
        if direction>0:
            for n in range(len(mps)):
                x[n+1,:]=TransferOperator(direction,mps[n],x[n,:])
            return x
        if direction<0:
            for n in range(len(mps)-1,-1,-1):
                x[len(mps)-1-n+1,:]=TransferOperator(direction,mps[n],x[len(mps)-1-n,:])
            return x


#returns the unitcellTO eigenvector with 'LR'
def UnitcellTMeigs(mps,direction,numeig,init=None,datatype='float64',nmax=800,tolerance=1e-12,ncv=10,which='LM'):
    #define the matrix vector product mv(v) using functools.partial and GeneralizedMatrixVectorProduct(direction,A,B,vector):
    D=mps[0].shape[0]

    mv=fct.partial(UnitcellTransferOperator,*[direction,mps,False])
    LOP=LinearOperator((D**2,D**2),matvec=mv,rmatvec=None,matmat=None,dtype=datatype)
    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m]))>1E-4:
        numeig=numeig+1
        print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
        print (eta)
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
    return eta[m],np.reshape(vec[:,m],D*D),numeig



#def UpdateLiebLinigerLeft(mpsl,mpsr,Cs,HLrenorm,HRrenorm,g,mass,mu):
