"""
@author: Martin Ganahl
"""

from sys import stdout
import numpy as np
import os,copy
import time
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.mps as mpslib
import lib.Lanczos.LanczosEngine as LZ
import lib.utils.utilities as utils
import lib.ncon as ncon
from scipy.sparse.linalg import ArpackNoConvergence
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

"""
DMRG engine for obtaining ground state of mpos
mps: initial state
mpo: the hamiltonian in mpo format
filename: filename for checkpointing and what not
lb, rb: left and right boundary conditions;  if None, obc are assumed;
"""
class DMRGengine:
    #left/rightboundary are boundary mpo expressions; pass lb=rb=np.ones((1,1,1)) for obc simulation
    def __init__(self,mps,mpo,filename,lb=None,rb=None):
        if (np.all(lb)!=None) and (np.all(rb)!=None):
            assert(mps[0].shape[0]==lb.shape[0])
            assert(mps[-1].shape[1]==rb.shape[0])
            assert(mpo[0].shape[0]==lb.shape[2])
            assert(mpo[-1].shape[1]==rb.shape[2])
            self._lb=np.copy(lb)
            self._rb=np.copy(rb)
            
        else:
            assert(mpo[0].shape[0]==1)
            assert(mpo[-1].shape[1]==1)
            assert(mps[0].shape[0]==1)
            assert(mps[-1].shape[1]==1)            
            
            self._lb=np.ones((1,1,1))
            self._rb=np.ones((1,1,1))
        
        self._mps=mps
        self._mpo=mpo
        self._filename=filename
        self._N=self._mps._N
        self._mps.__position__(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,np.copy(self._lb))
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,np.copy(self._rb))


    def __simulate__(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        performs a single site DMRG optimization
        Nmax: maximum number of sweeps
        Econv: desired convergence of energy
        tol: tolerance parameter of the eigensolver
        ncv: number of krylov vectors in the eigensolver
        cp: checkppointing step
        verbose: verbosity flag
        numvecs: number of eigenstates returned  by solver
        solver: type of solver; current support: 'AR' (arnoldi) or 'LAN' (lanczos)
        Ndiag: lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan: maximum number of lanczos stesp
        landelta: lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies

        """
        
        assert((solver=='AR') or (solver=='LAN'))
        converged=False
        energy=100000.0
        it=1
        Es=np.zeros(self._mps._N)
        self._mps.__position__(0)
        while not converged:
            for n in range(0,self._mps._N-1):
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),tol,numvecs,ncv)#mps._mat set to 11 during call of __tensor__()
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

                Dnew=opt.shape[1]
                if verbose>0:
                    stdout.write("\rSS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,self._N,np.real(e),np.imag(e),Dnew))
                    stdout.flush()                    
                    #print ('at iteration {2} optimization at site {0} returned E={1}'.format(n,e,it))
                Es[n]=e
                tensor,r=mf.prepareTensor(opt,1)
                self._mps[n]=tensor
                self._mps._mat=r
                self._mps._position=n+1

                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)
                
            for n in range(self._mps._N-1,0,-1):
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

                Dnew=opt.shape[1]
                if verbose>0:
                    stdout.write("\rSS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,self._N,np.real(e),np.imag(e),Dnew))
                    stdout.flush()                                        
                    #
                    #print ('at iteration {2} optimization at site {0} returned E={1}'.format(n,e,it))
                Es[n]=e
                tensor,r=mf.prepareTensor(opt,-1)
                self._mps[n]=tensor
                self._mps._mat=r
                self._mps._position=n
                
                self._R[self._mps._N-1-n+1]=mf.addLayer(self._R[self._mps._N-1-n],self._mps[n],self._mpo[n],self._mps[n],-1)

                    
            if np.abs(e-energy)<Econv:
                converged=True
            energy=e
            if cp!=None and it>0 and it%cp==0:
                np.save(self._filename+'_dmrg_cp',self._mps._tensors)
            it=it+1
            if it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        return e
        #returns the center bond matrix and the gs energy


        
    def __simulateTwoSite__(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        performs a two site DMRG optimization
        Nmax: maximum number of sweeps
        Econv: desired convergence of energy
        tol: tolerance parameter of the eigensolver
        ncv: number of krylov vectors in the eigensolver
        cp: checkppointing step
        verbose: verbosity flag
        numvecs: number of eigenstates returned  by solver
        solver: type of solver; current support: 'AR' (arnoldi) or 'LAN' (lanczos)
        Ndiag: lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan: maximum number of lanczos stesp
        landelta: lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies

        """

        assert((solver=='AR') or (solver=='LAN'))
        converged=False
        energy=100000.0
        it=1
        Es=np.zeros(self._mps._N)
        self._mps.__position__(0)        
        while not converged:
            for n in range(0,self._mps._N-2):
                self._mps.__position__(n+1)
                temp1=ncon.ncon([self._mps.__tensor__(n),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

                Es[n]=e
                temp3=np.reshape(np.transpose(np.reshape(opt,(Dl,Dr,dl,dr)),(0,2,3,1)),(Dl*dl,dr*Dr))
                U,S,V=np.linalg.svd(temp3,full_matrices=False)
                S=S[S>truncation]
                Dnew=len(S)
                Dnew=min(len(S),self._mps._D)
                S=S[0:Dnew]
                S/=np.linalg.norm(S)
                U=U[:,0:Dnew]
                V=V[0:Dnew,:]
                if verbose>0:
                    stdout.write("\rTS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,self._N,np.real(e),np.imag(e),Dnew))
                    stdout.flush()                    

                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)

            for n in range(self._mps._N-2,-1,-1):
                self._mps.__position__(n+1)
                temp1=ncon.ncon([self._mps.__tensor__(n),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

                Es[n]=e
                temp3=np.reshape(np.transpose(np.reshape(opt,(Dl,Dr,dl,dr)),(0,2,3,1)),(Dl*dl,dr*Dr))
                U,S,V=np.linalg.svd(temp3,full_matrices=False)
                S=S[S>truncation]
                Dnew=len(S)
                Dnew=min(len(S),self._mps._D)
                S=S[0:Dnew]
                S/=np.linalg.norm(S)
                U=U[:,0:Dnew]
                V=V[0:Dnew,:]
                if verbose>0:
                    stdout.write("\rTS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,self._N,np.real(e),np.imag(e),Dnew))
                    stdout.flush()                    

                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                self._R[self._mps._N-1-n]=mf.addLayer(self._R[self._mps._N-1-n-1],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)

            self._mps.__position__(0)
            self._R[self._mps._N-1-n]=mf.addLayer(self._R[self._mps._N-1-n-1],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)
            if np.abs(e-energy)<Econv:
                converged=True
            energy=e
            if cp!=None and it>0 and it%cp==0:
                np.save(self._filename+'_dmrg_cp',self._mps._tensors)
            it=it+1
            if it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        return e
        #returns the center bond matrix and the gs energy


class IDMRGengine(DMRGengine):

    """
    performs an IDMRG optimization of the ground-state of a Hamiltonian as given by mpo
    mps: MPS object with infinite boundary conditions
    mpo: MPO object with infinite boundary conditions
    filename: the name of the simulation; will be appended to the output (if there is any)
    """
    def __init__(self,mps,mpo,filename):
        lb,rb,lbound,rbound=mf.getBoundaryHams(mps,mpo)                            
        super().__init__(mps,mpo,filename,lb,rb)
    #shifts the unit-cell by N/2 by updating self._L, self._R, self._lb, self._rb, and cutting and patching self._mps and self._mpo
    def __update__(self):
        self._mps.__position__(self._mps._N)
        #update the left boundary
        for site in range(int(self._mps._N/2)):
            self._lb=mf.addLayer(self._lb,self._mps._tensors[site],self._mpo[site],self._mps._tensors[site],1)
            
        lamR=np.copy(self._mps._mat)
        mps=[]
        D=0
        #cut and patch the left half of the mps
        for n in range(int(self._mps._N/2),self._mps._N):
            if self._mps._tensors[n].shape[0]>D:
                D=self._mps._tensors[n].shape[0]
            mps.append(np.copy(self._mps._tensors[n]))

        self._mps.__position__(int(self._mps._N/2))
        lamC=np.copy(self._mps._mat)
        connector=np.linalg.pinv(lamC)

        #update the right boundary
        for site in range(self._mps._N-1,int(self._mps._N/2)-1,-1):
            self._rb=mf.addLayer(self._rb,self._mps._tensors[site],self._mpo[site],self._mps._tensors[site],-1)

        self._R[0]=np.copy(self._rb)
        self._L[0]=np.copy(self._lb)
        self._mps.__position__(0)
        lamL=np.copy(self._mps._mat)
        #cut and patch the right half of the mps
        for n in range(int(self._mps._N/2)):
            if self._mps._tensors[n].shape[0]>D:
                D=self._mps._tensors[n].shape[0]
            mps.append(np.copy(self._mps._tensors[n]))
        for n in range(self._mps._N):
            self._mps._tensors[n]=mps[n]
            
        self._mps._position=int(self._mps._N/2)
        self._mps._mat=lamR.dot(self._mps._connector).dot(lamL)
        self._mps._connector=connector
        self._mps.__position__(0)
        
        #cut and patch the mpo
        mf.patchmpo(self._mpo,int(self._mps._N/2))        
        return D

    def __simulate__(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):            
        """
        run a single site IDMRG simulation
        Nmax: number of outer iterations
        NUC: number of optimization sweeps when optimizing a single unitcell
        Econv: desired convergence of energy per unitcell
        tol: arnoldi tolerance
        ncv: number of krylov vectors in arnoldi or lanczos
        cp: chekpoint step
        verbose: verbosity flag
        numvecs: the number of eigenvectors to be calculated; should be 1
        solver: type of eigensolver: 'AR' or 'LAN' for arnoldi or lanczos
        Ndiag: lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan: maximum number of lanczos stesp
        landelta: lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies
        """
        print ('# simulation parameters:')
        print ('# of idmrg iterations: {0}'.format(Nmax))
        print ('# of sweeps per unit cell: {0}'.format(NUC))
        print ('# Econv: {0}'.format(Econv))
        print ('# Arnoldi tolerance: {0}'.format(tol))
        print ('# Number of Lanzcos vector in Arnoldi: {0}'.format(ncv))
        it=0
        converged=False
        #eold=super().__simulate__(NUC,Econv,tol,ncv,verbose=verbose-1)
        #self.__update__()
        eold=0.0
        while not converged:
            e=super().__simulate__(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,landelta=landelta,landeltaEta=landeltaEta)
            D=self.__update__()
            if verbose>0:
                stdout.write("\rSS-IDMRG: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real((e-eold)/(self._mps._N)),np.imag((e-eold)/(self._mps._N)),D))
                stdout.flush()  
                if verbose>1:
                    print('')
                #print ('at iteration {0} optimization returned E/N={1}'.format(it,(e-eold)/(dmrg._mps._N)))
            if cp!=None and it>0 and it%cp==0:                
                np.save(self._filename+'_dmrg_cp',self._mps._tensors)
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        it=it+1
    def __simulateTwoSite__(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        run a twos-site IDMRG simulation
        Nmax: number of outer iterations
        NUC: number of optimization sweeps when optimizing a single unitcell
        Econv: desired convergence of energy per unitcell
        tol: arnoldi tolerance
        ncv: number of krylov vectors in arnoldi or lanczos
        cp: chekpoint step
        verbose: verbosity flag
        numvecs: the number of eigenvectors to be calculated; should be 1
        solver: type of eigensolver: 'AR' or 'LAN' for arnoldi or lanczos
        Ndiag: lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan: maximum number of lanczos stesp
        landelta: lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies
        """
        
        print ('# simulation parameters:')
        print ('# of idmrg iterations: {0}'.format(Nmax))
        print ('# of sweeps per unit cell: {0}'.format(NUC))
        print ('# Econv: {0}'.format(Econv))
        print ('# Arnoldi tolerance: {0}'.format(tol))
        print ('# Number of Lanzcos vector in Arnoldi: {0}'.format(ncv))
        it=0
        converged=False
        #eold=super().__simulate__(NUC,Econv,tol,ncv,verbose=verbose-1)
        #self.__update__()
        eold=0.0
        while not converged:
            e=super().__simulateTwoSite__(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,numvecs=numvecs,truncation=truncation,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,\
                                          landelta=landelta,landeltaEta=landeltaEta)
            D=self.__update__()
            if verbose>0:
                stdout.write("\rTS-IDMRG: it=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real((e-eold)/(self._mps._N)),np.imag((e-eold)/(self._mps._N)),D))
                stdout.flush()  
                if verbose>1:
                    print('')
            if cp!=None and it>0 and it%cp==0:                
                np.save(self._filename+'_dmrg_cp',self._mps._tensors)
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        it=it+1


class HomogeneousIMPSengine:
    """
    MPS optimization methods for homogeneous systems
    uses gradient optimization to find the ground state of a Homogeneous system
    mps (np.ndarray): an initial mps tensor 
    mpo (np.ndarray): the mpo tensor
    filename (str): filename of the simulation
    alpha (float): initial steps size
    alphas (list of float): alphas[i] is the stepsizes to be use once gradient norm is smaller than normgrads[i] (see next)
    normgrads (list of float): alphas[i] is the stepsizes to be use once gradient norm is smaller than normgrads[i] (see next)
    dtype: type of the mps (float or complex)
    factor (float): factor by which internal stepsize is reduced in case divergence is detected
    normtol (float): absolute value by which gradient may increase without raising a "divergence" flag
    epsilon (float): desired convergence of the gradient
    tol (float): eigensolver tolerance used in regauging
    lgmrestol (float): eigensolver tolerance used for calculating the infinite environements
    ncv (int): number of krylov vectors used in sparse eigensolvers
    numeig (int): number of eigenvectors to be calculated in the sparse eigensolver
    Nmaxlgmres (int): max steps of the lgmres routine used to calculate the infinite environments
    """

    def __init__(self,Nmax,mps,mpo,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-4,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40):
        self._Nmax=Nmax
        self._mps=np.copy(mps)
        self._D=np.shape(mps)[0]
        self._d=np.shape(mps)[2]
        self._filename=filename
        self._dtype=dtype
        self._tol=tol
        self._lgmrestol=lgmrestol
        self._numeig=numeig
        self._ncv=ncv
        self._gamma=np.zeros(np.shape(self._mps),dtype=self._dtype)
        self._lam=np.ones(self._D)
        self._kleft=np.random.rand(self._D,self._D)
        self._kright=np.random.rand(self._D,self._D)
        self._alphas=alphas
        self._alpha=alpha
        self._alpha_=alpha

        self._normgrads=normgrads
        self._normtol=normtol
        self._factor=factor
        self._itreset=itreset
        self._epsilon=epsilon
        self._Nmaxlgmres=Nmaxlgmres

        self._it=1
        self._itPerDepth=0
        self._depth=0
        self._normgradold=10.0
        self._warmup=True
        self._reset=True
        #self._reject=False:
        [B1,B2,d1,d2]=np.shape(mpo)

        mpol=np.zeros((1,B2,d1,d2),dtype=self._dtype)
        mpor=np.zeros((B1,1,d1,d2),dtype=self._dtype)

        mpol[0,:,:,:]=mpo[-1,:,:,:]
        mpol[0,0,:,:]/=2.0
        mpor[:,0,:,:]=mpo[:,0,:,:]
        mpor[-1,0,:,:]/=2.0

        self._mpo=[]
        self._mpo.append(np.copy(mpol))
        self._mpo.append(np.copy(mpo))
        self._mpo.append(np.copy(mpor))



    def __doGradStep__(self):
        converged=False
        self._gamma,self._lam,trunc=mf.regauge(self._mps,gauge='symmetric',initial=np.reshape(np.diag(self._lam**2),self._D*self._D),\
                                               nmaxit=10000,tol=self._tol,ncv=self._ncv,numeig=self._numeig,trunc=1E-12,Dmax=self._D)
        
        if len(self._lam)!=self._D:
            Dchanged=True
            self._D=len(self._lam)

        self._A=np.tensordot(np.diag(self._lam),self._gamma,([1],[0]))
        self._B=np.transpose(np.tensordot(self._gamma,np.diag(self._lam),([1],[0])),(0,2,1))
        
        self._tensor=np.tensordot(np.diag(self._lam),self._B,([1],[0]))
        
        leftn=np.linalg.norm(np.tensordot(self._A,np.conj(self._A),([0,2],[0,2]))-np.eye(self._D))
        rightn=np.linalg.norm(np.tensordot(self._B,np.conj(self._B),([1,2],[1,2]))-np.eye(self._D))

        self._lb=mf.initializeLayer(self._A,np.eye(self._D),self._A,self._mpo[0],1) 
        ihl=mf.addLayer(self._lb,self._A,self._mpo[2],self._A,1)[:,:,0]

        Elocleft=np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))



        self._rb=mf.initializeLayer(self._B,np.eye(self._D),self._B,self._mpo[2],-1)


        ihr=mf.addLayer(self._rb,self._B,self._mpo[0],self._B,-1)[:,:,-1]
        Elocright=np.tensordot(ihr,np.diag(self._lam**2),([0,1],[0,1]))

        ihlprojected=(ihl-np.tensordot(ihl,np.diag(self._lam**2),([0,1],[0,1]))*np.eye(self._D))
        ihrprojected=(ihr-np.tensordot(np.diag(self._lam**2),ihr,([0,1],[0,1]))*np.eye(self._D))
        
        if self._kleft.shape[0]==len(self._lam):
            self._kleft=mf.RENORMBLOCKHAMGMRES(self._A,self._A,np.diag(self._lam**2),np.eye(self._D),ihlprojected,x0=np.reshape(self._kleft,self._D*self._D),tolerance=self._lgmrestol,\
                                               maxiteration=self._Nmaxlgmres,direction=1)
            self._kright=mf.RENORMBLOCKHAMGMRES(self._B,self._B,np.eye(self._D),np.diag(self._lam**2),ihrprojected,x0=np.reshape(self._kright,self._D*self._D),tolerance=self._lgmrestol,\
                                                maxiteration=self._Nmaxlgmres,direction=-1)
        else:
            self._kleft=mf.RENORMBLOCKHAMGMRES(self._A,self._A,np.diag(self._lam**2),np.eye(self._D),ihlprojected,x0=None,tolerance=self._lgmrestol,\
                                               maxiteration=self._Nmaxlgmres,direction=1)
            self._kright=mf.RENORMBLOCKHAMGMRES(self._B,self._B,np.eye(self._D),np.diag(self._lam**2),ihrprojected,x0=None,tolerance=self._lgmrestol,\
                                                maxiteration=self._Nmaxlgmres,direction=-1)
            Dchanged=False
        

        self._lb[:,:,0]+=np.copy(self._kleft)
        self._rb[:,:,-1]+=np.copy(self._kright)
        
        self._grad=np.reshape(mf.HAproductSingleSite(self._lb,self._mpo[1],self._rb,self._tensor),(self._D,self._D,self._d))-2*Elocleft*self._tensor
        self._normgrad=np.real(np.tensordot(self._grad,np.conj(self._grad),([0,1,2],[0,1,2])))
        self._alpha_,self._depth,self._itPerDepth,self._reset,self._reject,self._warmup=utils.determineNewStepsize(alpha_=self._alpha_,alpha=self._alpha,alphas=self._alphas,nxdots=self._normgrads,\
                                                                                                                   normxopt=self._normgrad,\
                                                                                                                   normxoptold=self._normgradold,normtol=self._normtol,warmup=self._warmup,it=self._it,\
                                                                                                                   rescaledepth=self._depth,factor=self._factor,itPerDepth=self._itPerDepth,\
                                                                                                                   itreset=self._itreset,reset=self._reset)
        if self._reject==True:
            self._grad=np.copy(self._gradbackup)
            self._tensor=np.copy(self._tensorbackup)
            self._lam=np.copy(self._lamold)
            self._D=len(self._lam)
            opt=self._tensor-self._alpha_*self._grad
            self._mps=np.transpose(np.tensordot(opt,np.diag(1.0/self._lam),([1],[0])),(0,2,1))
            print ('  norm increase from ||x||={1} --> {0} at normtolerance of {2}!'.format(self._normgrad,self._normgradold,self._normtol))
        
        if self._reject==False:
            #betanew,itstde,itbeta,dostde,printnlcgmessage,printstdemessage=utils.determineNonLinearCGBeta(self._nlcgupperthresh,self._nlcglowerthresh,self._nlcgnormtol,self._nlcgreset,self._normxopt,\
            #                                                                                       self._normxoptold,self._it,itstde,self._stdereset,dostde,itbeta,printnlcgmessage,printstdemessage)
            self._gradbackup=np.copy(self._grad)
            self._tensorbackup=np.copy(self._tensor)
            opt=self._tensor-self._alpha_*self._grad
            self._mps=np.transpose(np.tensordot(opt,np.diag(1.0/self._lam),([1],[0])),(0,2,1))
        
            self._normgradold=self._normgrad
            self._lamold=np.copy(self._lam)
        A_,s,v=mf.prepareTruncate(self._mps,direction=1,thresh=1E-14)
        s=s/np.linalg.norm(s)
        self._mps=np.transpose(np.tensordot(A_,np.diag(s).dot(v),([1],[0])),(0,2,1))
        if self._normgrad<self._epsilon:
            converged=True


        return Elocleft,leftn,rightn,converged

    #important note: at the moment, the user has to provide an mpo which covers TWO unitcells!
    def __simulate__(self,checkpoint=100):
        converged=False
        Dchanged=False
        while converged==False:
            Elocleft,leftn,rightn,converged=self.__doGradStep__()
            if self._it>=self._Nmax:
                break
            if self._it%checkpoint==0:
                np.save('CPTensor'+self._filename,self._mps)
            self._it+=1
            stdout.write("\rit %i: local E=%.16f, lnorm=%.6f, rnorm=%.6f, grad=%.16f, alpha=%.4f" %(self._it,np.real(Elocleft),leftn,rightn,self._normgrad,self._alpha_))
            stdout.flush()
        print
        if self._it>=self._Nmax and (converged==False):
            print ('simulation did not converge to {0} in {1} steps'.format(self._epsilon,self._Nmax))
        print
 

"""
a derived class for discretized Boson simulations; teh only difference to HomogeneousIMPS is the calculation of certain observables
"""
class HomogeneousDiscretizedBosonEngine(HomogeneousIMPSengine):
    def __init__(self,Nmax,mps,mpo,dx,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-4,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40):        
        super().__init__(Nmax,mps,mpo,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-4,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40)
        
    #important note: at the moment, the user has to provide an mpo which covers TWO unitcells!
    def __simulate__(self,dx,mu,checkpoint=100):
        converged=False
        Dchanged=False
        while converged==False:
            Elocleft,leftn,rightn,converged=self.__doGradStep__()
            if self._it>=self._Nmax:
                break
            self._it+=1
            dens=0
            for ind in range(1,self._tensor.shape[2]):
                dens+=np.trace(herm(self._tensor[:,:,ind]).dot(self._tensor[:,:,ind]))/dx
            kin=Elocleft/dx-mu*dens
            stdout.write("\rit %i at dx=%.5f: local E=%.8f, <h>=%.8f, <n>=%.8f, <h>/<n>**3=%.8f, lnorm=%.6f, rnorm=%.6f, grad=%.10f, alpha=%.4f" %(self._it,dx,np.real(Elocleft/dx),np.real(kin),\
                                                                                                                                                   np.real(dens),\
                                                                                                                                                   kin/dens**3,leftn,rightn,self._normgrad,self._alpha_))
            stdout.flush()

        print
        if self._it>=self._Nmax and (converged==False):
            print ('simulation did not converge to {0} in {1} steps'.format(self._epsilon,self._Nmax))
        print
        

"""
this is an engine for real or imaginary time evolution using TDVP; its not parallel, i.e. its a bit slow;
"mps": an initial mp
"gatecontainer": an object or method; gatecontainer(n,n+1,tau) has to return the gate to be applied at sites n and n+1
                 tau is a negative real (for imaginary time) or complex number with negative imaginary part (for real time)
"filename" is hte file under which cp results will be stored (not yet implemented)
"""

class TEBDEngine:
    def __init__(self,mps,gatecontainer,filename):
        self._mps=copy.deepcopy(mps)
        self._gates=copy.deepcopy(gatecontainer)
        self._filename=filename
        self._N=mps._N

    def applyEven(self,tau,Dmax,tr_thresh):
        for n in range(0,self._mps._N-1,2):
            tw_,D=self._mps.__applyTwoSiteGate__(gate=self._gates.twoSiteGate(n,n+1,tau),site=n,Dmax=Dmax,thresh=tr_thresh)
            self._maxD=max(self._maxD,D)
            self._tw+=tw_

    def applyOdd(self,tau,Dmax,tr_thresh):            
        if self._mps._N%2==0:
            lstart=self._mps._N-3
        elif self._mps._N%2==1:
            lstart=self._mps._N-2
        for n in range(lstart,-1,-2):
            tw_,D=self._mps.__applyTwoSiteGate__(gate=self._gates.twoSiteGate(n,n+1,tau),site=n,Dmax=Dmax,thresh=tr_thresh)
            self._maxD=max(self._maxD,D)                
            self._tw+=tw_
        
    def __doTEBD__(self,dt,numsteps,Dmax,tr_thresh,verbose=1,cnterset=0,tw=0,cp=None):
        """
        uses a second order trotter decomposition to evolve the state using TEBD
        dt: step size
        numsteps: total number of evolution steps
        Dmax: maximum bond dimension to be kept
        tr_thresh: truncation threshold 
        verbose: verbosity flag; put to 0 for no output
        cnterset: sets the internal iteration counter to cnterset; the internal iteration counter
                  is printed out if verbosity > 0, to have some idea of the simulation progress; 
                  cnterset is useful when when chaining multiple doTEBD calls, for example between measurements;
        """

        self._tw=tw
        it=cnterset
        self._maxD=1
        #even half-step:        
        self.applyEven(dt/2.0,Dmax,tr_thresh)
        for step in range(numsteps):
            #odd step updates:
            self.applyOdd(dt,Dmax,tr_thresh)
            if verbose==1:
                stdout.write("\rTEBD engine: t=%4.4f truncated weight=%.16f at D=%i"%(np.abs(np.imag(it*dt)),self._tw,self._maxD))
                stdout.flush()

            #if this is a cp step, save between two half-steps
            if (cp!=None) and (it>0) and (it%cp==0):
                #if the cp step does not coincide with the last step, do a half step, save, and do another half step
                if step<(numsteps-1):                
                    self.applyEven(dt/2.0,Dmax,tr_thresh)                
                    np.save(self._filename+'_tdvp_cp',self._mps._tensors)
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
                #if the cp step coincides with the last step, only do a half step and save the state
                else:
                    self.applyEven(dt/2.0,Dmax,tr_thresh)                
                    np.save(self._filename+'_tdvp_cp',self._mps._tensors)
                    
            #if step is not a cp step:
            else:
                #do a regular full step, unless step is the last step
                if step<(numsteps-1):
                    self.applyEven(dt,Dmax,tr_thresh)
                #if step is the last step, do a half step
                else:
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
            it=it+1
            
        return self._tw,it


class TDVPEngine:
    def __init__(self,mps,mpo,filename,lb=None,rb=None):
        if (np.all(lb)!=None) and (np.all(rb)!=None):
            assert(mps[0].shape[0]==lb.shape[0])
            assert(mps[-1].shape[1]==rb.shape[0])
            assert(mpo[0].shape[0]==lb.shape[2])
            assert(mpo[-1].shape[1]==rb.shape[2])
            self._lb=np.copy(lb)
            self._rb=np.copy(rb)
            
        else:
            assert(mpo[0].shape[0]==1)
            assert(mpo[-1].shape[1]==1)
            assert(mps[0].shape[0]==1)
            assert(mps[-1].shape[1]==1)            
            
            self._lb=np.ones((1,1,1))
            self._rb=np.ones((1,1,1))
        
        self._mps=copy.deepcopy(mps)
        self._mpo=copy.deepcopy(mpo)                
        self._filename=filename
        self._N=mps._N
        self._mps.__position__(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)


        
    def __doTDVP__(self,dt,numsteps,krylov_dim=20,cnterset=0,cp=None,verbose=1,use_split_step=False):
        """
        does a TDVP real or imaginary time evolution
        dt: step size
        numsteps: number of steps to be performed
        krylov_dim: if use_split_step=False, krylov_dim is the dimension of the krylov space used to perform evolution with lanczos
                    if use_split_step=True, method uses Ash Milsted's implementation of gexpmv (see evoMPS)
        cnterset: sets the iteration counter of the simulation to cnterset; effects terminal output; useful for chaining simulations
        cp: checkpointing (currently not implemented)
        verbose: verbosity flag
        use_split_step: if False, use Lanczos time evolution WITHOUT step-size optimization
                        if True, use Lanczos time evolution WITH step-size optimization using gexpmv from Ash Milsted's evoMPS package
        """
        converged=False
        it=cnterset
        self._mps.__position__(0)
        for step in range(numsteps):
            for n in range(self._mps._N):
                if n==self._mps._N-1:
                    dt_=dt
                else:
                    dt_=dt/2.0
                self._mps.__position__(n+1)
                #evolve tensor forward
                if not use_split_step:
                    evTen=mf. evolveTensorLan(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_,krylov_dimension=krylov_dim) #clear=True resets self._mat to identity
                else:
                    evTen=mf.evolveTensorSexpmv(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_)                
                tensor,mat=mf.prepareTensor(evTen,1)
                self._mps[n]=tensor
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)

                #evolve matrix backward                    
                if n<(self._mps._N-1):
                    if not use_split_step:
                        evMat=mf. evolveMatrixLan(self._L[n+1],self._R[self._mps._N-1-n],mat,-dt_,krylov_dimension=krylov_dim)
                    else:
                        evMat=mf.evolveMatrixSexpmv(self._L[n+1],self._R[self._mps._N-1-n],mat,-dt_)                                            
                    evMat/=np.linalg.norm(evMat)
                    self._mps._mat=evMat
                else:
                    self._mps._mat=mat
                    
            for n in range(self._mps._N-2,-1,-1):
                dt_=dt/2.0
                #evolve matrix backward; note that in the previous loop the last matrix has not been evolved yet; we'll rectify this now
                self._mps.__position__(n+1)
                self._R[self._mps._N-n-1]=mf.addLayer(self._R[self._mps._N-n-2],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)
                
                if not use_split_step:
                    evMat=mf. evolveMatrixLan(self._L[n+1],self._R[self._mps._N-1-n],self._mps._mat,-dt_,krylov_dimension=krylov_dim)
                else:
                    evMat=mf.evolveMatrixSexpmv(self._L[n+1],self._R[self._mps._N-1-n],self._mps._mat,-dt_)                                    
                evMat/=np.linalg.norm(evMat)#normalize wavefunction
                self._mps._mat=evMat        #set evolved matrix as new center-matrix
            
            
                #evolve tensor forward: the back-evolved center matrix is absorbed into the left-side tensor, and the product is evolved forward in time
                if not use_split_step:                
                    evTen=mf. evolveTensorLan(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_,krylov_dimension=krylov_dim)
                else:
                    evTen=mf.evolveTensorSexpmv(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),dt_)
                    
                #split of a center matrix C ("mat" in my notation)
                tensor,mat=mf.prepareTensor(evTen,-1) #mat is already normalized (happens in prepareTensor)
                self._mps[n]=tensor

                self._mps._mat=mat
                self._mps._position=n

            if verbose==1:
                stdout.write("\rTDVP engine: t=%4.4f"%(np.abs(np.imag(it*dt))))
                stdout.flush()

            if (cp!=None) and (it>0) and (it%cp==0):
                np.save(self._filename+'_tdvp_cp',self._mps._tensors)
            it=it+1
        self._mps.__position__(0)            
        return it
        #returns the center bond matrix and the gs energy



# ============================================================================     everything below this line is still in development =================================================
"""

calculates excitation spectrum for a lattice MPS
basemps has to be left orthogonal


"""
class ExcitationEngine:
    def __init__(self,basemps,basempstilde,mpo):
        NotImplemented
        self._mps=basemps
        assert(np.linalg.norm(np.tensordot(basemps,np.conj(basemps),([0,2],[0,2]))-np.eye(basemps.shape[1]))<1E-10)
        self._mpstilde=basempstilde
        self._mpo=mpo
        self._dtype=basemps.dtype
        
    def __simulate__(self,k,numeig,regaugetol,ncv,nmax,pinv=1E-14):
        return NotImplemented
        stdout.write("\r computing excitations at momentum k=%.4f" %(k))
        stdout.flush()

        self._k=k
        D=np.shape(self._mps)[0]
        d=np.shape(self._mps)[2]
        
        l=np.zeros((D,D),dtype=self._dtype)
        L=np.zeros((D,D,1),dtype=self._dtype)
        LAA=np.zeros((D,D,1),dtype=self._dtype)
        LAAAA=np.zeros((D,D,1),dtype=self._dtype)
        LAAAAEAAinv=np.zeros((D,D,1),dtype=self._dtype)

        r=np.zeros((D,D),dtype=self._dtype)
        R=np.zeros((D,D,1),dtype=self._dtype)
        RAA=np.zeros((D,D,1),dtype=self._dtype)
        RAAAA=np.zeros((D,D,1),dtype=self._dtype)
        REAAinvAAAA=np.zeros((D,D,1),dtype=self._dtype)

        #[etal,vl,numeig]=mf.TMeigs(self._mps,direction=1,numeig=numeig,init=None,nmax=nmax,tolerance=regaugetol,ncv=ncv,which='LR' )       
        #l=mf.fixPhase(np.reshape(vl,(D,D)))
        #l=l/np.trace(l)*D
        #hermitization of l
        l=np.eye(D)
        sqrtl=np.eye(D)#sqrtm(l)
        invsqrtl=np.eye(D)#np.linalg.pinv(sqrtl,rcond=1E-8)
        invl=np.eye(D)#np.linalg.pinv(l,rcond=1E-8)
        
        L[:,:,0]=np.copy(l)

        [etar,vr,numeig]=mf.TMeigs(self._mpstilde,direction=-1,numeig=numeig,init=None,nmax=nmax,tolerance=regaugetol,ncv=ncv,which='LR' )
        r=(mf.fixPhase(np.reshape(vr,(D,D))))
        #r=np.reshape(vr,(D,D))
        #hermitization of r
        r=((r+herm(r))/2.0)
        Z=np.real(np.trace(l.dot(r)))
        r=r/Z
        #print()
        #print (np.sqrt(np.abs(np.diag(r)))/np.linalg.norm(np.sqrt(np.abs(np.diag(r)))))
        #print (np.sqrt((np.diag(r))))
        #input()

        R[:,:,0]=np.copy(r)
        print ('norm of the state:',np.trace(R[:,:,0].dot(L[:,:,0])))

        #construct all the necessary left and right expressions:
        
        bla=np.tensordot(sqrtl,self._mps,([1],[0]))
        temp=np.reshape(np.transpose(bla,(2,0,1)),(d*D,D))
        random=np.random.rand(d*D,(d-1)*D)
        temp2=np.append(temp,random,1)
        [q,b]=np.linalg.qr(temp2)
        [size1,size2]=q.shape
        VL=np.transpose(np.reshape(q[:,D:d*D],(d,D,(d-1)*D)),(1,2,0))
        #print np.tensordot(VL,np.conj(self._mps),([0,2],[0,2]))
        #input()
        di,u=np.linalg.eigh(r)
        #print u.dot(np.diag(di)).dot(herm(u))-r
        #input()
        di[np.nonzero(di<1E-15)]=0.0
        invd=np.zeros(len(di)).astype(self._dtype)
        invsqrtd=np.zeros(len(di)).astype(self._dtype)
        invd[np.nonzero(di>pinv)]=1.0/di[np.nonzero(di>pinv)]
        invsqrtd[np.nonzero(di>pinv)]=1.0/np.sqrt(di[np.nonzero(di>pinv)])
        sqrtd=np.sqrt(di)
        sqrtr= u.dot(np.diag(sqrtd)).dot(herm(u))
        invr= u.dot(np.diag(invd)).dot(herm(u))
        invsqrtr= u.dot(np.diag(invsqrtd)).dot(herm(u))
        
        #sqrtr=sqrtm(r)
        #invsqrtr=np.linalg.pinv(sqrtr,rcond=1E-8)
        #invsqrtr=(invsqrtr+herm(invsqrtr))/2.0
        #invr=np.linalg.pinv(r,rcond=1E-14)

        RAA=mf.addLayer(R,self._mpstilde,self._mpo[1],self._mpstilde,-1)        
        RAAAA=mf.addLayer(RAA,self._mpstilde,self._mpo[0],self._mpstilde,-1)
        GSenergy=np.trace(RAAAA[:,:,0])
        print ('GS energy from RAAAA',GSenergy)
        LAA=mf.addLayer(L,self._mps,self._mpo[0],self._mps,1)        
        LAAAA=mf.addLayer(LAA,self._mps,self._mpo[1],self._mps,1)

        
        ih=np.reshape(LAAAA[:,:,0]-np.trace(np.dot(LAAAA[:,:,0],r))*l,(D*D))

        bla=mf.TDVPGMRES(self._mps,self._mps,r,l,ih,direction=1,momentum=0.0,tolerance=1e-12,maxiteration=2000,x0=None)
        #print np.tensordot(bla,r,([0,1],[0,1]))
        LAAAA_OneMinusEAAinv=np.reshape(bla,(D,D,1))
        
        ih=np.reshape(RAAAA[:,:,0]-np.trace(np.dot(l,RAAAA[:,:,0]))*r,(D*D))
        bla=mf.TDVPGMRES(self._mpstilde,self._mpstilde,r,l,ih,direction=-1,momentum=0.0,tolerance=1e-12,maxiteration=2000,x0=None)
        OneMinusEAAinv_RAAAA=np.reshape(bla,(D,D,1))
        
        HAM=np.zeros(((d-1)*D*(d-1)*D,(d-1)*D*(d-1)*D)).astype(self._mps.dtype)
        for index in range(D*D):
            vec1=np.zeros((D*D))
            vec1[index]=1.0
            out=mf.ExHAproductSingle(l,L,LAA,LAAAA,LAAAA_OneMinusEAAinv,self._mps,self._mpstilde,VL,invsqrtl,invsqrtr,self._mpo,r,R,RAA,RAAAA,OneMinusEAAinv_RAAAA,GSenergy,k,1E-12,vec1)
            HAM[index,:]=out

        print
        print (np.linalg.norm(HAM-herm(HAM)))
        [w,vl]=np.linalg.eigh(1.0/2.0*(HAM+herm(HAM)))
        hg=np.sort(w)
        #print
        #print (hg)
        e=hg[0:10]-GSenergy

        #print e
        #print(k)
        #print(l)
        #e,xopt=mf.eigshExSingle(l,L,LAA,LAAAA,LAAAAEAAinv,self._mps,self._mpstilde,VL,invsqrtl,invsqrtr,self._mpo,r,R,RAA,RAAAA,REAAinvAAAA,GSenergy,k,tolerance=1e-14,numvecs=2,numcv=100,datatype=self._dtype)
        return e,GSenergy
        
