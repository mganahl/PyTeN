"""
@author: Martin Ganahl
"""
from __future__ import absolute_import, division, print_function
from sys import stdout
import pickle
import sympy
import numpy as np
import os,copy
import time
import scipy as sp
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


class Container:

    def __init__(self):
        """
        Base class for simulation objects;
        """
        pass
    
    def save(self,filename):
        """
        dumps a simulation into a pickle file named "filename"
        Parameters:
        -----------------------------------
        filename: str
                  the filename of the file
        """

        with open(filename+'.pickle', 'wb') as f:
            pickle.dump(self,f)
            
    def load(self,filename):
        """
        reads a simulation from a pickle file "filename".pickle
        and returns a container object
        Parameters:
        -----------------------------------
        filename: str
                  the filename of the object to be loaded
        """
        with open(filename+'.pickle', 'rb') as f:
            return pickle.load(f)

    @property
    def mps(self):
        """
        Container.mps:
        return the underlying MPS object
        """
        return self._mps

    
    def measureLocal(self,operators):
        """
        Container.measure(operators):
        measures the expectation values of a list of local operators;
        len(operators) has to be the same as len(self.mps) 
        Parameters:
        --------------------------------------------
        operators: list of np.ndarrays
                   local operators to be measured



        returns:
        --------------------------------------------
        a list of floats containing the expectation values
        """
        return self.mps.measureList(operators)
    def truncateMPS(self,truncation_threshold=1E-10,D=None,tol=1E-10,ncv=20,pinv=1E-200):
        """
        Container.truncateMPS(truncation_threshold=1E-10,D=None,tol=1E-10,ncv=20,pinv=1E-200):
        truncates the mps. After truncation, the mps is in right-orthogonal form, 
        i.e. the mps.pos attribute is mps.pos=0. left and right Hamiltonian environments
        are recalculated as well using the truncated msp, with self._L (left environment)
        being a list of length len(mps)+1 with the only non-empty element being the first one:
        self._L[0]=self._lb. self._lb is the left bundary condition of the simulation
        self._R is a list of length len(mps)+1 holding all right-environment for all bipartitions
        of the chain.

        Parameters:
        -------------------------------------------------------------------------
        truncation_threshold: float
                              desired truncation threshold
        D:                    int
                              maximally allowed bond-dimension after truncation
        tol:                  float
                              precision for the transfer-matrix eigensolver; relevant only for infinite MPS
        ncv:                  int
                              number of krylov vectors in the implicitly restarted arnoldi solver used for transfer-matrix
                              eigen-decomposition; relevanty only for infinite MPS
        pinv:                 float:
                              pseudo-inverse parameter for invertion of reduced density matrices and Schmidt-values
                              change this value with caution; too large values (e.g even values pinv=1E-16) can 
                              cause erratic behaviour

        returns: self
        """
        self._mps.truncate(schmidt_thresh=truncation_threshold,D=D,nmaxit=100000,tol=tol,ncv=ncv,pinv=pinv,r_thresh=1E-14)
        self._mps.position(0)
        self._L=[]*len(self._mps)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)
        return self
        
    def position(self,n):

        """

        Container.position(n)
        shifts the center position of Container.mps to bond n, updates left and right environments
        accordingly
        Parameters:
        ------------------------------------
        n: int
           the bond to which the position should be shifted

        returns: self
        """
        if n>len(self.mps):
            raise IndexError("Container.position(n): n>len(mps)")
        if n<0:
            raise IndexError("Container.position(n): n<0")
        
        if n>=self.mps.pos:
            pos=self.mps.pos            
            self.mps.position(n)
            for m in range(pos,n):
                self._L[m+1]=mf.addLayer(self._L[m],self._mps[m],self._mpo[m],self._mps[m],1)
        if n<self.mps.pos:
            pos=self.mps.pos            
            self.mps.position(n)
            for m in range(pos-1,n-1,-1):
                self._R[len(self.mps)-m]=mf.addLayer(self._R[len(self.mps)-1-m],self._mps[m],self._mpo[m],self._mps[m],-1)                
                
        return self
        
    def update(self):
        """
        Container.update():
        make the Container internally consistent after e.g. changing the mps object.
        The mps.pos attribute after update() is mps.pos=0 (i.e. mps.position(0) is called)
        The left and right Hamiltonian environments  are recalculated using msp, with self._L (left environment)
        being a list of length len(mps)+1 with the only non-empty element being the first one:
        self._L[0]=self._lb. self._lb is the left bundary condition of the simulation
        self._R is a list of length len(mps)+1 holding all right-environment for all bipartitions
        of the chain.

        returns: self
        """
        self._mps.position(0)
        self._L=[]*len(self._mps)        
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)
        return self
        
    
class DMRGengine(Container):
    """
    DMRGengine
    simulation container for density matrix renormalization group optimization

    """
    #left/rightboundary are boundary mpo expressions; pass lb=rb=np.ones((1,1,1)) for obc simulation
    def __init__(self,mps,mpo,filename,lb=None,rb=None):
        """
        initialize an MPS object
        mps:      MPS object
                  the initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        filename: str
                  the name of the simulation
        lb,rb:    None or np.ndarray
                  left and right environment boundary conditions
                  if None, obc are assumed
                  user can provide lb and rb to fix the boundary condition of the mps
                  shapes of lb, rb, mps[0] and mps[-1] have to be consistent
        """
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
        self._mps.position(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,np.copy(self._lb))
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,np.copy(self._rb))

        
    def optimize(self,n,tol=1E-6,ncv=40,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,verbose=0):
        if solver=='LOBPCG':
            e,opt=mf.lobpcg(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),tol)#mps._mat set to 11 during call of __tensor__()
        elif solver=='AR':
            e,opt=mf.eigsh(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),tol,numvecs,ncv)#mps._mat set to 11 during call of __tensor__()
        elif solver=='LAN':
            e,opt=mf.lanczos(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                              deltaEta=landeltaEta)
        else:
            raise ValueError("DMRGengine.optimize: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))
        Dnew=opt.shape[1]
        if verbose>0:
            stdout.write("\rSS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,self._it,self._Nmax,n,self._N,np.real(e),np.imag(e),Dnew))
            stdout.flush()
        if verbose>1:
            print("")
            
            #print ('at iteration {2} optimization at site {0} returned E={1}'.format(n,e,it))
        return e,opt
    
                
    def __simulate__(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        DMRGengine.__simulate__(Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
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
        if solver not in ['AR','LAN','LOBPCG']:
            raise ValueError("DMRGengine.__simulate__: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))
        self._Nmax=Nmax
        converged=False
        energy=100000.0
        self._it=1

        while not converged:
            for n in range(self._mps._N-1):
                self.position(n)
                e,opt=self.optimize(n,tol,ncv,numvecs,solver,Ndiag,nmaxlan,landelta,landeltaEta,verbose)
                self.mps[n]=opt
            for n in range(self._mps._N-1,0,-1):
                self.position(n+1)                
                e,opt=self.optimize(n,tol,ncv,numvecs,solver,Ndiag,nmaxlan,landelta,landeltaEta,verbose)                
                self.mps[n]=opt
                    
            if np.abs(e-energy)<Econv:
                converged=True
            energy=e
            if cp!=None and self._it>0 and self._it%cp==0:
                self._mps.save(self._filename+'_dmrg_cp')
            self._it=self._it+1
            if self._it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        self._mps.resetZ()
        return e

    def simulate(self,*args,**kwargs):
        """
        see __simulate__
        """
        return self.__simulate__(*args,**kwargs)

        
    def __simulateTwoSite__(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        DMRGengine.__simulateTwoSite__(Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
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
        if solver not in ['AR','LAN','LOBPCG']:
            raise ValueError("DMRGengine.__simulateTwoSite__: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))
        self._Nmax=Nmax

        converged=False
        energy=100000.0
        it=1
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
                if solver=='LOBPCG':
                    e,opt=mf.lobpcg(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol)#mps._mat set to 11 during call of __tensor__()
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

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
                if solver=='LOBPCG':
                    e,opt=mf.lobpcg(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol)#mps._mat set to 11 during call of __tensor__()
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[self._mps._N-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                                      deltaEta=landeltaEta)

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
                self._mps.save(self._filename+'_dmrg_cp')                                                
            it=it+1
            if it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        self._mps.resetZ()            
        return e
        #returns the center bond matrix and the gs energy
    def simulateTwoSite(self,*args,**kwargs):
        """
        see __simulateTwoSite__
        """
        return self.__simulateTwoSite__(*args,**kwargs)


class IDMRGengine(DMRGengine):
    """
    IDMRGengine
    container object for performing an IDMRG optimization of the ground-state of a Hamiltonian
    """
    def __init__(self,mps,mpo,filename):
        """
        initialize an IDMRG object
        mps:      MPS object
                  initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        filename: str
                  the name of the simulation
        """
        
        lb,rb,lbound,rbound,self._hl,self._hr=mf.getBoundaryHams(mps,mpo)                            
        super().__init__(mps,mpo,filename,lb,rb)
    #shifts the unit-cell by N/2 by updating self._L, self._R, self._lb, self._rb, and cutting and patching self._mps and self._mpo

    def __update__(self,regauge=False):
        
        self._mps.__position__(self._mps._N)
        #update the left boundary
        for site in range(int(self._mps._N/2)):
            self._lb=mf.addLayer(self._lb,self._mps._tensors[site],self._mpo[site],self._mps._tensors[site],1)
            
        lamR=np.copy(self._mps._mat)
        mps=[]
        D=0
        #cut and patch the right half of the mps
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

        self._mps.__position__(0)
        lamL=np.copy(self._mps._mat)
        #cut and patch the left half of the mps
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
        if not regauge:
            self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
            self._L.insert(0,self._lb)
            self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
            self._R.insert(0,self._rb)
        elif regauge:
            lb,rb,lbound,rbound,self._hl,self._hr=mf.getBoundaryHams(self._mps,self._mpo,regauge=True)
            super().__init__(self._mps,self._mpo,self._filename,lb,rb)        
        return D

    
    def __simulate__(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0):            
        """
        IDMRGengine.__simulate__(Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0)
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
        regaugestep: if > 0, the mps is regauged into symmetric form when it%regaugestep==0; the effective Hamiltonians 
                     are recalculated in this case; do NOT use regaugestep==1
        """
        if regaugestep==1:
            raise ValueError("IDMRGengine.__simulate__(): regaugestep=1 can cause problems, use regaugestep>1 or regaugestep=0 (no regauging)")
        
        print ('# simulation parameters:')
        print ('# of idmrg iterations: {0}'.format(Nmax))
        print ('# of sweeps per unit cell: {0}'.format(NUC))
        print ('# Econv: {0}'.format(Econv))
        print ('# Arnoldi tolerance: {0}'.format(tol))
        print ('# Number of Lanzcos vector in Arnoldi: {0}'.format(ncv))
        it=0
        converged=False
        eold=0.0
        verbose=1
        skip=False
        while not converged:
            regauge=False
            e=super().__simulate__(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,landelta=landelta,landeltaEta=landeltaEta)
            if regaugestep>0 and it%regaugestep==0 and it>0:
                regauge=True
                skip=True
            D=self.__update__(regauge)
            if verbose>0:
                if regauge==False:
                    if skip==False:
                        stdout.write("\rSS-IDMRG using %s solver: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(solver,it,Nmax,np.real((e-eold)/(self._mps._N)),np.imag((e-eold)/(self._mps._N)),D))
                    if skip==True:
                        skip=False
                if regauge==True:
                    stdout.write("\rSS-IDMRG using %s solver: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(solver,it,Nmax,np.real(self._hl/self._N),np.imag(self._hl/self._N),D))
                stdout.flush()  
                if verbose>1:
                    print('')
            #if regauge==True:
                #input('finished updating and regauing, starting fresh now')
                #print ('at iteration {0} optimization returned E/N={1}'.format(it,(e-eold)/(dmrg._mps._N)))
            if cp!=None and it>0 and it%cp==0:
                self._mps.save(self._filename+'_dmrg_cp')                
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        self._mps.resetZ()            
        it=it+1

        
    def simulate(self,*args,**kwargs):
        """
        see __simulate__
        """
        self.__simulate__(*args,**kwargs)

        
    def __simulateTwoSite__(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0):
        """
        IDMRGengine.__simulateTwoSite__(Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0)
        run a twos-site IDMRG simulation

        Nmax:        number of outer iterations
        NUC:         number of optimization sweeps when optimizing a single unitcell
        Econv:       desired convergence of energy per unitcell
        tol:         arnoldi tolerance
        ncv:         number of krylov vectors in arnoldi or lanczos
        cp:          chekpoint step
        verbose:     verbosity flag
        numvecs:     the number of eigenvectors to be calculated; should be 1
        solver:      type of eigensolver: 'AR' or 'LAN' for arnoldi or lanczos
        Ndiag:       lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan:     maximum number of lanczos stesp
        landelta:    lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies
        regaugestep: if > 0, the mps is regauged into symmetric form when it%regaugestep==0; the effective Hamiltonians 
                     are recalculated in this case; do NOT use regaugestep==1

        """
        if regaugestep==1:
            raise ValueError("IDMRGengine.__simulateTwoSite__(): regaugestep=1 can cause problems, use regaugestep>1 or regaugestep=0 (no regauging)")
        print ('# simulation parameters:')
        print ('# of idmrg iterations: {0}'.format(Nmax))
        print ('# of sweeps per unit cell: {0}'.format(NUC))
        print ('# Econv: {0}'.format(Econv))
        print ('# Arnoldi tolerance: {0}'.format(tol))
        print ('# Number of Lanzcos vector in Arnoldi: {0}'.format(ncv))
        it=0
        converged=False
        eold=0.0
        skip=False
        while not converged:
            regauge=False
            e=super().__simulateTwoSite__(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,numvecs=numvecs,truncation=truncation,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,\
                                          landelta=landelta,landeltaEta=landeltaEta)
            if regaugestep>0 and it%regaugestep==0 and it>0:
                regauge=True
                skip=True
            D=self.__update__(regauge)
            if verbose>0:
                if regauge==False:
                    if skip==False:
                        stdout.write("\rSS-IDMRG: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real((e-eold)/(self._mps._N)),np.imag((e-eold)/(self._mps._N)),D))
                    if skip==True:
                        skip=False
                if regauge==True:
                    stdout.write("\rSS-IDMRG: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real(self._hl/self._N),np.imag(self._hl/self._N),D))
                stdout.flush()  
                if verbose>1:
                    print('')
            #if regauge==True:
                #input('finished updating and regauing, starting fresh now')
                #print ('at iteration {0} optimization returned E/N={1}'.format(it,(e-eold)/(dmrg._mps._N)))
            if cp!=None and it>0 and it%cp==0:
                self._mps.save(self._filename+'_dmrg_cp')                                
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        it=it+1
        self._mps.resetZ()
    def simulateTwoSite(self,*args,**kwargs):
        """
        see __simulateTwoSite__
        """
        self.__simulateTwoSite__(*args,**kwargs)

        
class HomogeneousIMPSengine(Container):
    """
    HomogeneousIMPSengine
    container object for homogeneous MPS optimization using a gradient descent method
    """
    def __init__(self,Nmax,mps,mpo,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-10,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40,pinv=1E-100,trunc=1E-16):
        """
        HomogeneousIMPSengine.__init__(Nmax,mps,mpo,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-4,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40):
        initialize a homogeneous gradient optimization
        
        MPS optimization methods for homogeneous systems
        uses gradient optimization to find the ground state of a Homogeneous system
        mps (np.ndarray of shape (D,D,d)): an initial mps tensor 
        mpo (np.ndarray of shape (M,M,d,d)): the mpo tensor
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
        self._pinv=pinv
        self._trunc=trunc
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

    def update(self):
        raise NotImplementedError("")

    def position(self):
        raise NotImplementedError("")        

    def mps(self):
        raise NotImplementedError("")
    def truncateMPS(self):
        raise NotImplementedError("")
    
    
    def __doGradStep__(self):
        converged=False
        self._gamma,self._lam,trunc=mf.regauge(self._mps,gauge='symmetric',initial=np.reshape(np.diag(self._lam**2),self._D*self._D),\
                                               nmaxit=10000,tol=self._tol,ncv=self._ncv,numeig=self._numeig,trunc=self._trunc,Dmax=self._D,pinv=self._pinv)
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
        A_,s,v,Z=mf.prepareTruncate(self._mps,direction=1,thresh=1E-14)
        s=s/np.linalg.norm(s)
        self._mps=np.transpose(np.tensordot(A_,np.diag(s).dot(v),([1],[0])),(0,2,1))
        if self._normgrad<self._epsilon:
            converged=True


        return Elocleft,leftn,rightn,converged
    
    def simulate(self,*args,**kwargs):
        self.__simulate__(*args,**kwargs)
        
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
            stdout.write("\rit %i: local E=%.16f, lnorm=%.6f, rnorm=%.6f, grad=%.16f, alpha=%.4f, D=%i" %(self._it,np.real(Elocleft),leftn,rightn,self._normgrad,self._alpha_,self._D))
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
        
    def simulate(self,*args,**kwargs):
        self.__simulate__(*args,**kwargs)
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
        


class VUMPSengine(Container):
    """

    VUMPSengine
    container object for mps ground-state optimization using the VUMPS algorithm
    """
    def __init__(self,mps,mpo,filename):
        """
        initialize a VUMPS simulation object
        mps: a list of a single np.ndarray of shape(D,D,d), or an MPS object of length 1
        mpo: an MPO object
        filename: str
                  the name of the simulation
        """
        if len(mps)>1:
            raise ValueError("VUMPSengine: got an mps of len(mps)>1; VUMPSengine can only handle len(mps)=1")
        if len(mpo)>1:
            raise ValueError("VUMPSengine: got an mpo of len(mps)>1; VUMPSengine can only handle len(mpo)=1")
        
        self._dtype=np.result_type(mps[0].dtype,mpo.dtype)

        self._mps=copy.deepcopy(mps)
        self._D=np.shape(mps[0])[0]
        self._filename=filename
        
        self._kleft=np.random.rand(self._D,self._D)
        self._kright=np.random.rand(self._D,self._D)
        self._it=1
        [B1,B2,d1,d2]=np.shape(mpo[0])

        mpol=np.zeros((1,B2,d1,d2),dtype=self._dtype)
        mpor=np.zeros((B1,1,d1,d2),dtype=self._dtype)

        mpol[0,:,:,:]=mpo[0][-1,:,:,:]
        mpol[0,0,:,:]/=2.0
        mpor[:,0,:,:]=mpo[0][:,0,:,:]
        mpor[-1,0,:,:]/=2.0
        self._mpo=H.MPO.fromlist([mpol,mpo[0],mpor])

    def __doStep__(self):
        [etar,vr,numeig]=mf.TMeigs(self._A,direction=-1,numeig=self._numeig,init=self._r,nmax=10000,tolerance=self._tol,ncv=self._ncv,which='LR')
        [etal,vl,numeig]=mf.TMeigs(self._B,direction=1,numeig=self._numeig,init=self._r,nmax=10000,tolerance=self._tol,ncv=self._ncv,which='LR')
        l=np.reshape(vl,(self._D,self._D))
        r=np.reshape(vr,(self._D,self._D))
        l=l/np.trace(l)
        r=r/np.trace(r)
        
        self._l=(l+herm(l))/2.0
        self._r=(r+herm(r))/2.0            

        leftn=np.linalg.norm(np.tensordot(self._A,np.conj(self._A),([0,2],[0,2]))-np.eye(self._D))
        rightn=np.linalg.norm(np.tensordot(self._B,np.conj(self._B),([1,2],[1,2]))-np.eye(self._D))

        self._lb=mf.initializeLayer(self._A,np.eye(self._D),self._A,self._mpo[0],1) 
        ihl=mf.addLayer(self._lb,self._A,self._mpo[2],self._A,1)[:,:,0]
        Elocleft=np.tensordot(ihl,self._r,([0,1],[0,1]))
        self._rb=mf.initializeLayer(self._B,np.eye(self._D),self._B,self._mpo[2],-1)
        ihr=mf.addLayer(self._rb,self._B,self._mpo[0],self._B,-1)[:,:,-1]
        Elocright=np.tensordot(ihr,self._l,([0,1],[0,1]))

        ihlprojected=(ihl-np.tensordot(ihl,r,([0,1],[0,1]))*np.eye(self._D))
        ihrprojected=(ihr-np.tensordot(l,ihr,([0,1],[0,1]))*np.eye(self._D))
        
        self._kleft=mf.RENORMBLOCKHAMGMRES(self._A,self._A,self._l,np.eye(self._D),ihlprojected,x0=np.reshape(self._kleft,self._D*self._D),tolerance=self._lgmrestol,\
                                           maxiteration=self._Nmaxlgmres,direction=1)
        self._kright=mf.RENORMBLOCKHAMGMRES(self._B,self._B,np.eye(self._D),self._r,ihrprojected,x0=np.reshape(self._kright,self._D*self._D),tolerance=self._lgmrestol,\
                                            maxiteration=self._Nmaxlgmres,direction=-1)
        
        self._lb[:,:,0]+=np.copy(self._kleft)
        self._rb[:,:,-1]+=np.copy(self._kright)


        AC_=mf.HAproductSingleSiteMPS(self._lb,self._mpo[1],self._rb,self._mps[0])
        C_=mf.HAproductZeroSiteMat(self._lb,self._mpo[1],self._A,self._rb,position='right',mat=self._mat)
        self._gradnorm=np.linalg.norm(AC_-ncon.ncon([self._A,C_],[[-1,1,-3],[1,-2]]))

        if self._solver=='AR':
            e1,mps=mf.eigsh(self._lb,self._mpo[1],self._rb,self._mps[0],self._artol_,numvecs=1,numcv=self._arncv,numvecs_returned=self._arnumvecs)#mps._mat set to 11 during call of __tensor__()
            e2,self._mat=mf.eigshbond(self._lb,self._mpo[1],self._A,self._rb,self._mat,position='right',tolerance=self._artol_,numvecs=self._arnumvecs,numcv=self._arncv)
            if self._arnumvecs>1:        
                self._mat/=np.linalg.norm(self._mat[0])
                self._gap=e1[1]-e1[0]
                self._mps[0]=mps[0]
            else:
                self._mps[0]=mps
        elif self._solver=='LAN':
            e1,mps=mf.lanczos(self._lb,self._mpo[1],self._rb,self._mps[0],self._artol,self._Ndiag,self._nmaxlan,self._arnumvecs,self._landelta,deltaEta=self._artol)
            e2,self._mat=mf.lanczosbond(self._lb,self._mpo[1],self._A,self._rb,self._mat,'right',self._Ndiag,self._nmaxlan,self._arnumvecs,delta=self._landelta,deltaEta=self._artol)
            if self._arnumvecs>1:        
                self._gap=e1[1]-e1[0]
                self._mps[0]=mps[0]
                self._mat=self._mat[0]
            else:
                self._mps[0]=mps
        else:
            raise ValueError("in VUMPSengine: unknown solver type; use 'AR' or 'LAN'")
        D1,D2,d=self._mps[0].shape

        if self._svd:
            ACC_l=np.reshape(ncon.ncon([self._mps[0],herm(self._mat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
            CAC_r=np.reshape(ncon.ncon([herm(self._mat),self._mps[0]],[[-1,1],[1,-2,-3]]),(D1,d*D2))
            Ul,Sl,Vl=mf.svd(ACC_l)
            Ur,Sr,Vr=mf.svd(CAC_r)
            self._A=np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
            self._B=np.reshape(Ur.dot(Vr),(D1,D2,d))

        else:
            AC_l=np.reshape(np.transpose(self._mps[0],(0,2,1)),(D1*d,D2))
            AC_r=np.reshape(self._mps[0],(D1,d*D2))
            
            UAC_l,PAC_l=sp.linalg.polar(AC_l,side='right')
            UAC_r,PAC_r=sp.linalg.polar(AC_r,side='left')
            
            UC_l,PC_l=sp.linalg.polar(self._mat,side='right')
            UC_r,PC_r=sp.linalg.polar(self._mat,side='left')
            
            self._A=np.transpose(np.reshape(UAC_l.dot(herm(UC_l)),(D1,d,D2)),(0,2,1))
            self._B=np.reshape(herm(UC_r).dot(UAC_r),(D1,D2,d))

        return Elocleft,leftn,rightn

    
    def simulate(self,*args,**kwargs):
        """
        see __simulate__
        """
        return self.__simulate__(*args,**kwargs)
        
    def __simulate__(self,Nmax,epsilon=1E-10,tol=1E-10,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40,\
                     artol=1E-10,arnumvecs=1,arncv=20,svd=False,Ndiag=10,nmaxlan=500,landelta=1E-8,solver='AR',checkpoint=100):

        """
        do a VUMPS simulation:
        Parameters
        ---------------------------------------------
        Nmax:            int
                         number of iterations
        epsilon:         float
                         desired convergence
        tol:             float
                         precision of the left and right reduced steady-density matrices
        lgmrestol:       float
                         precision of the left and right renormalized environments
        ncv:             int
                         number of krylov vectors used in sparse transfer-matrix eigendecomposition
        numeig:          number of eigenvectors to be returned bei eigs when computing the left and right reduced steady state density matrices
        Nmaxlgmres:      int
                         maximum iteration steps of lgmres when calculating the left and right renormalized environments
        artol:           float
                         precision of arnoldi eigsh eigensolver
        arnumvecs:       int
                         number of eigenvectors to be calculated by arnoldi; if > 1, the gap to the second eigenvalue is printed out during simulation
        arncv:           int
                         number of krylov vectors used in sparse eigsh of the effective Hamiltonian
        svd:             bool
                         if True, do an svd instead of polar decomposition for gauge matching
        checkpoint:      int
                         if > 0, simulation is checkpointed every "checkpoint" steps
        """
        converged=False
        self._tol=tol
        self._svd=svd
        self._artol=artol
        self._arnumvecs=arnumvecs
        self._arncv=arncv
        self._lgmrestol=lgmrestol
        self._numeig=numeig
        self._ncv=ncv
        self._epsilon=epsilon
        self._Nmaxlgmres=Nmaxlgmres
        self._Ndiag=Ndiag
        self._nmaxlan=nmaxlan
        self._landelta=landelta
        self._solver=solver
        self._A,self._l=mf.regauge(self._mps[0],gauge='left',initial=None,nmaxit=10000,tol=self._tol,ncv=self._ncv,numeig=self._numeig)
        self._B,self._r=mf.regauge(self._mps[0],gauge='right',initial=None,nmaxit=10000,tol=self._tol,ncv=self._ncv,numeig=self._numeig)
        self._mat=np.eye(self._A.shape[1])

        while converged==False:
            if self._it<10:
                self._artol_=1E-6
            else:
                self._artol_=self._artol
            Edens,leftn,rightn=self.__doStep__()
            if self._it>=Nmax:
                break
            if (checkpoint>0) and (self._it%checkpoint==0):
                np.save('CP_mps'+self._filename,self._mps)
            self._it+=1
            if self._arnumvecs==1:
                stdout.write("\rusing %s solver: it %i: local E=%.16f, D=%i, gradient norm=%.16f" %(self._solver,self._it,np.real(Edens),self._D,self._gradnorm))
                stdout.flush()
            if self._arnumvecs>1:
                stdout.write("\rusing %s solver: it %i: local E=%.16f, gap=%.16f, D=%i, gradient norm=%.16f" %(self._solver,self._it,np.real(Edens),np.real(self._gap),self._D,self._gradnorm))
                stdout.flush()
            if self._gradnorm<self._epsilon:
                converged=True
        print
        print()
        if self._it>=Nmax and (converged==False):
            print ('simulation reached maximum number of steps ({1}) and stopped at precision of {0}'.format(self._gradnorm,Nmax))
        if converged==True:
            print ('simulation converged to {0} in {1} steps'.format(self._epsilon,Nmax))
        print
        return Edens
        

class TimeEvolutionEngine(Container):
    """
    TimeEvolutionEngine(Container):
    container object for performing real/imaginary time evolution using TEBD or TDVP algorithm for finite systems 
    """

    @classmethod
    def TEBD(cls,mps,gatecontainer,filename):
        """
        TimeEvolutionEngine.TEBD(mps,mpo,filename):
        initialize a TEBD vsimulation; this is an engine for real or imaginary time evolution using TEBD
        Parameters:
        --------------------------------------------------------
        mps:           MPS object
                       the initial state 
        gatecontainer: nearest neighbor MPO object or a method f(n,m) which returns two-site gates at sites (n,m)
                       The Hamiltonian/generator of time evolution
        filename:      str
                       the filename under which cp results will be stored (not yet implemented)
        lb,rb:         None or np.ndarray
                       left and right environment boundary conditions
                       if None, obc are assumed
        """
        
        cls._gates=copy.deepcopy(gatecontainer)
        cls._mps=copy.deepcopy(mps)
        cls._mpo=copy.deepcopy(mpo)
        cls._filename=filename        
        return cls

    @classmethod
    def TDVP(cls,mps,mpo,filename,lb=None,rb=None):
        """
        TimeEvolutionEngine.TDVP(mps,mpo,filename):
        initialize a TDVP simulation; this is an engine for real or imaginary time evolution using TDVP
        Parameters:
        --------------------------------------------------------
        mps:           MPS object
                       the initial state 
        mpo:           MPO object, or (for TEBD) a method f(n,m) which returns two-site gates at sites (n,m)
                       The Hamiltonian/generator of time evolution
        filename:      str
                       the filename under which cp results will be stored (not yet implemented)
        lb,rb:         None or np.ndarray
                       left and right environment boundary conditions
                       if None, obc are assumed
        """
        
        return cls(mps,mpo,filename,lb,rb)
    
    def __init__(self,mps,mpo,filename,lb=None,rb=None):
        """
        TimeEvolutionEngine.__init__(mps,mpo,filename):
        initialize a TDVP or TEBD  simulation; this is an engine for real or imaginary time evolution
        Parameters:
        --------------------------------------------------------
        mps:           MPS object
                       the initial state 
        mpo:           MPO object, or (for TEBD) a method f(n,m) which returns two-site gates at sites (n,m), or a nearest neighbor MPO
                       The Hamiltonian/generator of time evolution
        filename:      str
                       the filename under which cp results will be stored (not yet implemented)
        lb,rb:         None or np.ndarray
                       left and right environment boundary conditions
                       if None, obc are assumed
        """
        
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


        self._gates=copy.deepcopy(mpo)
        
        self._filename=filename
        self._N=mps._N
        self._mps.__position__(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)
        self._t0=0.0
        self._it=0
        self._tw=0
        self._maxD=max(self._mps.D)
        
    @property
    def iteration(self):
        return self._it
    
    @property
    def time(self):
        return self._t0
    @property
    def truncatedWeight(self):
        return self._tw

    def reset(self):
        """
        resets iteration counter, time-accumulator and truncated-weight accumulator,
        i.e. self.time=0.0 self.iteration=0, self.truncatedWeight=0.0 afterwards.
        """
        self._t0=0.0
        self._it=0
        self._tw=0.0
        
    def initializeTDVP(self):
        """
        updates the TimeevolutionEngine by recalculating left and right environment blocks
        such that the mps can be evolved with the mpo
        """
        self._mps.__position__(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)

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

    def doTEBD(self,*args,**kwargs):
        """
        see __doTEBD__
        """
        return self.__doTEBD__(*args,**kwargs)
    
    def __doTEBD__(self,dt,numsteps,Dmax,tr_thresh,verbose=1,cp=None,keep_cp=False):
        """
        TEBDengine.__doTEBD__(self,dt,numsteps,Dmax,tr_thresh,verbose=1,cnterset=0,tw=0,cp=None):
        uses a second order trotter decomposition to evolve the state using TEBD
        Parameters:
        -------------------------------
        dt:        float
                   step size (scalar)
        numsteps:  int
                   total number of evolution steps
        Dmax:      int
                   maximum bond dimension to be kept
        tr_thresh: float
                   truncation threshold 
        verbose:   int
                   verbosity flag; put to 0 for no output
        cp:        int or None
                   checkpointing flag: checkpoint every cp steps
        keep_cp:   bool
                   if True, keep all checkpointed files, if False, only keep the last one
        """

        #even half-step:
        current='None'
        self.applyEven(dt/2.0,Dmax,tr_thresh)
        for step in range(numsteps):
            #odd step updates:
            self.applyOdd(dt,Dmax,tr_thresh)
            if verbose>=1:
                self._t0+=np.abs(np.imag(dt))
                stdout.write("\rTEBD engine: t=%4.4f truncated weight=%.16f at D/Dmax=%i/%i, truncation threshold=%1.16f, |dt|=%1.5f"%(self._t0,self._tw,np.max(self.mps.D),Dmax,tr_thresh,np.abs(dt)))
                stdout.flush()
            if verbose>=2:
                print('')
            #if this is a cp step, save between two half-steps
            if (cp!=None) and (self._it>0) and (self._it%cp==0):
                #if the cp step does not coincide with the last step, do a half step, save, and do another half step
                if step<(numsteps-1):                
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
                    if not keep_cp:
                        if os.path.exists(current+'.pickle'):
                            os.remove(current+'.pickle')
                        current=self._filename+'_tebd_cp'+str(self._it)
                        self.save(current)
                    else:
                        current=self._filename+'_tebd_cp'+str(self._it)
                        self.save(current)
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
                #if the cp step coincides with the last step, only do a half step and save the state
                else:
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
                    newname=self._filename+'_tebd_cp'+str(self._it)
                    self.save(newname)                    
            #if step is not a cp step:
            else:
                #do a regular full step, unless step is the last step
                if step<(numsteps-1):
                    self.applyEven(dt,Dmax,tr_thresh)
                #if step is the last step, do a half step
                else:
                    self.applyEven(dt/2.0,Dmax,tr_thresh)
            self._it=self._it+1
            self._mps.resetZ()
        return self._tw,self._t0


    def doTDVP(self,*args,**kwargs):
        return self.__doTDVP__(*args,**kwargs)

    def __doTDVP__(self,dt,numsteps,krylov_dim=10,cp=None,keep_cp=False,verbose=1,solver='RK45',rtol=1E-6,atol=1e-12):
        """
        TDVPengine.__doTDVP__(dt,numsteps,krylov_dim=20,cp=None,verbose=1,use_split_step=False)
        does a TDVP real or imaginary time evolution

        dt: step size
        numsteps: number of steps to be performed
        krylov_dim: if use_split_step=False, krylov_dim is the dimension of the krylov space used to perform evolution with lanczos
                    if use_split_step=True, method uses Ash Milsted's implementation of gexpmv (see evoMPS)
        cp: checkpointing (currently not implemented)
        keep_cp: bool
                 if True, keep all checkpointed files, if False, only keep the last one
        verbose: verbosity flag
        solver: str in {'LAN','RK45,'Radau','SEXPMV','LSODA','BDF'}: 
                different intergration schemes
        """

        if solver not in ['LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23']:
            raise ValueError("TDVPengine.__doTDVP__(): unknown solver type {0}; use {'LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23'}".format(solver))
        converged=False
        current='None'
        self._mps.__position__(0)
        for step in range(numsteps):
            for n in range(self._mps._N):
                if n==self._mps._N-1:
                    dt_=dt
                else:
                    dt_=dt/2.0
                self._mps.__position__(n+1)
                #evolve tensor forward
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                    evTen=mf.evolveTensorsolve_ivp(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),np.imag(dt_),method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
                elif solver=='LAN':
                    evTen=mf.evolveTensorLan(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_,krylov_dimension=krylov_dim) #clear=True resets self._mat to identity
                elif solver=='SEXPMV':                    
                    evTen=mf.evolveTensorSexpmv(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_)
                    
                tensor,mat,Z=mf.prepareTensor(evTen,1)
                self._mps[n]=tensor
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)

                #evolve matrix backward                    
                if n<(self._mps._N-1):
                    if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                        evMat=mf.evolveMatrixsolve_ivp(self._L[n+1],self._R[self._mps._N-1-n],mat,-np.imag(dt_),method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
                    elif solver=='LAN':                        
                        evMat=mf. evolveMatrixLan(self._L[n+1],self._R[self._mps._N-1-n],mat,-dt_,krylov_dimension=krylov_dim)
                    elif solver=='SEXPMV':                                            
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
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:                    
                    evMat=mf.evolveMatrixsolve_ivp(self._L[n+1],self._R[self._mps._N-1-n],self._mps._mat,-np.imag(dt_),method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity                    
                elif solver=='LAN':                
                    evMat=mf. evolveMatrixLan(self._L[n+1],self._R[self._mps._N-1-n],self._mps._mat,-dt_,krylov_dimension=krylov_dim)
                elif solver=='SEXPMV':                                    
                    evMat=mf.evolveMatrixSexpmv(self._L[n+1],self._R[self._mps._N-1-n],self._mps._mat,-dt_)                                    
                evMat/=np.linalg.norm(evMat)#normalize wavefunction
                self._mps._mat=evMat        #set evolved matrix as new center-matrix
            
            
                #evolve tensor forward: the back-evolved center matrix is absorbed into the left-side tensor, and the product is evolved forward in time
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:                                    
                    evTen=mf.evolveTensorsolve_ivp(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),np.imag(dt_),method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
                elif solver=='LAN':                                
                    evTen=mf. evolveTensorLan(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=True),dt_,krylov_dimension=krylov_dim)
                elif solver=='SEXPMV':                                                        
                    evTen=mf.evolveTensorSexpmv(self._L[n],self._mpo[n],self._R[self._mps._N-1-n],self._mps.__tensor__(n,clear=False),dt_)
                    
                #split of a center matrix C ("mat" in my notation)
                tensor,mat,Z=mf.prepareTensor(evTen,-1) #mat is already normalized (happens in prepareTensor)
                self._mps[n]=tensor

                self._mps._mat=mat
                self._mps._position=n
            if verbose>=1:
                self._t0+=np.abs(np.imag(dt))                
                stdout.write("\rTDVP engine using %s solver: t=%4.4f, D=%i, |dt|=%1.5f"%(solver,self._t0,np.max(self._mps.D),np.abs(dt)))
                stdout.flush()
            if verbose>=2:                
                print('')
            if (cp!=None) and (self._it>0) and (self._it%cp==0):
                if not keep_cp:
                    if os.path.exists(current+'.pickle'):
                        os.remove(current+'.pickle')
                    current=self._filename+'_tdvp_cp'+str(self._it)
                    self.save(current)
                else:
                    current=self._filename+'_tdvp_cp'+str(self._it)
                    self.save(current)

            self._it=self._it+1
        self._mps.position(0)
        self._mps.resetZ()        
        return self._t0
        #returns the center bond matrix and the gs energy

# ============================================================================     everything below this line is still in development =================================================
def matvec(Heff,Henv,mpo,vec):
    D=Heff.shape[0]
    d=mpo.shape[2]
    tensor=np.reshape(vec,(D,D,d))
    return np.reshape(ncon.ncon([Heff,mpo,tensor],[[1,-1,3,4,-2,5],[3,5,2,-3],[1,4,2]])+ncon.ncon([Henv,tensor],[[1,-1,2,-2],[1,2,-3]]),(D*D*d))

def gram(Neff,mpo,vec):

    D=Neff.shape[0]
    d=mpo.shape[2]
    tensor=np.reshape(vec,(D,D,d))
    return np.reshape(ncon.ncon([Neff,mpo,tensor],[[1,-1,3,4,-2,5],[3,5,2,-3],[1,4,2]]),(D*D*d))

#def gram(Neff,d,vec):
#    D=Neff.shape[0]
#    tensor=np.reshape(vec,(D,D,d))
#    return np.reshape(ncon.ncon([Neff,tensor],[[1,-1,2,-2],[1,2,-3]]),(D*D*d))


def matvecbond(Heff,vec):
    D=Heff.shape[0]
    mat=np.reshape(vec,(D,D))
    return np.reshape(ncon.ncon([Heff,mat],[[1,-1,2,-2],[1,2]]),(D*D))

def grambond(Neff,vec):
    D=Neff.shape[0]
    mat=np.reshape(vec,(D,D))
    return np.reshape(ncon.ncon([Neff,mat],[[1,-1,2,-2],[1,2]]),(D*D))



class PeriodicMPSengine(Container):
    def __init__(self,mps,mpo,N,filename):
        """
        initialize a VUMPS simulation object
        mps: a list of a single np.ndarray of shape(D,D,d), or an MPS object of length 1
        mpo: an MPO object
        filename: str
                  the name of the simulation
        """
        if len(mps)>1:
            raise ValueError("PeriodicMPSengine: got an mps of len(mps)>1; VUMPSengine can only handle len(mps)=1")
        if (mps.obc==True):
            raise ValueError("PeriodicMPSengine: got an mps with obc=True")
        self._dtype=np.result_type(mps[0].dtype,mpo.dtype)
        self._N=N

        [B1,B2,d1,d2]=mpo[0].shape
        mpol=np.zeros((1,B2,d1,d2),dtype=self._dtype)
        mpor=np.zeros((B1,1,d1,d2),dtype=self._dtype)
        
        
        mpolist2=[]
        mpolist2.append(np.copy(mpo[0][-1,:,:,:]))
        mpolist2.append(np.copy(mpo[0]))
        mpolist2.append(np.copy(mpo[0][:,0,:,:]))
        self._mpo2=H.MPO.fromlist(mpolist2)
        
        self._mpo3=H.MPO.fromlist(mpolist2)
        self._mpo3[0][0,:,:]/=2.0
        self._mpo3[2][-1,:,:]/=2.0
        


        mpol[0,:,:,:]=mpo[0][-1,:,:,:]
        mpor[:,0,:,:]=mpo[0][:,0,:,:]
        mpol[0,0,:,:]*=0.0
        mpor[-1,0,:,:]*=0.0

        mpolist=[]
        mpolist.append(np.copy(mpol))
        mpolist.append(np.copy(mpo[0]))
        mpolist.append(np.copy(mpor))
        self._mpo=H.MPO.fromlist(mpolist)
        self._mps=copy.deepcopy(mps)
        self._D=np.shape(mps[0])[0]
        self._filename=filename
        self._it=1


    def empty_pow(self,N):
        empty=ncon.ncon([self._mps[0],np.conj(self._mps[0])],[[-3,-1,1],[-4,-2,1]])
        for n in range(N-1):
            empty=ncon.ncon([empty,self._mps[0],np.conj(self._mps[0])],[[-1,-2,1,3],[-3,1,2],[-4,3,2]])
        return empty

    def checkH(self,N):
        Hlocall=ncon.ncon([self._mps[0],self._mpo2[0],np.conj(self._mps[0]),self._mps[0],self._mpo3[2],np.conj(self._mps[0])],[[-3,1,2],[3,2,4],[-4,5,4],[1,-1,7],[3,7,6],[5,-2,6]])
        Hlocal=ncon.ncon([self._mps[0],self._mpo3[0],np.conj(self._mps[0]),self._mps[0],self._mpo3[2],np.conj(self._mps[0])],[[-3,1,2],[3,2,4],[-4,5,4],[1,-1,7],[3,7,6],[5,-2,6]])
        Hlocalr=ncon.ncon([self._mps[0],self._mpo3[0],np.conj(self._mps[0]),self._mps[0],self._mpo2[2],np.conj(self._mps[0])],[[-3,1,2],[3,2,4],[-4,5,4],[1,-1,7],[3,7,6],[5,-2,6]])                
        empty=ncon.ncon([self._mps[0],np.conj(self._mps[0])],[[-3,-1,1],[-4,-2,1]])
        H=np.zeros(Hlocal.shape,dtype=Hlocal.dtype)
        for n in range(N):
            if n>0 and (N-n-2)>0:
                H+=ncon.ncon([self.empty_pow(n),Hlocal,self.empty_pow(N-n-2)],[[3,4,-3,-4],[1,2,3,4],[-1,-2,1,2]])
            if n==0 and (N-n-2)>0:
                H+=ncon.ncon([Hlocall,self.empty_pow(N-n-2)],[[1,2,-3,-4],[-1,-2,1,2]])
            if n>0 and (N-n-2)==0:
                H+=ncon.ncon([self.empty_pow(n),Hlocalr],[[1,2,-3,-4],[-1,-2,1,2]])
        return H
        
    def getHeffNeff(self):
        [D1,D2,d]=self._mps[0].shape
        Hbla=ncon.ncon([self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[-4,-1,1],[-6,-3,1,2],[-5,-2,2]])
        n=1
        while n<(self._N-1):
            Hbla=ncon.ncon([Hbla,self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[-1,-2,-3,1,5,3],[-4,1,2],[-6,3,2,4],[-5,5,4]])
            #bla=np.zeros((D1,D1),dtype=Hbla.dtype)
            #for k in range(D1):
            #    bla+=Hbla[:,:,0,k,k,0]
            #print(bla)
            #input()
            n+=1            
            #if n==(self._N-3):
            #    #N_temp contains all mps-tensors except those at 0,N-2 and N-1; will be contracted with local mpo to connect both ends
            #    N_temp=np.copy(Hbla[:,:,0,:,:,0])
            #    #print('length of N_temp: ',n,self._N)
            #print('contracted ',n,self._N)

        #Neffbond has all mps tensors contracted into it
        #Neffbond=ncon.ncon([Hbla,self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[-1,-2,-3,1,5,3],[-4,1,2],[-6,3,2,4],[-5,5,4]])[:,:,0,:,:,0]
        Neffbond=self.empty_pow(self._N)
        N_temp=self.empty_pow(self._N-3)
        #Neffbond has all but the N-1st mps tensors contracted into it        
        Neffsite=np.zeros((D1,D1,1,D2,D2,1),dtype=Hbla.dtype)            
        Neffsite[:,:,0,:,:,0]=Hbla[:,:,0,:,:,0]
        #bla=np.zeros((D1,D1),dtype=Neffsite.dtype)
        #for k in range(D1):
        #    #bla+=Neffsite[:,:,0,k,k,0]
        #    bla+=Neffbond[:,:,k,k]
        #print(bla)
        #input()        
        Henvsite=Hbla[:,:,0,:,:,-1]
        #H2=self.checkH(self._N-1)
        #print(np.linalg.norm(Henvsite-H2))
        #print(np.linalg.norm(Neffsite[:,:,0,:,:,0]-self.empty_pow(self._N-1)))
        #print(np.linalg.norm(Neffbond-self.empty_pow(self._N)))
        #print(np.linalg.norm(N_temp-self.empty_pow(self._N-3)))        
        #input()
        #Hevn contains all hamiltonian contributions that don't act on the last site N-1
        #N_temp contains the mps overlap on sites 1,...,N-3
        #Heffsite has will be contracted with a local mpo self._mpo[1] to get the full Hamiltonian
        Heffsite=ncon.ncon([N_temp,self._mps[0],self._mpo[0],np.conj(self._mps[0]),self._mps[0],self._mpo[2],np.conj(self._mps[0])],[[1,4,5,8],[1,-1,2],[9,-3,2,3],[4,-2,3],[-4,5,6],[-6,9,6,7],[-5,8,7]])

        #Heffsite[:,:,-1,:,:,-1]+=Henv
        Heffbond=ncon.ncon([Heffsite,self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[1,5,3,-3,-4,6],[1,-1,2],[3,6,2,4],[5,-2,4]])
        #Henvbond=ncon.ncon([Henvsite,self._mps[0],np.conj(self._mps[0]),self._mps[0],np.conj(self._mps[0])],[[1,3,4,6],[1,-1,2],[3,-2,2],[-3,4,5],[-4,6,5]])
        Henvbond=ncon.ncon([Henvsite,self._mps[0],np.conj(self._mps[0])],[[1,3,-3,-4],[1,-1,2],[3,-2,2]])
        #Heffsite has will be contracted with a local mpo self._mpo[1] to get the full Hamiltonian        

        return Heffsite,Neffsite,Henvsite,Heffbond,Neffbond,Henvbond
        
    def gradient_optimize(self,alpha=0.05,Econv=1E-3,Nmax=1000,verbose=1):
        converged=False
        it=0
        eold=1E10
        while not converged:
            self._mps.canonize()
            Heffsite,Neffsite,Henvsite,Heffbond,Neffbond,Henvbond=self.getHeffNeff()
            [D1,D2,d]=self._mps[0].shape
            mvsite=fct.partial(matvec,*[Heffsite,Henvsite,self._mpo[1]])
            #mvbond=fct.partial(matvecbond,*[Heffbond+Henvbond])            
            LOPsite=LinearOperator((D1*D2*d,D1*D2*d),matvec=mvsite,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            #LOPbond=LinearOperator((D1*D2,D1*D2),matvec=mvbond,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            #print(Neffsite.shape)
            #input()
            Neffsite_=ncon.ncon([Neffsite,self._mps._mat,np.conj(self._mps._mat)],[[-1,-2,-3,1,2,-6],[-4,1],[-5,2]])
            
            #AC=ncon.ncon([self._mps[0],self._mps._mat],[[-1,1,-3],[1,-2]])
            #gradmps=np.reshape(mvsite(AC),(self._mps[0].shape))                        
            gradAC=np.reshape(mvsite(self._mps[0]),(self._mps[0].shape))
            gradmps=ncon.ncon([gradmps,np.diag(1.0/np.diag(self._mps._mat))],[[-1,1,-3],[1,-2]])            

            #gradmat=np.reshape(mvbond(np.eye(D1)),(D1,D2))
            
            energy=np.tensordot(np.conj(self._mps[0]),gradmps,([0,1,2],[0,1,2]))
            Z=np.trace(np.reshape(Neffbond,(D1*D1,D2*D2)))
            edens=energy/Z/self._N
            
            self._mps[0]-=np.copy(alpha*gradmps)
            self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(self._N))))

            #mps=self._mps[0]-alpha*gradmps
            #mat=np.eye(D1)-alpha*gradmat
            #ACC_l=np.reshape(ncon.ncon([mps,herm(mat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
            #Ul,Sl,Vl=mf.svd(ACC_l)
            #self._mps[0]=np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
            #self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(self._N))))
            
            self._mps._mat=np.eye(D1)
            self._mps._connector=np.eye(D1)
            if verbose>0:
                stdout.write("\rPeriodic MPS gradient optimization for N=%i sites: it=%i/%i, E=%.16f+%.16f at alpha=%1.5f, D=%i"%(self._N,it,Nmax,np.real(edens),np.imag(edens),alpha,D1))
                stdout.flush()

            if np.abs(eold-edens)<Econv:
                converged=True
                if verbose>0:
                    print()
                    print ('energy converged to within {0} after {1} iterations'.format(Econv,it))
            eold=edens
            it+=1
            if it>Nmax:
                if verbose>0:
                    print()
                    print ('simulation did not converge to desired accuracy of {0} within {1} iterations '.format(Econv,Nmax))
                break

    def simulateVUMPS(self):
        converged=False
        while not converged:
            calls=[0]
            stop=2
            
            self._mps.canonize()
            Heffsite,Neffsite,Henvsite,Heffbond,Neffbond,Henvbond=self.getHeffNeff()
            [D1,D2,d]=self._mps[0].shape                                

            mvsite=fct.partial(matvec,*[calls,stop,Heffsite,Henvsite,self._mpo[1]])
            vvsite=fct.partial(gram,*[Neffsite,np.reshape(np.eye(d),(1,1,d,d))])

            LOPsite=LinearOperator((D1*D2*d,D1*D2*d),matvec=mvsite,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            Msite=LinearOperator((D1*D2*d,D1*D2*d),matvec=vvsite,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            
            mvbond=fct.partial(matvecbond,*[calls,stop,Heffbond+Henvbond])
            vvbond=fct.partial(matvecbond,*[[0],1000,Neffbond])
            
            LOPbond=LinearOperator((D1*D2,D1*D2),matvec=mvbond,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            Mbond=LinearOperator((D1*D2,D1*D2),matvec=vvbond,rmatvec=None,matmat=None,dtype=Heffsite.dtype)
            
            nmax=10000
            tolerance=10.0
            ncv=1

            gradmps=np.reshape(mvsite(self._mps[0]),(self._mps[0].shape))
            energy=np.tensordot(np.conj(self._mps[0]),gradmps,([0,1,2],[0,1,2]))
            Z=np.trace(np.reshape(Neffbond,(D1*D1,D2*D2)))


            etasite,vecsite=sp.sparse.linalg.eigs(LOPsite,k=6,M=Msite,which='SR',v0=np.reshape(self._mps[0],(D1*D2*d)),maxiter=nmax,tol=tolerance,ncv=ncv)
            #etasite,vecsite=sp.sparse.linalg.eigs(LOPsite,k=6,which='SR',v0=np.reshape(self._mps[0],(D1*D2*d)),maxiter=nmax,tol=tolerance,ncv=ncv)
            etabond,vecbond=sp.sparse.linalg.eigs(LOPbond,k=6,M=Mbond,which='SR',v0=np.reshape(np.eye(D1),(D1*D2)),maxiter=nmax,tol=tolerance,ncv=ncv)
            #etabond,vecbond=sp.sparse.linalg.eigs(LOPbond,k=6,which='SR',v0=np.reshape(self._mps._mat,(D1*D2)),maxiter=nmax,tol=tolerance,ncv=ncv)
            indsite=np.nonzero(np.real(etasite)==min(np.real(etasite)))[0][0]
            indbond=np.nonzero(np.real(etabond)==min(np.real(etabond)))[0][0]
            
            mps=np.reshape(vecsite[:,indsite],(D1,D2,d))
            mat=np.reshape(vecbond[:,indbond],(D1,D2))
            ACC_l=np.reshape(ncon.ncon([mps,herm(mat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
            Ul,Sl,Vl=mf.svd(ACC_l)

            self._mps[0]=mps#np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
            print(mps.shape)
            self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(self._N))))            
            self._mps._mat=np.eye(D1)
            self._mps._connector=np.eye(D1)
            
            #eta1,U1=np.linalg.eig(Hsite)
            #eta2,U2=np.linalg.eig(Hbond)            
            #print(np.sort(etasite))
            #print('the dimension of kernel of Nsite: ',D1**2*d-np.linalg.matrix_rank(Nsite))
            #print('the dimension of kernel of Nbond: ',D1**2-np.linalg.matrix_rank(Nbond))
            
            #print()
            #print('lowest eigenvalue of Hsite from sparse: ',np.sort(etasite)[0])
            #print('lowest eigenvalue of Hsite from dense:  ',np.sort(eta1)[0])            
            #print(np.sort(eta1))
            #print()
            #print('lowest eigenvalue of Hbond from sparse: ',np.sort(etabond)[0])
            #print('lowest eigenvalue of Hbond from dense:  ',np.sort(eta2)[0])                        

            #print(np.sort(eta2))
            print('etasite:',np.sort(etasite)[0]/self._N)
            print('etabond:',np.sort(etabond)[0]/self._N)
            print('energy ={0}, Z={1}, energy/N*Z={2}'.format(energy,Z,energy/Z/self._N))
            input()
            #print(self._mps[0])
            #print(self._mps._mat)                



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
        
