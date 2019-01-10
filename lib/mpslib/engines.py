"""
@author: Martin Ganahl
"""
from __future__ import absolute_import, division, print_function
from distutils.version import StrictVersion
from sys import stdout
import pickle,warnings
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


class Container(object):

    def __init__(self,mps,mpo,filename,lb=None,rb=None):
        """
        Base class for simulation objects;
        mps:      MPS object
                  the initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        filename: str
                  the name of the simulation
        lb:       np.ndarray of shape (D,D,M), or None
                  the left environment; 
                  lb has to have shape (mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0])
                  if None, obc are assumed, and lb=np.ones((mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0]))
        rb:       np.ndarray of shape (D,D,M), or None
                  the right environment
                  rb has to have shape (mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1])
                  if None, obc are assumed, and rb=np.ones((mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1]))
        """
        self._filename=filename        
        self._mps=mps
        self._mpo=mpo
        if len(mps)!=len(mpo):
            raise ValueError('len(mps)!=len(mpo)')
        
        #TODO: make checks for verbose
        if (np.all(lb)!=None) and (np.all(rb)!=None):
            assert(mps[0].shape[0]==lb.shape[0])
            assert(mps[-1].shape[1]==rb.shape[0])
            assert(mpo[0].shape[0]==lb.shape[2])
            assert(mpo[-1].shape[1]==rb.shape[2])
            #if not np.issubdtype(lb.dtype.type,self.dtype):
            #    raise TypeError('Container.__init__: lb.dtype={0} is not a subdtype of self.dtype={1}'.format(lb.dtype.type,self.dtype))
            #if not np.issubdtype(rb.dtype,self.dtype):
            #    raise TypeError('Container.__init__: rb.dtype={0} is not a subdtype of self.dtype={1}'.format(rb.dtype,self.dtype))
            self._lb=lb
            self._rb=rb
        else:
            self._lb=np.ones((mps[0].shape[0],mps[0].shape[0],mpo[0].shape[0]),dtype=self.dtype)
            self._rb=np.ones((mps[-1].shape[1],mps[-1].shape[1],mpo[-1].shape[1]),dtype=self.dtype)
        
    def __len__(self):
        """
        return the length of the MPS/MPO
        """
        return len(self._mps)
    
    @property
    def dtype(self):
        """
        return the data-type of the Container
        type is obtained from applying np.result_type 
        to the mps and mpo objects
        """
        return np.result_type(self.mps.dtype,self.mpo.dtype).type        
        
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

            
    @classmethod
    def read(self,filename):
        """
        reads a simulation from a pickle file "filename".pickle
        and returns a container object
        Parameters:
        -----------------------------------
        filename: str
                  the filename of the object to be loaded
        """
        with open(filename, 'rb') as f:
            out=pickle.load(f)
        return out

    
    def load(self,filename):
        """
        Container.load(filename):
        unpickles a Container object from a file "filename".pickle
        and stores the result in self
        """
        with open(filename, 'rb') as f:
            cls=pickle.load(f)
        #delete all attributes of self which are not present in cls
        todelete=[attr for attr in vars(self) if not hasattr(cls,attr)]
        for attr in todelete:
            delattr(self,attr)
            
        for attr in cls.__dict__.keys():
            setattr(self,attr,getattr(cls,attr))

            
    @property
    def mps(self):
        """
        Container.mps:
        return the underlying MPS object
        """
        return self._mps

    @property
    def mpo(self):
        """
        Container.mps:
        return the underlying MPS object
        """
        return self._mpo

    
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
        The left and right Hamiltonian bulk-environments are recalculated using self._mps, 
        with self._L (left environments) being a list of length len(mps)+1 with the only non-empty 
        element being the first one: self._L[0]=self._lb, where self._lb is unchanged. 
        self._lb is the left bundary condition of the simulation
        self._R is a list of length len(mps)+1 holding all right-environment for all bipartitions
        of the chain (again with self._R[0]=self._rb unchanged).

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
    def __init__(self,mps,mpo,filename='DMRG',lb=None,rb=None):
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
        super(DMRGengine,self).__init__(mps,mpo,filename,lb,rb)
        self._mps.position(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,np.copy(self._lb))
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,np.copy(self._rb))

        
    def optimize(self,n,tol=1E-6,ncv=40,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,verbose=0):
        if solver=='LOBPCG':
            e,opt=mf.lobpcg(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),tol)#mps._mat set to 11 during call of __tensor__()
        elif solver=='AR':
            e,opt=mf.eigsh(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),tol,numvecs,ncv)#mps._mat set to 11 during call of __tensor__()
        elif solver=='LAN':
            e,opt=mf.lanczos(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
                              deltaEta=landeltaEta)
        else:
            raise ValueError("DMRGengine.optimize: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))
        Dnew=opt.shape[1]
        if verbose>0:
            stdout.write("\rSS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,self._it,self._Nmax,n,len(self),np.real(e),np.imag(e),Dnew))
            stdout.flush()
        if verbose>1:
            print("")
            
            #print ('at iteration {2} optimization at site {0} returned E={1}'.format(n,e,it))
        return e,opt
    
                
    def simulate(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        DMRGengine.simulate(Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
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
            raise ValueError("DMRGengine.simulate: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))

        solver=solver
        self._Nmax=Nmax
        converged=False
        energy=100000.0
        self._it=1

        while not converged:
            for n in range(len(self._mps)-1):
                self.position(n)
                e,opt=self.optimize(n,tol,ncv,numvecs,solver,Ndiag,nmaxlan,landelta,landeltaEta,verbose)
                self.mps[n]=opt
            for n in range(len(self._mps)-1,0,-1):
                self.position(n+1)                
                e,opt=self.optimize(n,tol,ncv,numvecs,solver,Ndiag,nmaxlan,landelta,landeltaEta,verbose)                
                self.mps[n]=opt
                    
            if np.abs(e-energy)<Econv:
                converged=True
            energy=e
            if cp!=None and self._it>0 and self._it%cp==0:
                self.save(self._filename+'_dmrg_cp')                
            self._it=self._it+1
            if self._it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        self._mps.resetZ()
        return e

        
    def simulateTwoSite(self,Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
        """
        DMRGengine.simulateTwoSite(Nmax=4,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5):
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
            raise ValueError("DMRGengine.simulateTwoSite: unknown solver type {0}: use {'AR','LAN','LOBPCG'}".format(solver))
        solver=solver
        
        self._Nmax=Nmax

        converged=False
        energy=100000.0
        it=1
        self._mps.position(0)        
        while not converged:
            for n in range(0,len(self._mps)-2):
                self._mps.position(n+1)
                temp1=ncon.ncon([self._mps.tensor(n),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                if solver=='LOBPCG':
                    e,opt=mf.lobpcg(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol)#mps._mat set to 11 during call of __tensor__()
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
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
                    stdout.write("\rTS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,len(self),np.real(e),np.imag(e),Dnew))
                    stdout.flush()                    

                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)

            for n in range(len(self._mps)-2,-1,-1):
                self._mps.position(n+1)
                temp1=ncon.ncon([self._mps.tensor(n),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                if solver=='LOBPCG':
                    e,opt=mf.lobpcg(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol)#mps._mat set to 11 during call of __tensor__()
                if solver=='AR':
                    e,opt=mf.eigsh(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol,numvecs,ncv)
                if solver=='LAN':
                    e,opt=mf.lanczos(self._L[n],twositempo,self._R[len(self._mps)-1-n-1],twositemps,tol,Ndiag=Ndiag,nmax=nmaxlan,numeig=1,delta=landelta,\
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
                    stdout.write("\rTS-DMRG using %s solver: it=%i/%i, site=%i/%i: optimized E=%.16f+%.16f at D=%i"%(solver,it,Nmax,n,len(self),np.real(e),np.imag(e),Dnew))
                    stdout.flush()                    

                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                self._R[len(self._mps)-1-n]=mf.addLayer(self._R[len(self._mps)-1-n-1],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)

            self._mps.position(0)
            self._R[len(self._mps)-1-n]=mf.addLayer(self._R[len(self._mps)-1-n-1],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)
            if np.abs(e-energy)<Econv:
                converged=True
            energy=e
            if cp!=None and it>0 and it%cp==0:
                self.save(self._filename+'_2sdmrg_cp')                                
            it=it+1
            if it>Nmax:
                if verbose>0:
                    print()
                    print ('reached maximum iteration number ',Nmax)
                break
        self._mps.resetZ()            
        return e
        #returns the center bond matrix and the gs energy

class IDMRGengine(DMRGengine):
    """
    IDMRGengine
    container object for performing an IDMRG optimization of the ground-state of a Hamiltonian
    """
    def __init__(self,mps,mpo,filename='IDMRG'):
        """
        initialize an IDMRG object
        mps:      MPS object
                  initial mps
        mpo:      MPO object
                  Hamiltonian in MPO format
        filename: str
                  the name of the simulation
        """
        
        lb,rb,lbound,rbound,self._hl,self._hr=mf.getBoundaryHams(mps,mpo,numeig=1)                            
        super(IDMRGengine,self).__init__(mps,mpo,filename,lb,rb)
    #shifts the unit-cell by N/2 by updating self._L, self._R, self._lb, self._rb, and cutting and patching self._mps and self._mpo

    def update(self,regauge=False):
        """
        overrides self.update method of the Container class
        """
        self._mps.position(len(self._mps))
        #update the left boundary
        for site in range(int(len(self._mps)/2)):
            self._lb=mf.addLayer(self._lb,self._mps._tensors[site],self._mpo[site],self._mps._tensors[site],1)
            
        lamR=np.copy(self._mps._mat)
        mps=[]
        D=0
        #cut and patch the right half of the mps
        for n in range(int(len(self._mps)/2),len(self._mps)):
            if self._mps._tensors[n].shape[0]>D:
                D=self._mps._tensors[n].shape[0]
            mps.append(np.copy(self._mps._tensors[n]))

        self._mps.position(int(len(self._mps)/2))
        lamC=np.copy(self._mps._mat)
        connector=np.linalg.pinv(lamC)

        #update the right boundary
        for site in range(len(self._mps)-1,int(len(self._mps)/2)-1,-1):
            self._rb=mf.addLayer(self._rb,self._mps._tensors[site],self._mpo[site],self._mps._tensors[site],-1)

        self._mps.position(0)
        lamL=np.copy(self._mps._mat)
        #cut and patch the left half of the mps
        for n in range(int(len(self._mps)/2)):
            if self._mps._tensors[n].shape[0]>D:
                D=self._mps._tensors[n].shape[0]
            mps.append(np.copy(self._mps._tensors[n]))
        for n in range(len(self._mps)):
            self._mps._tensors[n]=mps[n]
            
        self._mps._position=int(len(self._mps)/2)
        self._mps._mat=lamR.dot(self._mps._connector).dot(lamL)
        self._mps._connector=connector
        self._mps.position(0)
        
        #cut and patch the mpo
        mf.patchmpo(self._mpo,int(len(self._mps)/2))
        if not regauge:
            self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
            self._L.insert(0,self._lb)
            self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
            self._R.insert(0,self._rb)
        elif regauge:
            lb,rb,lbound,rbound,self._hl,self._hr=mf.getBoundaryHams(self._mps,self._mpo,regauge=True)
            super(IDMRGengine,self).__init__(self._mps,self._mpo,self._filename,lb,rb)        
        return D

    
    def simulate(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0):            
        """
        IDMRGengine.simulate(Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0)
        run a single site IDMRG simulation

        Nmax: number of outer iterations
        NUC: number of optimization sweeps when optimizing a single unitcell
        Econv: desired convergence of energy per unitcell
        tol: arnoldi tolerance
        ncv: number of krylov vectors in arnoldi or lanczos
        cp: chekpoint step
        verbose: verbosity flag
        numvecs: the number of eigenvectors to be calculated; should be 1
        solver: type of eigensolver: 'AR' or 'LAN' or 'LOBPCG' for arnoldi or lanczos or conjugate gradient
        Ndiag: lanczos parameter; diagonalize tridiagonal Hamiltonian every Ndiag steps to check convergence;
        nmaxlan: maximum number of lanczos stesp
        landelta: lanczos stops if a krylov vector with norm < landelta is encountered
        landeltaEta: desired convergence of lanzcos eigenenergies
        regaugestep: if > 0, the mps is regauged into symmetric form when it%regaugestep==0; the effective Hamiltonians 
                     are recalculated in this case; do NOT use regaugestep==1
        """
        if regaugestep==1:
            raise ValueError("IDMRGengine.simulate(): regaugestep=1 can cause problems, use regaugestep>1 or regaugestep=0 (no regauging)")
        
        it=0
        converged=False
        eold=0.0
        verbose=1
        skip=False
        while not converged:
            regauge=False
            e=super(IDMRGengine,self).simulate(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,landelta=landelta,landeltaEta=landeltaEta)
            if regaugestep>0 and it%regaugestep==0 and it>0:
                regauge=True
                skip=True
            D=self.update(regauge)
            if verbose>0:
                if regauge==False:
                    if skip==False:
                        stdout.write("\rSS-IDMRG using %s solver: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(solver,it,Nmax,np.real((e-eold)/(len(self._mps))),np.imag((e-eold)/(len(self._mps))),D))
                    if skip==True:
                        skip=False
                if regauge==True:
                    stdout.write("\rSS-IDMRG using %s solver: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(solver,it,Nmax,np.real(self._hl/len(self)),np.imag(self._hl/len(self)),D))
                stdout.flush()  
                if verbose>1:
                    print('')
            #if regauge==True:
                #input('finished updating and regauing, starting fresh now')
                #print ('at iteration {0} optimization returned E/N={1}'.format(it,(e-eold)/(dmrg._mps._N)))
            if cp!=None and it>0 and it%cp==0:
                self.save(self._filename+'_idmrg_cp')                                                
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        self._mps.resetZ()            
        it=it+1

        
    def simulateTwoSite(self,Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0):
        """
        IDMRGengine.simulateTwoSite(Nmax=10,NUC=1,Econv=1E-6,tol=1E-6,ncv=40,cp=10,verbose=0,numvecs=1,truncation=1E-10,solver='AR',Ndiag=10,nmaxlan=500,landelta=1E-8,landeltaEta=1E-5,regaugestep=0)
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
            raise ValueError("IDMRGengine.simulateTwoSite(): regaugestep=1 can cause problems, use regaugestep>1 or regaugestep=0 (no regauging)")
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
            e=super(IDMRGengine,self).simulateTwoSite(Nmax=NUC,Econv=Econv,tol=tol,ncv=ncv,cp=cp,verbose=verbose-1,numvecs=numvecs,truncation=truncation,solver=solver,Ndiag=Ndiag,nmaxlan=nmaxlan,\
                                          landelta=landelta,landeltaEta=landeltaEta)
            if regaugestep>0 and it%regaugestep==0 and it>0:
                regauge=True
                skip=True
            D=self.update(regauge)
            if verbose>0:
                if regauge==False:
                    if skip==False:
                        stdout.write("\rSS-IDMRG: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real((e-eold)/(len(self._mps))),np.imag((e-eold)/(len(self._mps))),D))
                    if skip==True:
                        skip=False
                if regauge==True:
                    stdout.write("\rSS-IDMRG: rit=%i/%i, energy per unit-cell E/N=%.16f+%.16f at D=%i"%(it,Nmax,np.real(self._hl/len(self)),np.imag(self._hl/len(self)),D))
                stdout.flush()  
                if verbose>1:
                    print('')
            #if regauge==True:
                #input('finished updating and regauing, starting fresh now')
                #print ('at iteration {0} optimization returned E/N={1}'.format(it,(e-eold)/(dmrg._mps._N)))
            if cp!=None and it>0 and it%cp==0:
                self.save(self._filename+'_2sidmrg_cp')                                                                
            eold=e
            it=it+1
            if it>Nmax:
                converged=True
                break
        it=it+1
        self._mps.resetZ()
        
        
class HomogeneousIMPSengine(Container):
    """
    HomogeneousIMPSengine
    container object for homogeneous MPS optimization using a gradient descent method
    """
    def __init__(self,Nmax,mps,mpo,filename,alpha,alphas,normgrads,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-10,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40,pinv=1E-100,trunc=1E-16):
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
        super(HomogeneousIMPSengine,self).__init__(mps,mpo,filename)
        self._Nmax=Nmax
        #self._mps=np.copy(mps)
        self._D=np.shape(mps)[0]
        self._d=np.shape(mps)[2]
        #self._filename=filename
        self._dtype=mps.dtype.type
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

        #overwrite Container's _mpo attribute:
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
        self._gamma,self._lam,trunc=mf.regauge(self._mps,gauge='symmetric',initial=np.reshape(np.diag(self._lam**2),
                                                                                              self._D*self._D),
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
    
    def simulate(self,checkpoint=100):
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
        super(HomogeneousDiscretizedBosonEngine,self).__init__(Nmax,mps,mpo,filename,alpha,alphas,normgrads,dtype,factor=2.0,itreset=10,normtol=0.1,epsilon=1E-10,tol=1E-4,lgmrestol=1E-10,ncv=30,numeig=3,Nmaxlgmres=40)
        
    def simulate(self,dx,mu,checkpoint=100):
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
    container object for mps ground-state optimization  using the VUMPS algorithm and time evolution using the TDVP 
    """
    def __init__(self,mps,mpo,filename,regaugetol=1E-12,ncv=40,pinv=1E-200,numeig=6):
        """
        initialize a VUMPS simulation object

        Parameters:
        ---------------------------------------
        mps:            list of a single np.ndarray of shape(D,D,d), or an MPS object of length 1
                        the initial state
        mpo:            MPO object
                        Hamiltonian in MPO format
        filename:       str
                        the name of the simulation
        regaugetol: float  
                    desired accuracy when regauging
        ncv:        int
                    number of krylov vectors in eigs
        pinv:       float 
                    pseudoinverse cutoff
        numeig:     int
                    number of eigenvector-eigenvalue pairs to be calculated in eigs (hyperparameter)
        """
        if len(mps)>1:
            raise ValueError("VUMPSengine: got an mps of len(mps)>1; VUMPSengine can only handle len(mps)=1")
        if len(mpo)>1:
            raise ValueError("VUMPSengine: got an mpo of len(mps)>1; VUMPSengine can only handle len(mpo)=1")

        super(VUMPSengine,self).__init__(mps,mpo,filename,lb=None,rb=None)#initializes lb and rb
        self._kleft=np.random.rand(self.mps.D[0],self.mps.D[0]).astype(self.dtype)
        self._kright=np.random.rand(self.mps.D[-1],self.mps.D[-1]).astype(self.dtype)
        self._it=1
        [B1,B2,d1,d2]=np.shape(mpo[0])

        #initialize the mpo again
        mpol=np.zeros((1,B2,d1,d2),dtype=self.dtype)
        mpor=np.zeros((B1,1,d1,d2),dtype=self.dtype)
        mpol[0,:,:,:]=mpo[0][-1,:,:,:]
        mpol[0,0,:,:]/=2.0
        mpor[:,0,:,:]=mpo[0][:,0,:,:]
        mpor[-1,0,:,:]/=2.0
        self._mpo=H.MPO.fromlist([mpol,mpo[0],mpor])
        self._t0=self.dtype(0.0)
        self.mps.regauge('symmetric',tol=regaugetol,ncv=ncv,pinv=pinv,numeig=numeig)

        self._A=np.copy(self.mps[0])
        self._B=ncon.ncon([np.diag(1.0/np.diag(self.mps._mat)),self._A,self.mps._mat],[[-1,1],[1,2,-3],[2,-2]])
        self._r=np.eye(self._B.shape[0],dtype=self.dtype)
        self._l=np.eye(self._A.shape[0],dtype=self.dtype)
        #print([self._l.dtype,self._r.dtype,self._A.dtype,self._B.dtype]+[a.dtype for a in self.mps])

    def reset(self):
        """
        resets internal counters and density matrices of the VUMPS engine to self._it=1, self._t0=0.0
        and self._l=11, self._r=11
        """
        self._it=1
        self._t0=0.0
        self._r=np.eye(self._B.shape[0],dtype=self.dtype)
        self._l=np.eye(self._A.shape[0],dtype=self.dtype)        
        
    def _prepareStep(self,tol=1E-12,ncv=30,numeig=1,lgmrestol=1E-10,Nmaxlgmres=40):
        [etar,vr,numeig]=mf.TMeigs(self._A,direction=-1,numeig=numeig,init=self._r,nmax=10000,tolerance=tol,ncv=ncv,which='LR')
        [etal,vl,numeig]=mf.TMeigs(self._B,direction=1,numeig=numeig,init=self._l,nmax=10000,tolerance=tol,ncv=ncv,which='LR')
        l=np.reshape(vl,(self.mps.D[0],self.mps.D[0]))
        r=np.reshape(vr,(self.mps.D[-1],self.mps.D[-1]))
        l=l/np.trace(l)
        r=r/np.trace(r)
        
        self._l=(l+herm(l))/2.0
        self._r=(r+herm(r))/2.0            


        leftn=np.linalg.norm(np.tensordot(self._A,np.conj(self._A),([0,2],[0,2]))-np.eye(self.mps.D[0],dtype=self.dtype)
        rightn=np.linalg.norm(np.tensordot(self._B,np.conj(self._B),([1,2],[1,2]))-np.eye(self.mps.D[-1],dtype=self.dtype)

        self._lb=mf.initializeLayer(self._A,np.eye(self.mps.D[0],dtype=self.dtype),self._A,self._mpo[0],1)


        ihl=mf.addLayer(self._lb,self._A,self._mpo[2],self._A,1)[:,:,0]
        Elocleft=np.tensordot(ihl,self._r,([0,1],[0,1]))
        self._rb=mf.initializeLayer(self._B,np.eye(self.mps.D[-1],dtype=self.dtype),self._B,self._mpo[2],-1)


        ihr=mf.addLayer(self._rb,self._B,self._mpo[0],self._B,-1)[:,:,-1]
        Elocright=np.tensordot(ihr,self._l,([0,1],[0,1]))

        ihlprojected=(ihl-np.tensordot(ihl,l,([0,1],[0,1]))*np.eye(self.mps.D[0],dtype=self.dtype)
        ihrprojected=(ihr-np.tensordot(r,ihr,([0,1],[0,1]))*np.eye(self.mps.D[-1],dtype=self.dtype)
        
        self._kleft=mf.RENORMBLOCKHAMGMRES(self._A,self._A,self._l,np.eye(self.mps.D[0]).astype(self.dtype),ihlprojected,x0=np.reshape(self._kleft,self.mps.D[0]*self.mps.D[0]),tolerance=lgmrestol,\
                                           maxiteration=Nmaxlgmres,direction=1)
        self._kright=mf.RENORMBLOCKHAMGMRES(self._B,self._B,np.eye(self.mps.D[-1]).astype(self.dtype),self._r,ihrprojected,x0=np.reshape(self._kright,self.mps.D[-1]*self.mps.D[-1]),tolerance=lgmrestol,\
                                            maxiteration=Nmaxlgmres,direction=-1)
        
        self._lb[:,:,0]+=np.copy(self._kleft)
        self._rb[:,:,-1]+=np.copy(self._kright)
        return Elocleft,Elocright,leftn,rightn

    
    def _doOptimStep(self,svd=False,artol=1E-5,arnumvecs=1,arncv=20,Ndiag=10,nmaxlan=500,landelta=1E-8,solver='AR'):
        AC_=mf.HAproductSingleSiteMPS(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False))
        if self.mps._position==1:
            C_=mf.HAproductZeroSiteMat(self._lb,self._mpo[1],self._A,self._rb,position='right',mat=self.mps._mat)
            self._gradnorm=np.linalg.norm(AC_-ncon.ncon([self._A,C_],[[-1,1,-3],[1,-2]]))            
        if self.mps._position==0:
            C_=mf.HAproductZeroSiteMat(self._lb,self._mpo[1],self._B,self._rb,position='left',mat=self.mps._mat)
            self._gradnorm=np.linalg.norm(AC_-ncon.ncon([C_,self._B],[[-1,1],[1,-2,-3]]))            

        if solver.upper()=='AR':
            e1,mps=mf.eigsh(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),artol,numvecs=arnumvecs,numcv=arncv,numvecs_returned=arnumvecs)
            e2,mat=mf.eigshbond(self._lb,self._mpo[1],self._A,self._rb,self.mps._mat,position='right',tolerance=artol,numvecs=arnumvecs,numcv=arncv)
            if arnumvecs>1:
                self._gap=e1[1]-e1[0]
                mps=mps[0]                
            mat/=np.linalg.norm(mat)                

        elif solver.upper()=='LAN':
            e1,mps=mf.lanczos(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),artol,Ndiag,nmaxlan,arnumvecs,landelta,deltaEta=artol)
            e2,mat=mf.lanczosbond(self._lb,self._mpo[1],self._A,self._rb,self.mps._mat,'right',Ndiag,nmaxlan,arnumvecs,delta=landelta,deltaEta=artol)
            if arnumvecs>1:
                if len(e1)>1:
                    self._gap=e1[1]-e1[0]
                mat=mat[0]
                mps=mps[0]                
            mat/=np.linalg.norm(mat)                
        else:
            raise ValueError("in VUMPSengine: unknown solver type; use 'AR' or 'LAN'")
        D1,D2,d=mps.shape
        if svd:
            ACC_l=np.reshape(ncon.ncon([mps,herm(mat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
            CAC_r=np.reshape(ncon.ncon([herm(mat),mps],[[-1,1],[1,-2,-3]]),(D1,d*D2))
            Ul,Sl,Vl=mf.svd(ACC_l)
            Ur,Sr,Vr=mf.svd(CAC_r)
            self._A=np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
            self._B=np.reshape(Ur.dot(Vr),(D1,D2,d))

        else:
            AC_l=np.reshape(np.transpose(mps,(0,2,1)),(D1*d,D2))
            AC_r=np.reshape(mps,(D1,d*D2))
            
            UAC_l,PAC_l=sp.linalg.polar(AC_l,side='right')
            UAC_r,PAC_r=sp.linalg.polar(AC_r,side='left')
            
            UC_l,PC_l=sp.linalg.polar(mat,side='right')
            UC_r,PC_r=sp.linalg.polar(mat,side='left')
            
            self._A=np.transpose(np.reshape(UAC_l.dot(herm(UC_l)),(D1,d,D2)),(0,2,1))
            self._B=np.reshape(herm(UC_r).dot(UAC_r),(D1,D2,d))
            
        self.mps[0]=np.copy(self._A)
        self.mps._mat=np.copy(mat)
        self.mps._connector=np.linalg.pinv(mat)
        self.mps._position=1
        
    def simulate(self,*args,**kwargs):
        """
        see __simulate__

        """
        warnings.warn('VUMPS.simulate is deprecated; use optimize')
        self.optimize(*args,**kwargs)
        
    def optimize(self,Nmax=1000,epsilon=1E-10,regaugetol=1E-10,ncv=30,numeig=1,
                 lgmrestol=1E-10,Nmaxlgmres=40,\
                 artol=1E-10,arnumvecs=1,arncv=20,svd=False,Ndiag=10,nmaxlan=500,
                 landelta=1E-8,solver='AR',cp=None,keep_cp=False):

        """
        do a VUMPS ground-state optimization
        currently only nearest neighbor Hamiltonians are supported
        Parameters
        ---------------------------------------------
        Nmax:            int
                         number of iterations
        epsilon:         float
                         desired convergence
        regaugetol:      float
                         precision of the left and right dominant eigenvectors of the transfer operator
        ncv:             int
                         number of krylov vectors used for diagonalizeing transfer operator
        numeig:          int 
                         number of eigenvector-eigenvalues pairs calculated when diagonlizing transfer 
                         operator (hyperparameter)
        lgmrestol:       float
                         precision of the left and right renormalized environments
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
        cp:              int>0, or None
                         if > 0, simulation is checkpointed every "cp" steps

        Returns:
        -------------------------
        None
        """
        converged=False
        

        current='None'
        while converged==False:
            if self._it<10:
                artol_=1E-6
            else:
                artol_=artol
            Edens,Elocright,leftn,rightn=self._prepareStep(tol=regaugetol,ncv=ncv,numeig=numeig,
                                                           lgmrestol=lgmrestol,Nmaxlgmres=Nmaxlgmres)
            
            self._doOptimStep(svd=svd,artol=artol_,arnumvecs=arnumvecs,arncv=arncv,Ndiag=Ndiag,nmaxlan=nmaxlan,
                              landelta=landelta,solver=solver)
            if self._it>=Nmax:
                break
            if (cp!=None) and (cp>0) and (self._it%cp==0):
                if not keep_cp:
                    if os.path.exists(current+'.pickle'):
                        os.remove(current+'.pickle')
                    current=self._filename+'_vumps_cp'+str(self._it)
                    self.save(current)
                else:
                    current=self._filename+'_vumps_cp'+str(self._it)
                    self.save(current)
            self._it+=1
            if arnumvecs==1:
                stdout.write("\rusing %s solver: it %i: local E=%.16f, D=%i, gradient norm=%.16f" %(solver,self._it,np.real(Edens),np.max(self.mps.D),self._gradnorm))
                stdout.flush()
            if arnumvecs>1:
                stdout.write("\rusing %s solver: it %i: local E=%.16f, gap=%.16f, D=%i, gradient norm=%.16f" %(solver,self._it,np.real(Edens),np.real(self._gap),np.max(self.mps.D),self._gradnorm))
                stdout.flush()
            if self._gradnorm<epsilon:
                converged=True
        print
        print()

        if self._it>=Nmax and (converged==False):
            print ('simulation reached maximum number of steps ({1}) and stopped at precision of {0}'.format(self._gradnorm,Nmax))
        if converged==True:
            print ('simulation converged to {0} in {1} steps'.format(epsilon,self._it))
        print


    def _evolveTensor(self,solver,dt,krylov_dim,rtol,atol):
        """
        time-evolves the tensor at site n; 
        The caller has to ensure that self.L[n],self._R[len(self._mps)-1-n] are consistent with the mps
        n and self._mps._position have to match

        Parameters:
        -----------------------------------
        N:   int
             the lattice site
        solver: str, any from {'LAN','SEXPMV','Radau','RK45','RK23','BDF','LSODA','RK23'}
                the solver to do the time evolution
        dt:  float or complex:
             time step
        krylov_dim:  int
                     the number of krylov vectors to be used with solver='LAN'
        rtol,atol:  float
                    relative and absolute tolerance to be used with solver={'Radau','RK45','RK23','BDF','LSODA','RK23'}
        
        """
        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evTen=mf.evolveTensorsolve_ivp(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':
            evTen=mf.evolveTensorLan(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt,krylov_dimension=krylov_dim) #clear=True resets self._mat to identity
        elif solver=='SEXPMV':                    
            evTen=mf.evolveTensorSexpmv(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt)
        return evTen

    def _evolveMatrix(self,solver,dt,krylov_dim,rtol,atol):
        """
        time-evolves the center-matrix at bond n; 
        The caller has to ensure that self.L[n+1],self._R[len(self._mps)-1-n] are consistent with the mps
        n and self._mps._position have to match

        Parameters:
        -----------------------------------
        N:   int
             the lattice site
        solver: str, any from {'LAN','SEXPMV','Radau','RK45','RK23','BDF','LSODA','RK23'}
                the solver to do the time evolution
        dt:  float or complex:
             time step
        krylov_dim:  int
                     the number of krylov vectors to be used with solver='LAN'
        rtol,atol:  float
                    relative and absolute tolerance to be used with solver={'Radau','RK45','RK23','BDF','LSODA','RK23'}
        
        """
        
        L=mf.addLayer(self._lb,self._A,self._mpo[1],self._A,direction=1)        
        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evMat=mf.evolveMatrixsolve_ivp(L,self._rb,self.mps._mat,dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':                        
            evMat=mf. evolveMatrixLan(L,self._rb,self.mps._mat,dt,krylov_dimension=krylov_dim)
        elif solver=='SEXPMV':                                            
            evMat=mf.evolveMatrixSexpmv(L,self._rb,self.mps._mat,dt)                                            
        evMat/=np.linalg.norm(evMat)
        return evMat
            

    def _doEvoStep(self,solver,dt,krylov_dim,rtol,atol):
        """
        does a single time evolution step 
        Parameters:
        -----------------------------------
        solver: str, any from {'LAN','SEXPMV','Radau','RK45','RK23','BDF','LSODA','RK23'}
                the solver to do the time evolution
        dt:  float or complex:
             time step
        krylov_dim:  int
                     the number of krylov vectors to be used with solver='LAN'
        rtol,atol:  float
                    relative and absolute tolerance to be used with solver={'Radau','RK45','RK23','BDF','LSODA','RK23'}
        
        """


        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evTen=mf.evolveTensorsolve_ivp(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':
            evTen=mf.evolveTensorLan(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt,krylov_dimension=krylov_dim) #clear=True resets self._mat to identity
        elif solver=='SEXPMV':                    
            evTen=mf.evolveTensorSexpmv(self._lb,self._mpo[1],self._rb,self.mps.tensor(0,clear=False),dt)
     
        #evTen=self._evolveTensor(solver,dt,krylov_dim,rtol,atol)
        L=mf.addLayer(self._lb,self._A,self._mpo[1],self._A,direction=1)        
        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evMat=mf.evolveMatrixsolve_ivp(L,self._rb,self.mps._mat,dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':                        
            evMat=mf. evolveMatrixLan(L,self._rb,self.mps._mat,dt,krylov_dimension=krylov_dim)
        elif solver=='SEXPMV':                                            
            evMat=mf.evolveMatrixSexpmv(L,self._rb,self.mps._mat,dt)                                            
        evMat/=np.linalg.norm(evMat)
        
        D1,D2,d=evTen.shape        
        #evMat=self._evolveMatrix(solver,dt,krylov_dim,rtol,atol)
        ACC_l=np.reshape(ncon.ncon([evTen,herm(evMat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
        CAC_r=np.reshape(ncon.ncon([herm(evMat),evTen],[[-1,1],[1,-2,-3]]),(D1,d*D2))
        Ul,Sl,Vl=mf.svd(ACC_l)
        Ur,Sr,Vr=mf.svd(CAC_r)
        self._A=np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
        self._B=np.reshape(Ur.dot(Vr),(D1,D2,d))
        self.mps[0]=np.copy(self._B)
        self.mps._mat=np.copy(evMat)
        self.mps._connector=np.linalg.pinv(evMat)
        self.mps._position=0

    def doTDVP(self,dt,numsteps,solver='LAN',krylov_dim=10,rtol=1E-6,atol=1e-12,regaugetol=1E-10,ncv=40,numeig=1,lgmrestol=1E-10,Nmaxlgmres=40,cp=None,keep_cp=False,verbose=1):

        """
        !!!!!!!!!!!!!!!!!!!!!!         This has not yet been tested    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        to real or imaginary time evolution for an infinite homogeneous systems using the TDVP
        currently only nearest neighbor Hamiltonians are supported
        Parameters:
        -------------------------------
        dt:             float or complex:
                        time step for real or imaginary time evolution; dt has to either real of imaginary
        numsteps:       int
                        number of time steps
        solver:         str
                        type of solver to be used, can be either of {LAN, RK45, RK32, SEXPMV}
        krylov_dim:     int
                        number of krylov vectors to be used with solver=LAN
        atol,rtol:      float
                        absolute and relative tolerance of the RK45 and RK23 solvers
        regaugetol:     float
                        precision of the left and right dominant eigenvectors of the transfer operator
        ncv:            int
                        number of krylov vectors used for diagonalizeing transfer operator
        numeig:         int 
                        number of eigenvector-eigenvalues pairs calculated when diagonlizing transfer 
                        operator (hyperparameter)
        lgmrestol:      float
                        precision of lgmres used in determining left and right Hamiltonian environments 
        Nmaxlgmres:     int
                        maximum number of lgmres steps for determining for determining left and right Hamiltonian environments 
        cp:             int or None
                        if int>0, checkpoints are written every cp steps
        keep_cp:        bool
                        if True, intermediate checkpoints are kept on disk
        verbose:        int
                        verbosity flag; larger value produces more output, use 0 for no output
        """

        if solver not in ['LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23']:
            raise ValueError("VUMPSengine.doTDVP(): unknown solver type {0}; use {'LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23'}".format(solver))

        if solver in ['Radau','RK45','BDF','LSODA','RK23']:
            if StrictVersion(sp.__version__)<StrictVersion('1.1.0'):
                warnings.warn('{0} solver is only available for scipy versions >= 1.1.0. Switching to LAN time evolution'.format(solver),stacklevel=2)
                solver='LAN'

        current='None'
        while self._it <=numsteps:
            Edens,Elocright,leftn,rightn=self._prepareStep(tol=regaugetol,ncv=ncv,numeig=numeig,
                                                           lgmrestol=lgmrestol,Nmaxlgmres=Nmaxlgmres)
            self._doEvoStep(solver,dt,krylov_dim,rtol,atol)
            if verbose>=1:
                self._t0+=np.abs(dt)
                stdout.write("\rTDVP using %s solver: it/Nmax=%i/%i: t/T= %1.6f/%1.6flocal E=%.16f, D=%i, |dt|=%1.5f" %(solver,self._it,numsteps,self._t0,np.abs(dt)*numsteps,np.real(Edens),max(self.mps.D),np.abs(dt)))

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

            self._it+=1
        self.mps.position(0)
        self.mps.resetZ()        
        return self._t0


        
    
    
class TimeEvolutionEngine(Container):
    """
    TimeEvolutionEngine(Container):
    container object for performing real/imaginary time evolution using TEBD or TDVP algorithm for finite systems 
    """

    @classmethod
    def TEBD(cls,mps,gatecontainer,filename='TEBD'):
        """
        TimeEvolutionEngine.TEBD(mps,mpo,filename):
        initialize a TEBD simulation for finite systems. this is an engine for real or imaginary time evolution using TEBD
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

        return cls(mps,gatecontainer,filename)

    @classmethod
    def TDVP(cls,mps,mpo,filename='TDVP',lb=None,rb=None):
        """
        TimeEvolutionEngine.TDVP(mps,mpo,filename):
        initialize a TDVP simulation for finite systems; this is an engine for real or imaginary time evolution using TDVP
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
        initialize a TDVP or TEBD  simulation for finite systems; this is an engine for real or imaginary time evolution
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

        super(TimeEvolutionEngine,self).__init__(mps,mpo,filename,lb,rb)        
        self.gates=self.mpo
        self._mps.position(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)
        self._t0=0.0
        self._it=0
        self._tw=0
        
    @property
    def iteration(self):
        """
        return the current value of the iteration counter
        """
        return self._it
    
    @property
    def time(self):
        """
        return the current time self._t0 of the simulation
        """
        return self._t0
    @property
    def truncatedWeight(self):
        """
        returns the accumulated truncated weight of the simulation (if accessible)
        """
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
        self._mps.position(0)
        self._L=mf.getL(self._mps._tensors,self._mpo,self._lb)
        self._L.insert(0,self._lb)
        self._R=mf.getR(self._mps._tensors,self._mpo,self._rb)
        self._R.insert(0,self._rb)

    def applyEven(self,tau,Dmax,tr_thresh):
        """
        apply the TEBD gates on all even sites
        Parameters:
        ------------------------------------------
        tau:       float
                   the time-stepsize
        Dmax:      int
                   The maximally allowed bond dimension after the gate application (overrides tr_thresh)
        tr_tresh:  float
                   threshold for truncation
        """
        for n in range(0,len(self._mps)-1,2):
            tw_,D=self._mps.applyTwoSiteGate(gate=self.gates.twoSiteGate(n,n+1,tau),site=n,Dmax=Dmax,thresh=tr_thresh)
            self._tw+=tw_

    def applyOdd(self,tau,Dmax,tr_thresh):
        """
        apply the TEBD gates on all odd sites
        Parameters:
        ------------------------------------------
        tau:       float
                   the time-stepsize
        Dmax:      int
                   The maximally allowed bond dimension after the gate application (overrides tr_thresh)
        tr_tresh:  float
                   threshold for truncation
        """
        
        if len(self._mps)%2==0:
            lstart=len(self._mps)-3
        elif len(self._mps)%2==1:
            lstart=len(self._mps)-2
        for n in range(lstart,-1,-2):
            tw_,D=self._mps.applyTwoSiteGate(gate=self.gates.twoSiteGate(n,n+1,tau),site=n,Dmax=Dmax,thresh=tr_thresh)
            self._tw+=tw_

    def doTEBD(self,dt,numsteps,Dmax,tr_thresh=1E-10,verbose=1,cp=None,keep_cp=False):
        """
        TEBDengine.doTEBD(self,dt,numsteps,Dmax,tr_thresh,verbose=1,cnterset=0,tw=0,cp=None):
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

        Returns:
        a tuple containing the truncated weight and the simulated time
        """

        #even half-step:
        current='None'
        self.applyEven(dt/2.0,Dmax,tr_thresh)
        for step in range(numsteps):
            #odd step updates:
            self.applyOdd(dt,Dmax,tr_thresh)
            if verbose>=1:
                self._t0+=np.abs(dt)
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



    def _evolveTensor(self,n,solver,dt,krylov_dim,rtol,atol):
        """
        time-evolves the tensor at site n; 
        The caller has to ensure that self.L[n],self._R[len(self._mps)-1-n] are consistent with the mps
        n and self._mps._position have to match

        Parameters:
        -----------------------------------
        N:   int
             the lattice site
        solver: str, any from {'LAN','SEXPMV','Radau','RK45','RK23','BDF','LSODA','RK23'}
                the solver to do the time evolution
        dt:  float or complex:
             time step
        krylov_dim:  int
                     the number of krylov vectors to be used with solver='LAN'
        rtol,atol:  float
                    relative and absolute tolerance to be used with solver={'Radau','RK45','RK23','BDF','LSODA','RK23'}
        
        """
        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evTen=mf.evolveTensorsolve_ivp(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':
            evTen=mf.evolveTensorLan(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),dt,krylov_dimension=krylov_dim) #clear=True resets self._mat to identity
        elif solver=='SEXPMV':                    
            evTen=mf.evolveTensorSexpmv(self._L[n],self._mpo[n],self._R[len(self._mps)-1-n],self._mps.tensor(n,clear=True),dt)
        return evTen

    def _evolveMatrix(self,n,solver,dt,krylov_dim,rtol,atol):
        """
        time-evolves the center-matrix at bond n; 
        The caller has to ensure that self.L[n+1],self._R[len(self._mps)-1-n] are consistent with the mps
        n and self._mps._position have to match

        Parameters:
        -----------------------------------
        N:   int
             the lattice site
        solver: str, any from {'LAN','SEXPMV','Radau','RK45','RK23','BDF','LSODA','RK23'}
                the solver to do the time evolution
        dt:  float or complex:
             time step
        krylov_dim:  int
                     the number of krylov vectors to be used with solver='LAN'
        rtol,atol:  float
                    relative and absolute tolerance to be used with solver={'Radau','RK45','RK23','BDF','LSODA','RK23'}
        
        """
        
        if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
            evMat=mf.evolveMatrixsolve_ivp(self._L[n+1],self._R[len(self._mps)-1-n],self._mps._mat,dt,method=solver,rtol=rtol,atol=atol) #clear=True resets self._mat to identity
        elif solver=='LAN':                        
            evMat=mf. evolveMatrixLan(self._L[n+1],self._R[len(self._mps)-1-n],self._mps._mat,dt,krylov_dimension=krylov_dim)
        elif solver=='SEXPMV':                                            
            evMat=mf.evolveMatrixSexpmv(self._L[n+1],self._R[len(self._mps)-1-n],self._mps._mat,dt)                                            
        evMat/=np.linalg.norm(evMat)
        return evMat

    
    
    def doTDVP(self,dt,numsteps,krylov_dim=10,cp=None,keep_cp=False,verbose=1,solver='LAN',rtol=1E-6,atol=1e-12):
        """
        do real or imaginary time evolution for finite systems using single-site TDVP 
        Parameters:
        ----------------------------------------
        dt:         complex or float: 
                    step size
        numsteps:   int
                    number of steps to be performed
        krylov_dim: int 
                    dimension of the krylov space used to perform evolution with solver='LAN' (see below)
        cp:         int or None
                    if int>0, do checkpointing every cp steps
        keep_cp:    bool
                    if True, keep all checkpointed files, if False, only keep the last one
        verbose:    int
                    verbosity flag
        solver:     str, can be any of {'LAN','RK45,'Radau','SEXPMV','LSODA','BDF'}
                    different intergration schemes; note that only RK45, RK23, LAN and SEXPMV work for complex arguments
        rtol/atol:  float
                    relative and absolute precision of the RK45 and RK23 solver
        Returns:
        the simulated time
        
        """
        
        if solver not in ['LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23']:
            raise ValueError("TDVPengine.doTDVP(): unknown solver type {0}; use {'LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23'}".format(solver))

        if solver in ['Radau','RK45','BDF','LSODA','RK23']:
            if StrictVersion(sp.__version__)<StrictVersion('1.1.0'):
                warnings.warn('{0} solver is only available for scipy versions >= 1.1.0. Switching to LAN time evolution'.format(solver),stacklevel=2)
                solver='LAN'

        self._solver=solver
        converged=False
        current='None'
        self._mps.position(0)
        for step in range(numsteps):
            for n in range(len(self._mps)):
                if n==len(self._mps)-1:
                    dt_=dt
                else:
                    dt_=dt/2.0
                self._mps.position(n+1)
                #evolve tensor forward
                evTen=self._evolveTensor(n,solver=solver,dt=dt_,krylov_dim=krylov_dim,rtol=rtol,atol=atol)                
                tensor,mat,Z=mf.prepareTensor(evTen,1)
                self._mps[n]=tensor
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)
                self._mps._mat=mat
                #evolve matrix backward                    
                if n<(len(self._mps)-1):
                    evMat=self._evolveMatrix(n,solver=solver,dt=-dt_,krylov_dim=krylov_dim,rtol=rtol,atol=atol)                                    
                    self._mps._mat=evMat
                else:
                    self._mps._mat=mat
                    
            for n in range(len(self._mps)-2,-1,-1):
                dt_=dt/2.0
                #evolve matrix backward; note that in the previous loop the last matrix has not been evolved yet; we'll rectify this now
                self._mps.position(n+1)
                self._R[len(self._mps)-n-1]=mf.addLayer(self._R[len(self._mps)-n-2],self._mps[n+1],self._mpo[n+1],self._mps[n+1],-1)
                evMat=self._evolveMatrix(n,solver=solver,dt=-dt_,krylov_dim=krylov_dim,rtol=rtol,atol=atol)                                                    
                self._mps._mat=evMat        #set evolved matrix as new center-matrix
            
                #evolve tensor at bond n forward: the back-evolved center matrix is absorbed into the left-side tensor, and the product is evolved forward in time
                evTen=self._evolveTensor(n,solver=solver,dt=dt_,krylov_dim=krylov_dim,rtol=rtol,atol=atol)
                
                #split off a center matrix 
                tensor,mat,Z=mf.prepareTensor(evTen,-1) #mat is already normalized (happens in prepareTensor)
                self._mps[n]=tensor

                self._mps._mat=mat
                self._mps._position=n
            if verbose>=1:
                self._t0+=np.abs(dt)
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


    def doTwoSiteTDVP(self,dt,numsteps,Dmax,tr_thresh,krylov_dim=12,cp=None,keep_cp=False,verbose=1,solver='LAN',rtol=1E-6,atol=1e-12):
        """
        do real or imaginary time evolution for finite systems using a two-site TDVP 
        
        Parameters:
        --------------------------------------------
        dt:         complex or float: 
                    step size
        numsteps:   int
                    number of steps to be performed
        Dmax:       int
                    maximum bond dimension to be kept (overrides tr_thresh)
        tr_thresh:  treshold for truncation of Schmidt values
        krylov_dim: int 
                    dimension of the krylov space used to perform evolution with solver='LAN' (see below)
        cp:         int or None
                    if int>0, do checkpointing every cp steps
        keep_cp:    bool
                    if True, keep all checkpointed files, if False, only keep the last one
        verbose:    int
                    verbosity flag
        solver:     str, can be any of {'LAN','RK45,'Radau','SEXPMV','LSODA','BDF'}
                    different intergration schemes; note that only RK45, RK23, LAN and SEXPMV work for complex arguments
        rtol/atol:  float
                    relative and absolute precision of the RK45 and RK23 solver
        Returns:
        a tuple containing the truncated weight and the simulated time

        """
        
        if solver not in ['LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23']:
            raise ValueError("TDVPengine.doTwoSiteTDVP(): unknown solver type {0}; use {'LAN','Radau','SEXPMV','RK45','BDF','LSODA','RK23'}".format(solver))

        if solver in ['Radau','RK45','BDF','LSODA','RK23']:
            if StrictVersion(sp.__version__)<StrictVersion('1.1.0'):
                warnings.warn('{0} solver is only available for scipy versions >= 1.1.0. Switching to LAN time evolution'.format(solver),stacklevel=2)
                solver='LAN'

        solver=solver
        converged=False
        current='None'
        self._mps.position(0)
        self._mps._D=Dmax
        
        for step in range(numsteps):
            for n in range(len(self._mps)-1):
                if n==len(self._mps)-2:
                    dt_=dt
                else:
                    dt_=dt/2.0
                self._mps.position(n+1)
                #build the twosite mps
                temp1=ncon.ncon([self._mps.tensor(n,clear=True),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                    evTen=mf.evolveTensorsolve_ivp(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_,method=solver,rtol=rtol,atol=atol)
                elif solver=='LAN':
                    evTen=mf.evolveTensorLan(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_,krylov_dimension=krylov_dim) 
                elif solver=='SEXPMV':                    
                    evTen=mf.evolveTensorSexpmv(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_)

                temp3=np.reshape(np.transpose(np.reshape(evTen,(Dl,Dr,dl,dr)),(0,2,3,1)),(Dl*dl,dr*Dr))
                U,S,V=np.linalg.svd(temp3,full_matrices=False)
                self._tw+=np.sum(S[S<=tr_thresh]**2)
                S=S[S>tr_thresh]
                Dnew=len(S)
                Dnew=min(len(S),Dmax)
                self._tw+=np.sum(S[Dnew::]**2)                
                S=S[0:Dnew]
                S/=np.linalg.norm(S)
                U=U[:,0:Dnew]
                V=V[0:Dnew,:]
                
                #the local two-site tensors has now been evolved forward by a half-time-step.
                #the right-most twosite mps is evolved by a full time step.
                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                self._L[n+1]=mf.addLayer(self._L[n],self._mps[n],self._mpo[n],self._mps[n],1)

                if n<len(self._mps)-2:
                    if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                        evTen=mf.evolveTensorsolve_ivp(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_,method=solver,rtol=rtol,atol=atol)
                    elif solver=='LAN':
                        evTen=mf.evolveTensorLan(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_,krylov_dimension=krylov_dim)
                    elif solver=='SEXPMV':                    
                        evTen=mf.evolveTensorSexpmv(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_)
                
                    tensor,mat,Z=mf.prepareTensor(evTen,1)
                    self._mps[n+1]=tensor
                    self._mps._mat=mat
                    self._mps._position=n+2
                    
            for n in range(len(self._mps)-3,-1,-1):
                dt_=dt/2.0
                #evolve the right tensor at positoin n+1 backwards. Note that the tensor at N-1 has already been fully evolved forward at this point.
                self._mps.position(n+1)
                self._R[len(self._mps)-n-2]=mf.addLayer(self._R[len(self._mps)-3-n],self._mps[n+2],self._mpo[n+2],self._mps[n+2],-1)
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                    evTen=mf.evolveTensorsolve_ivp(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_,method=solver,rtol=rtol,atol=atol)
                elif solver=='LAN':
                    evTen=mf.evolveTensorLan(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_,krylov_dimension=krylov_dim)
                elif solver=='SEXPMV':                    
                    evTen=mf.evolveTensorSexpmv(self._L[n+1],self._mpo[n+1],self._R[len(self._mps)-2-n],self._mps.tensor(n+1,clear=True),-dt_)

                tensor,mat,Z=mf.prepareTensor(evTen,-1) 
                self._mps[n+1]=tensor
                self._mps._mat=mat

                #now evolve the two-site tensor forward
                #build the twosite mps
                temp1=ncon.ncon([self._mps.tensor(n,clear=True),self._mps[n+1]],[[-1,1,-2],[1,-4,-3]])
                Dl,dl,dr,Dr=temp1.shape
                twositemps=np.transpose(np.reshape(temp1,(Dl,dl*dr,Dr)),(0,2,1))                    
                temp2=ncon.ncon([self._mpo[n],self._mpo[n+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
                Ml,Mr,dlin,drin,dlout,drout=temp2.shape                
                twositempo=np.reshape(temp2,(Ml,Mr,dlin*drin,dlout*drout))
                
                if solver in ['Radau','RK45','RK23','BDF','LSODA','RK23']:
                    evTen=mf.evolveTensorsolve_ivp(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_,method=solver,rtol=rtol,atol=atol)
                elif solver=='LAN':
                    evTen=mf.evolveTensorLan(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_,krylov_dimension=krylov_dim) 
                elif solver=='SEXPMV':                    
                    evTen=mf.evolveTensorSexpmv(self._L[n],twositempo,self._R[len(self._mps)-2-n],twositemps,dt_)

                temp3=np.reshape(np.transpose(np.reshape(evTen,(Dl,Dr,dl,dr)),(0,2,3,1)),(Dl*dl,dr*Dr))
                U,S,V=np.linalg.svd(temp3,full_matrices=False)
                self._tw+=np.sum(S[S<=tr_thresh]**2)                
                S=S[S>tr_thresh]
                Dnew=len(S)
                Dnew=min(len(S),Dmax)
                self._tw+=np.sum(S[Dnew::]**2)
                S=S[0:Dnew]
                S/=np.linalg.norm(S)
                U=U[:,0:Dnew]
                V=V[0:Dnew,:]
                #the local two-site tensors has now been evolved forward by a half-time-step.
                #the right-most twosite mps is evolved by a full time step.
                self._mps[n]=np.transpose(np.reshape(U,(Dl,dl,Dnew)),(0,2,1))
                self._mps[n+1]=np.transpose(np.reshape(V,(Dnew,dr,Dr)),(0,2,1))
                self._mps._mat=np.diag(S)
                
            if verbose>=1:
                self._t0+=np.abs(dt)
                stdout.write("\rTwo-site TDVP engine using %s solver: t=%4.4f, truncated weight=%.16f at D/Dmax=%i/%i, truncation threshold=%1.16f, |dt|=%1.5f"%(solver,self._t0,self._tw,np.max(self.mps.D),Dmax,tr_thresh,np.abs(dt)))
                stdout.flush()
            if verbose>=2:                
                print('')
            if (cp!=None) and (self._it>0) and (self._it%cp==0):
                if not keep_cp:
                    if os.path.exists(current+'.pickle'):
                        os.remove(current+'.pickle')
                    current=self._filename+'_two_site_tdvp_cp'+str(self._it)
                    self.save(current)
                else:
                    current=self._filename+'_two_site_tdvp_cp'+str(self._it)
                    self.save(current)

            self._it=self._it+1
        self._mps.position(0)
        self._mps.resetZ()        
        return self._tw,self._t0    

    
class ITEBDengine(TimeEvolutionEngine):
    """
    a simulation engine for doing iTEBD for an infinite system
    with a two-site unitcell
    """
    def __init__(self,mps,mpo,filename='ITEBD'):
        if len(mps)!=2:
            raise ValueError('ITEBD: len(mps)!=2: only two-site unitcells are supported')
        if len(mpo)!=2:
            raise ValueError('ITEBD: len(mpo)!=2: only two-site unitcells are supported')
        if not mps.obc==False:
            raise ValueError('ITEBD: mpo.obc=True; use and mpo with mpo.obc=False')
        super(ITEBDengine,self).__init__(mps,mpo,filename,lb=None,rb=None)
        self._it=0
        self._time=0

    def applyGates(self,dt,Dmax,tr_thresh=1E-10):
        """
        uses a first order trotter decomposition to evolve the state using ITEBD
        Parameters:
        -------------------------------
        dt:        float
                   step size (scalar)
        Dmax:      int
                   maximum bond dimension to be kept
        tr_thresh: float
                   truncation threshold 
        Returns:
        -----------------------
        tw:  float 
             the truncated weight of the evolution step
        """
        tw=0.0
        self.mps.absorbConnector('left')
        tw_,D=self.mps.applyTwoSiteGate(gate=self.gates.twoSiteGate(0,1,dt,obc=False),site=0,Dmax=Dmax,thresh=tr_thresh)
        #obc has to be False here. The local part of the MPO must be divided by two, since there is no boundary
        tw+=tw_
        #swap the mps and mpo tensors
        self.mps.cutandpatch(1)
        self.mpo[0],self.mpo[1]=self.mpo[1],self.mpo[0]
        tw_,D=self.mps.applyTwoSiteGate(gate=self.gates.twoSiteGate(0,1,dt,obc=False),site=0,Dmax=Dmax,thresh=tr_thresh)
        tw+=tw_
        #swap back
        
        self.mps.cutandpatch(1)
        self.mpo[0],self.mpo[1]=self.mpo[1],self.mpo[0]
        return tw
    
    def doITEBD(self,dt,numsteps,Dmax,recanonizestep=1,regaugetol=1E-10,ncv=30,pinv=1E-200,numeig=1,
               tr_thresh=1E-10,verbose=1,cp=None,keep_cp=False):
        """
        uses a first order trotter decomposition to evolve the state using iTEBD
        Parameters:
        -------------------------------
        dt:             float
                        step size (scalar)
        numsteps:       int
                        total number of evolution steps
        Dmax:           int
                        maximum bond dimension to be kept
        recanonizestep: int
                        recanonize the mps every ```recanonizestep``` steps
        regaugetol: float  
                    desired accuracy when regauging
        ncv:        int
                    number of krylov vectors in eigs
        pinv:       float 
                    pseudoinverse cutoff
        numeig:     int
                    number of eigenvector-eigenvalue pairs to be calculated in eigs (hyperparameter)
        tr_thresh:      float
                        truncation threshold 
        verbose:        int
                        verbosity flag; put to 0 for no output
        cp:             int or None
                        checkpointing flag: checkpoint every cp steps
        keep_cp:        bool
                        if True, keep all checkpointed files, if False, only keep the last one

        Returns:
        ---------------------------------
        (E,tw,t0)
        E:    float or complex
              the current energy density, averaged over two unitcells
        tw:   float
              accumulated truncated weight
        t0:   float or complex
              total simulation time 
        """
        
        current='None'
        E=None
        for step in range(numsteps):
            #odd step updates:
            self._tw+=self.applyGates(dt=dt,Dmax=Dmax,tr_thresh=tr_thresh)
            if recanonizestep>0 and step%recanonizestep==0:            
                self.mps.canonize(nmaxit=1000,tol=regaugetol,ncv=ncv,pinv=pinv,numeig=numeig)
                mpsl=np.tensordot(np.diag(self.mps.lambdas[0]),self.mps.gammas[0],([1],[0]))
                mpsr=np.tensordot(np.diag(self.mps.lambdas[1]),self.mps.gammas[1],([1],[0]))                
                h=self.mpo.getTwoSiteHamiltonian(0,1,False)
                E1=ncon.ncon([mpsl,mpsr,np.conj(mpsl),np.conj(mpsr),
                              np.diag(self.mps.lambdas[2]**2.0),h],
                             [[1,6,3],[6,9,7],[1,4,2],[4,8,5],[8,9],[3,7,2,5]])
                E2=ncon.ncon([mpsr,mpsl,np.conj(mpsr),np.conj(mpsl),
                              np.diag(self.mps.lambdas[1]**2.0),h],
                             [[1,6,3],[6,9,7],[1,4,2],[4,8,5],[8,9],[3,7,2,5]])
                E=(E1+E2)/2.0
                
            if verbose>=1:
                self._t0+=np.abs(dt)
                stdout.write("\rITEBD engine: t=%4.4f truncated weight=%.16f at D/Dmax=%i/%i, truncation threshold=%1.16f, |dt|=%1.5f, Energy =%2.6f"%(self._t0,self._tw,np.max(self.mps.D),Dmax,tr_thresh,np.abs(dt),E))
                stdout.flush()
            if verbose>=2:
                print('')
            #if this is a cp step, save between two half-steps
            if (cp!=None) and (self._it>0) and (self._it%cp==0):
                
                #if the cp step does not coincide with the last step, do a half step, save, and do another half step
                if not keep_cp:
                    if os.path.exists(current+'.pickle'):
                        os.remove(current+'.pickle')
                    current=self._filename+'_itebd_cp'+str(self._it)
                    self.save(current)
                else:
                    current=self._filename+'_itebd_cp'+str(self._it)
                    self.save(current)
            self._it+=1
            self._mps.resetZ()
                      
        self.mps.canonize(nmaxit=1000,tol=regaugetol,ncv=ncv,pinv=pinv,numeig=numeig)
        mpsl=np.tensordot(np.diag(self.mps.lambdas[0]),self.mps.gammas[0],([1],[0]))
        mpsr=np.tensordot(np.diag(self.mps.lambdas[1]),self.mps.gammas[1],([1],[0]))                
        h=self.mpo.getTwoSiteHamiltonian(0,1,False)
        E1=ncon.ncon([mpsl,mpsr,np.conj(mpsl),np.conj(mpsr),
                      np.diag(self.mps.lambdas[2]**2.0),h],
                     [[1,6,3],[6,9,7],[1,4,2],[4,8,5],[8,9],[3,7,2,5]])
        E2=ncon.ncon([mpsr,mpsl,np.conj(mpsr),np.conj(mpsl),
                      np.diag(self.mps.lambdas[1]**2.0),h],
                     [[1,6,3],[6,9,7],[1,4,2],[4,8,5],[8,9],[3,7,2,5]])
        E=(E1+E2)/2.0
        stdout.write("\rITEBD finished at: t=%4.4f, truncated weight=%.16f at D/Dmax=%i/%i, truncation threshold=%1.16f, |dt|=%1.5f, Energy =%2.6f"%(self._t0,self._tw,np.max(self.mps.D),Dmax,tr_thresh,np.abs(dt),E))
            
        return E,self._tw,self._t0
        


        
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



class PUMPSengine(Container):
    def __init__(self,mps,mpo,filename):
        """
        initialize a PUMPS simulation object
        mps: a list of a single np.ndarray of shape(D,D,d), or an MPS object of length 1
        mpo: an MPO object
        filename: str
                  the name of the simulation
        """
        if len(mps)>1:
            raise ValueError("PUMPSengine: got an mps of len(mps)>1: can only handle len(mps)=1")
        if (mps.obc==True):
            raise ValueError("PUMPSengine: got an mps with obc=True")
        super(PUMPSengine,self).__init__(mps,mpo,filename,lb=None,rb=None)#initializes lb and rb                      \
                      
        [B1,B2,d1,d2]=self.mpo[0].shape
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
        while n<(len(self)-1):
            Hbla=ncon.ncon([Hbla,self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[-1,-2,-3,1,5,3],[-4,1,2],[-6,3,2,4],[-5,5,4]])
            #bla=np.zeros((D1,D1),dtype=Hbla.dtype)
            #for k in range(D1):
            #    bla+=Hbla[:,:,0,k,k,0]
            #print(bla)
            #input()
            n+=1            
            #if n==(len(self)-3):
            #    #N_temp contains all mps-tensors except those at 0,N-2 and N-1; will be contracted with local mpo to connect both ends
            #    N_temp=np.copy(Hbla[:,:,0,:,:,0])
            #    #print('length of N_temp: ',n,len(self))
            #print('contracted ',n,len(self))

        #Neffbond has all mps tensors contracted into it
        #Neffbond=ncon.ncon([Hbla,self._mps[0],self._mpo[1],np.conj(self._mps[0])],[[-1,-2,-3,1,5,3],[-4,1,2],[-6,3,2,4],[-5,5,4]])[:,:,0,:,:,0]
        Neffbond=self.empty_pow(len(self))
        N_temp=self.empty_pow(len(self)-3)
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
        #H2=self.checkH(len(self)-1)
        #print(np.linalg.norm(Henvsite-H2))
        #print(np.linalg.norm(Neffsite[:,:,0,:,:,0]-self.empty_pow(len(self)-1)))
        #print(np.linalg.norm(Neffbond-self.empty_pow(len(self))))
        #print(np.linalg.norm(N_temp-self.empty_pow(len(self)-3)))        
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
            edens=energy/Z/len(self)
            
            self._mps[0]-=np.copy(alpha*gradmps)
            self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(len(self)))))

            #mps=self._mps[0]-alpha*gradmps
            #mat=np.eye(D1)-alpha*gradmat
            #ACC_l=np.reshape(ncon.ncon([mps,herm(mat)],[[-1,1,-2],[1,-3]]),(D1*d,D2))
            #Ul,Sl,Vl=mf.svd(ACC_l)
            #self._mps[0]=np.transpose(np.reshape(Ul.dot(Vl),(D1,d,D2)),(0,2,1))
            #self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(len(self)))))
            
            self._mps._mat=np.eye(D1)
            self._mps._connector=np.eye(D1)
            if verbose>0:
                stdout.write("\rPeriodic MPS gradient optimization for N=%i sites: it=%i/%i, E=%.16f+%.16f at alpha=%1.5f, D=%i"%(len(self),it,Nmax,np.real(edens),np.imag(edens),alpha,D1))
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
            self._mps[0]=np.copy(self._mps[0]/(Z**(0.5/(len(self)))))            
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
            print('etasite:',np.sort(etasite)[0]/len(self))
            print('etabond:',np.sort(etabond)[0]/len(self))
            print('energy ={0}, Z={1}, energy/N*Z={2}'.format(energy,Z,energy/Z/len(self)))
            input()
            #print(self._mps[0])
            #print(self._mps._mat)                



"""

calculates excitation spectrum for a lattice MPS
basemps has to be left orthogonal


"""
class ExcitationEngine(object):
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
        
