#!/usr/bin/env python
import numpy as np
import scipy as sp
import copy

class LanczosEngine:
    """
    This is a general purpose Lanczos-class. It performs a Lanczos tridiagonalization 
    of a Hamiltonian, defined by the matrix-vector product matvec. 
    matvec: python function performing matrix-vector multiplication (e.g. np.dot)
    vecvec: python function performing vector-vector dot product (e.g. np.dot)
    zeros_initializer: python function which returns a vector filled with zeros  (e.g. np.zeros), has to accept a shape and dtype argument, i.e.
    zeros_initializer(shape,dtype); dtype should be either float or complex
    Ndiag: iteration step at which diagonalization of the tridiagonal Hamiltonian 
    should be done, e.g. Ndiag=4 means every 4 steps the tridiagonal Hamiltonian is diagonalized
    and it is checked if the eigenvalues are sufficiently converged (see deltaEta)
    ncv: maximum number of steps 
    numeig: number of eigenvalue-eigenvector pairs to be returned by the routine
    delta: tolerance parameter, such that iteration stops when a vector with norm<delta
    is encountered
    deltaEta: the desired eigenvalue-accuracy.
    
    RETURNS: eta,v,converged
    eta: list of eigenvalues, in ascending order
    v: a list of eigenvectors
    converged: bool; if True, lanczos did converge to desired accuracy within the specified number of iterations ncv.
    if False, it didn't converge
    """
    
    def __init__(self,matvec,vecvec,zeros_initializer,Ndiag,ncv,numeig,delta,deltaEta):
        assert(ncv>=numeig)

        self._Ndiag=Ndiag
        self._ncv=ncv
        self._numeig=numeig
        self._delta=delta
        self._deltaEta=deltaEta
        self._matvec=matvec
        self._dot=vecvec
        self._zeros=zeros_initializer
        assert(Ndiag>0)

        
    def __simulate__(self,initialstate,reortho=False,verbose=False):
        """
        do a lanczos simulation
        initialstate: that's the initial state
        reortho: if True, krylov vectors are reorthogonalized at each step (costly)
        the current implementation is not optimal: there are better ways to do this
        verbose: verbosity flag
        """
        dtype=np.result_type(self._matvec(initialstate))
        Dim=1
        for d in initialstate.shape:
            Dim*=d
        #initialization:
        xn=copy.deepcopy(initialstate)
        xn/=np.sqrt(self._dot(xn.conjugate(),xn))

        xn_minus_1=self._zeros(initialstate.shape,dtype=dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        self._vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            knval=np.sqrt(self._dot(xn.conjugate(),xn))

            if knval<self._delta:
                converged=True
                break
            kn.append(knval)
            xn=xn/kn[-1]
            #store the Lanczos vector for later

            if reortho==True:
                for v in self._vecs:
                    xn-=self._dot(v.conjugate(),xn)*v
            self._vecs.append(xn)                    
            Hxn=self._matvec(xn)                    
            epsn.append(self._dot(xn.conjugate(),Hxn))
            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=eta[0:self._numeig]
            if it>0:
                Hxn-=(self._vecs[-1]*epsn[-1])
                Hxn-=(self._vecs[-2]*kn[-1])
            else:
                Hxn-=(self._vecs[-1]*epsn[-1])
            xn=Hxn
            it=it+1
            if it>self._ncv:
                break

        self._Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(self._Heff)
        states=[]
        for n2 in range(min(self._numeig,len(eta))):
            state=np.zeros(initialstate.shape,dtype=initialstate.dtype)
            for n1 in range(len(self._vecs)):
                state+=self._vecs[n1]*u[n1,n2]
            states.append(state/np.sqrt(self._dot(state.conjugate(),state)))
        return eta[0:min(self._numeig,len(eta))],states,converged
                
class LanczosTimeEvolution(LanczosEngine):
    """
    Lanzcos time evolution engine
    LanczosTimeEvolution(matvec,vecvec,zeros_initializer,ncv,delta)
    matvec: python function performing matrix-vector multiplication (e.g. np.dot)
    vecvec: python function performing vector-vector dot product (e.g. np.dot)
    zeros_initializer: python function which returns a vector filled with zeros  (e.g. np.zeros), has to accept a shape and dtype argument, i.e.
    zeros_initializer(shape,dtype); dtype should be either float or complex
    ncv: number of krylov vectors used for time evolution; 10-20 usually works fine; larger ncv causes longer runtimes
    delta: tolerance parameter, such that iteration stops when a vector with norm<delta
    is encountered
    """
    def __init__(self,matvec,vecvec,zeros_initializer,ncv=10,delta=1E-10):
        super().__init__(matvec=matvec,vecvec=vecvec,zeros_initializer=zeros_initializer,Ndiag=ncv,ncv=ncv,numeig=ncv,delta=delta,deltaEta=1E-10)


    def __doStep__(self,state,dt,verbose=False):
        """
        do a time evolution step
        state: initial state
        dt: time increment
        verbose: verbosity flag
        """
        self._dtype=type(dt)        
        self.__simulate__(state.astype(self._dtype),verbose=True,reortho=True)
        #take the expm of self._Heff
        U=sp.linalg.expm(dt*self._Heff)
        result=np.zeros(state.shape,dtype=self._dtype)
        for n in range(min(self._ncv,self._Heff.shape[0])):
            result+=self._vecs[n]*U[n,0]
        return result
        
