#!/usr/bin/env python
import numpy as np
import scipy as sp
import lib.ncon as ncon
import copy

class LanczosEngine(object):
    """
    This is a general purpose Lanczos-class. It performs a Lanczos tridiagonalization 
    of a Hamiltonian, defined by the matrix-vector product matvec. 
    """
    
    def __init__(self,matvec,Ndiag,ncv,numeig,delta,deltaEta):
        assert(ncv>=numeig)

        self.Ndiag=Ndiag
        self.ncv=ncv
        self.numeig=numeig
        self.delta=delta
        self.deltaEta=deltaEta
        self.matvec=matvec
        assert(Ndiag>0)
        
    def simulate(self,initialstate,reortho=False,verbose=False):
        """
        do a lanczos simulation
        initialstate: that's the initial state
        reortho: if True, krylov vectors are reorthogonalized at each step (costly)
        the current implementation is not optimal: there are better ways to do this
        verbose: verbosity flag
        """
        dtype=np.result_type(self.matvec(initialstate).dtype)
        #initialization:
        xn=copy.deepcopy(initialstate)
        Z=np.sqrt(ncon.ncon([xn,xn.conj()],[range(len(xn.shape)),range(len(xn.shape))]))
        xn/=Z

        xn_minus_1=xn.zeros(initialstate.shape,dtype=dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        self.vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            knval=np.sqrt(ncon.ncon([xn,xn.conj()],[range(len(xn.shape)),range(len(xn.shape))]))            
            if knval<self.delta:
                converged=True
                break
            kn.append(knval)
            xn=xn/kn[-1]
            #store the Lanczos vector for later
            if reortho==True:
                for v in self.vecs:
                    xn-=ncon.ncon([v.conj(),xn],[range(len(v.shape)),range(len(xn.shape))])*v
            self.vecs.append(xn)                    
            Hxn=self.matvec(xn)
            epsn.append(ncon.ncon([xn.conj(),Hxn],[range(len(xn.shape)),range(len(Hxn.shape))]))
            if ((it>0) and (it%self.Ndiag)==0)&(len(epsn)>=self.numeig):
                #diagonalize the effective Hamiltonian
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self.numeig]-etaold[0:self.numeig])<self.deltaEta:
                        converged=True
                first=False
                etaold=eta[0:self.numeig]
            if it>0:
                Hxn-=(self.vecs[-1]*epsn[-1])
                Hxn-=(self.vecs[-2]*kn[-1])
            else:
                Hxn-=(self.vecs[-1]*epsn[-1])
            xn=Hxn
            it=it+1
            if it>self.ncv:
                break

        self.Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(self.Heff)
        states=[]
        for n2 in range(min(self.numeig,len(eta))):
            state=initialstate.zeros(initialstate.shape,dtype=initialstate.dtype)
            for n1 in range(len(self.vecs)):
                state+=self.vecs[n1]*u[n1,n2]
                
            states.append(state/np.sqrt(ncon.ncon([state.conj(),state],[range(len(state.shape)),range(len(state.shape))])))
        return eta[0:min(self.numeig,len(eta))],states,converged
                
class LanczosTimeEvolution(LanczosEngine,object):
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
        super(LanczosTimeEvolution,self).__init__(matvec=matvec,Ndiag=ncv,ncv=ncv,numeig=ncv,delta=delta,deltaEta=1E-10)


    def __doStep__(self,state,dt,verbose=False):
        """
        do a time evolution step
        state: initial state
        dt: time increment
        verbose: verbosity flag
        """
        self.dtype=type(dt)        
        self.simulate(state.astype(self.dtype),verbose=True,reortho=True)
        #take the expm of self.Heff
        U=sp.linalg.expm(dt*self.Heff)
        result=state.zeros(state.shape,dtype=self.dtype)
        for n in range(min(self.ncv,self.Heff.shape[0])):
            result+=self.vecs[n]*U[n,0]
        return result
        
