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

    def __init__(self, matvec, scalar_product, Ndiag, ncv, numeig, delta,
                 deltaEta):
        assert (ncv >= numeig)

        self.Ndiag = Ndiag
        self.ncv = ncv
        self.numeig = numeig
        self.delta = delta
        self.deltaEta = deltaEta
        self.matvec = matvec
        self.scalar_product = scalar_product
        if not Ndiag > 0:
            raise ValueError('LanczosEngine: Ndiag has to be > 0')

    def simulate(self, initialstate, zeros=None,reortho=False, verbose=False):
        """
        do a lanczos simulation
        Parameters:
        -----------------------
        initialstate: Tensor:
                      that's the initial state
        
        reortho:     bool
                     if True, krylov vectors are reorthogonalized at each step (costly)
                     the current implementation is not optimal: there are better ways to do this
        verbose:     int
                     verbosity flag
        Returns:
        -----------------------
        eta,v:
        eta:        float 
                    the gs energy
        v:          Tensor
                    the lowest eigenvector

        """

        dtype = np.result_type(self.matvec(initialstate).dtype)
        #initialization:
        xn = copy.deepcopy(initialstate)
        #Z=np.sqrt(ncon.ncon([xn,xn.conj()],[range(len(xn.shape)),range(len(xn.shape))]))
        Z = np.linalg.norm(xn)
        xn /= Z

        converged = False
        it = 0
        kn = []
        epsn = []
        self.vecs = []
        first = True
        while converged == False:
            #normalize the current vector:
            knval = np.sqrt(self.scalar_product(xn, xn))
            if knval < self.delta:
                converged = True
                break
            kn.append(knval)
            xn = xn / kn[-1]
            #store the Lanczos vector for later
            if reortho == True:
                for v in self.vecs:
                    xn -= self.scalar_product(v, xn) * v
            self.vecs.append(xn)
            Hxn = self.matvec(xn)
            epsn.append(self.scalar_product(xn, Hxn))

            if ((it > 0) and
                (it % self.Ndiag) == 0) & (len(epsn) >= self.numeig):
                #diagonalize the effective Hamiltonian
                Heff = np.diag(epsn) + np.diag(kn[1:], 1) + np.diag(
                    np.conj(kn[1:]), -1)
                eta, u = np.linalg.eigh(Heff)
                if first == False:
                    if np.linalg.norm(eta[0:self.numeig] -
                                      etaold[0:self.numeig]) < self.deltaEta:
                        converged = True
                first = False
                etaold = eta[0:self.numeig]
            if it > 0:
                Hxn -= (self.vecs[-1] * epsn[-1])
                Hxn -= (self.vecs[-2] * kn[-1])
            else:
                Hxn -= (self.vecs[-1] * epsn[-1])
            xn = Hxn
            it = it + 1
            if it >= self.ncv:
                break
            
        self.Heff = np.diag(epsn) + np.diag(kn[1:], 1) + np.diag(
            np.conj(kn[1:]), -1)
        eta, u = np.linalg.eigh(self.Heff)
        states = []
        for n2 in range(min(self.numeig, len(eta))):
            if zeros is None:
                state = initialstate.zeros(
                    initialstate.shape, dtype=initialstate.dtype)
            else:
                state=copy.deepcopy(zeros)
            for n1 in range(len(self.vecs)):
                state += self.vecs[n1] * u[n1, n2]
            states.append(state / np.sqrt(self.scalar_product(state, state)))
        return eta[0:min(self.numeig, len(eta))], states[0:min(self.numeig, len(eta))], it  #,epsn,kn


class LanczosTimeEvolution(LanczosEngine, object):
    def __init__(self, matvec, scalar_product, ncv=10, delta=1E-10):

        super().__init__(matvec=matvec,
                         scalar_product=scalar_product,
                         Ndiag=ncv,
                         ncv=ncv,
                         numeig=ncv,
                         delta=delta,
                         deltaEta=1E-10)


    def do_step(self, state, dt, zeros=None, verbose=False):
        """
        Lanzcos time evolution engine
        LanczosTimeEvolution(matvec,vecvec,zeros_initializer,ncv,delta)
        matvec: python function performing matrix-vector multiplication (e.g. np.dot)
        vecvec: python function performing vector-vector dot product (e.g. np.dot)
        ncv: number of krylov vectors used for time evolution; 10-20 usually works fine; larger ncv causes longer runtimes
        delta: tolerance parameter, such that iteration stops when a vector with norm<delta
        is encountered
        """
        self.dtype = type(dt)
        self.simulate(state.astype(self.dtype), zeros=zeros, verbose=True, reortho=True)
        #take the expm of self.Heff
        U = sp.linalg.expm(dt * self.Heff)
        if zeros is None:
            result = state.zeros(state.shape, dtype=self.dtype)
        else:
            result = zeros
        for n in range(min(self.ncv, self.Heff.shape[0])):
            result += self.vecs[n] * U[n, 0]
        return result
