"""
@author: Martin Ganahl
"""
from __future__ import absolute_import, division, print_function
from sys import stdout
import sys,pickle
import numpy as np
import os,copy
import time
import operator as opr
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
import lib.mpslib.Hamiltonians as H
try:
    mf=sys.modules['lib.mpslib.mpsfunctions']
except KeyError:
    import lib.mpslib.mpsfunctions as mf

from scipy.sparse.linalg import ArpackNoConvergence
import lib.ncon as ncon
import warnings
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

def generate_unary_deferer(op_func):
    def deferer(cls, *args, **kwargs):
        return type(cls).__unary_operations__(cls, op_func, *args,**kwargs)
    return deferer

    
"""
mpsfunctions.py:

prepareTensor
prepareTruncate
(prepareTensorSVD)
UnitcellTMeigs


numpy:

np.linalg.pinv
np.linalg.inv
np.random.rand
np.copy
np.eye
np.shape
np.ones
np.transpose
np.tensordot
np.reshape
np.real
np.imag
np.diag
np.trace
np.abs
np.linalg.eigh
np.linalg.eig
np.zeros
np.sqrt
np.conj
"""
class MPS(object):

    """
    MPS.__init__(tensors,schmidt_thresh=1E-16,r_thresh=1E-14):
    MPS object representing a matrix product state
    

    Generic Structure of the state: self._tensors contains the MPS matrices. self._position 
    is the bond index at which orthonormalization of self._tensor matrices switches from
    left- to right-orthonormal. self._mat contains the bond matrix (if diagonalized, the schmdit values) 
    of the cut at self._position. 

    members:
    self.__absorbCenterMatrix__(direction) merges self._mat with self._tensors[self._position] (direction>0) or self._tensors[self._position-1] (direction<0).
    and puts the resulting tensor at the corresponding position.
    
    self._connector contains a connector matrix such that the tensors can be connected back to themselves: self._tensors[-1]*self._mat*self._connector*self._tensors[0]
    (where * is a tensor contraction) is correctly gauge-matched. This is relevant for infinite system simulations

    if schmidt_thresh is set below 1E-15, some mps methods, like e.g. position, switch 
    to qr instead of svd orthogonalization

    The maximum bond dimension of the MPS is self._D. Methods will try to truncate the state such that this limit is respected
    
    for generating an mps, use the decorators random, zeros, ones, fromList or productState (see below)
    """
    def __init__(self,tensors,schmidt_thresh=1E-16,r_thresh=1E-14,Dmax=None):
        """
        initialize an unnormalized MPS from a list of tensors
        Parameters
        ----------------------------------------------------
        tensors: list of np.ndarrays of shape (D1,D2,d)
                 a list of mps tensors
        schmidt_thresh: float
                        truncation limit of the mps
        r_thresh: float 
                  internal parameter (ignore it)
        Dmax:     int or None
                  the maximally allowed bond dimension
                  if ```None```, the maximal bond dimension of ```tensors``` is used as Dmax
        """
        #self._N=len(tensors)
        if Dmax==None:
            self._D=max(list(map(lambda x: max(np.shape(x)),tensors)))
        else:
            self._D=Dmax
            
        if (tensors[0].shape[0]==1) and (tensors[-1].shape[1]==1):
            self._obc=True
        else:
            self._obc=False


        self._schmidt_thresh=schmidt_thresh
        self._r_thresh=r_thresh
        self._tensors=[np.copy(tensors[n]) for n in range(len(tensors))]
        self._d=[a.shape[2] for a in self]
        self._mat=np.eye(np.shape(self._tensors[-1])[1]).astype(self.dtype)
        self._mat=self._mat/np.sqrt(np.trace(self._mat.dot(herm(self._mat))))

        self._connector=np.diag(1.0/np.diag(self._mat))
        self._position=self.N
        self._Z=-1e10
        self.gammas=[]
        self.lambdas=[]

    @property
    def N(self):
        """
        length of the mps
        Returns:
        ----------------
        int: the length of the MPS
        """
        return len(self._tensors)
    
    @property
    def dtype(self):
        """
        Returns:
        ----------------
        np.dtype.type: the data type of the MPS tensors

        """
        return np.result_type(*self._tensors).type
    
    @property
    def obc(self):
        """
        Returns:
        ---------------------
        bool:   if True, state has obc
        """
        return self._obc
    
    @property
    def pos(self):
        """
        Returns:
        ----------------------------
        int: the current position of the center bond
        """
        return self._position
    
    @property
    def mat(self):
        """
        Returns:
        ----------------------
        np.ndarray of shape (MPS.D[MPS.pos],MPS.D[MPS.pos]): the center matrix of the MPS
        """
        return self._mat
    
    @property
    def connector(self):
        """
        np.ndarray of shape (MPS.D[0],MPS.D[0]):   the connector matrix 
        """
        return self._connector

    @classmethod
    def random(cls,N,D,d,obc,scaling=0.5,dtype=np.float64,shift=0.5,schmidt_thresh=1E-16,r_thresh=1E-14):
        """
        generate a random MPS
        Parameters:
        ----------------------------------------------
        N: int
           length of the mps
        D: int
           initial maximal bond dimension
        d: int or array of int 
           local Hilbert space dimension(s)
        obc: bool
             boundary condition; if True, open boundary conditions are used
        scaling: float
                 initial scaling of mps matrices
        dtype: np.dtype.type
               dtype of the mps tensors (e.g. np.float64)
        shift: float
               shift of interval of random number generation for initial mps
        schmidt_thresh: float 
                        truncation limit of the mps
        r_thresh: float
                  internal parameter (ignore it)

        """
        tensors=mf.MPSinitializer(np.random.random_sample,N,D,d,obc,dtype,scaling,shift)
        mps=cls(tensors,schmidt_thresh,r_thresh)

        mps._obc=obc
        mps.position(0)        
        mps.position(mps.N)
        mps.Z=1.0

        return mps

    @classmethod
    def zeros(cls,N,D,d,obc,dtype=np.dtype(float),schmidt_thresh=1E-16,r_thresh=1E-16):

        """
        generate an MPS filled with zeros
        Parameters:
        ----------------------------------------------
        N: int
           length of the mps
        D: int
           initial maximal bond dimension
        d: int or array of int 
           local Hilbert space dimension(s)
        obc: bool
             boundary condition; if True, open boundary conditions are used
        dtype: np.dtype.type
               dtype of the mps tensors (e.g. np.float64)

        schmidt_thresh: float 
                        truncation limit of the mps
        r_thresh: float
                  internal parameter (ignore it)
        Returns:
        ----------------------------------
        MPS:      an MPS object with tensors initialized with zeros

        """

        tensors=mf.MPSinitializer(np.zeros,N,D,d,obc,dtype,scale=1.0,shift=0.0)
        mps=cls(tensors,schmidt_thresh,r_thresh)
        mps._obc=obc
        mps.position(0)        
        mps.position(mps.N)
        mps._Z=1.0        
        return mps

    @classmethod
    def ones(cls,N,D,d,obc,dtype=np.dtype(float),schmidt_thresh=1E-16,r_thresh=1E-16):
        """
        MPS.ones(N,D,d,obc,dtype=float,schmidt_thresh=1E-16,r_thresh=1E-14):
        generate an MPS filled with ones
        Parameters:
        ----------------------------------------------
        N: int
           length of the mps
        D: int
           initial maximal bond dimension
        d: int or array of int 
           local Hilbert space dimension(s)
        obc: bool
             boundary condition; if True, open boundary conditions are used
        dtype: np.dtype.type
               dtype of the mps tensors (e.g. np.float64)
        schmidt_thresh: float 
                        truncation limit of the mps
        r_thresh: float
                  internal parameter (ignore it)

        Returns:
        ----------------------------------
        MPS:      an MPS object with tensors initialized with ones
        """
        
        tensors=mf.MPSinitializer(np.ones,N,D,d,obc,dtype,scale=1.0,shift=0.0)
        mps=cls(tensors,schmidt_thresh,r_thresh)
        mps._obc=obc
        mps.position(0)        
        mps.position(mps.N)
        mps._Z=1.0                
        return mps


    @classmethod
    def fromList(cls,tensors,mat=None,schmidt_thresh=1E-16,r_thresh=1E-16):
        """
        MPS.fromList(tensors,mat=None,schmidt_thresh=1E-16,r_thresh=1E-14):
        generate an MPS from a list of np.ndarrays

        Parameters:
        ----------------------------
        tensors: list of np.ndarrays: 
                 mps tensors
        mat: np.ndarray of shape (D1,D2) 
             the center matrix of the mps; if None, it is assumed to be 11
        schmidt_thresh: float
                        truncation limit of the mps
        r_thresh: float 
                  internal parameter (ignore it)

        Returns:
        ----------------------------------
        MPS:      an MPS object with tensors initialized from ```tensors```
        """

        mps=cls(tensors,schmidt_thresh,r_thresh)
        if np.all(mat)!=None:
            mps._mat=np.copy(mat)

        mps.position(0)        
        mps.position(mps.N)
        mps._Z=1.0                
        return mps
        
    
    @classmethod
    def productState(cls,localstate,obc,dtype=np.float64,schmidt_thresh=1E-16,r_thresh=1E-16):
        """
        Parameters:
        --------------------------------------------
        initialize a product state MPS:

        localstate: list of np.ndarray of float
                    sets the local state at position n to sum_k localstate[n][k]|k>
        obc:        bool
                    boundary condition of the mps
        dtype: np.dtype.type
               dtype of the mps tensors (e.g. np.float64)
        schmidt_thresh: float 
                        truncation limit of the mps
        r_thresh: float 
                  internal parameter (ignore it)

        Returns:
        ----------------------------------
        MPS:      an MPS object initialized in a product state as given by ```localstate```

        """
        dtype=np.result_type(dtype,*localstate).type
        d=np.asarray([len(s) for s in localstate])
        if any([np.linalg.norm(t)<1E-10 for t in localstate]):
            inds=np.nonzero(np.asarray([np.linalg.norm(t)<1E-10 for t in localstate]))
            raise ValueError("in MPS.productState(localstate,obc,dtype=np.dtype(float),schmidt_thresh=1E-16,r_thresh=1E-16): found a local state of norm <1E-10 at positions {0}".format(inds))
        
        tensors=mf.MPSinitializer(np.zeros,len(localstate),D=1,d=d,obc=obc,dtype=dtype,scale=1.0,shift=0.0)
        for n in range(len(tensors)):
            for s in range(d[n]):
                tensors[n][0,0,s]=localstate[n][s]
        
        mps=cls(tensors,schmidt_thresh,r_thresh)
        mps._obc=obc
        mps._Z=1.0
        return mps
    
    def copy(self):
        """
        Returns: 
        -----------------------
        MPS: copy of the MPS object
        """
        return self.__copy__()
    
    def __copy__(self):
        """
        Returns: 
        -----------------------
        MPS: copy of the MPS object
        """
        cop=MPS(self._tensors)
        cop._D=self._D
        cop._tensors=copy.deepcopy(self._tensors)
        cop._mat=copy.deepcopy(self._mat)
        cop._connector=copy.deepcopy(self._connector)
        cop._position=self._position
        cop.gammas=copy.deepcopy(self.gammas)
        cop.lambdas=copy.deepcopy(self.lambdas)
        cop._obc-self._obc
        cop._schmidt_thresh=self._schmidt_thresh
        cop._r_thresh=self._r_thresh
        cop._Z=self._Z
        return cop

    def __str__(self):
        """
        printing function
        """
        b=['\n\n connector \n\n '+self._connector.__str__()]+['\n\n ']+['mps['+str(n)+']'+' \n\n '+self._tensors[n].__str__()+' \n\n ' for n in range(self.pos)]+\
            ['center matrix \n\n' + str(self._mat) +'\n\n ']+['mps['+str(n)+']'+' \n\n '+self._tensors[n].__str__()+' \n\n ' for n in range(self.pos,len(self))]
        return ''.join(b)
        
    
    def __getitem__(self,n):
        if isinstance(n,int):
            assert((n<len(self)) and (abs(n)<=len(self)))
            return self._tensors[n]

    def __setitem__(self,n,tensor):
        assert((n<len(self)) and (abs(n)<=len(self)))        
        self._tensors[n]=np.copy(tensor)


    def __in_place_unary_operations__(self,operation,*args,**kwargs):
        """
        implements unary operations on mps-tensors in-place 
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        None
        """
        for n in range(len(self)):
            self[n]=operation(self[n],*args,**kwargs)


    def __unary_operations__(self,operation,*args,**kwargs):
        """
        implements unary operations on mps-tensors
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        MPS:  MPS object obtained from acting with operation on each individual MPS tensor
        """
        res=self.__copy__()
        for n in range(len(res)):
            res[n]=operation(res[n],*args,**kwargs)
        return res

    def conj(self,*args,**kwargs):
        """
        in-place complex conjugation of the MPS
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.conj

        Returns:
        ------------------------
        None
        """
        self.__in_place_unary_operations__(np.conj,*args,**kwargs)
    

    def __in_place_unary_operations__(self,operation,*args,**kwargs):
        """
        implements in-place unary operations on TensorNetwork tensors
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        None
        """
        for n in range(len(res)):
            self[n]=operation(self[n],*args,**kwargs)


    def __unary_operations__(self,operation,*args,**kwargs):
        """
        implements unary operations on TensorNetwork tensors
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        MPS:  MPS object obtained from acting with operation on each individual MPS tensor
        """
        res=copy.deepcopy(self)
        for n in range(len(res)):
            res[n]=operation(res[n],*args,**kwargs)
        return res

    def __conj__(self,*args,**kwargs):
        """
        in-place complex conjugation of the TensorNetwork
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.conj

        Returns:
        ------------------------
        None
        """
        self.__in_place_unary_operations__(np.conj,*args,**kwargs)
    
    def __conjugate__(self,*args,**kwargs):
        """
        complex conjugation of TensorNetwork
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.conj

        Returns:
        ------------------------
        MPS:       complex-conjugated MPS object
        """
        
        res=self.__unary_operations__(np.conj,*args,**kwargs)
        return res

    def conjugate(self,*args,**kwargs):
        """
        complex conjugation of MPS
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.conj

        Returns:
        ------------------------
        MPS:       complex-conjugated MPS object
        """
        return self.__conjugate__(self,*args,**kwargs)

    def conj(self,*args,**kwargs):
        """
        in-place complex conjugation of the MPS
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.conj

        Returns:
        ------------------------
        None
        """
        return self.__conj__(self,*args,**kwargs)

    def __exp__(self,*args,**kwargs):
        """
        element-wise exponential of the mps tensors
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.exp

        Returns:
        ------------------------
        None
        """
        res=self.__unary_operations__(np.exp,*args,**kwargs)
        return res
    
    def exp(self,*args,**kwargs):
        """
        element-wise exponential of the mps tensors
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.exp

        Returns:
        ------------------------
        None
        """
        
        return self.__exp__(*args,**kwargs)
       
    def __sqrt__(self,*args,**kwargs):
        """
        element-wise sqareroot of the mps tensors
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.sqrt

        Returns:
        ------------------------
        None
        """
                
        res=self.__unary_operations__(np.sqrt,*args,**kwargs)
        return res
    def sqrt(self,*args,**kwargs):
        """
        element-wise sqareroot of the mps tensors
        Parameters:
        ------------------
        *args,**kwargs:   arguments to np.sqrt

        Returns:
        ------------------------
        None
        """
        
        return self.__sqrt__(*args,**kwargs)

    def __real__(self,*args,**kwargs):
        """
        return a new MPS with the real part of self
        """
        
        res=self.__unary_operations__(np.real,*args,**kwargs)
        res._dtype=np.dtype(float)
        return res

    def real(self,*args,**kwargs):
        """
        return a new MPS with the real part of self
        """
        
        return self.__real__(*args,**kwargs)

    def __imag_(self,*args,**kwargs):
        """
        return a new MPS with the imaginary part of self
        """
        
        res=self.__unary_operations__(np.imag,*args,**kwargs)
        res._dtype=np.dtype(float)
        return res
    
    def imag(self,*args,**kwargs):
        """
        return a new MPS with the imaginary part of self
        """
        return self.__imag__(*args,**kwargs)
    



    __neg__ = generate_unary_deferer(opr.neg)
    __pos__ = generate_unary_deferer(opr.pos)
    __abs__ = generate_unary_deferer(abs)
    __invert__ = generate_unary_deferer(opr.invert)
    
    def __mul__(self,num):
        """
        left-multiplies "num" with MPS, i.e. returns MPS*num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        Parameters:
        -----------------------
        num: float or complex
             to be multiplied into the MPS
        Returns:
        ---------------
        MPS:    the state obtained from multiplying ```num``` into MPS
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        cpy=self.__copy__()
        cpy._Z*=num
        cpy._dtype=np.result_type(type(num),self.dtype).type
        return cpy

    def __imul__(self,num):
        """
        left-multiplies "num" with MPS, i.e. returns MPS*num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        self._Z*=num
        return self
    

    def __idiv__(self,num):
        """
        left-divides "num" with MPS, i.e. returns MPS*num;
        note that "1./num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        self._Z/=num
        return self
    
    def __truediv__(self,num):
        """
        left-divides "num" with MPS, i.e. returns MPS/num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        cpy=self.__copy__()
        cpy._Z/=num
        cpy._dtype=np.result_type(type(num),self.dtype).type
        return cpy
    
    
    def __rmul__(self,num):
        """
        right-multiplies "num" with MPS, i.e. returns num*MPS;
        WARNING: if you are using numpy number types, i.e. np.float, np.int, ..., 
        the right multiplication of num with MPS, i.e. num*MPS, returns 
        an np.darray instead of an MPS. 
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state

        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__rmul__(self,num): num is not a number")
        cpy=self.__copy__()
        cpy._Z*=num        
        cpy._dtype=np.result_type(type(num),self.dtype).type
        return cpy

    def __add__(self,other):
        """
        adds self with other;
        returns an unnormalized mps
        """
        tens=mf.addMPS(self.tolist(),self._Z,other.tolist(),other._Z)
        out=MPS(tensors=tens,schmidt_thresh=1E-16,r_thresh=1E-16) #out is an unnormalized MPS
        out._Z=1.0 #MPS.__init__ sets self._Z to -1e10, so we have to reset it here to 1
        return out

    def __iadd__(self,other):
        """
        adds self with other;
        leaves self unnormalized
        """
        self._tensors=mf.addMPS(self.tolist(),self._Z,other.tolist(),other._Z)
        self._Z=1.0 
        return self
    
    def __sub__(self,other):
        """
        subtracts other from  self
        returns an unnormalized mps
        WARNING: use __sub__ with caution; for example, the result of mps-mps depends on the postprocessing;
        doing (mps-mps).position(0) or (mps-mps).position(N) will give back a normalized mps, and NOT
        0. The best practice is to avoid __sub__ whenever possible. If you need to use __sub__, be sure
        that any following operation doesn't use position or anything alike;
        """
        
        return (other*(-1))+self
        

    def connect(self):
        """
        merges self._mat and self._connector into the mps, such that it can be connected back to itself.
        changes self.position to self.N prior to merging, then merges np.dot(self._mat,self._connector) 
        into the rightmost mps tensor
        
        """
        self.position(self.N)
        self._tensors[self.N-1]=ncon.ncon([self._tensors[self.N-1],self.mat.dot(self.connector)],[[-1,1,-3],[1,-2]])
        self._mat=np.eye(self.D[-1],dtype=self.dtype)
        self._connector=np.eye(self.D[-1],dtype=self.dtype)
        
    def absorbConnector(self,location):
        """
        merges self._connector into the MPS. self._connector is merged into either the 'left' most or 
        the 'right' most mps-tensor. changes self._connector to be the identity: self._connector=11
        """


        if location in ('right','r',-1):
            if self.pos==self.N:
                warnings.warn("MPS.absorbConnector(location='right'): MPS.pos==MPS.N-1; cannot absorb on the right side; switching to location='left'")
                self.absorbConnector('left')
            else:
                self._tensors[self.N-1]=np.transpose(np.tensordot(self._tensors[self.N-1],self._connector,([1],[0])),(0,2,1))
                D=np.shape(self._tensors[self.N-1])[1]
                self._connector=np.eye(D,dtype=self.dtype)


        elif location in ('left','l',1):
            if self.pos==0:
                warnings.warn("MPS.absorbConnector(location='right'): MPS.pos==0; cannot absorb on the left side; switching to location='right'")
                self.absorbConnector('right')
            else:
                self._tensors[0]=np.tensordot(self._connector,self._tensors[0],([1],[0]))
                D=np.shape(self._tensors[0])[0]
                self._connector=np.eye(D,dtype=self.dtype)
        else:
            raise ValueError('mps.absorbConnector(self,location): wrong value for "location"; use "location"="left","l",1 or "right","r",-1')
        return self
                
    def absorbCenterMatrix(self,direction=1):
        """
        merges self._mat into the MPS. self._mat is merged into either the left (direction in (-1,'r','right)) or 
        the right (direction in (1,'l','left')) tensor at bond self._position
        changes self._mat to be the identity: self._mat=11; does not change self._connector
        """
        assert(direction!=0)
        if (self._position==self.N) and (direction in (1,'l','left')):
            direction=-1
            warnings.warn('mps.absorbCenterMatrix(self,direction): self._position=N and direction>0; cannot contract bond-matrix to the right because there is no right tensor')
        elif (self._position==0) and (direction<0):
            direction=1                
            warnings.warn('mps.absorbCenterMatrix(self,direction): self._position=0 and direction<0; cannot contract bond-matrix to the left tensor because there is no left tensor')

        if (direction in (1,'l','left')):
            self._tensors[self._position]=np.tensordot(self._mat,self._tensors[self._position],([1],[0]))
            D=self._mat.shape[0]
            self._mat=np.eye(D,dtype=self.dtype)
        elif (direction in (-1,'r','right')):
            self._tensors[self._position-1]=np.transpose(np.tensordot(self._tensors[self._position-1],self._mat,([1],[0])),(0,2,1))
            D=self._mat.shape[1]
            self._mat=np.eye(D,dtype=self.dtype)
        return self
    
    def __len__(self):
        """
        return the number of tensors in the MPS
        """
        return len(self._tensors)
    
    def __iter__(self):
        """
        return an iterator over the MPS-tensors
        """
        
        return iter(self._tensors)

    def __dot__(self,mps):
        """
        calculate the overlap of self with mps 
        mps: MPS
        returns: float
                 overlap of self with mps
        """
        
        return mf.overlap(self,mps)
    
    def dot(self,mps):
        """
        calculate the overlap of self with mps 
        mps: MPS
        returns: float
                 overlap of self with mps
        """
        return mf.overlap(self,mps)

    @property
    def D(self):
        """
        returns a list containig the bond-dimensions of the MPS
        """
        return list(map(lambda x: np.shape(x)[0],self._tensors))+[self._tensors[-1].shape[1]]

    
    @property
    def d(self):
        """
        returns a list containig the bond-dimensions of the MPS
        """
        return [t.shape[2] for t in self._tensors]
    
    @property
    def Dmax(self):
        """
        Dmax is the maximally allowed bond dimension of the MPS
        """
        return self._D
    @Dmax.setter
    def Dmax(self,D):
        """
        set Dmax to D
        """
        self._D=D


    @property
    def Z(self):
        """
        Returns:
        ----------------
        the norm of the state
        """
        return self._Z
    
    @Z.setter
    def Z(self,val):
        """
        set the norm of the state to ```val```
        """

        self._Z=self.dtype(val)
    
    
    def SchmidtSpectrum(self,n):
        """
        SchmidtSpectrum(n):

        return the Schmidt-values on bond n:
        Parameters:
        ---------------------------------
        n: int
           the bond number

        Returns:
        -------------------
        S: np.ndarray of shape (D[n],)
           the D[n] Schmidt values
        """
        self.position(n)
        U,S,V=mf.svd(self._mat)
        return S

        
    def position(self,bond,schmidt_thresh=1E-16,D=None,r_thresh=1E-14):
        """
        position(bond,schmidt_thresh=1E-16,D=None,r_thresh=1E-14):
        shifts the center site to "bond"
        Parameters:
        ---------------------------------
        bond: int
              the bond onto which to put the center site
        schmidt_thresh: float
                        truncation threshold at which Schmidt-values are truncated during shifting the position
                        overrides the self._schmidt_thresh parameter of the MPS temporarily
                        if schmidt_thresh>1E-15, then routine uses an svd to truncate the mps, otherwise no truncation is done
        D: int
           maximum bond dimension after Schmidt-value truncation
           The function does not modify self._connector
        r_thresh: float
                  internal parameter, has no relevance 
        """
        assert(bond<=self.N)
        assert(bond>=0)
        """
        set the values for the schmidt_thresh-threshold, D-threshold and r_thresh-threshold
        r_thresh is used in case that svd throws an exception (happens sometimes in python 2.x)
        in this case, one preconditions the method by first doing a qr, and setting all values in r
        which are smaller than r_thresh to 0, then does an svd.
        """
            
        if schmidt_thresh>1E-15:
            _schmidt_thresh=schmidt_thresh
        else:
            _schmidt_thresh=self._schmidt_thresh
        if r_thresh>1E-14:
            _r_thresh=r_thresh
        else:
            _r_thresh=self._r_thresh
        if bond==self._position:
            return
        
        if bond>self._position:
            self._tensors[self._position]=np.tensordot(self._mat,self._tensors[self._position],([1],[0]))
            for n in range(self._position,bond):
                if _schmidt_thresh < 1E-15 and D==None:
                    tensor,self._mat,Z=mf.prepareTensor(self._tensors[n],direction=1)
                    
                else:
                    tensor,s,v,Z=mf.prepareTruncate(self._tensors[n],direction=1,D=D,thresh=_schmidt_thresh,\
                                                    r_thresh=_r_thresh)
                    self._mat=np.diag(s).dot(v)
                self._Z*=Z                    
                self._tensors[n]=np.copy(tensor)
                if (n+1)<bond:
                    self._tensors[n+1]=np.tensordot(self._mat,self._tensors[n+1],([1],[0]))

        if bond<self._position:
            self._tensors[self._position-1]=np.transpose(np.tensordot(self._tensors[self._position-1],self._mat,([1],[0])),(0,2,1))
            for n in range(self._position-1,bond-1,-1):
                if _schmidt_thresh < 1E-15 and D==None:
                    tensor,self._mat,Z=mf.prepareTensor(self._tensors[n],direction=-1)

                else:
                    u,s,tensor,Z=mf.prepareTruncate(self._tensors[n],direction=-1,D=D,thresh=_schmidt_thresh,\
                                                    r_thresh=_r_thresh)
                    self._mat=u.dot(np.diag(s))
                self._Z*=Z                                        
                self._tensors[n]=np.copy(tensor)
                if n>bond:
                    self._tensors[n-1]=np.transpose(np.tensordot(self._tensors[n-1],self._mat,([1],[0])),(0,2,1))
        self._position=bond
        #print("after position: self._D=",self._D)

    
    def measureList(self,ops,lb=None,rb=None):
        """
        measure a list of N local operators "ops", where N=len(ops)=len(self)
        the routine moves the centersite to the left boundary, measures and moves
        it back to its original position; this might cause some overhead

        Parameters:
        --------------------------------
        ops:           list of np.ndarrays of shape (d,d)
                       local operators to be measured
        lb,rb:         np.ndarrays, or None
                       left and right boundary conditions to be applied if the state is pbc;
                       for obc they can be left None
                       only obc case is currently implemented

        Returns:  np.ndarray containing the matrix elements
        """
        
        if lb==None or rb==None:
            if self._obc==False:
                return NotImplemented
            else:
                pos=self._position
                self.position(0)

                L=mf.measureLocal(self,ops,lb=np.ones((1,1,1),dtype=self.dtype),rb=np.ones((1,1,1),dtype=self.dtype),ortho='right')
                self.position(pos)
                return L*self._Z*np.conj(self._Z)


        
    def measureMatrixElementList(self,mps,ops,lb=None,rb=None):

        """
        measure the matrix elements <self|ops[n]|mps> for a list of N local operators "ops", where N=len(ops)=len(mps)=len(self)
        the routine moves the centersite to the left boundary, measures and moves
        it back to its original position; this might cause some overhead

        Parameters:
        --------------------------------
        mps:           an MPS object
        ops:           list of np.ndarrays of shape (d,d)
                       local operators to be measured
        lb,rb:         np.ndarrays, or None
                       left and right boundary conditions to be applied if the state is pbc;
                       for obc they can be left None
                       only obc case is currently implemented

        Returns:  np.ndarray containing the matrix elements
        
        """
        if lb!=None or rb!=None:
            raise NotImplementedError("measureMatrixElementList: does currenlty not support non-trivial boundary conditions")
        if lb==None or rb==None:
            if self._obc==False:
                return NotImplemented
            else:
                pos=self._position
                mpspos=mps._position
                self.position(0)
                mps.position(0)
                L=mf.matrixElementLocal(self,mps,ops,lb=np.ones((1,1,1),dtype=self.dtype),rb=np.ones((1,1,1),dtype=self.dtype))
                v1=np.copy(self._mat[0][0])
                v2=np.copy(mps._mat[0][0])
                self.position(pos)
                mps.position(mpspos)                
                return L*v1*np.conj(v2)


    def measureLocal(self,op,site,lb=None,rb=None):

        """
        measure a local operator "op" at site "site"
        "lb" and "rb" are left and right boundary conditions to be applied 
        if the state is pbc (currently not implemented); for obc they can be left None
        the routine moves the centersite to "site", measures and moves
        it back to its original position; this might cause some overhead
        """
        
        if lb==None or rb==None:
            if self._obc==False:
                return NotImplemented
            else:
                pos=self._position
                self.position(site+1)                
                t=self.__tensor__(site)                
                self.position(pos)
                #return np.tensordot(np.tensordot(t,np.conj(t),([0,1],[0,1])),([0,1],[0,1]))*self._Z*np.conj(self._Z)
                return ncon.ncon([t,np.conj(t),op],[[1,3,2],[1,3,4],[2,4]])*self._Z*np.conj(self._Z)
            

    def measure(self,ops,sites,P=None):
        
        """ 
        This is a quite slow function to calculate correlation functions; 
        it takes a list of operators "ops" and a list of sites "sites" of where the operators live, and calculates <ops[site[0]]ops[sites[1]],...,ops[sites[n-1]]>
        "sites" has to be a list of (weakly) monotonically increasing numbers 
        the function can also measure fermionic correlation functions, if the jordan wigner strings are given in P; P is a list 
        containing the jordan-woigner operators for each site of the MPS (len(P)==len(mps)); for example, P=np.diag(1,-1) for spinless fermions
        
        BE WEARY: ROUTINE COULD BE WRONG 
        """
        
        if not isinstance(sites,list):
            ops=list([ops])
            sites=list([sites])            
        if P!=None:
            assert(len(P)==len(self))
        
        if not sorted(sites)==sites:
            sys.exit('mps.py.measure(self,ops,sites): sites are not in increasing order')
        if not self._obc==True:
            warnings.warn('calculating observable for an infinite MPS; be sure that it is regauged correctly')
        s=0
        self.position(sites[-1]+1)
        #the parity of the operator string one wants to measure:
        par=len(ops)%2
        if par==0 or P==None:
            L=np.zeros((self[sites[s]].shape[0],self[sites[s]].shape[0],1)).astype(self.dtype)
            L[:,:,0]=np.eye(self[sites[s]].shape[0],dtype=self.dtype)
        elif par==1 and P!=None:
            #for fermionic modes one has to take into consideration the fermionic minus signs from the start (if len(ops) is odd)
            L=np.zeros((self[0].shape[0],self[0].shape[0],1)).astype(self.dtype)
            L[:,:,0]=np.eye(self[0].shape[0],dtype=self.dtype)
            for n in range(sites[s]-1):
                L=mf.addLayer(L,self[n],P[n],self[n],1)

        while s<len(sites):
            if s<len(sites):
                n=s
                op=np.eye(self[s].shape[2],dtype=self.dtype)
                while n<len(sites) and sites[n]==sites[s]:
                    op=op.dot(ops[n])
                    par=(par+1)%2
                    n+=1

            assert(sites[s]==sites[n-1])
            mpo=np.zeros((1,1,op.shape[0],op.shape[1])).astype(self.dtype)
            if P==None or par==0:
                mpo[0,0,:,:]=op
            elif P!=None and par==1:
                mpo[0,0,:,:]=op.dot(P[sites[s]])                

            L=mf.addLayer(L,self[sites[s]],mpo,self[sites[s]],1)
            if n<len(sites):
                for m in range(sites[s]+1,sites[n]):
                    if par==0 or P==None:
                        L=mf.addELayer(L,self[m],self[m],1)
                    elif par==1 and P!=None:
                        L=mf.addLayer(L,self[m],np.reshape(P[m],(1,1,P[m].shape[0],P[m].shape[1])),self[m],1)
            s=n

        assert(self._position==sites[-1]+1)

        R=np.zeros((self[sites[s-1]].shape[1],self[sites[s-1]].shape[1],1),dtype=self.dtype)
        R[:,:,0]=np.eye(self[sites[s-1]].shape[1],dtype=self.dtype)
        Z=np.trace(self._mat.dot(herm(self._mat)))
        if not((np.abs(np.trace(self._mat.dot(herm(self._mat))))-1.0)<1E-10):
            warnings.warn('mps.py.measure(self,ops,sites): state is not normalized')

        return ncon.ncon([L,self._mat,np.conj(self._mat),R],[[1,2,3],[1,4],[2,5],[4,5,3]])*self._Z*np.conj(self._Z)



    def truncate(self,schmidt_thresh,D=None,returned_gauge=None,nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-200,r_thresh=1E-14):
        """ 
        a dedicated routine for truncating an mps (for obc, this can also be done using self.position(self,pos))
        For the case of obc==True (infinite system with finite unit-cell), the function modifies self._connector
        schmidt_thresh: truncation threshold
        D: maximum bond dimension; if None, the bond dimension is adapted to match schmidt_thresh
        r_thresh: internal parameter, has no effect on the outcome and can be ignored
        returned_gauge: 'left','right' or 'symmetric': the desired gauge after truncation
        """
        
        if D!=None and D>self._D:
            print('MPS.truncate(): D>self._D, no truncation neccessary')
            return
        else:
            if self._obc==True:
                self.position(0)
                self.position(self.N)
                #print (schmidt_thresh,D)
                self.position(0,schmidt_thresh=schmidt_thresh,D=D,r_thresh=r_thresh)
            if self._obc==False:
                if self._position<self.N:
                    self.absorbCenterMatrix(direction=1)
                    self.absorbConnector(location='left') #self._connector is set to 11
                if self._position==self.N:
                    self.absorbCenterMatrix(direction=-1)
                    self.absorbConnector('right')

                self._mat=mf.regaugeIMPS(self._tensors,gauge='symmetric',ldens=None,rdens=None,truncate=schmidt_thresh,D=D,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=1E-8)
                self._connector=np.diag(1.0/np.diag(self._mat)) #note that regaugeIMPS returns in any case a diagonal matrix
                self._position=self.N

                if returned_gauge!=None:
                    self.regauge(returned_gauge)                    


    def tensor(self,site,clear=False):
        """
        returns the tensor at site="site" contracted with self._mat; self._position has to be either site or site+1
        if clear=True, self._mat is replaced with a normalized identity matrix
        """
        assert((site==self._position) or (site==(self._position-1)))
        if (site==self._position):
            out=np.tensordot(self._mat,self._tensors[site],([1],[0]))
            if clear==True:
                self._mat=np.eye(np.shape(self._tensors[site])[0],dtype=self.dtype)
                self._mat/=np.linalg.norm(self._mat)                                
            return out

        if (site==(self._position-1)):
            out=np.transpose(np.tensordot(self._tensors[site],self._mat,([1],[0])),(0,2,1))            
            if clear==True:
                self._mat=np.eye(np.shape(self._tensors[site])[1],dtype=self.dtype)
                self._mat/=np.linalg.norm(self._mat)                
            return out
        
        
    def canonize(self,nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-200,numeig=6):
        """
        canonizes the mps, i.e. brings it into Gamma,Lambda form; Gamma and Lambda are stored in
        mps.gammas and mps.lambdas member lists;
        len(mps.lambdas) is len(mps)+1, i.e. there are boundary lambdas to the left and right of the mps; 
        for obc, these are just [1.0]
        The mps is left in a left orthogonal state

        Parameters:
        ------------------------------
        nmaxit:      int
                     maximum iteration number of sparse solver
        tol:         float
                     desired precision of eigenvalues/eigenvectors returned by sparse solver
        ncv:         int
                     number of krylov vectors used in sparse sovler
        pinv:        float
                     pseudo inverse cutoff
        numeig:      number of eigenvalue-eigenvector pairs to be calculated in sparse solver (hyperparameter)

        Returns:
        -------------------------
        None
        """
        if self._obc==False:
            self.regauge(gauge='symmetric',nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,numeig=numeig)
        self.gammas,self.lambdas=mf.canonizeMPS(self)
        

    def regauge(self,gauge,nmaxit=100000,tol=1E-12,ncv=40,pinv=1E-200,numeig=6):
        """
        regauge brings state in either left, right or symmetric orthonormal form
        the state should be a finite unitcell state on an infinite lattice
        gauge = 'left' or 'right' bring the state into left or right orthonormal form such that it can be connected back 
        to itself. For gauge='left' or 'right', self._mat=11 and self._connector=11 are set to be the identity. To calculate 
        any observables, one needs to additionally calculate the right (for gauge = 'left') or the left (for gauge = 'right') reduced
        density matrices
        if gauge='symmetric' the state is brought into symmetric gauge, such that it is totally left normalized, 
        and self._mat=lambda, self._connector=1/lambda contain the schmidt-values. note that regauge uses self._schmidt_thresh
        as effective truncation, so the bond dimensions might change
        """
            
        if self._obc==True:
            print ('in MPS.regauge(self,gauge,nmaxit=100000,tol=1E-10,ncv=20): state is OBC; regauging only applies to PBC states.')
            return 
        #merge self._mat into the mps tensor
        if self._position<self.N:
            self.absorbCenterMatrix(direction=1)
            self.absorbConnector(location='left') #self._connector is set to 11
            
        if self._position==self.N:
            self.absorbCenterMatrix(direction=-1)
            self.absorbConnector('right')

        self._mat=mf.regaugeIMPS(self._tensors,gauge=gauge,ldens=None,rdens=None,truncate=self._schmidt_thresh,
                                 D=self._D,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=1E-8,numeig=numeig)
        self._connector=np.diag(1.0/np.diag(self._mat)) #note that regaugeIMPS returns in any case a diagonal matrix

        if (gauge=='left') or (gauge=='symmetric'):
            self._position=self.N
        if gauge=='right':
            self._position=0
        return 


    def ortho(self,site,which):
        """
        checks if the orthonormalization of the mps is OK
        prints out some stuff
        """
        
        return self.orthonormalization(site,which)
    
    def orthonormalization(self,site,which,verbose=0):
        """
        checks if the orthonormalization of the mps is OK
        prints out some stuff
        """
        assert(site<self.N)
        if which in (1,'l','left'):
            if verbose>0:
                print ('deviation from left orthonormalization at site {0}: {1}.'.format(site,np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))-np.eye(np.shape(self._tensors[site])[1],dtype=self.dtype))))
            #print np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))
            return np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))-np.eye(np.shape(self._tensors[site])[1],dtype=self.dtype))
        if which in (-1,'r','right'):
            if verbose>0:            
                print ('deviation from right orthonormalization at site {0}: {1}.'.format(site,np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))-np.eye(np.shape(self._tensors[site])[0]),dtype=self.dtype)))
            #print np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))
            return np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))-np.eye(np.shape(self._tensors[site])[0],dtype=self.dtype))




    @property        
    def tensors(self):
        """
        MPS.tensors
        returns the list of mps-tensors as is, i.e. neither the center matrix self._mat nor self._connector 
        are included; it you wish to include them, use self.tolist() instead
        """
        return self._tensors
    
    def tolist(self):
        
        """
        MPS.tolist():
        returns a list of a copy of the mps-tensors; 
        the center matrix mps._mat and the connector matrix  mps._connector
        are absorbed into the mps

        """
        #pos=self._position
        #self.position(0)
        tensors=copy.deepcopy(self._tensors)
        if self._position<len(self):
            #tensors[self._position]=np.tensordot(self._mat,tensors[self._position],([1],[0]))
            tensors[self._position]=ncon.ncon([self._mat,tensors[self._position]],[[-1,1],[1,-2,-3]])
        elif self._position==len(self):
            #tensors[self._position-1]=np.transpose(np.tensordot(tensors[self._position-1],self._mat,([1],[0])),(0,2,1))
            tensors[self._position-1]=ncon.ncon([self._mat,tensors[self._position-1]],[[1,-2],[-1,1,-3]])            
        #tensors[-1]=ncon.ncon([tensors[-1],self._connector],[[-1,1,-3],[1,-2]])
        #self.position(pos)
        return tensors

    def applyTwoSiteGate(self,gate,site,Dmax,thresh=1E-16):
        """
        applies a two-site gate to the mps at site ```site```, 
        and does a truncation with truncation threshold "thresh"
        the center bond is shifted to bond site+1 and mps[site],mps._mat and mps[site+1] are updated


        Parameters:
        --------------------------
        gate:   np.ndarray of shape (dout1,dout2,din1,din2)
                the gate to be applied to the mps.
        site:   int
                the left-hand site of the support of ```gate```
        Dmax:   int
                the maximally allowed bond dimension
                bond dimension will never be larger than ```Dmax```, irrespecitive of ```thresh```
        thresh: float  
                truncation threshold; all schmidt values < ```thresh`` are discarded
        Returns:
        tuple (tw,D):
        tw: float
            the truncated weight
        D:  int
            the current bond dimension D on bond=site+1
        
        """
        assert(len(gate.shape)==4)
        assert(site<len(self)-1)
        #print("in applyToSite; site+1=",site+1,"thresh=",thresh)
        self.position(site+1)        
        newState=ncon.ncon([self.tensor(site,clear=True),self._tensors[site+1],gate],[[-1,1,2],[1,-4,3],[-2,-3,2,3]])
        [Dl,d1,d2,Dr]=newState.shape
        U,S,V=mf.svd(np.reshape(newState,(Dl*d1,Dr*d2)))
        tw=0
        Strunc=S[S>thresh]
        tw=sum(S[min(len(Strunc),Dmax)::]**2)
        Strunc=S[0:min(len(Strunc),Dmax)]
        Strunc/=np.linalg.norm(Strunc)
        U=U[:,0:len(Strunc)]
        V=V[0:len(Strunc),:]
        self._tensors[site]=np.transpose(np.reshape(U,(Dl,d1,len(Strunc))),(0,2,1))
        self._tensors[site+1]=np.transpose(np.reshape(V,(len(Strunc),d2,Dr)),(0,2,1))
        self._mat=np.diag(Strunc)
        return tw,len(Strunc)


    def applyOneSiteGate(self,gate,site,preserve_position=True):
        """
        applies a one-site gate to an mps at site "site"
        the center bond is shifted to bond site+1 
        the _Z norm of the mps is changed
        """
        assert(len(gate.shape)==2)
        assert(site<len(self))
        if preserve_position==True:
            self.position(site+1)
            tensor=ncon.ncon([self.tensor(site,clear=True),gate],[[-1,-2,1],[-3,1]])
            A,mat,Z=mf.prepareTensor(tensor,1)
            self._Z*=Z
            self._tensors[site]=A
            self._mat=mat
        else:
            tensor=ncon.ncon([self[site],gate],[[-1,-2,1],[-3,1]])
            self[site]=tensor


    def applyMPO(self,mpo):
        """
        applies an mpo to an mps; no truncation is done
        """
        assert(len(mpo)==len(self))
        self.position(0)
        self.absorbCenterMatrix(1)
        for n in range(len(self)):
            Ml,Mr,din,dout=mpo[n].shape 
            Dl,Dr,d=self[n].shape
            if n==0:
                self._mat=np.eye(Ml*Dl,dtype=self.dtype)
            self[n]=np.reshape(ncon.ncon([self[n],mpo[n]],[[-1,-3,1],[-2,-4,-5,1]]),(Ml*Dl,Mr*Dr,dout))
            
    def resetZ(self):
        """
        resets the norm-member _Z to 1.0; it does not fully normalize the state; this can be done by sweeping once back and forth
        through the system
        """
        self._Z=1.0
        
    def save(self,filename):
        """
        MPS.save(filename):
        pickles the MPS object to a file "filename".pickle
        
        
        """
        with open(filename+'.pickle', 'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def read(cls,filename):
        """
        MPS.load(filename):
        unpickles an MPS object from a file "filename".pickle
        returns an MPS object
        """
        
        with open(filename+'.pickle', 'rb') as f:
            mps=pickle.load(f)
        return mps
    
    def load(self,filename):
        """
        MPS.load(filename):
        unpickles an MPS object from a file "filename".pickle
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
            
    def cutandpatch(self,index):
        """
        cut an mps at ```index```, swap the pieces and patch it back together
        this is an in place operation
        Parameters:
        -------------------
        index:     int
                   bond at which to cut the state apart; labelling starts at ```0``` for the bond to the left 
                   of the first tensors, and ends at ```MPS.N``` for the right most bond to the right of 
                   the rightmost tensor
        Returns:
        -------------------
        None

        """
        self.position(index)
        self.absorbConnector('left')#self._connector is now an identity
        left=[self[n] for n in range(index)]
        for n in range(index,self.N):
            self[n-index]=self[n]
        for n in range(0,index):
            self[n+self.N-index]=left[n]
        self._position=0
        self._connector=np.eye(self.D[0],dtype=self.dtype)#create a new connector with the right shape
        if not self.D[0]==self.D[-1]:
            raise ValueError('cutandpatch: state does not connect back to itself')
        
