from sys import stdout
import numpy as np
import os,copy
import time
import operator as opr
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import functools as fct
import lib.mpslib.Hamiltonians as H
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
class MPS:

    """Generic Structure of the state: self._tensors contains the MPS matrices. self._position 
    is the bond index at which orthonormalization of self._tensor matrices switches from
    left- to right-orthonormal. self._mat contains the bond matrix (if diagonalized, the schmdit values) 
    of the cut at self._position. 
    
    self.__absorbCenterMatrix__(direction) merges self._mat with self._tensors[self._position] (direction>0) or self._tensors[self._position-1] (direction<0).
    and puts the resulting tensor at the corresponding position.
    
    self._connector contains a connector matrix such that the tensors can be connected back to themselves: self._tensors[-1]*self._mat*self._connector*self._tensors[0]
    (where * is a tensor contraction) is correctly gauge-matched.

    if schmidt_thresh is set below 1E-15, some mps methods, like e.g. __position__, switch 
    to qr instead of svd orthogonalization

    The maximum bond dimension of the MPS is self._D. 
    """

    """ 
    returns an mps with no tensors
    """

    """
    N (int): length of the mps
    D (int): initial bond dimensions
    d (int or array of int): local Hilbert space dimension
    obc (bool): boundary condition
    scaling (float): initial scaling of mps matrices
    dtype (python "float" type or "complex" type: data type
    shift: shift of interval of random number generation for initial mps
    schmidt_thresh: truncation limit of the mps
    r_thresh: internal parameter (ignore it)
    """
    
    def __init__(self):
        pass
    @classmethod
    def random(cls,N,D,d,obc,scaling=0.5,dtype=float,shift=0.5,schmidt_thresh=1E-16,r_thresh=1E-16):
        mps=cls()
        mps._N=N
        mps._D=D
        mps._obc=obc
        mps._dtype=dtype
        mps._schmidt_thresh=schmidt_thresh
        mps._r_thresh=r_thresh

        mps._tensors=mf.MPSinitializer(np.random.random_sample,N,D,d,obc,dtype,scaling,shift)
        mps._d=[shape[2] for shape in list(map(np.shape,mps._tensors))]
        mps._mat=np.eye(np.shape(mps._tensors[-1])[1])
        mps._mat=mps._mat/np.sqrt(np.trace(mps._mat.dot(herm(mps._mat))))
        mps._connector=np.diag(1.0/np.diag(mps._mat))
        mps._position=mps._N
        mps.__position__(0)
        mps.__position__(mps._N)
        mps._gamma=[]
        mps._lambda=[]
        return mps


    @classmethod
    def zeros(cls,N,D,d,obc,scaling=0.0,dtype=float,shift=0.0,schmidt_thresh=1E-16,r_thresh=1E-16):
        mps=cls()

        mps._N=N
        mps._D=D
        mps._obc=obc
        mps._dtype=dtype
        mps._schmidt_thresh=schmidt_thresh
        mps._r_thresh=r_thresh

        mps._tensors=mf.MPSinitializer(np.zeros,N,D,d,obc,dtype,scaling,shift)
        mps._d=[shape[2] for shape in list(map(np.shape,mps._tensors))]
        mps._mat=np.eye(np.shape(mps._tensors[-1])[1])
        mps._mat=mps._mat/np.sqrt(np.trace(mps._mat.dot(herm(mps._mat))))
        mps._connector=np.diag(1.0/np.diag(mps._mat))
        mps._position=mps._N
        mps._gamma=[]
        mps._lambda=[]
        return mps

    @classmethod
    def ones(cls,N,D,d,obc,scaling=1.0,dtype=float,shift=0.0,schmidt_thresh=1E-16,r_thresh=1E-16):
        mps=cls()
        mps._N=N
        mps._D=D
        mps._obc=obc
        mps._dtype=dtype
        mps._schmidt_thresh=schmidt_thresh
        mps._r_thresh=r_thresh

        mps._tensors=mf.MPSinitializer(np.ones,N,D,d,obc,dtype,scaling,shift)
        mps._d=[shape[2] for shape in list(map(np.shape,mps._tensors))]
        mps._mat=np.eye(np.shape(mps._tensors[-1])[1])
        mps._mat=mps._mat/np.sqrt(np.trace(mps._mat.dot(herm(mps._mat))))
        mps._connector=np.diag(1.0/np.diag(mps._mat))
        mps._position=mps._N
        mps.__position__(0)
        mps.__position__(mps._N)
        mps._gamma=[]
        mps._lambda=[]
        return mps
        
    @classmethod
    def productState(cls,localstate,d,obc,scaling=1.0,dtype=float,shift=0.0,schmidt_thresh=1E-16,r_thresh=1E-16):
        mps=cls()
        mps._N=len(localstate)
        mps._D=1
        mps._obc=obc
        mps._dtype=dtype
        mps._schmidt_thresh=schmidt_thresh
        mps._r_thresh=r_thresh
        mps._tensors=mf.MPSinitializer(np.zeros,mps._N,D=1,d=d,obc=obc,dtype=dtype,scale=1.0,shift=0.0)
        for n in range(mps._N):
            mps._tensors[n][0,0,localstate[n]]=1.0
        mps._d=[shape[2] for shape in list(map(np.shape,mps._tensors))]
        mps._mat=np.eye(np.shape(mps._tensors[-1])[1])
        mps._mat=mps._mat/np.sqrt(np.trace(mps._mat.dot(herm(mps._mat))))
        mps._connector=np.diag(1.0/np.diag(mps._mat))
        mps._position=mps._N
        mps.__position__(0)
        mps.__position__(mps._N)
        mps._gamma=[]
        mps._lambda=[]
        return mps
        
    def __copy__(self):
        cop=MPS.zeros(self._N,self._D,self._d,self._obc,dtype=self._dtype,schmidt_thresh=self._schmidt_thresh,r_thresh=self._r_thresh)

        cop._tensors=copy.deepcopy(self._tensors)
        cop._d=copy.deepcopy(self._d)
        cop._mat=copy.deepcopy(self._mat)
        cop._connector=copy.deepcopy(self._connector)
        cop._position=self._position
        cop._gamma=copy.deepcopy(self._gamma)
        cop._lambda=copy.deepcopy(self._lambda)
        return cop


    def __len__(self):
        return self._N
    
    def __getitem__(self,n):
        if isinstance(n,int):
            assert(abs(n)<len(self))
            return self._tensors[n]

    def __setitem__(self,n,tensor):
        assert(abs(n)<len(self))
        self._tensors[n]=np.copy(tensor)

    #act on all tensors with "operation"
    def __unary_operations__(self,operation,*args,**kwargs):
        res=self.__copy__()
        for n in range(len(res)):
            res[n]=operation(res[n],*args,**kwargs)
        return res

    def __conj__(self,*args,**kwargs):
        res=self.__unary_operations__(np.conj,*args,**kwargs)
        res._qflow=utils.flipsigns(res._qflow)                
        return res

    def __conjugate__(self,*args,**kwargs):
        return self.__conj__(self,*args,**kwargs)

    def conjugate(self,*args,**kwargs):
        return self.__conj__(self,*args,**kwargs)

    def conj(self,*args,**kwargs):
        return self.__conj__(self,*args,**kwargs)

    #take elementwise exp
    def __exp__(self,*args,**kwargs):
        res=self.__unary_operations__(np.exp,*args,**kwargs)
        return res
    def exp(self,*args,**kwargs):
        return self.__exp__(*args,**kwargs)
       
    #take elementwise sqrt
    def __sqrt__(self,*args,**kwargs):
        res=self.__unary_operations__(np.sqrt,*args,**kwargs)
        return res
    def sqrt(self,*args,**kwargs):
        return self.__sqrt__(*args,**kwargs)

    def __real__(self,*args,**kwargs):
        res=self.__unary_operations__(np.real,*args,**kwargs)
        res._dtype=float        
        return res

    def real(self,*args,**kwargs):
        return self.__real__(*args,**kwargs)

    def __imag_(self,*args,**kwargs):
        res=self.__unary_operations__(np.imag,*args,**kwargs)
        res._dtype=float        
        return res
    def imag(self,*args,**kwargs):
        return self.__imag__(*args,**kwargs)
    

    __neg__ = generate_unary_deferer(opr.neg)
    __pos__ = generate_unary_deferer(opr.pos)
    __abs__ = generate_unary_deferer(abs)
    __invert__ = generate_unary_deferer(opr.invert)

    
    """
    merges self._connector into the MPS. self._connector is merged into either the 'left' most or 
    the 'right' most mps-tensor. changes self._connector to be the identity: self._connector=11
    """

    def __absorbConnector__(self,location):
        try:
            assert(location=='left' or location=='right')
        except AssertionError:
            sys.exit('mps.__absorbConnector__(self,location): wrong value for "location"; use "location"="left" or "right"')

        if location=='right':
            self._tensors[self._N-1]=np.transpose(np.tensordot(self._tensors[self._N-1],self._connector,([1],[0])),(0,2,1))
            D=np.shape(self._tensors[self._N-1])[1]
            self._connector=np.eye(D)

        if location=='left':
            self._tensors[0]=np.tensordot(self._connector,self._tensors[0],([1],[0]))
            D=np.shape(self._tensors[0])[0]
            self._connector=np.eye(D)

    """
    merges self._mat into the MPS. self._mat is merged into either the left (direction<0) or 
    the right (direction>0) tensor at bond self._position
    changes self._mat to be the identity: self._mat=11; does not change self._connector
    """

    def __absorbCenterMatrix__(self,direction):
        assert(direction!=0)
        if self._position==self._N:
            if direction>0:
                warnings.warn('mps.__absorbCenterMatrix__(self,direction): self._position=N and direction>0; cannot contract bond-matrix to the right because there is no right tensor')
            assert(direction<0)
        if self._position==0:
            if direction<0:
                warnings.warn('mps.__absorbCenterMatrix__(self,direction): self._position=0 and direction<0; cannot contract bond-matrix to the left tensor because there is no left tensor')
            assert(direction>0)

        if (direction>0):
            self._tensors[self._position]=np.tensordot(self._mat,self._tensors[self._position],([1],[0]))
            D=np.shape(self._mat)[0]
            self._mat=np.eye(D)
        if (direction<0):
            self._tensors[self._position-1]=np.transpose(np.tensordot(self._tensors[self._position-1],self._mat,([1],[0])),(0,2,1))
            D=np.shape(self._mat)[0]
            self._mat=np.eye(D)
            
    def __len__(self):
        return len(self._tensors)
    
    def __iter__(self):
        return iter(self._tensors)

    def __dot__(self,mps):
        return mf.overlap(self,mps)

    def dot(self,mps):
        return mf.overlap(self,mps)

    def __D__(self):
        D=[]
        for n in range(len(self)):
            D.append(self[n].shape[0])
        D.append(self[len(self)-1].shape[1])
        return D


    """
    shifts the center site to "bond"
    "schmidt_thresh" overrides the self._schmidt_thresh parameter of the MPS temporarily
    if schmidt_thresh>1E-15, then routine uses an svd to truncate the mps
    If "D" is specified, the bond dimension after schmidt_thresh is additionally never larger than D.
    "D" will override self._D as the new maximum bond dimension of the MPS
    The function does not modify self._connector
    """
    def __position__(self,bond,schmidt_thresh=1E-16,D=None,r_thresh=1E-14):
        assert(bond<=self._N)
        assert(bond>=0)
        """
        set the values for the schmidt_thresh-threshold, D-threshold and r_thresh-threshold
        r_thresh is used in case that svd throws an exception (happens sometimes in python 2.x)
        in this case, one preconditions the method by first doing a qr, and setting all values in r
        which are smaller than r_thresh to 0, then does an svd.
        """
        if D!=None:
            self._D=D
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
                    tensor,self._mat=mf.prepareTensor(self._tensors[n],direction=1)
                else:
                    tensor,s,v=mf.prepareTruncate(self._tensors[n],direction=1,D=self._D,thresh=_schmidt_thresh,\
                                                  r_thresh=_r_thresh)
                    self._mat=np.diag(s).dot(v)
                self._tensors[n]=np.copy(tensor)
                if (n+1)<bond:
                    self._tensors[n+1]=np.tensordot(self._mat,self._tensors[n+1],([1],[0]))

        if bond<self._position:
            self._tensors[self._position-1]=np.transpose(np.tensordot(self._tensors[self._position-1],self._mat,([1],[0])),(0,2,1))
            for n in range(self._position-1,bond-1,-1):
                if _schmidt_thresh < 1E-15 and D==None:
                    tensor,self._mat=mf.prepareTensor(self._tensors[n],direction=-1)
                else:

                    u,s,tensor=mf.prepareTruncate(self._tensors[n],direction=-1,D=self._D,thresh=_schmidt_thresh,\
                                                  r_thresh=_r_thresh)
                    self._mat=u.dot(np.diag(s))
                    
                self._tensors[n]=np.copy(tensor)
                if n>bond:
                    self._tensors[n-1]=np.transpose(np.tensordot(self._tensors[n-1],self._mat,([1],[0])),(0,2,1))
        self._position=bond



    """
    measure a set of local N operators "ops", where N=len(ops)=len(mps)
    "lb" and "rb" are left and right boundary conditions to be applied 
    if the state is pbc; for obc they can be left None
    the routine moves the centersite to the left boundary, measures and moves
    it back to its original position; this might cause some overhead
    """
    
    def __measureLocal__(self,ops,lb=None,rb=None):
        if lb==None or rb==None:
            if self._obc==False:
                return NotImplemented
            else:
                pos=self._position
                self.__position__(0)
                L=mf.measureLocal(self,ops,lb=np.ones((1,1,1)),rb=np.ones((1,1,1)),ortho='right')
                self.__position__(pos)
                return L




        
    """ 
    This is a quite slow function to calculate observables; it takes a list of operators "ops" and a list of sites "sites" of where the operators live, and calculates <op[1]op[2],...,op[n]>
    "sites" has to be a list of (weakly) monotonically increasing numbers 
    the function can also measure fermionic correlation fucntions, if the jordan wigner strings are given in P; P is a list 
    containing the jordan-woigner operators for each site of the MPS (len(P)==len(mps))

    ROUTINE MAY BE WRONG 
    """
    def __measure__(self,ops,sites,P=None):
        if not isinstance(sites,list):
            ops=list([ops])
            sites=list([sites])            
        if P!=None:
            assert(len(P)==len(self))
        
        if not sorted(sites)==sites:
            sys.exit('mps.py.__measure__(self,ops,sites): sites are not in increasing order')
        if not self._obc==True:
            warnings.warn('calculating observable for an infinite MPS; be sure that it is regauged correctly')
        s=0
        self.__position__(sites[-1]+1)
        #the parity of the operator string one wants to measure:
        par=len(ops)%2
        if par==0 or P==None:
            L=np.zeros((self[sites[s]].shape[0],self[sites[s]].shape[0],1)).astype(self._dtype)
            L[:,:,0]=np.copy(np.eye(self[sites[s]].shape[0]))
        elif par==1 and P!=None:
            #for fermionic modes one has to take into consideration the fermionic minus signs from the start (if len(ops) is odd)
            L=np.zeros((self[0].shape[0],self[0].shape[0],1)).astype(self._dtype)
            L[:,:,0]=np.copy(np.eye(self[0].shape[0]))
            for n in range(sites[s]-1):
                L=mf.addLayer(L,self[n],P[n],self[n],1)

        while s<len(sites):
            if s<len(sites):
                n=s
                op=np.eye(self[s].shape[2])
                while n<len(sites) and sites[n]==sites[s]:
                    op=op.dot(ops[n])
                    par=(par+1)%2
                    n+=1

            assert(sites[s]==sites[n-1])
            mpo=np.zeros((1,1,op.shape[0],op.shape[1])).astype(self._dtype)
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

        R=np.zeros((self[sites[s-1]].shape[1],self[sites[s-1]].shape[1],1))
        R[:,:,0]=np.eye(self[sites[s-1]].shape[1])
        Z=np.trace(self._mat.dot(herm(self._mat)))
        if not((np.abs(np.trace(self._mat.dot(herm(self._mat))))-1.0)<1E-10):
            warnings.warn('mps.py.measure(self,ops,sites): state is not normalized')

        return ncon.ncon([L,self._mat,np.conj(self._mat),R],[[1,2,3],[1,4],[2,5],[4,5,3]])


    """ 
    a dedicated routine for truncating an mps (for obc, this can also be done using self.__position__(self,pos))
    For the case of obc==True (infinite system with finite unit-cell), the function modifies self._connector
    schmidt_thresh: truncation threshold
    D: maximum bond dimension; if None, the bond dimension is adapted to match schmidt_thresh
    r_thresh: internal parameter, has no effect on the outcome and can be ignored
    returned_gauge: 'left','right' or 'symmetric': the desired gauge after truncation
    """

    def __truncate__(self,schmidt_thresh,D=None,returned_gauge=None,nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12,r_thresh=1E-14):
        if D!=None and D>self._D:
            print('MPS.__truncate__(): D>self._D, no truncation neccessary')
            return
        else:
            if self._obc==True:
                self.__position__(0)
                self.__position__(self._N)
                print (schmidt_thresh,D)
                print('truncating')
                self.__position__(0,schmidt_thresh=schmidt_thresh,D=D,r_thresh=r_thresh)
            if self._obc==False:
                if self._position<self._N:
                    self.__absorbCenterMatrix__(direction=1)
                    self.__absorbConnector__(location='left') #self._connector is set to 11
                if self._position==self._N:
                    self.__absorbCenterMatrix__(direction=-1)
                    self.__absorbConnector__('right')

                self._mat=mf.regaugeIMPS(self._tensors,gauge='symmetric',ldens=None,rdens=None,truncate=schmidt_thresh,D=D,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=1E-8)
                self._connector=np.diag(1.0/np.diag(self._mat)) #note that regaugeIMPS returns in any case a diagonal matrix
                self._position=self._N

                if returned_gauge!=None:
                    self.__regauge__(returned_gauge)                    
                
    """
    returns the tensor at site="site" contracted with self._mat; self._position has to be either site or site+1
    if clear=True, self._mat is replaced with an identity matrix
    """
    
    def __tensor__(self,site,clear=False):
        assert((site==self._position) or (site==(self._position-1)))
        if (site==self._position):
            out=np.tensordot(self._mat,self._tensors[site],([1],[0]))
            if clear==True:
                self._mat=np.eye(np.shape(self._tensors[site])[0])
            return out

        if (site==(self._position-1)):
            out=np.transpose(np.tensordot(self._tensors[site],self._mat,([1],[0])),(0,2,1))            
            if clear==True:
                self._mat=np.eye(np.shape(self._tensors[site])[1])
            return out

    """
    canonizes the mps, i.e. brings it into Gamma,Lambda form; Gamma and Lambda are stored in the mps._gamma and mps._lambda member lists;
    len(mps._lambda) is len(mps)+1, i.e. there are boundary lambdas to the left and right of the mps; for obc, these are just 1.0
    """
    def __canonize__(self,nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12):
        self.mps.__regauge__(gauge='symmetric',nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv)
        self._gamma,self._lambda=mf.canonizeMPS(self.mps)
        
    
    """
    regauge brings state in either left, right or symmetric orthonormal form
    the state should be a finite unitcell state on an infinite lattice
    gauge = 'left' or 'right' bring the state into left or right orthonormal form such that it can be connected back 
        to itself. For gauge='left' or 'right', self._mat=11 and self._connector=11 are set to be the identity. To calculate 
        any observables, one needs to additionally calculate the right (for gauge = 'left') or the left (for gauge = 'right') reduced
        density matrices
    if gauge='symmetric' the state is brought into symmetric gauge, such that it is totally left normalized, 
    and self._mat=lambda, self._connector=1/lambda contain the schmidt-values. note that __regauge__ uses self._schmidt_thresh
    as effective truncation, so the bond dimensions might change
    """
    def __regauge__(self,gauge,nmaxit=100000,tol=1E-10,ncv=20,pinv=1E-12):
        if self._obc==True:
            print ('in MPOS.__regauge__(self,gauge,nmaxit=100000,tol=1E-10,ncv=20): state is OBC; regauging only applies to PBC states.')
            return 
        #merge self._mat into the mps tensor
        if self._position<self._N:
            self.__absorbCenterMatrix__(direction=1)
            self.__absorbConnector__(location='left') #self._connector is set to 11

        if self._position==self._N:
            self.__absorbCenterMatrix__(direction=-1)
            self.__absorbConnector__('right')

        self._mat=mf.regaugeIMPS(self._tensors,gauge=gauge,ldens=None,rdens=None,truncate=self._schmidt_thresh,D=self._D,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=1E-8)
        self._connector=np.diag(1.0/np.diag(self._mat)) #note that regaugeIMPS returns in any case a diagonal matrix

        if (gauge=='left') or (gauge=='symmetric'):
            self._position=self._N
        if gauge=='right':
            self._position=0
        return 


    
    """
    checks if the orthonormalization of the mps is OK
    """
    
    def __orthonormalization__(self,site,direction):
        assert(site<self._N)
        if direction>0:
            print ('deviation from left orthonormalization at site {0}: {1}.'.format(site,np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))-np.eye(np.shape(self._tensors[site])[1]))))
            #print np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))
            return np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([0,2],[0,2]))-np.eye(np.shape(self._tensors[site])[1]))
        if direction<0:
            print ('deviation from right orthonormalization at site {0}: {1}.'.format(site,np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))-np.eye(np.shape(self._tensors[site])[0]))))
            #print np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))
            return np.linalg.norm(np.tensordot(self._tensors[site],np.conj(self._tensors[site]),([1,2],[1,2]))-np.eye(np.shape(self._tensors[site])[0]))

    """
    returns the mps-tensors; if absorb_center=1, the centermatrix is absorbed in right direction into the MPS,
    if absorb_center=-1, it is absorbed in left direction.  absorb_connector='left' or 'right' absorbes the
    connector matrix into the mps
    """
    def __tensors__(self,absorb_center=None,absorb_connector=None):
        if absorb_center!=None:
            self.__absorbCenterMatrix__(direction=absorb_center)
        if absorb_connector!=None:
            self.__absorbConnector__(absorb_connector)

        return copy.deepcopy(self._tensors)


    """
    applies a two-site gate to amps and does an optional truncation with truncation threshold "thresh"
    "Dmax" is the maximally allowed bond dimension; bond dimension will never be larger than "Dmax", irrespecitive of "thresh"
    site has to be the left-hand site of the operator support
    """
    def __applyTwoSiteGate__(self,gate,site,Dmax=None,thresh=1E-10):
        assert(len(gate.shape)==4)
        assert(site<len(self)-1)
        self.__position__(site+1)
        newState=ncon.ncon([self.__tensor__(site,clear=True),self._tensors[site+1],gate],[[-1,1,2],[1,-4,3],[2,3,-2,-3]])
        [Dl,d1,d2,Dr]=newState.shape
        U,S,V=mf.svd(np.reshape(newState,(Dl*d1,Dr*d2)))
        S=S[S>thresh]
        tw=0
        if len(S)>Dmax:
            tw+=sum(S[Dmax::]**2)
            S=S[0:Dmax]
        S/=np.linalg.norm(S)
        U=U[:,0:len(S)]
        V=V[0:len(S),:]
        self._D=len(S)
        self._tensors[site]=np.transpose(np.reshape(U,(Dl,d1,len(S))),(0,2,1))
        self._tensors[site+1]=np.transpose(np.reshape(V,(len(S),d2,Dr)),(0,2,1))
        self._mat=np.diag(S)
        return tw,len(S)
    
    """
    applies an mpo to an mps; no truncation is done
    """
    def __applyMPO__(self,mpo):
        assert(len(mpo)==len(mps))
        self.__position__(0)
        self.__absorbCenterMatrix__(1)
        for n in range(len(mps)):
            Ml,Mr,din,dout=mpo[n].shape 
            Dl,Dr,d=self[n].shape
            if n==0:
                self._mat=np.eye(Ml*Dl)
            self[n]=np.reshape(ncon.ncon([self[n],mpo[n]],[[-1,-3,1],[-2,-4,1,-5]]),(Ml*Dl,Mr*Dr,dout))
