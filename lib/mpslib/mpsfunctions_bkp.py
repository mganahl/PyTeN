"""
@author: Martin Ganahl
"""

from __future__ import absolute_import, division, print_function
from distutils.version import StrictVersion
from sys import stdout
import sys,time,copy,warnings
import numpy as np
import lib.mpslib.sexpmv as sexpmv
try:
    MPSL=sys.modules['lib.mpslib.mps']
except KeyError:
    import lib.mpslib.mps as MPSL


import lib.ncon as ncon
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
#from scipy.integrate import solve_ivp

import functools as fct
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.interpolate import griddata
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import lgmres
from numpy.linalg.linalg import LinAlgError
import lib.Lanczos.LanczosEngine as lanEn
import lib.mpslib.Tensor as tnsr
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


def overlap(mps1,mps2):
    """
    overlap(mps1,mps2)
    calculates the overlap between two mps 

    Parameters:
    -----------------------------------
    mps1, mps2: list of mps tensors, or two MPS-objects

    Returns:
    -----------------------------------------
    complex: the overlap; 
             note that mps1._Z*np.conj(mps2._Z) is included in the result!
    """
    if isinstance(mps1,MPSL.MPS) and isinstance(mps2,MPSL.MPS):
        pos1=mps1._position
        pos2=mps2._position        
        if (mps1.N-mps1._position)>mps1._position:
            mps1.position(0)
            mps2.position(0)        
        else:
            mps1.position(mps1.N)
            mps2.position(mps2.N)
    #if (mps2.N-mps2._position)>mps2._position:
    #    mps2.position(0)
    #else:
    #    mps2.position(mps2.N)        


    if len(mps1)!=len(mps2):
        raise ValueError("overlap(mps1,mps2): mps have to be of same length")
    if (mps1[0].shape[0]!=1):
        raise ValueError("overlap(mps1,mps2): mps1[0].shape[0]!=1")
    if (mps1[-1].shape[1]!=1):
        raise ValueError("overlap(mps1,mps2): mps1[-1].shape[1]!=1")
    if (mps2[0].shape[0]!=1):
        raise ValueError("overlap(mps1,mps2): mps2[0].shape[0]!=1")
    if (mps2[-1].shape[1]!=1):
        raise ValueError("overlap(mps1,mps2): mps2[-1].shape[1]!=1")

    L=np.reshape(ncon.ncon([mps1[0],np.conj(mps2[0])],[[1,-1,2],[1,-2,2]]),(mps1[0].shape[1],mps2[0].shape[1],1))
    for n in range(1,len(mps1)):
        L=addELayer(L,mps1[n],mps2[n],direction=1)
        
    if isinstance(mps1,MPSL.MPS) and isinstance(mps2,MPSL.MPS):        
        mps1.position(pos1)
        mps2.position(pos2)    
    return np.trace(L[:,:,0])*mps1._Z*np.conj(mps2._Z)



def svd(mat,full_matrices=False,compute_uv=True,r_thresh=1E-14):
    """
    wrapper around numpy svd
    catches a weird LinAlgError exception sometimes thrown by lapack (cause not entirely clear, but seems 
    related to very small matrix entries)
    if LinAlgError is raised, precondition with a QR decomposition (fixes the problem)


    Parameters
    ----------
    mat:           array_like of shape (M,N)
                   A real or complex array with ``a.ndim = 2``.
    full_matrices: bool, optional
                   If True (default), `u` and `vh` have the shapes ``(M, M)`` and
                   (N, N)``, respectively.  Otherwise, the shapes are
                   (M, K)`` and ``(K, N)``, respectively, where
                   K = min(M, N)``.
    compute_uv :   bool, optional
                   Whether or not to compute `u` and `vh` in addition to `s`.  True
                   by default.

    Returns
    -------
    u : { (M, M), (M, K) } array
        Unitary array(s). The shape depends
        on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    s : (..., K) array
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    vh : { (..., N, N), (..., K, N) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    """
    try: 
        [u,s,v]=np.linalg.svd(mat,full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        [q,r]=np.linalg.qr(mat)
        r[np.abs(r)<r_thresh]=0.0
        u_,s,v=np.linalg.svd(r)
        u=q.dot(u_)
        print('caught a LinAlgError with dir>0')
    return u,s,v


def qr(mat,signfix):
    """
    a simple wrapper around numpy qr, allows signfixing of the diagonal of r or q
    """
    dtype=type(mat[0,0])
    q,r=np.linalg.qr(mat)
    if signfix=='q':
        sign=np.sign(np.diag(q))
        unit=np.diag(sign)
        return q.dot(unit),herm(unit).dot(r)
    if signfix=='r':
        sign=np.sign(np.diag(r))
        unit=np.diag(sign)
        return q.dot(herm(unit)),unit.dot(r)

    
def mpsTensorAdder(A,B,boundary_type,ZA=1.0,ZB=1.0):
    """
    adds to Tensors A and B in the MPS fashion
    A,B:    Tensor objects
    boundary_type:  str
                    can be ('l','left',-1) or ('r','right',1) or ('b','bulk',0)
    """
    dtype=np.result_type(A,B)
    if A.shape[2]!=B.shape[2]:
        raise ValueError('physical dimensions  of A and B are not compatible')
    if len(A.shape)!=3:
        raise ValueError('A is not an MPS tensor')
    if len(B.shape)!=3:
        raise ValueError('B is not an MPS tensor')
    if not type(A)==type(B):
        raise TypeError('type(A)!=type(B)')
    if boundary_type in ('left','l',-1):
        if np.sum(A.shape[0])!=1:
            raise ValueError('A.shape[0] is not one dimensional; this is incompatible with left open boundary conditions')
        if np.sum(B.shape[0])!=1:
            raise ValueError('B.shape[0] is not one dimensional; this is incompatible with left open boundary conditions')
        return A.concatenate([A*ZA,B*ZB],axis=1).view(type(A))
    
    elif boundary_type in ('right','r',1):
        if np.sum(A.shape[1])!=1:
            raise ValueError('A.shape[1] is not one dimensional; this is incompatible with right open boundary conditions')
        if np.sum(B.shape[1])!=1:
            raise ValueError('B.shape[1] is not one dimensional; this is incompatible with rig open boundary conditions')
        return A.concatenate([A*ZA,B*ZB],axis=0).view(type(A))
        
    elif boundary_type in (0,'b','bulk'):
        if isinstance(A,tnsr.Tensor) and isinstance(B,tnsr.Tensor):
            res=A.zeros((A.shape[0]+B.shape[0],A.shape[1]+B.shape[1],A.shape[2]),dtype=dtype)
            for indx in range(A.shape[2]):
                res[:,:,indx]=A.directSum(A[:,:,indx]*ZA,B[:,:,indx]*ZB)
            return res
        else:
            return NotImplemented
    
def addMPS(mps1,mps2,obc=True):
    """
    addMPS(mps1,mps2,obc=True)
    adds two mps
    mps1,mps2: two lists of mps tensors of dimension (D,D,d)
    """
    if len(mps1)!=len(mps2):
        raise ValueError("addMPS(mps1,mps2): mps1 and mps2 have different length")
    
    dtype=np.result_type(*mps1,*mps2).type
    if obc==True:

        mps=[]
        shape=(1,mps1[0].shape[1]+mps2[0].shape[1],mps1[0].shape[2])
        if mps1[0].shape[2]!=mps2[0].shape[2]:
            raise ValueError("in addMPS: mps1[0] and mps2[0] have different hilbert space dimensions")

        mat=(np.random.random_sample(shape).astype(dtype)-0.5)*1E-16
        for n in range(mps1[0].shape[2]):
            mat[0,0:mps1[0].shape[1],n]=mps1[0][0,:,n]*Z1
            mat[0,mps1[0].shape[1]:mps1[0].shape[1]+mps2[0].shape[1],n]=mps2[0][0,:,n]*Z2
        mps.append(mat)
        for site in range(1,len(mps1)-1):
            if mps1[site].shape[2]!=mps2[site].shape[2]:
                raise ValueError("in addMPS: mps1[{0}] and mps2[{0}] have different hilbert space dimensions".format(site))
            
            l=list(map(sum,zip(mps1[site].shape[0:2],mps2[site].shape[0:2])))
            l.append(mps1[site].shape[2])
            shape=tuple(l)
            mat=(np.random.random_sample(shape).astype(dtype)-0.5)*1E-16            
            for n in range(mps1[site].shape[2]):
                mat[:,:,n]=directsum(mps1[site][:,:,n],mps2[site][:,:,n])
            mps.append(mat)
        
        shape=(mps1[-1].shape[0]+mps2[-1].shape[0],1,mps1[-1].shape[2])
        if mps1[-1].shape[2]!=mps2[-1].shape[2]:
            raise ValueError("in addMPS: mps1[-1] and mps2[-1] have different hilbert space dimensions")
        mat=(np.random.random_sample(shape).astype(dtype)-0.5)*1E-16
        for n in range(mps1[-1].shape[2]):
            mat[0:mps1[-1].shape[0],0,n]=mps1[-1][:,0,n]
            mat[mps1[-1].shape[0]:mps1[-1].shape[0]+mps2[-1].shape[0],0,n]=mps2[-1][:,0,n]
        mps.append(mat)
        
    if obc==False:

        if len(mps1)!=len(mps2):
            raise ValueError("addMPS(mps1,mps2): mps1 and mps2 have different length")
        mps=[]
        for site in range(len(mps1)):
            if mps1[site].shape[2]!=mps2[site].shape[2]:
                raise ValueError("in addMPS: mps1[{0}] and mps2[{0}] have different hilbert space dimensions".format(site))
            
            l=list(map(sum,zip(mps1[site].shape[0:2],mps2[site].shape[0:2])))
            l.append(mps1[site].shape[2])
            shape=tuple(l)
            mat=(np.random.random_sample(shape).astype(dtype)-0.5)*1E-16
            if site==0:
                for n in range(mps1[site].shape[2]):
                    mat[:,:,n]=directsum(mps1[site][:,:,n]*Z1,mps2[site][:,:,n]*Z2)
            else:
                for n in range(mps1[site].shape[2]):
                    mat[:,:,n]=directsum(mps1[site][:,:,n],mps2[site][:,:,n])
            mps.append(mat)

    return mps



def applyMPO(mps,mpo,inplace=False):
    """
    applyMPO(mps,mpo,inplace=False)
    applies mpo to mps
    if inplace=True, mps is overwritten with the result
    """
    assert(len(mpo)==len(mps))
    if not inplace:
        out=[]
        for n in range(len(mps)):
            [M1,M2,d1,d2]=mpo[n].shape
            [D1,D2,d]=mps[n].shape            
            out.append(np.reshape(np.transpose(np.tensordot(mps[n],mpo[n],([2],[3])),(0,2,1,3,4)),(D1*M1,D2*M2,d2)))
        return out
    if inplace:
        for n in range(len(mps)):
            [M1,M2,d1,d2]=mpo[n].shape
            [D1,D2,d]=mps[n].shape            
            mps[n]=np.reshape(np.transpose(np.tensordot(mps[n],mpo[n],([2],[3])),(0,2,1,3,4)),(D1*M1,D2*M2,d2))
        return mps


def contractionH(L,mps,mpo,R):
    """
    contraction routine for ground-state optimization

    Parameters:
    ----------------------
    L:   np.ndarray of shape (Dl,Dl',Ml) 
         left Hamiltonian environment
    mps: np.ndarray of shape (Dl,Dr,d)
         mps-tensor
    mpo: np.ndarray of shape (Ml,Mr,d,d)
         MPO tensor
    R:   np.ndarray of shape (Dr,Dr',Mr)
         right Hamiltonian environment
    Returns:
    ----------------------
    np.ndarray of shape (Dl',Dr',d)
    routine contracts input and returns it in an np.ndarray of shape (Dl',Dr',d)
    """
    
    term1=np.tensordot(L,mps,([0],[0]))
    term2=np.tensordot(term1,mpo,([1,3],[0,3]))
    term3=np.tensordot(term2,R,([1,2],[0,2]))
    return np.transpose(term3,[0,2,1])

def contractMPO(mps1,mps2,mpo,tensor,site1,site2,direction):

    """
    contracts a left or right environment tensor "tensor" with a unitcell mps-mpo-mps expression,
    starting at site site1 and ending at site site2
    direction<0: start at site2 and end at site1
    direction>0: start at site1 and end at site2
    mps1 is the upper mps, mps2 the lower mps
    site2 is included in the result
    mps1 and mps2 can be lists of ndarrays of dimension (D,D,d) or MPS objects
    mpo can be a list of mpo ndarrays of dimension (M,M,d,d), or an MPO object
    returns: an ndarray of dimension (D,D,M) 
    """
    
    assert(site2>=site1)
    if direction>0:
        L=np.copy(tensor)
        for site in range(site1,site2+1):
            L=addLayer(L,mps1[site],mpo[site],mps2[site],direction)
        return L
    if direction<0:
        R=np.copy(tensor)
        for site in range(site2,site1-1,-1):
            R=addLayer(R,mps1[site],mpo[site],mps2[site],direction)
        return R


def contractE(mps1,mps2,tensor,site1,site2,direction):
    """
    contracts a left or right environment tensor "tensor" with a unitcell mps-mps expression,
    starting at site site1 and ending at site site2, and returns the result
    direction<0: start at site2 and end at site1
    direction>0: start at site1 and end at site2
    mps1 is the upper mps, mps2 the lower mps
    site2 is included in the result
    mps1 and mps2 can be lists of ndarrays of dimension (D,D,d) or MPS objects
    returns: an ndarray of dimension (D,D) 
    """
    assert(site2>=site1)
    if direction>0:
        L=np.copy(tensor)
        for site in range(site1,site2+1):
            L=addELayer(L,mps1[site],mps2[site],direction)
        return L

    if direction<0:
        R=np.copy(tensor)
        for site in range(site2,site1-1,-1):
            R=addELayer(R,mps1[site],mps2[site],direction)
        return R
        

def fixPhase(matrix):
    """
    fix the phase of the diagonal of matrix; 
    used for l and r matrices of infinite MPS 
    returns: the phase-fixed matrix
    """
    matrix=matrix/np.trace(matrix);
    return matrix/np.sqrt(np.trace(matrix.dot(herm(matrix))))


def measure4pointCorrelation(mps,ops,P=None):
    """
    measures a 4-point correlation function
    mps: an MPS object
    ops: a list of 4 operators to be measured
    """
    assert(len(ops)==4)
    N=len(mps)
    s=0
    mps.position(0)
    par=len(ops)%2    
    L0=np.zeros((mps[0].shape[0],mps[0].shape[0],1))
    L0[:,:,0]=np.copy(np.eye(mps[0].shape[0]))
    #at each site n insert an operator
    mpol=np.zeros((1,1,ops[0].shape[0],ops[0].shape[1]))
    #L1=mf.addLayer(L0,mps[0],mpol,mps[0],1)
    #L1s.append(np.copy(L1))
    L1s=[]
    for n in range(N):
        if P==None or par==0:        
            L0=mf.addELayer(L0,mps[n],mps[n],1)
        elif P!=None and par==1:        
            L0=mf.addLayer(L0,mps[n],P[n],mps[n],1)

        for m in range(len(L1s)):
            if P==None or (par+1)%2==0:
                L1s[m]=mf.addELayer(L1s[m],mps[n],mps[n],1)
            elif P!=None and (par+1)%2==1:
                L1s[m]=mf.addLayer(L1s[m],mps[n],P[n],mps[n],1)
        if par==0 and P!=None:
            mpol[0,0,:,:]=ops[0].P[n]
        if par==1 or P==None:
            mpol[0,0,:,:]=ops[0]
            L1s.append(np.copy(mf.addLayer(L0,mps[n],mpol,mps[n],1)))
    R0=np.zeros((mps[N-1].shape[1],mps[N-1].shape[1],1))
    R0[:,:,0]=np.copy(np.eye(mps[N-1].shape[1]))
    mpor=np.zeros((1,1,ops[3].shape[0],ops[3].shape[1]))
    mpor[0,0,:,:]=ops[3]
    #R1=mf.addLayer(R0,mps[N-1],mpor,mps[N-1][0],-1)
    #R1s.append(np.copy(R1))
    R1s=[]
    for n in range(N-1,-1,-1):
        R0=mf.addELayer(R0,mps[n],mps[n],-1)
        for m in range(len(R1s)):
            if P==None:
                R1s[m]=mf.addELayer(R1s[m],mps[n],mps[n],-1)
            else:
                R1s[m]=mf.addLayer(R1s[m],mps[n],P[n],mps[n],-1)
                
        R1s.append(np.copy(mf.addLayer(R0,mps[n],mpor,mps[n],-1)))
        

    if not((np.abs(np.trace(self._mat.dot(herm(self._mat))))-1.0)<1E-10):
        warnings.warn('mps.py.measure(self,ops,sites): state is not normalized')

    #print(L.shape,self._mat.shape,np.conj(self._mat).shape,R.shape)
    #return ncon.ncon([L,self._mat,np.conj(self._mat),R],[[1,2,3],[1,4],[2,5],[4,5,3]])
    
    
def measureLocal(mps,operators,lb,rb,ortho):
    """
    measure local operators, given an mps, a list of local operators, left and right boundaries of the MPS, and the orthogonal state of the MPS.
    len(operators) has to be the same as len(mps). The routine is not very efficient because the whole network is contracted. If the state is canonized, more
    efficient methods can be used.
    
    mps: a list of mps tensors (ndarrays), or an mpslib.mps.MPS object
    operators: a list of np.ndarrays of len(operators)=len(mps). The position of the operator within the list is taken to be 
               the site where it acts, i.e. operators[n] acts at site n. Each operator is measured and the  result is returned in a list
    lb: left boundary of the mps (for obc, lb=np.ones((1,1,1))
    rb: right boundary of the mps (for obc, rb=np.ones((1,1,1))
    ortho (str): can be {'left','right'} and denotes the orthogonal state of the mps
    """

    if isinstance(mps,list):
        #temp=[m.dtype for m in mps]+[o.dtype for o in operators]+[lb,rb]
        dtype=np.result_type(*mps,*operators,lb,rb).type
        
    elif isinstance(mps,MPSL.MPS):
        dtype=np.result_type(mps.dtype,*operators,lb,rb).type
    N=len(mps)
    if len(mps)!=len(operators):
        raise ValueError("measureLocal: len(operators) does not match len(mps)")
    if ortho != 'right' and ortho !='left':
        raise ValueError("measureLocal: unknown orthogonality state {0}".format(ortho))
    
    loc=np.zeros(N).astype(dtype)
    if ortho=='right':
        L0=np.copy(lb).astype(dtype)
        #print('L0: ',L0)
        for n in range(0,N-1):
            [D1,D2,d]=np.shape(mps[n])
            exp=np.expand_dims(np.expand_dims(operators[n],0),0)
            Lmeasure=addLayer(L0,mps[n],np.expand_dims(np.expand_dims(operators[n],0),0),mps[n],1)
            L0=addELayer(L0,mps[n],mps[n],1)
            m=np.trace(Lmeasure[:,:,0])/np.trace(L0)
            loc[n]=m
            if np.abs(np.imag(m))>1e-12:
                warnings.warn('detected imaginary value {0} for observable'.format(np.imag(m)))
            
        Lmeasure=addLayer(L0,mps[N-1],np.expand_dims(np.expand_dims(operators[N-1],0),0),mps[N-1],1)
        L0=addELayer(L0,mps[N-1],mps[N-1],1)        
        m=np.trace(Lmeasure[:,:,0])/np.trace(L0)
        loc[N-1]=m
        if np.abs(np.imag(m))>1e-12:
            warnings.warn('detected imaginary value {0} for observable'.format(np.imag(m)))            
        return loc

    if ortho=='left':
        R0=np.copy(rb)
        for n in range(N-1,0,-1):
            [D1,D2,d]=np.shape(mps[n])
            Rmeasure=addLayer(R0,mps[n],np.expand_dims(np.expand_dims(operators[n],0),0),mps[n],-1)
            R0=addELayer(R0,mps[n],mps[n],-1)            
            m=np.trace(Rmeasure[:,:,-1])/np.trace(R0)
            loc[n]=m
            if np.abs(np.imag(m))>1e-12:
                warnings.warn('detected imaginary value {0} for observable'.format(np.imag(m)))                            


        
        Rmeasure=addLayer(R0,mps[0],np.expand_dims(np.expand_dims(operators[0],0),0),mps[0],-1)
        R0=addELayer(R0,mps[0],mps[0],-1)                    
        m=np.trace(Rmeasure[:,:,-1])/np.trace(R0)
        loc[0]=m
        if np.abs(np.imag(m))>1e-12:
            warnings.warn('detected imaginary value {0} for observable'.format(np.imag(m)))                                        

        return loc



def matrixElementLocal(mps1,mps2,operators,lb,rb):
    """
    measure the matrix elements of a list of local operators "operators", given mps1, mps2, a list of local operators, 
    left and right boundaries of the MPS, and the orthogonal state of the MPS 
    len(operators) has to be the same as len(mps). The routine is not very efficient because the whole network is contracted. If the state is canonized, more
    efficient methods can be used.
    
    mps1,mps2: lists of mps tensors (ndarrays), or mpslib.mps.MPS objects
    operators: a list of operators of len(operators)=len(mps). The position within of an operator within the list is taken to be 
               the site where it acts. Each operator is measured and the  result is returned in a list
    lb:        left boundary of the mps (for obc, lb=np.ones((1,1,1))
    rb:        right boundary of the mps (for obc, rb=np.ones((1,1,1))
    """
    dtype=np.result_type(*mps1,*mps2).type
    N=len(mps1)
    if len(mps1)!=len(mps2):
        raise ValueError("matrixElementLocal: mps1 and mps2 have different lengths")
    if len(mps1)!=len(operators):
        raise ValueError("matrixElementLocal: len(operators) does not match len(mps1) or len(mps2)")
    
    loc=np.zeros(N).astype(dtype)
    L0=np.copy(lb)
    
    R=[np.copy(rb)]
    for n in range(N-1,-1,-1):
        R.append(addELayer(R[-1],mps1[n],mps2[n],-1))
    for n in range(0,N-1):
        [D1,D2,d]=np.shape(mps1[n])
        Lmeasure=addLayer(L0,mps1[n],np.expand_dims(np.expand_dims(operators[n],0),0),mps2[n],1)
        m=np.tensordot(Lmeasure,R[N-1-n],([0,1,2],[0,1,2]))
        loc[n]=m
        L0=addELayer(L0,mps1[n],mps2[n],1)

    
    Lmeasure=addLayer(L0,mps1[N-1],np.expand_dims(np.expand_dims(operators[N-1],0),0),mps2[N-1],1)
    m=np.tensordot(Lmeasure,R[0],([0,1,2],[0,1,2]))            
    loc[N-1]=m
    return loc
    

def measure(operator,lb,rb,mps,site):
    """
    measure a local operator, given an list of ndarrays of dimension (D,D,d), a list of local operators in mpo format, left and right boundaries of the MPS, and the orthogonal state of the MPS 
    len(operators) has to be the same as len(mps). The routine is not very efficient because the whole network is contracted. If the state is canonized, more
    efficient methods can be used.
    
    mps: a list of mps tensors (ndarrays) (or an mpslib.mps.MPS object)
    operator: a list of operators of len(operator)=len(mps). Each operator is measured and the 
    result is returned in a list
    lb: left boundary of the mps (for obc, lb=np.ones((1,1,1))
    rb: right boundary of the mps (for obc, rb=np.ones((1,1,1))
    ortho (str): can be {'left','right'} and denotes the orthogonal state of the mps
    """
    
    L=np.copy(lb)
    R=np.copy(rb)
    for n in range(site):
        L=addELayer(L,mps[n],mps[n],1)
    for n in range(len(mps)-1,site,-1):
        R=addELayer(R,mps[n],mps[n],-1)
    Lnorm=addELayer(L,mps[site],mps[site],1)
    L=addLayer(L,mps[site],operator,mps[site],1)
    #print site,(np.tensordot(Lnorm,R,([0,1],[0,1]))[0,0]),(np.tensordot(lb,rb,([0,1],[0,1]))[0,0])
    return np.tensordot(L,R,([0,1],[0,1]))[0,0]/(np.tensordot(Lnorm,R,([0,1],[0,1]))[0,0])


    


#used in the lattice-cMPS context; ignore for the case of lattice MPS
def prepareTensorfixA0(mps,direction):

    [D1,D2,d]=np.shape(mps)
    dtype=mps.dtype.type

    if direction>0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])
        mat[:,:,0]=np.eye(D1,dtype=dtype).view(mps)
        mat[:,:,1]=mps[:,:,1].dot(np.linalg.pinv(A0)).view(type(mps))
        tensor,r,Z=prepareTensor(mat,1,fixphase='q')
        matout=r.dot(A0).view(type(mps))

        return tensor,matout
        
    if direction<0:
        mat=np.zeros((D1,D2,d),dtype=dtype)
        A0=np.copy(mps[:,:,0])

        mat[:,:,0]=np.eye(D1,dtype=dtype)
        mat[:,:,1]=np.linalg.pinv(A0).dot(mps[:,:,1]).view(type(mps))
        
        tensor,r,Z=prepareTensor(mat,-1,fixphase='q')
        matout=A0.dot(r).view(type(mps))

        return tensor,matout


def orthonormalizeQR(mps,direction,site1,site2):
    
    """
    uses QR decomposition to orthonormalize an mps from site1 to site2; 
    routine returns the last computed "r" matrix from the QR decomposition
    
    for direction>0 and site2<len(mps)-1 the mps is not altered
    for direction>0 and site2=len(mps)-1, the original state is recovered by 
    contracting "r" into mps from the right 
    
    for direction<0 and site1>0 the mps is not altered
    for direction<0 and site1=0, the original state is recovered by 
    contracting "r" into mps from the left
    
    """
    
    assert(direction!=0),'do NOT use direction=0!'
    assert(site2<len(mps))
    assert(site1>=0)
    N=len(mps)
    Z=1
    if direction>0:
        for site in range(site1,site2+1):
            tensor,r,Z_=prepareTensor(mps[site],direction)
            mps[site]=np.copy(tensor)
            Z*=Z_
            if site<(N-1):
                mps[site+1]=np.tensordot(r,mps[site+1],([1],[0]))

    if direction<0:
        for site in range(site2,site1-1,-1):
            tensor,r,Z_=prepareTensor(mps[site],direction)
            mps[site]=np.copy(tensor)
            Z*=Z_            
            if site>0:
                mps[site-1]=np.transpose(np.tensordot(mps[site-1],r,([1],[0])),(0,2,1))
    return r*Z

def initializeLayer(A,x,B,mpo,direction):
    """
    takes two mps matrices A,B, a density matrix x (can be either left or
    right eigenmatrix of the Transfer Matrix), a boundary mpo of either left
    or right type and a direction argument dir > or < 0; if dir > 0, x is
    assumed to be a left vector, if dir<0, then it's assumed to
    be a right vector. returns a three index object (l,l',b) by contracting the input
    l and l' are upper and lower (A and B) matrix indices, and b is the left or right auxiliary index of the mpo
    """
    
    [chi1,chi2,d]=np.shape(A)
    [chi1_,chi2_,d_]=np.shape(B);
    [D1,D2,d1,d2]=np.shape(mpo);
    if direction>0:
        t=np.tensordot(np.tensordot(x,A,([0],[0])),np.conj(B),([0],[0]))#that is a tensor of the form l,s,lpime,sprime ???
        return np.reshape(np.tensordot(t,mpo,([1,3],[3,2])),(chi2,chi2_,D2))

    if direction<0:
        t=np.tensordot(np.tensordot(x,A,([0],[1])),np.conj(B),([0],[1]))#that is a tensor of the form l,s,lpime,sprime ???
        return np.reshape(np.tensordot(t,mpo,([1,3],[3,2])),(chi1,chi1_,D1))

def addLayer(E,A,mpo,B,direction):

    """
    adds an mps-mpo-mps layer to a left or right block "E"; used in dmrg to calculate the left and right
    environments
    E: a tensor of shape (D1,D1',M1) (for direction>0) or (D2,D2',M2) (for direction>0)
    A: (D1,D2,d) shaped mps tensor
    mpo: a tensor of dimension (M1,M2,d,d')
    B: (D1',D2',d') shaped mps tensor
    direction (int): if >0 add a layer to the right of "E", if <0 add a layer to the left of "E"
    returns: E' of shape (D2,D2',M2) for direction>0
             E' of shape (D1,D1',M1) for direction<0
    """

    if direction>0:
        return np.transpose(np.tensordot(np.tensordot(np.tensordot(E,A,([0],[0])),mpo,([1,3],[0,3])),np.conj(B),([0,3],[0,2])),(0,2,1))
    if direction<0:
        return np.transpose(np.tensordot(np.tensordot(np.tensordot(E,A,([0],[1])),mpo,([3,1],[3,1])),np.conj(B),([3,0],[2,1])),(0,2,1))
        
def addELayer(E,A,B,direction):
    
    """
    adds an mps-mps layer to a left or right block; used in dmrg to calculate the left and right
    environments
    E: a tensor of shape (D1,D1',M) (for direction>0) or (D2,D2',M) (for direction>0)
    A: (D1,D2,d) shaped mps tensor
    B: (D1',D2',d) shaped mps tensor
    direction (int): if >0 add a layer to the right of "E", if <0 add a layer to the left of "E"
    returns: E' of shape (D2,D2',M) for direction>0
             E' of shape (D1,D1',M) for direction<0
    """

    if direction>0:
        return np.transpose(np.tensordot(np.tensordot(E,A,([0],[0])),np.conj(B),([0,3],[0,2])),(1,2,0))
    if direction<0:
        return np.transpose(np.tensordot(np.tensordot(A,E,([1],[0])),np.conj(B),([1,2],[2,1])),(0,2,1))



#L is a chi,chi,D tensor, where D is the MPO bond dimension
#mps is a single mps matrix
#mpo is a single mpo matrix
#the order fo the returned vector is correct, i.e. no fortran like ordering
def HAproductSingleSite(L,mpo,R,vec):
    #print 'calling HAproduct'
    chi1=np.shape(L)[0]
    chi2=np.shape(R)[0]
    d=np.shape(mpo)[2]
    mps=np.reshape(vec,(chi1,chi2,d))#,order='f')
    term1=np.tensordot(L,mps,([0],[0]))
    term2=np.tensordot(term1,mpo,([1,3],[0,3]))
    term3=np.tensordot(term2,R,([1,2],[0,2]))
    return np.reshape(np.transpose(term3,[0,2,1]),chi1*chi2*d)


#L is a chi,chi,D tensor, where D is the MPO bond dimension
#mps is a single mps matrix
#mpo is a single mpo matrix
#the order fo the returned vector is correct, i.e. no fortran like ordering
def HAproductSingleSiteMPS(L,mpo,R,mps):
    term1=np.tensordot(L,mps,([0],[0]))
    term2=np.tensordot(term1,mpo,([1,3],[0,3]))
    term3=np.tensordot(term2,R,([1,2],[0,2]))
    return np.transpose(term3,[0,2,1])

def HAproductZeroSite(L,mpo,mps,R,position,vec):
    #contracts vec on the right site of mps
    if position=='right':
        chi1=np.shape(mps)[1]
        mat=np.reshape(vec,(chi1,chi1))
        L1=addLayer(L,mps,mpo,mps,1)
        term1=np.tensordot(L1,mat,([0],[0]))
        return np.reshape(np.tensordot(term1,R,([2,1],[0,2])),chi1*chi1)

    #contracts vec on the left site of mps
    if position=='left':
        chi1=np.shape(mps)[1]
        mat=np.reshape(vec,(chi1,chi1))
        R1=addLayer(R,mps,mpo,mps,-1)
        term1=np.tensordot(L,mat,([0],[0]))
        return np.reshape(np.tensordot(term1,R1,([2,1],[0,2])),chi1*chi1)

#the matrix version of the above function
def HAproductZeroSiteMat(L,mpo,mps,R,position,mat):
    if position=='right':
        chi1=np.shape(mps)[1]
        L1=addLayer(L,mps,mpo,mps,1)
        term1=np.tensordot(L1,mat,([0],[0]))
        return np.tensordot(term1,R,([2,1],[0,2]))

    if position=='left':
        chi1=np.shape(mps)[1]
        R1=addLayer(R,mps,mpo,mps,-1)
        term1=np.tensordot(L,mat,([0],[0]))
        return np.tensordot(term1,R1,([2,1],[0,2]))


#does the matrix vector product on the bond; vec is a vectorized bond matrix
def HAproductBond(L,R,vec):
    chi=np.shape(L)[0]
    mat=np.reshape(vec,(chi,chi))
    temp=np.tensordot(L,mat,([0],[0]))
    return np.reshape(np.tensordot(temp,R,([2,1],[0,2])),chi*chi)


def contractionE(Lexp,mps,Rexp):
    term1=np.tensordot(Rexp,mps,([0],[1]))
    return np.tensordot(Lexp,term1,([0,2],[2,1]))


#the matrix version of the above routine 
def HAproductBondMatrix(L,R,matrix):
    chi=np.shape(L)[0]
    temp=np.tensordot(L,matrix,([0],[0]))
    return np.tensordot(temp,R,([2,1],[0,2]))



def getR(mps,mpo,rightboundary):
    """
    calculates all R blocks; make sure the mps is right orthogonal
    rightboundary if the right boundary to which mps-mpo-mps layers
    are attached
    returns a list of blocks (np.ndarray)
    """
    
    R=[]
    N=len(mps)
    R.append(addLayer(rightboundary,mps[-1],mpo[-1],mps[-1],-1))
    for n in range(1,N):
        R.append(addLayer(R[n-1],mps[N-1-n],mpo[N-1-n],mps[N-1-n],-1))
    return R

def getL(mps,mpo,leftboundary):
    
    """
    calculates all L blocks; make sure the mps is left orthogonal
    leftboundary if the left boundary to which mps-mpo-mps layers
    are attached
    returns a list of blocks (np.ndarray)
    """
    
    L=[]
    N=len(mps)
    L.append(addLayer(leftboundary,mps[0],mpo[0],mps[0],1))
    for n in range(1,N):
        L.append(addLayer(L[n-1],mps[n],mpo[n],mps[n],1))
    return L


def getUCR(mps,mpo,rightboundary):
    """
    same as getR, but does not store all intermediate blocks. Instead, it returns
    the left-most block obtained after contracting everything
    """
    
    N=len(mps)
    R=addLayer(rightboundary,mps[-1],mpo[-1],mps[-1],-1)
    for n in range(1,N):
        R=addLayer(R,mps[N-1-n],mpo[N-1-n],mps[N-1-n],-1)
    return R

def getUCL(mps,mpo,leftboundary):
    
    """
    same as getL, but does not store all intermediate blocks. Instead, it returns
    the right-most block obtained after contracting everything
    """

    N=len(mps)
    L=addLayer(leftboundary,mps[0],mpo[0],mps[0],1)
    for n in range(1,N):
        L=addLayer(L,mps[n],mpo[n],mps[n],1)
    return L



def getRprojected(mps,mpo,rightboundary,ldens,rdens):
    
    """
    calculates all R blocks and projects them onto the subspace orthogonal to |r(n))(l(n)|
    at each site n of the lattice. make sure the mps is right orthogonal 
    rightboundary: the right boundary to which mps-mpo-mps layers are attached
    ldens: left reduced density matrices at sites n of the lattice (in matrix form). ldens[0] and ldens[N] should 
    be identical (the left unit-cell steady reduced density matrix). ldens[n] is located 
    between tensors mps[n] and mps[n+1] (for n<N-1)
    
    rdens: right reduced density matrices at sites n of the lattice (in matrix form). rdens[0] and rdens[N] should 
    be identical (the right unit-cell steady reduced density matrix). rdens[n] is located 
    between tensors mps[n] and mps[n+1] (for n<N-1)
    
    returns a list of blocks (np.ndarray)
    """
    
    R=[]
    N=len(mps)
    R.append(addLayer(rightboundary,mps[-1],mpo[-1],mps[-1],-1))
    lmat=ldens[-2]
    rmat=rdens[-2]
    R[0][:,:,-1]=R[0][:,:,-1]-np.trace(R[0][:,:,-1].dot(lmat))*rmat
    for n in range(1,N):
        R.append(addLayer(R[n-1],mps[N-1-n],mpo[N-1-n],mps[N-1-n],-1))
        lmat=ldens[N-1-n]
        rmat=rdens[N-1-n]
        R[n][:,:,-1]=R[n][:,:,-1]-np.trace(R[n][:,:,-1].dot(lmat))*rmat
    return R


def getLprojected(mps,mpo,leftboundary,ldens,rdens):
    """
    calculates all L blocks and projects them onto the subspace orthogonal to |r(n))(l(n)|
    at each site n of the lattice. make sure the mps is right orthogonal 
    rightboundary: the right boundary to which mps-mpo-mps layers are attached
    ldens: left reduced density matrices at sites n of the lattice (in matrix form). ldens[0] and ldens[N] should 
    be identical (the left unit-cell steady reduced density matrix). ldens[n] is located 
    between tensors mps[n] and mps[n+1] (for n<N-1)
    
    rdens: right reduced density matrices at sites n of the lattice (in matrix form). rdens[0] and rdens[N] should 
    be identical (the right unit-cell steady reduced density matrix). rdens[n] is located 
    between tensors mps[n] and mps[n+1] (for n<N-1)
    
    returns a list of blocks (np.ndarray)
    """
    
    L=[]
    N=len(mps)
    L.append(addLayer(leftboundary,mps[0],mpo[0],mps[0],1))
    rmat=rdens[1]
    lmat=ldens[1]    
    L[0][:,:,0]=L[0][:,:,0]-np.trace(L[0][:,:,0].dot(rmat))*lmat
    for n in range(1,N):
        L.append(addLayer(L[n-1],mps[n],mpo[n],mps[n],1))
        rmat=rdens[n+1]
        lmat=ldens[n+1]
        L[n][:,:,0]=L[n][:,:,0]-np.trace(L[n][:,:,0].dot(rmat))*lmat

    return L



def lanczos(L,mpo,R,mps0,tolerance=1e-6,Ndiag=10,nmax=500,numeig=1,delta=1E-8,deltaEta=1E-5):
    dtype=np.result_type(L,mpo,R,mps0).type
    [chi1,chi2,d]=np.shape(mps0)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    mv=fct.partial(HAproductSingleSite,*[L,mpo,R])
    lan=lanEn.LanczosEngine(mv,np.dot,np.zeros,Ndiag,nmax,numeig,delta,deltaEta)
    e,v,con=lan.simulate(initialstate=np.reshape(mps0,chi1*chi2*d).astype(dtype),verbose=False)
    if numeig==1:
        return e[0],np.reshape(v[0],(chi1p,chi2p,dp))
    else:
        es=[]
        vs=[]
        for n in range(min(len(e),numeig)):
            es.append(e[n])
            vs.append(np.reshape(v[n],(chi1p,chi2p,dp)))
        return es,vs


def lanczosbond(L,mpo,mps,R,mat0,position,Ndiag=10,nmax=500,numeig=1,delta=1E-10,deltaEta=1E-8):
    dtype=np.result_type(L,mpo,mps,R,mat0).type
    [chi1,chi2]=np.shape(mat0)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    mv=fct.partial(HAproductZeroSite,*[L,mpo,mps,R,position])
    lan=lanEn.LanczosEngine(mv,np.dot,np.zeros,Ndiag,nmax,numeig,delta,deltaEta)
    e,v,con=lan.simulate(initialstate=np.reshape(mat0,chi1*chi2).astype(dtype),verbose=False)

    if numeig==1:
        return e[0],np.reshape(v[0],(chi1p,chi2p))
    else:
        es=[]
        vs=[]
        for n in range(min(len(e),numeig)):        
            es.append(e[n])
            vs.append(np.reshape(v[n],(chi1p,chi2p)))
        return es,vs

#L is a chi,chi,D tensor, where D is the MPO bond dimension
#mps is a single mps matrix
#mpo is a single mpo matrix
#the order fo the returned vector is correct, i.e. no fortran like ordering
# def HAproductSingleSite(L,mpo,R,vec):
#     #print 'calling HAproduct'
#     chi1=np.shape(L)[0]
#     chi2=np.shape(R)[0]
#     d=np.shape(mpo)[2]
#     mps=np.reshape(vec,(chi1,chi2,d))#,order='f')
#     term1=np.tensordot(L,mps,([0],[0]))
#     term2=np.tensordot(term1,mpo,([1,3],[0,3]))
#     term3=np.tensordot(term2,R,([1,2],[0,2]))
#     return np.reshape(np.transpose(term3,[0,2,1]),chi1*chi2*d)

def contractionHelper(L,mpo,R,mps):
    return np.reshape(ncon.ncon([L,np.reshape(mps,(L.shape[0],R.shape[0],mpo.shape[2])),mpo,R],[[1,-1,3],[1,4,2],[3,5,-3,2],[4,-2,5]]),mps.shape)

def eigsh(L,mpo,R,mps0,tolerance=1e-6,numvecs=1,numcv=10,numvecs_returned=1):
    """
    sparse diagonalization of DMRG local hamiltonian
    L:    Tensor object of shape (Dl,Dl',Ml)
    mpo:  Tensor object of shape (Ml,Mr,d,d')
    L:    Tensor object of shape (Dr,Dr',Mr)
    mps0: Tensor object of shape (Dl,Dd,d)
    see scipy eigsh documentation for details on the other parameters
    """
    dtype=np.result_type(L.dtype,mpo.dtype,R.dtype,mps0.dtype)
    chil=np.sum(L.shape[0])
    chir=np.sum(R.shape[0])
    chilp=np.sum(L.shape[1])
    chirp=np.sum(R.shape[1])
    d=mpo.shape[2]
    dp=mpo.shape[3]

    mv=fct.partial(contractionHelper,*[L,mpo,R])
    LOP=LinearOperator((chil*chir*d,chilp*chirp*dp),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=1000000,tol=tolerance,v0=mps0.merge([0,1,2]),ncv=numcv)

    if numvecs_returned==1:
        ind=np.nonzero(e==min(e))
        return e[ind[0][0]],np.reshape(v[:,ind[0][0]],(chilp,chirp,dp))        

    elif numvecs_returned>1:
        if (numvecs_returned>numvecs):
            warnings.warn('mpsfunctions.eigsh: requestion to return more vectors than calcuated: setting numvecs_returned=numvecs',stacklevel=2)
            numvecs_returned=numvecs
        es=[]
        vs=[]
        esorted=np.sort(e)
        for n in range(numvecs_returned):
            es.append(esorted[n])
            ind=np.nonzero(e==esorted[n])
            vs.append(np.reshape(v[:,ind[0][0]],(chilp,chirp,dp)))
        return es,vs

def lobpcg(L,mpo,R,mps0,tolerance=1e-6,numvecs_returned=1):
    
    """
    calls a sparse eigensolver to find the lowest eigenvalues and eigenvectors
    of the4 effective DMRG hamiltonian as given by L, mpo and R
    L (np.ndarray of shape (Dl,Dl',d)): left hamiltonian environment
    R (np.ndarray of shape (Dr,Dr',d)): right hamiltonian environment
    mpo (np.ndarray of shape (Ml,Mr,d)): MPO
    mps0 (np.ndarray of shape (Dl,Dr,d)): initial MPS tensor for the arnoldi solver
    see scipy eigsh documentation for details on the other parameters
    """
    if not isinstance(mps0,tnsr.Tensor):
        raise TypeError('lobpcg currently only implemented for np.ndarray or derived types')
    dtype=np.result_type(L.dtype,mpo.dtype,R.dtype,mps0.dtype)
    chil=np.sum(L.shape[0])
    chir=np.sum(R.shape[0])
    chilp=np.sum(L.shape[1])
    chirp=np.sum(R.shape[1])
    d=mpo.shape[2]
    dp=mpo.shape[3]

    mv=fct.partial(contractionHelper,*[L,mpo,R])    
    LOP=LinearOperator((chil*chir*d,chilp*chirp*dp),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    X=mps0.random((chil*chir*d,numvecs_returned),dtype=dtype)
    X[:,0]=mps0.merge([0,1,2])
    e,v=sp.sparse.linalg.lobpcg(LOP,X=X,largest=False)

    if numvecs_returned==1:
        ind=np.nonzero(e==min(e))
        return e[ind[0][0]],np.reshape(v[:,ind[0][0]],(chi1p,chi2p,dp))

    elif numvecs_returned>1:
        if (numvecs_returned>numvecs):
            warnings.warn('mpsfunctions.eigsh: requestion to return more vectors than calcuated: setting numvecs_returned=numvecs',stacklevel=2)
            numvecs_returned=numvecs
        es=[]
        vs=[]
        esorted=np.sort(e)
        for n in range(numvecs_returned):
            es.append(esorted[n])
            ind=np.nonzero(e==esorted[n])
            vs.append(np.reshape(v[:,ind[0][0]],(chi1p,chi2p,dp)))
        return es,vs


def evolveTensorSexpmv(L,mpo,R,mps,tau):
    dtype=np.result_type(L,mpo,R,mps,tau).type
    [chi1,chi2,d]=np.shape(mps)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    if np.issubdtype(type(tau),np.complexfloating):
        fac=np.exp(1j*np.angle(tau))
        dt=np.abs(tau)
    elif np.issubdtype(type(tau),np.floating):
        dt=tau
        fac=1.0        
    else:
        raise TypeError("evolveTensorSexpmv: unkown type {0} for tau".format(type(tau)))

    mv=fct.partial(HAproductSingleSite,*[L,fac*mpo,R])
    LOP=LinearOperator((chi1*chi2*d,chi1*chi2*d),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    v, conv, nstep, ibrkflag,mbrkdwn=sexpmv.gexpmv(LOP, np.reshape(mps,chi1*chi2*d).astype(dtype), dt, anorm=1.0)
    return np.reshape(v,mps.shape)

def evolveMatrixSexpmv(L,R,mat,tau):

    dtype=np.result_type(L,R,mat,tau).type
    if np.issubdtype(type(tau),np.complexfloating):    
        fac=np.exp(1j*np.angle(tau))
        dt=np.abs(tau)
    elif np.issubdtype(type(tau),np.floating):                
        dt=tau
        fac=1.0        
    else:
        raise TypeError("evolveMatrixSexpmv: unkown type {0} for tau".format(type(tau)))        


    [chi1,chi2]=np.shape(mat)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    mv=fct.partial(HAproductBond,*[L,fac*R])
    LOP=LinearOperator((chi1*chi2,chi1*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)    
    v, conv, nstep, ibrkflag,mbrkdwn=sexpmv.gexpmv(LOP, np.reshape(mat,chi1*chi2).astype(dtype), dt, anorm=1.0)
    return np.reshape(v,mat.shape)

    
def evolveTensorLan(L,mpo,R,mps,tau,krylov_dimension=20,delta=1E-8):
    [chi1,chi2,d]=np.shape(mps)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    
    mv=fct.partial(HAproductSingleSite,*[L,mpo,R])
    lan=lanEn.LanczosTimeEvolution(mv,np.dot,np.zeros,ncv=krylov_dimension,delta=delta)
    v=lan.__doStep__(np.reshape(mps,(chi1*chi2*d)),tau)
    return np.reshape(v,mps.shape)

def evolveMatrixLan(L,R,mat,tau,krylov_dimension=20,delta=1E-8):
    [chi1,chi2]=np.shape(mat)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    mv=fct.partial(HAproductBond,*[L,R])
    lan=lanEn.LanczosTimeEvolution(mv,np.dot,np.zeros,ncv=krylov_dimension,delta=delta)
    v=lan.__doStep__(np.reshape(mat,(chi1*chi2)),tau)
    return np.reshape(v,mat.shape)

def evolveTensorsolve_ivp(L,mpo,R,mps,tau,method='RK45',rtol=1E-8,atol=1E-12):
    if (np.real(tau)!=0.0) and (np.imag(tau)!=0.0):
        raise TypeError(" evolveTensorsolve_ivp(L,mpo,R,mps,tau,rtol=1E-3,atol=1E-6): tau has to be either real or imaginary")
    try:    
        tau_=np.real(tau)+np.imag(tau)
        [chi1,chi2,d]=np.shape(mps)
        chi1p=np.shape(L)[1]
        chi2p=np.shape(R)[1]
        dp=np.shape(mpo)[3]
        mv=fct.partial(HAproductSingleSite,*[L,mpo,R])
        if np.real(tau)==0.0:
            out=sp.integrate.solve_ivp(lambda t,y:1j*mv(y),t_span=(0.0,tau_),y0=np.reshape(mps,(chi1*chi2*d)),method=method,rtol=rtol,atol=atol)
        else:
            out=sp.integrate.solve_ivp(lambda t,y:mv(y),t_span=(0.0,tau_),y0=np.reshape(mps,(chi1*chi2*d)),method=method,rtol=rtol,atol=atol)            
        return np.reshape(out.y[:,-1],mps.shape)        
    except AttributeError as err:
        if StrictVersion(sp.__version__)<StrictVersion('1.1.0'):
            warnings.warn('in evolveTensorsolve_ivp: RK45 and RK23 solvers are only available for scipy versions >= 1.1.0',stacklevel=2)
            raise err
        else:
            raise err

def evolveMatrixsolve_ivp(L,R,mat,tau,method='RK45',rtol=1E-3,atol=1E-6):
    if (np.real(tau)!=0.0) and (np.imag(tau)!=0.0):
        raise TypeError(" evolveTensorsolve_ivp(L,mpo,R,mps,tau,rtol=1E-3,atol=1E-6): tau has to be either real or imaginary")
    try:
        tau_=np.real(tau)+np.imag(tau)        
        [chi1,chi2]=np.shape(mat)
        chi1p=np.shape(L)[1]
        chi2p=np.shape(R)[1]
        mv=fct.partial(HAproductBond,*[L,R])
        if np.real(tau)==0.0:        
            out=sp.integrate.solve_ivp(lambda t,y:1j*mv(y),t_span=(0.0,tau_),y0=np.reshape(mat,(chi1*chi2)),method=method,rtol=rtol,atol=atol)
        else:
            out=sp.integrate.solve_ivp(lambda t,y:mv(y),t_span=(0.0,tau_),y0=np.reshape(mat,(chi1*chi2)),method=method,rtol=rtol,atol=atol)            
        return np.reshape(out.y[:,-1],mat.shape)
    except AttributeError as err:
        if StrictVersion(sp.__version__)<StrictVersion('1.1.0'):
            warnings.warn('in evolveMatrixsolve_ivp: RK45 and RK23 solvers are only available for scipy versions >= 1.1.0',stacklevel=2)
            raise err
        else:
            raise err


#calls a sparse eigensolver to find the lowest eigenvalue
#takes only L and R blocks, and finds the ground state
def eigshbondsimple(L,R,mat0,tolerance=1e-6,numvecs=1,numcv=10):
    [chi1,chi2]=np.shape(mat0)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    mv=fct.partial(HAproductBond,*[L,R])
    LOP=LinearOperator((chi1*chi2,chi1*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=mat0.dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=np.reshape(mat0,chi1*chi2),ncv=numcv)
    return [e,np.reshape(v,(chi1p,chi2p))]

#calls a sparse eigensolver to find the lowest eigenvalue 
def eigshbond(L,mpo,mps,R,mat0,position,tolerance=1e-6,numvecs=1,numcv=10,numvecs_returned=1):
    dtype=np.result_type(L,mpo,mps,R,mat0).type
    #if position=='right':
    [chi1,chi2]=np.shape(mat0)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    mv=fct.partial(HAproductZeroSite,*[L,mpo,mps,R,position])
    LOP=LinearOperator((chi1*chi2,chi1*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=np.reshape(mat0,chi1*chi2),ncv=numcv)

    if numvecs_returned==1:
        ind=np.nonzero(e==min(e))
        #return [e,np.reshape(v,(chi1p,chi2p,dp))]
        return e[ind[0][0]],np.reshape(v[:,ind[0][0]],(chi1p,chi2p))
    elif numvecs_returned>1:
        if (numvecs_returned>numvecs):
            warnings.warn('mpsfunctions.eigsh: request to return more vectors than calcuated: setting numvecs_returned=numvecs',stacklevel=2)
            numvecs_returned=numvecs
            es=[]
            vs=[]
            esorted=np.sort(e)
            for n in range(numvecs_returned):
                es.append(esorted[n])
                ind=np.nonzero(e==esorted[n])
                vs.append(np.reshape(v[:,ind[0][0]],(chi1p,chi2p)))
        return es,vs
        
    #if position=='left':
    #    [chi1,chi2]=np.shape(mat0)
    #    chi1p=np.shape(L)[1]
    #    chi2p=np.shape(R)[1]
    #    dp=np.shape(mpo)[3]
    #    mv=fct.partial(HAproductZeroSite,*[L,mpo,mps,R,position])
    #    LOP=LinearOperator((chi1*chi2,chi1*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    #    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=100000,tol=tolerance,v0=np.reshape(mat0,chi1*chi2),ncv=numcv)
    #
    #    if numvecs_returned==1:
    #        ind=np.nonzero(e==min(e))
    #        #return [e,np.reshape(v,(chi1p,chi2p,dp))]
    #        return e[ind[0][0]],np.reshape(v[:,ind[0][0]],(chi1p,chi2p))
    #    elif numvecs_returned>1:
    #        if (numvecs_returned>numvecs):
    #            sys.warning('mpsfunctions.eigsh: request to return more vectors than calcuated: setting numvecs_returned=numvecs',stacklevel=2)
    #            numvecs_returned=numvecs
    #            es=[]
    #            vs=[]
    #            esorted=np.sort(e)
    #            for n in range(numvecs_returned):
    #                es.append(esorted[n])
    #                ind=np.nonzero(e==esorted[n])
    #                vs.append(np.reshape(v[:,ind[0][0]],(chi1p,chi2p)))
    #        return es,vs


        #return [e,np.reshape(v,(chi1p,chi2p))]

#this function assumes that mps0has the usual cMPS form, with only two entries
#the incoming mpo HAS TO BE OF A PROJECTED FORM
def applyHsingleSite(L,mpo,R,mps):
    assert(np.shape(mps)[2]==2)
    [chi1,chi2,d]=np.shape(mps)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    vec=np.reshape(mps,chi1*chi2*d)
    dvec_dt=HAproductSingleSite(L,mpo,R,vec)
    return np.reshape(dvec_dt,(chi1p,chi2p,dp))

def applyHbond(L,R,mat):
    term1=np.tensordot(L,mat,([0],[0]))
    return np.tensordot(term1,R,([2,1],[0,2]))

#calculate the local gradient at mps0
def gradient(L,mpo,R,mps0):
    [chi1,chi2,d]=np.shape(mps0)
    chi1p=np.shape(L)[1]
    chi2p=np.shape(R)[1]
    dp=np.shape(mpo)[3]
    mv=fct.partial(HAproductSingleSite,*[L,mpo,R])
    return mv(mps0)


def TransferOperator(direction,mps,vector):
    """
    computes the transfer matrix vector product 
    Parameters:
    --------------------------
    mps:       ndarray or MPS object of shape(D1,D2,d)
    direction: int
               direction > 0 does a left-side product; direction < 0 does a right side product;
    vector:  ndarray 
             for direction > 0: shape of vector can b (D1,D1) or (D1*D1) 
             for direction < 0: shape of vector can b (D2,D2) or (D2*D2) 
             the index convention is that for either left (D1,D1) or right (D2,D2) ndarrays, the index 0 is on the unconjugated side the transfer-matrix
             
    Returns: 
    -------------------------
    np.ndarray of shape vector.shape
    a vector obtained from the contracting the input vector with the transfer operator
    """
    if isinstance(mps,np.ndarray):
        A=mps
        [D1,D2,d]= A.shape
    elif isinstance(mps,MPSL.MPS):
        if len(A)>1:
            raise ValueError("in TransferOperator: got an MPS object of len(MPS)>1; can only handle single-site objects")
        A=mps[0]
        [D1,D2,d]= A.shape
    else:
        raise TypeError("TransferOperator: got an unknown type for mps")
    if direction>0:
        x=np.reshape(vector,(D1,D1))
        if len(vector.shape)==1:
            return np.reshape(np.tensordot(np.tensordot(x,A,([0],[0])),np.conj(A),([0,2],[0,2])),D2*D2)
        elif len(vector.shape)==2:
            return np.tensordot(np.tensordot(x,A,([0],[0])),np.conj(A),([0,2],[0,2]))
    if direction<0:
        x=np.reshape(vector,(D2,D2))
        if len(vector.shape)==1:
            return np.reshape(np.tensordot(np.tensordot(x,A,([0],[1])),np.conj(A),([0,2],[1,2])),D1*D1)
        elif len(vector.shape)==2:        
            return np.tensordot(np.tensordot(x,A,([0],[1])),np.conj(A),([0,2],[1,2]))
def MixedTransferOperator(direction,mpsA,mpsB,vector):

    """
    computes the mixed transfer-matrix vector product 
    mpsA/mpsA: ndarrays or MPS objects of shape(D1,D2,d); mpsA is the unconjugated tensor and mpsB is the conjugated tensor of the transfer operator
    direction: int, direction > 0 does a left-side product; direction < 0 does a right side product;
    vector:  ndarray of shape (D1,D1) (direction > 0) or (D2,D2)  (direction < 0), reshaped into vector format
             the index convention is that for either left  (D1,D1) or right (D2,D2) ndarrays, the leg 0 is on the unconjugated side the transfer-matrix,
             
    returns: a vector obtained from the contracting the input vector with the transfer operator
    """
    
    if isinstance(mpsA,np.ndarray) and isinstance(mpsB,np.ndarray):
        Upper=mpsA
        Lower=mpsB
        [D1A,D2A,dA]= Lower.shape
        [D1B,D2B,dB]= Upper.shape
        
    elif isinstance(mpsA,MPSL.MPS) and isinstance(mpsB,MPSL.MPS):
        if len(A)>1:
            raise ValueError("in TransferOperator: got an MPS object mpsA of len(mpsA)>1; can only handle single-site objects")
        if len(B)>1:
            raise ValueError("in TransferOperator: got an MPS object mpsB of len(mpsB)>1; can only handle single-site objects")
        Upper=mpsA[0]
        Lower=mpsB[0]
        [D1A,D2A,dA]= Lower.shape
        [D1B,D2B,dB]= Upper.shape
        
    elif isinstance(mpsA,MPSL.MPS) and isinstance(mpsB,np.ndarray):
        if len(A)>1:
            raise ValueError("in TransferOperator: got an MPS object mpsA of len(mpsA)>1; can only handle single-site objects")
        Upper=mpsA[0]
        Lower=mpsB
        [D1A,D2A,dA]= Lower.shape
        [D1B,D2B,dB]= Upper.shape
    elif isinstance(mpsB,MPSL.MPS) and isinstance(mpsA,np.ndarray):
        if len(B)>1:
            raise ValueError("in TransferOperator: got an MPS object mpsA of len(mpsA)>1; can only handle single-site objects")
        Upper=mpsA
        Lower=mpsB[0]
        [D1A,D2A,dA]= Lower.shape
        [D1B,D2B,dB]= Upper.shape
        
    else:
        raise TypeError("TransferOperator: got an unknown type for mps")
    
    [D1A,D2A,dA]= Lower.shape
    [D1B,D2B,dB]= Upper.shape
    if direction>0:
        x=np.reshape(vector,(D1B,D1A))
        return np.reshape(np.tensordot(np.tensordot(x,Upper,([0],[0])),np.conj(Lower),([0,2],[0,2])),(D2A*D2B))
    if direction<0:
        x=np.reshape(vector,(D2B,D2A))
        return np.reshape(np.tensordot(np.tensordot(x,Upper,([0],[1])),np.conj(Lower),([0,2],[1,2])),(D1A*D1B))

def GeneralizedMatrixVectorProduct(direction,A,B,vector):
    """    
    defines the matrix vector product to find the largest eigenvalue of the transfermatrix;
    this function will later on be turned into a functools.partial object mv(v) using partial(GeneralizedMatrixVectorProduct,[tensor,direction])
    A is the lower, B is the upper matrix 
    """

    warnings.warn('GeneralizedMatrixVectorProduct(direction,A,B,vector) is deprecated; used MixedTransferOperator instead')
    return MixedTransferOperator(direction=direction,mpsA=B,mpsB=A,vector=vector)

def TMeigs(tensor,direction,numeig=6,init=None,nmax=6000,tolerance=1e-10,ncv=100,which='LR'):
    """
    sparse computation of the dominant left or right eigenvector of the transfer matrix, using the TransferOperator function to
    to do the matrix-vector multiplication
    tensor:     ndarray or MPS object of length 1
                the mps tensor
    direction:  int 
                direction>0 gives the left eigenvector, direction<0 gives the right eigenvector
    numeig:     int 
                hyperparameter, number of eigenvectors to be calculated by argpack; note that numeig is not the 
                number of returned eigenvectors, the number of returned eigenvectors is always one; numeig influences the arpack solver
                due to a bug in lapack/arpack, and should usually be chosen >4 for stability reasons
    init:       None or np.ndarray of shape D**2
                an initial guess for the eigenvector
    nmax:       int
                maximum numbers of iteration in eigs
    tolerance:  float
                precision of the eigen-values
    ncv:        int
                number of krylov vectors in eigs
    which:      str, any of {'LR','SM','LA','SA'}
                which eigenvector to calculate; use 'LR' or 'SM' 
    """

    dtype=tensor.dtype
    #define the matrix vector product mv(v) using functools.partial and GeneralizedMatrixVectorProduct(direction,A,B,vector):
    [chi1,chi2,d]=np.shape(tensor)
    if chi1!=chi2:
        raise ValueError(" in TMeigs: ancillary dimensions of the MPS tensor have to be the same on either side")
    mv=fct.partial(TransferOperator,*[direction,tensor])
    LOP=LinearOperator((chi1*chi1,chi2*chi2),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    if numeig>=LOP.shape[0]-1:
        warnings.warn('TMeigs: numeig+1 ({0}) > dimension of transfer operator ({1}) changing value to numeig={2}'.format(numeig+1,LOP.shape[0],LOP.shape[0]-2))
        numeig=LOP.shape[0]-2

    try:
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
        while np.abs(np.imag(eta[m]))/np.abs(np.real(eta[m]))>1E-5:
            print ('found TM eigenvalue eta ={0} with large imaginary part (ARPACK BUG); recalculating with a new initial state'.format(eta))
            eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which='LR',v0=np.random.rand(chi2*chi2),maxiter=nmax,tol=tolerance,ncv=ncv)
            m=np.argmax(np.real(eta))

        if np.issubdtype(dtype,np.floating):
            out=np.reshape(vec[:,m],chi1*chi1)
            if np.linalg.norm(np.imag(out))>1E-6:
                raise TypeError("TMeigs: dtype was float, but returned eigenvector had a large imaginary part; something went wrong here!")
            return np.real(eta[m]),np.real(out),numeig
        if np.issubdtype(dtype,np.complexfloating):        
            return eta[m],np.reshape(vec[:,m],chi1*chi1),numeig
            
        return eta[m],np.reshape(vec[:,m],chi2*chi2),numeig

    except ArpackError:
        print ('Arpack just threw an exception .... ' )
        return TMeigs(tensor,direction,numeig,np.random.rand(chi1*chi2),nmax,tolerance,ncv,which)
        

def UnitcellTransferOperator(direction,mps,vector):
    """
    computes the transfer matrix vector product 
    mps: list of ndarray or MPS object
    direction: int
               direction > 0 does a left-side product; direction < 0 does a right side product;
    vector:  ndarray of shape (D1,D1) (direction > 0) or (D2,D2)  (direction < 0), reshaped into vector format
             the index convention is that for either left  (D1,D1) or right (D2,D2) ndarrays, the leg 0 is on the unconjugated side the transfer-matrix,
             
    returns: a vector obtained from the contracting the input vector with the utni-cell transfer operator
    """
    
    #return MixedUnitcellTransferOperator(direction,mps,mps,vector)
    [D1l,D2l,dl]= np.shape(mps[0])
    [D1r,D2r,dr]= np.shape(mps[len(mps)-1])
    if D1l!=D2r:
        raise ValueError("UnitcellTransferOperator: mps[0] and mps[-1] have non-matching dimensions")
    x=np.copy(vector)
    if direction>0:
        x=np.reshape(vector,(D1l,D1l))
        for n in range(len(mps)):
            x=np.tensordot(np.tensordot(x,mps[n],([0],[0])),np.conj(mps[n]),([0,2],[0,2]))
        return np.reshape(x,(D2r*D2r))
    if direction<0:
        x=np.reshape(vector,(D2r,D2r))
        for n in range(len(mps)-1,-1,-1):
            x=np.tensordot(np.tensordot(x,mps[n],([0],[1])),np.conj(mps[n]),([0,2],[1,2]))
        return np.reshape(x,(D1l*D1l))
    

def MixedUnitcellTransferOperator(direction,Upper,Lower,vector):

    """
    computes the mixed transfer-matrix vector product 
    mps: list of ndarray or MPS object
    direction: int
               direction > 0 does a left-side product; direction < 0 does a right side product;
    vector:  ndarray of shape (D1,D1) (direction > 0) or (D2,D2)  (direction < 0), reshaped into vector format
             the index convention is that for either left  (D1,D1) or right (D2,D2) ndarrays, the leg 0 is on the unconjugated side the transfer-matrix,
             
    returns: a vector obtained from the contracting the input vector with the utni-cell transfer operator
    """
    
    if len(Upper)!=len(Lower):
        raise ValueError("MixedUnitcellTransferOperator: Upper and Lower mps have different lengths")
    
    [D1l,D2l,dl]= np.shape(Upper[0])
    [D1l_,D2l_,dl_]= np.shape(Lower[0])
    
    [D1r,D2r,dr]= np.shape(Upper[len(Upper)-1])
    [D1r_,D2r_,dr_]= np.shape(Lower[len(Lower)-1])    
    if D1l!=D2r:
        raise ValueError("MixedUnitcellTransferOperator: Upper[0] and Upper[-1] have non-matching dimensions")
    if D1l_!=D2r_:
        raise ValueError("MixedUnitcellTransferOperator: Lower[0] and Lower[-1] have non-matching dimensions")
    
    x=np.copy(vector)
    if direction>0:
        x=np.reshape(vector,(D1l,D1l_))
        for n in range(len(Upper)):
            x=np.tensordot(np.tensordot(x,Upper[n],([0],[0])),np.conj(Lower[n]),([0,2],[0,2]))
        return np.reshape(x,(D2r*D2r_))
    if direction<0:
        x=np.reshape(vector,(D2r,D2r_))
        for n in range(len(Upper)-1,-1,-1):
            x=np.tensordot(np.tensordot(x,Upper[n],([0],[1])),np.conj(Lower[n]),([0,2],[1,2]))
        return np.reshape(x,(D1l*D1l_))



def UnitcellTMeigs(mps,direction,numeig=6,init=None,nmax=800,tolerance=1e-12,ncv=10,which='LM'):
    """
    sparse computation of the dominant left or right eigenvector of the transfer matrix
    Parameters:
    -----------------------------------------------
    mps:        list of np.ndarray, or MPS object
                the mps tensor
    direction:  int 
                direction>0 gives the left eigenvector, direction<0 gives the right eigenvector
    numeig:     int 
                hyperparameter, number of eigenvectors to be calculated by argpack; note that numeig is not the 
                number of returned eigenvectors, the number of returned eigenvectors is always one; numeig influences the arpack solver
                due to a bug in lapack/arpack, and should usually be chosen >4 for stability reasons
    init:       None or np.ndarray of shape D**2
                an initial guess for the eigenvector
    nmax:       int
                maximum numbers of iteration in eigs
    tolerance:  float
                precision of the eigen-values
    ncv:        int
                number of krylov vectors in eigs
    which:      str, any of {'LR','SM','LA','SA'}
                which eigenvector to calculate; use 'LR' or 'SM' 

    Returns:
    ---------------------------------
    tuple (eta,vec,numeig)
    eta: float or complex
         eigenvalue
    vec: np.ndarray of shape=(mps[0].shape[0]**2,) (for direction < 0) or shape=(mps[-1].shape[1]**2,) (for direction > 0)
         dominant eigenvector of the transfer operator
    numeig: int
            the number of eigenvectors to be calculated in the sparse solver (hyperparameter, can usually be ignored)
    """


    if isinstance(mps,MPSL.MPS):
        dtype=mps.dtype
    elif isinstance(mps,list):
        dtype=np.result_type(*mps).type

    [D1l,D2l,dl]=np.shape(mps[0])
    [D1r,D2r,dr]=np.shape(mps[-1])
    if D1l!=D2r:
        raise ValueError(" in UnitcellTMeigs: ancillary dimensions of the MPS tensor have to be the same on left and right side")
    mv=fct.partial(UnitcellTransferOperator,*[direction,mps])

    LOP=LinearOperator((D1l*D1l,D2r*D2r),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    if numeig>=LOP.shape[0]-1:
        warnings.warn('UnitcellTMeigs: numeig+1 ({0}) > dimension of transfer operator ({1}) changing value to numeig={2}'.format(numeig+1,LOP.shape[0],LOP.shape[0]-2))
        numeig=LOP.shape[0]-2

    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.real(eta))

    while np.abs(np.imag(eta[m]))/np.abs(np.real(eta[m]))>1E-4:
        numeig=numeig+1
        print ('found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'.format(numeig))
        print (eta)
        eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
        m=np.argmax(np.real(eta))
    if np.issubdtype(dtype,np.floating):
        out=np.reshape(vec[:,m],D1l*D1l)
        if np.linalg.norm(np.imag(out))>1E-10:
            raise TypeError("UnitcellTMeigs: dtype was float, but returned eigenvector had a large imaginary part; something went wrong here!")
        return np.real(eta[m]),np.real(out),numeig
    if np.issubdtype(dtype,np.complexfloating):    
        return eta[m],np.reshape(vec[:,m],D1l*D1l),numeig


def MixedUnitcellTMeigs(Upper,Lower,direction,numeig=6,init=None,nmax=800,tolerance=1e-12,ncv=10,which='LM'):
    """
    sparse computation of the dominant left or right eigenvector of the mixed transfer matrix

    -----------------------------------------------
    Parameters:
    -----------------------------------------------
    Upper/Lower: list of np.ndarray, or MPS objects
                 the mps tensor
    direction:   int 
                 direction>0 gives the left eigenvector, direction<0 gives the right eigenvector
    numeig:      int (6)
                 hyperparameter, number of eigenvectors to be calculated by argpack; note that numeig is not the 
                 number of returned eigenvectors, the number of returned eigenvectors is always one; numeig influences the arpack solver
                 due to a bug in lapack/arpack, and should usually be chosen >4 for stability reasons
    init:        None or np.ndarray of shape D**2
                 an initial guess for the eigenvector
    nmax:        int
                 maximum numbers of iteration in eigs
    tolerance:   float
                 precision of the eigen-values
    ncv:         int
                 number of krylov vectors in eigs
    which:       str, any of {'LR','SM','LA','SA'}
                 which eigenvector to calculate; use 'LR' or 'SM' 
    -----------------------------------------------
    Returns:
    -----------------------------------------------
    tuple containing the eigenvalue,eigenvector pair


    """

    if isinstance(Upper,MPSL.MPS) and isinstance(Lower,MPSL.MPS):
        dtype=np.result_type(*Upper._tensors,*Lower._tensors).type

    elif isinstance(Upper,list) and isinstance(Lower,list):
        dtype=np.result_type(*Upper,*Lower).type
    else:
        raise TypeError("MixedUnitcellTMeigs: Upper and Lower have to be either both MPS objects or both lists of np.ndarrays")
    
    [D1l,D2l,dl]=np.shape(Upper[0])
    [D1r,D2r,dr]=np.shape(Upper[-1])

    [D1l_,D2l_,dl_]=np.shape(Lower[0])
    [D1r_,D2r_,dr_]=np.shape(Lower[-1])
    
    if D1l!=D2r:
        raise ValueError(" in UnitcellTMeigs: ancillary dimensions of the upper mps have to be the same on left and right side")
    if D1l_!=D2r_:
        raise ValueError(" in UnitcellTMeigs: ancillary dimensions of the lower mps have to be the same on left and right side")


    mv=fct.partial(MixedUnitcellTransferOperator,*[direction,Upper,Lower])
    LOP=LinearOperator((D1l*D1l_,D2r*D2r_),matvec=mv,rmatvec=None,matmat=None,dtype=dtype)
    eta,vec=sp.sparse.linalg.eigs(LOP,k=numeig,which=which,v0=init,maxiter=nmax,tol=tolerance,ncv=ncv)
    m=np.argmax(np.abs(eta))
    if direction<0:
        return eta[m],vec[:,m]
    if direction>0:
        return eta[m],vec[:,m]


#gets L and R at "index"; shifts position of mps to index
#L contains mps matrix at site index
#R contains mps matrix at site index+1
def getBlockHam(mps,lbold,rbold,mpo,index):
    lb=np.copy(lbold)
    rb=np.copy(rbold)
    mps.position(index)
    A,lam,Z=prepareTensor(mps._tensors[index],1)
    for n in range(index):
        lb=addLayer(lb,mps._tensors[n],mpo[n],mps._tensors[n],1)
    lb=addLayer(lb,A,mpo[index],A,1)
    for n in range(mps.N-1,index,-1):
        rb=addLayer(rb,mps._tensors[n],mpo[n],mps._tensors[n],-1)
    return lb,rb,lam



#cuts an mpo at "index", flips the parts and repatches them
def patchmpo(mpo,index):
    temp=[]
    for n in range(index,len(mpo)):
        temp.append(np.copy(mpo[n]))

    for n in range(0,index):
        temp.append(np.copy(mpo[n]))

    for n in range(len(mpo)):
        mpo[n]=np.copy(temp[n])

def computeDensity(dens0,mps,direction,dtype=np.dtype(float)):
    """
    comuputes a reduced density matrix obtained from 
    evolving dens0 over a mps unitcell;
    dens0: initial density matrix of shape (D_1,D_1) or (D_N,D_N)
    mps: list of mps tensors (ndarrays of shape(D_n,D_n,n)) for n=1,...,N, or an MPS object
    direction: direction of contraction: direction>0: do left-to right evolution, direction<0: do a right to left evolution
    returns dens: a list of density matrices on each bond (including the left and right bonds to the left and right of the mps, len(dens)=len(mps)+1)
    """
    if isinstance(mps,MPSL.MPS) or isinstance(mps,np.ndarray):
        dtype=np.result_type(dens0,mps.dtype).type
    elif isinstance(mps,list):
        dtype=np.result_type(dens0,*mps).type
    else:
        raise TypeError("computeDensity: unknow type for mps")
    
    dens=[]
    for n in range(len(mps)+1):
        dens.append(None)
    if direction>0:
        D,D2,d=np.shape(mps[0])
        L=np.zeros((D,D,1),dtype=dtype)
        L[:,:,0]=np.copy(dens0)
        dens[0]=L[:,:,0]
        for n in range(len(mps)):
            L=addELayer(L,mps[n],mps[n],1)
            dens[n+1]=L[:,:,0]
        return dens

    if direction<0:
        D,D2,d=np.shape(mps[-1])
        R=np.zeros((D2,D2,1),dtype=dtype)
        R[:,:,0]=np.copy(dens0)
        dens[-1]=R[:,:,0]
        for n in range(len(mps)-1,-1,-1):
            R=addELayer(R,mps[n],mps[n],-1)
            dens[n]=R[:,:,0]
        return dens


def getBoundaryHams(mps,mpo,regauge=False,numeig=6):
    """
    calculates the environment of an infinite MPS
    mps: an infinite system mps
    mpo: an infinite system mpo
    if regauge==True, the routine regauges mps in place into symmetric gauge
    ToDo: allow passing precision parameters to __regauge__
    """
    if regauge:
        mps.regauge('symmetric')
    else:
        if (len(mps)-mps.pos)<mps.pos:
            mps.position(len(mps))
        else:
            mps.position(0)
            
    if mps.pos==0:
        D=np.shape(mps[0])[0]
        if regauge:
            phasematrix=mps._connector.dot(mps._mat)
            mps[0]=ncon.ncon([phasematrix,mps[0]],[[-1,1],[1,-2,-3]])
            mps._mat=mps._mat.dot(herm(phasematrix))
        temp=mps.tensors            
        eta,lss,numeig=UnitcellTMeigs(temp,direction=1,numeig=numeig,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')#is type preserving, or raises an expception if it can't be
        lbound=np.reshape(lss,(D,D))
        lbound=fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        
        Z=np.trace(lbound)
        lbound=lbound/Z
        
        ldens=computeDensity(lbound,temp,direction=1)
        rdens=computeDensity(np.eye(D),temp,direction=-1)
        f0r=np.zeros((D*D))
        f0r,hr=computeUCsteadyStateHamiltonianGMRES(temp,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000)
        
        mps.position(mps.N)
        if regauge:
            phasematrix=mps._mat.dot(mps._connector)
            mps[-1]=ncon.ncon([mps[-1],phasematrix],[[-1,1,-3],[1,-2]])
            mps._mat=herm(phasematrix).dot(mps._mat)
        temp=mps.tensors
        
        eta,rss,numeig=UnitcellTMeigs(temp,direction=-1,numeig=numeig,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')#is type preserving, or raises an expception if it can't be
        rbound=np.reshape(rss,(D,D))
        rbound=fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        
        Z=np.trace(rbound)
        rbound=rbound/Z
        
        rdens=computeDensity(rbound,temp,direction=-1)
        ldens=computeDensity(np.eye(D),temp,direction=1)
        
        f0l=np.zeros((D*D))
        f0l,hl=computeUCsteadyStateHamiltonianGMRES(temp,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000)
        
        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound

    if mps.pos==len(mps):
        D=np.shape(mps[-1])[1]
        if regauge:
            
            phasematrix=mps._mat.dot(mps._connector)
            mps[-1]=ncon.ncon([mps[-1],phasematrix],[[-1,1,-3],[1,-2]])
            mps._mat=herm(phasematrix).dot(mps._mat)

        temp=mps.tensors
        eta,rss,numeig=UnitcellTMeigs(temp,direction=-1,numeig=numeig,init=np.reshape(np.eye(D),(D*D))/(D*1.0),nmax=10000,tolerance=1E-16,which='LR')#is type preserving, or raises an expception if it can't be
        rbound=np.reshape(rss,(D,D))
        rbound=fixPhase(np.reshape(rss,(D,D)))
        rbound=(rbound+herm(rbound))/2.0
        Z=np.trace(rbound)
        rbound=rbound/Z
        
        rdens=computeDensity(rbound,temp,direction=-1)
        ldens=computeDensity(np.eye(D),temp,direction=1)
    
    
        f0l=np.zeros((D*D))
        f0l,hl=computeUCsteadyStateHamiltonianGMRES(temp,mpo,f0l,ldens,rdens,direction=1,thresh=1E-10,imax=1000)
        lb=np.copy(f0l)
        mps.position(0)
        if regauge:
            phasematrix=mps._connector.dot(mps._mat)
            mps[0]=ncon.ncon([phasematrix,mps[0]],[[-1,1],[1,-2,-3]])
            mps._mat=mps._mat.dot(herm(phasematrix))
            
        temp=mps.tensors
        eta,lss,numeig=UnitcellTMeigs(temp,direction=1,numeig=numeig,init=np.reshape(np.eye(D),(D*D)),nmax=10000,tolerance=1E-16,which='LR')#is type preserving, or raises an expception if it can't be
        lbound=np.reshape(lss,(D,D))
    
        lbound=fixPhase(np.reshape(lss,(D,D)))
        lbound=(lbound+herm(lbound))/2.0
        Z=np.trace(lbound)
        lbound=lbound/Z
        lbound=(lbound+herm(lbound))/2.0
    
    
        ldens=computeDensity(lbound,temp,direction=1)
        rdens=computeDensity(np.eye(D),temp,direction=-1)
            
        f0r=np.zeros((D*D))
        f0r,hr=computeUCsteadyStateHamiltonianGMRES(temp,mpo,f0r,ldens,rdens,direction=-1,thresh=1E-10,imax=1000)
    
        lb=np.copy(f0l)
        rb=np.copy(f0r)
        return lb,rb,lbound,rbound,hl,hr


def computeUCsteadyStateHamiltonianGMRES(mps,mpopbc,boundary,ldens,rdens,direction,thresh,imax):

    if isinstance(mps,MPSL.MPS) or isinstance(mps,np.ndarray):
        dtype=np.result_type(mps.dtype,mpopbc.dtype,boundary.dtype,*ldens,*rdens).type
    elif isinstance(mps,list):
        dtype=np.result_type(*mps,mpopbc.dtype,boundary.dtype,*ldens,*rdens).type
    else:
        raise TypeError("computeUCsteadyStateHamiltonianGMRES: unknow type for mps")

    NUC=len(mps)
    [D1r,D2r,d]=np.shape(mps[NUC-1])
    [D1l,D2l,d]=np.shape(mps[0])
    [B1,B2,d1,d2]=np.shape(mpopbc[NUC-1])
    if direction>0:
        mpo=np.zeros((1,B2,d1,d2),dtype=dtype)
        mpo[0,:,:,:]=mpopbc[NUC-1][-1,:,:,:]
        L=initializeLayer(mps[NUC-1],np.eye(D1r),mps[NUC-1],mpo,1)

        for n in range(0,len(mps)):
            L=addLayer(L,mps[n],mpopbc[n],mps[n],1)    

        h=np.trace(L[:,:,0].dot(rdens[-1]))
        inhom=np.reshape(L[:,:,0]-h*np.transpose(np.eye(D2r)),D2r*D2r) 
        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D1l*D1l)),thresh,imax,datatype=dtype,direction=1)
        L[:,:,0]=np.reshape(k2,(D2r,D2r))
        return np.copy(L),h

    if direction<0:
        mpo=np.zeros((B1,1,d1,d2),dtype=dtype)
        mpo[:,0,:,:]=mpopbc[0][:,0,:,:]
        R=initializeLayer(mps[0],np.eye(D2l),mps[0],mpo,-1)
        for n in range(len(mps)-1,-1,-1):
            R=addLayer(R,mps[n],mpopbc[n],mps[n],-1)    

        h=np.trace(R[:,:,-1].dot(ldens[0]))
        inhom=np.reshape(R[:,:,-1]-h*np.transpose(np.eye(D1l)),D1l*D1l)

        [k2,info]=TDVPGMRESUC(mps,ldens,rdens,inhom,np.reshape(boundary,(D2r*D2r)),thresh,imax,datatype=dtype,direction=-1)
        R[:,:,-1]=np.reshape(k2,(D2r,D2r))
        return np.copy(R),h

def pseudoUnitcellTransferOperator(direction,mps,ldens,rdens,vector):
    [D1l,D2l,dl]= np.shape(mps[0])
    [D1r,D2r,dr]= np.shape(mps[-1])
    if direction>0:
        x=np.reshape(vector,(D1l,D1l))
        return UnitcellTransferOperator(direction,mps,vector)-np.reshape(np.trace(np.transpose(x).dot(rdens[-1]))*ldens[-1],(D2r*D2r))
    if direction<0:
        x=np.reshape(vector,(D2r,D2r))
        return UnitcellTransferOperator(direction,mps,vector)-np.reshape(np.trace(np.transpose(x).dot(ldens[0]))*rdens[0],(D1l*D1l))


def TDVPGMRESUC(mps,ldens,rdens,inhom,x0,tolerance=1e-10,maxiteration=2000,datatype=np.dtype(float),direction=1):
    if direction>0:
        [D1l,D2l,d]=np.shape(mps[0])
        [D1r,D2r,d]=np.shape(mps[-1])
        mv=fct.partial(OneMinusPseudoUnitcellTransferOperator,*[direction,mps,ldens,rdens])
        LOP=LinearOperator((D1l*D1l,D2r*D2r),matvec=mv,dtype=datatype)
        return sp.sparse.linalg.lgmres(LOP,inhom,x0,tol=tolerance,maxiter=maxiteration)
    if direction<0:
        [D1l,D2l,d]=np.shape(mps[0])
        [D1r,D2r,d]=np.shape(mps[-1])
        mv=fct.partial(OneMinusPseudoUnitcellTransferOperator,*[direction,mps,ldens,rdens])
        LOP=LinearOperator((D1l*D1l,D2r*D2r),matvec=mv,dtype=datatype)
        return sp.sparse.linalg.lgmres(LOP,inhom,x0,tol=tolerance,maxiter=maxiteration)



def regaugeIMPS(mps,gauge,ldens=None,rdens=None,truncate=1E-16,D=None,nmaxit=1000,tol=1E-10,ncv=30,pinv=1E-200,thresh=1E-8,numeig=6):
    """
    takes an mps (can either be a list of np.arrays or an object of type MPS from mps.py) and regauges it in place
    gauge can be either of {'left','right',symmetric'} (desired gauge of the output mps)
    if gauge=symmetric, mps is right orthogonal; in this case the routine returns the schmidt-values in a diagonal matrix
    Note that in the case that gauge=symmetric and schmidt coefficients <= 1E-15 are present, the state is truncated, and an
    additional back and forth sweep is done to ensure proper normalization

    Parameters:
    mps:      list() of np.ndarrays of shape or MPS object
              the mps to be regauged
    gauge:    str()
              the desired gauge; can be {'left','right','symmetric'}
    ldens:    np.ndarray of shape (D,D) with D=mps[0].shape[0], or None
              an initial guess for the left eigenvector of the transfer operator 
    rdens:    np.ndarray of shape (D,D) with D=mps[-1].shape[1], or None
              an initial guess for the right eigenvector of the transfer operator 
    truncate: float
              truncation threshold; if "truncate"<=1E-15 no truncation is applied, otherwise state is truncated 
              to the value of "truncate" or D, depending which one is reached first
              Note that truncation is only done in the gauge=symmetric mode, since for other gauges truncation is not well defined
    D:        int
              the maximally allowed bond dimension
    nmaxit:   int
              maximum number of iterations in sparse solver
    tol:      float
              desired tolerance of the eigenvectors
    ncv:      int
              number of krylov vectors used in the sparse eigensolver
    pinv:     float
              pseudo-inverse threshold
    thresh:   float
              output threshold parameter (has no effect on the return values and can be ignored)


    Returns:
    if gauge='left':      the mps is modified in place into left orthogonal form; 
                          a properly normalized identity matrix is returned (the new left steady state vector of the 
                          transfer operator)
    if gauge='right':     the mps is modified in place into right orthogonal form; 
                          a properly normalized identity matrix is returned (the new right steady state vector of the 
                          transfer operator)
    if gauge='symmetric': the mps is modified in place into right orthogonal form, such that the left steady state is a
                          diagonal matrix containing the Schmidt values; 
                          the properly normalized left eigenvector of the transfer operator is returned in matrix from
                          (a diagonal matrix containing the Schmidt values in descending order)
    """

    N=len(mps)
    dtype=np.result_type(*mps).type
    [D1l,D2l,d]=np.shape(mps[0])
    [D1r,D2r,dr]=np.shape(mps[-1])
    assert(D1l==D2r)
    if truncate<np.sqrt(tol):
        warnings.warn('regaugeIMPS: truncate ({0}) <np.sqrt(tol) ({1}); this can cause problems.'.format(truncate,np.sqrt(tol)))
    if np.any(ldens==None):
        initl=np.reshape(np.eye(D1l).astype(dtype),D1l*D1l)
    else:
        initl=np.reshape(ldens,D1l*D1l)
    if np.any(rdens==None):
        initr=np.reshape(np.eye(D2r).astype(dtype),D2r*D2r)
    else:
        initr=np.reshape(rdens,D2r*D2r)
    if gauge in ('left','l',1):
        [eta,vl,numeig]=UnitcellTMeigs(mps,direction=1,numeig=numeig,init=initl,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
        sqrteta=np.real(eta)**(1./(2.*N))
        for site in range(N):
            mps[site]=mps[site]/sqrteta

        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',eta)
    
        l=np.reshape(vl,(D1l,D1l))
        l=l/np.trace(l)
        l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=u.dot(np.diag(np.sqrt(eigvals)))
        invy=np.diag(np.sqrt(inveigvals)).dot(herm(u))
        #multiply y to the left and y^{-1} to the right bonds of the tensor:
        #the index contraction looks weird, but is correct; my l matrices have their 0-index on the non-conjugated top layer
        mps[0]=np.tensordot(y,mps[0],([0],[0]))
        mps[N-1]=np.transpose(np.tensordot(mps[N-1],invy,([1],[1])),(0,2,1))
        for n in range(N-1):
            tensor,rest,Z=prepareTensor(mps[n],1)
            mps[n]=np.copy(tensor)
            if n<N-1:
                mps[n+1]=np.tensordot(rest,mps[n+1],([1],[0]))

        Z=np.trace(np.tensordot(mps[N-1],np.conj(mps[N-1]),([0,2],[0,2])))/np.shape(mps[N-1])[1]
        mps[N-1]=mps[N-1]/np.sqrt(Z)
        return np.eye(mps[N-1].shape[1]).astype(dtype)/np.sqrt(mps[N-1].shape[1])
    if gauge in ('right','r',-1):

        [eta,vr,numeig]=UnitcellTMeigs(mps,direction=-1,numeig=numeig,init=initr,nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
        sqrteta=np.real(eta)**(1./(2.*N))
        for site in range(N):
            mps[site]=mps[site]/sqrteta
        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in mpsfunctions.py.regaugeIMPSe: warning: found eigenvalue eta with large imaginary part: ',eta)

        r=np.reshape(vr,(D2r,D2r))
        #fix phase of l and restore the proper normalization of l
        r=r/np.trace(r)
        r=(r+herm(r))/2.0

        eigvals,u=np.linalg.eigh(r)

        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        r=u.dot(np.diag(eigvals)).dot(herm(u))
        
        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        #lam,u=np.linalg.eigh(r)
        x=u.dot(np.diag(np.sqrt(eigvals)))
        invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
        #multiply y to the left and y^{-1} to the right bonds of the tensor:
        
        mps[N-1]=np.copy(np.transpose(np.tensordot(mps[N-1],x,([1],[0])),(0,2,1)))
        mps[0]=np.copy(np.tensordot(invx,mps[0],([1],[0])))
        for n in range(N-1,0,-1):
            tensor,rest,Z=prepareTensor(mps[n],direction=-1)
            mps[n]=np.copy(tensor)
            if n>0:
                mps[n-1]=np.transpose(np.tensordot(mps[n-1],rest,([1],[0])),(0,2,1))

        Z=np.trace(np.tensordot(mps[0],np.conj(mps[0]),([1,2],[1,2])))/np.shape(mps[0])[0]
        mps[0]=mps[0]/np.sqrt(Z)
        return np.eye(mps[0].shape[0]).astype(dtype)/np.sqrt(mps[0].shape[0])

    if gauge in ('symmetric','s',0):

        [eta,vr,numeig]=UnitcellTMeigs(mps,direction=-1,numeig=numeig,init=np.reshape(np.eye(D1l),D1l*D1l),nmax=nmaxit,tolerance=tol,\
                                       ncv=ncv,which='LM')
        
        sqrteta=np.real(eta)**(1./(2.*N))
        for site in range(N):
            mps[site]=mps[site]/sqrteta
        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',eta)

        r=np.reshape(vr,(D2r,D2r))
        #fix phase of l and restore the proper normalization of l
        r=r/np.trace(r)
        r=(r+herm(r))/2.0

        eigvals,u=np.linalg.eigh(r)
        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        r=u.dot(np.diag(eigvals)).dot(herm(u))
        
        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))

        [eta,vl,numeig]=UnitcellTMeigs(mps,direction=1,numeig=numeig,init=np.reshape(np.eye(D1l),D1l*D1l),nmax=nmaxit,tolerance=tol,ncv=ncv,which='LM')
        sqrteta=np.real(eta)**(1./(2.*N))
        for site in range(N):
            mps[site]=mps[site]/sqrteta
        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',eta)
    
        l=np.reshape(vl,(D1l,D1l))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        l=(l+herm(l))/2.0
            
        eigvals,u=np.linalg.eigh(l)
        eigvals=np.abs(eigvals)
        eigvals=eigvals/np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0
        
        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        [U,lam,Vdag]=np.linalg.svd(y.dot(x))        
        if truncate<=1E-15:
            lam=lam/np.linalg.norm(lam)
            Dold=len(lam)
            lam=lam[lam>1E-15]

            U=U[:,0:len(lam)]
            Vdag=Vdag[0:len(lam),:]
            lam/=np.linalg.norm(lam)

            mps[0]=np.tensordot(np.diag(lam).dot(Vdag).dot(invx),mps[0],([1],[0]))
            mps[N-1]=np.transpose(np.tensordot(mps[N-1],invy.dot(U),([1],[0])),(0,2,1))
            
            for n in range(N-1):#sum can stop at N-2 because state is not truncated
                tensor,rest,Z=prepareTensor(mps[n],1)
                mps[n]=np.copy(tensor)
                if n<N-1:
                    mps[n+1]=np.tensordot(rest,mps[n+1],([1],[0]))

            if Dold!=len(lam):
                for n in range(N-1,0,-1):
                    tensor,rest,Z=prepareTensor(mps[n],-1)
                    mps[n]=np.copy(tensor)
                    if n>0:
                        mps[n-1]=np.transpose(np.tensordot(mps[n-1],rest,([1],[0])),(0,2,1))

                for n in range(N-1):
                    tensor,rest,Z=prepareTensor(mps[n],1)
                    mps[n]=np.copy(tensor)
                    if n<N-1:
                        mps[n+1]=np.tensordot(rest,mps[n+1],([1],[0]))
        
            Z=np.trace(np.tensordot(mps[N-1],np.conj(mps[N-1]),([0,2],[0,2])))/np.shape(mps[N-1])[1]
            mps[N-1]=mps[N-1]/np.sqrt(Z)
            if Dold!=len(lam):
                warnings.warn('regaugeIMPS: the MPS has Schmidt-values<1E-15! switching to truncating the state even though flag "truncate"<1E-15 (indicating that no truncation should be done)!!')
                return regaugeIMPS(mps,gauge='symmetric',ldens=None,rdens=None,truncate=1E-16,D=D,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=thresh)        
            else:
                return np.diag(lam).astype(dtype)

        if truncate>1E-15:
            lam=lam/np.linalg.norm(lam)
            lam=lam[lam>truncate]
            if D!=None:
                lam=lam[0:min(len(lam),D)]
            U=U[:,0:len(lam)]
            Vdag=Vdag[0:len(lam),:]
            lam/=np.linalg.norm(lam)
            mps[0]=np.tensordot(np.diag(lam).dot(Vdag).dot(invx),mps[0],([1],[0]))
            mps[N-1]=np.transpose(np.tensordot(mps[N-1],invy.dot(U),([1],[0])),(0,2,1))
            
            for n in range(N-1):
                tensor,rest,Z=prepareTensor(mps[n],1)
                #tensor,S,V=prepareTruncate(mps[n],1,D=D,thresh=truncate)
                #rest=np.diag(S).dot(V)
                mps[n]=np.copy(tensor)
                if n<(N-1):
                    mps[n+1]=np.tensordot(rest,mps[n+1],([1],[0]))
            
            for n in range(N-1,0,-1):
                U,S,tensor,Z=prepareTruncate(mps[n],direction=-1,D=D,thresh=truncate,r_thresh=1E-14)
                mps[n]=np.copy(tensor)
                if n>0:
                    mps[n-1]=np.transpose(np.tensordot(mps[n-1],U.dot(np.diag(S)),([1],[0])),(0,2,1))

            Z=np.trace(np.tensordot(mps[0],np.conj(mps[0]),([1,2],[1,2])))/np.shape(mps[0])[0]
            mps[0]=mps[0]/np.sqrt(Z)
            return regaugeIMPS(mps,gauge='symmetric',ldens=None,rdens=None,truncate=1E-16,D=10,nmaxit=nmaxit,tol=tol,ncv=ncv,pinv=pinv,thresh=thresh)


        
def truncateMPS(mps,D):
    
    """
    simple truncation method for a list of mps tensors
    D (int): the desired bond dimension
    """
    
    orthonormalizeQR(mps,-1,0,len(mps)-1)
    N=len(mps)
    lams=[]                        
    for site in range(N):
        tensor,lam,v,Z=prepareTruncate(mps[site],1,D)
        lams.append(lam)
        mat=np.diag(lam).dot(v)
        mps[site]=tensor
        if site<(N-1):
            mps[site+1]=np.tensordot(mat,mps[site+1],([1],[0]))
        if site==(N-1):
            mps[site]=np.transpose(np.tensordot(mps[site],np.diag(lam),([1],[0])),(0,2,1))
    return lams


def canonizeMPS(mps,tr_thresh=1E-16,r_thresh=1E-14,numeig=6):
    
    """
    canonizes an mps, i.e. it returns the Gamma and lambda matrices in a list

    Parameters 
    ------------------------------------
    mps: list of np.ndarrays of shape (D_i,D_{i+1},d_i) or MPS object 
         the mps can have obc or pbc boundaries; in the latter case,
         the state has to be regauged into symmetric form prior to calling canonizeMPS
    
    tr_thresh: float
               threshold for truncation of the MPS
    r_thresh:  float
               internal parameter for capturing exceptions (ignore it)
    numeig:    int
               hyper parameter: number of eigenvectors to be calculated in eigs 
               (only relevant for infinite systems)
    Returns: 
    ------------------------------------
    Gamma: list of np.ndarrays of shape (Di,Di,di)
           of len(Gamma)=len(mps) containing the Gamma-matrices
    Lam: list of np.arrays of shape (Di,)
         containing the Schmidt-values, stored as one-dimensional np.arrays
    """
    
    if (mps[0].shape[0]==1) and (mps[-1].shape[1]==1):
        #check that the state is obc
        assert(mps[0].shape[0]==1)
        assert(mps[len(mps)-1].shape[1]==1)
        #make a copy of the state
        if isinstance(mps,MPSL.MPS):
            Gamma=copy.deepcopy(mps._tensors)
        elif isinstance(mps,list):
            Gamma=copy.deepcopy(mps)
        else:
            raise TypeError("in canonizeMPS: unknown type {0} for mps; use list of np.ndarrays or MPS object".format(type(mps)))
        Lam=[None]*(len(Gamma)-1)
        for n in range(len(mps)):
            #use QR decomposition on Gamma[n] to pruduce a left orthogonal tensor A 
            #and an upper triangular matrix r (r is normalized inside the routine)
            A,r,Z=prepareTensor(Gamma[n],1)
            Gamma[n]=A
            #multiply r to the right
            if n<(len(Gamma)-1):
                Gamma[n+1]=np.tensordot(r,Gamma[n+1],([1],[0]))
        
        for n in range(len(Gamma)-1,-1,-1):
            U,S,B,Z=prepareTruncate(Gamma[n],direction=-1,D=Gamma[n].shape[0],thresh=tr_thresh,r_thresh=r_thresh)
            if n==(len(Gamma)-1):
                Gamma[n]=B
            else:
                Gamma[n]=np.transpose(np.tensordot(B,np.diag(1.0/Lam[n]),([1],[0])),(0,2,1))
            if n>0:
                Lam[n-1]=S
                Gamma[n-1]=np.transpose(np.tensordot(Gamma[n-1],U.dot(np.diag(S)),([1],[0])),(0,2,1))

        Lam.append(np.ones(1))
        Lam.insert(0,np.ones(1))
        return Gamma,Lam


    elif (mps[0].shape[0]!=1) and (mps[-1].shape[1]!=1):
        if isinstance(mps,MPSL.MPS):
            Gamma=copy.deepcopy(mps._tensors)
        elif isinstance(mps,list):
            Gamma=copy.deepcopy(mps)
        else:
            raise TypeError("in canonizeMPS: unknown type {0} for mps; use list of np.ndarrays or MPS object".format(type(mps)))

        if tr_thresh<=1E-15:#no truncation is done
            D=max([s[0] for s in list(map(np.shape,Gamma))]+[Gamma[-1].shape[1]])
            
            #max(Gamma.__D__())#get the maximum bond dimension of the Gamma
            lam=regaugeIMPS(Gamma,gauge='symmetric',D=D,truncate=tr_thresh)#regaugeIMPS returns a matrix
        if tr_thresh>1E-15:#Gamma is truncated
            D=max([s[0] for s in list(map(np.shape,Gamma))]+[Gamma[-1].shape[1]])
            #D=max(Gamma.__D__())
            #regauge and truncate the state; note that "lam" is not diagonal now, 
            #and does not correpsond to the sq-root of reduced density matrix
            lam=regaugeIMPS(Gamma,gauge='symmetric',D=D,truncate=tr_thresh,numeig=numeig)

        Gamma[-1]=np.transpose(np.tensordot(Gamma[-1],lam,([1],[0])),(0,2,1))
        Lam=[None]*(len(Gamma)+1)
        Lam[-1]=np.diag(lam)
        Lam[0]=np.diag(lam)
        for n in range(len(Gamma)-1,-1,-1):
            U,S,B,Z=prepareTruncate(Gamma[n],direction=-1,D=Gamma[n].shape[0],thresh=1E-16,r_thresh=r_thresh)
            Gamma[n]=np.transpose(np.tensordot(B,np.diag(1.0/Lam[n+1]),([1],[0])),(0,2,1))
            Lam[n]=S
            if n>0:
                Gamma[n-1]=np.transpose(np.tensordot(Gamma[n-1],U.dot(np.diag(S)),([1],[0])),(0,2,1))
        return Gamma,Lam

    else:
        sys.exit('CanonizeMPS: mps has non-consistent boundary-bond-dimensions')



def regauge(tensor,gauge,initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-200,thresh=1E-8,trunc=1E-16,Dmax=100):
    """
    bring an mps tensor "tensor" (ndarray of shape (D,D,d) into gauge "gauge" (string)
    
    Parameters:
    -----------------------------------------------------
    tensor: ndarray of shape (D,D,d) 
            the mps tensor
    gauge: str() 
           can be {'left', 'right', 'symmetric'}; the desired gauge
    initial: np.ndarray of shape(D,D), or None
             initial guess for dominant eigenvector of the mps transfer operators
    nmaxit:   int
              maximum number of iterations in sparse solver
    tol:      float
              desired tolerance of the eigenvectors
    ncv:      int
              number of krylov vectors used in the sparse eigensolver
    pinv:     float
              pseudo-inverse threshold
    thresh:   float
              output threshold parameter (has no effect on the return values and can be ignored)
    numeig:   int
              the nuymber of eigenvectors to be calculated by arpack; hyperparameter, use numeig>4 if you want a stable algorithm
    trunc:    float
              truncation threshold, if trunc<1E-15, no truncation is done
    Dmax:     int 
              the maximum bond dimension to be retained if trunc > 1E-15

    Returns:
    -----------------------------------------------------
    if gauge='left' or 'right': a tuple (out,mat); 
                                out: np.ndarray of shape (D,D,d): either left or right orthonormal
                                mat: np.ndarray of shape (D,D):    the left or right dominant eigenvector of the transfer operator
    if gauge='symmetric':       a tuple (Gamma,lam,rest): 
                                Gamma: np.ndarrays of shape (D,D,d)
                                lam:   np.ndarray of shape (D,)
                                rest (float): the truncated weight

    """
    
    dtype=tensor.dtype.type
    if gauge in ('left','l',1):
        [chi,chi2,d]=np.shape(tensor)
        lam=np.eye(chi)
        #find left eigenvalue
        [eta,v,numeig]=TMeigs(tensor,direction=1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)
        #rescale so that eta=1
        #tensor=tensor/np.sqrt(np.real(eta))
        tensor/=np.sqrt(np.real(eta))        
        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))
        l=np.reshape(v,(chi,chi))
        #fix phase of l and restore the proper normalization of l
        l=l/np.trace(l)
        l=(l+herm(l))/2.0        

        eigvals,u=np.linalg.eigh(l)
        eigvals=np.abs(eigvals)
        eigvals=eigvals/np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals=eigvals/np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=u.dot(np.diag(np.sqrt(eigvals)))
        invy=np.diag(np.sqrt(inveigvals)).dot(herm(u))

        #y=np.diag(np.sqrt(eigvals)).dot(u)
        #invy=herm(u).dot(np.diag(np.sqrt(1.0/eigvals)))
        out=np.copy(np.tensordot(y,tensor,([0],[0])))
        out=np.copy(np.transpose(np.tensordot(out,invy,([1],[1])),(0,2,1)))

        return out,y

    if gauge in ('right','r',-1):
        
        [chi1,chi,d]=np.shape(tensor)

        #find right eigenvalue
        [eta,v,numeig]=TMeigs(tensor,direction=-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)
        #rescale so that eta=1
        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))
        tensor=tensor/np.sqrt(np.real(eta))
        r=np.reshape(v,(chi,chi))
        r=r/np.trace(r)
        r=(r+herm(r))/2.0

        eigvals,u=np.linalg.eigh(r)
        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        r=u.dot(np.diag(eigvals)).dot(herm(u))
        
        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        x=u.dot(np.diag(np.sqrt(eigvals)))
        invx=np.diag(np.sqrt(inveigvals)).dot(herm(u))
        out=np.copy(np.tensordot(invx,tensor,([1],[0])))
        out=np.copy(np.transpose(np.tensordot(out,x,([1],[0])),(0,2,1)))

        return out,x


    if gauge in ('symmetric','s',0):
        warnings.warn("calling regauge with gauge='symmetric' is deprecated; use canonize instead")
        [chi1 ,chi2,d]=np.shape(tensor)
        [eta,v,numeig]=TMeigs(tensor,direction=1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)

        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

        tensor=tensor/np.sqrt(np.real(eta))
        l=np.reshape(v,(chi1,chi1))

        l=l/np.trace(l)
        l=(l+herm(l))/2.0

        eigvals,u=np.linalg.eigh(l)
        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<=pinv)]=0.0
        eigvals/=np.sum(eigvals)
        l=u.dot(np.diag(eigvals)).dot(herm(u))

        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
        invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

        
        [eta,v,numeig]=TMeigs(tensor,direction=-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)

        if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
            print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))
        r=np.reshape(v,(chi2,chi2))
        r=r/np.trace(r)
        r=(r+herm(r))/2.0        

        eigvals,u=np.linalg.eigh(r)
        eigvals=np.abs(eigvals)
        eigvals/=np.sum(eigvals)
        eigvals[np.nonzero(eigvals<pinv)]=0.0
        eigvals/=np.sum(eigvals)

        r=u.dot(np.diag(eigvals)).dot(herm(u))
        
        inveigvals=np.zeros(len(eigvals),dtype=dtype)
        inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
        inveigvals[np.nonzero(eigvals<=pinv)]=0.0

        x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
        invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))
        
        [U,lam,Vdag]=svd(y.dot(x))
        D=len(lam)
        Z=np.linalg.norm(lam)

        gamma=np.tensordot(Vdag.dot(invx),tensor,([1],[0]))
        gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
        A=np.tensordot(np.diag(lam),gamma,([1],[0]))
        lam=lam/Z
        rest=[dtype(0.0)]
        if trunc>1E-15:
            rest=lam[lam<=trunc]
            lam=lam[lam>trunc]
            rest=np.append(lam[min(len(lam),Dmax)::],rest)
            lam=lam[0:min(len(lam),Dmax)]
            U=U[:,0:len(lam)]
            Vdag=Vdag[0:len(lam),:]
            Z=np.linalg.norm(lam)
            lam=lam/Z
            
        gamma=np.tensordot(Vdag.dot(invx),tensor,([1],[0]))
        gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
        A=np.tensordot(np.diag(lam),gamma,([1],[0]))
        Z=np.trace(np.tensordot(A,np.conj(A),([0,2],[0,2])))
        gamma/=np.sqrt((Z/len(lam)))
        A/=np.sqrt((Z/len(lam)))
        #if len(lam)<D:
        #    if (len(lam)==1):
        #        print('state has been truncated to a pure product state')
        #    else:
        #        gamma,lam,rest2=regauge(A,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14,thresh=1E-8,trunc=1E-16,Dmax=len(lam))            
        return gamma,lam,np.sum(rest)

    
def canonize(tensor,initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-200,thresh=1E-8,trunc=1E-16,Dmax=100):
    """
    canonize an infinite, translational invariant mps, i.e. bring it into Gamma-lambda form
    
    Parameters:
    -----------------------------------------------------
    tensor: ndarray of shape (D,D,d) 
            the mps tensor
    initial: np.ndarray of shape(D,D), or None
             initial guess for dominant eigenvector of the mps transfer operators
    nmaxit:   int
              maximum number of iterations in sparse solver
    tol:      float
              desired tolerance of the eigenvectors
    ncv:      int
              number of krylov vectors used in the sparse eigensolver
    pinv:     float
              pseudo-inverse threshold
    thresh:   float
              output threshold parameter (has no effect on the return values and can be ignored)
    numeig:   int
              the nuymber of eigenvectors to be calculated by arpack; hyperparameter, use numeig>4 if you want a stable algorithm
    trunc:    float
              truncation threshold, if trunc<1E-15, no truncation is done
    Dmax:     int 
              the maximum bond dimension to be retained if trunc > 1E-15

    Returns:
    -----------------------------------------------------
    (Gamma,lam,rest): 
    Gamma: np.ndarrays of shape (D,D,d)
    lam:   np.ndarray of shape (D,)
    rest (float): the truncated weight

    """

    dtype=tensor.dtype.type
    [chi1 ,chi2,d]=np.shape(tensor)
    [eta,v,numeig]=TMeigs(tensor,direction=1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)

    if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
        print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))

    tensor=tensor/np.sqrt(np.real(eta))
    l=np.reshape(v,(chi1,chi1))

    l=l/np.trace(l)
    l=(l+herm(l))/2.0

    eigvals,u=np.linalg.eigh(l)
    eigvals=np.abs(eigvals)
    eigvals/=np.sum(eigvals)
    eigvals[np.nonzero(eigvals<=pinv)]=0.0
    eigvals/=np.sum(eigvals)
    l=u.dot(np.diag(eigvals)).dot(herm(u))

    inveigvals=np.zeros(len(eigvals),dtype=dtype)
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0

    y=np.transpose(u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u)))
    invy=np.transpose(herm(u)).dot(np.diag(np.sqrt(inveigvals))).dot(np.transpose(u))

    
    [eta,v,numeig]=TMeigs(tensor,direction=-1,numeig=numeig,init=initial,nmax=nmaxit,tolerance=tol,ncv=ncv)

    if np.abs(np.imag(eta))/np.abs(np.real(eta))>thresh:
        print ('in regauge: warning: found eigenvalue eta with large imaginary part: {0}'.format(eta))
    r=np.reshape(v,(chi2,chi2))
    r=r/np.trace(r)
    r=(r+herm(r))/2.0        

    eigvals,u=np.linalg.eigh(r)
    eigvals=np.abs(eigvals)
    eigvals/=np.sum(eigvals)
    eigvals[np.nonzero(eigvals<pinv)]=0.0
    eigvals/=np.sum(eigvals)

    r=u.dot(np.diag(eigvals)).dot(herm(u))
    
    inveigvals=np.zeros(len(eigvals),dtype=dtype)
    inveigvals[np.nonzero(eigvals>pinv)]=1.0/eigvals[np.nonzero(eigvals>pinv)]
    inveigvals[np.nonzero(eigvals<=pinv)]=0.0

    x=u.dot(np.diag(np.sqrt(eigvals))).dot(herm(u))
    invx=u.dot(np.diag(np.sqrt(inveigvals))).dot(herm(u))
    
    [U,lam,Vdag]=svd(y.dot(x))
    D=len(lam)
    Z=np.linalg.norm(lam)

    gamma=np.tensordot(Vdag.dot(invx),tensor,([1],[0]))
    gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
    A=np.tensordot(np.diag(lam),gamma,([1],[0]))
    lam=lam/Z
    rest=[dtype(0.0)]
    if trunc>1E-15:
        rest=lam[lam<=trunc]
        lam=lam[lam>trunc]
        rest=np.append(lam[min(len(lam),Dmax)::],rest)
        lam=lam[0:min(len(lam),Dmax)]
        U=U[:,0:len(lam)]
        Vdag=Vdag[0:len(lam),:]
        Z=np.linalg.norm(lam)
        lam=lam/Z
        
    gamma=np.tensordot(Vdag.dot(invx),tensor,([1],[0]))
    gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
    A=np.tensordot(np.diag(lam),gamma,([1],[0]))
    Z=np.trace(np.tensordot(A,np.conj(A),([0,2],[0,2])))
    gamma/=np.sqrt((Z/len(lam)))
    A/=np.sqrt((Z/len(lam)))
    #if len(lam)<D:
    #    if (len(lam)==1):
    #        print('state has been truncated to a pure product state')
    #    else:
    #        gamma,lam,rest2=regauge(A,gauge='symmetric',initial=None,nmaxit=100000,tol=1E-10,ncv=50,numeig=6,pinv=1E-14,thresh=1E-8,trunc=1E-16,Dmax=len(lam))            
    return gamma,lam,np.sum(rest)
    



def load_mps():
    D=24
    A0r=h5py.File('A0vecreal.h5')
    A0i=h5py.File('A0vecimag.h5')
    A1r=h5py.File('A1vecreal.h5')
    A1i=h5py.File('A1vecimag.h5')
    A0= np.reshape(np.asarray(list(A0r['A0R']))+1j*np.asarray(list(A0i['A0I'])),(D,D))
    A1= np.reshape(np.asarray(list(A1r['A1R']))+1j*np.asarray(list(A1i['A1I'])),(D,D))
    A=np.zeros((D,D,2)).astype(complex)

    A[:,:,0]=A0
    A[:,:,1]=A1
    return A
 
            
def regaugeSingleLayer(mps,gauge,precision=1E-10,nmax=1000):
    """
    regauges in infinite translational invariant mps into left or right orthogonal form using
    single layer method
    
    Parameters:
    ---------------------------------------------------------------
    mps:       np.ndarray of shape=(D,D,d)
    gauge:     int or str
               if gauge in (1,'l','left'), bring mps into left-orthogonal form
               if gauge in (-1,'r','right'), bring mps into right orthogonal form
    precision: float
               desired precision
    nmax:      int
               maximum number of iterations
               
    Returns:
    ---------------------------------------------------------------
    (tensor,C,F,conv)
    tensor: np.ndarray of shape=(D,D,d)
            if conv==True: left/right orthogonal mps
    C:      np.ndarray of shape (D,D)
            gauge transformation; see below for use-cases
    F:      float or complex
            normalization constant, see below for use-case
    conv:   bool
            convergence flag
    
    The gauge transformation which brings the input mps into left/right orthogonal form is given by C:
    gauge in (1,'l','left'): mps=[C]^(-1) x tensor/F x C
                             mps=ncon.ncon([np.linalg.inv(C),tensor/F,C],[[-1,1],[1,2,-3],[2,-2]])
    gauge in (-1,'r','right'): mps=C x tensor/F x [C]^(-1)
                               mps=ncon.ncon([C,tensor/F,np.linalg.inv(C)],[[-1,1],[1,2,-3],[2,-2]])
       
    """
    tensor=np.copy(mps)
    n=0
    if not (mps.shape[0]==mps.shape[1]):
        raise ValueError("mps.shape[0] is unequal to mps.shape[1]")
    
    sold=np.random.rand(tensor.shape[1])
    C=np.eye(mps.shape[0])
    if gauge in (1,'left','l'):
        while True:
            A,s,v,z=prepareTensor(tensor,direction=1,fixphase=False)
            Z=np.linalg.norm(s-sold)
            if Z<precision:
                s/=(np.sum(s)/len(s))
                assert(np.linalg.norm(s-np.ones(s.shape))<(100*precision))
                A_=ncon.ncon([C,mps,np.linalg.inv(C)],[[-1,1],[1,2,-3],[2,-2]])
                F=1.0/np.sqrt(np.sum(np.diag(np.tensordot(A_,np.conj(A_),([0,2],[0,2]))))/A_.shape[1])
                return ncon.ncon([A,v],[[-1,1,-3],[1,-2]]),C,F,True
            n+=1
            if n>=nmax:
                s/=(np.sum(s)/len(s))
                A_=ncon.ncon([C,mps,np.linalg.inv(C)],[[-1,1],[1,2,-3],[2,-2]])
                F=1.0/np.sqrt(np.sum(np.diag(np.tensordot(A_,np.conj(A_),([0,2],[0,2]))))/A_.shape[1])
                return tensor,C,F,False            
            C=np.diag(s).dot(v).dot(C)
            C/=np.linalg.norm(C)            
            tensor=ncon.ncon([np.diag(s).dot(v),A],[[-1,1],[1,-2,-3]])
            sold=s            
    elif gauge in (-1,'r','right'):
        while True:
            u,s,B,z=prepareTensor(tensor,direction=-1,fixphase=False)
            Z=np.linalg.norm(s-sold)
            if Z<precision:
                s/=(np.sum(s)/len(s))
                assert(np.linalg.norm(s-np.ones(s.shape))<(100*precision))
                B_=ncon.ncon([np.linalg.inv(C),mps,C],[[-1,1],[1,2,-3],[2,-2]])
                F=1.0/np.sqrt(np.sum(np.diag(np.tensordot(B_,np.conj(B_),([1,2],[1,2]))))/B_.shape[0])
                return ncon.ncon([u,B],[[-1,1],[1,-2,-3]]),C,F,True
            n+=1
            if n>=nmax:
                s/=(np.sum(s)/len(s))
                B_=ncon.ncon([np.linalg.inv(C),mps,C],[[-1,1],[1,2,-3],[2,-2]])
                F=1.0/np.sqrt(np.sum(np.diag(np.tensordot(B_,np.conj(B_),([1,2],[1,2]))))/B_.shape[0])
                return tensor,C,F,False
            C=C.dot(u).dot(np.diag(s))
            C/=np.linalg.norm(C)            
            tensor=ncon.ncon([B,u.dot(np.diag(s))],[[-1,1,-3],[1,-2]])
            sold=s
            
def canonizeSingleLayer(mps,trunc=1E-16,Dmax=100,precision=1E-10,nmax=1000):
    """
    canonizes an infinite, translational invariant mps using a single layer procedure
    this method achieves twice the accuracy obtained using the double layer method
    Parameters:
    ----------------------
    mps:        np.ndarray of shape (D,D,d)
                the mps
    trunc:      float
                truncation threshold; Schmidt values < trunc are discarded
                if trunc<=1E-15, no truncation is applied
    Dmax:       int
                maximally allowed bond dimension; overrides the bond dimension of trunc
    precision:  float
                desired precision of the Schmidt values
    nmax:       int
                maximum number of iterations
                
    Returns:
    -----------------------
    (gamma,lambda,rest)
    gamma:     np.ndarrray of shape (D',D',d), where D'<=D
               the gamma matrix of the canonical form
    lambda:    np.ndarray of shape (D',)
               the Schmidt values
    rest:      float
               the truncated weight

    """
    B,x,FB,conv=regaugeSingleLayer(mps,gauge=-1,precision=precision,nmax=nmax)
    A,y,FA,conv=regaugeSingleLayer(mps,gauge=1,precision=precision,nmax=nmax)
    invx=np.linalg.inv(x)
    invy=np.linalg.inv(y)
    [U,lam,Vdag]=svd(y.dot(x))
    D=len(lam)
    Z=np.linalg.norm(lam)

    gamma=np.tensordot(Vdag.dot(invx),mps,([1],[0]))
    gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
    A=np.tensordot(np.diag(lam),gamma,([1],[0]))
    lam=lam/Z
    rest=[0.0]
    if trunc>1E-15:
        rest=lam[lam<=trunc]
        lam=lam[lam>trunc]
        rest=np.append(lam[min(len(lam),Dmax)::],rest)
        lam=lam[0:min(len(lam),Dmax)]
        U=U[:,0:len(lam)]
        Vdag=Vdag[0:len(lam),:]
        Z=np.linalg.norm(lam)
        lam=lam/Z

    gamma=np.tensordot(Vdag.dot(invx),mps,([1],[0]))
    gamma=np.transpose(np.tensordot(gamma,invy.dot(U),([1],[0])),(0,2,1))
    A=np.tensordot(np.diag(lam),gamma,([1],[0]))
    Z=np.trace(np.tensordot(A,np.conj(A),([0,2],[0,2])))
    gamma/=np.sqrt((Z/len(lam)))
    A/=np.sqrt((Z/len(lam)))
    return gamma,lam,np.sum(rest)
    

def OneMinusPseudoUnitcellTransferOperator(direction,mps,ldens,rdens,vector):
    [D1l,D2l,dl]= np.shape(mps[0])
    [D1r,D2r,dr]= np.shape(mps[-1])
    if direction>0:
        x=np.reshape(vector,(D1l,D1l))
        return vector-UnitcellTransferOperator(direction,mps,vector)+np.reshape(np.tensordot(x,rdens[-1],([0,1],[0,1]))*ldens[-1],(D2r*D2r))
        #eturn np.reshape(x+np.trace(np.transpose(x).dot(rdens[-1]))*ldens[-1],(D2r*D2r))-UnitcellTransferOperator(direction,mps,vector)

    if direction<0:
        x=np.reshape(vector,(D2r,D2r))
        return vector-UnitcellTransferOperator(direction,mps,vector)+np.reshape(np.tensordot(ldens[0],x,([0,1],[0,1]))*rdens[0],(D1l*D1l))
        #return np.reshape(x+np.trace(np.transpose(x).dot(ldens[0]))*rdens[0],(D1l*D1l))-UnitcellTransferOperator(direction,mps,vector)
        
#calculates ( 1-exp(1j*momentum)*E+|r)(l| )*|vector) (direction<0) or (vector|*( 1-exp(1j*momentum)*E+|r)(l| ) (direction>0), where E is the mixed transfer operator; 
#takes a vector
#note that Lower is the conjugated layer here!
def OneMinusPseudoTransferOperator(Upper,Lower,r,l,direction,momentum,vector):
    [chiU1,chiU2,dU]=np.shape(Upper)
    [chiL1,chiL2,dL]=np.shape(Lower)
    if direction>0:
        x=np.reshape(vector,(chiU1,chiL1))
        if np.abs(momentum<1E-8):
            return vector-MixedTransferOperator(direction,Upper,Lower,vector)+np.reshape(np.tensordot(x,r,([0,1],[0,1]))*l,(chiU2*chiL2))
        else:
            return vector-np.exp(1j*momentum)*MixedTransferOperator(direction,Upper,Lower,vector)
    if direction<0:
        x=np.reshape(vector,(chiU2,chiL2))
        if np.abs(momentum<1E-8):
            return vector-MixedTransferOperator(direction,Upper,Lower,vector)+np.reshape(np.tensordot(l,x,([0,1],[0,1]))*r,(chiU1*chiL1))
        else:
            return vector-np.exp(1j*momentum)*MixedTransferOperator(direction,Upper,Lower,vector)

#takes a vector
def pseudoTransferOperator(A,r,l,direction,vector):
    """
    pseudoTransferOperator(A,r,l,direction,vector):
    evolves "vector" with the pseudo transfer operator E-|r)(l|, where  E is the MPS transfer operator
    A: ndarray of shape (D,D,d), mps tensor
    r,l: left and right dominant eigenvectors of the mps transfer operator E
    direction: int; if > 0, calculate (v|E -(v|r)(l|
                    if < 0, calculate E|v) -|r)(l|v)
    vector ndarray of shape (D**2): matrix reshape into vector form, the vector to be acted upon

    returns: ndarray of shape (D**2)
    """
    D=np.shape(A)[0]
    if direction>0:
        x=np.reshape(vector,(D,D))
        return TransferOperator(direction,A,vector)-np.reshape(np.tensordot(x,r,([0,1],[0,1]))*l,(D*D))
    if direction<0:
        x=np.reshape(vector,(D,D))
        return TransferOperator(direction,A,vector)-np.reshape(np.tensordot(l,x,([0,1],[0,1]))*r,(D*D))


#this routine is deprecated; use RENORMBLOCKHAMGMRES instead
def TDVPGMRES(tensorU,tensorL,r,l,inhom,x0,tolerance=1e-10,maxiteration=2000,direction=1,momentum=0.0):
    """
    deprecated, see RENORMBLOCKHAMGMRES(tensorU,tensorL,r,l,inhom,x0,tolerance=1e-10,maxiteration=200,direction=1,momentum=0.0):

    TDVPGMRES calculates the renormalized left and right Hamiltonian environment for a translational invariant infinite system with a nearest neighbor 
    hamiltonian

    tensorU/tensorL: upper and lower mps tensors; upper refers to the non-conjugated leg, lower to the conjugated one;
                     the tensors do not have be conjugated, this is done inside the routine
    r,l            : right and left reduced steady state density matrices of the mps
    inhom          : the unit-cell energy-operators (l|H_{AAAA} (direcion>0) or H_{BBBB}|r) (doirection<0), where A and B 
                     are left and right orthogonal (see paper by Jutho Haegeman)
    x0=None        : initial guess for the inversion
    tolerance=1E-10: accuracy of the lgmres solver
    maxiteration=200: maximum number oif iterations of lgmres
    direction=1     : int, if > 0, get the left environment, if <0 get the right environment
    momentum=0      : float, momentum quantum number; used for calculating excitations
    """
    
    warnings.warn('mpsfunctions.TDVPGMRES is deprecated; use mpsfunctions.RENORMBLOCKHAMGMRES instead')
    return RENORMBLOCKHAMGMRES(tensorU,tensorL,r,l,inhom,x0,tolerance,maxiteration,direction)

#calculates the reduced block hamiltonian for left or right side (depending on dir)
def RENORMBLOCKHAMGMRES(tensorU,tensorL,r,l,inhom,x0,tolerance=1e-10,maxiteration=200,direction=1,momentum=0.0):
    """
    RENORMBLOCKHAMGMRES(tensorU,tensorL,r,l,inhom,x0,tolerance=1e-10,maxiteration=200,direction=1,momentum=0.0)
    calculates the renormalized left and right Hamiltonian environment for a translational invariant infinite system with a nearest neighbor 
    hamiltonian

    tensorU/tensorL: upper and lower mps tensors; upper refers to the non-conjugated leg, lower to the conjugated one;
                     the tensors do not have be conjugated, this is done inside the routine
    r,l            : right and left reduced steady state density matrices of the mps
    inhom          : the unit-cell energy-operators (l|H_{AAAA} (direcion>0) or H_{BBBB}|r) (doirection<0), where A and B 
                     are left and right orthogonal (see paper by Jutho Haegeman)
    x0=None        : initial guess for the inversion
    tolerance=1E-10: accuracy of the lgmres solver
    maxiteration=200: maximum number oif iterations of lgmres
    direction=1     : int, if > 0, get the left environment, if <0 get the right environment
    momentum=0      : float, momentum quantum number; used for calculating excitations
    """
    
    x0=None
    dtype=tensorU.dtype
    if np.any(x0==None):
        x0=np.random.random_sample(inhom.shape).astype(dtype)
    [chiU1,chiU2,d]=np.shape(tensorU)
    [chiL1,chiL2,d]=np.shape(tensorL)
    
    if direction>0:
        mv=fct.partial(OneMinusPseudoTransferOperator,*[tensorU,tensorL,r,l,direction,momentum])
        LOP=LinearOperator((chiL1*chiU1,chiL2*chiU2),matvec=mv,dtype=dtype)
        [x,info]=lgmres(LOP,np.reshape(inhom,chiL1*chiU1),x0=np.reshape(x0,chiU1*chiL1),tol=tolerance,maxiter=maxiteration,outer_k=6)
        while info<0:
            [x,info]=lgmres(LOP,np.reshape(inhom,chiU1*chiL1),x0=np.reshape(x0,chiU1*chiL1),tol=tolerance,maxiter=maxiteration,outer_k=6)
        return np.reshape(x,(chiU2,chiL2))

    if direction<0:
        mv=fct.partial(OneMinusPseudoTransferOperator,*[tensorU,tensorL,r,l,direction,momentum])
        LOP=LinearOperator((chiU1*chiL1,chiU2*chiL2),matvec=mv,dtype=dtype)
        [x,info]=lgmres(LOP,np.reshape(inhom,chiU2*chiL2),x0=np.reshape(x0,chiU2*chiL2),tol=tolerance,maxiter=maxiteration,outer_k=6)
        while info<0:
            [x,info]=lgmres(LOP,np.reshape(inhom,chiU2*chiL2),x0=np.reshape(x0,chiU2*chiL2),tol=tolerance,maxiter=maxiteration,outer_k=6)
        return np.reshape(x,(chiU1 ,chiL1))



        
def ExHAproductSingle(l,L,LAA,LAAAA,LAAAA_OneMinusEAAinv,mpsA,mpsAtilde,VL,invsqrtl,invsqrtr,mpo,r,R,RAA,RAAAA,OneMinusEAAinv_RAAAA,GSenergy,k,tol,vec1):
    """
    Implementation of the effective matrix-vector product for calculating excitations for a translational invariant system
    """

    [D1,D2,d]=np.shape(VL)
    facp=1.0
    facm=1.0
    if np.abs(k)>=1E-8:
        facp=np.exp(1j*k)
        facm=np.exp(-1j*k)

    #bring the input vector into matrix form
    #multiply all necessary factors
    x=np.reshape(vec1,(D2,D1)).dot(invsqrtr)
    VLl=np.tensordot(invsqrtl,VL,([1],[0]))

    B=np.transpose(np.tensordot(VLl,x,([1],[0])),(0,2,1))

    #term1
    term1=HAproductSingleSiteMPS(L,mpo[0],RAA,B)
    #term2
    term2=HAproductSingleSiteMPS(LAA,mpo[1],R,B)

    #term3
    temp=addLayer(R,B,mpo[1],mpsAtilde,-1)
    term3=facp*HAproductSingleSiteMPS(L,mpo[0],temp,mpsA)

    #term4
    temp=addLayer(L,B,mpo[0],mpsA,1)
    term4=facm*HAproductSingleSiteMPS(temp,mpo[1],R,mpsAtilde)

    #term5
    term5=contractionE(L,B,OneMinusEAAinv_RAAAA)

    #term6
    term6=contractionE(LAAAA_OneMinusEAAinv,B,R)

    #term7
    temp=addELayer(R,B,mpsAtilde,-1)
    if np.abs(k)<1E-8:
        ih=np.reshape(temp[:,:,0]-np.tensordot(L,temp,([0,1,2],[0,1,2]))*R[:,:,0],(D1*D1))
    else:
        #ih=np.reshape(temp[:,:,0]-np.tensordot(L,temp,([0,1,2],[0,1,2]))*R[:,:,0],(D1*D1))
        ih=np.reshape(temp[:,:,0],(D1*D1))
    bla=TDVPGMRES(mpsA,mpsAtilde,r,l,ih,direction=-1,momentum=k,tolerance=tol,maxiteration=2000,x0=None)
    save7=np.reshape(bla,(D1,D1,1))
    term7=facp*contractionE(LAAAA_OneMinusEAAinv,mpsA,save7)

    #term8
    temp=addELayer(LAAAA_OneMinusEAAinv,B,mpsA,1)
    if np.abs(k)<1E-8:
        ih=np.reshape(temp[:,:,0]-np.tensordot(temp,R,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
    else:
        ih=np.reshape(temp[:,:,0],(D1*D1))
        #ih=np.reshape(temp[:,:,0]-np.tensordot(temp,R,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
    bla=TDVPGMRES(mpsAtilde,mpsA,r,l,ih,tolerance=tol,maxiteration=2000,direction=1,momentum=-k,x0=None)
    temp2=np.reshape(bla,(D1,D1,1))
    term8=facm*contractionE(temp2,mpsAtilde,R)

    #term9
    term9=facp*HAproductSingleSiteMPS(LAA,mpo[1],save7,mpsA)

    #term10
    temp=addLayer(LAA,B,mpo[1],mpsA,1)
    if np.abs(k)<1E-8:
        ih=np.reshape(temp[:,:,0]-np.tensordot(temp,R,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
    else:
        ih=np.reshape(temp[:,:,0],(D1*D1))
        #ih=np.reshape(temp[:,:,0]-np.tensordot(temp,R,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
    bla=TDVPGMRES(mpsAtilde,mpsA,r,l,ih,x0=None,tolerance=tol,maxiteration=2000,direction=1,momentum=-k)

    temp2=np.reshape(bla,(D1,D1,1))
    term10=facm*contractionE(temp2,mpsAtilde,R)

    #term11
    temp=addLayer(save7,mpsA,mpo[1],mpsAtilde,-1)
    term11=facp*facp*HAproductSingleSiteMPS(L,mpo[0],temp,mpsA)
   
    #term12
    temp=addLayer(L,B,mpo[0],mpsA,1)
    temp2=addLayer(temp,mpsAtilde,mpo[1],mpsA,1)
    if np.abs(k)<1E-8:
        ih=np.reshape(temp2[:,:,0]-np.tensordot(R,temp2,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
    else:
        #ih=np.reshape(temp2[:,:,0]-np.tensordot(R,temp2,([0,1,2],[0,1,2]))*L[:,:,0],(D1*D1))
        ih=np.reshape(temp2[:,:,0],(D1*D1))
    bla=TDVPGMRES(mpsAtilde,mpsA,R[:,:,0],L[:,:,0],ih,x0=None,tolerance=tol,maxiteration=2000,direction=1,momentum=-k)

    temp=np.reshape(bla,(D1,D1,1))
    term12=facm*facm*contractionE(temp,mpsAtilde,R)

    out=term1+term2+term3+term4+term5+term6+term7+term8+term9+term10+term11+term12
    out=np.tensordot(out,np.conj(VLl),([0,2],[0,2]))
    return np.reshape(np.tensordot(out,invsqrtr,([0],[1])),(D1*D2))#-GSenergy*vec1

#tdvp upate, works only for nearest neighbors
def TDVPupdate(r,tensor,mpo,kold=None,tol=1E-10):
    """
    TDVPupdate(r,tensor,mpo,kold=None,tol=1E-10):
    calculates the TDVP update for real and imaginary time evolution for a translational invariant infinite system.
    

    r (np.ndarray): right reduced density matrices a translational invariant mps given by "tensor"
    tensor (np.ndarray of shape(D,D,d)): mps tensor, has to be left-orthogonal!
    mpo (list or array of np.ndarray of shape (M,M,d,d)): an MPO for a nearest neighbor Hamiltonian, e.g. mpo=Hamiltonians.XXZ(Jz,Jxy,B,obc=True) (see Hamiltonians.py)
                                                          should have obc; it can have any length, the routine will only take the mpo-tensor at position 0
    
    """

    dtype=type(tensor[0,0,0])

    [chi1,chi2,d]=np.shape(tensor)
    [D1,D2,d1,d2]=np.shape(mpo[0])
    l=np.eye(chi1)    
    lb=initializeLayer(tensor,l,tensor,mpo[0],1)
    lb=np.reshape(addLayer(lb,tensor,mpo[1],tensor,1),(chi2,chi2))
    h=np.tensordot(lb,r,([0,1],[0,1]))
    inhom=np.reshape(lb-h*l,chi2*chi2)
    #def TDVPGMRES(tensor,r,l,inhom,x0,tolerance=1e-10,maxiteration=2000,datatype=float,direction=1):
    k = TDVPGMRES(tensor,tensor,r,l,inhom,x0=kold,tolerance=tol,maxiteration=1000)
    sqrtl=sp.linalg.sqrtm(l)
    isqrtl=np.linalg.pinv(sqrtl,rcond=1E-15)
    sqrtr=sp.linalg.sqrtm(r)
    isqrtr=np.linalg.pinv(sqrtr,rcond=1E-15)
    restmat=np.random.rand(d*chi1,chi2*(d-1))*0.0001
    L=np.reshape(np.transpose(tensor,(2,0,1)),(chi1*d,chi2)) #L_{\alpha,(\beta s)}
    L=np.hstack([L,restmat])

    [U,lam,Vd]=svd(L,full_matrices=1)
    VL=np.transpose(np.reshape(U[:,chi1::],(d,chi1,chi2)),(1,2,0))


    left=initializeLayer(tensor,sqrtl,VL,mpo[0],1);
    A_=np.zeros(np.shape(tensor),dtype=dtype)
    for n in range(d):
        A_[:,:,n]=isqrtr.dot(tensor[:,:,n])
    right=initializeLayer(tensor,r,A_,mpo[-1],-1);
    term1=np.tensordot(left,right,([0,2],[0,2]))
    #second term:
    lb=initializeLayer(tensor,l,tensor,mpo[0],1);
    VL_=np.zeros(np.shape(VL),dtype=dtype)
    for n in range(d):
        VL_[:,:,n]=isqrtl.dot(VL[:,:,n])

    lb=np.reshape(addLayer(lb,tensor,mpo[-1],VL_,1),(chi1,chi1))
    term2=np.transpose(lb).dot(sqrtr)

    #third term:
    VL_=np.zeros(np.shape(VL),dtype=dtype)
    for n in range(d):
        VL_[:,:,n]=isqrtl.dot(VL[:,:,n])
    lb=np.reshape(GeneralizedMatrixVectorProduct(1,VL_,tensor,k),(chi1,chi1))
    term3=np.transpose(lb).dot(sqrtr)
    xopt=term1+term2+term3
    Adot=np.zeros(np.shape(tensor),dtype=dtype)
    for n in range(d):
        Adot[:,:,n]=isqrtl.dot(VL[:,:,n]).dot(xopt).dot(isqrtr)
    normxopt=np.sqrt(np.tensordot(xopt,np.conj(xopt),([0,1],[0,1])))

    return Adot,h,normxopt,k

""" ====================================================           The following should not be used ============================================================="""
def LiebLinigerHAproduct(Hl,mpo,Hr,A,mps,B):
    D=mps.shape[0]
    term1=np.tensordot(Hl,mps,([0],[0]))
    term2=np.transpose(np.tensordot(mps,Hr,([1],[0])),(0,2,1))
    L=initializeLayer(A,np.eye(D),A,mpo[0],1)
    temp=np.tensordot(L,mps,([0],[0]))
    term3=np.transpose(np.tensordot(temp,mpo[1],([1,3],[0,3])),(0,1,3,2))[:,:,:,0]

    R=initializeLayer(B,np.eye(D),B,mpo[1],-1)
    temp=np.tensordot(mps,R,([1],[0]))
    term4=np.transpose(np.tensordot(temp,mpo[0],([3,1],[1,3])),(0,1,3,2))[:,:,:,0]

    return term1+term2+term3+term4
    

def tangentSpaceHAproduct(kleft,A,B,center,VL,mpo):
    D1,D2,d=np.shape(A)

    left=initializeLayer(center,np.eye(D1),VL,mpo[0],1)
    right=initializeLayer(B,np.eye(D1),B,mpo[1],-1)
    term1=np.tensordot(left,right,([0,2],[0,2]))

    left=initializeLayer(A,np.eye(D1),A,mpo[0],1)
    term2=np.transpose(addLayer(left,center,mpo[1],VL,1)[:,:,0],(1,0))

    term3=np.transpose(np.reshape(GeneralizedMatrixVectorProduct(1,VL,center,np.reshape(kleft,D1*D1)),(D1,D1)),(1,0))


    return term1+term2+term3

