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
    
def transfer_operator(A,B,direction,x):
    """
    """
    if direction in ('l','left',1):
        return ncon.ncon([x,A,B.conj()],[[1,2],[1,-1,3],[2,-2,3]])
    if direction in ('r','right',-1):        
        return ncon.ncon([A,B.conj(),x],[[-1,1,3],[-2,2,3],[1,2]])


def prepare_tensor_SVD(tensor,direction,D=None,thresh=1E-32,r_thresh=1E-14):
    """
    prepares and truncates an mps tensor using svd
    Parameters:
    ---------------------
    tensor: np.ndarray of shape(D1,D2,d)
            an mps tensor
    direction: int
               if >0 returns left orthogonal decomposition, if <0 returns right orthogonal decomposition
    thresh: float
            cutoff of schmidt-value truncation
    r_thresh: float
              only used when svd throws an exception.
    D:        int or None
              the maximum bond-dimension to keep (hard cutoff); if None, no truncation is applied

    Returns:
    ----------------------------
    direction>0: out,s,v,Z
                 out: a left isometric tensor of dimension (D1,D,d)
                 s  : the singular values of length D
                 v  : a right isometric matrix of dimension (D,D2)
                 Z  : the norm of tensor, i.e. tensor"="out.dot(s).dot(v)*Z
    direction<0: u,s,out,Z
                 u  : a left isometric matrix of dimension (D1,D)
                 s  : the singular values of length D
                 out: a right isometric tensor of dimension (D,D2,d)
                 Z  : the norm of tensor, i.e. tensor"="u.dot(s).dot(out)*Z

    """

    assert(direction!=0),'do NOT use direction=0!'
    [l1,l2,d]=tensor.shape
    if direction in (1,'l','left'):
        temp,merge_data=tensor.merge([[0,2],[1]])
        [u,s,v]=temp.svd(full_matrices=False,truncation_threshold=thresh,D=D)
        Z=np.sqrt(ncon.ncon([s,s],[[1],[1]]))
        s/=Z
        [size1,size2]=u.shape
        out=u.split([merge_data[0],[size2]]).transpose(0,2,1)
        return out,s,v,Z

    if direction in (-1,'r','right'):
        temp,merge_data=tensor.merge([[0],[1,2]])
        [u,s,v]=temp.svd(full_matrices=False,truncation_threshold=thresh,D=D)
        Z=np.sqrt(ncon.ncon([s,s],[[1],[1]]))
        s/=Z
        [size1,size2]=v.shape
        out=v.split([[size1],merge_data[1]])
        return u,s,out,Z
    
def prepare_tensor_QR(tensor,direction):
    """
    orthogonalizes an mps tensor using qr decomposition 

    Parameters:
    ----------------------------------
    tensor: np.ndarray of shape(D1,D2,d)
            an mps tensor

    direction: int
               direction in {1,'l','left'}: returns left orthogonal decomposition, 
               direction in {-1,'r','right'}: returns right orthogonal decomposition, 
    fixphase:  str
              fixphase can be in {'q','r'} fixes the phase of the diagonal of q or r to be real and positive
    Returns: 
    -------------------------------------
    (out,r,Z)
    out: np.ndarray
         a left or right isometric mps tensor
    r:   np.ndarray
         an upper or lower triangular matrix
    Z:   float
         the norm of the input tensor, i.e. tensor"="out x r x Z (direction in {1.'l','left'} or tensor"=r x out x Z (direction in {-1,'r','right'}
    """
    
    if len(tensor.shape)!=3:
        raise ValueError('prepareTensor: ```tensor``` has to be of rank = 3. Found ranke = {0}'.format(len(tensor.shape)))
    [l1,l2,d]=tensor.shape
    if direction in (1,'l','left'):
        temp,merge_data=tensor.merge([[0,2],[1]])
        q,r=temp.qr()
        #normalize the bond matrix
        Z=np.sqrt(ncon.ncon([r,np.conj(r)],[[1,2],[1,2]]))
        r/=Z
        [size1,size2]=q.shape
        out=q.split([merge_data[0],[size2]]).transpose(0,2,1)
        return out,r,Z            
    elif direction in (-1,'r','right'):
        temp,merge_data=tensor.merge([[1,2],[0]])
        temp=np.conj(temp)
        q,r_=temp.qr()

        [size1,size2]=q.shape
        out=np.conj(q.split([merge_data[0],[size2]]).transpose(2,0,1))
        r=np.conj(np.transpose(r_,(1,0)))
        #normalize the bond matrix
        Z=np.sqrt(ncon.ncon([r,np.conj(r)],[[1,2],[1,2]]))
        r/=Z
        return r,out,Z
    else:
        raise ValueError("unkown value {} for input parameter direction".format(direction))


def ortho_deviation(tensor,which):
    """
    returns the deviation from left or right orthonormalization of the MPS tensors
    """
    if which in ('l','left',1):
        return np.linalg.norm(ncon.ncon([tensor,tensor.conj()],[[1,-1,2],[1,-2,2]])-tensor.eye(1))
    if which in ('r','right',-1):
        return np.linalg.norm(ncon.ncon([tensor,tensor.conj()],[[-1,1,2],[-2,1,2]])-tensor.eye(0))
    else:
        raise ValueError("wrong value {0} for variable ```which```; use ('l','r',1,-1,'left,'right')".format(which))
        

def check_ortho(tensor,which,thresh=1E-8):
    """
    checks if orthogonality condition on tensor is obeyed up to ```thresh```
    """
    return MPS.ortho_deviation(tensor,which)<thresh
        
        
    



