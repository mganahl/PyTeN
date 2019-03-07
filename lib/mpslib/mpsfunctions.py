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
    
def transferOperator(A,B,direction,x):
    """
    """
    if direction in ('l','left',1):
        return ncon([x,A,B.conj()],[(0,1),(0,2,-1),(1,2,-2)])
    if direction in ('r','right',-1):        
        return ncon([A,B.conj(),x],[(-1,1,0),(-2,1,2),(0,2)])



