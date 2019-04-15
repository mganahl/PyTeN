"""
@author: Martin Ganahl
"""

from __future__ import absolute_import, division, print_function
from distutils.version import StrictVersion
from sys import stdout
import sys, time, copy, warnings
import numpy as np
import lib.mpslib.sexpmv as sexpmv
try:
    MPSL = sys.modules['lib.mpslib.mps']
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
comm = lambda x, y: np.dot(x, y) - np.dot(y, x)
anticomm = lambda x, y: np.dot(x, y) + np.dot(y, x)
herm = lambda x: np.conj(np.transpose(x))


def svd(mat, full_matrices=False, compute_uv=True, r_thresh=1E-14):
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
        [u, s, v] = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        [q, r] = np.linalg.qr(mat)
        r[np.abs(r) < r_thresh] = 0.0
        u_, s, v = np.linalg.svd(r)
        u = q.dot(u_)
        print('caught a LinAlgError with dir>0')
    return u, s, v


def qr(mat, signfix):
    """
    a simple wrapper around numpy qr, allows signfixing of the diagonal of r or q
    """
    dtype = type(mat[0, 0])
    q, r = np.linalg.qr(mat)
    if signfix == 'q':
        sign = np.sign(np.diag(q))
        unit = np.diag(sign)
        return q.dot(unit), herm(unit).dot(r)
    if signfix == 'r':
        sign = np.sign(np.diag(r))
        unit = np.diag(sign)
        return q.dot(herm(unit)), unit.dot(r)


def mps_tensor_adder(A, B, boundary_type, ZA=1.0, ZB=1.0):
    """
    adds to Tensors A and B in the MPS fashion
    A,B:    Tensor objects
    boundary_type:  str
                    can be ('l','left',-1) or ('r','right',1) or ('b','bulk',0)
    ZA, ZB:   float 
              the coefficients in from of A and B, i.e. ZA * A + ZB * B
    """
    dtype = np.result_type(A, B)
    if A.shape[2] != B.shape[2]:
        raise ValueError('physical dimensions  of A and B are not compatible')
    if len(A.shape) != 3:
        raise ValueError('A is not an MPS tensor')
    if len(B.shape) != 3:
        raise ValueError('B is not an MPS tensor')
    if not type(A) == type(B):
        raise TypeError('type(A)!=type(B)')
    if boundary_type in ('left', 'l', -1):
        if np.sum(A.shape[0]) != 1:
            raise ValueError(
                'A.shape[0] is not one dimensional; this is incompatible with left open boundary conditions'
            )
        if np.sum(B.shape[0]) != 1:
            raise ValueError(
                'B.shape[0] is not one dimensional; this is incompatible with left open boundary conditions'
            )
        return A.concatenate([A * ZA, B * ZB], axis=1).view(type(A))

    elif boundary_type in ('right', 'r', 1):
        if np.sum(A.shape[1]) != 1:
            raise ValueError(
                'A.shape[1] is not one dimensional; this is incompatible with right open boundary conditions'
            )
        if np.sum(B.shape[1]) != 1:
            raise ValueError(
                'B.shape[1] is not one dimensional; this is incompatible with rig open boundary conditions'
            )
        return A.concatenate([A * ZA, B * ZB], axis=0).view(type(A))

    elif boundary_type in (0, 'b', 'bulk'):
        if isinstance(A, tnsr.Tensor) and isinstance(B, tnsr.Tensor):
            res = A.zeros(
                (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1], A.shape[2]),
                dtype=dtype)
            for indx in range(A.shape[2]):
                res[:, :, indx] = A.directSum(A[:, :, indx] * ZA,
                                              B[:, :, indx] * ZB)
            return res
        else:
            return NotImplemented


def transfer_operator(tensors_a, tensors_b, direction, x):
    """
    MPS transfer operator.
    Parameters:
    -----------------------
    tensors_a:  list of Tensor or np.ndarray
                mps tensors on the unconjugated side
    tensors_b:  list of Tensor or np.ndarray
                mps tensors on the conjugated side 
                user should pass them UNCONJUGATED, conjugation happens inside
    direction: int or str in ('l','left,1) or ('r','right',-1)
               direction 
    x:         np.ndarray or Tensor
               the input
    Returns: 
    ------------------------

    np.ndarray or Tensor:   result of the transfer operator acting on `x`
    
    """
    if len(tensors_a) != len(tensors_b):
        raise ValueError(
            'transfer_operator(): lengths of tensors_a and tensors_b are different'
        )
    if direction in ('l', 'left', 1):
        for n in range(len(tensors_a)):
            x = ncon.ncon([x, tensors_a[n], tensors_b[n].conj()],
                          [[1, 2], [1, -1, 3], [2, -2, 3]])
    if direction in ('r', 'right', -1):
        for n in reversed(range(len(tensors_a))):
            x = ncon.ncon([tensors_a[n], tensors_b[n].conj(), x],
                          [[-1, 1, 3], [-2, 2, 3], [1, 2]])
    return x


def prepare_tensor_SVD(tensor, direction, D=None, thresh=1E-32, r_thresh=1E-14):
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

    assert (direction != 0), 'do NOT use direction=0!'
    [l1, l2, d] = tensor.shape
    if direction in (1, 'l', 'left'):
        temp, merge_data = tensor.merge([[0, 2], [1]])
        u, s, v, _ = temp.svd(
            full_matrices=False, truncation_threshold=thresh, D=D)
        Z = np.sqrt(ncon.ncon([s, s], [[1], [1]]))
        s /= Z
        [size1, size2] = u.shape
        out = u.split([merge_data[0], [size2]]).transpose(0, 2, 1)
        return out, s, v, Z

    if direction in (-1, 'r', 'right'):
        temp, merge_data = tensor.merge([[0], [1, 2]])
        u, s, v, _ = temp.svd(
            full_matrices=False, truncation_threshold=thresh, D=D)
        Z = np.sqrt(ncon.ncon([s, s], [[1], [1]]))
        s /= Z
        [size1, size2] = v.shape
        out = v.split([[size1], merge_data[1]])
        return u, s, out, Z


def prepare_tensor_QR(tensor, direction,walltime_log=None):
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

    if len(tensor.shape) != 3:
        raise ValueError(
            'prepareTensor: ```tensor``` has to be of rank = 3. Found ranke = {0}'
            .format(len(tensor.shape)))
    [l1, l2, d] = tensor.shape
    if walltime_log:
        t1=time.time()
    if direction in (1, 'l', 'left'):
        temp, merge_data = tensor.merge([[0, 2], [1]])
        q, r = temp.qr()
        #normalize the bond matrix
        Z = np.sqrt(ncon.ncon([r, np.conj(r)], [[1, 2], [1, 2]]))
        r /= Z
        [size1, size2] = q.shape
        out = q.split([merge_data[0], [size2]]).transpose(0, 2, 1)
        if walltime_log:
            walltime_log(lan=[],QR=[time.time()-t1],add_layer=[],num_lan=[])
        
        return out, r, Z
    
    elif direction in (-1, 'r', 'right'):
        temp, merge_data = tensor.merge([[1, 2], [0]])
        temp = np.conj(temp)
        q, r_ = temp.qr()

        [size1, size2] = q.shape
        out = np.conj(q.split([merge_data[0], [size2]]).transpose(2, 0, 1))
        r = np.conj(np.transpose(r_, (1, 0)))
        #normalize the bond matrix
        Z = np.sqrt(ncon.ncon([r, np.conj(r)], [[1, 2], [1, 2]]))
        r /= Z
        if walltime_log:
            walltime_log(lan=[],QR=[time.time()-t1],add_layer=[],num_lan=[])
        return r, out, Z
    else:
        raise ValueError(
            "unkown value {} for input parameter direction".format(direction))


def ortho_deviation(tensor, which):
    """
    returns the deviation from left or right orthonormalization of the MPS tensors
    Parameters:
    -------------------
    tensor:   Tensor 
    which:    int or str 
              which should be either in ('l', 'left', 1) or ('r', 'right', -1)
    Returns:  
    ----------------
    float:    the deviation from orthogonality
    
    """
    if which in ('l', 'left', 1):
        return np.linalg.norm(
            ncon.ncon([tensor, tensor.conj()], [[1, -1, 2], [1, -2, 2]]) -
            tensor.eye(1))
    if which in ('r', 'right', -1):
        return np.linalg.norm(
            ncon.ncon([tensor, tensor.conj()], [[-1, 1, 2], [-2, 1, 2]]) -
            tensor.eye(0))
    else:
        raise ValueError(
            "wrong value {0} for variable ```which```; use ('l','r',1,-1,'left,'right')"
            .format(which))


def check_ortho(tensor, which, thresh=1E-8):
    """
    checks if orthogonality condition on tensor is obeyed up to ```thresh```
    """
    return MPS.ortho_deviation(tensor, which) < thresh


def add_layer(B, mps, mpo, conjmps, direction,walltime_log=None):
    """
        adds an mps-mpo-mps layer to a left or right block "E"; used in dmrg to calculate the left and right
        environments
        Parameters:
        ---------------------------
        B:        Tensor object  
                  a tensor of shape (D1,D1',M1) (for direction>0) or (D2,D2',M2) (for direction>0)
        mps:      Tensor object of shape =(Dl,Dr,d)
        mpo:      Tensor object of shape = (Ml,Mr,d,d')
        conjmps: Tensor object of shape =(Dl',Dr',d')
                 the mps tensor on the conjugated side
                 this tensor will be complex conjugated inside the routine; usually, the user will like to pass 
                 the unconjugated tensor
        direction: int or str
                  direction in (1,'l','left'): add a layer to the right of ```B```
                  direction in (-1,'r','right'): add a layer to the left of ```B```
        Return:
        -----------------
        Tensor of shape (Dr,Dr',Mr) for direction in (1,'l','left')
        Tensor of shape (Dl,Dl',Ml) for direction in (-1,'r','right')
        """
    
    if walltime_log:
        t1=time.time()
    if direction in ('l', 'left', 1):
        out = ncon.ncon([B, mps, mpo, np.conj(conjmps)],
                         [[1, 4, 3], [1, -1, 2], [3, -3, 5, 2], [4, -2, 5]])
    if direction in ('r', 'right', -1):
        out = ncon.ncon([B, mps, mpo, np.conj(conjmps)],
                         [[1, 5, 3], [-1, 1, 2], [-3, 3, 4, 2], [-2, 5, 4]])
    if walltime_log:
        walltime_log(lan=[],QR=[],add_layer=[time.time()-t1],num_lan=[])
    return out

def one_minus_pseudo_unitcell_transfer_op(direction, mps, left_dominant,
                                          right_dominant, vector):
    """
    calculates action of 11-Transfer-Operator +|r)(l|
    Parameters:
    ---------------------------
    direction:  int or str 
                if (1,'l','left'): do left multiplication
                if (-1,'r','right'): do right multiplication
    mps:        InfiniteMPSCentralGauge object
                an infinite mps
    left_dominant:  Tensor of shape (mps.D[0],mps.D[0])
                    left dominant eigenvvector of the unit-cell transfer operator of mps
    right_dominant: Tensor of shape (mps.D[-1],mps.D[-1])
                    right dominant eigenvvector of the unit-cell transfer operator of mps
    vector:         Tensor of shape (mps.D[0]*mps.D[0]) or (mps.D[-1]*mps.D[-1])
                    the input vector
    Returns
    ---------------------------
    np.ndarray of shape (mps.D[0]*mps.D[0]) or (mps.D[-1]*mps.D[-1])

    """

    if direction in (1, 'l', 'left'):
        x = type(mps[0]).from_dense(vector, [mps.D[0], mps.D[0]])
        temp = x - mps.unitcell_transfer_op('left', x) + ncon.ncon(
            [x, right_dominant], [[1, 2], [1, 2]]) * left_dominant
        return temp.to_dense()

    if direction in (-1, 'r', 'right'):
        x = type(mps[-1]).from_dense(vector, [mps.D[-1], mps.D[-1]])
        temp = x - mps.unitcell_transfer_op('right', x) + ncon.ncon(
            [left_dominant, x], [[1, 2], [1, 2]]) * right_dominant
        return temp.to_dense()


def LGMRES_solver(mps,
                  direction,
                  left_dominant,
                  right_dominant,
                  inhom,
                  x0,
                  precision=1e-10,
                  nmax=2000,
                  **kwargs):
    #mps.D[0] has to be mps.D[-1], so no distincion between direction='l' or direction='r' has to be made here
    if not mps.D[0] == mps.D[-1]:
        raise ValueError(
            'in LGMRES_solver: mps.D[0]!=mps.D[-1], can only handle intinite MPS!'
        )
    inhom_dense = inhom.to_dense()
    x0_dense = x0.to_dense()
    mv = fct.partial(one_minus_pseudo_unitcell_transfer_op,
                     *[direction, mps, left_dominant, right_dominant])

    LOP = LinearOperator((np.sum(mps.D[0])**2, np.sum(mps.D[-1])**2),
                         matvec=mv,
                         dtype=mps.dtype)
    out, info = lgmres(
        A=LOP,
        b=inhom_dense,
        x0=x0_dense,
        tol=precision,
        maxiter=nmax,
        **kwargs)

    return type(mps[0]).from_dense(out, [mps.D[0], mps.D[0]]), info


def compute_steady_state_Hamiltonian_GMRES(direction,
                                           mps,
                                           mpo,
                                           left_dominant,
                                           right_dominant,
                                           precision=1E-10,
                                           nmax=1000):
    """
    calculates the left or right Hamiltonain environment of an infinite MPS-MPO-MPS network
    Parameters:
    ---------------------------
    direction:  int or str 
                if (1,'l','left'): obtain left environment
                if (-1,'r','right'): obtain right environment
    mps:        InfiniteMPSCentralGauge object
                an infinite mps
    mpo:        MPO object
    left_dominant:  Tensor of shape (mps.D[0],mps.D[0])
                    left dominant eigenvvector of the unit-cell transfer operator of mps
    right_dominant: Tensor of shape (mps.D[-1],mps.D[-1])
                    right dominant eigenvvector of the unit-cell transfer operator of mps
    precision: float
               deisred precision of the environments
    nmax:      int
               maximum iteration numner

    Returns
    ---------------------------
    (H,h)
    H:    Tensor of shape (mps.D[0],mps.D[0],mpo.D[0])
          Hamiltonian environment
    h:    Tensor of shape (1)
          average energy per unitcell 
    """
    dummy1 = mpo.get_boundary_vector('l')
    dummy2 = mpo.get_boundary_vector('r')

    if direction in (1, 'l', 'left'):
        L = ncon.ncon(
            [
                mps.get_tensor(mps.num_sites - 1),
                #mpo.get_tensor(mps.num_sites-1),
                mpo.get_boundary_mpo('left'),
                mps.get_tensor(mps.num_sites - 1).conj()
            ],
            [[1, -1, 2], [-3, 4, 2], [1, -2, 4]])
        for n in range(len(mps)):
            L = add_layer(
                L,
                mps.get_tensor(n),
                mpo.get_tensor(n),
                mps.get_tensor(n),
                direction='l')

        h = ncon.ncon([L, dummy2, right_dominant], [[1, 2, 3], [3], [1, 2]])
        inhom = ncon.ncon([L, dummy2], [[-1, -2, 1], [1]]) - h * mps[-1].eye(1)
        [out, info] = LGMRES_solver(
            mps=mps,
            direction=direction,
            left_dominant=left_dominant,
            right_dominant=right_dominant,
            inhom=inhom,
            x0=inhom.zeros([mps.D[0], mps.D[0]], dtype=mps.dtype),
            precision=precision,
            nmax=nmax)
        L[:, :, 0] = out
        return L, h

    if direction in (-1, 'r', 'right'):
        R = ncon.ncon(
            [
                mps.get_tensor(0),
                #mpo.get_tensor(0),
                mpo.get_boundary_mpo('right'),
                mps.get_tensor(0).conj()
            ],
            [[-1, 1, 2], [-3, 4, 2], [-2, 1, 4]])
        for n in reversed(range(len(mps))):
            R = add_layer(
                R,
                mps.get_tensor(n),
                mpo.get_tensor(n),
                mps.get_tensor(n),
                direction='r')
        h = ncon.ncon([dummy1, left_dominant, R], [[3], [1, 2], [1, 2, 3]])
        inhom = ncon.ncon([dummy1, R], [[1], [-1, -2, 1]]) - h * mps[0].eye(0)
        [out, info] = LGMRES_solver(
            mps=mps,
            direction=direction,
            left_dominant=left_dominant,
            right_dominant=right_dominant,
            inhom=inhom,
            x0=inhom.zeros([mps.D[0], mps.D[0]], dtype=mps.dtype),
            precision=precision,
            nmax=nmax)

        R[:, :, -1] = out
        return R, h


def compute_Hamiltonian_environments(mps,
                                     mpo,
                                     precision=1E-10,
                                     precision_canonize=1E-10,
                                     nmax=1000,
                                     nmax_canonize=10000,
                                     ncv=40,
                                     numeig=6,
                                     pinv=1E-30):
    """
    calculates the Hamiltonain environments of an infinite MPS-MPO-MPS network
    Parameters:
    ---------------------------
    mps:        InfiniteMPSCentralGauge object
                an infinite mps
    mpo:        MPO object
    precision: float
               deisred precision of the environments
    precision_canonize: float
                        deisred precision for mps canonization
    nmax:      int
               maximum iteration numner
    nmax_canonize:      int
                        maximum iteration number in TMeigs during canonization
    ncv:       int
               number of krylov vectors in TMeigs during canonization
    numeig:    int
               number of eigenvectors targeted by sparse soler in TMeigs during canonization
    pinv:      float 
               pseudo inverse threshold during canonization

    Returns:
    --------------------
    (lb,rb,hl,hr)
    lb:      Tensor of shape (mps.D[0],mps.D[0],mpo.D[0])
             left Hamiltonian environment, including coupling of unit-cell to the left environment
    rb:      Tensor of shape (mps.D[-1],mps.D[-1],mpo.D[-1])
             right Hamiltonian environment, including coupling of unit-cell to the right environment
    hl:     Tensor of shape(1)
            average energy per left unitcell 
    hr:     Tensor of shape(1)
            average energy per right unitcell 
    NOTE:  hl and hr do not have to be identical
    """

    mps.canonize(
        precision=precision_canonize,
        ncv=ncv,
        nmax=nmax_canonize,
        numeig=numeig,
        pinv=pinv)
    mps.position(len(mps))
    lb, hl = compute_steady_state_Hamiltonian_GMRES(
        'l',
        mps,
        mpo,
        left_dominant=mps[-1].eye(1),
        right_dominant=ncon.ncon([mps.mat, mps.mat.conj()], [[-1, 1], [-2, 1]]),
        precision=precision,
        nmax=nmax)
    rmps = mps.get_right_orthogonal_imps(
        precision=precision_canonize,
        ncv=ncv,
        nmax=nmax_canonize,
        numeig=numeig,
        pinv=pinv,
        canonize=False)
    rb, hr = compute_steady_state_Hamiltonian_GMRES(
        'r',
        rmps,
        mpo,
        right_dominant=mps[0].eye(0),
        left_dominant=ncon.ncon([mps.mat, mps.mat.conj()], [[1, -1], [1, -2]]),
        precision=precision,
        nmax=nmax)
    return lb, rb, hl, hr


def HA_product(L, mpo, R, mps):
    """
    the local matrix vector product of the DMRG optimization
    Parameters:
    --------------------
    L:    tf.Tensor
          left environment of the local sites
    mpo:  tf.Tensor
          local mpo tensor
    R:    tf.Tensor
          right environment of the local sites
    mps: tf.Tensor
         local mps tensor
    Returns:
    ------------------
    tf.Tensor:   result of the local contraction
    
    """
    return ncon.ncon([L, mps, mpo, R],
                     [[1, -1, 2], [1, 4, 3], [2, 5, -3, 3], [4, -2, 5]])


def HA_product_vectorized(L, mpo, R, vector):
    x = type(L).from_dense(vector, [L.shape[0], R.shape[0], mpo.shape[2]])
    return HA_product(L, mpo, R, x).to_dense()


def eigsh(L,
          mpo,
          R,
          initial,
          precision=1e-6,
          numvecs=1,
          ncv=20,
          numvecs_calculated=1,
          *args,
          **kwargs):
    """
    sparse diagonalization of DMRG local hamiltonian
    L:                  Tensor object of shape (Dl,Dl',Ml)
    mpo:                Tensor object of shape (Ml,Mr,d,d')
    R:                  Tensor object of shape (Dr,Dr',Mr)
    initial:            Tensor object of shape (Dl,Dd,d)
    precision:          float
    numvecs:            int 
    ncv:                int 
    numvecs_calculated: int 

    """
    dtype = np.result_type(L.dtype, mpo.dtype, R.dtype, initial.dtype)
    chil = np.sum(L.shape[0])
    chir = np.sum(R.shape[0])
    chilp = np.sum(L.shape[1])
    chirp = np.sum(R.shape[1])
    d = mpo.shape[2]
    dp = mpo.shape[3]

    mv = fct.partial(HA_product_vectorized, *[L, mpo, R])
    LOP = LinearOperator((chil * chir * d, chilp * chirp * dp),
                         matvec=mv,
                         dtype=dtype)
    e, v = sp.sparse.linalg.eigsh(
        LOP,
        k=numvecs,
        which='SA',
        tol=precision,
        v0=initial.to_dense(),
        ncv=ncv)
    
    # if numvecs == 1:
    #     ind = np.nonzero(e == min(e))
    #     return e[ind[0][0]], initial.from_dense(v[:, ind[0][0]],
    #                                             (chilp, chirp, dp))

    # elif numvecs > 1:
    if (numvecs > numvecs_calculated):
        warnings.warn(
            'mpsfunctions.eigsh: requestion to return more vectors than calcuated: setting numvecs_returned=numvecs',
            stacklevel=2)
        numvecs = numvecs_calculated
    es = []
    vs = []
    esorted = np.sort(e)
    for n in range(numvecs):
        es.append(esorted[n])
        ind = np.nonzero(e == esorted[n])
        vs.append(initial.from_dense(v[:, ind[0][0]], (chilp, chirp, dp)))
    return es, vs


def lobpcg(L, mpo, R, initial, precision=1e-6, *args, **kwargs):
    """
    calls a sparse eigensolver to find the lowest eigenvalues and eigenvectors
    of the4 effective DMRG hamiltonian as given by L, mpo and R
    L (np.ndarray of shape (Dl,Dl',d)): left hamiltonian environment
    R (np.ndarray of shape (Dr,Dr',d)): right hamiltonian environment
    mpo (np.ndarray of shape (Ml,Mr,d)): MPO
    mps0 (np.ndarray of shape (Dl,Dr,d)): initial MPS tensor for the arnoldi solver
    see scipy eigsh documentation for details on the other parameters
    """

    dtype = np.result_type(L.dtype, mpo.dtype, R.dtype, initial.dtype)
    chil = np.sum(L.shape[0])
    chir = np.sum(R.shape[0])
    chilp = np.sum(L.shape[1])
    chirp = np.sum(R.shape[1])
    d = mpo.shape[2]
    dp = mpo.shape[3]
    mv = fct.partial(HA_product_vectorized, *[L, mpo, R])
    LOP = LinearOperator((chil * chir * d, chilp * chirp * dp),
                         matvec=mv,
                         dtype=dtype)

    X = np.expand_dims(initial.to_dense(), 1)

    e, v = sp.sparse.linalg.lobpcg(
        LOP, X=X, largest=False, tol=precision, *args, **kwargs)
    return e, [initial.from_dense(v, [chilp, chirp, dp])]




def TMeigs_naive(tensors,
                 direction,
                 init=None,
                 precision=1E-12,
                 nmax=100000):
    """
    calculate the left and right dominant eigenvector of the MPS-unit-cell transfer operator, 
    using the power method

    Parameters:
    ------------------------------
    tensors:       list of Tensor
    direction:     int or str

                   if direction in (1,'l','left')   return the left dominant EV
                   if direction in (-1,'r','right') return the right dominant EV
    init:          tf.tensor
                   initial guess for the eigenvector
    precision:     float
                   desired precision of the dominant eigenvalue
    nmax:          int
                   max number of iterations

    Returns:
    ------------------------------
    (eta,x, it, diff):
    eta:   float
           the eigenvalue
    x:     tf.tensor
           the dominant eigenvector (in matrix form)
    it:    int 
           the number of iterations taken
    diff:  float 
           the precision of the result
    """

    if not np.all(tensors[0].dtype == t.dtype for t in tensors):
        raise TypeError('TMeigs_naive: all tensors have to have the same dtype')

    if init:
        x = init
    else:
        x = tensors[0].eye(0)
    if not tensors[0].dtype == x.dtype:
        raise TypeError('TMeigs_naive: `init` has other dtype than `tensors`')

    x /= x.norm()
    diff = 1101001010001.0
    it = 0
    while diff > precision:
        x_new = transfer_operator(tensors, tensors, direction, x)
        eta = x_new.norm()
        x_new /= eta
        diff = (x - x_new).norm()
        x = x_new
        if it >= nmax:
            break
    return eta, x, it, diff




def TMeigs(tensors,
           direction,
           init=None,
           precision=1E-12,
           ncv=50,
           nmax=1000,
           numeig=6,
           which='LR'):
    """
    calculate the left and right dominant eigenvector of the MPS-unit-cell transfer operator

    Parameters:
    ------------------------------
    tensors:       list of Tensor
    direction:     int or str

                   if direction in (1,'l','left')   return the left dominant EV
                   if direction in (-1,'r','right') return the right dominant EV
    init:          tf.tensor
                   initial guess for the eigenvector
    precision:     float
                   desired precision of the dominant eigenvalue
    ncv:           int
                   number of Krylov vectors
    nmax:          int
                   max number of iterations
    numeig:        int
                   hyperparameter, passed to scipy.sparse.linalg.eigs; number of eigenvectors 
                   to be returned by scipy.sparse.linalg.eigs; leave at 6 to avoid problems with arpack
                   use numeig=1 for best performance (and sacrificing stability)
    which:         str
                   hyperparameter, passed to scipy.sparse.linalg.eigs; which eigen-vector to target
                   can be ('LM','LA,'SA','LR'), refer to scipy.sparse.linalg.eigs documentation for details

    Returns:
    ------------------------------
    (eta,x):
    eta: float
         the eigenvalue
    x:   tf.tensor
         the dominant eigenvector (in matrix form)
    """
    if not np.all(
        [tensors[0].dtype == tensors[m].dtype for m in range(len(tensors))]):
        raise TypeError('TMeigs: all tensors have to have the same dtype')
    dtype = tensors[0].dtype
    if np.sum(tensors[0].shape[0]) != np.sum(tensors[-1].shape[1]):
        raise ValueError(
            " in TMeigs: left and right ancillary dimensions of the MPS do not match"
        )
    if np.all(init != None):
        initial = init

    def mv(vector):
        return transfer_operator(
            tensors, tensors, direction,
            type(tensors[0]).from_dense(
                vector, [tensors[0].shape[0], tensors[0].shape[0]])).to_dense()

    LOP = LinearOperator(
        (np.sum(tensors[0].shape[0]) * np.sum(tensors[0].shape[0]),
         np.sum(tensors[-1].shape[1]) * np.sum(tensors[-1].shape[1])),
        matvec=mv,
        dtype=dtype)
    if numeig >= LOP.shape[0] - 1:
        warnings.warn(
            'TMeigs: numeig+1 ({0}) > dimension of transfer operator ({1}) changing value to numeig={2}'
            .format(numeig + 1, LOP.shape[0], LOP.shape[0] - 2))
        while numeig >= (LOP.shape[0] - 1):
            numeig -= 1

    eta, vec = eigs(
        LOP,
        k=numeig,
        which=which,
        v0=init,
        maxiter=nmax,
        tol=precision,
        ncv=ncv)
    m = np.argmax(np.real(eta))
    while np.abs(np.imag(eta[m])) / np.abs(np.real(eta[m])) > 1E-4:
        numeig = numeig + 1
        print(
            'found TM eigenvalue with large imaginary part (ARPACK BUG); recalculating with larger numeig={0}'
            .format(numeig))
        eta, vec = eigs(
            LOP,
            k=numeig,
            which=which,
            v0=init,
            maxiter=nmax,
            tol=precision,
            ncv=ncv)
        m = np.argmax(np.real(eta))

    if np.issubdtype(dtype, np.floating):
        out = type(tensors[0]).from_dense(
            vec[:, m], [tensors[0].shape[0], tensors[0].shape[0]])
        if np.linalg.norm(np.imag(out)) > 1E-10:
            raise TypeError(
                "TMeigs: dtype was float, but returned eigenvector had a large imaginary part; something went wrong here!"
            )
        return np.real(eta[m]), out.real
    elif np.issubdtype(dtype, np.complexfloating):
        return eta[m], type(tensors[0]).from_dense(
            vec[:, m], [tensors[0].shape[0], tensors[0].shape[0]])



    
def evolve_tensor_lan(left_env, mpo, right_env, mps, tau,
                    krylov_dimension=20, delta=1E-8):
    """
    evolve `mps` locally using a lanczos algorithm
    Parameters:
    ----------------
    left_env:           Tensor of shape (Dl,Dl,Ml)
    mpo:                Tensor of shape (Ml, Mr, d_out, d_in)
    right_env:          Tensor of shape (Dr,Dr,Mr)
    mps:                Tensor of shape (Dl, Dr, d_in)
    tau:                float or complex 
    krylov_dimension:   int 
    delta:              float

    Returns:
    ------------------
    Tensor: the evolved `mps`
    """
    def HAproduct(L, mpo, R, mps):
        return ncon.ncon([L, mps, mpo, R],
                         [[1, -1, 2], [1, 4, 3], [2, 5, -3, 3], [4, -2, 5]])
    
    def scalar_product(a, b):
        return ncon.ncon([a.conj(), b], [[1, 2, 3], [1, 2, 3]])
    
    mv=fct.partial(HAproduct,*[left_env, mpo, right_env])
    lan=lanEn.LanczosTimeEvolution(mv, scalar_product,
                                   ncv=krylov_dimension,
                                   delta=delta)
    return lan.do_step(mps,tau)



def evolve_matrix_lan(left_env, right_env, mat,
                    tau, krylov_dimension=20,
                    delta=1E-8):
    """
    evolve `mat` locally using a lanczos algorithm
    Parameters:
    ----------------
    left_env:           Tensor of shape (D,D,M)
    right_env:          Tensor of shape (D,D,M)
    mat:                Tensor of shape (D, D)
    tau:                float or complex 
    krylov_dimension:   int 
    delta:              float
    Returns:
    ------------------
    Tensor: the evolved `mat`

    """
    
    def HAproduct(L, R, mat):
        return ncon.ncon([L, mat, R],
                         [[1, -1, 2], [1, 3], [3, -2, 2]])
    
    def scalar_product(a, b):
        return ncon.ncon([a.conj(), b], [[1, 2], [1, 2]])
    
    mv=fct.partial(HAproduct,*[left_env,right_env])
    lan=lanEn.LanczosTimeEvolution(mv, scalar_product,
                                   ncv=krylov_dimension,
                                   delta=delta)
    return lan.do_step(mat,tau)
 
