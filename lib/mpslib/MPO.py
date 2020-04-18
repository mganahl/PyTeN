"""
@author: Martin Ganahl
"""
from __future__ import division
import pickle
import warnings
import os
import operator as opr
import copy
import numbers
import scipy as sp
import numpy as np
import lib.mpslib.mpsfunctions as mf
import lib.ncon as ncon
from lib.mpslib.TensorNetwork import TensorNetwork

comm = lambda x, y: np.dot(x, y) - np.dot(y, x)
anticomm = lambda x, y: np.dot(x, y) + np.dot(y, x)
herm = lambda x: np.conj(np.transpose(x))
from lib.mpslib.Tensor import TensorBase, Tensor


class MPOBase(TensorNetwork):
  """
    Base class for defining MPO; if you want to implement a custom MPO, derive from this class (see below for examples)
    Index convention:

           3
           |
          ___
         |   |
    0----|   |----1
         |___|
           |
           2

    2,3 are the physical outgoing and incoming indices, respectively. The conjugated 
    side of the MPS is on the bottom (at index 2)

    An MPS by this convention is contracted from above:
                ___
             --|   |--
                ---
                 |
                ___
               |   |
           ----|   |----
               |___|
                 |
    
    """

  def __init__(self, tensors, name, fromview):
    super().__init__(tensors=tensors, shape=(), name=name, fromview=fromview)

  @property
  def D(self):
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return ([self._tensors[0].shape[0]] +
            [self._tensors[n].shape[1] for n in range(len(self._tensors))])

  @property
  def d_in(self):
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return ([self._tensors[n].shape[2] for n in range(len(self._tensors))])

  @property
  def d_out(self):
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return ([self._tensors[n].shape[2] for n in range(len(self._tensors))])

  def get_tensor(self, n):
    return self._tensors[n]

  def get_2site_mpo(self, *args, **kwargs):
    raise NotImplementedError()

  def get_2site_hamiltonian(self, *args, **kwargs):
    raise NotImplementedError()

  def get_2site_gate(self, site1, site2, tau):
    """
        calculate the unitary two-site gates exp(tau*H(m,n))
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the gate
        tau:         float or complex
                     time-increment
        Returns:
        --------------------------------------------------
        A two-site gate "Gate" between sites m and n by summing up  (morally, for m<n)
        h=\sum_s np.kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:]) and exponentiating the result:
        Gate=scipy.linalg..expm(tau*h); 
        Gate is a rank-4 tensor with shape (dm,dn,dm,dn), with
        dm, dn the local hilbert space dimension at site m and n, respectively
        """
    if site2 < site1:
      d1 = self[site2].shape[2]
      d2 = self[site1].shape[2]
    elif site2 > site1:
      d1 = self[site1].shape[2]
      d2 = self[site2].shape[2]
    else:
      raise ValuError(
          'MPO.get_2site_gate: site1 has to be different from site2!')
    h = np.reshape(self.get_2site_hamiltonian(site1, site2), (d1 * d2, d1 * d2))
    return np.reshape(sp.linalg.expm(tau * h), (d1, d2, d1, d2)).view(type(h))


class FiniteMPO(MPOBase):

  def __init__(self, tensors, name=None, fromview=True):
    super().__init__(tensors=tensors, name=name, fromview=fromview)
    if not (np.sum(self.D[0]) == 1 and np.sum(self.D[-1]) == 1):
      raise ValueError(
          'FiniteMPO: left and right MPO ancillary dimension is different from 1'
      )

  def get_2site_mpo(self, site1, site2):
    if site2 < site1:
      mpo1 = self[site2][-1, :, :, :]
      mpo2 = self[site1][:, 0, :, :]

    if site2 > site1:
      mpo1 = self[site1][-1, :, :, :]
      mpo2 = self[site2][:, 0, :, :]

    d1 = mpo1.shape[1]
    d2 = mpo2.shape[1]

    return [
        np.expand_dims(mpo1, 0).view(type(mpo1)),
        np.expand_dims(mpo2, 1).view(type(mpo2))
    ]

  def get_2site_hamiltonian(self, site1, site2):
    """
        obtain a two-site Hamiltonian H_{mn} from MPO
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the Hamiltonian
        Returns:
        --------------------------------------------------
        np.ndarray of shape (d1,d2,d3,d4)
        A two-site Hamiltonian between sites ```site1``` and ```site2``` by summing up  
        (for site1<site2, and site1!=0, site2!=0)
        h=np.kron(mpo[m][-1,s=0,:,:]/2,mpo[n][s=0,0,:,:])+
          \sum_s={1}^{M-2} np.kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:])+
          np.kron(mpo[m][-1,s=M-1,:,:],mpo[n][s=M-1,0,:,:])+
        the returned np.ndarray is a rank-4 tensor with shape (dsite1,dsite2,dsite1,dsite2), with
        dsite1, dsite2 the local hilbert space dimension at sites ```site1``` and ```site2```, respectively,
        
        """
    mpo1, mpo2 = self.get_2site_mpo(site1, site2)
    if site2 < site1:
      nl = site2
      mr = site1
    elif site2 > site1:
      nl = site1
      nr = site2

    mpo1 = mpo1[0, :, :, :]
    mpo2 = mpo2[:, 0, :, :]
    d1 = mpo1.shape[1]
    d2 = mpo2.shape[1]
    if nl != 0 and nr != (len(self) - 1):
      h = np.kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
      for s in range(1, mpo1.shape[0] - 1):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
      h += np.kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)

    elif nl != 0 and nr == (len(self) - 1):
      h = np.kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
      for s in range(1, mpo1.shape[0]):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])

    elif nl == 0 and nr != (len(self) - 1):
      h = np.kron(mpo1[0, :, :], mpo2[0, :, :])
      for s in range(1, mpo1.shape[0] - 1):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
      h += np.kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)

    elif nl == 0 and nr == (len(self) - 1):
      h = np.kron(mpo1[0, :, :], mpo2[0, :, :])
      for s in range(1, mpo1.shape[0]):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
    return np.reshape(h, (d1, d2, d1, d2)).view(type(mpo1))


class InfiniteMPO(MPOBase):

  def __init__(self, tensors, name=None, fromview=True):
    super().__init__(tensors=tensors, name=name, fromview=fromview)
    if not (np.sum(self.D[0]) == np.sum(self.D[-1])):
      raise ValueError(
          'InfiniteMPO: left and right MPO ancicllary dimension differ')

  def get_boundary_vector(self, side):
    """
        return a one dimensional array which can be contracted with 
        the mpo to yield a boundary mpo
        Parameters:
        ----------------------
        side:  str in ('l','left','r','right)
               the side for which the boundary vector is required
        
        """

    if side.lower() in ('l', 'left'):
      v = np.zeros(self.D[0], dtype=self.dtype)
      v[-1] = 1.0
      return v.view(type(self._tensors[0]))

    if side.lower() in ('r', 'right'):
      v = np.zeros(self.D[-1], dtype=self.dtype)
      v[0] = 1.0
      return v.view(type(self._tensors[-1]))

  def get_boundary_mpo(self, side):
    """
        return a boundary mpo
        Parameters:
        -------------------
        side:  str in ('l','left','r','right)
               the side for which the boundary vector is required
        
        """

    if side.lower() in ('l', 'left'):
      out = copy.deepcopy(self._tensors[-1][-1, :, :, :])
      out[0, :, :] *= 0.0
    if side.lower() in ('r', 'right'):
      out = copy.deepcopy(self._tensors[0][:, 0, :, :])
      out[-1, :, :] *= 0.0
    return out.squeeze()

  def get_2site_mpo(self, site1, site2):
    if site2 < site1:
      mpo1 = copy.deepcopy(self[site2][-1, :, :, :])
      mpo2 = copy.deepcopy(self[site1][:, 0, :, :])
      if site2 == 0:
        mpo1[0, :, :] /= 2.0
      if site1 == (len(self) - 1):
        mpo2[-1, :, :] /= 2.0

    if site2 > site1:
      mpo1 = copy.deepcopy(self[site1][-1, :, :, :])
      mpo2 = copy.deepcopy(self[site2][:, 0, :, :])
      if site1 == 0:
        mpo1[0, :, :] /= 2.0
      if site2 == (len(self) - 1):
        mpo2[-1, :, :] /= 2.0

    assert (mpo1.shape[0] == mpo2.shape[0])
    d1 = mpo1.shape[1]
    d2 = mpo2.shape[1]

    return [
        np.expand_dims(mpo1, 0).view(type(mpo1)),
        np.expand_dims(mpo2, 1).view(type(mpo2))
    ]

  def get_2site_hamiltonian(self, site1, site2):
    """
        obtain a two-site Hamiltonian H_{mn} from MPO
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the Hamiltonian
        Returns:
        --------------------------------------------------
        np.ndarray of shape (d1,d2,d3,d4)
        A two-site Hamiltonian between sites ```site1``` and ```site2``` by summing up  
        (for site1<site2, and site1!=0, site2!=0)

        \sum_s={0}^{M-1} np.kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:])

        the returned np.ndarray is a rank-4 tensor with shape (dsite1,dsite2,dsite1,dsite2), with
        dsite1, dsite2 the local hilbert space dimension at sites ```site1``` and ```site2```, respectively,
        
        """
    mpo1, mpo2 = self.get_2site_mpo(site1, site2)
    if site2 < site1:
      nl = site2
      mr = site1
    elif site2 > site1:
      nl = site1
      nr = site2
    mpo1 = mpo1[0, :, :, :]
    mpo2 = mpo2[:, 0, :, :]

    d1 = mpo1.shape[1]
    d2 = mpo2.shape[1]

    h = np.kron(mpo1[0, :, :], mpo2[0, :, :])
    for s in range(1, mpo1.shape[0]):
      h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
    return np.reshape(h, (d1, d2, d1, d2)).view(type(mpo1))

  def roll(self, num_sites):
    tensors=[self._tensors[n] for n in range(num_sites,len(self._tensors))]\
        + [self._tensors[n] for n in range(num_sites)]
    self.set_tensors(tensors)


class FiniteTFI(FiniteMPO):
  """ 
    the good old transverse field Ising MPO
    convention: sigma_z=diag([-1,1])
    this is Jx * X * X + Bz * Z    
    """

  def __init__(self, Jx, Bz, dtype=np.float64):
    dtype = np.result_type(Jx.dtype, Bz.dtype, dtype)
    self.Jx = Jx.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([-1, 1]).astype(dtype)
    mpo = []
    temp = Tensor.zeros((1, 3, 2, 2), dtype)
    #Bsigma_z
    temp[0, 0, :, :] = self.Bz[0] * sigma_z
    #sigma_x
    temp[0, 1, :, :] = self.Jx[0] * sigma_x
    #11
    temp[0, 2, 0, 0] = 1.0
    temp[0, 2, 1, 1] = 1.0
    mpo.append(temp.copy())
    for n in range(1, N - 1):
      temp = Tensor.zeros((3, 3, 2, 2), dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #sigma_x
      temp[1, 0, :, :] = sigma_x
      #Bsigma_z
      temp[2, 0, :, :] = self.Bz[n] * sigma_z
      #sigma_x
      temp[2, 1, :, :] = self.Jx[n] * sigma_x
      #11
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp.copy())

    temp = Tensor.zeros((3, 1, 2, 2), dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #sigma_x
    temp[1, 0, :, :] = sigma_x
    #Bsigma_z
    temp[2, 0, :, :] = self.Bz[-1] * sigma_z

    mpo.append(temp.copy())

    super().__init__(tensors=mpo, name='FiniteTFI_MPO')


class FiniteTFI_2(FiniteMPO):
  """ 
    the good old transverse field Ising MPO
    convention: sigma_z=diag([-1,1])
    this is Jz * Z*Z + Bx * X
    
    """

  def __init__(self, Jz, Bx, dtype=np.float64):
    dtype = np.result_type(Jz.dtype, Bx.dtype, dtype)
    self.Jz = Jz.astype(dtype)
    self.Bx = Bx.astype(dtype)
    N = len(Bx)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([-1, 1]).astype(dtype)
    mpo = []
    temp = Tensor.zeros((1, 3, 2, 2), dtype)
    #Bsigma_z
    temp[0, 0, :, :] = self.Bx[0] * sigma_x
    #sigma_x
    temp[0, 1, :, :] = self.Jz[0] * sigma_z
    #11
    temp[0, 2, 0, 0] = 1.0
    temp[0, 2, 1, 1] = 1.0
    mpo.append(temp.copy())
    for n in range(1, N - 1):
      temp = Tensor.zeros((3, 3, 2, 2), dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #sigma_x
      temp[1, 0, :, :] = sigma_z
      #Bsigma_z
      temp[2, 0, :, :] = self.Bx[n] * sigma_x
      #sigma_x
      temp[2, 1, :, :] = self.Jz[n] * sigma_z
      #11
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp.copy())

    temp = Tensor.zeros((3, 1, 2, 2), dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #sigma_x
    temp[1, 0, :, :] = sigma_z
    #Bsigma_z
    temp[2, 0, :, :] = self.Bx[-1] * sigma_x

    mpo.append(temp.copy())

    super().__init__(tensors=mpo, name='FiniteTFI_MPO')


class FiniteTFI2D(FiniteMPO):
  """ 
  the good old transverse field Ising MPO
  convention: sigma_z=diag([-1,1])
  this is X * X + Z    
  """

  def __init__(self, Jx1, Jx2, Bz, N1, N2, dtype=np.float64):
    self.Jx1 = Jx1
    self.Jx2 = Jx2
    self.N1 = N1
    self.N2 = N2
    self.Bz = Bz
    eye = np.eye(2).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([-1, 1]).astype(dtype)
    mpo_dim = 3 + N1 - 1
    mpo_matrix = Tensor.zeros((1, mpo_dim, 2, 2), dtype=dtype)
    mpo_matrix[0, 0, :, :] = Bz * sigma_z
    mpo_matrix[0, 1, :, :] = Jx1 * sigma_x
    mpo_matrix[0, mpo_dim - 2, :, :] = Jx2 * sigma_x
    mpo_matrix[0, mpo_dim - 1, :, :] = eye
    mpo = []
    mpo.append(mpo_matrix.copy())
    n2 = 0
    #print('(n1, n2) = ({}, {}), Jx1, Jx2 = ({},{})'.format(
    #    0 % N1, n2, Jx1, Jx2))
    for n in range(1, N1 * N2 - 1):
      if (n + 1) % N2 == 0:
        jx1 = 0
      else:
        jx1 = Jx1

      if n < N1 * (N2 - 1):
        jx2 = Jx2
      else:
        jx2 = 0
      if n % N1 == 0:
        n2 += 1
      #print('(n1, n2) = ({}, {}), Jx1, Jx2 = ({},{})'.format(
      #    n % N1, n2, jx1, jx2))
      mpo_dim = 3 + N1 - 1
      mpo_matrix = Tensor.zeros((mpo_dim, mpo_dim, 2, 2), dtype=dtype)
      mpo_matrix[0, 0, :, :] = eye
      mpo_matrix[1, 0, :, :] = sigma_x
      for n1 in range(2, mpo_dim):
        mpo_matrix[n1, n1 - 1, :, :] = eye
      mpo_matrix[mpo_dim - 1, 0, :, :] = Bz * sigma_z
      mpo_matrix[mpo_dim - 1, 1, :, :] = jx1 * sigma_x
      mpo_matrix[mpo_dim - 1, mpo_dim - 2, :, :] = jx2 * sigma_x
      mpo_matrix[mpo_dim - 1, mpo_dim - 1, :, :] = eye
      mpo.append(mpo_matrix.copy())

    mpo_matrix = Tensor.zeros((mpo_dim, 1, 2, 2), dtype=dtype)
    mpo_matrix[0, 0, :, :] = eye
    mpo_matrix[1, 0, :, :] = sigma_x
    mpo_matrix[mpo_dim - 1, 0, :, :] = Bz * sigma_z
    #print('(n1, n2) = ({}, {}), Jx1, Jx2 = ({},{})'.format(
    #    N1 - 1 % N1, n2, 0, 0))
    mpo.append(mpo_matrix.copy())
    super().__init__(tensors=mpo, name='Finite2DTFI_MPO')


class InfiniteTFI(InfiniteMPO):
  """ 
    the good old transverse field Ising MPO
    convention: sigma_z=diag([-1,1])
    
    """

  def __init__(self, Jx, Bz, dtype=np.float64):
    dtype = np.result_type(Jx.dtype, Bz.dtype, dtype)
    self.Jx = Jx.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([-1, 1]).astype(dtype)
    mpo = []
    for n in range(0, N):
      temp = Tensor.zeros((3, 3, 2, 2), dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #sigma_x
      temp[1, 0, 1, 0] = 1
      temp[1, 0, 0, 1] = 1
      #Bsigma_z
      temp[2, 0:, :] = sigma_z * self.Bz[n]
      #sigma_x
      temp[2, 1, :, :] = sigma_x * self.Jx[n]
      #11
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp.copy())

    super().__init__(tensors=mpo, name='InfiniteTFI_MPO')


class FiniteXXZ(FiniteMPO):
  """
    the famous Heisenberg Hamiltonian, which we all know and love so much!
    """

  def __init__(self, Jz, Jxy, Bz, dtype=np.float64):
    dtype = np.result_type(Jz.dtype, Jxy.dtype, Bz.dtype, dtype)
    self.Jz = Jz.astype(dtype)
    self.Jxy = Jxy.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    mpo = []
    temp = Tensor.zeros((1, 5, 2, 2), dtype)
    #BSz
    temp[0, 0, 0, 0] = -0.5 * Bz[0]
    temp[0, 0, 1, 1] = 0.5 * Bz[0]

    #Sm
    temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
    #Sp
    temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
    #Sz
    temp[0, 3, 0, 0] = Jz[0] * (-0.5)
    temp[0, 3, 1, 1] = Jz[0] * 0.5

    #11
    temp[0, 4, 0, 0] = 1.0
    temp[0, 4, 1, 1] = 1.0
    mpo.append(temp.copy())
    for n in range(1, N - 1):
      temp = Tensor.zeros((5, 5, 2, 2), dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sp
      temp[1, 0, 1, 0] = 1.0
      #Sm
      temp[2, 0, 0, 1] = 1.0
      #Sz
      temp[3, 0, 0, 0] = -0.5
      temp[3, 0, 1, 1] = 0.5
      #BSz
      temp[4, 0, 0, 0] = -0.5 * Bz[n]
      temp[4, 0, 1, 1] = 0.5 * Bz[n]

      #Sm
      temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
      #Sp
      temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
      #Sz
      temp[4, 3, 0, 0] = Jz[n] * (-0.5)
      temp[4, 3, 1, 1] = Jz[n] * 0.5
      #11
      temp[4, 4, 0, 0] = 1.0
      temp[4, 4, 1, 1] = 1.0

      mpo.append(temp.copy())

    temp = Tensor.zeros((5, 1, 2, 2), dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #Sp
    temp[1, 0, 1, 0] = 1.0
    #Sm
    temp[2, 0, 0, 1] = 1.0
    #Sz
    temp[3, 0, 0, 0] = -0.5
    temp[3, 0, 1, 1] = 0.5
    #BSz
    temp[4, 0, 0, 0] = -0.5 * Bz[-1]
    temp[4, 0, 1, 1] = 0.5 * Bz[-1]

    mpo.append(temp.copy())
    super().__init__(mpo)


class FiniteXX(FiniteMPO):
  """
  """

  def __init__(self, Jxy, Bz, dtype=np.float64):
    dtype = np.result_type(Jxy.dtype, Bz.dtype, dtype)
    self.Jxy = Jxy.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    mpo = []
    temp = Tensor.zeros((1, 4, 2, 2), dtype)
    #BSz
    temp[0, 0, :, :] = np.diag([-0.5, 0.5]) * Bz[0]
    #Sm
    temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
    #Sp
    temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
    #11
    temp[0, 3, :, :] = np.eye(2)

    mpo.append(temp.copy())
    for n in range(1, N - 1):
      temp = Tensor.zeros((4, 4, 2, 2), dtype)
      #11
      temp[0, 0, :, :] = np.eye(2)
      #Sp
      temp[1, 0, 1, 0] = 1.0
      #Sm
      temp[2, 0, 0, 1] = 1.0
      #BSz
      temp[3, 0, :, :] = np.diag([-0.5, 0.5]) * Bz[n]

      #Sm
      temp[3, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
      #Sp
      temp[3, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
      #11
      temp[3, 3, :, :] = np.eye(2)
      mpo.append(temp.copy())

    temp = Tensor.zeros((4, 1, 2, 2), dtype)
    #11
    temp[0, 0, :, :] = np.eye(2)
    #Sp
    temp[1, 0, 1, 0] = 1.0
    #Sm
    temp[2, 0, 0, 1] = 1.0
    #BSz
    temp[3, 0, :, :] = np.diag([-0.5, 0.5]) * Bz[-1]

    mpo.append(temp.copy())
    super().__init__(mpo)


class Finite2DXXX(FiniteMPO):
  """
    the famous Heisenberg Hamiltonian, which we all know and love so much!
      |---J2---|
      |        |
     J1        J1
      |        |
      |---J2---|
    

    """

  def __init__(self, Jz1, Jxy1, Jz2, Jxy2, Bz, N1, N2, dtype=np.float64):
    self.Jz1 = Jz1
    self.Jxy1 = Jxy1
    self.Jz2 = Jz2
    self.Jxy2 = Jxy2

    self.N1 = N1
    self.N2 = N2
    self.Bz = Bz

    Sx = np.array([[0, 0.5], [0.5, 0]])
    Sy = np.array([[0, 0.5j], [-0.5j, 0]])
    Sz = np.diag([-0.5, 0.5])

    mpo_dim = 3 * N1 + 2
    mpo = []
    mat = Tensor.zeros((1, mpo_dim, 2, 2), dtype=np.complex128)
    mat[0, 0, :, :] = Bz * Sz
    mat[0, 1] = Jxy1 * Sx
    mat[0, 2] = Jxy1 * Sy
    mat[0, 3] = Jz1 * Sz
    mat[0, 3 * (N1 - 1) + 1, :, :] = Jxy2 * Sx
    mat[0, 3 * (N1 - 1) + 2, :, :] = Jxy2 * Sy
    mat[0, 3 * (N1 - 1) + 3, :, :] = Jz2 * Sz
    mat[0, -1, :, :] = np.eye(2)
    mpo.append(mat.copy())
    for n2, n1 in itertools.product(list(range(N2)), list(range(N1))):
      if ((n1, n2) == (0, 0)) or ((n1, n2) == (N1 - 1, N2 - 1)):
        continue
      if n1 == N1 - 1:
        jz1 = 0
        jxy1 = 0
      else:
        jz1 = Jz1
        jxy1 = Jxy1

      if n2 < N2 - 1:
        jz2 = Jz2
        jxy2 = Jxy2
      else:
        j1_hor = 0
        j2_1 = 0
        j2_2 = 0

      if (n + 1) % N2 == 0:
        j1 = 0
      else:
        j1 = J1

      if n < N1 * (N2 - 1):
        j2 = J2
      else:
        j2 = 0
      if n % N1 == 0:
        n2 += 1

      mat = Tensor.zeros((mpo_dim, mpo_dim, 2, 2), dtype=np.complex128)
      mat[0, 0] = np.eye(2)
      mat[1, 0] = Sx
      mat[2, 0] = Sy
      mat[3, 0] = Sz
      for ind in range(4, 3 * (N1 - 1) + 4):
        mat[ind, ind - 3, :, :] = np.eye(2)
      mat[-1, 0, :, :] = Bz * Sz
      mat[-1, 1] = j1 * Sx
      mat[-1, 2] = j1 * Sy
      mat[-1, 3] = j1 * Sz

      mat[-1, 3 * (N1 - 1) + 1, :, :] = j2 * Sx
      mat[-1, 3 * (N1 - 1) + 2, :, :] = j2 * Sy
      mat[-1, 3 * (N1 - 1) + 3, :, :] = j2 * Sz
      mat[-1, -1, :, :] = np.eye(2)
      mpo.append(mat.copy())

    mat = Tensor.zeros((mpo_dim, 1, 2, 2), dtype=np.complex128)
    mat[0, 0] = np.eye(2)
    mat[1, 0] = Sx
    mat[2, 0] = Sy
    mat[3, 0] = Sz
    mat[-1, 0, :, :] = Bz * Sz
    mpo.append(mat.copy())

    super().__init__(tensors=mpo, name='Finite2DXXZ_MPO')


class InfiniteXXZ(InfiniteMPO):

  def __init__(self, Jz, Jxy, Bz, dtype=np.float64):
    """
        XXZ model on an infinite systems
        Args:
            Jz  (np.ndarray):  Jz coupling
            Jxy (np.ndarray):  Jxy coupling
            Bz  (np.ndarray):  magnetic field
        """
    dtype = np.result_type(Jz.dtype, Jxy.dtype, Bz.dtype, dtype)
    self.Jz = Jz.astype(dtype)
    self.Jxy = Jxy.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    mpo = []
    for n in range(0, N):

      temp = Tensor.zeros((5, 5, 2, 2), dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sp
      temp[1, 0, 1, 0] = 1.0
      #Sm
      temp[2, 0, 0, 1] = 1.0
      #Sz
      temp[3, 0, 0, 0] = -0.5
      temp[3, 0, 1, 1] = 0.5
      #BSz
      temp[4, 0, 0, 0] = -0.5 * Bz[n]
      temp[4, 0, 1, 1] = 0.5 * Bz[n]

      #Sm
      temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
      #Sp
      temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
      #Sz
      temp[4, 3, 0, 0] = Jz[n] * (-0.5)
      temp[4, 3, 1, 1] = Jz[n] * 0.5
      #11
      temp[4, 4, 0, 0] = 1.0
      temp[4, 4, 1, 1] = 1.0

      mpo.append(temp.copy())
    super().__init__(mpo)


class FiniteJ1J2(FiniteMPO):
  """
      returns the MPO of the finite J1-J2 model
      Args:
        J1 (np.ndarray or tf.Tensor):  the S*S coupling strength between nearest neighbor lattice sites
        J2 (np.ndarray or tf.Tensor):  the S*S coupling strength between next-nearest neighbor lattice sites
        Bz (np.ndarray or tf.Tensor):  magnetic field on each lattice site
        dtype (tf-dtype or numpy dtype): the dtype of the MPO
      Returns:
        FiniteJ1J2:   the mpo of the finite J1J2 model
      """

  def __init__(self, J1, J2, Bz, dtype=np.float64):
    dtype = np.result_type(J1.dtype, J2.dtype, Bz.dtype, dtype)
    self.J1 = J1.astype(dtype)
    self.J2 = J2.astype(dtype)
    self.Bz = Bz.astype(dtype)
    Sm = np.array([[0, 1], [0, 0]]).astype(dtype)
    Sp = np.array([[0, 0], [1, 0]]).astype(dtype)
    Sz = np.diag([-0.5, 0.5]).astype(dtype)
    eye = np.eye(2).astype(dtype)

    N = len(Bz)
    mpo = []
    temp = Tensor.zeros((1, 8, 2, 2), dtype)

    temp[0, 0, :, :] = -Bz[0] * Sz
    temp[0, 1, :, :] = J1[0] / 2 * Sm
    temp[0, 2, :, :] = J1[0] / 2 * Sp
    temp[0, 3, :, :] = J1[0] * Sz
    temp[0, 4, :, :] = J2[0] / 2 * Sm
    temp[0, 5, :, :] = J2[0] / 2 * Sp
    temp[0, 6, :, :] = J2[0] * Sz
    temp[0, 7, :, :] = eye

    mpo.append(temp.copy())
    for n in range(1, N - 1):
      temp = Tensor.zeros((8, 8, 2, 2), dtype)
      temp[0, 0, :, :] = eye
      temp[1, 0, :, :] = Sp
      temp[2, 0, :, :] = Sm
      temp[3, 0, :, :] = Sz
      temp[4, 1, :, :] = eye
      temp[5, 2, :, :] = eye
      temp[6, 3, :, :] = eye

      temp[7, 0, :, :] = -Bz[n] * Sz
      temp[7, 1, :, :] = J1[n] / 2 * Sm
      temp[7, 2, :, :] = J1[n] / 2 * Sp
      temp[7, 3, :, :] = J1[n] * Sz
      temp[7, 4, :, :] = J2[n] / 2 * Sm
      temp[7, 5, :, :] = J2[n] / 2 * Sp
      temp[7, 6, :, :] = J2[n] * Sz
      temp[7, 7, :, :] = eye

      mpo.append(temp.copy())

    temp = Tensor.zeros((8, 1, 2, 2), dtype)
    temp[0, 0, :, :] = eye
    temp[1, 0, :, :] = Sp
    temp[2, 0, :, :] = Sm
    temp[3, 0, :, :] = Sz
    temp[7, 0, :, :] = -Bz[-1] * Sz

    mpo.append(temp.copy())
    super().__init__(mpo)


class Finite2D_J1J2(FiniteMPO):
  """ 
  
    
      |\-j1_hor/|
      | \     / |
      |  \  j2_1|
      |   \ /   |
  j1_vert  X  j1_vert
      |   / \   |
      |  /  j2_2|
      | /     \ |
      |/-j1_hor\|
    
  """

  def __init__(self, J1, J2, N1, N2, dtype=np.float64, points=[]):

    self.J1 = J1
    self.J2 = J2
    self.N1 = N1
    self.N2 = N2

    Sx = np.array([[0, 0.5], [0.5, 0]])
    Sy = np.array([[0, 0.5], [-0.5, 0]])  #imaginary part is added below
    Sz = np.diag([-0.5, 0.5])

    mpo_dim = 3 * (N1 + 1) + 2
    mpo = []

    mat = np.zeros((1, mpo_dim, 2, 2), dtype=dtype)
    mat[0, 1] = J1 * Sx
    mat[0, 2] = -J1 * Sy
    mat[0, 3] = J1 * Sz
    mat[0, 3 * (N1 - 2) + 1, :, :] = 0 * Sx
    mat[0, 3 * (N1 - 2) + 2, :, :] = -0 * Sy
    mat[0, 3 * (N1 - 2) + 3, :, :] = 0 * Sz

    mat[0, 3 * (N1 - 1) + 1, :, :] = J1 * Sx
    mat[0, 3 * (N1 - 1) + 2, :, :] = -J1 * Sy
    mat[0, 3 * (N1 - 1) + 3, :, :] = J1 * Sz

    mat[0, 3 * N1 + 1, :, :] = J2 * Sx
    mat[0, 3 * N1 + 2, :, :] = -J2 * Sy
    mat[0, 3 * N1 + 3, :, :] = J2 * Sz
    mat[0, -1, :, :] = np.eye(2)
    assert (mat.shape[1] == 3 * N1 + 5)
    mpo.append(mat.copy())
    points.append((0, 0, J1, J1, J2, 0))
    for n2, n1 in itertools.product(list(range(N2)), list(range(N1))):
      if ((n1, n2) == (0, 0)) or ((n1, n2) == (N1 - 1, N2 - 1)):
        continue
      if n1 == N1 - 1:
        j1_vert = 0
        j2_1 = 0
      else:
        j1_vert = J1
        j2_1 = J2

      if n1 == 0:
        j2_2 = 0
      else:
        j2_2 = J2

      if n2 < N2 - 1:
        j1_hor = J1
      else:
        j1_hor = 0
        j2_1 = 0
        j2_2 = 0
      points.append((n1, n2, j1_vert, j1_hor, j2_1, j2_2))
      mat = np.zeros((mpo_dim, mpo_dim, 2, 2), dtype=dtype)
      mat[0, 0] = np.eye(2)
      mat[1, 0] = Sx
      mat[2, 0] = Sy
      mat[3, 0] = Sz
      for ind in range(4, 3 * N1 + 4):  #one more than XXX
        mat[ind, ind - 3, :, :] = np.eye(2)

      mat[-1, 1] = j1_vert * Sx
      mat[-1, 2] = -j1_vert * Sy
      mat[-1, 3] = j1_vert * Sz

      mat[-1, 3 * (N1 - 2) + 1, :, :] = j2_2 * Sx
      mat[-1, 3 * (N1 - 2) + 2, :, :] = -j2_2 * Sy
      mat[-1, 3 * (N1 - 2) + 3, :, :] = j2_2 * Sz

      mat[-1, 3 * (N1 - 1) + 1, :, :] = j1_hor * Sx
      mat[-1, 3 * (N1 - 1) + 2, :, :] = -j1_hor * Sy
      mat[-1, 3 * (N1 - 1) + 3, :, :] = j1_hor * Sz

      mat[-1, 3 * N1 + 1, :, :] = j2_1 * Sx
      mat[-1, 3 * N1 + 2, :, :] = -j2_1 * Sy
      mat[-1, 3 * N1 + 3, :, :] = j2_1 * Sz

      mat[-1, -1, :, :] = np.eye(2)
      # print(
      #     '(n1, n2) = ({}, {}), J1_vert, J1_hor = ({},{}), J2_1, J2_2=({},{})'.
      #     format(n % N1, n2, j1_vert, j1_hor, j2_1, j2_2))
      mpo.append(mat.copy())

    mat = np.zeros((mpo_dim, 1, 2, 2), dtype=dtype)
    mat[0, 0] = np.eye(2)
    mat[1, 0] = Sx
    mat[2, 0] = Sy
    mat[3, 0] = Sz
    mpo.append(mat.copy())
    super().__init__(tensors=mpo, name='Finite2DJ1J2_MPO')

    super().__init__(tensors=mpo, name='Finite2DJ1J2_MPO')


# class XXZIsing(MPO):
#     """
#     the famous Heisenberg Hamiltonian, which we all know and love so much!
#     """

#     def __init__(self, J, w, obc=True):
#         self.obc = obc
#         self.J = J
#         self.w = w
#         N = len(w)
#         sig_x = np.asarray([[0, 1], [1, 0]]).astype(complex)
#         sig_y = np.asarray([[0, -1j], [1j, 0]]).astype(complex)
#         sig_z = np.diag([1, -1]).astype(complex)
#         if obc == True:
#             if len(J) != (N - 1):
#                 raise ValueError(
#                     "in JessHamiltonian: for obc=True, len(J) has to be N-1")

#             mpo = []
#             temp = Tensor.zeros((1, 5, 2, 2), complex)
#             temp[0, 0, :, :] = self.w[0] * sig_z
#             temp[0, 1, :, :] = self.J[0] * sig_x
#             temp[0, 2, :, :] = self.J[0] * sig_y
#             temp[0, 3, :, :] = self.J[0] * sig_z
#             temp[0, 4, :, :] = np.eye(2)

#             mpo.append(np.copy(temp))
#             for n in range(1, N - 1):
#                 temp = Tensor.zeros((5, 5, 2, 2), complex)
#                 temp[0, 0, :, :] = np.eye(2)
#                 temp[1, 0, :, :] = sig_x
#                 temp[2, 0, :, :] = sig_y
#                 temp[3, 0, :, :] = sig_z
#                 temp[4, 0, :, :] = self.w[n] * sig_z
#                 temp[4, 1, :, :] = self.J[n] * sig_x
#                 temp[4, 2, :, :] = self.J[n] * sig_y
#                 temp[4, 3, :, :] = self.J[n] * sig_z
#                 temp[4, 4, :, :] = np.eye(2)
#                 mpo.append(np.copy(temp))

#             temp = Tensor.zeros((5, 1, 2, 2), complex)
#             temp[0, 0, :, :] = np.eye(2)
#             temp[1, 0, :, :] = sig_x
#             temp[2, 0, :, :] = sig_y
#             temp[3, 0, :, :] = sig_z
#             temp[4, 0, :, :] = self.w[-1] * sig_z
#             mpo.append(np.copy(temp))
#             super(XXZIsing, self).__init__(mpo)

# class XXZflipped(MPO):
#     """
#     again the famous Heisenberg Hamiltonian, but in a slightly less common form
#     """

#     def __init__(self, Delta, J, obc=True, dtype=float):
#         self._Delta = Delta
#         self.J = J
#         self.obc = obc
#         mpo = []
#         if obc == True:
#             N = len(Delta) + 1
#             temp = Tensor.zeros((1, 5, 2, 2)).astype(dtype)
#             #Sx
#             temp[0, 1, 0, 1] = J[0]
#             temp[0, 1, 1, 0] = J[0]
#             #Sy
#             temp[0, 2, 0, 1] = +1j * (-J[0])
#             temp[0, 2, 1, 0] = -1j * (-J[0])

#             #Sz
#             temp[0, 3, 0, 0] = -Delta[0] * J[0] * (-1.0)
#             temp[0, 3, 1, 1] = -Delta[0] * J[0] * 1.0
#             #11
#             temp[0, 4, 0, 0] = 1.0
#             temp[0, 4, 1, 1] = 1.0
#             mpo.append(temp)
#             for site in range(1, N - 1):
#                 temp = Tensor.zeros((5, 5, 2, 2)).astype(dtype)
#                 #11
#                 temp[0, 0, 0, 0] = 1.0
#                 temp[0, 0, 1, 1] = 1.0
#                 #Sx
#                 temp[1, 0, 0, 1] = 1.0
#                 temp[1, 0, 1, 0] = 1.0
#                 #Sy
#                 temp[2, 0, 0, 1] = +1j
#                 temp[2, 0, 1, 0] = -1j

#                 #Sz
#                 temp[3, 0, 0, 0] = -1.0
#                 temp[3, 0, 1, 1] = +1.0

#                 #Sx
#                 temp[4, 1, 0, 1] = J[0]
#                 temp[4, 1, 1, 0] = J[0]
#                 #Sy
#                 temp[4, 2, 0, 1] = +1j * (-J[0])
#                 temp[4, 2, 1, 0] = -1j * (-J[0])

#                 #Sz
#                 temp[4, 3, 0, 0] = -Delta[site] * J[site] * (-1.0)
#                 temp[4, 3, 1, 1] = -Delta[site] * J[site] * 1.0
#                 #11
#                 temp[4, 4, 0, 0] = 1.0
#                 temp[4, 4, 1, 1] = 1.0
#                 mpo.append(temp)

#             temp = Tensor.zeros((5, 1, 2, 2)).astype(dtype)
#             #11
#             temp[0, 0, 0, 0] = 1.0
#             temp[0, 0, 1, 1] = 1.0
#             #Sx
#             temp[1, 0, 0, 1] = 1.0
#             temp[1, 0, 1, 0] = 1.0
#             #Sy
#             temp[2, 0, 0, 1] = +1j
#             temp[2, 0, 1, 0] = -1j
#             #Sz
#             temp[3, 0, 0, 0] = -1.0
#             temp[3, 0, 1, 1] = +1.0

#             mpo.append(temp)
#             super(XXZflipped, self).__init__(mpo)
#         if obc == False:
#             N = len(Delta)
#             for site in range(N):
#                 temp = Tensor.zeros((5, 5, 2, 2)).astype(dtype)
#                 #11
#                 temp[0, 0, 0, 0] = 1.0
#                 temp[0, 0, 1, 1] = 1.0
#                 #Sx
#                 temp[1, 0, 0, 1] = 1.0
#                 temp[1, 0, 1, 0] = 1.0
#                 #Sy
#                 temp[2, 0, 0, 1] = +1j
#                 temp[2, 0, 1, 0] = -1j

#                 #Sz
#                 temp[3, 0, 0, 0] = -1.0
#                 temp[3, 0, 1, 1] = +1.0

#                 #Sx
#                 temp[4, 1, 0, 1] = J[0]
#                 temp[4, 1, 1, 0] = J[0]
#                 #Sy
#                 temp[4, 2, 0, 1] = +1j * (-J[0])
#                 temp[4, 2, 1, 0] = -1j * (-J[0])

#                 #Sz
#                 temp[4, 3, 0, 0] = -Delta[site] * J[site] * (-1.0)
#                 temp[4, 3, 1, 1] = -Delta[site] * J[site] * 1.0
#                 #11
#                 temp[4, 4, 0, 0] = 1.0
#                 temp[4, 4, 1, 1] = 1.0
#                 mpo.append(temp)
#             super(XXZflipped, self).__init__(mpo)

# class SpinlessFermions(MPO):
#     """
#     Spinless Fermions, you know them ...
#     """

#     def __init__(self, interaction, hopping, chempot, obc, dtype=complex):
#         assert (len(interaction) == len(hopping))
#         assert (len(interaction) == len(chempot) - 1)

#         mpo = []
#         self._interaction = interaction
#         self._hoping = hopping
#         self._chempot = chempot
#         self.obc = obc

#         N = len(chempot)
#         c = Tensor.zeros((2, 2)).astype(dtype)
#         c[0, 1] = 1.0
#         P = np.diag([1.0, -1.0]).astype(dtype)

#         if obc == True:
#             tensor = Tensor.zeros((1, 5, 2, 2)).astype(dtype)
#             tensor[0, 0, :, :] = chempot[0] * herm(c).dot(c)
#             tensor[0, 1, :, :] = (-1.0) * hopping[0] * herm(c).dot(P)
#             tensor[0, 2, :, :] = (+1.0) * hopping[0] * c.dot(P)
#             tensor[0, 3, :, :] = interaction[0] * herm(c).dot(c)
#             tensor[0, 4, :, :] = np.eye(2)
#             mpo.append(np.copy(tensor))

#             for n in range(1, N - 1):
#                 tensor = Tensor.zeros((5, 5, 2, 2)).astype(dtype)
#                 tensor[0, 0, :, :] = np.eye(2)
#                 tensor[1, 0, :, :] = c
#                 tensor[2, 0, :, :] = herm(c)
#                 tensor[3, 0, :, :] = herm(c).dot(c)
#                 tensor[4, 0, :, :] = chempot[n] * herm(c).dot(c)

#                 tensor[4, 1, :, :] = (-1.0) * hopping[n] * herm(c).dot(P)
#                 tensor[4, 2, :, :] = (+1.0) * hopping[n] * c.dot(P)
#                 tensor[4, 3, :, :] = interaction[n] * herm(c).dot(c)
#                 tensor[4, 4, :, :] = np.eye(2)

#                 mpo.append(np.copy(tensor))
#             tensor = Tensor.zeros((5, 1, 2, 2)).astype(dtype)
#             tensor[0, 0, :, :] = np.eye(2)
#             tensor[1, 0, :, :] = c
#             tensor[2, 0, :, :] = herm(c)
#             tensor[3, 0, :, :] = herm(c).dot(c)
#             tensor[4, 0, :, :] = chempot[N - 1] * herm(c).dot(c)
#             mpo.append(np.copy(tensor))
#             super(SpinlessFermions, self).__init__(mpo)
#         if obc == False:
#             for n in range(N):
#                 tensor = Tensor.zeros((5, 5, 2, 2)).astype(dtype)
#                 tensor[0, 0, :, :] = np.eye(2)
#                 tensor[1, 0, :, :] = c.dot(P)
#                 tensor[2, 0, :, :] = herm(c).dot(P)
#                 tensor[3, 0, :, :] = herm(c).dot(c)
#                 tensor[4, 0, :, :] = chempot[n] * herm(c).dot(c)

#                 tensor[4, 1, :, :] = (-1.0) * hopping[n] * herm(c)
#                 tensor[4, 2, :, :] = (+1.0) * hopping[n] * c
#                 tensor[4, 3, :, :] = interaction[n] * herm(c).dot(c)
#                 tensor[4, 4, :, :] = np.eye(2)
#                 mpo.append(np.copy(tensor))
#             super(SpinlessFermions, self).__init__(mpo)

# class HubbardChain(MPO):
#     """
#     The good old Fermi Hubbard model
#     """

#     def __init__(self, U, t_up, t_down, mu_up, mu_down, obc, dtype=complex):
#         self._U = U
#         self._t_up = t_up
#         self._t_down = t_down
#         self._mu_up = mu_up
#         self._mu_down = mu_down
#         self.obc = obc
#         N = len(U)

#         assert (len(mu_up) == N)
#         assert (len(mu_down) == N)
#         assert (len(t_up) == N - 1)
#         assert (len(t_down) == N - 1)

#         mpo = []
#         c = Tensor.zeros((2, 2), dtype=dtype)
#         c[0, 1] = 1.0
#         c_down = np.kron(c, np.eye(2))
#         c_up = np.kron(np.diag([1.0, -1.0]).astype(dtype), c)
#         P = np.diag([1.0, -1.0, -1.0, 1.0]).astype(dtype)

#         if obc == True:
#             tensor = Tensor.zeros((1, 6, 4, 4)).astype(dtype)
#             tensor[0, 0, :, :] = mu_up[0] * herm(c_up).dot(
#                 c_up) + mu_down[0] * herm(c_down).dot(c_down) + U[0] * herm(
#                     c_up).dot(c_up).dot(herm(c_down).dot(c_down))
#             tensor[0, 1, :, :] = (-1.0) * t_up[0] * herm(c_up).dot(P)
#             tensor[0, 2, :, :] = (+1.0) * t_up[0] * c_up.dot(P)
#             tensor[0, 3, :, :] = (-1.0) * t_down[0] * herm(c_down).dot(P)
#             tensor[0, 4, :, :] = (+1.0) * t_down[0] * c_down.dot(P)
#             tensor[0, 5, :, :] = np.eye(4)
#             mpo.append(np.copy(tensor))

#             for n in range(1, N - 1):
#                 tensor = Tensor.zeros((6, 6, 4, 4)).astype(dtype)
#                 tensor[0, 0, :, :] = np.eye(4)
#                 tensor[1, 0, :, :] = c_up
#                 tensor[2, 0, :, :] = herm(c_up)
#                 tensor[3, 0, :, :] = c_down
#                 tensor[4, 0, :, :] = herm(c_down)

#                 tensor[5, 0, :, :] = mu_up[n] * herm(c_up).dot(
#                     c_up) + mu_down[n] * herm(c_down).dot(c_down) + U[n] * herm(
#                         c_up).dot(c_up).dot(herm(c_down).dot(c_down))

#                 tensor[5, 1, :, :] = (-1.0) * t_up[n] * herm(c_up).dot(P)
#                 tensor[5, 2, :, :] = (+1.0) * t_up[n] * c_up.dot(P)
#                 tensor[5, 3, :, :] = (-1.0) * t_down[n] * herm(c_down).dot(P)
#                 tensor[5, 4, :, :] = (+1.0) * t_down[n] * c_down.dot(P)
#                 tensor[5, 5, :, :] = np.eye(4)

#                 mpo.append(np.copy(tensor))

#             tensor = Tensor.zeros((6, 1, 4, 4)).astype(dtype)
#             tensor[0, 0, :, :] = np.eye(4)
#             tensor[1, 0, :, :] = c_up
#             tensor[2, 0, :, :] = herm(c_up)
#             tensor[3, 0, :, :] = c_down
#             tensor[4, 0, :, :] = herm(c_down)
#             tensor[5, 0, :, :] = mu_up[N - 1] * herm(c_up).dot(c_up) + mu_down[
#                 N - 1] * herm(c_down).dot(c_down) + U[N - 1] * herm(c_up).dot(
#                     c_up).dot(herm(c_down).dot(c_down))
#             mpo.append(np.copy(tensor))
#             super(HubbardChain, self).__init__(mpo)
#         if obc == False:
#             for n in range(N):
#                 tensor = Tensor.zeros((6, 6, 4, 4)).astype(dtype)
#                 tensor[0, 0, :, :] = np.eye(4)
#                 tensor[1, 0, :, :] = c_up
#                 tensor[2, 0, :, :] = herm(c_up)
#                 tensor[3, 0, :, :] = c_down
#                 tensor[4, 0, :, :] = herm(c_down)

#                 tensor[5, 0, :, :] = mu_up[n] * herm(c_up).dot(
#                     c_up) + mu_down[n] * herm(c_down).dot(c_down) + U[n] * herm(
#                         c_up).dot(c_up).dot(herm(c_down).dot(c_down))

#                 tensor[5, 1, :, :] = (-1.0) * t_up[n] * herm(c_up).dot(P)
#                 tensor[5, 2, :, :] = (+1.0) * t_up[n] * c_up.dot(P)
#                 tensor[5, 3, :, :] = (-1.0) * t_down[n] * herm(c_down).dot(P)
#                 tensor[5, 4, :, :] = (+1.0) * t_down[n] * c_down.dot(P)
#                 tensor[5, 5, :, :] = np.eye(4)

#                 mpo.append(np.copy(tensor))
#             super(HubbardChain, self).__init__(mpo)
