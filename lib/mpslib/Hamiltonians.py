"""
@author: Martin Ganahl
"""
import numpy as np
import time, copy
import scipy as sp

comm = lambda x, y: np.dot(x, y) - np.dot(y, x)
anticomm = lambda x, y: np.dot(x, y) + np.dot(y, x)
herm = lambda x: np.conj(np.transpose(x))


class MPO:
  """

    Base class for defining MPO; if you want to implement a custom MPO, derive from this class (see below for examples)
    
    """

  def __init__(self, mpos=[]):
    #a list of mpo-tensors, to be initialized in the derived class
    self._mpo = copy.deepcopy(mpos)
    self._N = len(self._mpo)

  @classmethod
  def fromMPO(cls, other):
    #a list of mpo-tensors, to be initialized in the derived class
    return cls(other._mpolist)

  @classmethod
  def fromlist(cls, mpolist):
    return cls(mpolist)
    #a list of mpo-tensors, to be initialized in the derived class
    #cls._mpo=copy.deepcopy(mpolist)
    #cls._N=len(mpolist)
    #return cls

  def twoSiteGate(self, m, n, tau):
    """
        returns a two-site gate "Gate" between sites m and n by summing up  (morally, for m<n)
        h=\sum_s np.kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:]) and exponentiating the result:
        Gate=scipy.linalg..expm(tau*h); Gate is a rank-4 tensor with shape (dm,dn,dm,dn), with
        dm, dn the local hilbert space dimension at site m and n, respectively
        """

    if n < m:
      mpo1 = self._mpo[n][-1, :, :, :]
      mpo2 = self._mpo[m][:, 0, :, :]
      nl = n
      mr = m
    if n > m:
      mpo1 = self._mpo[m][-1, :, :, :]
      mpo2 = self._mpo[n][:, 0, :, :]
      nl = m
      nr = n
    assert (mpo1.shape[0] == mpo2.shape[0])
    d1 = mpo1.shape[1]
    d2 = mpo2.shape[1]
    if nl != 0 and nr != (self._N - 1):
      h = np.kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
      for s in range(1, mpo1.shape[0] - 1):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
      h += np.kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)

    elif nl != 0 and nr == (self._N - 1):
      h = np.kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
      for s in range(1, mpo1.shape[0]):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
    elif nl == 0 and nr != (self._N - 1):
      h = np.kron(mpo1[0, :, :], mpo2[0, :, :])
      for s in range(1, mpo1.shape[0] - 1):
        h += np.kron(mpo1[s, :, :], mpo2[s, :, :])
      h += np.kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)
    Gate = np.reshape(sp.linalg.expm(tau * h), (d1, d2, d1, d2))
    #Gate=np.reshape(np.eye(4),(d1,d2,d1,d2))

    return Gate

  def __getitem__(self, n):
    return self._mpo[n]

  def __setitem__(self, n, tensor):
    self._mpo[n] = tensor

  def __len__(self):
    return len(self._mpo)

  @property
  def dtype(self):
    return self._dtype


class TFI_2(MPO):
  """ 
    transverse field Ising MPO
    convention: sigma_z=diag([-0.5,0.5])
    
    """

  def __init__(self, Jz, Bx, obc=True, dtype=float):
    self._obc = obc
    self._Jz = Jz
    self._Bx = Bx
    self._N = len(Bx)
    self._dtype = dtype
    sig_x = np.array([[0, 1], [1, 0]])
    sig_z = np.diag([-1, 1])
    if obc == True:
      self._mpo = []
      temp = np.zeros((1, 3, 2, 2), self.dtype)
      #Bx
      temp[0, 0, :, :] = self._Bz[0] * sig_x
      #Sz
      temp[0, 1, :, :] = self._Jz[0] * sig_z
      #11
      temp[0, 2, :, :] = np.eye(2)

      self._mpo.append(np.copy(temp))
      for n in range(1, self._N - 1):
        temp = np.zeros((3, 3, 2, 2), self.dtype)
        #11
        temp[0, 0, :, :] = np.eye(2)
        #Sz
        temp[1, 0, :, :] = sig_z

        #Bx Sx
        temp[2, 0, :, :] = self._Bx[n] * sig_x

        #Sz
        temp[2, 1, :, :] = self._Jz[n] * sig_z
        #11
        temp[2, 2, :, :] = np.eye(2)

        self._mpo.append(np.copy(temp))

      temp = np.zeros((3, 1, 2, 2), self.dtype)
      #11
      temp[0, 0, :, :] = np.eye(2)
      #Sz
      temp[1, 0, :, :] = sig_z
      #Bx Sx
      temp[2, 0, :, :] = sig_x * self._Bz[-1]
      self._mpo.append(np.copy(temp))

    if obc == False:
      assert (len(Bx) == len(Jz))
      self._mpo = []
      for n in range(0, self._N):
        temp = np.zeros((3, 3, 2, 2), self.dtype)
        #11
        temp[0, 0, :, :] = np.eye(2)
        #Sz
        temp[1, 0, :, :] = sig_z

        #Bx Sx
        temp[2, 0, :, :] = self._Bx[n] * sig_x

        #Sz
        temp[2, 1, :, :] = self._Jz[n] * sig_z
        #11
        temp[2, 2, :, :] = np.eye(2)
        self._mpo.append(np.copy(temp))


class TFI(MPO):
  """ 
    transverse field Ising MPO
    convention: sigma_z=diag([-0.5,0.5])
    
    """

  def __init__(self, Jx, Bz, obc=True, dtype=float):
    self._obc = obc
    self._Jx = Jx
    self._Bz = Bz
    self._N = len(Bz)
    self._dtype = dtype
    if obc == True:
      self._mpo = []
      temp = np.zeros((1, 3, 2, 2), self.dtype)
      #BSz
      temp[0, 0, 0, 0] = -self._Bz[0]
      temp[0, 0, 1, 1] = self._Bz[0]

      #Sx
      temp[0, 1, 0, 1] = self._Jx[0]
      temp[0, 1, 1, 0] = self._Jx[0]
      #11
      temp[0, 2, 0, 0] = 1.0
      temp[0, 2, 1, 1] = 1.0
      self._mpo.append(np.copy(temp))
      for n in range(1, self._N - 1):
        temp = np.zeros((3, 3, 2, 2), self.dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #Sx
        temp[1, 0, 1, 0] = 1.0
        temp[1, 0, 0, 1] = 1.0
        #BSz
        temp[2, 0, 0, 0] = -self._Bz[n]
        temp[2, 0, 1, 1] = self._Bz[n]
        #Sx
        temp[2, 1, 0, 1] = self._Jx[n]
        temp[2, 1, 1, 0] = self._Jx[n]
        #11
        temp[2, 2, 0, 0] = 1.0
        temp[2, 2, 1, 1] = 1.0

        self._mpo.append(np.copy(temp))

      temp = np.zeros((3, 1, 2, 2), self.dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sx
      temp[1, 0, 1, 0] = 1.0
      temp[1, 0, 0, 1] = 1.0
      #BSz
      temp[2, 0, 0, 0] = -Bz[-1]
      temp[2, 0, 1, 1] = Bz[-1]

      self._mpo.append(np.copy(temp))

    if obc == False:
      assert (len(Bz) == len(Jx))
      self._mpo = []
      for n in range(0, self._N):
        temp = np.zeros((3, 3, 2, 2), self.dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #Sx
        temp[1, 0, 1, 0] = 1
        temp[1, 0, 0, 1] = 1
        #BSz
        temp[2, 0, 0, 0] = -self._Bz[n]
        temp[2, 0, 1, 1] = self._Bz[n]
        #Sx
        temp[2, 1, 0, 1] = self._Jx[n]
        temp[2, 1, 1, 0] = self._Jx[n]
        #11
        temp[2, 2, 0, 0] = 1.0
        temp[2, 2, 1, 1] = 1.0
        self._mpo.append(np.copy(temp))


class JessHamiltonian(MPO):
  """
    
    """

  def __init__(self, J, w, chi, obc=True):
    self._obc = obc
    self._J = J
    self._w = w
    self._chi = chi
    self._dtype = complex
    self._N = len(chi)

    sigma_x = np.asarray([[0, 1], [1, 0]])
    sigma_y = np.asarray([[0, -1j], [1j, 0]])
    sigma_z = np.diag([1, -1])

    sig_x = np.kron(sigma_x, np.eye(2))
    sig_y = np.kron(sigma_y, np.eye(2))
    sig_z = np.kron(sigma_z, np.eye(2))
    tau_x = np.kron(np.eye(2), sigma_x)
    tau_y = np.kron(np.eye(2), sigma_y)
    tau_z = np.kron(np.eye(2), sigma_z)

    if obc == True:
      if len(J) != (self._N - 1):
        raise ValueError(
            "in JessHamiltonian: for obc=True, len(J) has to be N-1")

      self._mpo = []
      temp = np.zeros((1, 8, 4, 4), self.dtype)
      temp[0, 0, :, :] = self._w[0] * (sig_x + tau_x) + self._chi[0] * np.kron(
          sigma_x, sigma_x)
      temp[0, 1, :, :] = self._J[0] * sig_x
      temp[0, 2, :, :] = self._J[0] * sig_y
      temp[0, 3, :, :] = self._J[0] * sig_z
      temp[0, 4, :, :] = self._J[0] * tau_x
      temp[0, 5, :, :] = self._J[0] * tau_y
      temp[0, 6, :, :] = self._J[0] * tau_z
      temp[0, 7, :, :] = np.eye(4)

      self._mpo.append(np.copy(temp))
      for n in range(1, self._N - 1):
        temp = np.zeros((8, 8, 4, 4), self.dtype)
        #11
        temp[0, 0, :, :] = np.eye(4)
        temp[1, 0, :, :] = sig_x
        temp[2, 0, :, :] = sig_y
        temp[3, 0, :, :] = sig_z
        temp[4, 0, :, :] = tau_x
        temp[5, 0, :, :] = tau_y
        temp[6, 0, :, :] = tau_z

        temp[7, 0, :, :] = self._w[n] * (
            sig_x + tau_x) + self._chi[n] * np.kron(sigma_x, sigma_x)

        temp[7, 1, :, :] = self._J[n] * sig_x
        temp[7, 2, :, :] = self._J[n] * sig_y
        temp[7, 3, :, :] = self._J[n] * sig_z
        temp[7, 4, :, :] = self._J[n] * tau_x
        temp[7, 5, :, :] = self._J[n] * tau_y
        temp[7, 6, :, :] = self._J[n] * tau_z
        temp[7, 7, :, :] = np.eye(4)

        self._mpo.append(np.copy(temp))

      temp = np.zeros((8, 1, 4, 4), self.dtype)
      temp[0, 0, :, :] = np.eye(4)
      temp[1, 0, :, :] = sig_x
      temp[2, 0, :, :] = sig_y
      temp[3, 0, :, :] = sig_z
      temp[4, 0, :, :] = tau_x
      temp[5, 0, :, :] = tau_y
      temp[6, 0, :, :] = tau_z
      temp[7, 0, :, :] = self._w[-1] * (
          sig_x + tau_x) + self._chi[-1] * np.kron(sigma_x, sigma_x)
      self._mpo.append(np.copy(temp))

    if obc == False:
      if len(J) < self._N:
        raise ValueError(
            "in JessHamiltonian: for obc=False, len(J) has to be N")
      self._mpo = []
      for n in range(self._N):
        temp = np.zeros((8, 8, 4, 4), self.dtype)
        temp[0, 0, :, :] = np.eye(4)
        temp[1, 0, :, :] = sig_x
        temp[2, 0, :, :] = sig_y
        temp[3, 0, :, :] = sig_z
        temp[4, 0, :, :] = tau_x
        temp[5, 0, :, :] = tau_y
        temp[6, 0, :, :] = tau_z

        temp[7, 0, :, :] = self._w[n] * (
            sig_x + tau_x) + self._chi[n] * np.kron(sigma_x, sigma_x)

        temp[7, 1, :, :] = self._J[n] * sig_x
        temp[7, 2, :, :] = self._J[n] * sig_y
        temp[7, 3, :, :] = self._J[n] * sig_z
        temp[7, 4, :, :] = self._J[n] * tau_x
        temp[7, 5, :, :] = self._J[n] * tau_y
        temp[7, 6, :, :] = self._J[n] * tau_z
        temp[7, 7, :, :] = np.eye(4)
        self._mpo.append(np.copy(temp))


class XXZ(MPO):
  """
    the famous Heisenberg Hamiltonian, which we all know and love so much!
    """

  def __init__(self, Jz, Jxy, Bz, obc=True, dtype=float):
    self._obc = obc
    self._Jz = Jz
    self._Jxy = Jxy
    self._dtype = dtype
    self._Bz = Bz
    self._N = len(Bz)
    if obc == True:
      self._mpo = []
      temp = np.zeros((1, 5, 2, 2), self.dtype)
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
      self._mpo.append(np.copy(temp))
      for n in range(1, self._N - 1):
        temp = np.zeros((5, 5, 2, 2), self.dtype)
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

        self._mpo.append(np.copy(temp))

      temp = np.zeros((5, 1, 2, 2), self.dtype)
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

      self._mpo.append(np.copy(temp))
      #return mpo
    if obc == False:
      assert (len(Jz) == len(Jxy))
      assert (len(Bz) == len(Jz))
      self._mpo = []
      for n in range(0, self._N):

        temp = np.zeros((5, 5, 2, 2), self.dtype)
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

        #temp=np.zeros((5,5,2,2),self.dtype)
        ##11
        #temp[0,0,0,0]=1.0
        #temp[0,0,1,1]=1.0
        ##Sp
        #temp[1,0,1,0]=1.0
        ##Sm
        #temp[2,0,0,1]=1.0
        ##Sz
        #temp[3,0,0,0]=-0.5
        #temp[3,0,1,1]=0.5
        ##BSz
        #temp[4,0,0,0]=-0.5*Bz[n]
        #temp[4,0,1,1]= 0.5*Bz[n]
        #
        ##Sm
        #temp[4,1,0,1]=Jxy[n]/2.0*1.0
        ##Sp
        #temp[4,2,1,0]=Jxy[n]/2.0*1.0
        ##Sz
        #temp[4,3,0,0]=Jz[n]*(-0.5)
        #temp[4,3,1,1]=Jz[n]*0.5
        ##11
        #temp[4,4,0,0]=1.0
        #temp[4,4,1,1]=1.0
        self._mpo.append(np.copy(temp))


class XXZflipped(MPO):
  """
    again the famous Heisenberg Hamiltonian, but in a slightly less common form
    """

  def __init__(self, Delta, J, obc=True, dtype=float):
    self._Delta = Delta
    self._J = J
    self._dtype = dtype
    self._obc = obc
    self._mpo = []
    if obc == True:
      self._N = len(Delta) + 1
      temp = np.zeros((1, 5, 2, 2)).astype(self.dtype)
      #Sx
      temp[0, 1, 0, 1] = J[0]
      temp[0, 1, 1, 0] = J[0]
      #Sy
      temp[0, 2, 0, 1] = +1j * (-J[0])
      temp[0, 2, 1, 0] = -1j * (-J[0])

      #Sz
      temp[0, 3, 0, 0] = -Delta[0] * J[0] * (-1.0)
      temp[0, 3, 1, 1] = -Delta[0] * J[0] * 1.0
      #11
      temp[0, 4, 0, 0] = 1.0
      temp[0, 4, 1, 1] = 1.0
      self._mpo.append(np.copy(temp))
      for site in range(1, self._N - 1):
        temp = np.zeros((5, 5, 2, 2)).astype(self.dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #Sx
        temp[1, 0, 0, 1] = 1.0
        temp[1, 0, 1, 0] = 1.0
        #Sy
        temp[2, 0, 0, 1] = +1j
        temp[2, 0, 1, 0] = -1j

        #Sz
        temp[3, 0, 0, 0] = -1.0
        temp[3, 0, 1, 1] = +1.0

        #Sx
        temp[4, 1, 0, 1] = J[0]
        temp[4, 1, 1, 0] = J[0]
        #Sy
        temp[4, 2, 0, 1] = +1j * (-J[0])
        temp[4, 2, 1, 0] = -1j * (-J[0])

        #Sz
        temp[4, 3, 0, 0] = -Delta[site] * J[site] * (-1.0)
        temp[4, 3, 1, 1] = -Delta[site] * J[site] * 1.0
        #11
        temp[4, 4, 0, 0] = 1.0
        temp[4, 4, 1, 1] = 1.0
        self._mpo.append(np.copy(temp))

      temp = np.zeros((5, 1, 2, 2)).astype(self.dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sx
      temp[1, 0, 0, 1] = 1.0
      temp[1, 0, 1, 0] = 1.0
      #Sy
      temp[2, 0, 0, 1] = +1j
      temp[2, 0, 1, 0] = -1j
      #Sz
      temp[3, 0, 0, 0] = -1.0
      temp[3, 0, 1, 1] = +1.0

      self._mpo.append(np.copy(temp))

    if obc == False:
      self._N = len(Delta)
      for site in range(self._N):
        temp = np.zeros((5, 5, 2, 2)).astype(self.dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #Sx
        temp[1, 0, 0, 1] = 1.0
        temp[1, 0, 1, 0] = 1.0
        #Sy
        temp[2, 0, 0, 1] = +1j
        temp[2, 0, 1, 0] = -1j

        #Sz
        temp[3, 0, 0, 0] = -1.0
        temp[3, 0, 1, 1] = +1.0

        #Sx
        temp[4, 1, 0, 1] = J[0]
        temp[4, 1, 1, 0] = J[0]
        #Sy
        temp[4, 2, 0, 1] = +1j * (-J[0])
        temp[4, 2, 1, 0] = -1j * (-J[0])

        #Sz
        temp[4, 3, 0, 0] = -Delta[site] * J[site] * (-1.0)
        temp[4, 3, 1, 1] = -Delta[site] * J[site] * 1.0
        #11
        temp[4, 4, 0, 0] = 1.0
        temp[4, 4, 1, 1] = 1.0
        self._mpo.append(np.copy(temp))


class SpinlessFermions(MPO):
  """
    Spinless Fermions, you know them ...
    """

  def __init__(self, interaction, hopping, chempot, obc, dtype=complex):
    assert (len(interaction) == len(hopping))
    assert (len(interaction) == len(chempot) - 1)

    self._mpo = []
    self._interaction = interaction
    self._hoping = hopping
    self._chempot = chempot
    self._obc = obc
    self._dtype = dtype

    self._N = len(chempot)
    c = np.zeros((2, 2)).astype(dtype)
    c[0, 1] = 1.0
    P = np.diag([1.0, -1.0]).astype(dtype)

    if obc == True:
      tensor = np.zeros((1, 5, 2, 2)).astype(dtype)
      tensor[0, 0, :, :] = chempot[0] * herm(c).dot(c)
      tensor[0, 1, :, :] = (-1.0) * hopping[0] * herm(c).dot(P)
      tensor[0, 2, :, :] = (+1.0) * hopping[0] * c.dot(P)
      tensor[0, 3, :, :] = interaction[0] * herm(c).dot(c)
      tensor[0, 4, :, :] = np.eye(2)
      self._mpo.append(np.copy(tensor))

      for n in range(1, self._N - 1):
        tensor = np.zeros((5, 5, 2, 2)).astype(dtype)
        tensor[0, 0, :, :] = np.eye(2)
        tensor[1, 0, :, :] = c
        tensor[2, 0, :, :] = herm(c)
        tensor[3, 0, :, :] = herm(c).dot(c)
        tensor[4, 0, :, :] = chempot[n] * herm(c).dot(c)

        tensor[4, 1, :, :] = (-1.0) * hopping[n] * herm(c).dot(P)
        tensor[4, 2, :, :] = (+1.0) * hopping[n] * c.dot(P)
        tensor[4, 3, :, :] = interaction[n] * herm(c).dot(c)
        tensor[4, 4, :, :] = np.eye(2)

        self._mpo.append(np.copy(tensor))
      tensor = np.zeros((5, 1, 2, 2)).astype(dtype)
      tensor[0, 0, :, :] = np.eye(2)
      tensor[1, 0, :, :] = c
      tensor[2, 0, :, :] = herm(c)
      tensor[3, 0, :, :] = herm(c).dot(c)
      tensor[4, 0, :, :] = chempot[self._N - 1] * herm(c).dot(c)
      self._mpo.append(np.copy(tensor))

    if obc == False:
      for n in range(self._N):
        tensor = np.zeros((5, 5, 2, 2)).astype(dtype)
        tensor[0, 0, :, :] = np.eye(2)
        tensor[1, 0, :, :] = c.dot(P)
        tensor[2, 0, :, :] = herm(c).dot(P)
        tensor[3, 0, :, :] = herm(c).dot(c)
        tensor[4, 0, :, :] = chempot[n] * herm(c).dot(c)

        tensor[4, 1, :, :] = (-1.0) * hopping[n] * herm(c)
        tensor[4, 2, :, :] = (+1.0) * hopping[n] * c
        tensor[4, 3, :, :] = interaction[n] * herm(c).dot(c)
        tensor[4, 4, :, :] = np.eye(2)
        self._mpo.append(np.copy(tensor))


def PhiFourmpo(mu, nu, g, N, dx, cutoff, obc=False):
  """
    A discrete version of the phi^4 model!
    """

  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  cdag4 = cdag.dot(cdag).dot(cdag).dot(cdag)
  cdag3c = cdag.dot(cdag).dot(cdag).dot(c)
  cdag2c2 = cdag.dot(cdag).dot(c).dot(c)
  cdagc3 = cdag.dot(c).dot(c).dot(c)
  c4 = c.dot(c).dot(c).dot(c)
  #phi=cdag+c
  for n in range(0, N):
    temp = np.zeros((10, 10, cutoff, cutoff)).astype(complex)
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)

    temp[3, 0, :, :] = np.copy(num)
    temp[4, 0, :, :] = ((mu**2 + nu**2) / (8 * nu) + 1.0 /
                        (2 * nu * dx**2)) * np.eye(cutoff)

    temp[5, 0, :, :] = np.copy(c.dot(c))
    temp[6, 0, :, :] = ((mu**2 - nu**2) / (16 * nu)) * np.eye(cutoff)

    temp[7, 0, :, :] = np.copy(cdag.dot(cdag))
    temp[8, 0, :, :] = ((mu**2 - nu**2) / (16 * nu)) * np.eye(cutoff)

    temp[9, 0, :, :] = g / (96.0 * nu**2 * dx) * (cdag4 + 4 * cdag3c +
                                                  6 * cdag2c2 + 4 * cdagc3 + c4)
    #temp[9,0,:,:]=g/(96.0*nu**2*dx)*phi.dot(phi).dot(phi).dot(phi)

    temp[9, 1, :, :] = -1.0 / (2 * nu * dx**2) * c
    temp[9, 2, :, :] = -1.0 / (2 * nu * dx**2) * cdag

    temp[9, 3, :, :] = ((mu**2 + nu**2) / (8 * nu) + 1.0 /
                        (2 * nu * dx**2)) * np.eye(cutoff)
    temp[9, 4, :, :] = np.copy(num)

    temp[9, 5, :, :] = (mu**2 - nu**2) / (16 * nu) * np.eye(cutoff)
    temp[9, 6, :, :] = np.copy(c.dot(c))

    temp[9, 7, :, :] = (mu**2 - nu**2) / (16 * nu) * np.eye(cutoff)
    temp[9, 8, :, :] = np.copy(cdag.dot(cdag))

    temp[9, 9, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))

  if obc == True:
    [B1, B2, d1, d2] = np.shape(mpo[0])
    mpol = np.zeros((1, B2, d1, d2), dtype=complex)
    mpor = np.zeros((B1, 1, d1, d2), dtype=complex)
    mpol[0, :, :, :] = mpo[0][-1, :, :, :]
    mpor[:, 0, :, :] = mpo[0][:, 0, :, :]
    mpo[0] = np.copy(mpol)
    mpo[-1] = np.copy(mpor)

  return mpo


def PhiFourmpo2(mu, nu, g, N, dx, cutoff, obc=False):
  """
    Another discrete version of the phi^4 model!
    """

  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  cdag4 = cdag.dot(cdag).dot(cdag).dot(cdag)
  cdag3c = cdag.dot(cdag).dot(cdag).dot(c)
  cdag2c2 = cdag.dot(cdag).dot(c).dot(c)
  cdagc3 = cdag.dot(c).dot(c).dot(c)
  c4 = c.dot(c).dot(c).dot(c)
  #phi=cdag+c
  for n in range(0, N):
    temp = np.zeros((10, 10, cutoff, cutoff)).astype(complex)
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)

    temp[3, 0, :, :] = np.copy(num)
    temp[4, 0, :, :] = ((mu**2 + nu**2) / 4.0 + 1.0 / (dx**2)) * np.eye(cutoff)

    temp[5, 0, :, :] = np.copy(c.dot(c))
    temp[6, 0, :, :] = ((mu**2 - nu**2) / (8.0)) * np.eye(cutoff)

    temp[7, 0, :, :] = np.copy(cdag.dot(cdag))
    temp[8, 0, :, :] = ((mu**2 - nu**2) / (8.0)) * np.eye(cutoff)

    temp[9, 0, :, :] = g / (96.0 * nu * dx) * (cdag4 + 4 * cdag3c +
                                               6 * cdag2c2 + 4 * cdagc3 + c4)
    #temp[9,0,:,:]=g/(48.0*nu)*cdag2c2
    #temp[9,0,:,:]=g/(48.0*nu*dx)*phi.dot(phi).dot(phi).dot(phi)

    temp[9, 1, :, :] = -1.0 / (1.0 * dx**2) * c
    temp[9, 2, :, :] = -1.0 / (1.0 * dx**2) * cdag

    temp[9, 3, :, :] = ((mu**2 + nu**2) / 4.0 + 1.0 / (dx**2)) * np.eye(cutoff)
    temp[9, 4, :, :] = np.copy(num)

    temp[9, 5, :, :] = (mu**2 - nu**2) / (8.0) * np.eye(cutoff)
    temp[9, 6, :, :] = np.copy(c.dot(c))

    temp[9, 7, :, :] = (mu**2 - nu**2) / (8.0) * np.eye(cutoff)
    temp[9, 8, :, :] = np.copy(cdag.dot(cdag))

    temp[9, 9, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))

  if obc == True:
    [B1, B2, d1, d2] = np.shape(mpo[0])
    mpol = np.zeros((1, B2, d1, d2), dtype=complex)
    mpor = np.zeros((B1, 1, d1, d2), dtype=complex)
    mpol[0, :, :, :] = mpo[0][-1, :, :, :]
    mpor[:, 0, :, :] = mpo[0][:, 0, :, :]
    mpo[0] = np.copy(mpol)
    mpo[-1] = np.copy(mpor)

  return mpo


def PhiFourmpo3(mu, nu, g, N, dx, cutoff, obc=False):
  """
    Yet another discrete version of the phi^4 model!
    """

  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  cdag4 = cdag.dot(cdag).dot(cdag).dot(cdag)
  cdag3c = cdag.dot(cdag).dot(cdag).dot(c)
  cdag2c2 = cdag.dot(cdag).dot(c).dot(c)
  cdagc3 = cdag.dot(c).dot(c).dot(c)
  c4 = c.dot(c).dot(c).dot(c)
  #phi=cdag+c
  for n in range(0, N):
    temp = np.zeros((6, 6, cutoff, cutoff)).astype(complex)
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = np.eye(cutoff)
    temp[4, 0, :, :] = 1.0 / (dx**2) * np.copy(num)
    temp[5, 0, :, :] = (mu**2 + nu**2) / 2.0 * num + (mu**2 - nu**2) / 4.0 * (
        c.dot(c) + cdag.dot(cdag)) + g / (96.0 * nu * dx) * (
            cdag4 + 4 * cdag3c + 6 * cdag2c2 + 4 * cdagc3 + c4)
    temp[5, 1, :, :] = -1.0 / (dx**2) * c
    temp[5, 2, :, :] = -1.0 / (dx**2) * cdag
    temp[5, 3, :, :] = 1.0 / (dx**2) * np.copy(num)
    temp[5, 4, :, :] = np.eye(cutoff)
    temp[5, 5, :, :] = np.eye(cutoff)
    mpo.append(np.copy(temp))
  if obc == True:
    [B1, B2, d1, d2] = np.shape(mpo[0])
    mpol = np.zeros((1, B2, d1, d2), dtype=complex)
    mpor = np.zeros((B1, 1, d1, d2), dtype=complex)
    mpol[0, :, :, :] = mpo[0][-1, :, :, :]
    mpor[:, 0, :, :] = mpo[0][:, 0, :, :]
    mpo[0] = np.copy(mpol)
    mpo[-1] = np.copy(mpor)

  return mpo


def PhiFourmpo4(mu, nu, g, N, dx, cutoff, obc=False):
  """
    How many more Phi^4 models are there?
    """

  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  cdag4 = cdag.dot(cdag).dot(cdag).dot(cdag)
  cdag3c = cdag.dot(cdag).dot(cdag).dot(c)
  cdag2c2 = cdag.dot(cdag).dot(c).dot(c)
  cdagc3 = cdag.dot(c).dot(c).dot(c)
  c4 = c.dot(c).dot(c).dot(c)
  #phi=cdag+c
  for n in range(0, N):
    temp = np.zeros((4, 4, cutoff, cutoff)).astype(complex)
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = (
        (mu**2.0 + nu**2.0) / 2.0 + 2.0 /
        (dx**2.0)) * num + (mu**2.0 - nu**2.0) / 4.0 * (
            c.dot(c) + cdag.dot(cdag)) + g / (96.0 * nu * dx) * (
                cdag4 + 4 * cdag3c + 6.0 * cdag2c2 + 4 * cdagc3 + c4)
    #old version temp[3,0,:,:]=((mu**2.0+nu**2.0)/1.0+2.0/(dx**2.0))*num+(mu**2.0-nu**2.0)/2.0*(c.dot(c)+cdag.dot(cdag))+g/(48.0*nu*dx)*(cdag4+4*cdag3c+6.0*cdag2c2+4*cdagc3+c4)
    temp[3, 1, :, :] = -1.0 / (dx**2) * c
    temp[3, 2, :, :] = -1.0 / (dx**2) * cdag
    temp[3, 3, :, :] = np.eye(cutoff)
    mpo.append(np.copy(temp))
  if obc == True:
    [B1, B2, d1, d2] = np.shape(mpo[0])
    mpol = np.zeros((1, B2, d1, d2), dtype=complex)
    mpor = np.zeros((B1, 1, d1, d2), dtype=complex)
    mpol[0, :, :, :] = mpo[0][-1, :, :, :]
    mpor[:, 0, :, :] = mpo[0][:, 0, :, :]
    mpo[0] = np.copy(mpol)
    mpo[-1] = np.copy(mpor)

  return mpo


def TwoSitePhiFourMPO(mu, nu, g, dx, cutoff):
  """
    There is an proliferatoin of Phi Four models!
    """

  mpo = []

  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  cdag4 = cdag.dot(cdag).dot(cdag).dot(cdag)
  cdag3c = cdag.dot(cdag).dot(cdag).dot(c)
  cdag2c2 = cdag.dot(cdag).dot(c).dot(c)
  cdagc3 = cdag.dot(c).dot(c).dot(c)
  c4 = c.dot(c).dot(c).dot(c)
  temp = np.zeros((1, 4, cutoff, cutoff)).astype(complex)
  temp[0, 0, :, :] = np.copy(
      ((mu**2 + nu**2) / 2.0 + 2.0 / (dx**2)) * num + (mu**2 - nu**2) / 4.0 *
      (c.dot(c) + cdag.dot(cdag)) + g / (96.0 * nu * dx) *
      (cdag4 + 4 * cdag3c + 6 * cdag2c2 + 4 * cdagc3 + c4)) / 2.0
  temp[0, 1, :, :] = np.copy(-1.0 / (dx**2) * c)
  temp[0, 2, :, :] = np.copy(-1.0 / (dx**2) * cdag)
  temp[0, 3, :, :] = np.copy(np.eye(cutoff))
  mpo.append(np.copy(temp))

  temp = np.zeros((4, 1, cutoff, cutoff)).astype(complex)
  temp[0, 0, :, :] = np.eye(cutoff)
  temp[1, 0, :, :] = np.copy(cdag)
  temp[2, 0, :, :] = np.copy(c)
  temp[3, 0, :, :] = np.copy(
      ((mu**2 + nu**2) / 2.0 + 2.0 / (dx**2)) * num + (mu**2 - nu**2) / 4.0 *
      (c.dot(c) + cdag.dot(cdag)) + g / (96.0 * nu * dx) *
      (cdag4 + 4 * cdag3c + 6 * cdag2c2 + 4 * cdagc3 + c4)) / 2.0
  mpo.append(np.copy(temp))

  return mpo


class HubbardChain(MPO):
  """
    The good old Fermi Hubbard model
    """

  def __init__(self, U, t_up, t_down, mu_up, mu_down, obc, dtype=complex):
    self._U = U
    self._t_up = t_up
    self._t_down = t_down
    self._mu_up = mu_up
    self._mu_down = mu_down
    self._obc = obc
    self._dtype = dtype

    self._N = len(U)

    assert (len(mu_up) == self._N)
    assert (len(mu_down) == self._N)
    assert (len(t_up) == self._N - 1)
    assert (len(t_down) == self._N - 1)

    self._mpo = []
    c = np.zeros((2, 2), dtype=self.dtype)
    c[0, 1] = 1.0
    c_down = np.kron(c, np.eye(2))
    c_up = np.kron(np.diag([1.0, -1.0]).astype(self.dtype), c)
    P = np.diag([1.0, -1.0, -1.0, 1.0]).astype(self.dtype)

    if obc == True:
      tensor = np.zeros((1, 6, 4, 4)).astype(self.dtype)
      tensor[0, 0, :, :] = mu_up[0] * herm(c_up).dot(c_up) + mu_down[0] * herm(
          c_down).dot(c_down) + U[0] * herm(c_up).dot(c_up).dot(
              herm(c_down).dot(c_down))
      tensor[0, 1, :, :] = (-1.0) * t_up[0] * herm(c_up).dot(P)
      tensor[0, 2, :, :] = (+1.0) * t_up[0] * c_up.dot(P)
      tensor[0, 3, :, :] = (-1.0) * t_down[0] * herm(c_down).dot(P)
      tensor[0, 4, :, :] = (+1.0) * t_down[0] * c_down.dot(P)
      tensor[0, 5, :, :] = np.eye(4)
      self._mpo.append(np.copy(tensor))

      for n in range(1, self._N - 1):
        tensor = np.zeros((6, 6, 4, 4)).astype(self.dtype)
        tensor[0, 0, :, :] = np.eye(4)
        tensor[1, 0, :, :] = c_up
        tensor[2, 0, :, :] = herm(c_up)
        tensor[3, 0, :, :] = c_down
        tensor[4, 0, :, :] = herm(c_down)

        tensor[5, 0, :, :] = mu_up[n] * herm(c_up).dot(
            c_up) + mu_down[n] * herm(c_down).dot(c_down) + U[n] * herm(
                c_up).dot(c_up).dot(herm(c_down).dot(c_down))

        tensor[5, 1, :, :] = (-1.0) * t_up[n] * herm(c_up).dot(P)
        tensor[5, 2, :, :] = (+1.0) * t_up[n] * c_up.dot(P)
        tensor[5, 3, :, :] = (-1.0) * t_down[n] * herm(c_down).dot(P)
        tensor[5, 4, :, :] = (+1.0) * t_down[n] * c_down.dot(P)
        tensor[5, 5, :, :] = np.eye(4)

        self._mpo.append(np.copy(tensor))

      tensor = np.zeros((6, 1, 4, 4)).astype(self.dtype)
      tensor[0, 0, :, :] = np.eye(4)
      tensor[1, 0, :, :] = c_up
      tensor[2, 0, :, :] = herm(c_up)
      tensor[3, 0, :, :] = c_down
      tensor[4, 0, :, :] = herm(c_down)
      tensor[5, 0, :, :] = mu_up[self._N - 1] * herm(c_up).dot(c_up) + mu_down[
          self._N - 1] * herm(c_down).dot(c_down) + U[self._N - 1] * herm(
              c_up).dot(c_up).dot(herm(c_down).dot(c_down))
      self._mpo.append(np.copy(tensor))

    if obc == False:
      for n in range(self._N):
        tensor = np.zeros((6, 6, 4, 4)).astype(self.dtype)
        tensor[0, 0, :, :] = np.eye(4)
        tensor[1, 0, :, :] = c_up
        tensor[2, 0, :, :] = herm(c_up)
        tensor[3, 0, :, :] = c_down
        tensor[4, 0, :, :] = herm(c_down)

        tensor[5, 0, :, :] = mu_up[n] * herm(c_up).dot(
            c_up) + mu_down[n] * herm(c_down).dot(c_down) + U[n] * herm(
                c_up).dot(c_up).dot(herm(c_down).dot(c_down))

        tensor[5, 1, :, :] = (-1.0) * t_up[n] * herm(c_up).dot(P)
        tensor[5, 2, :, :] = (+1.0) * t_up[n] * c_up.dot(P)
        tensor[5, 3, :, :] = (-1.0) * t_down[n] * herm(c_down).dot(P)
        tensor[5, 4, :, :] = (+1.0) * t_down[n] * c_down.dot(P)
        tensor[5, 5, :, :] = np.eye(4)

        self._mpo.append(np.copy(tensor))


def projectedTwoBosonsModel(mu1, mu2, g1, g2, m1, m2, N, dx, obc, dtype=float):
  N = len(mu)
  c = np.zeros((3, 3), dtype=dtype)
  for n in range(1, 2):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)

  if obc == True:
    #dxtilde=dx[0]
    if n < (N - 1):
      dxtilde = (dx[n + 1] + dx[n]) / 2.
    if n == (N - 1):
      dxtilde = (dx[0] + dx[n]) / 2.

    mpo = []
    temp = np.zeros((1, 9, 2, 2), dtype=dtype)

    temp[0, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * num / dx[0]
    temp[0, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
    temp[0, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * c / np.sqrt(dx[0])
    temp[0, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * cdag / np.sqrt(dx[0])
    temp[0, 5, :, :] = dxtilde * interact * num / dx[0]
    temp[0, 6, :, :] = np.eye(2)
    temp[0, 7, :, :] = mu[0] / 2.0 * dxtilde * num / dx[n]
    temp[0, 8, :, :] = np.eye(2)
    mpo.append(np.copy(temp))

    for n in range(1, N - 1):
      dxtilde = dx[n]
      temp = np.zeros((9, 9, 2, 2), dtype=dtype)
      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = np.eye(2) - num
      temp[2, 0, :, :] = num / dx[n]
      temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
      temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
      temp[5, 0, :, :] = np.copy(num) / dx[n]
      temp[6, 0, :, :] = mu[n] / 2.0 * num
      temp[7, 0, :, :] = np.eye(2)

      temp[8, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * num / dx[n]
      temp[8, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
      temp[8, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * c / np.sqrt(dx[n])
      temp[8, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * cdag / np.sqrt(dx[n])
      temp[8, 5, :, :] = interact * dxtilde * num / dx[n]
      temp[8, 6, :, :] = np.eye(2)
      temp[8, 7, :, :] = mu[n] / 2.0 * dxtilde * num / dx[n]
      temp[8, 8, :, :] = np.eye(2)
      mpo.append(np.copy(temp))

    dxtilde = dx[N - 1]
    temp = np.zeros((9, 1, 2, 2), dtype=dtype)
    temp[0, 0, :, :] = np.eye(2)
    temp[1, 0, :, :] = np.eye(2) - num
    temp[2, 0, :, :] = num / dx[n]
    temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
    temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
    temp[5, 0, :, :] = np.copy(num) / dx[n]
    temp[6, 0, :, :] = mu[N - 1] / 2.0 * dxtilde * num / dx[n]
    temp[7, 0, :, :] = np.eye(2)
    mpo.append(np.copy(temp))

    return mpo

  if obc == False:
    mpo = []
    for n in range(0, N):
      #dxtilde=dx[n]
      if n < (N - 1):
        dxtilde = (dx[n + 1] + dx[n]) / 2.
      if n == (N - 1):
        dxtilde = (dx[0] + dx[n]) / 2.

      temp = np.zeros((9, 9, 2, 2), dtype=dtype)
      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = (np.eye(2) - num)
      temp[2, 0, :, :] = num / dx[n]
      temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
      temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
      temp[5, 0, :, :] = num / dx[n]
      temp[6, 0, :, :] = mu[n] / 2.0 * num
      temp[7, 0, :, :] = np.eye(2)

      temp[8, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * (num / dx[n])
      temp[8, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
      temp[8, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * (c / np.sqrt(dx[n]))
      temp[8, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * (cdag / np.sqrt(dx[n]))
      temp[8, 5, :, :] = interact * dxtilde * (num / dx[n])
      temp[8, 6, :, :] = np.eye(2)
      temp[8, 7, :, :] = mu[n] / 2.0 * num
      temp[8, 8, :, :] = np.eye(2)

      mpo.append(np.copy(temp))

    return mpo


def projectedLiebLinigermpo(mu, interact, mass, N, dx, obc):
  N = len(mu)
  c = np.zeros((2, 2))
  for n in range(1, 2):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)

  if obc == True:
    mpo = []
    temp = np.zeros((1, 7, 2, 2))
    temp[0, 0, :, :] = mu[0] / 2.0 * num
    temp[0, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
    temp[0, 2, :, :] = np.eye(2) - num
    #temp[0,2,:,:]=1.0/(2.0*mass*(dx**2.))*(np.eye(2))
    temp[0, 3, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * c
    temp[0, 4, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * cdag
    temp[0, 5, :, :] = np.copy((interact * 1.0 / dx) * num)
    temp[0, 6, :, :] = np.eye(2)
    mpo.append(np.copy(temp))

    for n in range(1, N - 1):
      temp = np.zeros((7, 7, 2, 2))

      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = np.eye(2) - num
      temp[2, 0, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
      temp[3, 0, :, :] = np.copy(cdag)
      temp[4, 0, :, :] = np.copy(c)
      temp[5, 0, :, :] = np.copy(num)

      temp[6, 0, :, :] = mu[n] * num
      temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
      temp[6, 2, :, :] = (np.eye(2) - num)
      temp[6, 3, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * c
      temp[6, 4, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * cdag
      temp[6, 5, :, :] = np.copy((interact * 1.0 / dx) * num)
      temp[6, 6, :, :] = np.eye(2)
      mpo.append(np.copy(temp))

    temp = np.zeros((7, 1, 2, 2))
    temp[0, 0, :, :] = np.eye(2)
    temp[1, 0, :, :] = np.eye(2) - num
    #temp[1,0,:,:]=np.eye(2)
    temp[2, 0, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
    temp[3, 0, :, :] = np.copy(cdag)
    temp[4, 0, :, :] = np.copy(c)
    temp[5, 0, :, :] = np.copy(num)
    temp[6, 0, :, :] = mu[N - 1] / 2.0 * num

    mpo.append(np.copy(temp))

    return mpo

  if obc == False:
    mpo = []
    for n in range(0, N):
      temp = np.zeros((7, 7, 2, 2))

      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = np.eye(2) - num
      temp[2, 0, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
      temp[3, 0, :, :] = np.copy(cdag)
      temp[4, 0, :, :] = np.copy(c)
      temp[5, 0, :, :] = np.copy(num)

      temp[6, 0, :, :] = mu[n] * num
      temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
      temp[6, 2, :, :] = (np.eye(2) - num)
      temp[6, 3, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * c
      temp[6, 4, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * cdag
      temp[6, 5, :, :] = np.copy((interact * 1.0 / dx) * num)
      temp[6, 6, :, :] = np.eye(2)
      mpo.append(np.copy(temp))
    return mpo


def projectedLiebLinigermpo2(mu, interact, mass, N, d, dx, obc):

  N = len(mu)
  c = np.zeros((d, d))
  proj = np.eye(d)
  for n in range(1, d):
    c[n - 1, n] = np.sqrt(n)
    proj[n, n] = 0.0
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  inpart = cdag.dot(cdag).dot(c).dot(c)

  if obc == True:
    mpo = []
    temp = np.zeros((1, 7, d, d))
    temp[0, 0, :, :] = mu[0] * num + interact / dx * inpart
    temp[0, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
    temp[0, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * proj
    #temp[0,2,:,:]=1.0/(2.0*mass*(dx**2.))*(np.eye(2))
    temp[0, 3, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * c
    temp[0, 4, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * cdag
    temp[0, 5, :, :] = np.copy((interact * 1.0 / dx) * num) * 0.0
    temp[0, 6, :, :] = np.eye(d)
    mpo.append(np.copy(temp))

    for n in range(1, N - 1):
      temp = np.zeros((7, 7, d, d))

      temp[0, 0, :, :] = np.eye(d)
      temp[1, 0, :, :] = proj
      temp[2, 0, :, :] = num
      temp[3, 0, :, :] = np.copy(cdag)
      temp[4, 0, :, :] = np.copy(c)
      temp[5, 0, :, :] = np.copy(num)

      temp[6, 0, :, :] = mu[n] * num + interact / dx * inpart
      temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
      temp[6, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * proj
      temp[6, 3, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * c
      temp[6, 4, :, :] = -1.0 / (2.0 * mass * (dx**2.)) * cdag
      temp[6, 5, :, :] = np.copy((interact * 1.0 / dx) * num) * 0.0
      temp[6, 6, :, :] = np.eye(d)
      mpo.append(np.copy(temp))

    temp = np.zeros((7, 1, d, d))
    temp[0, 0, :, :] = np.eye(d)
    temp[1, 0, :, :] = proj
    temp[2, 0, :, :] = num
    temp[3, 0, :, :] = np.copy(cdag)
    temp[4, 0, :, :] = np.copy(c)
    temp[5, 0, :, :] = np.copy(num) * 0.0
    temp[6, 0, :, :] = mu[N - 1] * num + interact / dx * inpart

    mpo.append(np.copy(temp))

    return mpo

  return mpo


def projectedLiebLinigermpo3(mu,
                             interact,
                             mass,
                             dx,
                             obc,
                             dtype=float,
                             proj=True):
  N = len(mu)
  c = np.zeros((2, 2), dtype=dtype)
  for n in range(1, 2):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)

  if obc == True:
    if n < (N - 1):
      dxtilde = (dx[n + 1] + dx[n]) / 2.
    if n == (N - 1):
      dxtilde = (dx[0] + dx[n]) / 2.

    mpo = []
    temp = np.zeros((1, 9, 2, 2), dtype=dtype)

    temp[0, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * num / dx[0]
    temp[0, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
    temp[0, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * c / np.sqrt(dx[0])
    temp[0, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * cdag / np.sqrt(dx[0])
    temp[0, 5, :, :] = dxtilde * interact * num / dx[0]
    temp[0, 6, :, :] = np.eye(2)
    temp[0, 7, :, :] = mu[0] / 2.0 * dxtilde * num / dx[n]
    temp[0, 8, :, :] = np.eye(2)
    mpo.append(np.copy(temp))

    for n in range(1, N - 1):
      dxtilde = dx[n]
      temp = np.zeros((9, 9, 2, 2), dtype=dtype)
      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = np.eye(2) - num
      temp[2, 0, :, :] = num / dx[n]
      temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
      temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
      temp[5, 0, :, :] = np.copy(num) / dx[n]
      temp[6, 0, :, :] = mu[n] / 2.0 * num
      temp[7, 0, :, :] = np.eye(2)

      temp[8, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * num / dx[n]
      temp[8, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
      temp[8, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * c / np.sqrt(dx[n])
      temp[8, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * cdag / np.sqrt(dx[n])
      temp[8, 5, :, :] = interact * dxtilde * num / dx[n]
      temp[8, 6, :, :] = np.eye(2)
      temp[8, 7, :, :] = mu[n] / 2.0 * dxtilde * num / dx[n]
      temp[8, 8, :, :] = np.eye(2)
      mpo.append(np.copy(temp))

    dxtilde = dx[N - 1]
    temp = np.zeros((9, 1, 2, 2), dtype=dtype)
    temp[0, 0, :, :] = np.eye(2)
    temp[1, 0, :, :] = np.eye(2) - num
    temp[2, 0, :, :] = num / dx[n]
    temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
    temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
    temp[5, 0, :, :] = np.copy(num) / dx[n]
    temp[6, 0, :, :] = mu[N - 1] / 2.0 * dxtilde * num / dx[n]
    temp[7, 0, :, :] = np.eye(2)
    mpo.append(np.copy(temp))

    return mpo

  if obc == False:
    mpo = []
    for n in range(0, N):
      #dxtilde=dx[n]
      if n < (N - 1):
        dxtilde = (dx[n + 1] + dx[n]) / 2.
      if n == (N - 1):
        dxtilde = (dx[0] + dx[n]) / 2.

      temp = np.zeros((9, 9, 2, 2), dtype=dtype)
      temp[0, 0, :, :] = np.eye(2)
      temp[1, 0, :, :] = (np.eye(2) - num)
      temp[2, 0, :, :] = num / dx[n]
      temp[3, 0, :, :] = np.copy(cdag) / np.sqrt(dx[n])
      temp[4, 0, :, :] = np.copy(c) / np.sqrt(dx[n])
      temp[5, 0, :, :] = num / dx[n]
      temp[6, 0, :, :] = mu[n] / 2.0 * num
      temp[7, 0, :, :] = np.eye(2)

      temp[8, 1, :, :] = 1.0 / (2.0 * mass * dxtilde) * (num / dx[n])
      temp[8, 2, :, :] = 1.0 / (2.0 * mass * dxtilde) * (np.eye(2) - num)
      temp[8, 3, :, :] = -1.0 / (2.0 * mass * dxtilde) * (c / np.sqrt(dx[n]))
      temp[8, 4, :, :] = -1.0 / (2.0 * mass * dxtilde) * (cdag / np.sqrt(dx[n]))
      temp[8, 5, :, :] = interact * dxtilde * (num / dx[n])
      temp[8, 6, :, :] = np.eye(2)
      temp[8, 7, :, :] = mu[n] / 2.0 * num
      temp[8, 8, :, :] = np.eye(2)

      mpo.append(np.copy(temp))

    return mpo


def LocalLiebLinigermpo(mu, intac, mass, dx, cutoff, obc=False):
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  inpart = cdag.dot(cdag).dot(c).dot(c)
  temp = np.zeros((1, 4, cutoff, cutoff))

  temp[0, 0, :, :] = (mu[0] / 2.0 + 1.0 /
                      (2.0 * mass * (dx**2))) * num + (intac /
                                                       (2.0 * dx)) * inpart
  temp[0, 1, :, :] = -1.0 / (2.0 * mass * dx**2) * c
  temp[0, 2, :, :] = -1.0 / (2.0 * mass * dx**2) * cdag
  temp[0, 3, :, :] = np.eye(cutoff)

  mpo.append(temp)
  temp = np.zeros((4, 1, cutoff, cutoff))
  temp[0, 0, :, :] = np.eye(cutoff)
  temp[1, 0, :, :] = np.copy(cdag)
  temp[2, 0, :, :] = np.copy(c)
  temp[3, 0, :, :] = (mu[-1] / 2.0 + 1.0 /
                      (2.0 * mass * dx**2)) * num + (intac /
                                                     (dx * 2.0)) * inpart
  mpo.append(temp)
  return mpo


def LiebLinigerEdens(mu, intac, mass, dx, cutoff, N, site):
  assert (site > 1)
  assert (site < (N - 1))
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  inpart = cdag.dot(cdag).dot(c).dot(c)
  temp = np.zeros((1, 1, cutoff, cutoff))
  temp[0, 0, :, :] = np.eye(cutoff)
  mpo.append(temp)
  for n in range(1, site - 1):
    temp[0, 0, :, :] = np.eye(cutoff)
    mpo.append(temp)

  temp = np.zeros((1, 3, cutoff, cutoff))
  temp[0, 1, :, :] = np.copy(c)
  temp[0, 2, :, :] = np.eye(cutoff)
  mpo.append(temp)

  temp = np.zeros((3, 3, cutoff, cutoff))
  temp[0, 0, :, :] = np.eye(cutoff)
  temp[1, 0, :, :] = np.copy(cdag) * (-1.0 / (2.0 * mass * dx**2))
  #temp[2,0,:,:]=np.copy(num)*(1.0/(mass*dx**2)+mu[site])+intac/dx*inpart
  temp[2, 0, :, :] = np.copy(num) * (1.0 / (mass * dx**2)) + intac / dx * inpart
  temp[2, 1, :, :] = np.copy(cdag) * (-1.0 / (2.0 * mass * dx**2))
  temp[2, 2, :, :] = np.eye(cutoff)
  mpo.append(temp)

  temp = np.zeros((3, 1, cutoff, cutoff))
  temp[0, 0, :, :] = np.eye(cutoff)
  temp[1, 0, :, :] = np.copy(c)
  mpo.append(temp)

  temp = np.zeros((1, 1, cutoff, cutoff))
  for n in range(site + 1, N):
    temp[0, 0, :, :] = np.eye(cutoff)
    mpo.append(temp)

  return mpo


def LiebLinigermpo(mu, g, mass, N, dx, cutoff):
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)

  for n in range(0, N):
    temp = np.zeros((4, 4, cutoff, cutoff))
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = (
        mu + 2.0 /
        (2.0 * mass * dx**2)) * num + g / dx * cdag.dot(cdag).dot(c).dot(
            c)  #+g/dx*num.dot(num-np.eye(cutoff))
    temp[3, 1, :, :] = -1.0 / (2 * mass * dx**2) * c
    temp[3, 2, :, :] = -1.0 / (2 * mass * dx**2) * cdag
    temp[3, 3, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))
  return mpo


def LiebLinigermpo1(mu, g, mass, N, dx, cutoff):
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)

  for n in range(0, N):
    temp = np.zeros((6, 6, cutoff, cutoff))
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = np.copy(num)
    temp[4, 0, :, :] = (mu / 2.0 + 1.0 / (2 * mass * dx**2)) * np.eye(cutoff)

    temp[5, 0, :, :] = g / dx * cdag.dot(cdag).dot(c).dot(
        c)  #g/dx*num.dot(num-np.eye(cutoff))
    temp[5, 1, :, :] = -1.0 / (2 * mass * dx**2) * c
    temp[5, 2, :, :] = -1.0 / (2 * mass * dx**2) * cdag
    temp[5, 3, :, :] = (mu / 2.0 + 1.0 / (2 * mass * dx**2)) * np.eye(cutoff)
    temp[5, 4, :, :] = np.copy(num)
    temp[5, 5, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))
  return mpo


def LiebLinigermpo2(mu, g, mass, N, dx, cutoff):
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  for n in range(0, N):
    temp = np.zeros((7, 7, cutoff, cutoff))
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = np.copy(num)
    temp[4, 0, :, :] = (mu / 2.0 + 1.0 / (2 * mass * dx**2)) * np.eye(cutoff)
    temp[5, 0, :, :] = np.copy(num)

    temp[6, 1, :, :] = -1.0 / (2 * mass * dx**2) * c
    temp[6, 2, :, :] = -1.0 / (2 * mass * dx**2) * cdag
    temp[6, 3, :, :] = (mu / 2.0 + 1.0 / (2 * mass * dx**2)) * np.eye(cutoff)
    temp[6, 4, :, :] = np.copy(num)
    temp[6, 5, :, :] = g / dx * np.copy(num)
    temp[6, 6, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))
  return mpo


def LiebLinigermpo3(mu, g, mass, N, dx, cutoff):
  mpo = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  for n in range(0, N):
    temp = np.zeros((7, 7, cutoff, cutoff))
    temp[0, 0, :, :] = np.eye(cutoff)
    temp[1, 0, :, :] = np.copy(cdag)
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = np.copy(num)
    temp[4, 0, :, :] = (1.0 / (2 * mass * dx**2)) * np.eye(cutoff)
    temp[5, 0, :, :] = np.copy(num)
    temp[6, 0, :, :] = np.copy(num) * mu
    temp[6, 1, :, :] = -1.0 / (2 * mass * dx**2) * c
    temp[6, 2, :, :] = -1.0 / (2 * mass * dx**2) * cdag
    temp[6, 3, :, :] = (1.0 / (2 * mass * dx**2)) * np.eye(cutoff)
    temp[6, 4, :, :] = np.copy(num)
    temp[6, 5, :, :] = g / dx * np.copy(num)
    temp[6, 6, :, :] = np.eye(cutoff)

    mpo.append(np.copy(temp))
  return mpo


def HomogeneousLiebLinigermpo(interact, mass, dtype=float):
  c = np.zeros((2, 2), dtype=dtype)
  for n in range(1, 2):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  P = (np.eye(2) - num)

  mpo = []
  temp = np.zeros((1, 7, 2, 2), dtype=dtype)

  temp[0, 1, :, :] = -1.0 / (2.0 * mass) * c
  temp[0, 2, :, :] = -1.0 / (2.0 * mass) * cdag
  temp[0, 3, :, :] = 1.0 / (2.0 * mass) * P
  temp[0, 4, :, :] = 1.0 / (2.0 * mass) * num
  temp[0, 5, :, :] = interact * num
  temp[0, 6, :, :] = np.eye(2)

  mpo.append(np.copy(temp))

  #the center MPO
  temp = np.zeros((7, 7, 2, 2), dtype=dtype)
  temp[0, 0, :, :] = np.eye(2)
  temp[1, 0, :, :] = cdag
  temp[2, 0, :, :] = c
  temp[3, 0, :, :] = num
  temp[4, 0, :, :] = P
  temp[5, 0, :, :] = num

  temp[6, 1, :, :] = -1.0 / (2.0 * mass) * c
  temp[6, 2, :, :] = -1.0 / (2.0 * mass) * cdag
  temp[6, 3, :, :] = 1.0 / (2.0 * mass) * P
  temp[6, 4, :, :] = 1.0 / (2.0 * mass) * num
  temp[6, 5, :, :] = interact * num
  temp[6, 6, :, :] = np.eye(2)
  mpo.append(np.copy(temp))

  temp = np.zeros((7, 1, 2, 2), dtype=dtype)
  temp[0, 0, :, :] = np.eye(2)
  temp[1, 0, :, :] = cdag
  temp[2, 0, :, :] = c
  temp[3, 0, :, :] = num
  temp[4, 0, :, :] = P
  temp[5, 0, :, :] = num

  mpo.append(np.copy(temp))

  return mpo


#this gives back two mpo's, one for each unitcell. This mpo is specific to the case
#of excitation spectrums for periodic LL model. The left mpol covers a left unitcell,
#the right mpor covers the right unitcell. For calculation of the excitation spectrum
#a SINGLE mps matrix is varied somewhere inside the unitcell at 'site'=site. In the excitation
#engine, a new unitcell is chosen such that in expressions like e.g. HABBA, the B matrices are
#at the boundaries of this unitcell; in fact the right B matrix has to be OUTSIDE the unitcell
def UnitcellExLiebLinigermpo(mu, intac, mass, N, dx, cutoff):
  mpol = []
  c = np.zeros((cutoff, cutoff))
  for n in range(1, cutoff):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  inpart = cdag.dot(cdag).dot(c).dot(c)
  temp = np.zeros((1, 4, cutoff, cutoff))
  #t=np.sqrt(1.0/(4.0*mass*(dx**2.)))
  t = 1.0 / (4.0 * mass * (dx**2.))
  temp[0, 0, :, :] = (mu[0] / 2.0 + 1.0 /
                      (2.0 * mass * (dx**2))) * num + (intac /
                                                       (2.0 * dx)) * inpart
  temp[0, 1, :, :] = -t * c
  temp[0, 2, :, :] = -t * cdag
  temp[0, 3, :, :] = np.eye(cutoff)
  mpol.append(temp)
  for n in range(1, N - 1):
    temp = np.zeros((4, 4, cutoff, cutoff))
    #11
    temp[0, 0, :, :] = np.eye(cutoff)
    #cdagger
    temp[1, 0, :, :] = np.copy(cdag)
    #c
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = (mu[n] / 2.0 + 1.0 /
                        (2.0 * mass * dx**2)) * num + (intac /
                                                       (dx * 2.0)) * inpart
    #c
    temp[3, 1, :, :] = -t * np.copy(c)
    #cdagger
    temp[3, 2, :, :] = -t * np.copy(cdag)
    #n
    temp[3, 3, :, :] = np.eye(cutoff)
    mpol.append(temp)

  temp = np.zeros((4, 4, cutoff, cutoff))
  #11
  temp[0, 0, :, :] = np.eye(cutoff)
  #cdagger
  temp[1, 0, :, :] = np.copy(cdag)
  #c
  temp[2, 0, :, :] = np.copy(c)
  temp[3, 0, :, :] = (mu[n] / 2 + 1.0 /
                      (2.0 * mass * dx**2)) * num + (intac /
                                                     (dx * 2.0)) * inpart
  #c
  temp[3, 1, :, :] = -2.0 * t * np.copy(c)
  #temp[3,1,:,:]=-t*np.copy(c)
  #cdagger
  temp[3, 2, :, :] = -2.0 * t * np.copy(cdag)
  #temp[3,2,:,:]=-t*np.copy(cdag)
  #n
  temp[3, 3, :, :] = np.eye(cutoff)
  mpol.append(temp)

  mpor = []

  temp = np.zeros((4, 4, cutoff, cutoff))
  #11
  temp[0, 0, :, :] = np.eye(cutoff)
  #cdagger
  temp[1, 0, :, :] = np.copy(cdag)
  #temp[1,0,:,:]=t*np.copy(cdag)
  #c
  temp[2, 0, :, :] = np.copy(c)
  #temp[2,0,:,:]=t*np.copy(c)

  temp[3, 0, :, :] = (mu[n] / 2.0 + 1.0 /
                      (2.0 * mass * dx**2)) * num + (intac /
                                                     (dx * 2.0)) * inpart
  #c
  temp[3, 1, :, :] = -t * np.copy(c)
  #cdagger
  temp[3, 2, :, :] = -t * np.copy(cdag)
  #n
  temp[3, 3, :, :] = np.eye(cutoff)
  mpor.append(temp)

  for n in range(1, N - 1):
    temp = np.zeros((4, 4, cutoff, cutoff))
    #11
    temp[0, 0, :, :] = np.eye(cutoff)
    #cdagger
    temp[1, 0, :, :] = np.copy(cdag)
    #c
    temp[2, 0, :, :] = np.copy(c)
    temp[3, 0, :, :] = (mu[n] / 2.0 + 1.0 /
                        (2.0 * mass * dx**2)) * num + (intac /
                                                       (dx * 2.0)) * inpart
    #c
    temp[3, 1, :, :] = -t * np.copy(c)
    #cdagger
    temp[3, 2, :, :] = -t * np.copy(cdag)
    #n
    temp[3, 3, :, :] = np.eye(cutoff)
    mpor.append(temp)

  #n=site
  temp = np.zeros((4, 1, cutoff, cutoff))
  temp[0, 0, :, :] = np.eye(cutoff)
  temp[1, 0, :, :] = np.copy(cdag)
  temp[2, 0, :, :] = np.copy(c)
  temp[3, 0, :, :] = (mu[N - 1] / 2.0 + 1.0 /
                      (2.0 * mass * (dx**2))) * num + (intac /
                                                       (dx * 2.0)) * inpart
  mpor.append(temp)
  return mpol, mpor


def projectedUnitcellExLiebLinigermpo(mu, interact, mass, N, dx):
  c = np.zeros((2, 2))
  for n in range(1, 2):
    c[n - 1, n] = np.sqrt(n)
  cdag = np.copy(herm(c))
  num = cdag.dot(c)
  t = 1.0 / (2.0 * mass * (dx**2.))

  mpol = []
  temp = np.zeros((1, 7, 2, 2))
  temp[0, 0, :, :] = mu[0] / 1.0 * num
  temp[0, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
  temp[0, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * (np.eye(2) - num)

  temp[0, 3, :, :] = -t * c
  temp[0, 4, :, :] = -t * cdag
  temp[0, 5, :, :] = np.copy((interact * 1.0 / (1.0 * dx)) * num)
  temp[0, 6, :, :] = np.eye(2)
  mpol.append(np.copy(temp))

  for n in range(1, N - 1):
    temp = np.zeros((7, 7, 2, 2))

    temp[0, 0, :, :] = np.eye(2)
    temp[1, 0, :, :] = np.eye(2) - num
    temp[2, 0, :, :] = num
    temp[3, 0, :, :] = np.copy(cdag)
    temp[4, 0, :, :] = np.copy(c)
    temp[5, 0, :, :] = np.copy(num)

    temp[6, 0, :, :] = mu[n] / 1.0 * num
    temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
    temp[6, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * (np.eye(2) - num)
    temp[6, 3, :, :] = -t * c
    temp[6, 4, :, :] = -t * cdag
    temp[6, 5, :, :] = np.copy((interact * 1.0 / (1.0 * dx)) * num)
    temp[6, 6, :, :] = np.eye(2)
    mpol.append(np.copy(temp))

  temp = np.zeros((7, 7, 2, 2))

  temp[0, 0, :, :] = np.eye(2)
  temp[1, 0, :, :] = np.eye(2) - num
  temp[2, 0, :, :] = num
  temp[3, 0, :, :] = np.copy(cdag)
  temp[4, 0, :, :] = np.copy(c)
  temp[5, 0, :, :] = np.copy(num)

  temp[6, 0, :, :] = mu[N - 1] / 1.0 * num
  temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
  temp[6, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * (np.eye(2) - num)
  #======================
  temp[6, 3, :, :] = -1.0 * t * c
  temp[6, 4, :, :] = -1.0 * t * cdag
  #======================
  temp[6, 5, :, :] = np.copy((interact * 1.0 / (1.0 * dx)) * num)
  temp[6, 6, :, :] = np.eye(2)
  mpol.append(np.copy(temp))

  mpor = []
  temp = np.zeros((7, 7, 2, 2))

  temp[0, 0, :, :] = np.eye(2)
  temp[1, 0, :, :] = np.eye(2) - num
  temp[2, 0, :, :] = num
  temp[3, 0, :, :] = np.copy(cdag)
  temp[4, 0, :, :] = np.copy(c)
  temp[5, 0, :, :] = np.copy(num)

  temp[6, 0, :, :] = mu[0] / 1.0 * num
  temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
  temp[6, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * (np.eye(2) - num)
  temp[6, 3, :, :] = -t * c
  temp[6, 4, :, :] = -t * cdag
  temp[6, 5, :, :] = np.copy((interact * 1.0 / (1.0 * dx)) * num)
  temp[6, 6, :, :] = np.eye(2)
  mpor.append(np.copy(temp))
  for n in range(1, N - 1):
    temp = np.zeros((7, 7, 2, 2))

    temp[0, 0, :, :] = np.eye(2)
    temp[1, 0, :, :] = np.eye(2) - num
    temp[2, 0, :, :] = num
    temp[3, 0, :, :] = np.copy(cdag)
    temp[4, 0, :, :] = np.copy(c)
    temp[5, 0, :, :] = np.copy(num)

    temp[6, 0, :, :] = mu[n] / 1.0 * num
    temp[6, 1, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * num
    temp[6, 2, :, :] = 1.0 / (2.0 * mass * (dx**2.)) * (np.eye(2) - num)
    temp[6, 3, :, :] = -t * c
    temp[6, 4, :, :] = -t * cdag
    temp[6, 5, :, :] = np.copy((interact * 1.0 / (1.0 * dx)) * num)
    temp[6, 6, :, :] = np.eye(2)
    mpor.append(np.copy(temp))

  #n=site
  temp = np.zeros((7, 1, 2, 2))
  temp[0, 0, :, :] = np.eye(2)
  temp[1, 0, :, :] = np.eye(2) - num
  #temp[1,0,:,:]=np.eye(2)
  temp[2, 0, :, :] = num
  temp[3, 0, :, :] = np.copy(cdag)
  temp[4, 0, :, :] = np.copy(c)
  temp[5, 0, :, :] = np.copy(num)
  temp[6, 0, :, :] = mu[N - 1] / 1.0 * num

  mpor.append(np.copy(temp))
  return mpol, mpor
