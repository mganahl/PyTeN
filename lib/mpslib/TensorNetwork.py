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
import numpy as np
import lib.mpslib.mpsfunctions as mf
from scipy.sparse.linalg import LinearOperator, eigs
import lib.ncon as ncon
from lib.mpslib.Container import Container

comm = lambda x, y: np.dot(x, y) - np.dot(y, x)
anticomm = lambda x, y: np.dot(x, y) + np.dot(y, x)
herm = lambda x: np.conj(np.transpose(x))
from lib.mpslib.Tensor import TensorBase, Tensor


def generate_unary_deferer(op_func):

    def deferer(cls, *args, **kwargs):
        try:
            return type(cls).__unary_operations__(cls, op_func, *args, **kwargs)
        except AttributeError:
            raise (AttributeError(
                "cannot generate unary deferer for class withtou __unary_operations__"
            ))

    return deferer


def ndarray_initializer(numpy_func, shapes, *args, **kwargs):
    """
    initializer to create a list of tensors of type Tensor
    Parameters:
    ---------------------
    numpy_func:       callable
                      a numpy function like np.random.random_sample
    shapes:           list of tuple
                      the shapes of the individual tensors

    
    *args,**kwargs:   if numpy_func is not one of np.random functions, these are passed on to numpy_func
    possible **kwargs are:
    minval:        float
                   lower bound for np.random.rand
    maxval:        float
                   upper bound for np.random.rand
    mean:          float
                   mean value for np.random.randn
    std:           float
                   standard deviation for np.random.randn
    Returns:
    -------------------------
    list of Tensor objects of shape ```shapes```, initialized with numpy_func
    """

    minval = kwargs.get('minval', -0.5)
    maxval = kwargs.get('maxval', 0.5)
    mean = kwargs.get('mean', 0.0)
    std = kwargs.get('std', 0.7)
    dtype = kwargs.get('dtype', np.float64)
    if numpy_func in (np.random.random_sample, np.random.rand):
        if np.issubdtype(dtype, np.complexfloating):
            return [
                ((maxval - minval) * numpy_func(shape).view(Tensor) + minval +
                 1j *
                 ((maxval - minval) * numpy_func(shape).view(Tensor) + minval)
                ).astype(dtype) for shape in shapes
            ]

        elif np.issubdtype(dtype, np.floating):
            return [((maxval - minval) * numpy_func(shape).view(Tensor) +
                     minval).astype(dtype) * std for shape in shapes]

    elif numpy_func == np.random.randn:
        if np.issubdtype(dtype, np.complexfloating):
            return [
                (std * numpy_func(*shape).view(Tensor) + mean + 1j * std *
                 (numpy_func(*shape).view(Tensor) + 1j * mean)).astype(dtype)
                for shape in shapes
            ]

        elif np.issubdtype(dtype, np.floating):
            tensors = [
                (std * numpy_func(*shape).view(Tensor) + mean).astype(dtype)
                for shape in shapes
            ]
            return [(std * numpy_func(*shape).view(Tensor) + mean).astype(dtype)
                    for shape in shapes]

    else:
        if np.issubdtype(dtype, np.complexfloating):
            return [
                numpy_func(shape, *args, **kwargs).view(Tensor) +
                1j * numpy_func(shape, *args, **kwargs).view(Tensor)
                for shape in shapes
            ]
        elif np.issubdtype(dtype, np.floating):
            return [
                numpy_func(shape, *args, **kwargs).view(Tensor)
                for shape in shapes
            ]


class TensorNetwork(Container, np.lib.mixins.NDArrayOperatorsMixin):
    _HANDLED_UFUNC_TYPES = (numbers.Number, np.ndarray)

    def __init__(self, tensors=[], shape=(), name=None, fromview=True):
        """
        initialize an unnormalized TensorNetwork from a list of tensors
        Parameters
        ----------------------------------------------------
        tensors: list() of tensors
                 can be either np.ndarray or any other object 
                 a list containing the tensors of the network
                 the entries in tensors should have certain properties that
                 tensors usually have
        shape:   tuple
                 the shape of the tensor networl
        """
        #TODO: add attribute checks for the elements of tensors
        super().__init__(name)  #initialize the Container base class
        if isinstance(tensors, np.ndarray):
            if not fromview:
                self._tensors = copy.deepcopy(tensors)
            else:
                self._tensors = tensors.view()
            self._norm = np.result_type(*tensors.ravel()).type(1.0)
            self.tensortype = type(tensors.ravel()[0])
        elif isinstance(tensors, list):
            N = len(tensors)
            if shape != ():
                if not (np.prod(shape) == N):
                    raise ValueError(
                        'shape={0} incompatible with len(tensors)={1}'.format(
                            shape, N))
                if not isinstance(shape, tuple):
                    raise TypeError(
                        'TensorNetwor.__init__(): got wrong type for shape; only tuples are allowed'
                    )
                _shape = shape
            else:
                _shape = tuple([N])
            if tensors != []:
                self.tensortype = type(tensors[0])
            else:
                self.tensortype = object

            self._tensors = np.empty(_shape, dtype=self.tensortype)
            for n in range(len(tensors)):
                self._tensors.flat[n] = tensors[n].view(Tensor)

            self._norm = np.result_type(*self._tensors).type(1.0)

        else:
            raise TypeError(
                'TensorNetwork.__init__(tensors): tensors has invlaid tupe {0}'.
                type(tensors))

    def set_tensors(self, tensors):
        for n in range(len(tensors)):
            self._tensors[n] = tensors[n]

    def __in_place_unary_operations__(self, operation, *args, **kwargs):
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
        for n, x in np.ndenumerate(self):
            self[n] = operation(self[n], *args, **kwargs)

    def __unary_operations__(self, operation, *args, **kwargs):
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
        obj = self.copy()
        obj.__in_place_unary_operations__(operation, *args, **kwargs)
        return obj

    def __array__(self):
        return self._tensors

    def reshape(self, newshape, order='C'):
        """
        returns a reshaped view of self
        compatible with np.reshape
        """
        view = self.view()
        view._tensors = np.reshape(view._tensors, newshape, order=order)
        return view

    @property
    def num_sites(self):
        """
        length of the Tensor network
        Returns:
        ----------------
        int: the length of the MPS
        """
        return np.prod(self._tensors.shape)

    @property
    def dtype(self):
        """
        Returns:
        ----------------
        np.dtype.type: the data type of the MPS tensors
        """
        return np.result_type(*[t.dtype for t in self._tensors], self._norm)

    @property
    def shape(self):
        return self._tensors.shape

    @property
    def tensors(self):
        """
        return a flat np.ndarray view of the tensors in TensorNetwork
        """
        return self._tensors.ravel().view()

    @classmethod
    def random(cls,
               shape=(),
               tensorshapes=(),
               name=None,
               initializer=ndarray_initializer,
               *args,
               **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        shape:           tuple
                         shape of the Tensor Network 
        tensorshapes:    tuple 
                         shapes if the tensors in TensorNetwork
        name:            str or None
                         name of the TensorNetwork
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        *args,**kwargs:  arguments and keyword arguments for ```initializer```
        """
        return cls(
            tensors=initializer(np.random.random_sample,
                                np.prod(shape) * [tensorshapes], *args,
                                **kwargs),
            name=name,
            shape=shape)

    @classmethod
    def zeros(cls,
              shape=(),
              tensorshapes=(),
              name=None,
              initializer=ndarray_initializer,
              *args,
              **kwargs):
        """
        generate a TensorNetwork of zeros-tensors
        Parameters:
        ----------------------------------------------
        shape:           tuple
                         shape of the Tensor Network 
        tensorshapes:    tuple 
                         shapes if the tensors in TensorNetwork
        name:            str or None
                         name of the TensorNetwork
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        *args,**kwargs:  arguments and keyword arguments for ```initializer```
                        
        """
        return cls(
            tensors=initializer(np.zeros,
                                np.prod(shape) * [tensorshapes], *args,
                                **kwargs),
            name=name,
            shape=shape)

    @classmethod
    def ones(cls,
             shape=(),
             tensorshapes=(),
             name=None,
             initializer=ndarray_initializer,
             *args,
             **kwargs):
        """
        generate a TensorNetwork of ones-tensors
        Parameters:
        ----------------------------------------------
        shape:           tuple
                         shape of the Tensor Network 
        tensorshapes:    tuple 
                         shapes if the tensors in TensorNetwork
        name:            str or None
                         name of the TensorNetwork
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        *args,**kwargs:  arguments and keyword arguments for ```initializer```
                        
        """
        return cls(
            tensors=initializer(np.ones,
                                np.prod(shape) * [tensorshapes], *args,
                                **kwargs),
            name=name,
            shape=shape)

    @classmethod
    def empty(cls,
              shape=(),
              tensorshapes=(),
              name=None,
              initializer=ndarray_initializer,
              *args,
              **kwargs):
        """
        generate a TensorNetwork of empty tensors
        Parameters:
        ----------------------------------------------
        shape:           tuple
                         shape of the Tensor Network 
        tensorshapes:    tuple 
                         shapes if the tensors in TensorNetwork
        name:            str or None
                         name of the TensorNetwork
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        *args,**kwargs:  arguments and keyword arguments for ```initializer```
        """
        return cls(
            tensors=initializer(np.empty,
                                np.prod(shape) * [tensorshapes], *args,
                                **kwargs),
            name=name,
            shape=shape)

    def __getitem__(self, n, **kwargs):
        return self._tensors[n]

    def __setitem__(self, n, tensor, **kwargs):
        self._tensors[n] = tensor

    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """

        inds = np.unravel_index(range(self.num_sites), dims=self.shape)
        inds = list(zip(*inds))
        return ''.join(['Name: ', str(self.name), '\n\n '] + [
            'TN' + str(index) + ' \n\n ' + self[index].__str__() + ' \n\n '
            for index in inds
        ] + ['\n\n Z=', str(self._norm)])

    def __len__(self):
        """
        """
        return len(self._tensors)

    def __iter__(self):
        """
        Returns:
        iterator:  an iterator over the tensors of the TensorNetwork
        """
        return iter(self._tensors)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self._norm is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """
        #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
        #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
        if ufunc == np.true_divide:
            return self.__idiv__(inputs[1])

        out = kwargs.get('out', ())
        for arg in inputs + out:
            if not isinstance(arg,
                              self._HANDLED_UFUNC_TYPES + (TensorNetwork,)):
                return NotImplemented
        if out:
            #takes care of in-place operations
            result = []
            for n, x in np.ndenumerate(self._tensors):
                kwargs['out'] = tuple(
                    o[n] if isinstance(o, type(self)) else o for o in out)
                ipts = [
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ]
                result.append(getattr(ufunc, method)(*ipts, **kwargs))
        else:
            result = [
                getattr(ufunc, method)(*[
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ], **kwargs)
                for n, x in np.ndenumerate(self._tensors)
            ]

        if method == 'reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis = kwargs.get('axis', ())
            if axis == None:
                return getattr(ufunc, method)(result)
            else:
                raise NotImplementedError(
                    'TensorNetwork.__array_ufunc__ with argument axis!=None not implemented'
                )

        else:
            return TensorNetwork(tensors=result, shape=self.shape, name=None)

    def __mul__(self, num):
        """
        left-multiplies "num" with TensorNetwork, i.e. returns TensorNetwork*num;
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
            raise TypeError(
                "in TensorNetwork.__mul__(self,num): num is not a number")
        new = self.copy()
        new._norm *= num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self._norm*num)

    def __imul__(self, num):
        """
        left-multiplies "num" with TensorNetwork, i.e. returns TensorNetwork*num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError(
                "in TensorNetwork.__mul__(self,num): num is not a number")
        self._norm *= num
        return self

    def __idiv__(self, num):
        """
        left-divides "num" with TensorNetwork, i.e. returns TensorNetwork*num;
        note that "1./num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError(
                "in TensorNetwork.__mul__(self,num): num is not a number")
        self._norm /= num
        return self

    def __truediv__(self, num):
        """
        left-divides "num" with TensorNetwork, i.e. returns TensorNetwork/num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError(
                "in TensorNetwork.__mul__(self,num): num is not a number")
        new = self.copy()
        new._norm /= num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self._norm/num)

    def __rmul__(self, num):
        """
        right-multiplies "num" with TensorNetwork, i.e. returns num*TensorNetwork;
        WARNING: if you are using numpy number types, i.e. np.float, np.int, ..., 
        the right multiplication of num with TensorNetwork, i.e. num*TensorNetwork, returns 
        an np.darray instead of an TensorNetwork. 
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state

        """
        if not np.isscalar(num):
            raise TypeError(
                "in TensorNetwork.__mul__(self,num): num is not a number")
        new = self.copy()
        new._norm *= num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self._norm*num)

    def __add__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented

    def contract(self, other):
        return NotImplemented

    @property
    def real(self):
        return self.__array_ufunc__(np.real, '__call__', self)

    @property
    def imag(self):
        return self.__array_ufunc__(np.imag, '__call__', self)


class MPSBase(TensorNetwork):
    """
    base class for all MPS (Finite, Infinite, Canonized)
    """

    def __init__(self, tensors=[], name=None, fromview=True):
        super().__init__(
            tensors=tensors, shape=(), name=name, fromview=fromview)

    @property
    def D(self):
        """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
        return ([self.get_tensor(0).shape[0]] +
                [self.get_tensor(n).shape[1] for n in range(self.num_sites)])

    @property
    def d(self):
        """
        returns a list containig the bond-dimensions of the MPS
        """
        return [self.get_tensor(n).shape[2] for n in range(self.num_sites)]

    def transfer_op(self, site, direction, x):
        """
        """
        A = self.get_tensor(site)
        return mf.transfer_operator([A], [A], direction=direction, x=x)

    def get_env_left(self, site):
        raise NotImplementedError()

    def get_env_right(self, site):
        raise NotImplementedError()

    def get_envs_right(self, sites):
        """Right environments for ```sites```
        This default implementation is not necessarily optimal.
        Returns the environments as a dictionary, indexed by site number.
        Parameters:
        ----------------------
        sites:   list of int 
                 the sites for which the environments should be calculated
        Returns:

        dict():  mapping int to tf.tensor
                 the right environment for each site in ```sites```
        """

        if not np.all(np.array(sites) >= 0):
            raise ValueError('get_envs_right: sites have to be >= 0')
        n2 = max(sites)
        n1 = min(sites)
        rs = {}
        r = self.get_env_right(n2)
        for n in range(n2, n1 - 1, -1):
            if n in sites:
                rs[n] = r
            r = self.transfer_op(n % self.num_sites, 'r', r)
        return rs

    def get_envs_left(self, sites):
        """left environments for ```sites```
        This default implementation is not necessarily optimal.
        Returns the environments as a dictionary, indexed by site number.
        Parameters:
        ----------------------
        sites:   list of int 
                 the sites for which the environments should be calculated
        Returns:

        dict():  mapping int to tf.tensor
                 the left environment for each site in ```sites```
        """

        if not np.all(np.array(sites) >= 0):
            raise ValueError('get_envs_left: sites have to be >= 0')
        n2 = max(sites)
        n1 = min(sites)
        ls = {}
        l = self.get_env_left(n1)
        for n in range(n1, n2 + 1):
            if n in sites:
                ls[n] = l
            l = self.transfer_op(n % self.num_sites, 'l', l)
        return ls

    def get_tensor(self, n):
        """
        get_tensor returns an mps tensors, possibly contracted with the center matrix
        by convention, the center matrix is contracted if n==self.pos
        the centermatrix is always absorbed from the left into the mps tensor, unless site==N-1, 
        in which case it is absorbed from the right

        the connector matrix is always absorbed into the right-most mps tensors

        """
        n = n % self.num_sites
        if n < 0:
            raise ValueError("n={} is less than zero!".format(n))

        if self.pos < len(self):
            if n == self.pos:
                out = ncon.ncon([self.centermatrix, self._tensors[n]],
                                [[-1, 1], [1, -2, -3]])
            else:
                out = self._tensors[n]
        elif self.pos == len(self):
            if n == (self.pos - 1):
                out = ncon.ncon([self._tensors[n], self.centermatrix],
                                [[-1, 1, -3], [1, -2]])
            else:
                out = self._tensors[n]
        else:
            raise ValueError("Invalid tensor index {}".format(n))

        if n == (len(self) - 1):
            return ncon.ncon([out, self.connector], [[-1, 1, -3], [1, -2]])
        else:
            return out

    def measure_1site_correlator(self, op1, op2, n1, n2_max):
        """
        state has top be normalized
        """
        N = self.num_sites
        if n1 < 0 or n1 >= N:
            raise ValueError(
                "Site n1 out of range: {} not between 0 and {}.".format(n1, N))
        ls = self.get_envs_left([n1])
        rs = self.get_envs_right([n % N for n in range(n1 + 1, n2_max + 1)])

        A = self.get_tensor(n1)
        l = ncon.ncon([ls[n1], A, op1, A.conj()], [(0, 1), (0, -1, 2), (3, 2),
                                                   (1, -2, 3)])
        c = []
        for n in range(n1 + 1, n2_max + 1):
            r = rs[n % N]
            A = self.get_tensor(n % N)
            res = ncon.ncon([l, A, op2, A.conj(), r],
                            [[1, 4], [1, 5, 2], [3, 2], [4, 6, 3], [5, 6]])
            c.append(res)

            if n < n2_max:
                l = self.transfer_op(n % N, 'left', l)

        return np.array(c)

    
    def measure_1site_ops(self, ops, sites):
        """
        Expectation value of list of  single-site operators on sites
        this is still not optimal but much faster than running expval_1site on many sitesg
        Parameters
        --------------------------
        ops:    list of tf.tensor
                local operators to be measure
        sites:  list of int 
                sites where the operators live
                ```sites``` can be in any order and have any number of sites appear arbitralily often
        Returns:
        --------------------------             
        a list of measurements, in the same order as sites were passed
        """
        if not len(ops) == len(sites):
            raise ValueError(
                'measure_1site_ops: len(ops) has to be len(sites)!')
        right_envs = self.get_envs_right(sites)
        left_envs = self.get_envs_left(sites)
        res = []
        #norm=self.norm()
        for n in range(len(sites)):
            op = ops[n]
            r = right_envs[sites[n]]
            l = left_envs[sites[n]]
            A = self.get_tensor(sites[n])
            res.append(
                ncon.ncon([l, A, op, A.conj(), r], [(0, 1), (0, 4, 2), (3, 2),
                                                    (1, 5, 3), (4, 5)]))

        return np.array(res)  #[o/norm for o in res]

    def norm(self):
        return self._norm

    def normalize(self):
        raise NotImplementedError()


class MPS(MPSBase):

    @classmethod
    def from_tensors(cls, tensors, name=None):
        return cls(tensors, name=name, fromview=True)

    def __init__(self, tensors, name=None, fromview=True):
        """
        no checks are performed to see wheter the provided tensors can be contracted
        """
        super().__init__(tensors=tensors, name=name, fromview=fromview)
        self.mat = tensors[-1].eye(1)
        self._right_mat = tensors[-1].eye(1)
        self._connector = self.mat.inv()
        #do not shift mps position here! some routines assume that the mps tensors are not changed at initialization
        self._position = self.num_sites

    def __len__(self):
        return self.num_sites

    @property
    def centermatrix(self):
        return self.mat


    @property
    def connector(self):
        return self._connector

    def normalize(self, **kwargs):
        self.mat/=np.sqrt(self.mat.norm())        
        #self.canonize(numeig=1, **kwargs)
        self._norm = self.dtype.type(1)

    def roll(self, num_sites):
        """
        roll the mps unitcell by num_sites (shift it by num_sites)
        """
        self.position(num_sites)
        centermatrix = self.mat  #copy the center matrix
        self.position(len(self))  #move cenermatrix to the right
        new_center_matrix = ncon.ncon([self.mat, self.connector], [[-1, 1], [1, -2]])

        self._position = num_sites
        self.mat = centermatrix
        self.position(0)
        new_center_matrix = ncon.ncon([new_center_matrix, self.mat],
                                 [[-1, 1], [1, -2]])
        tensors=[self._tensors[n] for n in range(num_sites,len(self._tensors))]\
            + [self._tensors[n] for n in range(num_sites)]
        self.set_tensors(tensors)
        self._connector = np.linalg.inv(centermatrix)
        self._right_mat = centermatrix
        self.mat = new_center_matrix
        self._position = len(self) - num_sites

    def schmidt_spectrum(self, n, canonize=True, **kwargs):
        """
        schmidt_spectrum(n):

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
        if canonize == True:
            self.canonize(**kwargs)
        self.position(n)
        U, S, V, _ = self.mat.svd(full_matrices=False)
        return S

    def get_env_left(self, site):
        """
        obtain the left environment of ```site```
        """
        site = site % len(self)
        if site >= len(self) or site < 0:
            raise IndexError(
                'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'
                .format(site, len(self)))

        if site <= self.pos:
            return self[site - 1].eye(1)
        else:
            l = self[self.pos].eye(0)
            for n in range(self.pos, site):
                l = self.transfer_op(n, direction='l', x=l)
            return l

    def get_env_right(self, site):
        """
        obtain the right environment of ```site```
        """

        site = site % len(self)
        if site >= len(self) or site < 0:
            raise IndexError(
                'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'
                .format(site, len(self)))

        if site == len(self) - 1:
            return ncon.ncon(
                [self._right_mat, self._right_mat.conj()], [[-1, 1], [-2, 1]])

        elif site >= self.pos and site < len(self) - 1:
            return self[site].eye(1)
        else:
            r = ncon.ncon(
                [self.centermatrix, self.centermatrix.conj()],
                [[-1, 1], [-2, 1]])
            for n in range(self.pos - 1, site, -1):
                r = self.transfer_op(n, 'r', r)
            return r

    def get_unitcell_transfer_op(self, direction):
        """
        Returns a function that implements the transfer operator for the
        entire unit cell.
        """
        if direction in ('l', 'left', 1):
            As = [self.get_tensor(n) for n in range(len(self))]
        elif direction in ('r', 'right', -1):
            As = [self.get_tensor(n) for n in reversed(range(len(self)))]
        else:
            raise ValueError("Invalid direction: {}".format(direction))

        def t_op(x):
            for A in As:
                x = mf.transfer_operator([A], [A], direction, x)
            return x

        return t_op

    def unitcell_transfer_op(self, direction, x):
        """
        use get_unitcell_transfer_op for sparse diagonalization
        """

        if direction in ('l', 'left', 1):
            if not x.shape[0] == self.D[0]:
                raise ValueError(
                    'shape of x[0] does not match the shape of mps.D[0]')
            if not x.shape[1] == self.D[0]:
                raise ValueError(
                    'shape of x[1] does not match the shape of mps.D[0]')

            l = x
            for n in range(len(self)):
                l = self.transfer_op(n, direction='l', x=l)
            return l
        if direction in ('r', 'right', -1):
            if not x.shape[0] == self.D[-1]:
                raise ValueError(
                    'shape of x[0] does not match the shape of mps.D[-1]')
            if not x.shape[1] == self.D[-1]:
                raise ValueError(
                    'shape of x[1] does not match the shape of mps.D[-1]')
            r = x
            for n in range(len(self) - 1, -1, -1):
                r = self.transfer_op(n, direction='r', x=r)
            return r

    def TMeigs_naive(self, direction, init=None, precision=1E-12, nmax=100000):
        """
        calculate the left and right dominant eigenvector of the MPS-unit-cell transfer operator
        usint power method

        Parameters:
        ------------------------------
        direction:     int or str

                       if direction in (1,'l','left')   return the left dominant EV
                       if direction in (-1,'r','right') return the right dominant EV
        init:          Tensor
                       initial guess for the eigenvector
        precision:     float
                       desired precision of the dominant eigenvalue
        nmax:          int
                       max number of iterations

        Returns:
        ------------------------------
        (eta,x,it,diff):
        eta:  float
              the eigenvalue
        x:    Tensor
              the dominant eigenvector (in matrix form)
        it:   int 
              number of iterations
        diff: float
              the final precision
        
        """

        if self.D[0] != self.D[-1]:
            raise ValueError(
                " in TMeigs: left and right ancillary dimensions of the MPS do not match"
            )
        if np.all(init != None):
            initial = init
        tensors = [self.get_tensor(n) for n in range(len(self))]
        return mf.TMeigs_naive(
            tensors=tensors,
            direction=direction,
            init=init,
            precision=precision,
            nmax=nmax)

    def TMeigs(self,
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
        direction:     int or str

                       if direction in (1,'l','left')   return the left dominant EV
                       if direction in (-1,'r','right') return the right dominant EV
        init:          Tensor
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
        x:   Tensor
             the dominant eigenvector (in matrix form)
        """
        if self.D[0] != self.D[-1]:
            raise ValueError(
                " in TMeigs: left and right ancillary dimensions of the MPS do not match"
            )
        if np.all(init != None):
            initial = init
        tensors = [self.get_tensor(n) for n in range(len(self))]
        return mf.TMeigs(
            tensors=tensors,
            direction=direction,
            init=init,
            precision=precision,
            ncv=ncv,
            nmax=nmax,
            numeig=numeig,
            which='LR')

    def regauge(self,
                gauge,
                init=None,
                precision=1E-12,
                ncv=50,
                nmax=1000,
                numeig=6,
                pinv=1E-50,
                warn_thresh=1E-8):
        raise NotImplementedError()
        """
        regauge the MPS into left or right canonical form (inplace)

        Parameters:
        ------------------------------
        
        gauge:         int or str
                       for (1,'l','left'): bring into left gauge
                       for (-1,'r','right'): bring into right gauge

        init:          Tensor
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
        pinv:          float
                       pseudoinverse cutoff
        warn_thresh:   float 
                       threshold value; if TMeigs returns an eigenvalue with imaginary value larger than 
                       ```warn_thresh```, a warning is issued 

        Returns:
        ----------------------------------
        None
        """

        if gauge in ('left', 'l', 1):
            self.position(0)
            eta, l = self.TMeigs(
                direction='left',
                init=init,
                precision=precision,
                ncv=ncv,
                nmax=nmax,
                numeig=numeig)
            self.mat /= np.sqrt(eta)
            if np.abs(np.imag(eta)) / np.abs(np.real(eta)) > warn_thresh:
                print(
                    'in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',
                    eta)

            l = l / l.tr()
            l = (l + l.conj().transpose()) / 2.0

            eigvals, u = l.eigh()
            eigvals[eigvals < 0.0] = 0.0                                    
            eigvals /= np.sqrt(ncon.ncon([eigvals, eigvals.conj()], [[1], [1]]))
            abseigvals = eigvals.abs()
            eigvals[abseigvals <= pinv] = 0.0

            inveigvals = 1.0 / eigvals
            inveigvals[abseigvals <= pinv] = 0.0

            y = ncon.ncon([u, np.sqrt(eigvals).diag()], [[-2, 1], [1, -1]])
            invy = ncon.ncon([np.sqrt(inveigvals).diag(),
                              u.conj()], [[-2, 1], [-1, 1]])

            #multiply y to the left and y^{-1} to the right bonds of the tensor:
            #the index contraction looks weird, but is correct; my l matrices have their 0-index on the non-conjugated top layer
            self.mat = ncon.ncon(
                [y, self.connector, self.mat], [[-1, 1], [1, 2], [2, -2]]
            )  #connector is absorbed at the left end here because this is the  convention if self.pos==0 (which is the case here)
            self._tensors[-1] = ncon([self._tensors[-1], invy],
                                     [[-1, -2, 1], [1, -3]])

            #the easier solution is to do
            #self.position(len(self))

            #however, the current implementatin serves as a check that indeed the rightmost tensor is right orthogonal
            #also, below self.mat and self.connector are left as 11, which is nice
            self.position(len(self) - 1)
            self._tensors[-1] = ncon.ncon([self.mat, self._tensors[-1]],
                                          [[-1, 1], [1, -2, -3]])
            Z = ncon.ncon([self._tensors[-1], self._tensors[-1].conj()],
                          [[1, 2, 3], [1, 2, 3]]) / np.sum(self.D[-1])
            self._tensors[-1] /= np.sqrt(Z)
            self.mat = self._tensors[-1].eye(1)
            self._position = len(self)
            self._connector = self._tensors[-1].eye(1)

        if gauge in ('right', 'r', -1):
            self.position(len(self))
            eta, r = self.TMeigs(
                direction='right',
                init=init,
                precision=precision,
                ncv=ncv,
                nmax=nmax,
                numeig=numeig)
            self.mat /= np.sqrt(eta)
            if np.abs(np.imag(eta)) / np.abs(np.real(eta)) > warn_thresh:
                print(
                    'in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',
                    eta)

            r = r / r.tr()
            r = (r + r.conj().transpose()) / 2.0

            eigvals, u = r.eigh()
            eigvals[eigvals < 0.0] = 0.0                        
            eigvals /= np.sqrt(ncon.ncon([eigvals, eigvals.conj()], [[1], [1]]))
            abseigvals = eigvals.abs()
            eigvals[abseigvals <= pinv] = 0.0

            inveigvals = 1.0 / eigvals
            inveigvals[abseigvals <= pinv] = 0.0

            x = ncon.ncon([u, np.sqrt(eigvals).diag()], [[-1, 1], [1, -2]])
            invx = ncon.ncon([np.sqrt(inveigvals).diag(),
                              u.conj()], [[-1, 1], [-2, 1]])

            #multiply y to the left and y^{-1} to the right bonds of the tensor:
            #the index contraction looks weird, but is correct; my l matrices have their 0-index on the non-conjugated top layer
            self.mat = ncon.ncon([self.mat, self.connector, x],
                                 [[-1, 1], [1, 2], [2, -2]])

            self.position(1)
            self._tensors[0] = ncon.ncon([invx, self._tensors[0], self.mat],
                                         [[-1, 1], [1, 2, -3], [2, -2]])
            Z = ncon.ncon([self._tensors[0], self._tensors[0].conj()],
                          [[1, 2, 3], [1, 2, 3]]) / np.sum(self.D[-1])
            self._tensors[0] /= np.sqrt(Z)

            self.mat = self._tensors[0].eye(0) / np.sum(self.D[0])
            self._position = 0
            self._connector = self._tensors[0].eye(0) * np.sum(self.D[0])

    def canonize(self,
                 init=None,
                 precision=1E-12,
                 ncv=50,
                 nmax=1000,
                 numeig=1,
                 power_method=False,
                 pinv=1E-30,
                 truncation_threshold=1E-15,
                 D=None,
                 warn_thresh=1E-8):
        """
        bring the MPS into Schmidt canonical form

        Parameters:
        ------------------------------
        init:          Tensor
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
        pinv:          float
                       pseudoinverse cutoff
        truncation_threshold: float 
                              truncation threshold for the MPS, if < 1E-15, no truncation is done
        D:             int or None 
                       if int is given, bond dimension will be reduced to `D`; `D=None` has no effect
        warn_thresh:   float 
                       threshold value; if TMeigs returns an eigenvalue with imaginary value larger than 
                       ```warn_thresh```, a warning is issued 

        Returns:
        ----------------------------------
        None
        """
        self.position(0)
        if not power_method:
            eta, l = self.TMeigs(
                direction='left',
                init=init,
                nmax=nmax,
                precision=precision,
                ncv=ncv,
                numeig=numeig)
        elif power_method:
            eta, l, _, _ = self.TMeigs_naive(
                direction='left', init=init, nmax=nmax, precision=precision)
        sqrteta = np.real(eta)
        self.mat /= sqrteta

        if np.abs(np.imag(eta)) / np.abs(np.real(eta)) > warn_thresh:
            print(
                'in mpsfunctions.py.regaugeIMPS: warning: found eigenvalue eta with large imaginary part: ',
                eta)

        l = l / l.tr()
        l = (l + l.conj().transpose()) / 2.0

        eigvals_left, u_left = l.eigh()
        eigvals_left /= np.sqrt(
            ncon.ncon([eigvals_left, eigvals_left.conj()], [[1], [1]]))
        inveigvals_left=eigvals_left.zeros(eigvals_left.shape[0])
        eigvals_left[eigvals_left <= pinv] = 0.0
        inveigvals_left[eigvals_left > pinv] = 1.0/eigvals_left[eigvals_left > pinv]

        y = ncon.ncon([u_left, np.sqrt(eigvals_left).diag()],
                      [[-2, 1], [1, -1]])
        invy = ncon.ncon([np.sqrt(inveigvals_left).diag(),
                          u_left.conj()], [[-2, 1], [-1, 1]])
        if not power_method:
            eta, r = self.TMeigs(
                direction='right',
                init=init,
                nmax=nmax,
                precision=precision,
                ncv=ncv,
                numeig=numeig)
        elif power_method:
            eta, r, _, _ = self.TMeigs_naive(
                direction='right', init=init, nmax=nmax, precision=precision)

        r = r / r.tr()
        r = (r + r.conj().transpose()) / 2.0
        eigvals_right, u_right = r.eigh()
        eigvals_right[eigvals_right <= pinv] = 0.0
        
        eigvals_right /= np.sqrt(
            ncon.ncon([eigvals_right, eigvals_right.conj()], [[1], [1]]))
        inveigvals_right=eigvals_right.zeros(eigvals_right.shape[0])
        inveigvals_right[eigvals_right > pinv] = 1.0/eigvals_right[eigvals_right> pinv]


        x = ncon.ncon([u_right, np.sqrt(eigvals_right).diag()],
                      [[-1, 1], [1, -2]])
        invx = ncon.ncon([np.sqrt(inveigvals_right).diag(),
                          u_right.conj()], [[-1, 1], [-2, 1]])
        U, lam, V, _ = ncon.ncon([y, x], [[-1, 1], [1, -2]]).svd(truncation_threshold=truncation_threshold,D=D)
        
        self._tensors[0] = ncon.ncon(
            [lam.diag(), V, invx, self.mat, self._tensors[0]],
            [[-1, 1], [1, 2], [2, 3], [3, 4], [4, -2, -3]])
        self._tensors[-1] = ncon.ncon(
            [self._tensors[-1], self.connector, invy, U],
            [[-1, 1, -3], [1, 2], [2, 3], [3, -2]])
        self.mat = self._tensors[0].eye(0)
        self._connector = self._tensors[-1].eye(1)

        self.position(len(self) - 1)
        self._tensors[-1] = self.get_tensor(len(self) - 1)
        Z = ncon.ncon([self._tensors[-1], self._tensors[-1].conj()],
                          [[1, 2, 3], [1, 2, 3]]) / np.sum(self.D[-1])
        
        self._tensors[-1] /= np.sqrt(Z)
        lam_norm = np.sqrt(ncon.ncon([lam, lam], [[1], [1]]))
        lam = lam / lam_norm
        self.mat = lam.diag()
        self._position = len(self)
        self._connector = (1.0 / lam).diag()
        self._right_mat = lam.diag()
        self._norm = self.dtype.type(1)

    def __in_place_unary_operations__(self, operation, *args, **kwargs):
        """
        implements in-place unary operations on MPS._tensors and MPS.mat
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        None
        """
        super().__in_place_unary_operations__(operation, *args, **kwargs)
        self.mat = operation(self.mat, *args, **kwargs)

    def __unary_operations__(self, operation, *args, **kwargs):
        """
        implements unary operations on MPS tensors
        Parameters:
        ----------------------------------------
        operation: method
                   the operation to be applied to the mps tensors
        *args,**kwargs: arguments of operation

        Returns:
        -------------------
        MPS:  MPS object obtained from acting with operation on each individual MPS tensor
        """
        obj = self.copy()
        obj.__in_place_unary_operations__(operation, *args, **kwargs)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self._norm is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """

        if ufunc == np.true_divide:
            #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
            #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
            return self.__idiv__(inputs[1])

        out = kwargs.get('out', ())
        for arg in inputs + out:
            if not isinstance(arg, self._HANDLED_UFUNC_TYPES + (type(self),)):
                return NotImplemented

        if out:
            #takes care of in-place operations
            result = []
            for n, x in np.ndenumerate(self):
                kwargs['out'] = tuple(
                    o[n] if isinstance(o, type(self)) else o for o in out)
                ipts = [
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ]
                result.append(getattr(ufunc, method)(*ipts, **kwargs))
            matipts = [
                ipt.mat if isinstance(ipt, type(self)) else ipt
                for ipt in inputs
            ]
            matresult = getattr(ufunc, method)(*matipts, **kwargs)
        else:
            result = [
                getattr(ufunc, method)(*[
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ], **kwargs)
                for n, x in np.ndenumerate(self)
            ]
            matresult = getattr(ufunc, method)(*[
                ipt.mat if isinstance(ipt, type(self)) else ipt
                for ipt in inputs
            ], **kwargs)

        if method == 'reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis = kwargs.get('axis', ())
            if axis == None:
                return getattr(ufunc, method)(result + [matresult])
            else:
                raise NotImplementedError(
                    'MPS.__array_ufunc__ with argument axis!=None not implemented'
                )

        else:
            obj = MPS(tensors=result, name=None)
            obj.mat = matresult
            return obj

    @property
    def pos(self):
        """
        Returns:
        ----------------------------
        int: the current position of the center bond
        """
        return self._position

    @classmethod
    def random(cls,
               D=[2, 2],
               d=[2],
               name=None,
               initializer=ndarray_initializer,
               numpy_func=np.random.random_sample,
               *args,
               **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D) != len(d) + 1:
            raise ValueError('len(D)!=len(d)+1')

        return cls(
            tensors=initializer(
                numpy_func=numpy_func,
                shapes=[(D[n], D[n + 1], d[n]) for n in range(len(d))],
                *args,
                **kwargs),
            name=name)

    @classmethod
    def zeros(cls,
              D=[2, 2],
              d=[2],
              name=None,
              initializer=ndarray_initializer,
              *args,
              **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D) != len(d) + 1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(
            tensors=initializer(
                numpy_func=np.zeros,
                shapes=[(D[n], D[n + 1], d[n]) for n in range(len(d))],
                *args,
                **kwargs),
            name=name)

    @classmethod
    def ones(cls,
             D=[2, 2],
             d=[2],
             name=None,
             initializer=ndarray_initializer,
             *args,
             **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D) != len(d) + 1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(
            tensors=initializer(
                numpy_func=np.ones,
                shapes=[(D[n], D[n + 1], d[n]) for n in range(len(d))],
                *args,
                **kwargs),
            name=name)

    @classmethod
    def empty(cls,
              D=[2, 2],
              d=[2],
              name=None,
              initializer=ndarray_initializer,
              *args,
              **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D) != len(d) + 1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(
            tensors=initializer(
                numpy_func=np.empty,
                shapes=[(D[n], D[n + 1], d[n]) for n in range(len(d))],
                *args,
                **kwargs),
            name=name)

    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """
        inds = range(len(self))
        s1=['Name: ',str(self.name),'\n\n ']+['MPS['+str(ind)+'] of shape '+str(self[ind].shape)+'\n\n '+self[ind].__str__()+' \n\n ' for ind in range(self.pos)]+\
            ['center matrix \n\n ',self.mat.__str__()]+['\n\n MPS['+str(ind)+'] of shape '+str(self[ind].shape)+'\n\n '+self[ind].__str__()+' \n\n ' for ind in range(self.pos,len(self))]+['\n\n Z=',str(self._norm)]
        return ''.join(s1)

    def diagonalize_center_matrix(self):
        """
        diagonalizes the center matrix and pushes U and V onto the left and right MPS tensors
        """

        U, S, V, _ = self.mat.svd()
        self.mat = S.diag()
        if self.pos > 0 and self.pos < len(self):
            self[self.pos - 1] = ncon.ncon([self[self.pos - 1], U],
                                           [[-1, 1, -3], [1, -2]])
            self[self.pos] = ncon.ncon([V, self[self.pos]],
                                       [[-1, 1], [1, -2, -3]])
        elif self.pos == len(self):
            self[self.pos - 1] = ncon.ncon([self[self.pos - 1], U],
                                           [[-1, 1, -3], [1, -2]])
            self._connector = ncon.ncon([V, self._connector],
                                        [[-1, 1], [1, -2]])
        elif self.pos == 0:
            self._connector = ncon.ncon([self._connector, U],
                                        [[-1, 1], [1, -2]])
            self[self.pos] = ncon.ncon([V, self[self.pos]],
                                       [[-1, 1], [1, -2, -3]])

    def position(self, bond, schmidt_thresh=1E-16, D=None, r_thresh=1E-14, walltime_log=None):
        """
        position(bond,schmidt_thresh=1E-16,D=None,r_thresh=1E-14):
        shifts the center site to "bond"
        Parameters:
        ---------------------------------
        bond: int
              the bond onto which to put the center site
        schmidt_thresh: float
                        truncation threshold at which Schmidt-values are truncated during shifting the position
                        if schmidt_thresh>1E-15, then routine uses an svd to truncate the mps, otherwise no truncation is done
        D: int
           maximum bond dimension after Schmidt-value truncation
           The function does not modify self._connector
        r_thresh: float
                  internal parameter, has no relevance 
        """
        """
        set the values for the schmidt_thresh-threshold, D-threshold and r_thresh-threshold
        r_thresh is used in case that svd throws an exception (happens sometimes in python 2.x)
        in this case, one preconditions the method by first doing a qr, and setting all values in r
        which are smaller than r_thresh to 0, then does an svd.
        """
        if bond == self._position:
            return

        if bond > self._position:
            self[self._position] = ncon.ncon([self.mat, self[self._position]],
                                             [[-1, 1], [1, -2, -3]])
            for n in range(self._position, bond):
                if schmidt_thresh < 1E-15 and D == None:
                    tensor, self.mat, Z = mf.prepare_tensor_QR(
                        self[n], direction=1, walltime_log=walltime_log)
                else:
                    tensor,s,v,Z=mf.prepare_tensor_SVD(self[n],direction=1,D=D,thresh=schmidt_thresh,\
                                                                    r_thresh=r_thresh)
                    self.mat = s.diag().dot(v)

                self._norm *= self.dtype.type(Z)
                self[n] = tensor
                if (n + 1) < bond:
                    self[n + 1] = ncon.ncon([self.mat, self[n + 1]],
                                            [[-1, 1], [1, -2, -3]])
        if bond < self._position:
            self[self._position - 1] = ncon.ncon(
                [self[self._position - 1], self.mat], [[-1, 1, -3], [1, -2]])
            for n in range(self._position - 1, bond - 1, -1):
                if schmidt_thresh < 1E-15 and D == None:
                    self.mat, tensor, Z = mf.prepare_tensor_QR(
                        self[n], direction=-1, walltime_log=walltime_log)
                else:
                    u,s,tensor,Z=mf.prepare_tensor_SVD(self[n],direction=-1,D=D,thresh=schmidt_thresh,\
                                                                 r_thresh=r_thresh)
                    self.mat = u.dot(s.diag())

                self._norm *= self.dtype.type(Z)
                self[n] = tensor
                if n > bond:
                    self[n - 1] = ncon.ncon([self[n - 1], self.mat],
                                            [[-1, 1, -3], [1, -2]])
        self._position = bond

    @staticmethod
    def ortho_deviation(tensor, which):
        return mf.ortho_deviation(tensor, which)

    @staticmethod
    def check_ortho(tensor, which, thresh=1E-8):
        """
        checks if orthogonality condition on tensor is obeyed up to ```thresh```
        """
        return mf.ortho_deviation(tensor, which) < thresh

    def check_form(self, thresh=1E-8):
        """
        check if the MPS is in canonical form, i.e. if all tensors to the left of self.pos are left isometric, and 
        all tensors to the right of self.pos are right isometric

        Parameters:
        thresh:    float
                   threshold for allowed deviation from orthogonality
        Returns:
        ---------------------
        bool
        """
        return np.all([
            self.check_ortho(self[site], 'l', thresh)
            for site in range(self.pos)
        ] + [
            self.check_ortho(self[site], 'r', thresh)
            for site in range(self.pos, len(self))
        ])

    def set_tensor(self, n, tensor):
        raise NotImplementedError()

    def get_left_orthogonal_imps(self,
                                 init=None,
                                 precision=1E-12,
                                 ncv=50,
                                 nmax=1000,
                                 numeig=1,
                                 pinv=1E-30,
                                 warn_thresh=1E-8,
                                 canonize=True,
                                 name=None):
        if canonize:
            self.canonize(
                init=init,
                precision=precision,
                ncv=ncv,
                nmax=nmax,
                numeig=numeig,
                pinv=pinv,
                warn_thresh=warn_thresh)

        return MPS(
            tensors=[self.get_tensor(n) for n in range(len(self))], name=name)

    def get_right_orthogonal_imps(self,
                                  init=None,
                                  precision=1E-12,
                                  ncv=50,
                                  nmax=1000,
                                  numeig=1,
                                  pinv=1E-30,
                                  warn_thresh=1E-8,
                                  canonize=True):
        if canonize:
            self.canonize(
                init=init,
                precision=precision,
                ncv=ncv,
                nmax=nmax,
                numeig=numeig,
                pinv=pinv,
                warn_thresh=warn_thresh)
        self.position(0)
        A = ncon.ncon([self.connector, self.mat, self._tensors[0]],
                      [[-1, 1], [1, 2], [2, -2, -3]])
        tensors = [A] + [self._tensors[n] for n in range(1, len(self))]
        imps = MPS(tensors=tensors)
        imps._position = 0
        return imps

    def apply_2site_gate(self, gate, site, truncation_threshold=1E-16, D=None):
        """
        applies a two-site gate to the mps at site ```site```, 
        and does a truncation with truncation threshold "thresh"
        the center bond is shifted to bond site+1 and mps[site],mps._mat and mps[site+1] are updated


        Parameters:
        --------------------------
        gate:   np.ndarray of shape (dout1,dout2,din1,din2)
                the gate to be applied to the mps.
        site:   int
                the left-hand site of the support of `gate`
        thresh: float  
                truncation threshold; all schmidt values < `thresh` are discarded
        D:      int or None
                the maximally allowed bond dimension
                bond dimension will never be larger than `D`, irrespecitive of `thresh`
        Returns:
        ---------------------
        tw: float
            the truncated weight
        """

        self.position(site + 1)
        
        newState = ncon.ncon(
            [self._tensors[site],self.mat,
             self._tensors[site + 1], gate],
            [[-1, 1, 3], [1,2], [2, -4, 4], [-2, -3, 3, 4]])
        [Dl, d1, d2, Dr] = newState.shape
        newState, merge_data = newState.merge([[0, 1], [2, 3]])
        U, S, V, tw = newState.svd(
            truncation_threshold=truncation_threshold, D=D, full_matrices=False)
        S /= S.norm()
        self[site] = U.split([merge_data[0], [S.shape[0]]]).transpose(0, 2, 1)
        self[site + 1] = V.split([[S.shape[0]], merge_data[1]]).transpose(
            0, 2, 1)
        self.mat = S.diag()
        return tw

    def apply_1site_gate(self, gate, site):
        """
        applies a one-site gate to an mps at site "site"
        the center bond is shifted to bond site+1 
        the _Z norm of the mps is changed
        """
        self.position(site)
        tensor = ncon.ncon([self.mat,self._tensors[site], gate],
                           [[-1,1],[1, -2, 2], [-3, 2]])
        mat,B, Z = mf.prepare_tensor_QR(tensor, -1)
        self._norm *= Z
        self._tensors[site] = B
        self.mat = mat
        return self


class FiniteMPS(MPS):

    @classmethod
    def random(cls,
               D,
               d,
               name=None,
               initializer=ndarray_initializer,
               numpy_func=np.random.random_sample,
               *args,
               **kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D) != len(d) - 1:
            raise ValueError('len(D)!=len(d)-1')
        D = [1] + D + [1]
        return cls(
            tensors=initializer(
                numpy_func=numpy_func,
                shapes=[(D[n], D[n + 1], d[n]) for n in range(len(d))],
                *args,
                **kwargs),
            name=name)

    def __init__(self, tensors=[], name=None, fromview=True):
        if not np.sum(tensors[0].shape[0]) == 1:
            raise ValueError(
                'FiniteMPS got a wrong shape {0} for tensor[0]'.format(
                    tensors[0].shape))
        if not np.sum(tensors[-1].shape[1]) == 1:
            raise ValueError(
                'FiniteMPS got a wrong shape {0} for tensor[-1]'.format(
                    tensors[-1].shape))

        super().__init__(tensors=tensors, name=name, fromview=fromview)
        #self.position(0)
        #self.position(len(self))

    @classmethod
    def zeros(self, *args, **kwargs):
        raise NotImplementedError(
            'FiniteMPS.zeros(*args,**kwargs) not implemented')

    def __add__(self, other):
        """
        adds self with other;
        returns an unnormalized mps
        """
        tensors=[mf.mps_tensor_adder(self.get_tensor(0),other.get_tensor(0),boundary_type='l',ZA=self._norm,ZB=other._norm)]+\
            [mf.mps_tensor_adder(self.get_tensor(n),other.get_tensor(n),boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        return FiniteMPS(tensors=tensors)  #out is an unnormalized MPS

    def __iadd__(self, other):
        """
        adds self with other;
        returns an unnormalized mps
        """
        tensors=[mf.mps_tensor_adder(self.get_tensor(0),other.get_tensor(0),boundary_type='l',ZA=self._norm,ZB=other._norm)]+\
            [mf.mps_tensor_adder(self.get_tensor(n),other.get_tensor(n),boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        self = FiniteMPS(tensors=tensors)  #out is an unnormalized MPS

    def __sub__(self, other):
        """
        subtracts other from self
        returns an unnormalized mps
        """
        tensors=[mf.mps_tensor_adder(self.get_tensor(0),other.get_tensor(0),boundary_type='l',ZA=self._norm,ZB=-other._norm)]+\
            [mf.mps_tensor_adder(self.get_tensor(n),other.get_tensor(n),boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        return FiniteMPS(tensors=tensors)  #out is an unnormalized MPS

    def normalize(self):
        self.mat/=np.sqrt(self.mat.norm())
        # if self.pos == len(self):
        #     self.position(0)
        # elif self.pos == 0:
        #     self.position(len(self))
        # else:
        #     self.position(0)
        #     self.position(len(self))
        # self.position(self.pos)
        self._norm = self.dtype.type(1)
        
    def norm(self):
        return self._norm*self.mat.norm()

    #def norm(self):
    #    return np.sqrt(ncon.ncon([self.centermatrix,self.centermatrix.conj()],[[1,2],[1,2]]))

    def schmidt_spectrum(self, n):
        """
        schmidt_spectrum(n):

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
        U, S, V, _ = self.mat.svd(full_matrices=False)
        return S

    def regauge(self, gauge):
        if gauge in (1, 'l', 'left'):
            self.position(0)
            self.position(len(self))
        if gauge in (-1, 'r', 'right'):
            self.position(len(self))
            self.position(0)

    def canonize(self, name=None):
        """
        """
        Lambdas, Gammas = [], []

        self.position(len(self))
        self.position(0)
        Lambdas.append(self.mat.diag())
        for n in range(len(self)):
            self.position(n + 1)
            self.diagonalize_center_matrix()
            Gammas.append(
                ncon.ncon([(1.0 / Lambdas[-1]).diag(), self[n]],
                          [[-1, 1], [1, -2, -3]]))
            Lambdas.append(self.mat.diag())
        return CanonizedFiniteMPS(gammas=Gammas, lambdas=Lambdas, name=name)

    def truncate(self, schmidt_thresh=1E-16, D=None, presweep=True):
        """ 
        truncates the mps
        Parameters
        ---------------------------------
        schmidt_thresh: float 
                        truncation threshold
        D:              int 
                        maximum bond dimension; if None, the bond dimension is adapted to match schmidt_thresh
        presweep:       bool
                        if True, the routine sweeps the center site through the MPS once prior to truncation (extra cost)
                        this ensures that the  truncation is done in the optimal basis; use False only if the 
                        state is in standard form
        """

        if D and D > max(self.D):
            return self
        else:
            pos = self.pos
            if self.pos == 0:
                self.position(len(self), schmidt_thresh=schmidt_thresh, D=D)
            elif self.pos == len(self):
                self.position(0, schmidt_thresh=schmidt_thresh, D=D)
            else:
                if presweep:
                    self.position(0)
                    self.position(self.num_sites)
                self.position(0, schmidt_thresh=schmidt_thresh, D=D)
                self.position(self.pos)
            return self

    def apply_MPO(self, mpo):
        """
        applies an mpo to an mps; no truncation is done
        """
        assert (len(mpo) == len(self))
        tensors = [
            ncon.ncon(
                [self.get_tensor(n), mpo.get_tensor(n)],
                [[-1, -3, 1], [-2, -4, -5, 1]]).merge([[0, 1], [2, 3], [4]])[0]
            for n in range(len(self))
        ]
        return self.from_tensors(tensors)

    def dot(self, mps):
        """
        calculate the overlap of self with mps 
        mps: MPS
        returns: float
                 overlap of self with mps
        """
        if not len(self) == len(mps):
            raise ValueError('FiniteMPS.dot(other): len(other)!=len(self)')
        # if not isinstance(mps,FiniteMPS):
        #     raise TypeError('can only calculate overlaps with FiniteMPS')

        O = self[0].eye(0)
        for n in range(len(self)):
            O = mf.transfer_operator([self.get_tensor(n)], [mps.get_tensor(n)],
                                     'l', O)
        return O


class CanonizedMPS(MPSBase):
    """
    """

    class GammaTensors(object):
        """
        helper class to implement calls such as camps.Gamma[site]
        camps.Gamma[site]=Tensor
        GammaTensors holds a view of the tensors in CanonizedMPS
        
        """

        def __init__(self, tensors):
            self._data = tensors

        def __getitem__(self, site):
            N = np.prod(self._data.shape)
            if site >= N:
                raise IndexError(
                    'CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'
                    .format(site, N))
            return self._data[int(2 * site + 1)]

        def __setitem__(self, site, val):
            N = np.prod(self._data.shape)
            if site >= N:
                raise IndexError(
                    'CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'
                    .format(site, N))
            self._data[int(2 * site + 1)] = val

        def __len__(self):
            return int((len(self._data) - 1) / 2)

        def __iter__(self):
            return (self[n] for n in range(len(self)))

    class LambdaTensors(object):
        """
        helper class to implement calls such as camps.Lambda[site]
        camps.Lambda[site]=Tensor
        LambdaTensors holds a view of the tensors in CanonizedMPS
        """

        def __init__(self, tensors):
            self._data = tensors

        def __getitem__(self, site):
            N = np.prod(self._data.shape)
            if site >= N:
                raise IndexError(
                    'CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'
                    .format(site, N))
            return self._data[int(2 * site)]

        def __setitem__(self, site, val):
            N = np.prod(self._data.shape)
            if site >= N:
                raise IndexError(
                    'CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'
                    .format(site, N))
            self._data[int(2 * site)] = val

        def __len__(self):
            return int((len(self._data) - 1) / 2 + 1)

        def __iter__(self):
            return (self[n] for n in range(len(self)))

    def __init__(self, gammas=[], lambdas=[], name=None, fromview=True):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """

        assert ((len(gammas) + 1) == len(lambdas))
        tensors = [lambdas[0]]
        for n in range(len(gammas)):
            tensors.append(gammas[n])
            tensors.append(lambdas[n + 1])
        super().__init__(tensors=tensors, name=name, fromview=fromview)
        self.Gamma = self.GammaTensors(self._tensors.view())
        self.Lambda = self.LambdaTensors(self._tensors.view())

    @classmethod
    def from_tensors(cls, tensors, name=None):
        """
        construct a CanonizedMPS from a list containing Gamma and Lambda matrices
        Parameters:
        ------------
        tensors:   list of Tensors
                   is a list of Tensor objects which alternates between ```Lambda`` and ```Gamma``` tensors,
                   starting and stopping with a boundary ```Lambda```
                   ```Lambda``` are in vector-format (i.e. it is the diagonal), ```Gamma``` is a full matrix.
        name:      str or None
                   name of the CanonizedMPS
        Returns:
        CanonizeMPS or subclass 
        
        """
        if not np.sum(tensors[0].shape[0]) == 1:
            raise ValueError(
                'CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[0]'
                .format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[0]) == 1:
            raise ValueError(
                'CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[-1]'
                .format(tensors[-1].shape))
        assert (len(tensors) % 2)
        return cls(gammas=tensors[1::2], lambdas=tensors[0::2], name=name)

    def __len__(self):
        """
        return the length of the CanonizedMPS, i.e. the number of physical sites

        Parameters: None
        --------------
        Returns:
        int
        """
        return len(self.Gamma)

    @property
    def num_sites(self):
        return len(self.Gamma)

    def get_tensor(self, n, mult='l'):
        if mult in (1, 'l', 'left'):
            out = ncon.ncon([self.Lambda[n].diag(), self.Gamma[n]],
                            [[-1, 1], [1, -2, -3]])
            if n == (self.num_sites - 1):
                out = ncon.ncon([out, self.Lambda[n + 1].diag()],
                                [[-1, 1, -3], [1, -2]])
        elif mult in (-1, 'r', 'right'):
            out = ncon.ncon([self.Gamma[n], self.Lambda[n + 1].diag()],
                            [[-1, 1, -3], [1, -2]])
            if n == 0:
                out = ncon.ncon([self.Lambda[n].diag(), out],
                                [[-1, 1], [1, -2, -3]])

        return out

    def get_env_left(self, site):
        """
        obtain the left environment of ```site```
        """
        if site >= len(self) or site < 0:
            raise IndexError(
                'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'
                .format(site, len(self)))

        return self.Lambda[site].eye(0)

    def get_env_right(self, site):
        """
        obtain the right environment of ```site```
        """

        if site >= len(self) or site < 0:
            raise IndexError(
                'index {0} out of bounds for MPSUnitCellCentralGauge of length {1}'
                .format(site, len(self)))

        return self.Lambda[site + 1].diag()

    def __iter__(self):
        """
        iterates through the mps tensors
        lambda is absorbed from the left into the Gammas, result is returned
        """
        return (self.get_tensor(n) for n in range(self.num_sites))

    @property
    def D(self):
        """
        return a list of bond dimensions for each bond
        """
        return [l.shape[0] for l in self.Lambda]

    @property
    def d(self):
        """
        return a list of physicsl dimensions for each site
        """
        return [g.shape[2] for g in self.Gamma]

    @classmethod
    def random(*args, **kwargs):
        raise NotImplementedError(
            'CanonizedMPS.random() only implemented in subclasses')

    @classmethod
    def zeros(*args, **kwargs):
        raise NotImplementedError(
            'CanonizedMPS.zeros() only implemented in subclasses')

    @classmethod
    def ones(*args, **kwargs):
        raise NotImplementedError(
            'CanonizedMPS.ones() only implemented in subclasses')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self._norm is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """
        if ufunc == np.true_divide:
            #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
            #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
            return self.__idiv__(inputs[1])

        out = kwargs.get('out', ())
        for arg in inputs + out:
            if not isinstance(arg,
                              self._HANDLED_UFUNC_TYPES + (TensorNetwork,)):
                return NotImplemented
        if out:
            #takes care of in-place operations
            result = []
            for n, x in np.ndenumerate(self._tensors):
                kwargs['out'] = tuple(
                    o[n] if isinstance(o, type(self)) else o for o in out)
                ipts = [
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ]
                result.append(getattr(ufunc, method)(*ipts, **kwargs))
        else:
            result = [
                getattr(ufunc, method)(*[
                    ipt[n] if isinstance(ipt, type(self)) else ipt
                    for ipt in inputs
                ], **kwargs)
                for n, x in np.ndenumerate(self._tensors)
            ]
        if method == 'reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis = kwargs.get('axis', ())
            if axis == None:
                return getattr(ufunc, method)(result)
            else:
                raise NotImplementedError(
                    'CanonizedFiniteMPS.__array_ufunc__ with argument axis!=None not implemented'
                )

        else:
            return self.from_tensors(tensors=result, name=None)


class CanonizedFiniteMPS(CanonizedMPS):

    def __init__(self, gammas=[], lambdas=[], name=None, fromview=True):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """
        if not np.sum(gammas[0].shape[0]) == 1:
            raise ValueError(
                'CanonizedFiniteMPS got a wrong shape {0} for gammas[0]'.format(
                    gammas[0].shape))
        if not np.sum(gammas[-1].shape[1]) == 1:
            raise ValueError(
                'CanonizedFiniteMPS got a wrong shape {0} for gammas[-1]'.
                format(gammas[-1].shape))

        super().__init__(
            gammas=gammas, lambdas=lambdas, name=name, fromview=fromview)

    @classmethod
    def random(cls,
               D=[1, 2, 1],
               d=[2, 2],
               name=None,
               initializer=ndarray_initializer,
               *args,
               **kwargs):
        mps = FiniteMPS.random(
            D=D, d=d, name=name, initialize=initializer, *args, **kwargs)
        return mps.canonize()

    @classmethod
    def zeros(*args, **kwargs):
        raise NotImplementedError

    @classmethod
    def ones(*args, **kwargs):
        raise NotImplementedError

    @classmethod
    def empty(*args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """
        inds = range(len(self))
        s1=['Name: ',str(self.name),'\n\n ']+\
            ['Lambda[0] of shape '+str(self.Lambda[0].shape)+'\n\n '+self.Lambda[0].__str__()+' \n\n ']+\
            ['Gamma['+str(ind)+'] of shape '+str(self.Gamma[ind].shape)+'\n\n '+self.Gamma[ind].__str__()+' \n\n '+'Lambda['+str(ind+1)+'] of shape '+str(self.Lambda[ind+1].shape)+'\n\n '+self.Lambda[ind+1].__str__()+' \n\n ' for ind in range(len(self))]

        return ''.join(s1)

    def toMPS(self, name=None):
        """
        cast the CanonizedFiniteMPS to a FiniteMPS
        Returns:
        ----------------
        FiniteMPS
        """
        tensors = [
            ncon.ncon([self.Lambda[n].diag(), self.Gamma[n]],
                      [[-1, 1], [1, -2, -3]]) for n in range(len(self))
        ]
        return FiniteMPS(tensors=tensors, name=name)

    def canonize(self, name=None):
        """
        re-canonize the CanonizedFiniteMPS
        Returns:
        ---------------
        CanonizedFiniteMPS
        """
        return self.toMPS().canonize(name=self.name)

    def iscanonized(self, thresh=1E-10):
        left = [self.ortho(n, 'l') < thresh for n in range(len(self))]
        right = [self.ortho(n, 'r') < thresh for n in range(len(self))]
        if np.all(left) and np.all(right):
            return True
        if not np.all(right):
            right = [self.ortho(n, 'r') >= thresh for n in range(len(self))]
            print('{1} is not right canonized at site(s) {0} within {2}'.format(
                np.nonzero(right)[0][:], type(self), thresh))

        if not np.all(left):
            left = [self.ortho(n, 'l') >= thresh for n in range(len(self))]
            print('{1} is not left canonized at site(s) {0} within {2}'.format(
                np.nonzero(left)[0][:], type(self), thresh))
        return False
