import numpy as np
import warnings
import lib.ncon as ncon
from numpy.linalg.linalg import LinAlgError


#TODO:   merge returns data necessary to unmerge
#        prepareTensor has to be removed as a staticmethod from Tensor; logically does not nelong to it
#        split: takes data from merge and un-merges the indices
#        SparseTensor needs to have __len__ implemented
#        truncation is done in svd, this has to implemented in SparseTensor as well
class TensorBase:

    @classmethod
    def random(cls, *args, **kwargs):
        return cls._initializer(np.random.random_sample, *args, **kwargs)

    @classmethod
    def zeros(cls, *args, **kwargs):
        return cls._initializer(np.zeros, *args, **kwargs)

    @classmethod
    def ones(cls, *args, **kwargs):
        return cls._initializer(np.ones, *args, **kwargs)

    @classmethod
    def empty(cls, *args, **kwargs):
        return cls._initializer(np.empty, *args, **kwargs)

    @classmethod
    def eye(cls, *args, **kwargs):
        return cls.eye(*args, **kwargs)

    @classmethod
    def diag(cls, *args, **kwargs):
        return cls.diag(*args, **kwargs)

    @staticmethod
    def directSum(*args, **kwargs):
        raise NotImplementedError('TensorBase.directSum: not implemented')

    @staticmethod
    def concatenate(arrs, *args, **kwargs):
        raise NotImplementedError('TensorBase.concatenate: not implemented')

    def squeeze(*args, **kwargs):
        raise NotImplementedError('TensorBase.squeeze: not implemented')

    def to_dense(self):
        raise NotImplementedError('TensorBase.to_dense: not implemented')

    @staticmethod
    def from_dense(self):
        raise NotImplementedError('TensorBase.from_dense: not implemented')

    def herm(self):
        raise NotImplementedError()

    def trace(self):
        raise NotImplementedError()

    def abs(self):
        raise NotImplementedError()


class Tensor(np.ndarray, TensorBase):

    def __new__(cls, *args, **kwargs):
        """
        note: np.ndarray has no __init__, only __new__
        """
        return super(Tensor, cls).__new__(cls, *args, **kwargs)

    def merge(self, indices):
        """
        merge has to return the data neccessary to unmerge indices again. In the generic case, merging is irreversible
        if no additional information is given.

        merges indices in `indices`, respecting the order of the elements of `indices`.
        For each list in `indices`, the indices in this list are merged into a single new index.
        The order of these new merged indices is given by the order in which the lists are passed.
        All indices of self which are not in any of the lists in `indices` are transposed and placed in the following way:
        let `comp` be an ordered list (small to large) of the complementary indices to `indices`, i.e. `comp` contains all 
        indices not contained in `indices`,i.e. `comb=[c1,c2,c3,...]`. self is then first transposed into an index-order
        [sorted(all elements of `comp` which are smaller than min(indices[0]))],indices[0],[sorted(all elements of `comp` 
        wich are smaller than min(indices[1]) and larger than min(indices[0])],indices[1],...,indices[-1],sorted([all elements of `comp` which have not yet been placed])]

        Parameters:
        ----------------------
        indices:   tuple of list of int
                   the indices which should be merged
        Returns:
        ----------------------
        (Tensor, data):    
        Tensor:            the tensor with merged indices
        data:              information neccessary to undo the merge (the old shape)
        
        """
        t = list(indices)
        for n in range(len(indices)):
            if isinstance(indices[n], int):
                t[n] = [indices[n]]
            else:
                t[n] = indices[n]
        indices = tuple(t)
        flatinds = list(indices[0])
        shape = np.array(self.shape)
        [flatinds.extend(n) for n in indices[1:]]
        if len(set(flatinds)) < len(flatinds):
            elems, cnts = np.unique(np.asarray(flatinds), return_counts=True)
            raise ValueError(
                'Tensor.merge(): indices {0} are appearing more than once!'.
                format(elems[cnts > 1]))

        if max(flatinds) >= len(self.shape):
            raise ValueError("Tensor.merge(indices): max(indices)>=len(shape)")
        complement = sorted(
            list(set(range(len(self.shape))).difference(set(flatinds))))
        complist = [list() for n in range(len(indices) + 1)]
        for c in complement:
            n = 0
            while (n < len(indices)) and (c > min(indices[n])):
                n += 1
            complist[n].append(c)
        neworder = complist[0]
        newshape = list(shape[complist[0]])
        oldshape = list(shape[complist[0]])

        for n in range(len(indices)):
            neworder.extend(indices[n])
            neworder.extend(complist[n + 1])
            newshape.append(np.prod(shape[indices[n]]))
            newshape.extend(list(shape[complist[n + 1]]))
            oldshape.append(list(shape[indices[n]]))
            oldshape.extend(list(shape[complist[n + 1]]))
        t = list(oldshape)

        for n in range(len(oldshape)):
            if isinstance(oldshape[n], list):
                t[n] = oldshape[n]
            else:
                t[n] = [oldshape[n]]

        oldshape = list(t)
        return np.reshape(np.transpose(self, neworder),
                          newshape).view(Tensor), oldshape

    def split(self, merge_data):
        """
        splits the tensor indices into its constituent parts, using
        the information provided in `merge_data`.
        len(merge_data) = len(self.shape), i.e. each index of the tensor is represented in `merge_data`
        """
        if len(merge_data) != len(self.shape):
            raise ValueError(
                'length of merge_data is not compatible with the shape of tensor'
            )

        newshape = merge_data[0]
        [newshape.extend(m) for m in merge_data[1:]]
        return np.reshape(self, newshape)

    @staticmethod
    def _initializer(numpy_func, *args, **kwargs):
        """
        initializer function for Tensor
        """
        return numpy_func(*args, **kwargs).view(Tensor)

    def eye(self, rank_index, *args, **kwargs):
        """
        returns an identity matrix of shape matching with `self.shape[index]`
        """
        if rank_index >= len(self.shape):
            raise IndexError("rank_index out of range")

        return np.eye(self.shape[rank_index], *args,
                      **kwargs).astype(self.dtype.type).view(Tensor)

    def diag(self, **kwargs):
        """
        wrapper for np.diag
        returns either a diagonal of self, or constructs a matrix from self, if self is a vector
        """
        return np.diag(self, **kwargs).view(type(self))

    def svd(self,
            truncation_threshold=1E-16,
            D=None,
            r_thresh=1E-14,
            *args,
            **kwargs):
        tw = 0.0
        try:
            u, s, v = np.linalg.svd(self, *args, **kwargs)
            Z = np.linalg.norm(s)
            s /= Z
            if truncation_threshold > 1E-16:
                mask = s > truncation_threshold
                tw = np.sum(s[~mask]**2)
                s = s[mask]
            if D:
                if D < len(s):
                    warnings.warn(
                        'Tensors.svd: desired thresh imcompatible with max bond dimension; truncating',
                        stacklevel=3)
                tw += np.sum(s[min(D, len(s))::])
                s = s[0:min(D, len(s))]
            if len(s)==0:
                s=np.array([1.0],dtype=s.dtype)
            u = u[:, 0:len(s)]
            v = v[0:len(s), :]
            s *= Z
            return u, s.view(type(self)), v, tw
        except LinAlgError:
            [q, r] = self.qr()
            r[np.abs(r) < r_thresh] = 0.0
            u_, s, v = r.svd(*args, **kwargs)
            s /= Z
            u = q.dot(u_).view(Tensor)
            warnings.warn(
                'svd: prepareTruncate caught a LinAlgError with dir>0')
            if truncation_threshold > 1E-16:
                mask = s > truncation_threshold
                tw = np.sum(s[~mask]**2)
                s = s[mask]

            if D != None:
                if D < len(s):
                    warnings.warn(
                        'Tensors.svd: desired thresh imcompatible with max bond dimension; truncating',
                        stacklevel=3)
                tw += np.sum(s[min(D, len(s))::])
                s = s[0:min(D, len(s))]

            if len(s)==0:
                s=np.array([1.0],dtype=s.dtype)
                
            u = u[:, 0:len(s)]
            v = v[0:len(s), :]

            s *= Z
            return u.view(type(self)), s.view(type(self)), v.view(
                type(self)), tw

    def qr(self, **kwargs):
        return np.linalg.qr(self, **kwargs)

    def eigh(self, **kwargs):
        eta, u = np.linalg.eigh(self, **kwargs)
        return eta.view(type(self)), u.view(type(self))

    def truncate(self, thresh):
        """
        truncate a one-dimension array
        """

        if len(self.shape) != 1:
            raise ValueError('Tensors.truncate works only on rank 1 tensors')
        return self[self >= thresh]

    def norm(self, **kwargs):
        """
        the norm of the state
        """
        return np.linalg.norm(self, **kwargs)

    def prune(self, newshape):
        """
        truncate a one-dimensional array
        Parameters:
        -----------------------
        newshape: tuple or list
                  the new shape of the array
                  if newshape[n]=None, the dimension `n` is not truncated
        """
        if not len(newshape) == len(self.shape):
            raise ValueError(
                'Tensor.truncate(newshape): newshape has different rank than self'
            )
        shape = [
            slice(0, newshape[n], 1) if newshape[n] != None else slice(
                0, self.shape[n], 1) for n in range(len(newshape))
        ]
        return self[shape]

    @staticmethod
    def directSum(A, B):
        if len(A.shape) != 2:
            raise ValueError(
                'A.shape!=2: directSum can only operate on matrices')
        if len(B.shape) != 2:
            raise ValueError(
                'B.shape!=2: directSum can only operate on matrices')
        dtype = np.result_type(A.dtype, B.dtype)
        out = A.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]),
                      dtype=dtype)
        out[0:A.shape[0], 0:A.shape[1]] = A
        out[A.shape[0]:A.shape[0] + B.shape[0], A.shape[1]:A.shape[1] +
            B.shape[1]] = B
        return out

    @staticmethod
    def concatenate(arrs, axis=0, out=None):
        dtype = np.result_type(*[a.dtype for a in arrs])
        return np.concatenate(arrs, axis, out)

    def herm(self):
        return np.conj(self.T)

    def inv(self):
        return np.linalg.inv(self)

    def to_dense(self):
        return self.reshape(np.prod(self.shape))

    @staticmethod
    def from_dense(dense, shape):
        return dense.reshape(shape).view(Tensor)

    def herm(self):
        return self.conj().T

    def tr(self, **kwargs):
        return np.trace(self, **kwargs)

    def abs(self):
        return np.abs(self)
