import numpy as np


class TensorBase(object):
    @classmethod
    def random(cls,*args,**kwargs):
        return cls._initializer(np.random.random_sample,*args,**kwargs)
    @classmethod        
    def zeros(cls,*args,**kwargs):
        return cls._initializer(np.zeros,*args,**kwargs)
    @classmethod        
    def ones(cls,*args,**kwargs):
        return cls._initializer(np.ones,*args,**kwargs)
    @classmethod        
    def empty(cls,*args,**kwargs):
        return cls._initializer(np.empty,*args,**kwargs)

    @classmethod
    def eye(cls,*args,**kwargs):
        return cls.eye(*args,**kwargs)
    
    @classmethod
    def diag(cls,*args,**kwargs):
        return cls.diag(*args,**kwargs)
    
    @staticmethod
    def directSum(*args,**kwargs):
        pass
    @staticmethod
    def concatenate(arrs,*args,**kwargs):
        print('wll hello')
        pass
    
class Tensor(np.ndarray,TensorBase):
    
    def __new__(cls,*args,**kwargs):
        """
        note: np.ndarray has no __init__, only __new__
        """
        return super(Tensor,cls).__new__(cls,*args,**kwargs)
    
    def merge(self,indices):
        """
        merges indices in the list ```indices```, respecting the order of the elements of ```indices```,
        array is transposed such that: all indices smaller or equal to the smalles element in ```indices``` remain at their positions;
        these indices are followed by all elements of ```indices``, followed by all indices larger than ```min(indices)``` and not contained in ```indices```.
        Parameters:
        ----------------------
        indices:   list of int
                   the indices which should be merged
        Returns:
        ----------------------
        Tensor:    the tensor with merged indices
        """
        if max(indices)>=len(self.shape):
            raise ValueError("Tensor.merge(indices): max(indices)>=len(shape)")
        left=list(range(0,min(indices)))
        complement=sorted(list(set(range(len(self.shape))).difference(set(left+list(indices)))))
        neworder=left+list(indices)+complement
        newshape=tuple([self.shape[l] for l in left]+[np.prod([self.shape[l] for l in indices])]+[self.shape[l] for l in complement])
        return np.reshape(np.transpose(self,neworder),newshape).view(Tensor)
    
    @classmethod
    def _initializer(cls,numpy_func,*args,**kwargs):
        """
        initializer function for Tensor
        """
        return numpy_func(*args,**kwargs).view(Tensor)
    
    def eye(self,index,*args,**kwargs):
        """
        returns an identity matrix of shape matching with ```self.shape[index]```
        """
        if index>=len(self.shape):
            raise IndexError("index out of range")
        return np.eye(self.shape[index],*args,**kwargs).view(type(self))

    
    def diag(self,*args,**kwargs):
        """
        wrapper for np.diag
        """
        return np.diag(*args,**kwargs).view(type(self))
        
    def svd(self,*args,**kwargs):
        try:        
            return np.linalg.svd(self,*args,**kwargs)
        except LinAlgError:
            [q,r]=temp.qr()
            r[np.abs(r)<r_thresh]=0.0
            u_,s,v=r.svd(*args,**kwargs)
            u=q.dot(u_).view(Tensor)
            warnings.warn('svd: prepareTruncate caught a LinAlgError with dir>0')
            return u,s,v
    def qr(self,**kwargs):
        return np.linalg.qr(self,**kwargs)
    
    @staticmethod
    def directSum(A,B):
        if len(A.shape)!=2:
            raise ValueError('A.shape!=2: directSum can only operate on matrices')
        if len(B.shape)!=2:
            raise ValueError('B.shape!=2: directSum can only operate on matrices')
        dtype=np.result_type(A.dtype,B.dtype)
        out=A.zeros((A.shape[0]+B.shape[0],A.shape[1]+B.shape[1]),dtype=dtype)
        out[0:A.shape[0],0:A.shape[1]]=A
        out[A.shape[0]:A.shape[0]+B.shape[0],A.shape[1]:A.shape[1]+B.shape[1]]=B
        return out

    
    @staticmethod
    def concatenate(arrs,axis=0,out=None):
        dtype=np.result_type(*[a.dtype for a in arrs])
        return np.concatenate(arrs,axis,out)

    def reduce(self):
        print('calling reduce')
    # def __array_ufunc__(self,ufunc,method,*args,**kwargs):
    #     if method=='reduce':
    #         ufunc.reduce()
    #     else:
    #         getattr(ufunc,method)(*args,**kwargs)


        
