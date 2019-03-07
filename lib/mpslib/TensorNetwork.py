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
import lib.ncon as ncon
from lib.mpslib.Container import Container

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
from lib.mpslib.Tensor import TensorBase, Tensor



def generate_unary_deferer(op_func):
    def deferer(cls, *args, **kwargs):
        try:
            return type(cls).__unary_operations__(cls, op_func, *args,**kwargs)
        except AttributeError:
            raise(AttributeError("cannot generate unary deferer for class withtou __unary_operations__"))
    return deferer


def ndarray_initializer(numpy_func,shapes,*args,**kwargs):
    """
    initializer to create a list of tensors of type Tensor
    Parameters:
    ---------------------
    numpy_func:       callable
                      a numpy function like np.random.random_sample
    shapes:           list of tuple
                      the shapes of the individual tensors
    *args,**kwargs:   if numpy_func is not one of np.random functions, these are passed on to numpy_func

    Returns:
    -------------------------
    list of Tensor objects of shape ```shapes```, initialized with numpy_func
    """

    minval=kwargs.get('mean',-0.5)
    maxval=kwargs.get('std',0.5)

    mean=kwargs.get('mean',0.0)
    std=kwargs.get('std',0.1)
    dtype=kwargs.get('dtype',np.float64)
    if numpy_func in (np.random.random_sample,np.random.rand):
        if np.issubdtype(dtype,np.complexfloating):
            return [((maxval-minval)*numpy_func(shape).view(Tensor)+minval+1j*((maxval-minval)*numpy_func(shape).view(Tensor)+minval)).astype(dtype) for shape in shapes]

        elif np.issubdtype(dtype,np.floating):
           return [((maxval-minval)*numpy_func(shape).view(Tensor)+minval).astype(dtype)*std for shape in shapes]
       
    elif numpy_func ==np.random.randn:
        if np.issubdtype(dtype,np.complexfloating):
            return [(std*numpy_func(*shape).view(Tensor)+mean+1j*std*(numpy_func(*shape).view(Tensor)+1j*mean)).astype(dtype) for shape in shapes]

        elif np.issubdtype(dtype,np.floating):
           return [(std*numpy_func(*shape).view(Tensor)+mean).astype(dtype) for shape in shapes]
       
    else:
        if np.issubdtype(dtype,np.complexfloating):
            return [numpy_func(shape,*args,**kwargs).view(Tensor)+1j*numpy_func(shape,*args,**kwargs).view(Tensor) for shape in shapes]
        elif np.issubdtype(dtype,np.floating):            
            return [numpy_func(shape,*args,**kwargs).view(Tensor) for shape in shapes]


class TensorNetwork(Container,np.lib.mixins.NDArrayOperatorsMixin):
    _HANDLED_UFUNC_TYPES=(numbers.Number,np.ndarray)
    def __init__(self,tensors=[],shape=(),name=None,fromview=True):
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
        super(TensorNetwork,self).__init__(name) #initialize the Container base class
        if isinstance(tensors,np.ndarray):
            if not fromview:
                self._tensors=copy.deepcopy(tensors)
            else:
                self._tensors=tensors.view()
            self.Z=np.result_type(*tensors.ravel()).type(1.0)
            self.tensortype=type(tensors.ravel()[0])
        elif isinstance(tensors,list):
            N=len(tensors)
            if shape!=():
                if not (np.prod(shape)==N):
                    raise ValueError('shape={0} incompatible with len(tensors)={1}'.format(shape,N))
                if not isinstance(shape,tuple):
                    raise TypeError('TensorNetwor.__init__(): got wrong type for shape; only tuples are allowed')
                _shape=shape
            else:
                _shape=tuple([N])
            if tensors!=[]:
                self.tensortype=type(tensors[0])
            else:
                self.tensortype=object
                
            self._tensors=np.empty(_shape,dtype=self.tensortype)
            for n in range(len(tensors)):
                self._tensors.flat[n]=tensors[n].view(Tensor)
            self.Z=np.result_type(*self._tensors).type(1.0)
        else:
            raise TypeError('TensorNetwork.__init__(tensors): tensors has invlaid tupe {0}'.type(tensors))

    
    def __in_place_unary_operations__(self,operation,*args,**kwargs):
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
        for n ,x in np.ndenumerate(self):
            self[n]=operation(self[n],*args,**kwargs)


    def __unary_operations__(self,operation,*args,**kwargs):
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
        obj=self.copy()
        obj.__in_place_unary_operations__(operation,*args,**kwargs)
        return obj

    
    def __array__(self):
        return self._tensors
    
    def reshape(self,newshape,order='C'):
        """
        returns a reshaped view of self
        compatible with np.reshape
        """
        view=self.view()
        view._tensors=np.reshape(view._tensors,newshape,order=order)
        return view
    
    @property
    def N(self):
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
        return np.result_type(*[t.dtype for t in self._tensors],self.Z)
    
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
    def random(cls,shape=(),tensorshapes=(),name=None,initializer=ndarray_initializer,*args,**kwargs):
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
        return cls(tensors=initializer(np.random.random_sample,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape)

        
    @classmethod
    def zeros(cls,shape=(),tensorshapes=(),name=None,initializer=ndarray_initializer,*args,**kwargs):
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
        return cls(tensors=initializer(np.zeros,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape)

    @classmethod
    def ones(cls,shape=(),tensorshapes=(),name=None,initializer=ndarray_initializer,*args,**kwargs):
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
        return cls(tensors=initializer(np.ones,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape)
    
    @classmethod
    def empty(cls,shape=(),tensorshapes=(),name=None,initializer=ndarray_initializer,*args,**kwargs):
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
        return cls(tensors=initializer(np.empty,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape)
        

    def __getitem__(self,n,**kwargs):
        return self._tensors[n]

    def __setitem__(self,n,tensor,**kwargs):
        self._tensors[n]=tensor

    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """
        
        inds=np.unravel_index(range(len(self)),dims=self.shape)
        inds=list(zip(*inds))
        return ''.join(['Name: ',str(self.name),'\n\n ']+['TN'+str(index)+' \n\n '+self[index].__str__()+' \n\n ' for index in inds]+['\n\n Z=',str(self.Z)])

    def __len__(self):
        """
        return the total number of tensors in the TensorNetwork
        Returns:
        ---------------
        int:    the total number of tensors in TensorNetwork
        """
        return np.prod(self.shape)

    def __iter__(self):
        """
        Returns:
        iterator:  an iterator over the tensors of the TensorNetwork
        """
        return iter(self._tensors)
    
    
    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self.Z is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """
        #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
        #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
        if ufunc==np.true_divide:
            return self.__idiv__(inputs[1])


        out=kwargs.get('out',())
        for arg in inputs+out:
            if not isinstance(arg,self._HANDLED_UFUNC_TYPES+(TensorNetwork,)):
                return NotImplemented
        if out:
            #takes care of in-place operations
            result=[]
            for n,x in np.ndenumerate(self._tensors):
                kwargs['out']=tuple(o[n] if isinstance(o,type(self)) else o for o in out)
                ipts=[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs]
                result.append(getattr(ufunc,method)(*ipts,**kwargs))
        else:
            result=[getattr(ufunc,method)(*[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs],**kwargs)
                    for n,x in np.ndenumerate(self._tensors)]

        if method=='reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis=kwargs.get('axis',())
            if axis==None:
                return getattr(ufunc,method)(result)
            else:
                raise NotImplementedError('TensorNetwork.__array_ufunc__ with argument axis!=None not implemented')

        else:
            return TensorNetwork(tensors=result,shape=self.shape,name=None)            
            


    
    def __mul__(self,num):
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
            raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        new=self.copy()
        new.Z*=num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self.Z*num)

    def __imul__(self,num):
        """
        left-multiplies "num" with TensorNetwork, i.e. returns TensorNetwork*num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        self.Z*=num
        return self
    

    def __idiv__(self,num):
        """
        left-divides "num" with TensorNetwork, i.e. returns TensorNetwork*num;
        note that "1./num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        self.Z/=num
        return self
    
    def __truediv__(self,num):
        """
        left-divides "num" with TensorNetwork, i.e. returns TensorNetwork/num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        new=self.copy()
        new.Z/=num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self.Z/num)

     
    def __rmul__(self,num):
        """
        right-multiplies "num" with TensorNetwork, i.e. returns num*TensorNetwork;
        WARNING: if you are using numpy number types, i.e. np.float, np.int, ..., 
        the right multiplication of num with TensorNetwork, i.e. num*TensorNetwork, returns 
        an np.darray instead of an TensorNetwork. 
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state

        """
        if not np.isscalar(num):
            raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        new=self.copy()
        new.Z*=num
        return new
        #return TensorNetwork(tensors=copy.deepcopy(self._tensors),shape=self.shape,name=None,Z=self.Z*num)                


    def __add__(self,other):
        return NotImplemented

    def __iadd__(self,other):
        return NotImplemented        
    
    def __sub__(self,other):
        return NotImplemented        

    def contract(self,other):
        return NotImplemented
    
    @property
    def real(self):
        return self.__array_ufunc__(np.real,'__call__',self)
    @property    
    def imag(self):
        return self.__array_ufunc__(np.imag,'__call__',self)        
    
        
class MPS(TensorNetwork):

    @staticmethod
    def prepareTruncate(tensor,direction,D=None,thresh=1E-32,r_thresh=1E-14):
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
    
    @staticmethod
    def prepareTensor(tensor,direction):
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


    
    def __init__(self,tensors=[],Dmax=None,name=None,fromview=False):
        """
        no checks are performed to see wheter the provided tensors can be contracted
        """
        super(MPS,self).__init__(tensors=tensors,shape=(),name=name,fromview=fromview)
        if not Dmax:
            self._D=max([tensors[0].shape[0]]+[t.shape[1] for t in tensors])
        else:
            self._D=Dmax
        self.mat=tensors[-1].eye(1)
        self.mat=self.mat/np.sqrt(np.trace(self.mat.dot(self.mat.herm())))
        self.connector=self.mat.inv()
        self._position=self.N


    def normalize(self):
        pos=self.pos
        if self.pos==len(self):
            self.position(0)
        elif self.pos==0:
            self.position(len(self))
        else:
            self.position(0)
            self.position(len(self))
        self.position(self.pos)                        
        self.Z=self.dtype.type(1)
 

    def transfer_op(self,site,direction,x):
        """
        """
        A = self.get_tensor(site)
        return mf.TransferOperator(A,A,direction=direction, x=x)

    def unitcell_transfer_op(self,direction,x):
        """
        FIXME: absorbing self.connector and self.mat each time
               is suboptimal when using unitcell_transfer_op within a sparse solver
        """
        
        if direction in ('l','left',1):
            if not x.shape[0]==self.D[0]:
                raise ValueError('shape of x[0] does not match the shape of mps.D[0]')
            if not x.shape[1]==self.D[0]:
                raise ValueError('shape of x[1] does not match the shape of mps.D[0]')
                
            l=x
            for n in range(len(self)):
                l=self.transfer_op(n,direction='l',x=l)
            return l
        if direction in ('r','right',-1):
            if not x.shape[0]==self.D[-1]:
                raise ValueError('shape of x[0] does not match the shape of mps.D[-1]')
            if not x.shape[1]==self.D[-1]:
                raise ValueError('shape of x[1] does not match the shape of mps.D[-1]')
            r=x
            for n in range(len(self)-1,-1,-1):
                r=self.transfer_op(n,direction='r',x=r)
            return r

    def __in_place_unary_operations__(self,operation,*args,**kwargs):
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
        super(MPS,self).__in_place_unary_operations__(operation,*args,**kwargs)
        self.mat=operation(self.mat,*args,**kwargs)
        
    def __unary_operations__(self,operation,*args,**kwargs):
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
        obj=self.copy()
        obj.__in_place_unary_operations__(operation,*args,**kwargs)
        return obj


    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self.Z is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """
        
        if ufunc==np.true_divide:
            #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
            #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
            return self.__idiv__(inputs[1])

        out=kwargs.get('out',())
        for arg in inputs+out:
            if not isinstance(arg,self._HANDLED_UFUNC_TYPES+(type(self),)):
                return NotImplemented

        if out:
            #takes care of in-place operations
            result=[]
            for n,x in np.ndenumerate(self):
                kwargs['out']=tuple(o[n] if isinstance(o,type(self)) else o for o in out)
                ipts=[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs]
                result.append(getattr(ufunc,method)(*ipts,**kwargs))
            matipts=[ipt.mat if isinstance(ipt,type(self)) else ipt for ipt in inputs]                
            matresult=getattr(ufunc,method)(*matipts,**kwargs)
        else:
            result=[getattr(ufunc,method)(*[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs],**kwargs)
                    for n,x in np.ndenumerate(self)]
            matresult=getattr(ufunc,method)(*[ipt.mat if isinstance(ipt,type(self)) else ipt for ipt in inputs],**kwargs)
            
        if method=='reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis=kwargs.get('axis',())
            if axis==None:
                return getattr(ufunc,method)(result+[matresult])
            else:
                raise NotImplementedError('MPS.__array_ufunc__ with argument axis!=None not implemented')

        else:
            obj=MPS(tensors=result,Dmax=self.Dmax,name=None)
            obj.mat=matresult
            return obj

    @property
    def D(self):
        """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
        return (
            [self.get_tensor(0).shape[0]] + 
            [self.get_tensor(n).shape[2] for n in range(len(self))]
        )

    
    @property
    def d(self):
        """
        returns a list containig the bond-dimensions of the MPS
        """
        return [m.shape[2] for m in self]
    
    @property
    def pos(self):
        """
        Returns:
        ----------------------------
        int: the current position of the center bond
        """
        return self._position
       
    @property
    def Dmax(self):
        """
        Dmax is the maximally allowed bond dimension of the MPS
        """
        return self._D
    
    @Dmax.setter
    def Dmax(self,D):
        """
        set Dmax to D
        """

    @classmethod
    def random(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,numpy_func=np.random.random_sample,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')

        return cls(tensors=initializer(numpy_func=numpy_func,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name)

    @classmethod
    def zeros(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.zeros,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name)
        
    @classmethod
    def ones(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.ones,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name)
        
    @classmethod
    def empty(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.empty,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name)


    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """
        inds=range(len(self))
        s1=['Name: ',str(self.name),'\n\n ']+['MPS['+str(ind)+'] of shape '+str(self[ind].shape)+'\n\n '+self[ind].__str__()+' \n\n ' for ind in range(self.pos)]+\
            ['center matrix \n\n ',self.mat.__str__()]+['\n\n MPS['+str(ind)+'] of shape '+str(self[ind].shape)+'\n\n '+self[ind].__str__()+' \n\n ' for ind in range(self.pos,len(self))]+['\n\n Z=',str(self.Z)]
        return ''.join(s1)

    def position(self,bond,schmidt_thresh=1E-16,D=None,r_thresh=1E-14):
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

        assert(bond<=self.N)
        assert(bond>=0)
        """
        set the values for the schmidt_thresh-threshold, D-threshold and r_thresh-threshold
        r_thresh is used in case that svd throws an exception (happens sometimes in python 2.x)
        in this case, one preconditions the method by first doing a qr, and setting all values in r
        which are smaller than r_thresh to 0, then does an svd.
        """
        if bond==self._position:
            return
        
        if bond>self._position:
            self[self._position]=ncon.ncon([self.mat,self[self._position]],[[-1,1],[1,-2,-3]])
            for n in range(self._position,bond):
                if schmidt_thresh < 1E-15 and D==None:
                    tensor,self.mat,Z=self.prepareTensor(self[n],direction=1)
                else:
                    tensor,s,v,Z=self.prepareTruncate(self[n],direction=1,D=D,thresh=schmidt_thresh,\
                                                                    r_thresh=r_thresh)
                    self.mat=s.diag().dot(v)
                self.Z*=Z                    
                self[n]=tensor
                if (n+1)<bond:
                    self[n+1]=ncon.ncon([self.mat,self[n+1]],[[-1,1],[1,-2,-3]])
        if bond<self._position:
            self[self._position-1]=ncon.ncon([self[self._position-1],self.mat],[[-1,1,-3],[1,-2]])

            for n in range(self._position-1,bond-1,-1):
                if schmidt_thresh < 1E-15 and D==None:
                    self.mat,tensor,Z=self.prepareTensor(self[n],direction=-1)
                else:
                    u,s,tensor,Z=self.prepareTruncate(self[n],direction=-1,D=D,thresh=schmidt_thresh,\
                                                                 r_thresh=r_thresh)
                    self.mat=u.dot(s.diag())
                self.Z*=Z                                        
                self[n]=tensor
                if n>bond:
                    self[n-1]=ncon.ncon([self[n-1],self.mat],[[-1,1,-3],[1,-2]])
        self._position=bond



        
    def ortho(self,sites,which):
        """
        checks if the orthonormalization of the mps is OK
        prints out some stuff
        """
        if which in (1,'l','left'):
            if hasattr(sites,'__iter__'):
                return [np.linalg.norm(ncon.ncon([self[site],np.conj(self[site])],[[1,-1,2],[1,-2,2]])-\
                                       self[site].eye(1,dtype=self.dtype)) for site in sites]
            else:
                return np.linalg.norm(ncon.ncon([self[sites],np.conj(self[sites])],[[1,-1,2],[1,-2,2]])-\
                                      self[sites].eye(1,dtype=self.dtype))

        elif which in (-1,'r','right'):
            if hasattr(sites,'__iter__'):
                return [np.linalg.norm(ncon.ncon([self[site],np.conj(self[site])],[[-1,1,2],[-2,1,2]])-self[site].eye(0,dtype=self.dtype)) for site in sites]
            else:
                return np.linalg.norm(ncon.ncon([self[sites],np.conj(self[sites])],[[-1,1,2],[-2,1,2]])-self[sites].eye(0,dtype=self.dtype))
        else:
            raise ValueError("wrong value {0} for variable ```which```; use ('l','r',1,-1,'left,'right')".format(which))



    def get_tensor(self, n):
        """
        get_tensor returns an mps tensors, possibly contracted with the center matrix
        by convention, the center matrix is contracted if n==self.pos
        """

        if (n != self.pos) and (self.pos<len(self)):
            out = self._tensors[n]
        elif (n == self.pos)  and (self.pos<len(self)):
            out = ncon([self.centermatrix,self._tensors[n]],[[-1,1],[1,-2,-3]])
        elif (n != (self.pos-1))  and (self.pos==len(self)):
            out = self._tensors[n]
        elif (n == (self.pos-1))  and (self.pos==len(self)):
            out = ncon([self._tensors[n],self.centermatrix],[[-1,-2,1],[1,-3]])
        
        if n==(len(self)-1):
            return ncon([out,self.connector],[[-1,-2,1],[1,-3]])
        else:
            return out
            
    def set_tensor(self, n, tensor):
        raise NotImplementedError()
        
        
    def applyTwoSiteGate(self,gate,site,Dmax=None,thresh=1E-16,preserve_position=True):
        """
        applies a two-site gate to the mps at site ```site```, 
        and does a truncation with truncation threshold "thresh"
        the center bond is shifted to bond site+1 and mps[site],mps._mat and mps[site+1] are updated


        Parameters:
        --------------------------
        gate:   np.ndarray of shape (dout1,dout2,din1,din2)
                the gate to be applied to the mps.
        site:   int
                the left-hand site of the support of ```gate```
        Dmax:   int
                the maximally allowed bond dimension
                bond dimension will never be larger than ```Dmax```, irrespecitive of ```thresh```
        thresh: float  
                truncation threshold; all schmidt values < ```thresh`` are discarded
        Returns:
        tuple (tw,D):
        tw: float
            the truncated weight
        D:  int
            the current bond dimension D on bond=site+1
        
        """
        assert(len(gate.shape)==4)
        assert(site<len(self)-1)
        if type(gate)!=type(self[site]):
            raise TypeError('MPS.applyTwoSiteGate(): provided gate has to be of same type as MPS tensors')
        if preserve_position==True:
            self.position(site)
            newState=ncon.ncon([self.localWaveFunction(length=1),self[site+1],gate],[[-1,1,2],[1,-4,3],[-2,-3,2,3]])
            [Dl,d1,d2,Dr]=newState.shape
            newState=newState.merge([[0,1],[2,3]])
            U,S,V=newState.svd(full_matrices=False)
            Strunc=S.truncate([Dmax]).squeeze(thresh)
            tw=float(np.sum(S**2)-np.sum(Strunc**2))
            Strunc/=Strunc.norm()
            U=U.truncate([U.shape[0],Strunc.shape[0]]) 
            V=V.truncate([Strunc.shape[0],V.shape[1]])
            self[site]=np.transpose(np.reshape(U,(Dl,d1,Strunc.shape[0])),(0,2,1))
            self[site+1]=np.transpose(np.reshape(V,(Strunc.shape[0],d2,Dr)),(0,2,1))
            self.mat=Strunc.diag()
            self._position=site+1
            return tw,Strunc.shape[0]
        else:
            newState=ncon.ncon([self[site],self[site+1],gate],[[-1,1,2],[1,-4,3],[-2,-3,2,3]])
            [Dl,d1,d2,Dr]=newState.shape
            newState=newState.merge([[0,1],[2,3]])
            U,S,V=newState.svd(full_matrices=False)
            self[site]=np.transpose(np.reshape(ncon.ncon([U,S.diag()],[[-1,1],[1,-2]]),(Dl,d1,S.shape[0])),(0,2,1))
            self[site+1]=np.transpose(np.reshape(V,(S.shape[0],d2,Dr)),(0,2,1))
            return 0.0,S.shape[0]

    def applyOneSiteGate(self,gate,site,preserve_position=True):
        """
        applies a one-site gate to an mps at site "site"
        the center bond is shifted to bond site+1 
        the _Z norm of the mps is changed
        """
        assert(len(gate.shape)==2)
        assert(site<len(self))
        if preserve_position==True:
            self.position(site)
            tensor=ncon.ncon([self.localWaveFunction(length=1),gate],[[-1,-2,1],[-3,1]])
            A,mat,Z=self.prepareTensor(tensor,1)
            self.Z*=Z
            self[site]=A
            self.mat=mat
        else:
            tensor=ncon.ncon([self[site],gate],[[-1,-2,1],[-3,1]])
            self[site]=tensor
        return self

class FiniteMPS(MPS):
    
    def __init__(self,tensors=[],Dmax=None,name=None,fromview=True):
        if not np.sum(tensors[0].shape[0])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[0]'.format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[1])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[-1]'.format(tensors[-1].shape))
        
        super(FiniteMPS,self).__init__(tensors=tensors,Dmax=Dmax,name=name,fromview=fromview)
        self.position(0)
        self.position(len(self))
        
    def diagonalize_center_matrix(self):
        """
        diagonalizes the center matrix and pushes U and V onto the left and right MPS tensors
        """

        if self.pos==0:
            return 
        elif self.pos==len(self):
            return
        else:
            U,S,V=self.mat.svd()
            self[self.pos-1]=ncon.ncon([self[self.pos-1],U],[[-1,1,-3],[1,-2]])
            self[self.pos]=ncon.ncon([V,self[self.pos]],[[-1,1],[1,-2,-3]])
            self.mat=S.diag()
    @classmethod
    def zeros(self,*args,**kwargs):
        raise NotImplementedError('FiniteMPS.zeros(*args,**kwargs) not implemented')
    
    def __add__(self,other):
        """
        adds self with other;
        returns an unnormalized mps
        """
        tensors=[mf.mpsTensorAdder(self[0],other[0],boundary_type='l',ZA=self.Z,ZB=other.Z)]+\
            [mf.mpsTensorAdder(self[n],other[n],boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax+other.Dmax,Z=1.0) #out is an unnormalized MPS

    def __iadd__(self,other):
        """
        adds self with other;
        returns an unnormalized mps
        """
        tensors=[mf.mpsTensorAdder(self[0],other[0],boundary_type='l',ZA=self.Z,ZB=other.Z)]+\
            [mf.mpsTensorAdder(self[n],other[n],boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        self=FiniteMPS(tensors=tensors,Dmax=self.Dmax+other.Dmax,Z=1.0) #out is an unnormalized MPS
    
    def __sub__(self,other):
        """
        subtracts other from self
        returns an unnormalized mps
        """
        tensors=[mf.mpsTensorAdder(self[0],other[0],boundary_type='l',ZA=self.Z,ZB=-other.Z)]+\
            [mf.mpsTensorAdder(self[n],other[n],boundary_type=bt)
             for n, bt in zip(range(1,len(self)),['b']*(len(self)-2)+['r'])]
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax+other.Dmax) #out is an unnormalized MPS

    def schmidt_spectrum(self,n):
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
        U,S,V=self.mat.svd(full_matrices=False)
        return S

    def canonize(self,name=None):
        """
        """
        Lambdas,Gammas=[],[]
        
        self.position(len(self))
        self.position(0)
        Lambdas.append(self.mat.diag())                        
        for n in range(len(self)):
            self.position(n+1)
            self.diagonalize_center_matrix()
            Gammas.append(ncon.ncon([(1.0/Lambdas[-1]).diag(),self[n]],[[-1,1],[1,-2,-3]]))
            Lambdas.append(self.mat.diag())
        return CanonizedFiniteMPS(gammas=Gammas,lambdas=Lambdas,name=name)
        
            
    def truncate(self,schmidt_thresh=1E-16,D=None,presweep=True):
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
        
        if D and D>max(self.D):
            return self
        else:
            pos=self.pos
            if self.pos==0:
                self.position(len(self),schmidt_thresh=schmidt_thresh,D=D)
            elif self.pos==len(self):
                self.position(0,schmidt_thresh=schmidt_thresh,D=D)                
            else:
                if presweep:
                    self.position(0)
                    self.position(self.N)
                self.position(0,schmidt_thresh=schmidt_thresh,D=D)                                    
                self.position(self.pos)
            return self

    def apply_MPO(self,mpo):
        """
        applies an mpo to an mps; no truncation is done
        """
        assert(len(mpo)==len(self))
        res=self.copy()
        res.position(0)
        res.absorbCenterMatrix(1)
        for n in range(len(res)):
            Ml,Mr,din,dout=mpo[n].shape 
            Dl,Dr,d=res[n].shape
            res[n]=ncon.ncon([res[n],mpo[n]],[[-1,-3,1],[-2,-4,-5,1]]).merge([[0,1],[2,3],[4]])
        res.mat=res[0].eye(0,dtype=res.dtype)
        return res
            


    def dot(self,mps):
        """
        calculate the overlap of self with mps 
        mps: MPS
        returns: float
                 overlap of self with mps
        """
        if not len(self)==len(mps):
            raise ValueError('FiniteMPS.dot(other): len(other)!=len(self)')
        if not isinstance(mps,FiniteMPS):
            raise TypeError('can only calculate overlaps with FiniteMPS')
        O=self[0].eye(0,dtype=np.result_type(self.dtype,mps.dtype))
        for site in range(min(self.pos,mps.pos)):
            O=ncon.ncon([O,self[site],np.conj(mps[site])],[[1,2],[1,-1,3],[2,-2,3]])
        if self.pos<mps.pos:
            O=ncon.ncon([O,self.mat],[[1,-2],[1,-1]])
        elif self.pos==mps.pos:
            O=ncon.ncon([O,self.mat,np.conj(mps.mat)],[[1,2],[1,-1],[2,-2]])
        elif self.pos>mps.pos:
            O=ncon.ncon([O,np.conj(mps.mat)],[[-1,1],[1,-2]])
            
        for site in range(min(self.pos,mps.pos),max(self.pos,mps.pos)):
            O=ncon.ncon([O,self[site],np.conj(mps[site])],[[1,2],[1,-1,3],[2,-2,3]])
            
        if self.pos<mps.pos:
            O=ncon.ncon([O,np.conj(mps.mat)],[[-1,1],[1,-2]])
        elif self.pos>mps.pos:
            O=ncon.ncon([O,self.mat],[[1,-2],[1,-1]])            
        for site in range(max(self.pos,mps.pos),len(self)):
            O=ncon.ncon([O,self[site],np.conj(mps[site])],[[1,2],[1,-1,3],[2,-2,3]])
        return O.dtype.type(O)


    def measureMatrixElement(self,mps,op,site,preserve_position=True):
        pos1=self._position
        pos2=mps._position        
        self.position(site)
        mps.position(site)                        
        t1=self.localWaveFunction(length=1)
        t2=mps.localWaveFunction(length=1)        
        if np.abs(self.Z-1.0)>1E-10:
            warnings.warn('MPS.measureMatrixElement(): norm self.Z != 1; discarding it anyway')
        if np.abs(mps.Z-1.0)>1E-10:
            warnings.warn('MPS.measureMatrixElement(): norm mps.Z != 1; discarding it anyway')
            
        if preserve_position:
            self.position(pos1)
            self.position(pos2)            
            
        return ncon.ncon([t1,np.conj(t2),op],[[1,3,2],[1,3,4],[4,2]])
        
    def measureList(self,ops):
        """
        measure a list of N local operators "ops", where N=len(ops)=len(self)
        the routine moves the centersite to the right boundary, measures and moves
        it back to its original position; this might cause some overhead

        Parameters:
        --------------------------------
        ops:           list of np.ndarrays of shape (d,d)
                       local operators to be measured
        Returns:  np.ndarray containing the matrix elements
        """
        assert(len(self)==len(ops))
        return [self.measureLocal(ops[site],site,preserve_position=False) for site in range(len(self))]


    def measureLocal(self,op,site,preserve_position=False):

        """
        measure a local operator "op" at site "site"
        the routine moves the centersite to "site" and measures the operator
        if preserve_position=True, the center site is moved back to its original position
        """
        return self.measureMatrixElement(self,op=op,site=site,preserve_position=preserve_position)


class  CanonizedMPS(TensorNetwork):
    class GammaTensors(object):
        """
        helper class to implement calls such as camps.Gamma[site]
        camps.Gamma[site]=Tensor
        GammaTensors holds a view of the tensors in CanonizedMPS
        
        """
        def __init__(self,tensors):
            self._data=tensors
        def __getitem__(self,site):
            N=np.prod(self._data.shape)
            if site>=N:
                raise IndexError('CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'.format(site,N))
            return self._data[int(2*site+1)]
        
        def __setitem__(self,site,val):
            N=np.prod(self._data.shape)
            if site>=N:
                raise IndexError('CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'.format(site,N))
            self._data[int(2*site+1)]=val

    class LambdaTensors(object):
        """
        helper class to implement calls such as camps.Lambda[site]
        camps.Lambda[site]=Tensor
        LambdaTensors holds a view of the tensors in CanonizedMPS
        """
        
        def __init__(self,tensors):
            self._data=tensors
        def __getitem__(self,site):
            N=np.prod(self._data.shape)
            if site>=N:
                raise IndexError('CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'.format(site,N))
            return self._data[int(2*site)]
        
        def __setitem__(self,site,val):
            N=np.prod(self._data.shape)
            if site>=N:
                raise IndexError('CanonizedMPS.Gamms[index]: index {0} out of bounds or CanonizedMPS of length {1}'.format(site,N))
            self._data[int(2*site)]=val

            
    def __init__(self,gammas=[],lambdas=[],Dmax=None,name=None):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """

        assert((len(gammas)+1)==len(lambdas))
        tensors=[lambdas[0]]
        for n in range(len(gammas)):
            tensors.append(gammas[n])
            tensors.append(lambdas[n+1])
        super(CanonizedMPS,self).__init__(tensors=tensors,shape=(),name=name)
        self.Gamma=self.GammaTensors(self._tensors.view())
        self.Lambda=self.LambdaTensors(self._tensors.view())                        
        if not Dmax:
            self._D=max(self.D)
        else:
            self._D=Dmax


    @classmethod
    def fromList(cls,tensors,Dmax=None,name=None):
        """
        construct a CanonizedMPS from a list containing Gamma and Lambda matrices
        Parameters:
        ------------
        tensors:   list of Tensors
                   is a list of Tensor objects which alternates between ```Lambda`` and ```Gamma``` tensors,
                   starting and stopping with a boundary ```Lambda```
                   ```Lambda``` are in vector-format (i.e. it is the diagonal), ```Gamma``` is a full matrix.
        Dmax:      int
                   maximally allowed bond dimension
        name:      str or None
                   name of the CanonizedMPS
        Returns:
        CanonizeMPS or subclass 
        
        """
        if not np.sum(tensors[0].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[0]'.format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[-1]'.format(tensors[-1].shape))
        assert(len(tensors)%2)
        return cls(gammas=tensors[1::2],lambdas=tensors[0::2],Dmax=Dmax,name=name)

    def __len__(self):
        """
        return the length of the CanonizedMPS, i.e. the number of physical sites

        Parameters: None
        --------------
        Returns:
        int
        """

        return int((len(self._tensors)-1)/2)
    
    @property
    def D(self):
        """
        return a list of bond dimensions for each bond
        """
        return [self.Gamma[n].shape[0] for n in range(len(self))]+[self.Gamma[len(self)-1].shape[1]]
    @property
    def d(self):
        """
        return a list of physicsl dimensions for each site
        """
        return [self.Gamma[n].shape[2] for n in range(len(self))]
    
    @property
    def Dmax(self):
        """
        Return the maximally allowed bond dimension of the CanonizedMPS
        """
        return self._D
    
    @Dmax.setter
    def Dmax(self,D):
        """
        Set the maximally allowed bond dimension of the CanonizedMPS
        """
    @classmethod
    def random(*args,**kwargs):
        raise NotImplementedError('CanonizedMPS.random() only implemented in subclasses')
    
    @classmethod
    def zeros(*args,**kwargs):
        raise NotImplementedError('CanonizedMPS.zeros() only implemented in subclasses')        
    
    @classmethod
    def ones(*args,**kwargs):
        raise NotImplementedError('CanonizedMPS.ones() only implemented in subclasses')                

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        note that the attribute self.Z is NOT operated on with ufunc. ufunc is 
        currently applied elementwise to the tensors in the tensor network
        """
        if ufunc==np.true_divide:
            #this is a dirty hack: division of TensorNetwork by a scalar currently triggers use of
            #__array_ufunc__ method which applies the division to the individual elements of TensorNetwork
            return self.__idiv__(inputs[1])


        out=kwargs.get('out',())
        for arg in inputs+out:
            if not isinstance(arg,self._HANDLED_UFUNC_TYPES+(TensorNetwork,)):
                return NotImplemented
        if out:
            #takes care of in-place operations
            result=[]
            for n,x in np.ndenumerate(self._tensors):
                kwargs['out']=tuple(o[n] if isinstance(o,type(self)) else o for o in out)
                ipts=[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs]
                result.append(getattr(ufunc,method)(*ipts,**kwargs))
        else:
            result=[getattr(ufunc,method)(*[ipt[n] if isinstance(ipt,type(self)) else ipt for ipt in inputs],**kwargs)
                    for n,x in np.ndenumerate(self._tensors)]
        if method=='reduce':
            #reduce is not well defined for mps because the center matrix has different dimension than the other tensors
            #furthermore, reduce weith axis==None reduces ndarrays to a number, not a tensor. This is nonsensical for
            #MPS. Thus, if axis==None, reapply the ufunc to the list of obtained results and return the result
            axis=kwargs.get('axis',())
            if axis==None:
                return getattr(ufunc,method)(result)
            else:
                raise NotImplementedError('CanonizedFiniteMPS.__array_ufunc__ with argument axis!=None not implemented')

        else:
            return self.fromList(tensors=result,Dmax=self.Dmax,name=None)                        

        
class  CanonizedFiniteMPS(CanonizedMPS):
    def __init__(self,gammas=[],lambdas=[],Dmax=None,name=None,fromview=True):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """
        if not np.sum(gammas[0].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS got a wrong shape {0} for gammas[0]'.format(gammas[0].shape))
        if not np.sum(gammas[-1].shape[1])==1:
            raise ValueError('CanonizedFiniteMPS got a wrong shape {0} for gammas[-1]'.format(gammas[-1].shape))

        super(CanonizedFiniteMPS,self).__init__(gammas=gammas,lambdas=lambdas,Dmax=None,name=name)
            
    
    @classmethod
    def random(cls,D=[1,2,1],d=[2,2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):    
        mps=FiniteMPS.random(D=D,d=d,Dmax=Dmax,name=name,initialize=initializer,*args,**kwargs)
        return mps.canonize()
    
    @classmethod
    def zeros(*args,**kwargs):
        raise NotImplementedError
    
    @classmethod
    def ones(*args,**kwargs):
        raise NotImplementedError
        
    @classmethod
    def empty(*args,**kwargs):
        raise NotImplementedError
        

    def __str__(self):
        """
        return a str representation of the TensorNetwork
        """
        inds=range(len(self))
        s1=['Name: ',str(self.name),'\n\n ']+\
            ['Lambda[0] of shape '+str(self.Lambda[0].shape)+'\n\n '+self.Lambda[0].__str__()+' \n\n ']+\
            ['Gamma['+str(ind)+'] of shape '+str(self.Gamma[ind].shape)+'\n\n '+self.Gamma[ind].__str__()+' \n\n '+'Lambda['+str(ind+1)+'] of shape '+str(self.Lambda[ind+1].shape)+'\n\n '+self.Lambda[ind+1].__str__()+' \n\n ' for ind in range(len(self))]


        return ''.join(s1)

    def toMPS(self,name=None):
        """
        cast the CanonizedFiniteMPS to a FiniteMPS
        Returns:
        ----------------
        FiniteMPS
        """
        tensors=[ncon.ncon([self.Lambda[n].diag(),self.Gamma[n]],[[-1,1],[1,-2,-3]]) for n in range(len(self))]
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax,name=name)

    
    def canonize(self,name=None):
        """
        re-canonize the CanonizedFiniteMPS
        Returns:
        ---------------
        CanonizedFiniteMPS
        """
        return self.toMPS().canonize(name=self.name)

    def iscanonized(self,thresh=1E-10):
        left=[self.ortho(n,'l')<thresh for n in range(len(self))]
        right=[self.ortho(n,'r')<thresh for n in range(len(self))]
        if np.all(left) and np.all(right):
            return True
        if not np.all(right):
            right=[self.ortho(n,'r')>=thresh for n in range(len(self))]
            print('{1} is not right canonized at site(s) {0} within {2}'.format(np.nonzero(right)[0][:],type(self),thresh))

        if not np.all(left):
            left=[self.ortho(n,'l')>=thresh for n in range(len(self))]
            print('{1} is not left canonized at site(s) {0} within {2}'.format(np.nonzero(left)[0][:],type(self),thresh))
        return False
        
    def ortho(self,sites,which):
        """
        checks if the orthonormalization of the CanonizedMPS is OK
        """
        if which in (1,'l','left'):
            if hasattr(sites,'__iter__'):
                tensors=[ncon.ncon([self.Lambda[site].diag(),self.Gamma[site]],[[-1,1],[1,-2,-3]]) for site in sites]
                return [np.linalg.norm(ncon.ncon([tensors[n],np.conj(tensors[n])],[[1,-1,2],[1,-2,2]])-\
                                       tensors[n].eye(1,dtype=self.dtype)) for n in range(len(tensors))]
            else:
                tensor=ncon.ncon([self.Lambda[sites].diag(),self.Gamma[sites]],[[-1,1],[1,-2,-3]])
                return np.linalg.norm(ncon.ncon([tensor,np.conj(tensor)],[[1,-1,2],[1,-2,2]])-\
                                      tensor.eye(1,dtype=self.dtype))

        elif which in (-1,'r','right'):
            if hasattr(sites,'__iter__'):
                tensors=[ncon.ncon([self.Lambda[site+1].diag(),self.Gamma[site]],[[1,-2],[-1,1,-3]]) for site in sites]
                return [np.linalg.norm(ncon.ncon([tensors[n],np.conj(tensors[n])],[[-1,1,2],[-2,1,2]])-\
                                       tensors[n].eye(0,dtype=self.dtype)) for n in range(len(tensors))]

            else:
                tensor=ncon.ncon([self.Lambda[sites+1].diag(),self.Gamma[sites]],[[1,-2],[-1,1,-3]])
                return np.linalg.norm(ncon.ncon([tensor,np.conj(tensor)],[[-1,1,2],[-2,1,2]])-\
                                      tensor.eye(0,dtype=self.dtype))

        else:
            raise ValueError("wrong value {0} for variable ```which```; use ('l','r',1,-1,'left,'right')".format(which))
        








