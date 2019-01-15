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
    initializer function for numpy.npdaarys
    Parameters:
    ---------------------
    numpy_func:       callable
                      a numpy function like np.random.random_sample
    shapes:           list of tuple
                      the shapes of the individual tensors
    *args,**kwargs:   if numpy_func is not one of np.random functions, these are passed on to numpy_func

    Returns:
    -------------------------
    list of np.ndarrays of shape ```shapes```, initialized with numpy_func
    """
    mean=kwargs.get('mean',0.5)
    scale=kwargs.get('scale',0.1)
    dtype=kwargs.get('dtype',np.float64)

    if numpy_func in (np.random.random_sample,np.random.rand,np.random.randn):
        if np.issubdtype(dtype,np.complexfloating):
            return [(numpy_func(shape).view(Tensor)-mean+1j*(numpy_func(shape).view(Tensor)-mean)).astype(dtype)*scale for shape in shapes]

        elif np.issubdtype(dtype,np.floating):
           return [(numpy_func(shape).view(Tensor)-mean).astype(dtype)*scale for shape in shapes]
    else:
        if np.issubdtype(dtype,np.complexfloating):
            return [numpy_func(shape,*args,**kwargs).view(Tensor)+1j*numpy_func(shape,*args,**kwargs).view(Tensor) for shape in shapes]
        elif np.issubdtype(dtype,np.floating):            
            return [numpy_func(shape,*args,**kwargs).view(Tensor) for shape in shapes]
        
    
class Container(object):

    def __init__(self,name=None):
        """
        Base class for holdig all objects
        filename: str
                  the name of the object, used for storing things
        """
        self.name=name        
        
    def save(self,filename=None):
        """
        dumps the container into a pickle file named "filename"
        Parameters:
        -----------------------------------
        filename: str
                  the filename of the file
        """
        if filename != None:
            with open(filename+'.pickle', 'wb') as f:
                pickle.dump(self,f)
        elif self.name != None:
            with open(self.name+'.pickle', 'wb') as f:
                pickle.dump(self,f)
        else:
            raise ValueError('Cotainer has no name; cannot save it')

    def copy(self):
        """
        for now this does a deep copy using copy module
        """
        return copy.deepcopy(self)
            
    @classmethod
    def read(self,filename=None):
        """
        reads a Container from a pickle file "filename".pickle
        and returns a Container object
        Parameters:
        -----------------------------------
        filename: str
                  the filename of the object to be loaded
        Returns:
        ---------------------
        a Container object holding the loaded data
        """
        if filename is not None:
            with open(filename, 'rb') as f:
                out=pickle.load(f)
        elif self.name is not None:
            with open(self.name, 'rb') as f:
                out=pickle.load(f)
        else:
            raise ValueError('cannot read Container-file: all no name given')            
        return out

    
    def load(self,filename):
        """
        Container.load(filename):
        unpickles a Container object from a file "filename".pickle
        and stores the result in self
        """
        if filename is not None:        
            with open(filename, 'rb') as f:
                cls=pickle.load(f)
            #delete all attributes of self which are not present in cls
            todelete=[attr for attr in vars(self) if not hasattr(cls,attr)]
            for attr in todelete:
                delattr(self,attr)
                
            for attr in cls.__dict__.keys():
                setattr(self,attr,getattr(cls,attr))
        elif self.name is not None:
            with open(self.name, 'rb') as f:
                cls=pickle.load(f)
            #delete all attributes of self which are not present in cls
            todelete=[attr for attr in vars(self) if not hasattr(cls,attr)]
            for attr in todelete:
                delattr(self,attr)
                
            for attr in cls.__dict__.keys():
                setattr(self,attr,getattr(cls,attr))

        else:
            raise ValueError('cannot load Container-file: all no name given')            

        
class TensorNetwork(Container,np.lib.mixins.NDArrayOperatorsMixin):
    _HANDLED_UFUNC_TYPES=(numbers.Number,np.ndarray)
    _order='C'
    def __init__(self,tensors=[],shape=(),name=None,Z=1.0):
        """
        initialize an unnormalized TensorNetwork from a list of tensors
        Parameters
        ----------------------------------------------------
        tensors: list() of tensors
                 can be either np.ndarray or any other object 
                 a list containing the tensors of the network
                 the entries in tensors should have certain properties that
                 tensors usually have
        """
        #TODO: add attribute checks for the elements of tensors
        super(TensorNetwork,self).__init__(name) #initialize the Container base class            
        if shape:
            if not np.prod(shape)==len(tensors):
                raise ValueError('shape={0} incompatible with len(tensors)={1}'.format(shape,len(tensors)))
            if not isinstance(shape,tuple):
                raise TypeError('TensorNetwor.__init__(): got wrong type for shape; only tuples are allowed')
            _shape=shape
        else:
            _shape=tuple([len(tensors)])

        self._tensors=np.empty(_shape,dtype=Tensor)
        for n in range(len(tensors)):
            self._tensors.flat[n]=tensors[n].view(Tensor)
        self.Z=np.result_type(*self._tensors,Z).type(Z)

    def view(self):
        """
        return a view of self
        Parameters:
        ----------------------------------------------
        Return:   TensorNetwork
        """
        obj=self.__new__(type(self))
        obj.__init__(tensors=self.tensors,shape=self.shape,name=self.name,Z=self.Z)
        return obj
    
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
        return len(self._tensors)
    
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
        return a list (view) of the tensors in TensorNetwork
        """
        return list(self._tensors.flat)

    
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
                   name=name,shape=shape,Z=1.0)

        
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
                   name=name,shape=shape,Z=1.0)

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
                   name=name,shape=shape,Z=1.0)
    
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
                   name=name,shape=shape,Z=1.0)
        

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
    
    def _ufunc_handler(self,tensors):
        return TensorNetwor(tensors=tensors,name=None)
    
    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        """
        if ufunc==np.true_divide:
            print('sdf')
            return self.__idiv__(inputs[1])


        out=kwargs.get('out',())
        for arg in inputs+out:
            if not isinstance(arg,self._HANDLED_UFUNC_TYPES+(TensorNetwork,)):
                return NotImplemented

        if out:
            #takes care of in-place operations
            result=[]
            for n in range(len(self)):
                kwargs['out']=tuple(o[n] if isinstance(o,TensorNetwork) else o for o in out)
                ipts=[ipt[n] if isinstance(ipt,TensorNetwork) else ipt for ipt in inputs]
                result.append(getattr(ufunc,method)(*ipts,**kwargs))
        else:
            result=[getattr(ufunc,method)(*[ipt[n] if isinstance(ipt,TensorNetwork) else ipt for ipt in inputs],**kwargs)
                    for n in range(len(self))]

        return self._ufunc_handler(tensors=result)
    
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
        print('calling truediv')
        
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
    
    def __init__(self,tensors=[],Dmax=None,name=None,Z=1.0):    
        super(MPS,self).__init__(tensors=tensors,shape=(),name=name,Z=1.0)
        if not Dmax:
            self._D=max(self.D)
        else:
            self._D=Dmax
        self.mat=tensors[-1].eye(1)
        self.mat=self.mat/np.sqrt(np.trace(self.mat.dot(herm(self.mat))))
        self._position=self.N

 
    def view(self):
        """
        return a view of self
        Parameters:
        ----------------------------------------------
        """
        obj= self.__new__(type(self))
        obj.__init__(tensors=self.tensors,Dmax=self.Dmax,name=self.name,Z=self.Z)
        return obj
        
    # def __unary_operations__(self,operation,*args,**kwargs):
    #     """
    #     implements unary operations on MPS tensors
    #     Parameters:
    #     ----------------------------------------
    #     operation: method
    #                the operation to be applied to the mps tensors
    #     *args,**kwargs: arguments of operation

    #     Returns:
    #     -------------------
    #     MPS:  MPS object obtained from acting with operation on each individual MPS tensor
    #     """
    #     return MPS(tensor=[operation(self[n],*args,**kwargs) for n in range(len(self))],Dmax=self.Dmax,name=None,Z=1.0)

    # def __mul__(self,num):
    #     """
    #     left-multiplies "num" with TensorNetwork, i.e. returns TensorNetwork*num;
    #     note that "num" is not really multiplied into the mps matrices, but
    #     instead multiplied into the internal field _Z which stores the norm of the state
    #     Parameters:
    #     -----------------------
    #     num: float or complex
    #          to be multiplied into the MPS
    #     Returns:
    #     ---------------
    #     MPS:    the state obtained from multiplying ```num``` into MPS
    #     """
    #     if not np.isscalar(num):
    #         raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
    #     return MPS(tensors=copy.deepcopy(self._tensors),Dmax=self.Dmax,name=None,Z=self.Z*num)
    
    # def __truediv__(self,num):
    #     """
    #     left-divides "num" with TensorNetwork, i.e. returns TensorNetwork/num;
    #     note that "num" is not really multiplied into the mps matrices, but
    #     instead multiplied into the internal field _Z which stores the norm of the state
    #     """
    #     if not np.isscalar(num):
    #         raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
    #     return MPS(tensors=copy.deepcopy(self._tensors),Dmax=self.Dmax,name=None,Z=self.Z/num)        

     
    # def __rmul__(self,num):
    #     """
    #     right-multiplies "num" with TensorNetwork, i.e. returns num*TensorNetwork;
    #     WARNING: if you are using numpy number types, i.e. np.float, np.int, ..., 
    #     the right multiplication of num with TensorNetwork, i.e. num*TensorNetwork, returns 
    #     an np.darray instead of an TensorNetwork. 
    #     note that "num" is not really multiplied into the mps matrices, but
    #     instead multiplied into the internal field _Z which stores the norm of the state

    #     """
    #     if not np.isscalar(num):
    #         raise TypeError("in TensorNetwork.__mul__(self,num): num is not a number")
        return MPS(tensors=copy.deepcopy(self._tensors),Dmax=self.Dmax,name=None,Z=self.Z*num)
        
    def _ufunc_handler(self,tensors):
        """
        subclasses of TensorNetwork should implement _ufunc_handler to ensure
        that numpy ufunc calls are type preserving
        """
        return MPS(tensors=tensors,Dmax=self.Dmax,name=None,Z=1.0)

    @property
    def D(self):
        """
        returns a list containig the bond-dimensions of the MPS
        """
        return [m.shape[0] for m in self]+[self[-1].shape[1]]

    
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

    def _mpsinitializer(self,numpy_func,D,d,Dmax,name,initializer=ndarray_initializer,*args,**kwargs):
        """
        initializes the tensor network, using the ```initializer``` function
        subclasses of MPS should implement _mpsinitializer to enable initialization via
        classmethods random, ones, zeros and empty
        
        numpy_func:   callable: 
                      numpy function\
        D:            list of in 

        d:            list of int

        name:         str
        
        initializer:  callable
                      signature: initializer(numpy_func,shapes=[],*args,**kwargs)
                      returns a list of tensors
        *args,**kwargs:  additional parameters to ```initializer```
        """
        return MPS(tensors=initializer(numpy_func=numpy_func,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],
                                       *args,**kwargs),
                   Dmax=Dmax,name=name,Z=1.0)
    @classmethod
    def random(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls._mpsinitializer(cls,numpy_func=np.random.random_sample,D=D,d=d,Dmax=Dmax,name=name,
                                   initializer=ndarray_initializer,*args,**kwargs)

    @classmethod
    def zeros(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls._mpsinitializer(cls,numpy_func=np.zeros,D=D,d=d,Dmax=Dmax,name=name,
                                   initializer=ndarray_initializer,*args,**kwargs)
        
    @classmethod
    def ones(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls._mpsinitializer(cls,numpy_func=np.ones,D=D,d=d,Dmax=Dmax,name=name,
                                   initializer=ndarray_initializer,*args,**kwargs)

    @classmethod
    def empty(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls._mpsinitializer(cls,numpy_func=np.empty,D=D,d=d,Dmax=Dmax,name=name,
                                   initializer=ndarray_initializer,*args,**kwargs)

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
                    tensor,self.mat,Z=mf.prepareTensor(self[n],direction=1)
                else:
                    tensor,s,v,Z=mf.prepareTruncate(self[n],direction=1,D=D,thresh=schmidt_thresh,\
                                                    r_thresh=r_thresh)
                    self.mat=s.diag(s).dot(v)
                self.Z*=Z                    
                self[n]=tensor
                if (n+1)<bond:
                    self[n+1]=ncon.ncon([self.mat,self[n+1]],[[-1,1],[1,-2,-3]])
        if bond<self._position:
            self[self._position-1]=ncon.ncon([self[self._position-1],self.mat],[[-1,1,-3],[1,-2]])

            for n in range(self._position-1,bond-1,-1):
                if schmidt_thresh < 1E-15 and D==None:
                    tensor,self.mat,Z=mf.prepareTensor(self[n],direction=-1)
                else:
                    u,s,tensor,Z=mf.prepareTruncate(self[n],direction=-1,D=D,thresh=schmidt_thresh,\
                                                    r_thresh=r_thresh)
                    self.mat=u.dot(s.diag(s))
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



        
        
    def absorbCenterMatrix(self,direction):
        """
        merges self.mat into the MPS. self.mat is merged into either the left (direction in (-1,'l','left')) or 
        the right (direction in (1,'r','right')) tensor at bond self.pos
        changes self.mat to be the identity: self.mat=11
        """
        assert(direction!=0)
        if (self.pos==len(self)) and (direction in (1,'r','right')):
            direction=-1
            warnings.warn('mps.absorbCenterMatrix(self,direction): self.pos=len(self) and direction>0; cannot contract bond-matrix to the right because there is no right tensor; absorbing into the left tensor')
        elif (self.pos==0) and (direction in (-1,'l','left')):
            direction=1                
            warnings.warn('mps.absorbCenterMatrix(self,direction): self.pos=0 and direction<0; cannot contract bond-matrix to the left tensor because there is no left tensor; absorbing into right tensor')

        if (direction in (1,'r','right')):
            self[self.pos]=ncon.ncon([self.mat,self[self.pos]],[[-1,1],[1,-2,-3]])
            self.mat=self.mat.eye(0,dtype=self.dtype)
        elif (direction in (-1,'l','left')):
            self[self.pos-1]=ncon.ncon([self[self.pos-1],self.mat],[[-1,1,-3],[1,-2]])
            self.mat=self.mat.eye(1,dtype=self.dtype)
        return self

    
    def localWaveFunction(self,length=0):
        
        """
        """
        if length==0:
            return copy.deepcopy(self.mat)
        elif length==1:
            if self.pos<len(self):
                return ncon.ncon([self.mat,self[self.pos]],[[-1,1],[1,-2,-3]])
            elif self.pos==len(self):
                return ncon.ncon([self[self.pos-1],self.mat],[[-1,1,-3],[1,-2]])
        elif length==2:
            if (self.pos<len(self)) and (self.pos>0):
                shape=(self.D[self.pos-1],self.D[self.pos+1],self.d[self.pos-1]*self.d[self.pos])
                return ncon.ncon([self[self.pos-1],self.mat,self[self.pos]],[[-1,1,-3],[1,2],[2,-2,-4]]).reshape(shape)
            elif self.pos==len(self):
                shape=(self.D[self.pos-2],self.D[self.pos],self.d[self.pos-2]*self.d[self.pos-1])
                return ncon.ncon([self[self.pos-2],self[self.pos-1],self.mat],[[-1,1,-3],[1,2,-4],[2,-2]]).reshape(shape)
            elif self.pos==0:
                shape=(self.D[self.pos],self.D[self.pos+2],self.d[self.pos]*self.d[self.pos+1])                
                return ncon.ncon([self.mat,self[self.pos],self[self.pos+1]],[[-1,1],[1,2,-3],[2,-2,-4]]).reshape(shape)
        else:
            return NotImplemented


        
class FiniteMPS(MPS):
    
    def __init__(self,tensors=[],Dmax=None,name=None,Z=1.0):
        if not np.sum(tensors[0].shape[0])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[0]'.format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[1])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[-1]'.format(tensors[-1].shape))
        
        super(FiniteMPS,self).__init__(tensors=tensors,Dmax=Dmax,name=name,Z=Z)
        self.gammas=[]
        self.lambdas=[]
        
    def _ufunc_handler(self,tensors):
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax,name=None,Z=1.0)

    def _mpsinitializer(self,numpy_func,D,d,Dmax,name,initializer=ndarray_initializer,*args,**kwargs):
        """
        initializes the tensor network, using the ```initializer``` function
        subclasses of MPS should implement _mpsinitializer to enable initialization via
        classmethods random, ones, zeros and empty
        
        numpy_func:   callable: 
                      numpy function\
        D:            list of in 

        d:            list of int

        name:         str
        
        initializer:  callable
                      signature: initializer(numpy_func,shapes=[],*args,**kwargs)
                      returns a list of tensors
        *args,**kwargs:  additional parameters to ```initializer```
        """
        return FiniteMPS(tensors=initializer(numpy_func=numpy_func,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],
                                             *args,**kwargs),
                         Dmax=Dmax,name=name,Z=1.0)
    
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
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax+other.Dmax,Z=1.0) #out is an unnormalized MPS


