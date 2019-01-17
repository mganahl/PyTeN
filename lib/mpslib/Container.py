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
        if tensors:
            self.tensortype=type(tensors[0])

        
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
            return TensorNetwork(tensors=result,shape=self.shape,name=None,Z=self.Z)            
            


    
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
    def __init__(self,tensors=[],Dmax=None,name=None,Z=1.0):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """
        super(MPS,self).__init__(tensors=tensors,shape=(),name=name,Z=1.0)
        if not Dmax:
            self._D=max(self.D)
        else:
            self._D=Dmax
        self.mat=tensors[-1].eye(1)
        self.mat=self.mat/np.sqrt(np.trace(self.mat.dot(herm(self.mat))))
        self._position=self.N
        
    def normalize(self):
        self.position(len(self))
        self.position(0)        
        self.Z=self.dtype.type(1)

 
    def view(self):
        """
        return a view of self
        Parameters:
        ----------------------------------------------
        """
        obj= self.__new__(type(self))
        obj.__init__(tensors=self.tensors,Dmax=self.Dmax,name=self.name,Z=self.Z)
        return obj
        
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
            obj=MPS(tensors=result,Dmax=self.Dmax,name=None,Z=1.0)
            obj.mat=matresult
            return obj


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

    @classmethod
    def random(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')

        return cls(tensors=initializer(numpy_func=np.random.random_sample,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name,Z=1.0)

    @classmethod
    def zeros(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.zeros,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name,Z=1.0)
        
    @classmethod
    def ones(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.ones,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name,Z=1.0)
        
    @classmethod
    def empty(cls,D=[2,2],d=[2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        """
        if len(D)!=len(d)+1:
            raise ValueError('len(D)!=len(d)+1')
        return cls(tensors=initializer(numpy_func=np.empty,shapes=[(D[n],D[n+1],d[n]) for n in range(len(d))],*args,**kwargs), Dmax=Dmax,name=name,Z=1.0)


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
                    tensor,self.mat,Z=self.tensortype.prepareTensor(self[n],direction=1)
                else:
                    tensor,s,v,Z=self.tensortype.prepareTruncate(self[n],direction=1,D=D,thresh=schmidt_thresh,\
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
                    tensor,self.mat,Z=self.tensortype.prepareTensor(self[n],direction=-1)
                else:
                    u,s,tensor,Z=self.tensortype.prepareTruncate(self[n],direction=-1,D=D,thresh=schmidt_thresh,\
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
        return the local wavefunction over ```length``` sites
        if length==0: self.mat is returned
        if length==1: self.mat contracted with the next-right mps-tensor is returned, unless
                      self.pos==len(self), in which case self.mat contracted with the next-left mps-tensor 
                      is returned

        if length==2: self.mat contracted with the nex-left and next-right mps-tensor is returned, unless:
                      1) self.pos in (len(self)-1,len(self)), in which case self.mat contracted with the two leftmost 
                         mps-tensors is returned
                      2) self.pos in (0,1), in which case self.mat contracted with the two rightmost 
                         mps-tensors is returned
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
            newState=newState.merge([0,1],[2,3])
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
            newState=newState.merge([0,1],[2,3])
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
            A,mat,Z=self.tensortype.prepareTensor(tensor,1)
            self.Z*=Z
            self[site]=A
            self.mat=mat
        else:
            tensor=ncon.ncon([self[site],gate],[[-1,-2,1],[-3,1]])
            self[site]=tensor
        return self

class FiniteMPS(MPS):
    
    def __init__(self,tensors=[],Dmax=None,name=None,Z=1.0):
        if not np.sum(tensors[0].shape[0])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[0]'.format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[1])==1:
            raise ValueError('FiniteMPS got a wrong shape {0} for tensor[-1]'.format(tensors[-1].shape))
        
        super(FiniteMPS,self).__init__(tensors=tensors,Dmax=Dmax,name=name,Z=Z)
        self.position(0)
        self.position(len(self))
        self.gammas=[]
        self.lambdas=[]
        
    def diagonalizeCenterMatrix(self):
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
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax+other.Dmax,Z=1.0) #out is an unnormalized MPS

    def SchmidtSpectrum(self,n):
        """
        SchmidtSpectrum(n):

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

    def canonize(self):
        """
        canonizes the mps, i.e. brings it into Gamma,Lambda form; Gamma and Lambda are stored in
        mps.gammas and mps.lambdas member lists;
        len(mps.lambdas) is len(mps)+1, i.e. there are boundary lambdas to the left and right of the mps; 
        for obc, these are just [1.0]
        The mps is left in a left orthogonal state

        Parameters:
        ------------------------------
        nmaxit:      int
                     maximum iteration number of sparse solver
        tol:         float
                     desired precision of eigenvalues/eigenvectors returned by sparse solver
        ncv:         int
                     number of krylov vectors used in sparse sovler
        pinv:        float
                     pseudo inverse cutoff
        numeig:      number of eigenvalue-eigenvector pairs to be calculated in sparse solver (hyperparameter)

        Returns:
        -------------------------
        None
        """
        Lambdas,Gammas=[],[]
        
        self.position(len(self))
        self.position(0)
        Lambdas.append(self.mat.diag())                        
        for n in range(len(self)):
            self.position(n+1)
            self.diagonalizeCenterMatrix()
            Gammas.append(ncon.ncon([(1.0/Lambdas[-1]).diag(),self[n]],[[-1,1],[1,-2,-3]]))
            Lambdas.append(self.mat.diag())
        return CanonizedFiniteMPS(gammas=Gammas,lambdas=Lambdas,name=None,Z=1.0)
        
            
    def truncate(self,schmidt_thresh=1E-16,D=None,presweep=True):
        """ 
        a dedicated routine for truncating an mps (for obc, this can also be done using self.position(self,pos))
        For the case of obc==True (infinite system with finite unit-cell), the function modifies self._connector
        schmidt_thresh: truncation threshold
        D: maximum bond dimension; if None, the bond dimension is adapted to match schmidt_thresh
        returned_gauge: 'left','right' or 'symmetric': the desired gauge after truncation
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

    def applyMPO(self,mpo):
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
            res[n]=ncon.ncon([res[n],mpo[n]],[[-1,-3,1],[-2,-4,-5,1]]).merge([0,1],[2,3],[4])
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
    
                
class  CanonizedFiniteMPS(TensorNetwork):
    def __init__(self,gammas=[],lambdas=[],Dmax=None,name=None,Z=1.0):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """
        assert((len(gammas)+1)==len(lambdas))
        tensors=[lambdas[0]]
        for n in range(len(gammas)):
            tensors.append(gammas[n])
            tensors.append(lambdas[n+1])
        if not np.sum(gammas[0].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS got a wrong shape {0} for gammas[0]'.format(gammas[0].shape))
        if not np.sum(gammas[-1].shape[1])==1:
            raise ValueError('CanonizedFiniteMPS got a wrong shape {0} for gammas[-1]'.format(gammas[-1].shape))

        super(CanonizedFiniteMPS,self).__init__(tensors=tensors,shape=(),name=name,Z=1.0)
        if not Dmax:
            self._D=max(self.D)
        else:
            self._D=Dmax
            
    @classmethod
    def fromList(cls,tensors=[],Dmax=None,name=None,Z=1.0):
        """
        no checks are performed to see wheter the prived tensors can be contracted
        """
        if not np.sum(tensors[0].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[0]'.format(tensors[0].shape))
        if not np.sum(tensors[-1].shape[0])==1:
            raise ValueError('CanonizedFiniteMPS.fromList(tensors) got a wrong shape {0} for tensors[-1]'.format(tensors[-1].shape))
        assert(len(tensors)%2)
        return cls(gammas=tensors[1::2],lambdas=tensors[0::2],Dmax=Dmax,name=name,Z=Z)
        
    def __len__(self):
        return int((len(self._tensors)-1)/2)
    
    def Gamma(self,site):
        if site>=len(self):
            raise IndexError('CanonizedFiniteMPS.Gammas(index): index {0} out of bounds or CanonizedFiniteMPS of length {1}'.format(bond,len(self)))
        return super(CanonizedFiniteMPS,self).__getitem__(int(2*site+1))
                             
    def Lambda(self,bond):
        if bond>len(self):
            raise IndexError('CanonizediniteMPS.Lambda(index): index {0} out of bounds or CanonizedFiniteMPS of length {1}'.format(bond,len(self)))
        
        return super(CanonizedFiniteMPS,self).__getitem__(int(2*bond))        

    @property
    def D(self):
        return [self.Gamma(n).shape[0] for n in range(len(self))]+[self.Gamma(len(self)-1).shape[1]]
    @property
    def d(self):
        return [self.Gamma(n).shape[2] for n in range(len(self))]
    
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
    def random(cls,D=[1,2,1],d=[2,2],Dmax=None,name=None,initializer=ndarray_initializer,*args,**kwargs):    
        mps=FiniteMPS.random(D=D,d=d,Dmax=Dmax,name=name,initialize=initializer,*args,**kwargs)
        return mps.canonize()
        

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
                raise NotImplementedError('CanonizedFiniteMPS.__array_ufunc__ with argument axis!=None not implemented')

        else:
            return CanonizedFiniteMPS.fromList(tensors=result,Dmax=self.Dmax,name=None,Z=self.Z)                        
        
    def toMPS(self):
        tensors=[ncon.ncon([self.Lambda(n).diag(),self.Gamma(n)],[[-1,1],[1,-2,-3]]) for n in range(len(self))]
        return FiniteMPS(tensors=tensors,Dmax=self.Dmax,name=None,Z=self.Z)

        
    
    def canonize(self):
        return self.toMPS().canonize()
    
    def ortho(self,sites,which):
        """
        checks if the orthonormalization of the mps is OK
        prints out some stuff
        """
        if which in (1,'l','left'):
            if hasattr(sites,'__iter__'):
                tensors=[ncon.ncon([self.Lambda(site).diag(),self.Gamma(site)],[[-1,1],[1,-2,-3]]) for site in sites]
                return [np.linalg.norm(ncon.ncon([tensors[n],np.conj(tensors[n])],[[1,-1,2],[1,-2,2]])-\
                                       tensors[n].eye(1,dtype=self.dtype)) for n in range(len(tensors))]
            else:
                tensor=ncon.ncon([self.Lambda(sites).diag(),self.Gamma(sites)],[[-1,1],[1,-2,-3]])
                return np.linalg.norm(ncon.ncon([tensor,np.conj(tensor)],[[1,-1,2],[1,-2,2]])-\
                                      tensor.eye(1,dtype=self.dtype))

        elif which in (-1,'r','right'):
            if hasattr(sites,'__iter__'):
                tensors=[ncon.ncon([self.Lambda(site+1).diag(),self.Gamma(site)],[[1,-2],[-1,1,-3]]) for site in sites]
                return [np.linalg.norm(ncon.ncon([tensors[n],np.conj(tensors[n])],[[-1,1,2],[-2,1,2]])-\
                                       tensors[n].eye(0,dtype=self.dtype)) for n in range(len(tensors))]

            else:
                tensor=ncon.ncon([self.Lambda(sites+1).diag(),self.Gamma(sites)],[[1,-2],[-1,1,-3]])
                return np.linalg.norm(ncon.ncon([tensor,np.conj(tensor)],[[-1,1,2],[-2,1,2]])-\
                                      tensor.eye(0,dtype=self.dtype))

        else:
            raise ValueError("wrong value {0} for variable ```which```; use ('l','r',1,-1,'left,'right')".format(which))
