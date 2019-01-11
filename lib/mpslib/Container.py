"""
@author: Martin Ganahl
"""
import pickle
import warnings
import os
import operator as opr
import copy
import numbers
import numpy as np
def generate_unary_deferer(op_func):
    def deferer(cls, *args, **kwargs):
        try:
            return type(cls).__unary_operations__(cls, op_func, *args,**kwargs)
        except AttributeError:
            raise(AttributeError("cannot generate unary deferer for class withtou __unary_operations__"))
    return deferer

def numpy_initializer(numpy_func,shapes,*args,**kwargs):
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
            return [(numpy_func(shape)-mean+1j*(numpy_func(shape)-mean)).astype(dtype)*scale for shape in shapes]
        elif np.issubdtype(dtype,np.floating):
            return [(numpy_func(shape)-mean).astype(dtype)*scale for shape in shapes]
    else:
        if np.issubdtype(dtype,np.complexfloating):
            return [numpy_func(shape,*args,**kwargs)+1j*numpy_func(shape,*args,**kwargs) for shape in shapes]
        elif np.issubdtype(dtype,np.floating):            
            return [numpy_func(shape,*args,**kwargs) for shape in shapes]
        
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
            
        self._tensors=np.empty((len(tensors)),dtype=np.ndarray)
        for n in range(len(tensors)):
            self._tensors[n]=tensors[n]
        self._tensors=self._tensors.reshape(_shape)
        self.Z=np.result_type(*self._tensors,Z).type(Z)

        
    def reshape(self,shape):
        """
        returns a reshaped view of self
        """
        view=TensorNetwork.view(self)
        view._tensors=np.reshape(view._tensors,shape)
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
    def view(cls,other):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        filename:        str or None
                        
        """
        return cls(tensors=other.tensors,shape=other.shape,name=other.name,Z=other.Z)
        
    
    @classmethod
    def random(cls,shape=(),tensorshapes=(),name=None,initializer=numpy_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        filename:        str or None
        """
        return cls(tensors=initializer(np.random.random_sample,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape,Z=1.0)

        
    @classmethod
    def zeros(cls,shapes=[(3,3,2)],name=None,shape=(),initializer=numpy_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        filename:        str or None
                        
        """
        return cls(tensors=initializer(np.zeros,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape,Z=1.0)

    @classmethod
    def ones(cls,shapes=[(3,3,2)],name=None,shape=(),initializer=numpy_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        filename:        str or None
                        
        """
        return cls(tensors=initializer(np.ones,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape,Z=1.0)
    
    @classmethod
    def empty(cls,shapes=[(3,3,2)],name=None,shape=(),initializer=numpy_initializer,*args,**kwargs):
        """
        generate a random TensorNetwork
        Parameters:
        ----------------------------------------------
        initializer:     callable
                         initializer(*args,**kwargs) should return the initial tensors
        filename:        str or None
                        
        """
        return cls(tensors=initializer(np.empty,np.prod(shape)*[tensorshapes],*args,**kwargs),
                   name=name,shape=shape,Z=1.0)
        

    @classmethod
    def fromList(cls,tensors,name=None):
        """
        generate an TN from a list of tensors

        Parameters:
        ----------------------------
        tensors: list tensor objects 
        Returns:
        ----------------------------------
        TensorNetwork object with tensors initialized from ```tensors```
        """

        TN=cls(tensors=tensors,name=name)
        TN.Z=self.dtype.type(1.0)
        return TN

    def append(self,t):
        self._tensors.append(t)
        
    def extend(self,ts):
        self._tensors.extend(ts)
    def insert(self,index,t):
        self._tensors.insert(index,t)
    def pop(self,index):
        return self._tensors.pop(index)

    def __getitem__(self,n,**kwargs):
        return self._tensors[n]

    def __setitem__(self,n,tensor,**kwargs):
        self._tensors[n]=tensor

    def __str__(self):
        """
        printing function
        """
        
        inds=np.unravel_index(range(len(self)),dims=self.shape)
        inds=list(zip(*inds))
        return ''.join(['\n\n ']+['TN'+str(index)+' \n\n '+self[index].__str__()+' \n\n ' for index in inds]+['\n\n Z=',str(self.Z)])

    def __len__(self):
        return np.prod(self.shape)

    def __iter__(self):
        return self._tensors.__iter__()
    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        """
        implements np.ufuncs for the TensorNetwork
        for numpy compatibility
        """
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

        return TensorNetwork(tensors=result,name=None)
    
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
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        return TensorNetwork(tensors=copy.deepcopy(self._tensors),name=None,Z=self.Z*num)


    def __imul__(self,num):
        """
        left-multiplies "num" with MPS, i.e. returns MPS*num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        self.Z*=num
        return self
    

    def __idiv__(self,num):
        """
        left-divides "num" with MPS, i.e. returns MPS*num;
        note that "1./num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        self.Z/=num
        return self
    
    def __truediv__(self,num):
        """
        left-divides "num" with MPS, i.e. returns MPS/num;
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state
        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__mul__(self,num): num is not a number")
        return TensorNetwork(tensors=copy.deepcopy(self._tensors),name=None,Z=self.Z/num)        
    
    def __rmul__(self,num):
        """
        right-multiplies "num" with MPS, i.e. returns num*MPS;
        WARNING: if you are using numpy number types, i.e. np.float, np.int, ..., 
        the right multiplication of num with MPS, i.e. num*MPS, returns 
        an np.darray instead of an MPS. 
        note that "num" is not really multiplied into the mps matrices, but
        instead multiplied into the internal field _Z which stores the norm of the state

        """
        if not np.isscalar(num):
            raise TypeError("in MPS.__rmul__(self,num): num is not a number")
        return TensorNetwork(tensors=copy.deepcopy(self._tensors),name=None,Z=self.Z*num)                

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
        for n in range(len(res)):
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
        return TensorNetwork(tensor=[operation(self[n],*args,**kwargs) for n in range(len(self))],name=None)



    
class MPS(TensorNetwork):
    
    def __init__(self,tensors=[],Dmax=None,name=None,Z=1.0):    
        super(MPS,self).__init__(tensors=tensors,shape=(),name=name,Z=1.0)
        if not Dmax:
            self._D=max(self.D)
        else:
            self._D=Dmax
            

        self._mat=np.eye(np.shape(self._tensors[-1])[1]).astype(self.dtype)
        self._mat=self._mat/np.sqrt(np.trace(self._mat.dot(herm(self._mat))))
        self._position=self.N
        self.gammas=[]
        self.lambdas=[]

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
        self._D=D


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
            self[self._position]=np.tensordot(self._mat,self[self._position],([1],[0]))
            for n in range(self._position,bond):
                if schmidt_thresh < 1E-15 and D==None:
                    tensor,self._mat,Z=mf.prepareTensor(self[n],direction=1)
                    
                else:
                    tensor,s,v,Z=mf.prepareTruncate(self[n],direction=1,D=D,thresh=schmidt_thresh,\
                                                    r_thresh=r_thresh)
                    self._mat=np.diag(s).dot(v)
                self.Z*=Z                    
                self[n]=tensor
                if (n+1)<bond:
                    self[n+1]=np.tensordot(self._mat,self[n+1],([1],[0]))

        if bond<self._position:
            self[self._position-1]=np.transpose(np.tensordot(self[self._position-1],self._mat,([1],[0])),(0,2,1))
            for n in range(self._position-1,bond-1,-1):
                if schmidt_thresh < 1E-15 and D==None:
                    tensor,self._mat,Z=mf.prepareTensor(self[n],direction=-1)

                else:
                    u,s,tensor,Z=mf.prepareTruncate(self[n],direction=-1,D=D,thresh=schmidt_thresh,\
                                                    r_thresh=r_thresh)
                    self._mat=u.dot(np.diag(s))
                self._Z*=Z                                        
                self[n]=tensor
                if n>bond:
                    self[n-1]=np.transpose(np.tensordot(self[n-1],self._mat,([1],[0])),(0,2,1))
        self._position=bond
        #print("after position: self._D=",self._D)
        
        


    
