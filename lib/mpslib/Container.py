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

        
    
class Container(object):
    """
    Base class for all tensor networks; implements loading, saving and copying
    """
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

        


        
