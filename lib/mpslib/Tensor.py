import numpy as np
import warnings

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
    
    # def merge(self,*indices):
    #     """
    #     merges indices in the list ```indices```, respecting the order of the elements of ```indices```,
    #     array is transposed such that: all indices smaller or equal to the smalles element in ```indices``` remain at their positions;
    #     these indices are followed by all elements of ```indices``, followed by all indices larger than ```min(indices)``` and not contained in ```indices```.
    #     Parameters:
    #     ----------------------
    #     indices:   list of int
    #                the indices which should be merged
    #     Returns:
    #     ----------------------
    #     Tensor:    the tensor with merged indices
    #     """
    #     inds=list(indices[0])
    #     [inds.extend(n) for n in indices[1:]]
    #     print(inds)
    #     if len(set(inds))<len(inds):
    #         elems,cnts=np.unique(np.asarray(inds),return_counts=True)
    #         raise ValueError('Tensor.merge(): indices {0} are appearing more than once!'.format(elems[cnts>1]))

    #     if max(indices[0])>=len(self.shape):
    #         raise ValueError("Tensor.merge(indices): max(indices)>=len(shape)")
    #     left=list(range(0,min(indices[0])))
    #     complement=sorted(list(set(range(len(self.shape))).difference(set(left+list(indices[0])))))
    #     neworder=left+list(indices[0])+complement
    #     newshape=tuple([self.shape[l] for l in left]+[np.prod([self.shape[l] for l in indices[0]])]+[self.shape[l] for l in complement])
    #     if len(indices)==1:        
    #         return np.reshape(np.transpose(self,neworder),newshape).view(Tensor)
    #     else:
    #         print(indices[1::])
    #         for otherindices in indices[1::]:
    #             for m in range(len(otherindices)):
    #                 cnt=0
    #                 indx=otherindices[m]
    #                 for n in sorted(indices[0]):
    #                     if n<indx:
    #                         cnt+=1

    #                 otherindices[m]-=(cnt-1)
    #         newshape=tuple([self.shape[l] for l in left]+[np.prod([self.shape[l] for l in indices[0]])]+[self.shape[l] for l in complement])

    #         print(indices[1::])
    #         out=np.reshape(np.transpose(self,neworder),newshape).view(Tensor)
    #         return out.merge(*indices[1:])


    def merge(self,*indices):
        """
        merges indices in the tuple ```indices```, respecting the order of the elements of ```indices```.
        for each list in ```indices```, the indices in this list are merged into a single index.
        The order of these merged indices is given by the order in which the lists are passed.
        All indices of self which are not in any of the lists in ```indices``` are transposed and placed in the following way:
        let ```comp``` be an ordered list (small to large) of the complementary indices to ```indices```, i.e. ```comb``` contains all indices not contained in ```indices```,i.e.
        ```comb=[c1,c2,c3,...]```. self is then first transposed into an index-order

        [sorted(all elements of comb which are smaller than min(indices[0]))],indices[0],[sorted(all elements of comb wich are smaller than min(indices[1]) and larger than min(indices[0])],indices[1],...,indices[-1],sorted([all elements of comb which have not yet been placed])]

        Parameters:
        ----------------------
        indices:   tuple of list of int
                   the indices which should be merged
        Returns:
        ----------------------
        Tensor:    the tensor with merged indices
        
        """
        flatinds=list(indices[0])
        shape=np.array(self.shape)        
        [flatinds.extend(n) for n in indices[1:]]
        if len(set(flatinds))<len(flatinds):
            elems,cnts=np.unique(np.asarray(flatinds),return_counts=True)
            raise ValueError('Tensor.merge(): indices {0} are appearing more than once!'.format(elems[cnts>1]))

        if max(flatinds)>=len(self.shape):
            raise ValueError("Tensor.merge(indices): max(indices)>=len(shape)")
        complement=sorted(list(set(range(len(self.shape))).difference(set(flatinds))))
        complist=[list() for n in range(len(indices)+1)]
        for c in complement:
            n=0
            while (n < len(indices)) and (c > min(indices[n])):
                n+=1
            complist[n].append(c)
        neworder=complist[0]
        newshape=list(shape[complist[0]])
        for n in range(len(indices)):
            neworder.extend(indices[n])
            neworder.extend(complist[n+1])
            newshape.append(np.prod(shape[indices[n]]))
            newshape.extend(list(shape[complist[n+1]]))

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
            u,s,v=np.linalg.svd(self,*args,**kwargs)
            return u,s.view(type(self)),v
        except LinAlgError:
            [q,r]=temp.qr()
            r[np.abs(r)<r_thresh]=0.0
            u_,s,v=r.svd(*args,**kwargs)
            u=q.dot(u_).view(Tensor)
            warnings.warn('svd: prepareTruncate caught a LinAlgError with dir>0')
            return u.view(type(self)),s.view(type(self)),v.view(type(self))

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

    @staticmethod
    def prepareTensor(tensor,direction,fixphase=None):
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
            temp=tensor.merge([0,2],[1])
            temp1=np.reshape(np.transpose(tensor,(0,2,1)),(d*l1,l2))
            print(temp-temp1)
            q,r=temp.qr()
            #fix the phase freedom of the qr
            if fixphase=='r':
                phase=np.angle(np.diag(r)).view(type(tensor))
                unit=np.diag(np.exp(-1j*phase)).view(type(tensor))
                q=q.dot(herm(unit)).view(type(tensor))
                r=unit.dot(r).view(type(tensor))
            if fixphase=='q':
                phase=np.angle(np.diag(q)).view(type(tensor))
                unit=np.diag(np.exp(-1j*phase)).view(type(tensor))
                q=q.dot(unit).view(type(tensor))
                r=herm(unit).dot(r).view(type(tensor))
    
            #normalize the bond matrix
            Z=np.linalg.norm(r)        
            r/=Z
            [size1,size2]=q.shape
            out=np.transpose(np.reshape(q,(l1,d,size2)),(1,2,0))
        elif direction in (-1,'r','right'):
            temp=np.conj(tensor.merge([1,2],[0]))
            #temp=np.conjugate(np.transpose(np.reshape(tensor,(l1,l2*d)),(1,0)))

            q,r_=temp.qr()
            #fix the phase freedom of the qr        
            if fixphase=='r':
                phase=np.angle(np.diag(r_)).view(type(tensor))
                unit=np.diag(np.exp(-1j*phase)).view(type(tensor))
                q=q.dot(herm(unit)).view(type(tensor))
                r_=unit.dot(r_).view(type(tensor))
    
            if fixphase=='q':
                phase=np.angle(np.diag(q)).view(type(tensor))
                unit=np.diag(np.exp(-1j*phase)).view(type(tensor))
                q=q.dot(unit).view(type(tensor))
                r_=herm(unit).dot(r_).view(type(tensor))
    
            [size1,size2]=q.shape
            out=np.conjugate(np.transpose(np.reshape(q,(l2,d,size2)),(2,0,1)))            
            r=np.conjugate(np.transpose(r_,(1,0)))
            #normalize the bond matrix
            Z=np.linalg.norm(r)
            r/=Z
        else:
            raise ValueError("unkown value {} for input parameter direction".format(direction))
        return out,r,Z



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
        #print("in prepareTruncate: thresh=",thresh,"D=",D)
        assert(direction!=0),'do NOT use direction=0!'
        [l1,l2,d]=tensor.shape
        if direction in (1,'l','left'):
            temp=tensor.merge([2,0],[1])
            #temp=np.reshape(np.transpose(tensor,(2,0,1)),(d*l1,l2))
            [u,s,v]=temp.svd(full_matrices=False)
            Z=np.linalg.norm(s)            
            if thresh>1E-16:
                s=s[s>thresh]
    
            if D!=None:
                if D<len(s):
                    warnings.warn('Tensors.prepareTruncate with dir={0}: desired thresh imcompatible with max bond dimension; truncating'.format(direction),stacklevel=3)
                s=s[0:min(D,len(s))]
                u=u[:,0:len(s)]
                v=v[0:len(s),:]
            elif D==None:
                u=u[:,0:len(s)]
                v=v[0:len(s),:]
                s=s[0:len(s)]
    
            s/=np.linalg.norm(s)            
            [size1,size2]=u.shape
            out=np.transpose(np.reshape(u,(d,l1,size2)),(1,2,0))
            return out,s.view(type(tensor)),v,Z

        if direction in (-1,'r','right'):
            temp=tensor.merge([0],[1,2])
            #temp=np.reshape(tensor,(l1,d*l2))
            [u,s,v]=temp.svd(full_matrices=False)
            Z=np.linalg.norm(s)                        
            if thresh>1E-16:
                s=s[s>thresh]
            if D!=None:
                if D<len(s):
                    warnings.warn('Tensors.prepareTruncate with dir={0}: desired thresh imcompatible with max bond dimension; truncating'.format(direction),stacklevel=3)                    
                s=s[0:min(D,len(s))]
                u=u[:,0:len(s)]
                v=v[0:len(s),:]
    
            elif D==None:
                u=u[:,0:len(s)]
                v=v[0:len(s),:]
                s=s[0:len(s)]
    
    
            s/=np.linalg.norm(s)            
            [size1,size2]=v.shape
            out=np.reshape(v,(size1,l2,d))
        s=s.view(type(tensor))        
        return u,s,out,Z
