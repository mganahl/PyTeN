import numpy as np
cimport numpy as np
from scipy.special import binom
from sys import stdout
import sys
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.pair cimport pair
import time
#note: total run time for building+ doiagonalization using lil_matrix is slower by a factor of 40 than using python lists
#and building csc_matrix from that
#filling a csc_matrix is fastest by doing it from the constructor (rather than filling in element by element)

from scipy.sparse import lil_matrix,csc_matrix
from cython.operator cimport dereference as deref, preincrement as inc

ctypedef np.uint64_t ITYPE_t
ctypedef np.float64_t DTYPE_t




"""
generate a list of uint64 representing the U1 and Z2 symmetric basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration; entries of returned list are sorted in ascending order

"""
cdef binarybasisU1Z2TI_(int N,vector[long unsigned int]& basis):
    cdef int N_up=int(N/2)
    cdef int dim
    dim = int(binom(N, N_up)/2)
    cdef long unsigned int count,state
    cdef int k,nleft,p
    basis.resize(dim)
    #cdef np.ndarray[ITYPE_t, ndim=1] basis = np.empty([dim],dtype=np.uint64)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis
    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    '''state!= (2**N-1)-(2**(N-N_up)-1):'''
    while count<(dim-1):
        basis[count] = state
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state



"""
generate a list of uint64 representing the U1 and Z2 symmetric basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration; entries of returned list are sorted in ascending order

"""
def binarybasisU1Z2(int N):
    cdef int N_up=int(N/2)
    cdef int dim
    dim = int(binom(N, N_up)/2)
    cdef long unsigned int count,state
    cdef int k,nleft,p

    cdef np.ndarray[ITYPE_t, ndim=1] basis = np.empty([dim],dtype=np.uint64)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis
    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    '''state!= (2**N-1)-(2**(N-N_up)-1):'''
    while count<(dim-1):
        basis[count] = state
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state
    return basis


"""
generate a list of uint64 representing the U1 and Z2 symmetric basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration; entries of returned list are sorted in ascending order

"""
cdef binarybasisU1Z2_(int N,vector[long unsigned int]& basis):
    cdef int N_up=int(N/2)
    cdef int dim
    dim = int(binom(N, N_up)/2)
    cdef long unsigned int count,state
    cdef int k,nleft,p
    basis.resize(dim)
    #cdef np.ndarray[ITYPE_t, ndim=1] basis = np.empty([dim],dtype=np.uint64)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis
    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    '''state!= (2**N-1)-(2**(N-N_up)-1):'''
    while count<(dim-1):
        basis[count] = state
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state

"""
generate a list of uint64 representing U1 symmetric basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration
a spin configuration; entries of returned list are sorted in ascending order
"""
def binarybasisU1(N, N_up):
    cdef int dim
    dim = int(binom(N, N_up))
    cdef long unsigned int count,state
    cdef int k,nleft,p

    cdef np.ndarray[ITYPE_t, ndim=1] basis = np.empty([dim],dtype=np.uint64)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis
    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    '''state!= (2**N-1)-(2**(N-N_up)-1):'''
    while count<(dim-1):
        basis[count] = state
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state
    return basis

"""
generate a list of uint64 representing U1 symmetric basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration
a spin configuration; entries of returned list are sorted in ascending order
"""
cdef binarybasisU1_(N, N_up,vector[long unsigned int]& basis):
    cdef int dim
    dim = int(binom(N, N_up))
    cdef long unsigned int count,state
    cdef int k,nleft,p

    basis.resize(dim)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis
    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    '''state!= (2**N-1)-(2**(N-N_up)-1):'''
    while count<(dim-1):
        basis[count] = state
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state

"""
generate a list of uint64 representing the basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins. Uses recursive function call
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration
np.uint64(n+2**p))
"""
def binarybasisrecursive(int N,int Nup):
    cdef int p
    cdef list basis=[]
    cdef list init=[np.uint64(0)]
    cdef list rest
    cdef long unsigned int n
    if Nup==0:
        return init
    else:
        for p in range(Nup-1,N):
            rest=binarybasisrecursive(p,Nup-1)
            for n in rest:
                basis.append(n+setBit(0,p))
        return basis


def flip(long unsigned int state,int N):
    return flipSpins(state,N)

"""
flips all spins of a binary representation state
"""
cdef long unsigned int flipSpins(unsigned long int state,unsigned int N):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    cdef unsigned int p
    for p in range(N):
        state=flipBit(state,p)
    return state

    
"""
flips a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int flipBit(unsigned long int b,unsigned int pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    if b&mask==0:#b(pos)==0
        return b|mask
    elif b&mask!=0:#b(pos)==1
        return b&(~mask)

"""
sets a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int setBit(long unsigned int b, unsigned int  pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    return mask|b


"""
sets a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int unsetBit(long unsigned int b, unsigned int  pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    return (~mask)&b

cdef int getBit(unsigned long int  b,unsigned int pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask = a << pos
    return int((b & mask)>0)


"""
calculates all non-zero matrix elements of the XXZ Hamiltonian on a grid "grid" with interactions
Jz and Jxy and for total number of N spins. "basis" is a list of unit64 numbers encoding the basis-states

grid (list() of list()) of length N: grid[n] is a list of neighbors of spin n

Jz,Jxy: Jz[n] and Jxy[n] is an array of the interaction and hopping parameters of all neighbors of spin n,
such that Jz[n][i] corresponds to the interaction of spin n with spin grid[n][i] (similar for Jxy)

RETURNS: inddiag,diag,nondiagindx,nondiagindy,nondiag

diag:   a list of non-zero diagonal element from the Jz Sz*Sz -part of the Hamiltonian
inddiag: a list indices of the non-zero matrix elements from the Sz*Sz part, such that H[inddiag[n],inddiag[n]]=diag[n]


nondiag:   a list of non-zero matrix elements form the Sx*Sx+Sy*Sy-part of the Hamiltonian
nondiagindx,nondiagindy: a list x- and y-indices of the non-zero values form the Jxy(Sx*Sx+Sy*Sy) part
                         such that H[nondiagindx[n],nondiagindy[n]]=nondiag[n]

"""
cdef XXZU1(np.ndarray[DTYPE_t, ndim=2] Jxy,np.ndarray[DTYPE_t, ndim=2] Jz,int N,vector[long unsigned int]& basis,grid):
    cdef long unsigned int state,newstate,N0
    cdef int s,p,nei
    cdef long unsigned int n
    cdef float sz,szsz
    cdef map[long unsigned int,long unsigned int] num2ind
    for n in range(basis.size()):
        num2ind[basis[n]]=n
    
    cdef np.ndarray[DTYPE_t, ndim=2] Jp=Jxy/2.0
    cdef list diag=[]
    cdef list inddiag=[]
    cdef list nondiag=[]
    cdef list nondiagindx=[]
    cdef list nondiagindy=[]
    N0=num2ind.size()
    basis.clear()
    t0=time.time()
    for it in num2ind:
        n=it.second
        if n%100000==0:
            stdout.write("\rbuilding Hamiltonian ... finished %2.2f percent, 100000 states took %2.4f seconds" %(100.0*n/N0,time.time()-t0))
            t0=time.time()
            stdout.flush()
        
        state=it.first
        szsz=0
        for s in range(N):
            sz=(getBit(state,s)-0.5)*2
            for p in range(len(grid[s])):
                nei=grid[s][p]
                szsz+=sz*(getBit(state,nei)-0.5)*2*Jz[s,p]/4.0
        if (abs(szsz)>1E-5):
            diag.append(szsz)
            inddiag.append(n)
        for s in range(N):
            for p in range(len(grid[s])):
                nei=grid[s][p]
                if getBit(state,s)!=getBit(state,nei):
                    newstate=flipBit(flipBit(state,s),nei)
                    nondiagindx.append(num2ind[newstate])
                    nondiagindy.append(n)
                    nondiag.append(Jp[s,p])
    print('')		    
    stdout.write("\rfinished building Hamiltonian")
    stdout.flush()


    print('')
    return inddiag,diag,nondiagindx,nondiagindy,nondiag


"""
calculates all non-zero matrix elements of the XXZ Hamiltonian on a grid "grid" with interactions
Jz and Jxy and for total number of N spins. "basis" is a list of unit64 numbers encoding the basis-states

grid (list() of list()) of length N: grid[n] is a list of neighbors of spin n

Jz,Jxy: Jz[n] and Jxy[n] is an array of the interaction and hopping parameters of all neighbors of spin n,
such that Jz[n][i] corresponds to the interaction of spin n with spin grid[n][i] (similar for Jxy)

RETURNS: inddiag,diag,nondiagindx,nondiagindy,nondiag

diag:   a list of non-zero diagonal element from the Jz Sz*Sz -part of the Hamiltonian
inddiag: a list indices of the non-zero matrix elements from the Sz*Sz part, such that H[inddiag[n],inddiag[n]]=diag[n]


nondiag:   a list of non-zero matrix elements form the Sx*Sx+Sy*Sy-part of the Hamiltonian
nondiagindx,nondiagindy: a list x- and y-indices of the non-zero values form the Jxy(Sx*Sx+Sy*Sy) part
                         such that H[nondiagindx[n],nondiagindy[n]]=nondiag[n]

"""
cdef XXZU1Z2(np.ndarray[DTYPE_t, ndim=2] Jxy,np.ndarray[DTYPE_t, ndim=2] Jz,int N,int symmetry,vector[long unsigned int]& basis,grid):

    if (symmetry!=1) and (symmetry!=(-1)):
        sys.exit('XXZED.py: XXZU1Z2(Jxy,Jz,N,symmetry,basis,grid): symmetry is not 1 or -1! ')
        
    cdef int Nup=int(N/2)
    cdef long unsigned int state,newstate,N0,maxstate

    cdef int s,p,nei
    cdef long unsigned int n
    cdef float sz,szsz,val
    cdef map[long unsigned int, DTYPE_t ] stateset
    cdef map[long unsigned int, DTYPE_t ].iterator setit
    cdef pair[long unsigned int, DTYPE_t ] ppair
    
    cdef map[long unsigned int,long unsigned int] num2ind
    for n in range(len(basis)):
        num2ind[basis[n]]=n
    
    cdef np.ndarray[DTYPE_t, ndim=2] Jp=Jxy/2.0
    cdef list diag=[]
    cdef list inddiag=[]
    cdef list nondiag=[]
    cdef list nondiagindx=[]
    cdef list nondiagindy=[]
    N0=num2ind.size()
    maxstate=basis[N0-1]    
    basis.clear()
    t0=time.time()
    for it in num2ind:
        n=it.second
        if n%100000==0:
            stdout.write("\rbuilding Hamiltonian ... finished %2.2f percent, 100000 states took %2.4f seconds" %(100.0*n/N0,time.time()-t0))
            t0=time.time()
            stdout.flush()
        
        state=it.first
        szsz=0
        for s in range(N):
            sz=(getBit(state,s)-0.5)*2
            for p in range(len(grid[s])):
                nei=grid[s][p]
                szsz+=sz*(getBit(state,nei)-0.5)*2*Jz[s,p]/4.0
        if (abs(szsz)>1E-5):
            diag.append(szsz)
            inddiag.append(n)
            
        stateset.clear()
        for s in range(N):
            for p in range(len(grid[s])):
                nei=grid[s][p]
                if getBit(state,s)!=getBit(state,nei):
                    newstate=flipBit(flipBit(state,s),nei)
                    ppair.first=newstate
                    ppair.second=Jp[s,p]
                    stateset.insert(ppair)

        while stateset.size()!=0:
            setit=stateset.begin()
            newstate=deref(setit).first
            val=deref(setit).second
            if newstate<=maxstate:
                nondiagindx.append(num2ind[newstate])
                nondiagindy.append(n)
                nondiag.append(val)

            elif newstate>maxstate:
                nondiagindx.append(num2ind[flipSpins(newstate,N)])
                nondiagindy.append(n)
                nondiag.append(symmetry*val)
            else:
                sys.exit('found a state that is not contained in the Hilbert-space')
            stateset.erase(setit)                        
    print('')		    
    stdout.write("\rfinished building Hamiltonian")    
    stdout.flush()
    print('')
    return inddiag,diag,nondiagindx,nondiagindy,nondiag    



def XXZSparseHam(np.ndarray[DTYPE_t, ndim=2] Jxy,np.ndarray[DTYPE_t, ndim=2] Jz,int N,int Nup,int Z2,grid):
    cdef vector[long unsigned int] basis
    cdef long unsigned int dim
    if ((Z2==1) or (Z2==-1)) and (Nup*2==N):
        print('                            doing Z2 symmetric diagonalization for Z2={0}, N={1}, Nup={2}                         '.format(Z2,N,Nup))        
        t1=time.time()
        binarybasisU1Z2_(N,basis)
        t2=time.time()        
        print('calculating {1} basis-states took {0} seconds'.format(t2-t1,len(basis)))
        dim=basis.size()
        t3=time.time()            
        inddiag,diag,nondiagindx,nondiagindy,nondiag=XXZU1Z2(Jxy,Jz,N,Z2,basis,grid)
        stdout.write("\rgenerating csc_matrix, be patient with me ...")
        stdout.flush()	
        Hsparse=csc_matrix((diag,(inddiag,inddiag)),shape=(dim,dim))+csc_matrix((nondiag,(nondiagindx,nondiagindy)),shape=(dim,dim))	
        stdout.write("\rgenerating csc_matrix, be patient with me ... done")
        stdout.flush()
        print('')
        t4=time.time()
        print('calculating sparse Hamiltonian took {0} seconds'.format(t4-t3))
        return Hsparse
    else:
        print('                            doing non-Z2-symmetric diagonalization for, N={1}, Nup={2}                             '.format(Z2,N,Nup))            
        t1=time.time()        
        binarybasisU1_(N,Nup,basis)
        t2=time.time()
        print('calculating {1} basis-states took {0} seconds'.format(t2-t1,len(basis)))    
        dim=basis.size()        
        t3=time.time()    
        inddiag,diag,nondiagindx,nondiagindy,nondiag=XXZU1(Jxy,Jz,N,basis,grid)
        stdout.write("\rgenerating csc_matrix, be patient with me ...")
        stdout.flush()	
        Hsparse=csc_matrix((diag,(inddiag,inddiag)),shape=(dim,dim))+csc_matrix((nondiag,(nondiagindx,nondiagindy)),shape=(dim,dim))
        stdout.write("\rgenerating csc_matrix, be patient with me ... done")
        stdout.flush()
        print('')
        t4=time.time()
        print('calculating sparse Hamiltonian took {0} seconds'.format(t4-t3))
        return Hsparse
    
def testbinops(unsigned long int b,int pos):
    print('b={2}, bit {0} of b={1}'.format(pos,getBit(b,pos),bin(b)))
    print('b before flipping bit {0}:'.format(pos),bin(b))
    print('b after flipping bit {0}:'.format(pos),bin(flipBit(b,pos)))
    print('b before setting bit {0}:'.format(pos),bin(b))
    print('b after setting bit {0}:'.format(pos),bin(setBit(b,pos)))
