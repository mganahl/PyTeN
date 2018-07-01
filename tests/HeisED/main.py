#!/usr/bin/env python
import functools as fct
import numpy as np
import XXZED as ed
import scipy as sp
import time,sys
import LanczosEngine as lanEn
import argparse
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

from scipy.sparse import csc_matrix

if __name__ == "__main__":
    """
    ED calculation of groundstate of XXZ model. You can use either arnoldi (scipy implementation) 
    or a lanczos (my implementation). The latter is faster for large matrices
    """
    parser = argparse.ArgumentParser('HeisED')
    parser.add_argument('--N',help='Number of sites (10)',type=int,default=10)
    parser.add_argument('--Nup',help='Number of up-spins (5)',type=int,default=5)    
    parser.add_argument('--Z2',help='Z2 symmetry; use (-1),1 for (anti)-symmetric ground-states; Z2!=-1 or Z2!=1 uses no Z2 symmetry (more costly) (-1)',type=int,default=-1)    
    parser.add_argument('--LAN', help='use lanczos (False)',action='store_true' )
    parser.add_argument('--AR', help='use arnoldi (False)',action='store_true')
    parser.add_argument('--pbc', help='boundadry condition (False)',action='store_true')
    parser.add_argument('--save', help='save sparse Hamiltonian (not stored)',type=str,default=None)    
    parser.add_argument('--Jxy',help='Jxy coupling (1.0)',type=float,default=1.0)
    parser.add_argument('--Jz',help='Jz couplong (1.0)',type=float,default=1.0)
    args=parser.parse_args()    
    N=args.N #system size
    Nup=args.Nup #number of up-spins
    Jz=args.Jz 
    Jxy=args.Jxy
    Z2=args.Z2
    LAN=args.LAN
    AR=args.AR
    if (LAN==True) and (AR==True):
        print('--LAN and --AR are set; unset one of them')
        sys.exit()
    if (LAN==False) and (AR==False):
        print('no solver flag set; use --LAN or --AR')
        sys.exit()

        
    if not ((args.LAN==False) and (args.AR==False) and (args.save==None)):
        if args.save!=None:
            filename=args.save+'XXZsparseN{0}Nup{1}Jz{2}Jxy{3}'.format(N,Nup,Jz,Jxy)
        print('###########################    running ED for N={0}, Nup={1}, Jz={2}, Jxy={3}   #############################'.format(N,Nup,Jz,Jxy))
            
        #grid: a list of length N; grid[n] is a list of neighbors of spin n
        grid=[None]*N
        #Jzar,Jxyar: a list of length N; Jz[n] and Jxy[n] is an array of the interaction and hopping parameters of all neighbors of spin n,
        #such that Jz[n][i] corresponds to the interaction of spin n with spin grid[n][i]
        Jzar=[0.0]*N
        Jxyar=[0.0]*N
        for n in range(N-1):
            grid[n]=[n+1]
            Jzar[n]=np.asarray([Jz])
            Jxyar[n]=np.asarray([Jxy])
        if args.pbc:
            grid[N-1]=[0]
            Jzar[N-1]=np.asarray([1.0])
            Jxyar[N-1]=np.asarray([1.0])
        else:
            grid[N-1]=[]
            Jzar[N-1]=np.asarray([0.0])
            Jxyar[N-1]=np.asarray([0.0])
            
        Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
        t0=time.time()
        Hsparse=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid)
        
        if args.save!=None:
            sp.sparse.save_npz(filename,Hsparse)
        t1=time.time()
        if AR==True:
            e,v=sp.sparse.linalg.eigsh(Hsparse,k=2,which='SA',maxiter=1000000,tol=1E-5,v0=None,ncv=40)
            t2=time.time()        
        
        if LAN==True:
            def matvec(mat,vec):
                return mat.dot(vec)
            mv=fct.partial(matvec,*[Hsparse])
            lan=lanEn.LanczosEngine(mv,np.dot,np.zeros,Ndiag=10,ncv=500,numeig=1,delta=1E-8,deltaEta=1E-10)
            e,v,conv=lan.__simulate__(np.random.rand(Hsparse.shape[1]),verbose=False)
            t2=time.time()
        if (AR==True) or (LAN==True):        
            print('sparse diagonalization took {0} seconds'.format(t2-t1))
        t3=time.time()        
        print('total runtime: {0} seconds'.format(t3-t0))
        if args.save!=None:
            print('sparse Hamiltonian has been stored to disc in {0}.npz'.format(filename))
        print('lowest energies:')
        print(e)
    
