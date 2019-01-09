#!/usr/bin/env python

#program for ground state calculations of the Heisenberg model in the thermodynamic limit
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import argparse
import numpy as np
import math
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib
import datetime
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('HeisIMPS.py: ground-state simulation for the infinite XXZ model using gradient optimization')    
    parser.add_argument('--dtype', help='type of the matrix (float)',type=str,default='float64')
    parser.add_argument('--D', help='initial MPS bond dimension (8)',type=int,default=8)
    parser.add_argument('--Dmax', help='maximum MPS bond dimension (32)',type=int,default=32)    
    parser.add_argument('--verbose', help='verbosity (1)',type=int,default=1)    
    parser.add_argument('--cp', help='do checkpointing at specified steps (0, no checkpointing)',type=int)
    parser.add_argument('--keep_cp',action='store_true',help='if this flag is set, keep the checkpoint files; otherwise, only the last checkpoint will be kept (False)')            
    parser.add_argument('--Jx', help='Jx intercation (-1.0)',type=float,default=-1.0)
    parser.add_argument('--Bz', help='magnetic field (1.0)',type=float,default=1.0)
    parser.add_argument('--load', help='load a file (None)',type=str)        
    parser.add_argument('--scaling',help='scaling of the initial MPS entries (0.5)',type=float,default=0.5)    
    parser.add_argument('--lgmrestol', help='lgmres tolerance for reduced hamiltonians (1E-12)',type=float,default=1E-12)
    parser.add_argument('--regaugetol', help='tolerance of eigensolver for finding left and right reduced DM (1E-12)',type=float,default=1E-12)
    parser.add_argument('--dt', help='time step; use negative imaginary for imaginary time evolution (-0.01j)',type=complex,default=-0.01j)
    parser.add_argument('--imax', help='maximum number of time steps (1000)',type=int,default=1000)
    parser.add_argument('--saveit', help='save the simulation every saveit iterations for checkpointing (10)',type=int,default=10)
    parser.add_argument('--filename', help='filename for output (TFI_ITEBD)',type=str,default='TFI_ITEBD')
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs (40)',type=int,default=40)
    parser.add_argument('--svd', help='do svd instead of polar decompostion for  guage matching',action="store_true")    
    parser.add_argument('--krylov_dim', help='number of Krylov vectors in lan (10)',type=int,default=10)
    parser.add_argument('--atol', help='absolute tolerance of RK45 and RK32 solver (1E-12)',type=float,default=1E-12)
    parser.add_argument('--rtol', help='relative tolerance of RK45 and RK32 solver (1E-12)',type=float,default=1E-6)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs (5)',type=int,default=6)    
    parser.add_argument('--Nmaxlgmres', help='Maximum number of lgmres steps (see scipy.sparse.linalg.lgmres help); total number is (Nmaxlgmres+1)*innermlgmres',type=int,default=40)
    args=parser.parse_args()
    date=datetime.datetime.now()
    today=str(date.year)+str(date.month)+str(date.day)
    d=2
    N=2
    filename=today+'_'+args.filename+'D{0}_Jx{1}_B{2}_dt{3}'.format(args.D,args.Jx,args.Bz,args.dt)
    root=os.getcwd()
    if os.path.exists(filename):
        print('folder',filename,'exists already. Resuming will likely overwrite existing data. Hit enter to confirm')
        input()
    elif not os.path.exists(filename):
        os.mkdir(filename)


    if args.dtype=='complex128':
        dtype=np.complex128
    elif args.dtype=='complex64':
        dtype=np.complex64
    elif args.dtype=='complex32':
        dtype=np.complex32
    elif args.dtype=='float64':
        dtype=np.float64
    elif args.dtype=='float32':
        dtype=np.float32
    elif args.dtype=='float16':
        dtype=np.float16
    else:
        sys.exit('unknown type args.dtype={0}'.format(args.dtype))
    Jx=args.Jx*np.ones(N)
    B=args.Bz*np.ones(N)
    mpo=H.TFI(Jx,B,obc=False,dtype=dtype)

    dt=(-1j*args.dt)
    if np.imag(dt)<1E-12:
        dt=dtype(np.real(dt))
    if args.load==None:
        os.chdir(filename)        
        try:
            mps=mpslib.MPS.random(N=N,D=args.D,d=d,obc=False,dtype=dtype)  #initialize a random MPS with bond dimension D'=10
            #normalize the state by sweeping the orthogonalizty center once back and forth through the system
            mps.regauge(gauge='right')
            iTEBD=en.ITEBDengine(mps,mpo,args.filename)
            iTEBD.doITEBD(dt,numsteps=args.imax,Dmax=args.Dmax,recanonize=True,regaugetol=args.regaugetol,ncv=args.ncv,numeig=1,
                          tr_thresh=1E-10,verbose=args.verbose,cp=args.cp,keep_cp=args.keep_cp)            
        except TypeError:
            dtype=complex            
            mps=mpslib.MPS.random(N=N,D=args.D,d=d,obc=False,dtype=dtype)  #initialize a random MPS with bond dimension D'=10
            #normalize the state by sweeping the orthogonalizty center once back and forth through the system
            mps.regauge(gauge='right')
            iTEBD=en.ITEBDengine(mps,mpo,args.filename)
            iTEBD.doITEBD(dt,numsteps=args.imax,Dmax=args.Dmax,recanonize=True,regaugetol=args.regaugetol,ncv=args.ncv,numeig=1,
                          tr_thresh=1E-10,verbose=args.verbose,cp=args.cp,keep_cp=args.keep_cp)            
    else:
        pass
        
    [Gamma,lam,r]=mf.regauge(iMPS._B,gauge='symmetric',tol=args.regaugetol)
    print('Schmidt values, normalization')
    print(lam,np.sum(lam**2))
    print('normalized and rescaled natural logarithm of Schmidt values')    
    loglam=np.log(lam)
    print((loglam-loglam[0])/(loglam[1]-loglam[0]))
    np.save('mps',iMPS._A)
    np.save('lam',lam)    