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
comm=lambda x,y:np.do(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('HeisIMPS.py: ground-state simulation for the infinite XXZ model using gradient optimization')    
    parser.add_argument('--dtype', help='type of the matrix (float64)',type=str,default='float64')
    parser.add_argument('--D', help='MPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--cp', help='do checkpointing at specified steps (0, no checkpointing)',type=int,default=0)
    parser.add_argument('--solver', help='saolver type;  us lan or ar (ar)',type=str,default='ar')        
    parser.add_argument('--Jx', help='Jx intercation (-1.0)',type=float,default=-1.0)
    parser.add_argument('--Bz', help='magnetic field (1.0)',type=float,default=1.0)
    parser.add_argument('--scaling',help='scaling of the initial MPS entries (0.5)',type=float,default=0.5)    
    parser.add_argument('--lgmrestol', help='lgmres tolerance for reduced hamiltonians (1E-12)',type=float,default=1E-12)
    parser.add_argument('--regaugetol', help='tolerance of eigensolver for finding left and right reduced DM (1E-12)',type=float,default=1E-12)
    parser.add_argument('--epsilon', help='desired convergence of the gradient (1E-10)',type=float,default=1E-10)
    parser.add_argument('--landelta', help='lanczos stopping criterion (1E-10)',type=float,default=1E-10)    
    parser.add_argument('--imax', help='maximum number of iterations (100)',type=int,default=100)
    parser.add_argument('--saveit', help='save the simulation every saveit iterations for checkpointing (10)',type=int,default=10)
    parser.add_argument('--filename', help='filename for output (_TFI_VUMPS)',type=str,default='TFI_VUMPS')
    parser.add_argument('--load', help='load a file (None)',type=str)    
    parser.add_argument('--seed', help='seed for initialization of Q and R matrices',type=int)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs (5)',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs (20)',type=int,default=20)
    parser.add_argument('--svd', help='do svd instead of polar decompostion for  guage matching',action="store_true")    
    parser.add_argument('--arncv', help='number of Krylov vectors in arnoldi (20)',type=int,default=20)
    parser.add_argument('--arnumvecs', help='number of exciteds states (1)',type=int,default=1)    
    parser.add_argument('--artol', help='tolerance of the arnoldi sovler (1E-12)',type=float,default=1E-12)    
    parser.add_argument('--Nmaxlgmres', help='Maximum number of lgmres steps (see scipy.sparse.linalg.lgmres help); total number is (Nmaxlgmres+1)*innermlgmres',type=int,default=50)
    parser.add_argument('--outerklgmres', help='Number of vectors to carry between inner GMRES iterations. According to [R271], good values are in the range of 1...3. However, note that if you want to use the additional vectors to accelerate solving multiple similar problems, larger values may be beneficial (from scipy.sparse.linalg.lgmres manual)',type=int,default=10)
    parser.add_argument('--innermlgmres', help='Number of inner GMRES iterations per each outer iteration (from scipy.sparse.linalg.lgmres manual)',type=int,default=30)
    args=parser.parse_args()
    date=datetime.datetime.now()
    today=str(date.year)+str(date.month)+str(date.day)
    
    d=2
    N=1
    filename=today+'_'+args.filename+'D{0}_Jx{1}_B{2}'.format(args.D,args.Jx,args.Bz)
    root=os.getcwd()
    if os.path.exists(filename):
        print('folder',filename,'exists already. Resuming will likely overwrite existing data. Hit enter to confirm')
        input()
    elif not os.path.exists(filename):
        os.mkdir(filename)

    
    Jx=args.Jx*np.ones(N)
    B=args.Bz*np.ones(N)
    mpo=H.TFI(Jx,B,False,dtype=args.dtype)
    
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
    mps=mpslib.MPS.random(N=N,D=args.D,d=d,obc=False,dtype=dtype)  #initialize a random MPS with bond dimension D'=10
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.regauge(gauge='right')
    if args.load==None:
        os.chdir(filename)        
        iMPS=en.VUMPSengine(mps=mps,mpo=mpo,filename=args.filename,regaugetol=args.regaugetol,ncv=args.ncv,numeig=args.numeig)
        iMPS.optimize(Nmax=args.imax,epsilon=args.epsilon,
                      regaugetol=args.regaugetol,
                      ncv=args.ncv,
                      numeig=args.numeig,
                      lgmrestol=args.lgmrestol,
                      Nmaxlgmres=args.Nmaxlgmres,
                      artol=args.artol,arnumvecs=args.arnumvecs,
                      arncv=args.arncv,svd=args.svd,cp=args.cp,
                      solver=args.solver.upper(),keep_cp=False,landelta=args.landelta)
    else:
        iMPS=en.VUMPSengine.load(args.load)
        os.chdir(filename)
        iMPS.optimize(Nmax=args.imax,epsilon=args.epsilon,
                      regaugetol=args.regaugetol,
                      ncv=args.ncv,
                      numeig=args.numeig,
                      lgmrestol=args.lgmrestol,
                      Nmaxlgmres=args.Nmaxlgmres,
                      artol=args.artol,arnumvecs=args.arnumvecs,
                      arncv=args.arncv,svd=args.svd,cp=args.cp,
                      solver=args.solver.upper(),keep_cp=False,landelta=args.landelta)
        
        # iMPS.optimize(args.imax,args.epsilon,args.lgmrestol,args.Nmaxlgmres,artol=args.artol,arnumvecs=args.arnumvecs,\
        #               arncv=args.arncv,svd=args.svd,cp=args.cp,solver=args.solver.upper(),keep_cp=False)
        
    [Gamma,lam,r]=mf.regauge(iMPS._A,gauge='symmetric',tol=args.regaugetol)
    print('Schmidt values, normalization')
    print(lam,np.sum(lam**2))
    print('normalized and rescaled natural logarithm of Schmidt values')    
    loglam=np.log(lam)
    print((loglam-loglam[0])/(loglam[1]-loglam[0]))
    np.save('mps',iMPS._A)
    np.save('lam',lam)    
