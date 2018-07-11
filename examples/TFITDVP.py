#!/usr/bin/env python
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)
from sys import stdout
import numpy as np

import lib.mpslib.mpsfunctions as mf
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as Hams
import lib.mpslib.mps as mpslib

import argparse
herm=lambda x:np.conj(np.transpose(x))
if __name__ == "__main__":


    parser = argparse.ArgumentParser('HeisTDVP.py: ground-state simulation for the infinite XXZ model using TDVP imaginary time evolution')
    parser.add_argument('--D', help='cMPS bond dimension (8)',type=int,default=8)
    parser.add_argument('--dtype', help='type of the matrix (float)',type=str,default='float')    
    parser.add_argument('--Jx', help='Jx intercation (1.0)',type=float,default=1.0)
    parser.add_argument('--B', help='magnetic field (0.5)',type=float,default=0.5)
    parser.add_argument('--dt',help='time step for imaginary time evolution (0.05)',type=float,default=0.005)
    parser.add_argument('--regaugetol', help='tolerance of eigensolver for finding left and right reduced DM (1E-10)',type=float,default=1E-10)
    parser.add_argument('--lgmrestol', help='lgmres tolerance for reduced hamiltonians (1E-10)',type=float,default=1E-10)
    parser.add_argument('--epsilon', help='desired convergence of the state (1E-5)',type=float,default=1E-6)
    parser.add_argument('--imax', help='maximum number of iterations (20000)',type=int,default=20000)
    parser.add_argument('--filename', help='filename for output (_cMPSoptnew)',type=str,default='_HeisTDVP')
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs',type=int,default=20)
    args=parser.parse_args()
    #run in left mps gauge and left mps-tangent gauge
    
    N=2
    d=2
    Jx=args.Jx*np.ones(N)
    B=args.B*np.ones(N)
    mpo=Hams.TFI(Jx,B,obc=True)
    
    if args.dtype=='complex':
        tensor=(np.random.rand(args.D,args.D,d)-0.5)*0.9+1j*(np.random.rand(args.D,args.D,d)-0.5)*0.9
        dtype=complex
    elif args.dtype=='float':
        tensor=(np.random.rand(args.D,args.D,d)-0.5)*0.9
        dtype=float        
    else:
        sys.exit('unknown type args.dtype={0}'.format(args.dtype))


    [gamma,lam]=mf.regauge(tensor,gauge='left',tol=args.regaugetol)

    A=np.copy(gamma)

    l=np.eye(args.D)
    converged=False
    it=0

    rold=np.random.rand(args.D*args.D)*0.1
    kold=np.random.rand(args.D*args.D)*0.1


    while converged==False:
        it=it+1
        [etar,vr,numeig]=mf.TMeigs(A,direction=-1,numeig=args.numeig,init=rold,nmax=50000,tolerance=args.regaugetol,ncv=args.ncv)
        A=A/np.sqrt(np.real(etar))
        r=np.reshape(vr,(args.D,args.D))
        r=r/np.trace(r)

        if dtype==float:
            r=np.real(r+herm(r))/2.0
        if dtype==complex:
            r=(r+herm(r))/2.0
        [Bx,h,normxopt,kold]=mf.TDVPupdate(r,A,mpo,kold,tol=args.lgmrestol)
        A=A-Bx*args.dt
        [A,y]=mf.regauge(A,gauge='left',tol=args.regaugetol,ncv=args.ncv)
        [gamma,lam,trunc]=mf.regauge(A,gauge='symmetric',tol=args.regaugetol,ncv=args.ncv)
        stdout.write("\rit %i: ||xdot||=%.16f, h=%.12f, dt=%.4f" %(it,normxopt,np.real(h),args.dt))
        stdout.flush()

        if normxopt<args.epsilon:
           converged=True

        if it>args.imax:
            converged=True
    np.save(args.filename,A)
