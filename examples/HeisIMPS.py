#!/usr/bin/env python

#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code
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
import matplotlib.pyplot as plt
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":
    
    plt.ion()
    parser = argparse.ArgumentParser('HeisIMPS.py')
    parser.add_argument('--type', help='matrix type (complex128)',type=str,default='complex128')
    parser.add_argument('--dev', help='run __simulatetest__()',action='store_true')
    parser.add_argument('--D', help='cMPS bond dimension (8)',type=int,default=8)
    parser.add_argument('--Jz', help='Sz-Sz intercation (1.0)',type=float,default=1.0)
    parser.add_argument('--B', help='magnetic field (0.0)',type=float,default=0.0)
    parser.add_argument('--rescalingfactor',help='rescaling factor by which time step is rescaled if norm increase is detected (2.0)',type=float,default=2.0)
    parser.add_argument('--normtolerance',help='tolerance of relative normincrease (0.1)',type=float,default=0.1)
    parser.add_argument('--alpha',help='time step for imaginary time evolution (0.05)',type=float,default=0.1)
    parser.add_argument('--alphas', nargs='+',help='list of time steps for imaginary time evolution',type=float)
    parser.add_argument('--normgrads', nargs='+',help='list of length N=len(dts) (see above); if norm(xdot)<nxdots[i], use dts[i] for imaginary time evolution',type=float)
    parser.add_argument('--lgmrestol', help='lgmres tolerance for reduced hamiltonians (1E-10)',type=float,default=1E-6)
    parser.add_argument('--regaugetol', help='tolerance of eigensolver for finding left and right reduced DM (1E-10)',type=float,default=1E-6)
    parser.add_argument('--epsilon', help='desired convergence of the state (1E-5)',type=float,default=1E-6)
    parser.add_argument('--imax', help='maximum number of iterations (20000)',type=int,default=20000)
    parser.add_argument('--saveit', help='save the simulation every saveit iterations for checkpointing (10)',type=int,default=10)
    parser.add_argument('--nreset', help='number of steps at each level of the adaptive dt refinement (10)',type=int,default=10)
    parser.add_argument('--filename', help='filename for output (_cMPSoptnew)',type=str,default='_HeisIMPS')
    parser.add_argument('--seed', help='seed for initialization of Q and R matrices',type=int)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs',type=int,default=20)
    parser.add_argument('--Nmaxlgmres', help='Maximum number of lgmres steps (see scipy.sparse.linalg.lgmres help); total number is (Nmaxlgmres+1)*innermlgmres',type=int,default=50)
    parser.add_argument('--outerklgmres', help='Number of vectors to carry between inner GMRES iterations. According to [R271], good values are in the range of 1...3. However, note that if you want to use the additional vectors to accelerate solving multiple similar problems, larger values may be beneficial (from scipy.sparse.linalg.lgmres manual)',type=int,default=10)
    parser.add_argument('--innermlgmres', help='Number of inner GMRES iterations per each outer iteration (from scipy.sparse.linalg.lgmres manual)',type=int,default=30)
    args=parser.parse_args()
    if (args.alpha==None):
        print ('please enter value for --alpha')
        sys.exit()

    if (args.alphas!=None) and (args.normgrads==None):
        print ('please enter values for --normgrads')
        sys.exit()

    if (args.alphas!=None) and (args.normgrads!=None):
        if len(args.alphas)!= len(args.normgrads):
            print ('please enter same number of values for --alphas and --normgrads')
            sys.exit()



    args=parser.parse_args()
    d=2
    N=1
    Jz=args.Jz*np.ones(N)
    Jxy=np.ones(N)
    B=args.B*np.ones(N)
    mpo=H.XXZ(Jz,Jxy,B,False)[0]
    print (mpo.shape)

    #mps=(np.random.rand(args.D,args.D,2)-0.5+1j*(np.random.rand(args.D,args.D,2)-0.5))*0.5
    if args.type=='complex128':
        tensor=(np.random.rand(args.D,args.D,d)-0.5)*0.9+1j*(np.random.rand(args.D,args.D,d)-0.5)*0.9
        dtype=complex
    elif args.type=='float64':
        tensor=(np.random.rand(args.D,args.D,d)-0.5)*0.9
        dtype=float        
    else:
        sys.exit('unknown type args.type={0}'.format(args.type))
    
        
    filename=args.filename+'D{0}_Jz{1}_B{2}'.format(args.D,args.Jz,args.B)
    [mps,lam]=mf.regauge(tensor,gauge='left',tol=args.regaugetol)
    iMPS=en.HomogeneousIMPSengine(args.imax,mps,mpo,args.filename,args.alpha,args.alphas,args.normgrads,dtype,args.rescalingfactor,\
                                  args.nreset,args.normtolerance,args.epsilon,args.regaugetol,args.lgmrestol,args.ncv,args.numeig,\
                                  args.Nmaxlgmres)

    if not args.dev:
        iMPS.__simulate__()
    elif args.dev:
        iMPS.__simulatetest__(Nmax=args.imax,Econv=1E-10,arnolditol=1E-10,arnoldinumvecs=1,arnoldincv=100)
    np.save(args.filename,iMPS._mps)
    print()
