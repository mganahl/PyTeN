#!/usr/bin/env python3
import numpy as np
import time
import pickle
import lib.mpslib.SimContainer as SCT
import lib.mpslib.TensorNetwork as TN
import lib.mpslib.MPO as MPO
import os

def run_bench(N, D,
              dtype,
              prefix='DMRG_benchmark',
              save=True,
              Nsweeps=2,
              ncv=100,
              verbose=1):
    """
    runs a DMRG benchmark and produces some files with walltimes
    
    Parameters:
    --------------------
    N:              int
                    system size
    D:              int
                    bond dimension
    prefix:         str 
                    the file prefix for the benchmark files
    compile_graph:  bool 
                    compile the graph
    save:           bool 
                    save files 
    Nsweeps:        int 
                    number of DMRG sweeps
    ncv:            int 
                    number of krylov vectors in DMRG 
    verbose:        int 
                    verbosity flag
    """
    if dtype==np.float64:
        the_dtype='float64'
    if dtype==np.float32:
        the_dtype='float32'
    if dtype==np.complex128:
        the_dtype='complex128'
    if dtype==np.complex64:
        the_dtype='complex64'
    
    lan_times=[]
    QR_times=[]
    add_layer_times=[]
    num_lanczos=[]
    def walltime_log(lan=[],QR=[],add_layer=[],num_lan=[]):
        lan_times.extend(lan)
        QR_times.extend(QR)
        add_layer_times.extend(add_layer)
        num_lanczos.extend(num_lan)
    
    mps=TN.FiniteMPS.random(d=[2]*N, D=[D]*(N-1), dtype=dtype)
    mps.position(0)
    mps.position(N)
    mps.position(0)    
    mpo=MPO.FiniteXXZ(Jz=np.ones([N-1]),
                      Jxy=np.ones([N-1]),
                      Bz=np.zeros([N]),
                      dtype=dtype)
    dmrg=SCT.FiniteDMRGEngine(mps,mpo)
    t1=time.time()
    dmrg.run_one_site(verbose=verbose,
                      Nsweeps=Nsweeps,
                      ncv=ncv,
                      precision=1E-200,
                      solver='lan',
                      walltime_log=walltime_log)
    out = {'lanczos':lan_times,
           'QR': QR_times,
           'add_layer': add_layer_times,
           'num_lanczos': num_lanczos,
           'total': time.time()-t1
    }
    if save:
        with open(prefix+'_N_{1}_D_{0}_dtype_{2}_ncv_{3}.pickle'.format(D,N,the_dtype,ncv),'wb') as f:
            pickle.dump(out,f)
            
    return out


def benchmark_1(prefix):
    """
    runs a DMRG benchmark and produces some files with walltimes
    
    Parameters:
    --------------------
    prefix:   str 
              the file prefix for the benchmark files
    """
    N=32 #system size
    Ds=[32, 64, 128, 256, 512, 1024]#bond dims
    Nsweeps=6#number of sweeps
    ncv=10#number of krylov vectors
    dtype = np.float64
    res_compiled = {D: run_bench(N=N, D=D, dtype=dtype,
                                    prefix=prefix,
                                    save=True,
                                    Nsweeps=Nsweeps, ncv=ncv)     for D in Ds}        
def benchmark_2(prefix):
    """
    runs a DMRG benchmark and produces some files with walltimes
    
    Parameters:
    --------------------
    prefix:   str 
              the file prefix for the benchmark files
    """
    
    N=64 #system size
    Ds=[64, 128, 256, 512, 1024] #bond dims
    Nsweeps=6 #number of sweeps
    ncv=10 #number of krylov vectors
    dtype = np.float64
    res_compiled = {D: run_bench(N=N, D=D, dtype=dtype,
                                    prefix=prefix,
                                    save=True,
                                    Nsweeps=Nsweeps, ncv=ncv)     for D in Ds}
    
def benchmark_3(prefix):
    """
    runs a DMRG benchmark and produces some files with walltimes
    
    Parameters:
    --------------------
    prefix:   str 
              the file prefix for the benchmark files
    """
    
    N=128 #system size
    Ds=[32,64, 128, 256, 512]#, 1024]#bond dims
    Nsweeps=6#number of sweeps
    ncv=10#number of krylov vectors
    dtype = np.float64
    res_compiled = {D: run_bench(N=N, D=D, dtype=dtype,
                                    prefix=prefix,
                                    save=True,
                                    Nsweeps=Nsweeps, ncv=ncv)     for D in Ds}        
        
    

        
    
    
if __name__ == "__main__":
    benchmark_3('CPU_GSS_DMRG_benchmark')

    
