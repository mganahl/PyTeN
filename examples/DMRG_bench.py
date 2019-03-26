#!/usr/bin/env python3
import numpy as np
import time
import pickle
import lib.mpslib.SimContainer as SCT
import lib.mpslib.TensorNetwork as TN
import lib.mpslib.MPO as MPO
import os
os.chdir('benchmarks')

def run_bench(N, D,
              dtype,
              save=True,
              Nsweeps=2,
              ncv=100,
              verbose=1):
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
    def walltime_log(lan=[], QR=[], add_layer=[], num_lan=[]):
        lan_times.extend(lan)
        QR_times.extend(QR)
        add_layer_times.extend(add_layer)
        num_lanczos.extend(num_lan)
    
    mps=TN.FiniteMPS.random(d=[2]*N, D=[D]*(N-1), dtype=dtype)
    mps.position(0)
    mpo=MPO.FiniteXXZ(Jz=np.ones([N-1]), Jxy=np.ones([N-1]), Bz=np.zeros([N]), dtype=dtype)
    dmrg=SCT.FiniteDMRGEngine(mps,mpo)
    t1=time.time()
    dmrg.run_one_site(verbose=verbose, Nsweeps=Nsweeps, ncv=ncv, precision=1E-200, solver='lan', walltime_log=walltime_log)
    out = {'lanczos':lan_times,
           'QR': QR_times,
           'add_layer': add_layer_times,
           'num_lanczos': num_lanczos,
           'total': time.time()-t1
    }
    if save:
        with open('PYTEN_DMRG_benchmark_N_{0}_D_{1}_dtype_{2}_ncv_{3}.pickle'.format(N,D,the_dtype,ncv),'wb') as f:
            pickle.dump(out,f)
            
    return out

if __name__ == "__main__":
    N=32
    Ds=[32, 64, 128, 256]#, 512, 1024]
    names = ['DMRG_benchmark_D{0}'.format(D) for D in Ds]
    dtype = np.float64
    res = {name: run_bench(N=N, D=D, dtype=dtype, save=True,  Nsweeps=2, ncv=10)     for D,name in zip(Ds,names)}
    # with open('PYTEN_DMRG_benchmarks_all.pickle','wb') as f:
    #     pickle.dump(res,f)

        
    
    


    
