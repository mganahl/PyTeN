"""
@author: Martin Ganahl
"""
import numpy as np
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.SimContainer as SCT
import lib.mpslib.MPO as MPO
import lib.mpslib.TensorNetwork  as TN

if __name__ == "__main__":
    D, d, N = 50, 2, 2
    Jz=np.ones(N)
    Jxy=np.ones(N)
    dtype=np.float64
    mps=TN.MPS.random(D=[D] * (N+1) ,d=[d] * N, dtype=dtype)
    mpo=MPO.InfiniteXXZ(Jz, Jxy, np.zeros(N), dtype=dtype)
    idmrg=SCT.InfiniteDMRGEngine(mps, mpo, 'insert_name_here')
    print(idmrg.run_one_site.__doc__)
    idmrg.run_one_site(Nsweeps=100, solver='LAN', precision=1E-10, ncv=40, verbose=1)

    
