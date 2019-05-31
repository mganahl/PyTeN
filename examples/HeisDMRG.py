"""
@author: Martin Ganahl
"""
import numpy as np
import matplotlib.pyplot as plt
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.SimContainer as SCT
import lib.mpslib.MPO as MPO
import lib.mpslib.TensorNetwork  as TN

if __name__ == "__main__":
    D, d, N = 50, 2, 100
    Jz=np.ones(N)
    Jxy=np.ones(N)
    dtype=np.float64
    mps=TN.FiniteMPS.random(D=[D] * (N - 1) ,d=[d] * N, dtype=dtype)
    mpo=MPO.FiniteXXZ(Jz, Jxy, np.zeros(N), dtype=dtype)
    dmrg=SCT.FiniteDMRGEngine(mps, mpo, 'insert_name_here')
    print(dmrg.run_one_site.__doc__)
    dmrg.run_one_site(Nsweeps=2, solver='LAN', precision=1E-10, ncv=40, verbose=1)
    
    #a list of measurement operators to measured (here sz)
    Sz=[np.diag([0.5,-0.5]) for n in range(N)]
    #measure the local spin-density
    meanSz=dmrg.mps.measure_1site_ops(Sz, range(N))
    meanSzSz=dmrg.mps.measure_1site_correlator(Sz[0], Sz[0], N//2, range(N))    
    Dt=20
    dmrg.mps.truncate(schmidt_thresh=1E-8,D=Dt)    
    print (dmrg.mps.D)

    meanSztrunc = dmrg.mps.measure_1site_ops(Sz, range(N))
    meanSzSztrunc = dmrg.mps.measure_1site_correlator(Sz[0], Sz[0], N//2, range(N))        

    #compare results before and after truncation

    plt.figure(1)
    plt.plot(range(len(meanSz)),meanSz,range(len(meanSztrunc)),meanSztrunc,'--')
    plt.ylabel(r'$\langle S^z_i\rangle$')
    plt.xlabel(r'$i$')    
    plt.legend(['before truncation (D={0})'.format(D),'after truncation (D={0})'.format(Dt)])

    plt.figure(2)
    plt.plot(range(len(meanSzSz)),meanSzSz,range(len(meanSzSztrunc)),meanSzSztrunc,'--')
    plt.ylabel(r'$\langle S^z_{N/2} S^z_{i}\rangle$')
    plt.xlabel(r'$i$')        
    plt.legend(['before truncation (D={0})'.format(D),'after truncation (D={0})'.format(Dt)])    

    plt.draw()
    plt.show()
    
    plt.draw()
    plt.show()
    input()
    

    
