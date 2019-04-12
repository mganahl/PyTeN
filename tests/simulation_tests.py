#!/usr/bin/env python
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)
import unittest
import numpy as np
import scipy as sp
import math
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.Container as CO
import lib.mpslib.TensorNetwork as TN
import lib.mpslib.MPO as MPO
import lib.mpslib.SimContainer as SCT
from lib.mpslib.Tensor import Tensor
import lib.Lanczos.LanczosEngine as LZ
import lib.ncon as ncon
import lib.HeisED.XXZED as ed
from scipy.sparse import csc_matrix
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


class TestFiniteXXZ(unittest.TestCase):        
    def setUp(self):
        self.D=60
        d=2
        self.N=10
        jz=1.0 
        jxy=1.0
        self.Jz=np.ones(self.N)*jz
        self.Jxy=np.ones(self.N)*jxy

        N=self.N #system size
        Nup=int(self.N/2) #number of up-spins
        Z2=-1
        Jzar=[0.0]*N
        Jxyar=[0.0]*N
        grid=[None]*N
        for n in range(N-1):
            grid[n]=[n+1]
            Jzar[n]=np.asarray([jz])
            Jxyar[n]=np.asarray([jxy])
            
        grid[N-1]=[]
        Jzar[N-1]=np.asarray([0.0])
        Jxyar[N-1]=np.asarray([0.0])
        np.asarray(Jxyar)

        Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
        Hsparse=ed.XXZSparseHam(Jxyar,Jzar,N,Nup,Z2,grid)
        e,v=sp.sparse.linalg.eigsh(Hsparse,k=1,which='SA',maxiter=1000000,tol=1E-8,v0=None,ncv=40)
        self.Eexact=e[0]
        self.eps = 1E-5


    def test_one_site_dmrg_float_ar(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2,precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_one_site_dmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2,precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)



    def test_one_site_dmrg_float_lan(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_one_site_dmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)



    def test_two_site_dmrg_float_ar(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_two_site_dmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_two_site_dmrg_float_lan(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')

        
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_two_site_dmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_two_site(Nsweeps=2, thresh=1e-10, D=100, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

class TestMPSTrunc(unittest.TestCase):        
    def setUp(self):
        self.D=64
        d=2
        self.N=32
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        N=self.N #system size
        self.eps = 1E-5

    def test_mps_trunc_float(self):
        dtype = np.float64
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')        
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        Sz=np.diag([0.5,-0.5])
        meanSz = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N))
        meanSzSz = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))
        
        Dt=32
        dmrg.mps.truncate(schmidt_thresh=1E-8,D=Dt)
        meanSztrunc = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N) )  
        meanSzSztrunc = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))

        self.assertTrue(np.linalg.norm(meanSztrunc-meanSz)< 1E-8)
        self.assertTrue(np.linalg.norm(meanSzSztrunc-meanSzSz)< 1E-4)        

        
    def test_mps_trunc_complex(self):
        dtype = np.complex128
        mps = TN.FiniteMPS.random(D=[self.D]*(self.N-1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.FiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.FiniteDMRGEngine(mps,mpo,'testXXZ')                
        edmrg = dmrg.run_one_site(Nsweeps=2, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        Sz=np.diag([0.5,-0.5])

        meanSz = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N))
        meanSzSz = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))
        
        Dt=20
        dmrg.mps.truncate(schmidt_thresh=1E-8,D=Dt)
        meanSztrunc = dmrg.mps.measure_1site_ops(ops=[Sz]*self.N, sites=range(self.N) )  
        meanSzSztrunc = dmrg.mps.measure_1site_correlator(Sz, Sz, self.N//2, range(self.N))

        self.assertTrue(np.linalg.norm(meanSztrunc-meanSz)< 1E-8)
        self.assertTrue(np.linalg.norm(meanSzSztrunc-meanSzSz)< 1E-4)        

        
class TestInfiniteXXZ(unittest.TestCase):        
    def setUp(self):
        self.D=60
        d=2
        self.N=2
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        self.Eexact = 0.25 - np.log(2) #exact GS energy of XXX model in thermodynamic limit
        self.eps = 1E-4
        self.Nsweeps=100

        
    def test_one_site_idmrg_float_ar(self):
        dtype = np.float64
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2] * self.N, dtype=dtype, minval=-0.3, maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


    def test_one_site_idmrg_complex_ar(self):
        dtype = np.complex128
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'ar')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)
        
    def test_one_site_idmrg_float_lan(self):
        dtype = np.float64
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)

    def test_one_site_idmrg_complex_lan(self):
        dtype = np.complex128
        mps = TN.MPS.random(D=[self.D]*(self.N + 1), d=[2]*self.N, dtype=dtype,minval=-0.3,maxval=0.3)
        mps.position(self.N)
        mps.position(0)
        mpo = MPO.InfiniteXXZ(self.Jz,self.Jxy,np.zeros(self.N), dtype=mps.dtype)
        dmrg = SCT.InfiniteDMRGEngine(mps,mpo,'testXXZ')
        edmrg = dmrg.run_one_site(Nsweeps=self.Nsweeps, precision=1e-10, ncv=10, verbose=1, solver= 'lan')    
        self.assertTrue((abs(edmrg-self.Eexact)/abs(self.Eexact))<self.eps)


class TestITEBD(unittest.TestCase):        
    def setUp(self):
        self.D=40
        d=2
        self.N=2
        self.Jz=np.ones(self.N)
        self.Jxy=np.ones(self.N)
        self.Eexact = 0.25 - np.log(2) #exact GS energy of XXX model in thermodynamic limit
        self.eps = 1E-4
        self.Nsweeps=100
        self.Eexact = 0.25 - np.log(2)
        
    def test_itebd_imag_float(self):
        dtype = np.float64
        imps = TN.MPS.random(D=[self.D]*(self.N + 1),d=[2] * self.N ,minval=-0.5,maxval=0.5)
        impo = MPO.InfiniteXXZ(Jz=np.ones(len(imps)),Jxy = np.ones(len(imps)), Bz=np.zeros(len(imps)), dtype=imps.dtype)
        itebd = SCT.InfiniteTEBDEngine(imps,impo)
        itebd.canonize_mps()
        Nmx = 500
        Sz = np.diag([-0.5, 0.5])
        Sx = np.array([[0,0.5],[0.5,0]])
        Sy = np.array ([[0, 0.5j],[-0.5j, 0]]) 
        from sys import stdout
        dt = 0.01
        numsteps=20
        for n in range(Nmx):
            itebd.do_steps(dt=-dt, numsteps=numsteps, D = self.D,recanonize = None, verbose=0)
            imps.canonize(precision=1E-12)
            energy = imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1])  #the minus accounts for the missing i above
            imps.roll(1)
            energy += imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1]) #the minus accounts for the missing i above           
            
            stdout.write('\r time: %0.4f/%0.4f, energy: %.8f (exact: %0.8f)'%(itebd.time, dt*Nmx*numsteps, np.real(energy[0])/2.0, 0.25 -np.log(2)))
        self.assertTrue(np.abs(energy[0] - self.Eexact) < 1E-4)
        
    def test_itebd_imag_complex(self):
        dtype = np.complex128
        imps = TN.MPS.random(D=[self.D]*(self.N + 1),d=[2] * self.N ,minval=-0.5,maxval=0.5)
        impo = MPO.InfiniteXXZ(Jz=np.ones(len(imps)),Jxy = np.ones(len(imps)), Bz=np.zeros(len(imps)), dtype=imps.dtype)
        itebd = SCT.InfiniteTEBDEngine(imps,impo)
        itebd.canonize_mps()
        Nmx = 500
        Sz = np.diag([-0.5, 0.5])
        Sx = np.array([[0,0.5],[0.5,0]])
        Sy = np.array ([[0, 0.5j],[-0.5j, 0]]) 
        from sys import stdout
        dt = 0.01
        numsteps=20
        for n in range(Nmx):
            itebd.do_steps(dt=-dt, numsteps=numsteps, D = self.D,recanonize = None, verbose=0)
            imps.canonize(precision=1E-12)
            energy = imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1])  
            imps.roll(1)
            energy += imps.measure_1site_correlator(Sz, Sz, 0, [1])
            energy += imps.measure_1site_correlator(Sx, Sx, 0, [1])
            energy += imps.measure_1site_correlator(Sy, Sy, 0, [1]) 
            
            stdout.write('\r time: %0.4f/%0.4f, energy: %.8f (exact: %0.8f)'%(itebd.time, dt*Nmx*numsteps, np.real(energy[0])/2.0, 0.25 -np.log(2)))
        self.assertTrue(np.abs(energy[0] - self.Eexact) < 1E-4)
            
if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFiniteXXZ)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestMPSTrunc)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestInfiniteXXZ)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestITEBD)            
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)
    unittest.TextTestRunner(verbosity=2).run(suite3)
    #unittest.TextTestRunner(verbosity=2).run(suite4)             

