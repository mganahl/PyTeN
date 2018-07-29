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
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import lib.mpslib.mps as mpslib
import tests.HeisED.XXZED as ed
import scipy as sp
import lib.Lanczos.LanczosEngine as lanEn
from scipy.sparse import csc_matrix
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


class TestTFI(unittest.TestCase):
    def setUp(self):
        Jx=-1.0*np.ones(1)
        B=np.ones(1)
        self.mpo=H.TFI(Jx,B,False)
        self.D=20
    def testTFI_AR_float(self):
        mps=mpslib.MPS.random(N=1,D=self.D,d=2,obc=False,dtype=float)
        mps.regauge(gauge='right')
        iMPS=en.VUMPSengine(mps,self.mpo,'erase_me')
        e=iMPS.simulate(Nmax=10000,epsilon=1E-8,tol=1E-12,lgmrestol=1E-12,\
                        ncv=30,numeig=3,artol=1E-12,arnumvecs=1,\
                        arncv=30,svd=False,solver='AR')
        self.assertTrue(np.abs(-1.2732394-e)<1E-5)
    def testTFI_AR_complex(self):
        mps=mpslib.MPS.random(N=1,D=self.D,d=2,obc=False,dtype=complex)
        mps.regauge(gauge='right')
        iMPS=en.VUMPSengine(mps,self.mpo,'erase_me')
        e=iMPS.simulate(Nmax=10000,epsilon=1E-8,tol=1E-12,lgmrestol=1E-12,\
                        ncv=30,numeig=3,artol=1E-12,arnumvecs=1,\
                        arncv=30,svd=False,solver='AR')
        self.assertTrue(np.abs(-1.2732394-e)<1E-5)
    def testTFI_LAN_float(self):
        mps=mpslib.MPS.random(N=1,D=self.D,d=2,obc=False,dtype=float)
        mps.regauge(gauge='right')
        iMPS=en.VUMPSengine(mps,self.mpo,'erase_me')
        e=iMPS.simulate(Nmax=10000,epsilon=1E-8,tol=1E-12,lgmrestol=1E-12,\
                        ncv=30,numeig=3,artol=1E-12,arnumvecs=1,\
                        arncv=30,svd=False,solver='LAN')
        self.assertTrue(np.abs(-1.2732394-e)<1E-5)        
    def testTFI_LAN_complex(self):
        mps=mpslib.MPS.random(N=1,D=self.D,d=2,obc=False,dtype=complex)
        mps.regauge(gauge='right')
        iMPS=en.VUMPSengine(mps,self.mpo,'erase_me')
        e=iMPS.simulate(Nmax=10000,epsilon=1E-8,tol=1E-12,lgmrestol=1E-12,\
                        ncv=30,numeig=3,artol=1E-12,arnumvecs=1,\
                        arncv=30,svd=False,solver='LAN')
        self.assertTrue(np.abs(-1.2732394-e)<1E-5)        



if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTFI)
    unittest.TextTestRunner(verbosity=2).run(suite1) 
