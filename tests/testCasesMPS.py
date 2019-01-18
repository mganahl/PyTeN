#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code

import sys,os
import unittest
import lib.mpslib.Container as CO
import lib.mpslib.mpsfunctions as mf
import copy
import lib.mpslib.Tensor as tnsr
import numpy as np
import random

class TestTensorNetwork(unittest.TestCase):
    def setUp(self):
        self.shape=tuple(np.random.randint(1,4,3))
        self.tshape=tuple(np.random.randint(3,4,3))        
        self.TN=CO.TensorNetwork.random(shape=self.shape,tensorshapes=self.tshape)
    def testTypePreservation(self):
        for n,x in np.ndenumerate(self.TN):
            self.assertTrue(type(self.TN[n])==tnsr.Tensor)
    def testInit(self):
        tn1=CO.TensorNetwork.random(shape=self.shape,tensorshapes=self.tshape)
        tn2=CO.TensorNetwork.ones(shape=self.shape,tensorshapes=self.tshape)
        tn3=CO.TensorNetwork.zeros(shape=self.shape,tensorshapes=self.tshape)
        tn4=CO.TensorNetwork.empty(shape=self.shape,tensorshapes=self.tshape)
        
        self.assertTrue(np.all(tn2==1))
        self.assertTrue(np.all(tn3==0))
    
    def testScalarOps(self):
        tn=self.TN
        self.assertEqual(random.random()*tn,tn)
        self.assertEqual(tn*random.random(),tn)
        tn2=tn.copy()
        tn2*=random.random()
        self.assertEqual(tn2,tn)        
        tn2=tn.copy()
        tn2/=random.random()
        self.assertEqual(tn2,tn)
        
        self.assertEqual(type(random.random()*tn),type(tn))
        self.assertEqual(type(tn*random.random()),type(tn))
        self.assertEqual(type(tn/random.random()),type(tn))
        tn2*=random.random()
        self.assertEqual(type(tn2),type(tn))        
        tn2=tn.copy()
        tn2/=random.random()
        self.assertEqual(type(tn2),type(tn))                
        
    def testufuncs(self):
        tn=self.TN
        self.assertEqual(random.random()*tn,tn)
        self.assertEqual(tn*random.random(),tn)
        tn2=tn.copy()
        tn2*=random.random()
        self.assertEqual(tn2,tn)        
        tn2=tn.copy()
        tn2/=random.random()
        self.assertEqual(tn2,tn)
        
    def test_view(self):
        tn=self.TN
        tn.tensors[0]=tnsr.Tensor.zeros(tn.tensors[0].shape)
        for n,x in np.ndenumerate(tn):
            self.assertTrue(np.all(tn[n]==0))
            break
        tn2=tn.view()
        #print(tn2._tensors)
        for n,x in np.ndenumerate(tn._tensors):
            tn2[n]=tnsr.Tensor.ones(tn.tensors[0].shape)
            break
        for n,x in np.ndenumerate(tn._tensors):        
            self.assertTrue(np.all(tn[n]==1))        
            self.assertTrue(np.all(tn2[n]==1))
            break


class TestMPS(TestTensorNetwork):
    def setUp(self):
        N=random.randint(10,20)
        self.D=np.random.randint(1,10,N+1)
        self.d=[random.randint(2,4)]*N
        self.TN=CO.MPS.random(D=self.D,d=self.d)
        
        
    def testTypePreservation(self):
        self.TN.position(0)
        self.TN.position(len(self.TN))
        for n,x in np.ndenumerate(self.TN):
            self.assertTrue(type(self.TN[n])==tnsr.Tensor)
        self.assertTrue(type(self.TN.mat)==tnsr.Tensor)

    def testInit(self):
        tn1=CO.MPS.random(D=self.D,d=self.d)
        tn2=CO.MPS.ones(D=self.D,d=self.d)
        tn3=CO.MPS.empty(D=self.D,d=self.d)



class TestFiniteMPS(TestMPS):
    def setUp(self):
        N=random.randint(10,20)
        self.D=[1]+list(np.random.randint(1,10,N-1))+[1]
        self.d=[random.randint(2,4)]*N
        self.TN=CO.FiniteMPS.random(D=self.D,d=self.d)

    def testTypePreservation(self):
        super(TestFiniteMPS,self).testTypePreservation()
        S=self.TN.SchmidtSpectrum(random.sample(range(len(self.TN)),1)[0])

        
    def testInit(self):
        tn1=CO.FiniteMPS.random(D=self.D,d=self.d)
        tn2=CO.FiniteMPS.ones(D=self.D,d=self.d)
        tn3=CO.FiniteMPS.empty(D=self.D,d=self.d)


class TestCanonizedFiniteMPS(TestTensorNetwork):
    def setUp(self):
        N=random.randint(10,20)
        self.D=[1]+list(np.random.randint(1,10,N-1))+[1]
        self.d=[random.randint(2,4)]*N
        self.TN=CO.FiniteMPS.random(D=self.D,d=self.d).canonize()
    def testInit(self):
        pass


if __name__ == "__main__":
    suite0 = unittest.TestLoader().loadTestsFromTestCase(TestTensorNetwork)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestMPS)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestFiniteMPS)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestCanonizedFiniteMPS)            
    unittest.TextTestRunner(verbosity=2).run(suite0)
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)
    unittest.TextTestRunner(verbosity=2).run(suite3)            
