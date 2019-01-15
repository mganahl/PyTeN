#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code

import sys,os
import unittest
import lib.mpslib.Container as CO
import lib.mpslib.mpsfunctions as mf
import copy
import numpy as np
import random

class TestTensorNetwork(unittest.TestCase):
    def setUp(self):
        self.shape=tuple(np.random.randint(1,4,3))
        self.tshape=tuple(np.random.randint(3,4,3))        
        self.TN=CO.TensorNetwork.random(shape=self.shape,tensorshapes=self.tshape)
        
    def testInit(self):
        tn1=CO.TensorNetwork.random(shape=self.shape,tensorshapes=self.tshape)
        tn2=CO.TensorNetwork.ones(shape=self.shape,tensorshapes=self.tshape)
        tn3=CO.TensorNetwork.zeros(shape=self.shape,tensorshapes=self.tshape)
        tn4=CO.TensorNetwork.empty(shape=self.shape,tensorshapes=self.tshape)
        
        self.assertTrue(np.all(tn1==1))
        self.assertTrue(np.all(tn2==0))
    
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

class TestMPS(TestTensorNetwork):
    def setUp(self):
        N=random.randint(10,20)
        self.D=np.random.randint(1,10,N+1)
        self.d=[random.randint(2,4)]*N
        self.TN=CO.MPS.random(D=self.D,d=self.d)
        
    def testInit(self):
        tn1=CO.MPS.random(D=self.D,d=self.d)
        tn2=CO.MPS.ones(D=self.D,d=self.d)
        tn3=CO.MPS.zeros(D=self.D,d=self.d)
        tn4=CO.MPS.empty(D=self.D,d=self.d)

        self.assertTrue(np.all([np.all(t==1) for t in tn2]))
        self.assertTrue(np.all([np.all(t==0) for t in tn3]))        


class TestFiniteMPS(TestMPS):
    def setUp(self):
        N=random.randint(10,20)
        self.D=[1]+list(np.random.randint(1,10,N-1))+[1]
        self.d=[random.randint(2,4)]*N
        self.TN=CO.FiniteMPS.random(D=self.D,d=self.d)
        
    def testInit(self):
        tn1=CO.FiniteMPS.random(D=self.D,d=self.d)
        tn2=CO.FiniteMPS.ones(D=self.D,d=self.d)
        tn3=CO.FiniteMPS.zeros(D=self.D,d=self.d)
        tn4=CO.FiniteMPS.empty(D=self.D,d=self.d)

        self.assertTrue(np.all([np.all(t==1) for t in tn2]))
        self.assertTrue(np.all([np.all(t==0) for t in tn3]))        


if __name__ == "__main__":
    suite0 = unittest.TestLoader().loadTestsFromTestCase(TestTensorNetwork)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestMPS)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestFiniteMPS)        
    unittest.TextTestRunner(verbosity=2).run(suite0)
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)        
