#!/usr/bin/env python
import sys,copy
import numpy as np
import random


a=[]
for n in range(10):
    a.append(np.random.rand(2,3,4))
b=copy.deepcopy(a)
print type(b)
print type(b[0])
