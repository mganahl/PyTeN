import numpy as np
import scipy as sp
import os,pickle
import math,io
import datetime as dt
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import lib.mpslib.mps as mpslib
import lib.mpslib.engines as en
import lib.mpslib.mpsfunctions as mf
import lib.mpslib.Hamiltonians as H

out_s = io.BytesIO()
N=10
D=10
d=2
mps=mpslib.MPS(N,D,d,obc=True)
pickle.dump(mps,out_s)

in_s=io.BytesIO(out_s.getvalue())
while True:
    try:
        mps2=pickle.load(in_s)
    except EOFError:
        break


print (mps2==mps)

