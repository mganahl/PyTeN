"""
@author: Martin Ganahl
"""

from __future__ import absolute_import, division, print_function
from sys import stdout
import sys, time, copy, warnings
import numpy as np
import lib.ncon as ncon
import functools as fct

import lib.mpslib.Tensor as tnsr
comm = lambda x, y: np.dot(x, y) - np.dot(y, x)
anticomm = lambda x, y: np.dot(x, y) + np.dot(y, x)
herm = lambda x: np.conj(np.transpose(x))

def finegrain(tensor):
    """
    fine-grains an MPS tensor by splitting a single site into 2 sites (see paper by Dolfi et al)
    """
    if not tensor.shape[2] == 2:
        raise TypeError('finegrain: only d == 2 mps are supported')

    T=np.zeros((2,2,2)).astype(tensor.dtype)
    T[0,0,0] = 1.0
    T[1,1,0] = 1.0/np.sqrt(2.0)
    T[1,0,1] = 1.0/np.sqrt(2.0)
    D1, D2, d = tensor.shape
    tensor = ncon.ncon([tensor, T],[[-1, -3, 1], [1, -2, -4]])
    matrix = np.reshape(tensor,(D1 * d, D2 * d))
    U, S, V = np.linalg.svd(matrix)
    leftmat = U.dot(np.diag(np.sqrt(S)))
    rightmat = np.diag(np.sqrt(S)).dot(V)
    lefttens = np.transpose(np.reshape(leftmat,(D1, d, len(S))),(0, 2, 1))
    righttens = np.reshape(rightmat, (len(S), D1, d))
    return lefttens.view(tnsr.Tensor), righttens.view(tnsr.Tensor)
