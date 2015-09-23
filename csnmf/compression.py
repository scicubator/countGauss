"""
   Copyright (c) 2015, Mariano Tepper, Duke University.
   All rights reserved.

   This file is part of RCNMF and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import, print_function
import numpy as np
import dask.array as da
import csnmf.tsqr
from gauss_sim import Random_Projection

def compression_level(n, q):
    return min(max(20, q + 10), n)


def _inner_compress(data, omega, n_power_iter=0, qr=np.linalg.qr):
    mat_h = data.dot(omega)
    for j in range(n_power_iter):
        mat_h = data.dot(data.T.dot(mat_h))
    q, _ = qr(mat_h)
    comp = q.T
    return comp.dot(data), comp

def _our_compress(data, comp_level):
     seed = np.random.randint(1000)
     M    = Random_Projection(data.T, comp_level, seed=seed)
     tmp  = 0
     return M.T, tmp

def compress(data, q, n_power_iter=0, our=False):

    n = data.shape[1]
    comp_level = compression_level(n, q)

    if our:
        return _our_compress(data,comp_level)

    else:
        if isinstance(data, np.ndarray):
            omega = np.random.standard_normal(size=(n, comp_level))
            qr = np.linalg.qr
        elif isinstance(data, da.Array):
            omega = da.random.standard_normal(size=(n, comp_level),
                                              chunks=(data.chunks[1],
                                                         (comp_level,)))
            qr = csnmf.tsqr.qr
        else:
            raise TypeError('Cannot compress data of type ' + type(data).__name__)

        return _inner_compress(data, omega, n_power_iter=n_power_iter, qr=qr)
