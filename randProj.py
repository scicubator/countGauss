from __future__ import division
import numpy as np
#from fht import fht,fht2
#import scipy.linalg
from time import time
from anchor_new import fwht_spiral

#  Need to finish this ...still incomplete (make use fht library import fht)
def shift_bit_length(x):
    return 1<<(x-1).bit_length()

# Project the columns of the matrix M into the
# lower dimension new_dim
def randProj(M, new_dim, seed=23):
    np.random.seed(seed=seed)
    m,old_dim = np.shape(M)
    pow_two   = shift_bit_length(old_dim)
    if old_dim==pow_two:
        MM = M
    else:
        MM = np.zeros((m,pow_two))
        MM[:,:old_dim] = M

    B = np.random.rand(pow_two)
    B[B<0.5]  = -1.0
    B[B>=0.5] =  1.0
    G = np.random.randn(pow_two)
    P = np.int32(range(pow_two));  np.random.shuffle(P)

    print "computing product"
    #M_red = fht2(np.dot(G, np.dot(P, fht2(np.dot(B,M),axes=0))),axes=0)
    #M_red = fht_v(np.dot(G, np.dot(P, fht_v(np.dot(B,M)))))
    #T = B*MM
    #fwht_spiral(T)
    #np.random.shuffle(T)
    #M_red = G*np.dot(P,T)
    #M_red = G*T
    #fwht_spiral(M_red)
    #H = scipy.linalg.hadamard(old_dim)/np.sqrt(old_dim)
    #M_red_1 = np.dot(H,(np.dot(G, np.dot(P, np.dot(H,(np.dot(B,M)))))))

    #M_red = M_red[:new_dim,:]
    #import ipdb
    #ipdb.set_trace()




    fwht_spiral(MM,B,G,P)

    M_red = MM[:,:new_dim]
    return M_red
