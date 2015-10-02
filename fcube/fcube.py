from __future__ import division
import numpy as np
from anchor_new import fwht_spiral
from anchor_mic import fwht_mic

def shift_bit_length(x):
    return 1<<(x-1).bit_length()
# Project the columns of the matrix M into the
# lower dimension new_dim using michael's algorithm
def mic_projection(M, new_dim, seed=23):
    UPPER = 262144*2
    np.random.seed(seed=seed)
    m,old_dim = np.shape(M)
    pow_two   = shift_bit_length(old_dim)

    G_1 = np.random.randn(pow_two)
    G_2 = np.random.randn(pow_two)
    P = np.arange(pow_two,dtype="intc");  np.random.shuffle(P)
    P = np.sort(P[:new_dim])

    print "computing product"

    #M_red = M_red[:new_dim,:]
    M_red = np.zeros((m,new_dim))
    MM    = np.zeros((1,pow_two))
    for i in range(m):
        MM[:,:old_dim] = M[i,:]
        fwht_mic(MM,G_1,G_2)
        M_red[i,:] = MM[:,P]
        if i%200==0:
            print i
        MM.fill(0)
    #import ipdb
    #ipdb.set_trace()
    return M_red

# Project the columns of the matrix M into the
# lower dimension new_dim using fastfood
def fcube_projection(M, new_dim, seed=23):
    UPPER = 262144*2
    np.random.seed(seed=seed)
    m,old_dim = np.shape(M)
    pow_two   = shift_bit_length(old_dim)

    B = np.random.rand(pow_two)
    B[B<0.5]  = -1.0
    B[B>=0.5] =  1.0
    G = np.random.randn(pow_two)
    P = np.arange(pow_two,dtype="intc");  np.random.shuffle(P)

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
    M_red = np.zeros((m,new_dim))
    MM    = np.zeros((1,pow_two))
    for i in range(m):
        MM[:,:old_dim] = M[i,:]
        fwht_spiral(MM,B,G,P)
        M_red[i,:] = MM[:,:new_dim]
        if i%200==0:
            print i
        MM.fill(0)
    #import ipdb
    #ipdb.set_trace()
    return M_red

if __name__ == '__main__':
    print "fcube"
