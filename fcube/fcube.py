from __future__ import division
import numpy as np
from anchor_new import fwht_spiral
from anchor_mic import fwht_mic
import scipy.sparse

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

# Project the columns of the matrix M into the
# lower dimension new_dim using gaussian
def sign_projection(M, new_dim, seed=23):
    #np.random.seed(seed=seed)
    np.random.seed()
    m, old_dim = np.shape(M)

    SMALL=True
    if SMALL:
        S = np.random.randint(2,size=old_dim*new_dim)*2-1
        M_red = scipy.sparse.csr_matrix.dot(M,np.reshape(S,[old_dim,new_dim]))
    else:
        M_red = np.zeros((m,new_dim))
        for i in range(new_dim):
            g = np.sign(np.random.randn(old_dim))
            M_red[:,i]=scipy.sparse.csr_matrix.dot(M,g)

    return M_red/np.sqrt(new_dim)

# Project the columns of the matrix M into the
# lower dimension new_dim using gaussian
def gauss_projection(M, new_dim, G=None, seed=23):
    #np.random.seed(seed=seed)
    np.random.seed()
    m, old_dim = np.shape(M)
    M_red = np.zeros((m,new_dim))

    SMALL=True
    if SMALL:
        if G is not None:
            M_red = scipy.sparse.csr_matrix.dot(M,G)
        else:
            M_red = scipy.sparse.csr_matrix.dot(M,np.random.randn(old_dim,new_dim))
    else:
        for i in range(new_dim):
            g = np.random.randn(old_dim)
            M_red[:,i]=scipy.sparse.csr_matrix.dot(M,g)

    return M_red/np.sqrt(new_dim)

# Project the columns of the matrix M into the
# lower dimension new_dim using count sketch
def countSketch_projection(M, new_dim, seed=23):
    np.random.seed()
    m,old_dim = np.shape(M)

    #ksq = new_dim * new_dim
    ksq = new_dim
    #G = np.random.randn(ksq,new_dim)
    #R = np.array([np.random.random_integers(ksq)-1 for i in range(old_dim)])
    C = np.arange(old_dim)
    R = np.random.random_integers(ksq,size=old_dim)-1
    D = np.random.randint(2,size=old_dim)*2-1
    #D = np.array([1 if np.random.rand(1)>0.5 else -1 for i in range(old_dim)])

    S = scipy.sparse.csr_matrix((D,(R,C)), shape=(ksq,old_dim))

    #M_red = scipy.sparse.csc_matrix.dot(scipy.sparse.csr_matrix.dot(S,M.T).T,G)
    M_red = scipy.sparse.csr_matrix.dot(S,M.T).T
    return M_red

# Project the columns of the matrix M into the
# lower dimension new_dim using count sketch + gaussian algorithm
def countGauss_projection(M, new_dim, seed=23):
    np.random.seed()
    m,old_dim = np.shape(M)

    #ksq = new_dim * new_dim
    ksq = np.int(new_dim*5) # This was not converging for scree plots of damle/sun
    #ksq = np.int(new_dim*new_dim)
    G = np.random.randn(ksq,new_dim)
    #R = np.array([np.random.random_integers(ksq)-1 for i in range(old_dim)])
    #D = np.array([1 if np.random.rand(1)>0.5 else -1 for i in range(old_dim)])
    C = np.arange(old_dim)
    R = np.random.random_integers(ksq,size=old_dim)-1
    D = np.random.randint(2,size=old_dim)*2-1

    S = scipy.sparse.csr_matrix((D,(R,C)), shape=(ksq,old_dim))

    if scipy.sparse.issparse(M):
        MSt = scipy.sparse.csr_matrix.dot(M,S.T)
        M_red = scipy.sparse.csr_matrix.dot(MSt,G)
    else:
        SMt = scipy.sparse.csr_matrix.dot(S, M.T)
        M_red = np.dot(SMt.T, G)
    return M_red/np.sqrt(new_dim)

# Project the columns of the matrix M into the
# lower dimension new_dim using michael's algorithm
def mic_projection(M, new_dim, seed=23):
    UPPER = 262144*2
    #np.random.seed(seed=seed)
    np.random.seed()
    m,old_dim = np.shape(M)
    pow_two   = shift_bit_length(old_dim)

    G_1 = np.random.randn(pow_two)
    G_2 = np.random.randint(2,size=pow_two)
    G_2[G_2==0] = -1
    G_2 = np.array(G_2, dtype="intc")
    P = np.arange(pow_two,dtype="intc");  np.random.shuffle(P)
    P = np.sort(P[:new_dim])

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
    return M_red

# Project the columns of the matrix M into the
# lower dimension new_dim using fastfood
def fcube_projection(M, new_dim, seed=23):
    UPPER = 262144*2
    #np.random.seed(seed=seed)
    np.random.seed()
    m,old_dim = np.shape(M)
    pow_two   = shift_bit_length(old_dim)

    B = np.random.rand(pow_two)
    B[B<0.5]  = -1.0
    B[B>=0.5] =  1.0
    G = np.random.randn(pow_two)
    P = np.arange(pow_two,dtype="intc");  np.random.shuffle(P)

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
