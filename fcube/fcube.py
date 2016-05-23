from __future__ import division
import numpy as np
import scipy.sparse

def sign_projection(M, new_dim, seed=23):
    """Project the columns of the matrix M into the  lower dimension new_dim using gaussian """
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

def gauss_projection(M, new_dim, G=None, seed=23):
    """ Project the columns of the matrix M into the lower dimension new_dim using gaussian"""
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

def countSketch_projection(M, new_dim, seed=23):
    """ Project the columns of the matrix M into the lower dimension new_dim using count sketch"""
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

def countGauss_projection(M, new_dim, seed=23):
    """ Project the columns of the matrix M into the lower dimension new_dim using count sketch + gaussian algorithm"""
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

if __name__ == '__main__':
    print "fcube"
