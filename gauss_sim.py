from __future__ import division
import numpy as np
from fht import fht
#import scipy.linalg
from time import time
from anchor_new import fwht_spiral
import pylab as plt
from lshash import LSHash

def gauss_sim(X,k,maxiter=10, alg='gauss'):
    m,n  = np.shape(X)
    Idx  = set()
    for iT in range(maxiter):
        if alg == 'gauss':
            G    = np.random.randn(n,k)
            Z    = np.dot(X,G)
        if alg == 'fastfood':
            Z    = Random_Projection(X,k)
        if alg == 'hash':
            if iT == 0:
                lsh  = LSHash(20,n)
                for j in range(m):
                    lsh.index(X[j])
                Z = np.random.rand(m,k)
            for k in range(k):
                lsh.query(np.random.rand(n),num_results=1,distance_func='euclidean')
        if alg != 'hash':
            ind  = np.argmax(Z,axis=0)
            Idx  = set(ind).union(Idx)
            ind  = np.argmin(Z,axis=0)
            Idx  = set(ind).union(Idx)
        #print Idx
    return Idx

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

#def fht_v(M):
#    m,n=np.shape(M)
#    for i in range(n):
#        M[:,i]=fht(M[:,i])
#    return M

# Project the columns of the matrix M into the
# lower dimension new_dim
def Random_Projection(M, new_dim, seed=23):
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
    r   =  200
    m   =  400
    maxiter = 4

    #n   =  4096
    two_L =  7
    two_U =  15
    FF    = np.zeros(len(range(two_L,two_U)))
    GP    = np.zeros(len(range(two_L,two_U)))
    HH    = np.zeros(len(range(two_L,two_U)))
    iT    = 0
    nList =  [np.power(2,i) for i in range(two_L,two_U)]
    for n in nList:
        alg =  'fastfood'
        U   =  np.random.rand(m,r)
        V   =  np.random.rand(r,n)
        U[:r,:r] = np.eye(r)
        #for i in range(m):
        #    V[:,i] = V[:,i]/np.sum(V[:,i])

        X   =  np.dot(U,V)

        t0  = time()
        I   =  gauss_sim(X,r,maxiter,alg='fastfood')
        dur = time() - t0
        FF[iT] = dur
        print "fast food took", dur, " seconds \n"

        t0  = time()
        I   =  gauss_sim(X,r,maxiter,alg='gauss')
        dur = time() - t0
        GP[iT] = dur
        print "Gaussian proj took", dur, " seconds \n"

        t0  = time()
        I   =  gauss_sim(X,r,maxiter,alg='hash')
        dur = time() - t0
        HH[iT] = dur
        print "Hashing took", dur, " seconds \n"
        iT += 1
    x = range(two_U-two_L)

    plt.ion()
    plt.plot(x,FF,'bo--')
    plt.plot(x,GP,'gs--')
    plt.plot(x,HH,'ro--')
    plt.xticks(x,[str(label) for label in nList],rotation='horizontal', fontsize=20)
    plt.legend(('FastFood','Gaussian', 'Hashing'), fontsize=22)
    plt.title('Number of Anchors and Samples are %s, %s'%(str(r), str(m)), fontsize=25)
    plt.xlabel('Dimension of data', fontsize=22)
    plt.ylabel('Running times', fontsize=22)
    plt.show(True)
    #plt.savefig('comp_GP_FF.png')
