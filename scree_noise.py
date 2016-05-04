from __future__ import division
import numpy as np
#from fht import fht,fht2
#import scipy.linalg
from time import time
#from anchor_new import fwht_spiral
import pylab as plt
from lshash import LSHash
from collections import Counter
from fcube.fcube import fcube_projection, countGauss_projection, countSketch_projection

def gauss_sim(X,k,maxiter=10, alg='gauss'):
    m,n  = np.shape(X)
    idxT = Counter()
    for iT in range(maxiter):
        if alg == 'gauss':
            G    = np.random.randn(n,k)
            Z    = np.dot(X,G)
        if alg == 'countGauss':
            Z    = countGauss_projection(X,k)
        if alg == 'hash':
            if iT == 0:
                lsh  = LSHash(20,n)
                for j in range(m):
                    lsh.index(X[j])
                Z = np.random.rand(m,k)
            for k in range(k):
                lsh.query(np.random.rand(n),num_results=1,distance_func='euclidean')
        if alg != 'hash':
            indMax  = np.argmax(Z,axis=0)
            idxT.update(indMax)
            indMin  = np.argmin(Z,axis=0)
            idxT.update(indMin)
        #print idxT
    return sorted([k for k,v in idxT.most_common(k)])
    #return idxT

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

# Project the columns of the matrix M into the
# lower dimension new_dim
def Random_Projection(M, new_dim):
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
    fwht_spiral(MM,B,G,P)

    M_red = MM[:,:new_dim]
    return M_red

def gen_data_noise(m,n,r, sigma):
    U   =  np.zeros((m,r))
    V   =  np.random.rand(r,n)
    U[:r,:r] = np.eye(r)
    cnt      = r
    for j in range(r):
        for k in range(r):
            if k<j:
                U[cnt,j] = 1
                U[cnt,k] = 1
                cnt     += 1
    for i in range(m):
        U[i,:] = U[i,:]/np.sum(U[i,:])
    X     =  np.dot(U,V) + sigma*np.random.randn(m,n)
    return X,U,V


if __name__ == '__main__':
    m   =  210
    n   =  1000
    r   =  20
    maxiter = 4

    #rList  = range(5,50)
    #maxiter= range(5,300,5)
    #R      = 500
    sList  = [0.01,0.02, 0.03, 0.05, 0.08, 0.12,0.22,0.36,0.6,1]
    maxiter= 4
    R      = 100
    succ   = np.zeros((m,len(sList)))
    for iT, sigma in enumerate(sList):
           for run in range(R):
                #alg   =  'fastfood'
                X,U,V = gen_data_noise(m,n,r, sigma)

                t0  = time()
                #I   =  gauss_sim(X,r,maxiter,alg='fastfood')
                I   =  gauss_sim(X,r,maxiter,alg='countGauss')
                dur = time() - t0
                for elem in I:
                    succ[elem,iT] +=1
                print "noise, runs", iT,run
    import ipdb
    ipdb.set_trace()
    x = range(two_U-two_L)

    plt.ion()
    plt.plot(x,FF,'bo--')
    plt.plot(x,GP,'gs--')
    plt.plot(x,HH,'ro--')
    plt.xticks(x,[str(label) for label in rList],rotation='horizontal', fontsize=20)
    plt.legend(('FastFood','Gaussian', 'Hashing'), fontsize=22)
    plt.title('Number of Anchors and Samples are %s, %s'%(str(r), str(m)), fontsize=25)
    plt.xlabel('Dimension of data', fontsize=22)
    plt.ylabel('Running times', fontsize=22)
    plt.show(True)
    #plt.savefig('comp_GP_FF.png')
