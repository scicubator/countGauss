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
        if alg == 'fastfood':
            Z    = Random_Projection(X,k)
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

def gen_data(m,n,r):
    U   =  np.random.rand(m,r)
    V   =  np.random.rand(r,n)
    U[:r,:r] = np.eye(r)
    for i in range(m):
        U[i,:] = U[i,:]/np.sum(U[i,:])
    X     =  np.dot(U,V)
    return X,U,V


if __name__ == '__main__':
    m   =  500
    n   =  1000
    maxiter = 4

    nList  = range(5,50,2)
    maxiter= range(1,5)
    R      = 100
    #nList  = range(5,50,2)
    #maxiter= range(1,5)
    #R      = 200
    succ   = np.zeros((len(nList),len(maxiter)))
    for iT, r in enumerate(nList):
       for iC,citer in enumerate(maxiter):
           for run in range(R):
                #alg   =  'fastfood'
                X,U,V = gen_data(m,n,r)

                t0  = time()
                I   =  gauss_sim(X,r,citer,alg='countGauss')
                #I   =  gauss_sim(X,r,citer,alg='gauss')
                dur = time() - t0
                if set(I)==set(range(r)):
                   succ[iT,iC]+=1
                print "rank, projections, runs", iT,iC,run
    plt.ion()
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    major_ticks = np.arange(5,50,5)
    ax.set_xticks(major_ticks)
    plt.imshow(succ.T, cmap='gray_r', origin="lower")

    import ipdb
    ipdb.set_trace()
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
