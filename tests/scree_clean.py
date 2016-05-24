from __future__ import division
import numpy as np
from time import time
import pylab as plt
from collections import Counter
from fcube.fcube import countGauss_projection, countSketch_projection

def gauss_sim(X,k,maxiter=10, alg='gauss'):
    m,n  = np.shape(X)
    idxT = Counter()
    for iT in range(maxiter):
        if alg == 'gauss':
            G    = np.random.randn(n,k)
            Z    = np.dot(X,G)
        if alg == 'countGauss':
            Z    = countGauss_projection(X,k)
        if alg != 'hash':
            indMax  = np.argmax(Z,axis=0)
            idxT.update(indMax)
            indMin  = np.argmin(Z,axis=0)
            idxT.update(indMin)
        #print idxT
    return sorted([k for k,v in idxT.most_common(k)])
    #return idxT


def gen_data(m,n,r):
    U   =  np.random.rand(m,r)
    V   =  np.random.rand(r,n)
    U[:r,:r] = np.eye(r)
    for i in range(m):
        U[i,:] = U[i,:]/np.sum(U[i,:])
    X     =  np.dot(U,V)
    return X,U,V

def compute_sol(nList, maxiter, alg='gauss'):
    succ   = np.zeros((len(nList),len(maxiter)))
    for iT, r in enumerate(nList):
       for iC,citer in enumerate(maxiter):
           for run in range(R):
                #alg   =  'fastfood'
                X,U,V = gen_data(m,n,r)

                t0  = time()
                I   =  gauss_sim(X,r,citer,alg=alg)
                #I   =  gauss_sim(X,r,citer,alg='gauss')
                dur = time() - t0
                if set(I)==set(range(r)):
                   succ[iT,iC]+=1
                print "rank, projections, runs", iT,iC,run
    return succ


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

    succ  = compute_sol(nList, maxiter, alg='gauss')
    succG = compute_sol(nList, maxiter, alg='countGauss')

    plt.ion()

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    major_ticks = np.arange(5,50,5)
    ax.set_xticks(major_ticks)
    plt.imshow(succ.T, cmap='gray_r', origin="lower")

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    major_ticks = np.arange(5,50,5)
    ax.set_xticks(major_ticks)
    plt.imshow(succG.T, cmap='gray_r', origin="lower")
    import ipdb
    ipdb.set_trace()

    #savefig('out.svg', transparent=True, bbox_inches='tight', pad_inches=0)
