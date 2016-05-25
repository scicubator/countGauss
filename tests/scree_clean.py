from __future__ import division
import numpy as np
from time import time
import pylab as plt
from collections import Counter
from fcube.fcube import countGauss_projection, countSketch_projection
import pickle

def gauss_sim(X,k,alg='gauss'):
    m,n  = np.shape(X)
    idxT = Counter()
    if alg == 'gauss':
        G    = np.random.randn(n,k)
        Z    = np.dot(X,G)
    if alg == 'countGauss':
        Z    = countGauss_projection(X,k)
    indMax  = np.argmax(Z,axis=0)
    idxT.update(indMax)
    indMin  = np.argmin(Z,axis=0)
    idxT.update(indMin)
    return sorted([k for k,v in idxT.most_common(k)])


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
                X,U,V = gen_data(m,n,r)
                I   =  gauss_sim(X, int(r*citer),alg=alg)
                if set(I)==set(range(r)):
                   succ[iT,iC]+=1
                print "rank, projections, runs", iT,iC,run
    return succ


if __name__ == '__main__':
    m   =  500
    n   =  1000

    nList  = range(5,50,1)
    maxiter= [0.1*x+1 for x in np.arange(50)]
    R      = 100

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

    D={}
    D['gauss'] = succ
    D['countGauss'] = succG
    pickle.dump(D, open("tests/clean.pkl","wb"))

    import ipdb
    ipdb.set_trace()

    #savefig('out.svg', transparent=True, bbox_inches='tight', pad_inches=0)
