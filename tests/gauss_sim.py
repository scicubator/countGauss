from __future__ import division
import numpy as np
from fht import fht
from time import time
import pylab as plt
from lshash import LSHash
from csnmf.third_party.mrnmf.nmf_process_algorithms import xray,spa
from csnmf.compression import compress
from fcube.fcube import fcube_projection, mic_projection

def gauss_sim(X,k,maxiter=10, alg='gauss'):
    m,n  = np.shape(X)
    Idx  = set()
    for iT in range(maxiter):
        if alg == 'gauss':
            G    = np.random.randn(n,k)
            Z    = np.dot(X,G)
        if alg == 'fastfood':
            Z    = fcube_projection(X,k)
        if alg == 'mic':
            Z    = mic_projection(X,k)
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


if __name__ == '__main__':
    r   =  200
    m   =  400
    maxiter = 4

    #n   =  4096
    two_L =  8
    two_U =  15
    FF    = np.zeros(len(range(two_L,two_U)))
    GP    = np.zeros(len(range(two_L,two_U)))
    HM    = np.zeros(len(range(two_L,two_U)))
    SP    = np.zeros(len(range(two_L,two_U)))
    XR    = np.zeros(len(range(two_L,two_U)))
    SC    = np.zeros(len(range(two_L,two_U)))
    iT    = 0
    nList =  [np.power(2,i) for i in range(two_L,two_U)]
    for n in nList:
        U   =  np.random.rand(m,r)
        V   =  np.random.rand(r,n)
        U[:r,:r] = np.eye(r)
        #for i in range(m):
        #    V[:,i] = V[:,i]/np.sum(V[:,i])

        X   =  np.dot(U,V)

        t0  = time()
        I_f   =  gauss_sim(X,r,maxiter,alg='fastfood')
        dur = time() - t0
        FF[iT] = dur
        print "fast food took", dur, " seconds \n"

        t0  = time()
        I_g   =  gauss_sim(X,r,maxiter,alg='gauss')
        dur = time() - t0
        GP[iT] = dur
        print "Gaussian proj took", dur, " seconds \n"

        t0  = time()
        I_m   =  gauss_sim(X,r,maxiter,alg='mic')
        dur = time() - t0
        HM[iT] = dur
        print "Michael took", dur, " seconds \n"

        t0  = time()
        I_m   =  compress(X,r,0)
        dur = time() - t0
        SC[iT] = dur
        print "Mariano took", dur, " seconds \n"
        #t0  = time()
        #I_s   =  spa(X,r,None)
        #dur = time() - t0
        #SP[iT] = dur
        #print "SPA took", dur, " seconds \n"

        #t0  = time()
        #I_s   =  xray(X,r)
        #dur = time() - t0
        #XR[iT] = dur
        #print "XRAY took", dur, " seconds \n"

        iT += 1
    x = range(two_U-two_L)

    plt.ion()
    plt.plot(x,GP,'gs--')
    plt.plot(x,FF,'bo--')
    plt.plot(x,HM,'ro--')
    plt.plot(x,SC,'k+--')
    plt.xticks(x,[str(label) for label in nList],rotation='horizontal', fontsize=20)
    plt.legend(('Gaussian','FastFood', 'Michael', 'Mariano'), fontsize=22)
    plt.title('Number of Anchors and Samples are %s, %s'%(str(r), str(m)), fontsize=25)
    plt.xlabel('Dimension of data', fontsize=22)
    plt.ylabel('Running times', fontsize=22)
    plt.show(True)
    #plt.savefig('comp_GP_FF.png')
