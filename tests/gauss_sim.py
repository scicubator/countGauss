from __future__ import division
import numpy as np
from fht import fht
from time import time
import pylab as plt
#import seaborn as sns
from lshash import LSHash
from csnmf.third_party.mrnmf.nmf_process_algorithms import xray,spa
from csnmf.compression import compress
from fcube.fcube import fcube_projection, countGauss_projection, countSketch_projection

def gauss_sim(X,k,maxiter=10, alg='gauss'):
    m,n  = np.shape(X)
    Idx  = set()
    for iT in range(maxiter):
        if alg == 'gauss':
            G    = np.random.randn(n,k)
            Z    = np.dot(X,G)
        if alg == 'fastfood':
            Z    = fcube_projection(X,k)
        if alg == 'countGauss':
            Z    = countGauss_projection(X,k)
        if alg == 'countSketch':
            Z    = countSketch_projection(X,k)
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
    #maxiter = 4
    maxiter = 1

    #n   =  4096
    two_L =  8
    two_U =  18
    FF    = np.zeros(len(range(two_L,two_U)))
    GP    = np.zeros(len(range(two_L,two_U)))
    HM    = np.zeros(len(range(two_L,two_U)))
    SP    = np.zeros(len(range(two_L,two_U)))
    XR    = np.zeros(len(range(two_L,two_U)))
    SC    = np.zeros(len(range(two_L,two_U)))
    CG    = np.zeros(len(range(two_L,two_U)))
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
        I_g   =  gauss_sim(X,r,maxiter,alg='gauss')
        dur = time() - t0
        GP[iT] = dur
        print "Gaussian proj took", dur, " seconds \n"

        #t0  = time()
        #I_m   =  compress(X,r,0)
        #dur = time() - t0
        #SC[iT] = dur
        #print "Mariano took", dur, " seconds \n"

        t0  = time()
        I_f   =  gauss_sim(X,r,maxiter,alg='countGauss')
        dur = time() - t0
        FF[iT] = dur
        print "Count Gauss took", dur, " seconds \n"

        t0  = time()
        I_f   =  gauss_sim(X,r,maxiter,alg='countSketch')
        dur = time() - t0
        CG[iT] = dur
        print "Count sketch took", dur, " seconds \n"
        #t0  = time()
        #I_m   =  gauss_sim(X,r,maxiter,alg='mic')
        #dur = time() - t0
        #HM[iT] = dur
        #print "Michael took", dur, " seconds \n"

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
    y = ['$2^'+str(i)+'$' for i in range(two_L,two_U)]


    FS=60
    MS=30
    plt.ion()
    plt.plot(x,GP,'gs--', markersize=MS)
    #plt.plot(x,SC,'k+--', markersize=MS)
    plt.plot(x,FF,'bo--', markersize=MS)
    plt.plot(x,CG,'ro--', markersize=MS)
    #plt.plot(x,HM,'ro--')
    #plt.xticks(x,[str(label) for label in nList], rotation='horizontal', fontsize=30)
    plt.xticks(x,["$2^"+"{" + str(i)+ "}"+ "$" for i in range(two_L, two_U)], rotation='horizontal', fontsize=FS)
    plt.yticks(fontsize=FS)
    #ax.xticks(y,rotation='horizontal', fontsize=20)
    #plt.legend(('GP','SC', 'CountGauss', 'CountSketch'), fontsize=45, loc='upper left')
    plt.legend(('GP','CountGauss', 'CountSketch'), fontsize=FS, loc='upper left')
    plt.title('Anchors:  %s,  Samples: %s'%(str(r), str(m)), fontsize=FS)
    plt.xlabel('Dimension of data', fontsize=FS)
    plt.ylabel('Running times (seconds)', fontsize=FS)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show(True)

    #plt.savefig('comp_GP_FF.png')
    #plt.savefig('figures/fcube_GP_SC.pdf',transparent=False, bbox_inches='tight', pad_inches=0)
