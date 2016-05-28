from __future__ import division
import numpy as np
import pylab as plt
import timeit
from fcube.fcube import countGauss_projection, countSketch_projection


def gauss_sim(X, k, maxiter=10, alg='gauss'):
    m, n = np.shape(X)
    idx = set()
    for _ in range(maxiter):
        if alg == 'gauss':
            G = np.random.randn(n, k)
            Z = np.dot(X, G)
        if alg == 'countGauss':
            Z = countGauss_projection(X, k)
        if alg == 'countSketch':
            Z = countSketch_projection(X, k)
        ind = np.argmax(Z, axis=0)
        idx = set(ind).union(idx)
        ind = np.argmin(Z, axis=0)
        idx = set(ind).union(idx)
        # idx = idx.union(np.argmax(Z, axis=0), np.argmin(Z, axis=0))
        # print Idx
    return idx


if __name__ == '__main__':
    r = 200
    m = 400
    # maxiter = 4
    maxiter = 1

    # n   =  4096
    two_L = 8
    two_U = 18
    range_L_U = range(two_L, two_U)
    FF = np.zeros(len(range_L_U))
    GP = np.zeros(len(range_L_U))
    HM = np.zeros(len(range_L_U))
    SP = np.zeros(len(range_L_U))
    XR = np.zeros(len(range_L_U))
    SC = np.zeros(len(range_L_U))
    CG = np.zeros(len(range_L_U))

    nList = [np.power(2, i) for i in range_L_U]
    for iT, n in enumerate(nList):
        U = np.random.rand(m, r)
        V = np.random.rand(r, n)
        U[:r, :r] = np.eye(r)
        # for i in range(m):
        #     V[:,i] = V[:,i]/np.sum(V[:,i])

        X = np.dot(U, V)

        t0 = timeit.default_timer()
        I_g = gauss_sim(X, r, maxiter, alg='gauss')
        dur = timeit.default_timer() - t0
        GP[iT] = dur
        print "Gaussian proj took", dur, " seconds \n"

        # t0  = time()
        # I_m   =  compress(X,r,0)
        # dur = time() - t0
        # SC[iT] = dur
        # print "Mariano took", dur, " seconds \n"

        t0 = timeit.default_timer()
        I_f = gauss_sim(X, r, maxiter, alg='countGauss')
        dur = timeit.default_timer() - t0
        FF[iT] = dur
        print "Count Gauss took", dur, " seconds \n"

        t0 = timeit.default_timer()
        I_f = gauss_sim(X, r, maxiter, alg='countSketch')
        dur = timeit.default_timer() - t0
        CG[iT] = dur
        print "Count sketch took", dur, " seconds \n"

        # t0  = timeit.default_timer()
        # I_s   =  spa(X,r,None)
        # dur = timeit.default_timer() - t0
        # SP[iT] = dur
        # print "SPA took", dur, " seconds \n"

        # t0  = timeit.default_timer()
        # I_s   =  xray(X,r)
        # dur = timeit.default_timer() - t0
        # XR[iT] = dur
        # print "XRAY took", dur, " seconds \n"

    x = range(two_U - two_L)
    y = ['$2^' + str(i) + '$' for i in range_L_U]

    FS = 60
    MS = 30
    plt.ion()
    plt.plot(x, GP, 'gs--', markersize=MS)
    # plt.plot(x,SC,'k+--', markersize=MS)
    plt.plot(x, FF, 'bo--', markersize=MS)
    plt.plot(x, CG, 'ro--', markersize=MS)
    # plt.plot(x, HM, 'ro--')
    # plt.xticks(x, [str(label) for label in nList], rotation='horizontal',
    #            fontsize=30)
    plt.xticks(x, ["$2^" + "{" + str(i) + "}" + "$" for i in range_L_U],
               rotation='horizontal', fontsize=FS)
    plt.yticks(fontsize=FS)
    # ax.xticks(y,rotation='horizontal', fontsize=20)
    # plt.legend(('GP','SC', 'CountGauss', 'CountSketch'), fontsize=45,
    #            loc='upper left')
    plt.legend(('GP', 'CountGauss', 'CountSketch'), fontsize=FS,
               loc='upper left')
    plt.title('Anchors: {0},  Samples: {1}'.format(r, m), fontsize=FS)
    plt.xlabel('Dimension of data', fontsize=FS)
    plt.ylabel('Running times (seconds)', fontsize=FS)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show(True)

    # plt.savefig('comp_GP_FF.png')
    # plt.savefig('figures/fcube_GP_SC.pdf', transparent=False,
    #             bbox_inches='tight', pad_inches=0)
