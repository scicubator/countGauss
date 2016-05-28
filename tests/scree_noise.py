from __future__ import division
import numpy as np
from collections import Counter
from fcube.fcube import countGauss_projection, countSketch_projection


def gauss_sim(X, k, alg='gauss'):
    m, n = np.shape(X)
    idxT = Counter()
    if alg == 'gauss':
        G = np.random.randn(n, k)
        Z = np.dot(X, G)
    if alg == 'countGauss':
        Z = countGauss_projection(X, k)
    indMax = np.argmax(Z, axis=0)
    idxT.update(indMax)
    indMin = np.argmin(Z, axis=0)
    idxT.update(indMin)
    return idxT


def gen_data_noise(m, n, r, sigma):
    U = np.zeros((m, r))
    V = np.random.rand(r, n)
    U[:r, :r] = np.eye(r)
    cnt = r
    for j in range(r):
        for k in range(r):
            if k < j:
                U[cnt, j] = 1
                U[cnt, k] = 1
                cnt += 1
    for i in range(m):
        U[i, :] = U[i, :] / np.sum(U[i, :])
    X = np.dot(U, V) + sigma * np.random.randn(m, n)
    return X, U, V


if __name__ == '__main__':
    m = 210
    n = 1000
    r = 20

    sList = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.22, 0.36, 0.6, 1]
    R = 100
    succ = np.zeros((m, R * len(sList)))
    for iT, sigma in enumerate(sList):
        for run in range(R):
            X, U, V = gen_data_noise(m, n, r, sigma)

            k = int(2 * r * np.log(r))
            # I   =  gauss_sim(X,k,alg='countGauss')
            I = gauss_sim(X, k, alg='gauss')
            maxV = I.most_common(1)[0][1]
            for key, val in I.iteritems():
                succ[key, R * iT + run] = val / maxV
            print "noise, runs", iT, run

    succG = np.zeros((m, R * len(sList)))
    for iT, sigma in enumerate(sList):
        for run in range(R):
            X, U, V = gen_data_noise(m, n, r, sigma)

            k = int(2 * r * np.log(r))
            # I   =  gauss_sim(X,k,alg='countGauss')
            I = gauss_sim(X, k, alg='gauss')
            maxV = I.most_common(1)[0][1]
            for key, val in I.iteritems():
                succG[key, R * iT + run] = val / maxV
            print "noise, runs", iT, run
    import pickle

    D = {}
    D['gauss'] = succ
    D['countGauss'] = succG
    pickle.dump(D, open("tests/noise.pkl", "wb"))
