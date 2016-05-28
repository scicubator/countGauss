from __future__ import division
import numpy as np
from time import time
import matplotlib
import pylab as plt
from collections import Counter
from fcube.fcube import countGauss_projection


def gauss_sim(X, k, maxiter=10, alg='gauss'):
    m, n = np.shape(X)
    idxT = Counter()
    for _ in range(maxiter):
        if alg == 'gauss':
            G = np.random.randn(n, k)
            Z = np.dot(X, G)
        if alg == 'countGauss':
            Z = countGauss_projection(X, k)
        indMax = np.argmax(Z, axis=0)
        idxT.update(indMax)
        indMin = np.argmin(Z, axis=0)
        idxT.update(indMin)
    B = sorted([key for key, val in idxT.most_common(k)])
    return B


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


def run_nmf(X, r, maxiter, alg):
    m, n = np.shape(X)
    I = gauss_sim(X, r, maxiter, alg=alg)
    U = np.random.rand(m, r)
    V = np.copy(X[I])
    print iT, np.shape(X), np.shape(U), np.shape(V)
    print I
    for _ in range(100):
        U = U * (np.dot(X, V.T) / (np.dot(U, np.dot(V, V.T)) + 1e-9))
    Obj = np.linalg.norm(X - np.dot(U, V), 'fro') / np.linalg.norm(X, 'fro')
    return Obj, I, U, V


if __name__ == '__main__':
    m = 22
    n = 3226
    maxiter = 5
    norm = "l1"

    rList = range(1, 20)
    # maxiter= range(5,300,5)
    # R      = 500
    maxiter = 30
    R = 100
    f = open("data/log_brca3226.txt")
    d = f.readlines()
    X = np.zeros((n, m))
    for i in range(n):
        X[i] = np.array(
            [np.exp(float(x)) for j, x in enumerate(d[i].split("\t")) if j])
    X = X.T
    for j in range(m):
        if norm == "l2":
            X[j, :] = X[j, :] / np.linalg.norm(X[j, :])
        if norm == "l1":
            X[j, :] = X[j, :] / np.sum(X[j, :])

    F_f = {}
    F_g = {}
    FF = np.zeros(len(rList))
    GP = np.zeros(len(rList))
    Obj_f = np.zeros(len(rList))
    Obj_g = np.zeros(len(rList))
    for iT, r in enumerate(rList):
        t0 = time()
        # Obj_f[iT], F_f[iT] , U_f, V_f =  run_nmf(X,r,maxiter,alg='fastfood')
        Obj_f[iT], F_f[iT], U_f, V_f = run_nmf(X, r, maxiter, alg='countGauss')
        dur = time() - t0
        FF[iT] = dur
        t0 = time()
        Obj_g[iT], F_g[iT], U_g, V_g = run_nmf(X, r, maxiter, alg='gauss')
        dur = time() - t0
        GP[iT] = dur
    plt.ion()

    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    plt.plot(Obj_f, "ro--", markersize=15)
    plt.plot(Obj_g, "bs--", markersize=15)
    plt.xlabel("Extreme points selected", fontsize=40)
    plt.ylabel("Relative error", fontsize=38)
    # plt.title("Breast cancer dataset", fontsize=50)
    plt.legend(['CountGauss', 'GP'], fontsize=30)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("tests/breast.pdf", transparent=True, bbox_inches='tight',
                pad_inches=0)

    # plt.figure(2)
    # plt.plot(FF,"ro--")
    # plt.plot(GP,"bs--")
    # plt.xlabel("Number of extreme points selected")
    # plt.ylabel("Running time")
    # plt.title("NMF on Breast cancer dataset")
    # plt.legend(['fastfood', 'gauss'])
    # import ipdb
    # ipdb.set_trace()
