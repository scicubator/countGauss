from __future__ import division
import numpy as np
#from fht import fht,fht2
#import scipy.linalg
from time import time
#from anchor_new import fwht_spiral
import pylab as plt
from lshash import LSHash
from collections import Counter
from fcube.fcube import countGauss_projection, countSketch_projection

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
    B =  sorted([key for key,val in idxT.most_common(k)])
    return B
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

def run_nmf(X,r, maxiter, alg):
        m,n = np.shape(X)
        I   =  gauss_sim(X,r,maxiter,alg=alg)
        U = np.random.rand(m,r)
        V = np.copy(X[I])
        print iT, np.shape(X), np.shape(U), np.shape(V)
        print I
        for p in range(100):
            U = U* (np.dot(X,V.T)/(np.dot(U, np.dot(V,V.T)) + 1e-9))
        Obj = np.linalg.norm(X-np.dot(U,V),'fro')/np.linalg.norm(X,'fro')
        return Obj, I , U, V

if __name__ == '__main__':
    m   = 22
    n   = 3226
    maxiter = 5
    norm  ="l1"

    rList  = range(1,20)
    #maxiter= range(5,300,5)
    #R      = 500
    maxiter= 30
    R      = 100
    f=open("log_brca3226.txt")
    d=f.readlines()
    X=np.zeros((n,m))
    for i in range(n):
        X[i] = np.array([np.exp(float(x)) for j,x in enumerate(d[i].split("\t")) if j])
    X = X.T
    for j in range(m):
        if norm=="l2":
            X[j,:] = X[j,:]/np.linalg.norm(X[j,:])
        if norm=="l1":
            X[j,:] = X[j,:]/np.sum(X[j,:])

    F_f = {}
    F_g = {}
    FF  = np.zeros(len(rList))
    GP  = np.zeros(len(rList))
    Obj_f=np.zeros(len(rList))
    Obj_g=np.zeros(len(rList))
    for iT, r in enumerate(rList):

        t0  = time()
        #Obj_f[iT], F_f[iT] , U_f, V_f =  run_nmf(X,r,maxiter,alg='fastfood')
        Obj_f[iT], F_f[iT] , U_f, V_f =  run_nmf(X,r,maxiter,alg='countGauss')
        dur = time() - t0
        FF[iT] = dur
        t0  = time()
        Obj_g[iT], F_g[iT] , U_g, V_g =  run_nmf(X,r,maxiter,alg='gauss')
        dur = time() - t0
        GP[iT] = dur
    plt.ion()
    import matplotlib
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    plt.plot(Obj_f,"ro--", markersize=10)
    plt.plot(Obj_g,"bs--", markersize=10)
    plt.xlabel("Number of extreme points selected", fontsize=40)
    plt.ylabel("Relative error", fontsize=38)
    plt.title("NMF on Breast cancer dataset", fontsize=50)
    plt.legend(['CountGauss', 'GP'],fontsize=50)
    plt.gcf().subplots_adjust(bottom=0.15)

    #plt.figure(2)
    #plt.plot(FF,"ro--")
    #plt.plot(GP,"bs--")
    #plt.xlabel("Number of extreme points selected")
    #plt.ylabel("Running time")
    #plt.title("NMF on Breast cancer dataset")
    #plt.legend(['fastfood', 'gauss'])
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
