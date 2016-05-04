from __future__ import division
import numpy as np
import scipy.io
import pylab as plt
import sys
import os, re
import glob
import nipy
from nipy import save_image, load_image
from nipy.core.api import Image
from time import time
from anchor_new import fwht_spiral
from lshash import LSHash
from gauss_sim import Random_Projection
from collections import Counter
#from ica.ica import ica1
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import auc
from sklearn.cross_validation import train_test_split
#from polyssifier import main_data

def orth_nmf(X,r,maxiter,blk=1,W=None,H=None):
    m,n=np.shape(X)
    if not W and not H:
        W, H = init_nmf(X,r)
    Obj=np.zeros(maxiter)
    for i in range(maxiter):
        Obj[i] = np.linalg.norm(X-np.dot(W,H),'fro')
        print'iter:', i+1, 'Obj: ', Obj[i]
        W = update_W(X,W,H)
        for siteNo in range(int(n/blk)):
            H[:,siteNo*blk:(siteNo+1)*blk]=update_H(X,W,H,siteNo,blk)
        #WtX=np.dot(W.T,X)
        #WtW=np.dot(W.T,W)
        #for j in range(4):
        #    H = H*WtX/(np.dot(WtW,H)+np.spacing(1))
    return W,H,Obj

def init_nmf(X,r):
    m,n=np.shape(X)
    #for i in range(r):
    #        W[i*blk:(i+1)*blk,i] = np.random.rand(blk)
    W = np.random.rand(m,r);
    H = np.random.rand(r,n);
    return W,H

def update_W(X,W,H):
        m,n=np.shape(X)
        m,r=np.shape(W)
        cach=np.zeros((m,r))
        HHt=np.zeros((r,r))
        XHt=np.zeros((m,r))
        for siteNo in range(int(n/blk)):
            Xi=X[:,siteNo*blk:(siteNo+1)*blk]
            Hi=H[:,siteNo*blk:(siteNo+1)*blk]
            HHt+=np.dot(Hi,Hi.T)
            XHt+=np.dot(Xi,Hi.T)
            cach+= -np.dot(Xi,Hi.T) + np.dot(W,np.dot(Hi,Hi.T))
        #alg='MU'
        alg='our'
        noise=True
        if noise:
        	HHt=HHt+0.1*np.random.rand(r,r)
        	cach=cach+0.001*np.random.randn(m,r)
        	XHt=XHt+0.0001*np.random.rand(m,r)
        for j in range(10):
            if alg=='our':
            	(W,cach) = W_orth_all(W,HHt,cach);
            if alg=='MU':
	    	W = W*XHt/(np.dot(W,np.dot(XHt.T,W)+np.spacing(1)))
        return W

def update_H(Y,W,H,siteNo,blk):
    m,n=np.shape(Y)
    X=Y[:,siteNo*blk:(siteNo+1)*blk]
    Hblk=H[:,siteNo*blk:(siteNo+1)*blk]
    WtX=np.dot(W.T,X)
    WtW=np.dot(W.T,W)
    for j in range(4):
        Hblk = Hblk*WtX/(np.dot(WtW,Hblk)+np.spacing(1))
    return Hblk

def W_orth_all(W,HHt,cach):
    m,r=np.shape(W)
    C = cach -np.dot(W,HHt)
    d = np.diag(HHt)
    V=np.zeros((m,r))
    obj=np.zeros(r)
    for i in np.random.permutation(range(m)):
        ind = np.argwhere(C[i]<0);
        obj[ind] = -np.square(C[i,ind])/(2*d[ind]+np.spacing(1))
        val=min(obj)
        p=np.argmin(obj)
        if C[i,p]!=0:
            V[i,p] = 2*val/C[i,p]
    cach = cach + np.dot(V-W,HHt)
    W=V
    return (W,cach)

def pre_proc(X, fudgefactor=0.25):
    X = X.T
    # zero mean, unit variance for each voxel
    X = fudgefactor*(X - np.mean(X,axis=0))/np.std(X,axis=0) + fudgefactor
    X = np.clip(X/X.max(),0,1)
    return X.T

def get_files(pathname, patternlist=['H*nii', 'vH*nii', 'S*nii', 'vS*nii']):
    l = []
    for p in patternlist:
        l.extend([x for x in glob.iglob(pathname+'/'+p)])
    return l

def get_masked(pathname, maskfile='groupmeanmask3mm.nii'):
    mask = nipy.load_image(pathname+maskfile)
    idx = np.where(mask.get_data().flatten()==1)[0]
    return idx

def get_X(pathname, patternlist=['H*nii', 'vH*nii', 'S*nii', 'vS*nii']):
    idx = get_masked(pathname)
    l = get_files(pathname, patternlist=patternlist)
    data = np.zeros([len(idx),len(l)]);
    for i in range(0,len(l)):
        print '.',
        d = nipy.load_image(l[i])
        data[:,i] = d.get_data().flatten()[idx]
    print ''
    return pre_proc(data)

def X_to_nii(input, pathname, ONAME, BASENIFTI,
             maskfile='groupmeanmask3mm.nii',
             dims = (52,63,45)):

   # set the variables first
   bnifti = load_image(BASENIFTI)
   mask   = load_image(pathname+maskfile)
   idx    = get_masked(pathname)

   mx       = max(abs(input.flatten()))
   f        = np.zeros(np.prod(mask.shape))
   features  = input.shape[1]

   data = np.zeros((dims)+(features,))

   for i in range(0, features):
      w = input[:,i]
      f[idx] = w
      data[:,:,:,i] = np.reshape(f, dims)

   img = Image.from_image(bnifti,data=data)
   if ONAME != "": save_image(img,pathname+ONAME)
   return img


def fit_test(clf, train_tuple, test_tuple):
    '''
    fit_test function that fits a classifier in train_tuple and
    report AUC results on test_tuple
    The tuples should be given as (data, label)
    '''
    data_train, labels_train = train_tuple
    data_test, labels_test = test_tuple
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    clf.fit(data_train, labels_train)
    fpr, tpr, _ = rc(labels_test, clf.predict_proba(data_test)[:, 1])
    return auc(fpr, tpr)

def classify(data, labels, seed):
    '''
    Classifier function that receives data, labels and a seed for
    random number generation.
    '''
    # Data split in 90% train 10% test
    data_train, data_test, labels_train, labels_test\
        = train_test_split(data, labels, test_size=.1, random_state=seed)

    clf = SVC(kernel='linear', C=0.05, probability=True)

    # Fit on real data
    result = fit_test(clf, (data_train, labels_train),
                      (data_test, labels_test))

    print 'Process %d: (%.2f) AUC' % (seed,result)
    return result


if __name__=='__main__':
    r=9
    numP= 200
    if len(sys.argv)>1:
	maxiter=int(sys.argv[1])
    else:
    	maxiter=30
    #pattern = ['H*nii', 'vH*nii', 'S*nii', 'vS*nii']
    pattern = ['H*nii']
    pathname = '/home/ismav/prog/csnmf/anchor-word-recovery/MRI/'
    bnfti  = pathname+ 'hartford_rest_agg__component_ica_.nii'
    l = get_files(pathname, patternlist=pattern)

    X   = get_X(pathname, patternlist=pattern)
    X   = X.T
    m,n = np.shape(X)
    X   = X/np.outer(np.sum(X,axis=1),np.ones(n))

    #X=scipy.io.loadmat('/scratch/vp335/data/nmf/orlfaces.mat')
    pattern = ['S*nii']
    Y    = get_X(pathname, patternlist=pattern)
    Y    = Y.T
    Y    = Y/np.outer(np.sum(Y,axis=1),np.ones(n))

    cross_auc = np.zeros((10,2))
    topK = 100
    sIdx = 150
    for cIdx in range(10):
        np.random.seed(seed=cIdx)
        #mapRPos   = dict([(v,k) for k,v in enumerate(np.random.permutation(range(len(X))))])
        #mapRNeg   = dict([(v,k) for k,v in enumerate(np.random.permutation(range(len(Y))))])
        randPos   = np.random.permutation(range(len(X)))
        randNeg   = np.random.permutation(range(len(Y)))
        trainList = np.concatenate((randPos[:sIdx],[i+len(X) for i in randNeg[:sIdx]]))
        testList  = np.concatenate((randPos[sIdx:],[i+len(X) for i in randNeg[sIdx:]]))
        Z         = np.concatenate((X,Y))
        ZTr       = Z[trainList]
        ZTe       = Z[testList]
        lTr       = np.concatenate((np.ones(sIdx), -np.ones(sIdx)))
        lTe       = np.concatenate((np.ones(len(X)-sIdx), -np.ones(len(Y)-sIdx)))

        P         = np.zeros((m,numP))
        P         = Random_Projection(ZTr, numP)

        C = Counter(np.argmax(P,axis=0)) + Counter(np.argmin(P,axis=0))
        s=sorted([k for k,v  in C.most_common(r)])[::-1]

        V = sorted(C.values())[::-1]
        L = s
        print sorted(L[:r])

        #A,S  = ica1(ZTr,r)
        ica   = FastICA(n_components=r,max_iter=1000)
        S     = ica.fit_transform(ZTr)
        A     = ica.mixing_
        STe  = np.dot(ZTe, np.linalg.pinv(A).T)
        clf_ica  = LogisticRegression()
        clf_ica.fit(S,lTr)

        main_data(S, lTr)

        iterations = 4
        labels     = np.concatenate((np.ones(len(X)),-np.ones(len(Y))))
        #main_data(Z,labels)
        #results = [classify(Z,labels,seed) for seed in range(iterations)]
        def nmf_fit(X,W,H=None):
            if not H:
                m = len(X)
                r = len(W)
                H    = np.random.rand(len(X),r)
            for i in range(20):
                H = H*np.dot(X,W.T)/(np.dot(H,np.dot(W,W.T))+1e-9)
            return H

        W    = ZTr[L[:r]]
        HTr  = nmf_fit(ZTr,W)
        HTe  = nmf_fit(ZTe,W)
        obj  = np.linalg.norm(ZTr-np.dot(HTr,W),'fro')
        clf_nmf  = LogisticRegression()
        clf_nmf.fit(HTr,lTr)

        main_data(HTr, lTr)
        import ipdb
        ipdb.set_trace()


        cross_auc[cIdx] = [roc_auc_score(lTe, clf_nmf.predict_proba(HTe)[:,1]), roc_auc_score(lTe, clf_ica.predict_proba(STe)[:,1])]
        print cross_auc[cIdx]
    print np.mean(cross_auc[:,0]),np.mean(cross_auc[:,1])

    plt.ion()
    plt.plot(sorted(C.values())[::-1],"s--")
    plt.plot(obj[:-1],"o--")


    X_to_nii(W, pathname,'features_orthNMF_rank_'+str(r)+'.nii', bnfti)
