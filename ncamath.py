"""
Python package for NCA utility
Developer: Harry He
Algorithm:  Takuya Isomura, Taro Toyoizumi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time, os, pickle
from sklearn.cluster import KMeans
import scipy.linalg as la
import torch

def cluster(data,n_clusters,mode="kmeans"):
    """
    Do clustering
    :param data: data to be clustered
    :param n_clusters: number of clusters
    :param mode: "kmeans"
    :return:
    """
    startt = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init="random", ).fit(data)
    center=np.zeros((n_clusters,len(data[0])))
    clscounter=np.zeros(n_clusters)
    for iid in range(len(data)):
        ncls=kmeans.labels_[iid]
        center[ncls]=center[ncls]+data[iid]
        clscounter[ncls]=clscounter[ncls]+1
    for iic in range(n_clusters):
        center[iic]=center[iic]/clscounter[iic]
    endt = time.time()
    print("Time used in training:", endt - startt)
    return kmeans.labels_, center

def pca(data,D):
    """
    Principle component analysis
    :param data: dataset of shape (mdim,N)
    :param D: output data dimension
    :return: output data of shape (D,N)
    """
    data=np.array(data)
    assert len(data.shape)==2
    assert data.shape[1]>=data.shape[0]
    assert data.shape[0]>=D

    # PCA
    N=data.shape[1]
    U, S, V = np.linalg.svd(data.dot(data.T) / N)  # eigenvalue decomposition of the covariance matrix of Y(t)
    res = np.diag(1 / np.sqrt(S[0:D])).dot((V[0:D, :])).dot(data)  # Project the input pattern to leading PC directions
    return res

def pca_approx(data,D,dataT=None):
    """
    Approximate data with leading D eigen_vector, the result is an D-rank approximation, not a projection.
    :param data: input textmat
    :param D: rank
    :param dataT: Using eigenvalue of data to project dataT
    :return: data_app
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    assert data.shape[0] >= D

    # PCA
    N = data.shape[1]
    U, S, V = np.linalg.svd(data.dot(data.T) / N)# eigenvalue decomposition of the covariance matrix
    if type(dataT)==type(None):
        print("Project original data")
        res = (V[0:D, :].T).dot((V[0:D, :])).dot(data)  # Project the input pattern to leading PC directions
    else:
        print("Project test data")
        res = (V[0:D, :].T).dot((V[0:D, :])).dot(dataT)
    return res

def pca_proj(data,D,dataT=None):
    """
    Project data with leading D eigen_vector
    :param data: input textmat
    :param D: rank
    :param dataT: Using eigenvalue of data to project dataT
    :return: data_app
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    assert data.shape[0] >= D

    # PCA
    N = data.shape[1]
    U, S, V = np.linalg.svd(data.dot(data.T) / N)# eigenvalue decomposition of the covariance matrix
    pM=np.diag(1/np.sqrt(S[0:D])).dot((V[0:D,:]))
    if type(dataT)==type(None):
        print("Project original data")
        res = pM.dot(data)  # Project the input pattern to leading PC directions
    else:
        print("Project test data")
        res = pM.dot(dataT)
    return res,pM

def ppca_approx(data,D,dataT=None):
    """
    Try to rank decrease using ppca projection of data
    :param data: Training data
    :param D: Dimension
    :param dataT: test data
    :return: data_app
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    assert data.shape[0] >= D

    # PCA
    N = data.shape[1]
    data1 = np.roll(data, 1, axis=1)
    C0 = data1.dot(data1.T) / N
    C1 = data.dot(data1.T) / N
    pCov = C1.dot(np.linalg.pinv(C0)).dot(C1.T)
    U, S, V = np.linalg.svd(pCov)
    if type(dataT)==type(None):
        print("Project original data")
        # res = (V[0:D, :].T).dot((V[0:D, :])).dot(C1).dot(C0).dot(data)# Prodiction, not that working
        res = (V[0:D, :].T).dot((V[0:D, :])).dot(data)
    else:
        print("Project test data")
        # res = (V[0:D, :].T).dot((V[0:D, :])).dot(C1).dot(C0).dot(dataT)
        res = (V[0:D, :].T).dot((V[0:D, :])).dot(dataT)
    return res

def ppca_proj(data,D,dataT=None):
    """
    Try to project data using ppca of data
    :param data: Training data
    :param D: Dimension
    :param dataT: test data
    :return: data_app
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    assert data.shape[0] >= D

    # PCA
    N = data.shape[1]
    data1 = np.roll(data, 1, axis=1)
    C0 = data1.dot(data1.T) / N
    C1 = data.dot(data1.T) / N
    pCov = C1.dot(np.linalg.pinv(C0)).dot(C1.T)
    U, S, V = np.linalg.svd(pCov)
    pM=np.diag(1/np.sqrt(S[0:D])).dot(U[:,0:D].T)
    if type(dataT)==type(None):
        print("Project original data")
        # res = (V[0:D, :].T).dot((V[0:D, :])).dot(C1).dot(C0).dot(data)# Project the input pattern to leading PC directions
        res = pM.dot(data)
    else:
        print("Project test data")
        # res = (V[0:D, :].T).dot((V[0:D, :])).dot(C1).dot(C0).dot(dataT)
        res = pM.dot(dataT)
    return res, pM

def ica(data,LR=1e-2,step=1e4,show_step=1e2):
    """
    Independent component analysis using Amari's learning rule
    :param data: dataset of signal mixing
    :param LR: learning rate
    :param step: learning step
    :return: independent source
    """
    print("Using Amari's ICA rule to do independent component analysis. Don't forget PCA projection beforehand.")
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]

    def g(x):
        res = np.sign(x) * (1 - np.exp(-np.sqrt(2) * np.abs(x))) / 2  # nonlinear function for Laplace distribution
        return res

    data = np.array(data)
    D=data.shape[0]
    N=data.shape[1]
    W = np.eye(D)  # initial weights
    ltab=[]
    for k in range(int(step)):
        if k%show_step==0:
            print("Step "+str(k)+" of "+str(step)+" total step.")
        Xh = W.dot(data)
        DW=((np.eye(D) - g(Xh).dot(Xh.T / N)).dot(W))
        W = W + LR * DW # Amari's ICA rule
        ltab.append(np.linalg.norm(DW))
    Xh = W.dot(data)
    plt.plot(ltab)
    plt.show()
    return Xh,W

def plot_eig_pca(data):
    """
    Plot the eigenvalue of covariance
    :param data: data matrix
    :return: eigs
    """
    startt = time.time()
    data=np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1]>=data.shape[0]
    N=data.shape[1]
    Cov= data.dot(data.T) / N
    S = la.svdvals(Cov)
    endt = time.time()
    print("Time used in calculation:", endt - startt)
    plt.plot(S, 'b*-')
    plt.yscale("log")
    plt.show()
    return S

def plot_eig_ppca(data,history=1):
    """
    Plot the eigenvalue of ppca
    :param data: data matrix
    :return: eigs
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]

    N = data.shape[1]
    data1 = np.roll(data, 1, axis=1)
    if history>1:
        for ii in range(history-1):
            data_shf = np.roll(data, ii+2, axis=1)
            data1=np.concatenate((data1,data_shf))
    C0 = data1.dot(data1.T) / N
    C1 = data.dot(data1.T) / N
    pCov=C1.dot(np.linalg.pinv(C0)).dot(C1.T)
    S = la.svdvals(pCov)
    plt.plot(S, 'b+-')
    plt.show()
    return S

def plot_cov(data,text=None,texty=None):
    """
    Plot covariance
    :param data: data matrix
    :return:
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    N = data.shape[1]
    for ii in range(data.shape[0]):
        data[ii, :] = data[ii, :] - np.mean(data[ii, :])
    Cov = data.dot((data).T) / N
    fig, ax = plt.subplots()
    fig = ax.imshow(Cov, cmap='seismic',clim=(-np.amax(np.abs(Cov)),np.amax(np.abs(Cov))))
    if text != None:
        st, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(st + 0.5, end + 0.5, 1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for ii in range(len(labels)):
            labels[ii] = str(text[ii])
        ax.set_xticklabels(labels, rotation=70)
    if texty != None:
        st, end = ax.get_ylim()
        if st < end:
            ax.yaxis.set_ticks(np.arange(st + 0.5, end + 0.5, 1))
        else:
            ax.yaxis.set_ticks(np.arange(end + 0.5, st + 0.5, 1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        labels = [item.get_text() for item in ax.get_yticklabels()]
        for ii in range(len(labels)):
            labels[ii] = str(texty[ii])
        ax.set_yticklabels(labels, rotation=0)
    plt.xlabel("Data 1")
    plt.ylabel("Data 1")
    plt.title("cov(|D1|,|D1|)")
    plt.colorbar(fig)
    plt.show()
    return Cov

def plot_corr(data,text=None,texty=None):
    """
    Plot correlation
    :param data: data matrix
    :return:
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    N = data.shape[1]
    for ii in range(data.shape[0]):
        mn = np.mean(data[ii, :])
        data[ii, :] = data[ii, :] - mn
    Cov = data.dot((data).T) / N
    d=np.diag(1/np.sqrt(np.diag(Cov)))
    Corr=d.dot(Cov).dot(d)
    fig, ax = plt.subplots()
    fig = ax.imshow(Corr, cmap='seismic', clim=(-np.amax(np.abs(Corr)), np.amax(np.abs(Corr))))
    if text != None:
        st, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(st + 0.5, end + 0.5, 1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for ii in range(len(labels)):
            labels[ii] = str(text[ii])
        ax.set_xticklabels(labels, rotation=70)
    if texty != None:
        st, end = ax.get_ylim()
        if st < end:
            ax.yaxis.set_ticks(np.arange(st + 0.5, end + 0.5, 1))
        else:
            ax.yaxis.set_ticks(np.arange(end + 0.5, st + 0.5, 1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        labels = [item.get_text() for item in ax.get_yticklabels()]
        for ii in range(len(labels)):
            labels[ii] = str(texty[ii])
        ax.set_yticklabels(labels, rotation=0)
    plt.xlabel("Data 1")
    plt.ylabel("Data 1")
    plt.title("corr(|D1|,|D1|)")
    plt.colorbar(fig)
    plt.show()
    return Corr

def plot_mucov(data1,data2):
    """
    Plot mutual covariance
    :param data1: data matrix 1
    :param data2: data matrix 2
    :return:
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    assert len(data1.shape) == 2
    assert len(data2.shape) == 2
    assert data1.shape[1] >= data1.shape[0]
    assert data2.shape[1] >= data2.shape[0]
    assert data1.shape==data2.shape
    D=data1.shape[0]
    muCov=np.cov(np.abs(data1), np.abs(data2))[0:D, D:2 * D]
    plt.imshow(muCov, cmap='seismic', clim=(-np.amax(muCov), np.amax(muCov)))
    plt.xlabel("Data 1")
    plt.ylabel("Data 2")
    plt.title("cov(|D1|,|D2|)")
    plt.colorbar()
    plt.show()

def cal_cosdist(v1,v2):
    """
    Calculate cosine distance between two word embedding vectors
    :param self:
    :param v1:
    :param v2:
    :return:
    """
    return np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)

def cal_pdist(data):
    """
    Cal probabilistic distribution distance of data
    :param data:
    :return:
    """
    data = np.array(data)
    data = data / np.sum(data)
    assert len(data.shape) == 1
    orin=np.ones(len(data))/len(data)
    dist=np.sum(np.abs(data-orin))*np.sum(np.abs(data-orin))/2
    return dist

def cal_entropy(data,logit=False):
    """
    Cal entropy of a vector
    :param data:
    :param logit: if input data is logit mode
    :return:
    """
    if logit:
        data=np.exp(data)
    else:
        data=np.array(data)+1e-9
    data=data/np.sum(data)
    assert len(data.shape) == 1
    assert np.min(data)>0
    ent=-np.sum(data*np.log(data))
    return ent

def cal_kldiv(p,q):
    """
    Cal KL divergence of p over q
    :param data:
    :return:
    """
    p = np.array(p)+1e-9
    q = np.array(q)+1e-9
    p = p / np.sum(p)
    q = q / np.sum(q)
    assert len(q.shape) == 1
    assert len(p.shape) == 1
    assert p.shape[0] == q.shape[0]
    assert np.min(p)>0
    assert np.min(q)>0
    kld=np.sum(p*np.log(p/q))
    return kld

def cal_kappa_stat(seq1,seq2):
    """
    Calculate kappa statistics of two one-hot matrix
    from "Pruning Adaptive Boosting"
    :param seq1:
    :param seq2:
    :return:
    """
    assert len(seq1.shape)==2
    assert seq1.shape == seq2.shape
    assert len(seq1)>len(seq1[0])
    Cij=seq1.T.dot(seq2)
    m=len(seq1)
    L=len(seq1[0])
    Theta1=np.trace(Cij)/m
    Theta2=0
    for ii in range(L):
        t1=0
        t2=0
        for jj in range(L):
            t1=t1+Cij[ii,jj]
            t2=t2+Cij[jj,ii]
        Theta2=Theta2+t1/m*t2/m
    kappa=(Theta1-Theta2)/(1-Theta2)
    print(Cij, Theta1-1/L, Theta2)
    return kappa

def genDist(N):
    """
    Generate 1D distribution vector of N entry, sum(V)=1
    :param N:
    :return:
    """
    vec=np.random.random(N)
    return vec/np.sum(vec)

def sigmoid(x):
    res=np.exp(x)/(np.exp(x)+1)
    return res

def relu(x):
    x=np.array(x)
    res=(x+np.abs(x))/2
    return res

# def softmax(x,dim=-1):
#     NAN error
#     xdown=(torch.sum(torch.exp(x),keepdim=True,dim=dim)+1e-9)
#     res=torch.exp(x)/xdown
#     if torch.isnan(res).any():
#         save_data(x, file="data_output1.pickle")
#         save_data(xdown, file="data_output2.pickle")
#         raise Exception("NaN Error 1")
#     return res

def softmax(x,dim=-1):
    sfm=torch.nn.Softmax(dim=dim)
    res=sfm(x)
    if torch.isnan(res).any():
        raise Exception("NaN Error 1")
    return res

####### Section Logit Dynamic study

def build_basis(n):
    """
    A handy set of basis for logit (normalized), v1: (1,-1/(n-1),-1/(n-1),...)/L, v2: (0,1,-1/(n-2),-1/(n-2)...)
    :param n: number of classes
    :return: a base vector with length n-1
    """
    vecb=[]
    for iib in range(n-1):
        veciib=np.zeros(n)
        veciib[iib]=1
        for iibd in range(n-iib-1):
            veciib[iib+iibd+1]=-1/(n-1-iib)
        vecb.append(np.array(veciib)/np.linalg.norm(veciib))
    return np.array(vecb)

def proj(vec,vecb):
    """
    Project vec onto predifined orthoganal vecbasis
    """
    res=[]
    vec=np.array(vec)
    for base in vecb:
        base=np.array(base)/np.linalg.norm(base)
        xp=vec.dot(base)
        res.append(xp)
    return np.array(res)

def aproj(vec,vecb):
    """
    Reverse operation of projection
    :param vec:
    :param vecb:
    :return:
    """
    res=np.zeros(len(vecb[0]))
    for ii in range(len(vec)):
        res=res+vec[ii]*np.array(vecb[ii])
    return res

def logit_space_transfer(seq):
    """
    Transfer a logit sequence to non-redundant sub-space
    :param seq: a sequence of logit
    :return: transfered sequence
    """
    seq=np.array(seq)
    assert len(seq.shape)==2
    assert seq.shape[0]>seq.shape[1]
    resp = []
    basis = build_basis(len(seq[0]))
    for vec in seq:
        vecp = proj(vec, basis)
        resp.append(vecp)
    return resp

def logit_space_atransfer(seq):
    """
    Reverse transfer a non-redundant sub-space vec back to logit sequence
    :param seq: a sequence non-redundant sub-space vec
    :return: transfered back sequence
    """
    seq=np.array(seq)
    assert len(seq.shape)==2
    resp = []
    basis = build_basis(len(seq[0])+1)
    for vec in seq:
        vecp = aproj(vec, basis)
        resp.append(vecp)
    return resp