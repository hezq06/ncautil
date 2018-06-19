"""
Python package for NCA learning algorithm
Algorithm:  Takuya Isomura, Taro Toyoizumi
Developer: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

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
    Approximate data with leading D eigen_vector
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

def pl_eig_pca(data):
    """
    Plot the eigenvalue of covariance
    :param data: data matrix
    :return: eigs
    """
    data=np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1]>=data.shape[0]
    N=data.shape[1]
    Cov= data.dot(data.T) / N
    S = la.svdvals(Cov)
    plt.plot(S, 'b*-')
    plt.show()
    return S

def pl_eig_ppca(data,history=1):
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

def pl_cov(data):
    """
    Plot covariance
    :param data: data matrix
    :return:
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    N = data.shape[1]
    for ii in range(data.shape[1]):
        data[:,ii]=data[:,ii]-np.mean(data[:,ii])
    Cov = data.dot((data).T) / N
    plt.imshow(Cov, cmap='seismic',clim=(-np.amax(np.abs(Cov)),np.amax(np.abs(Cov))))
    plt.xlabel("Data 1")
    plt.ylabel("Data 1")
    plt.title("cov(|D1|,|D1|)")
    plt.colorbar()
    plt.show()

def pl_corr(data):
    """
    Plot correlation
    :param data: data matrix
    :return:
    """
    data = np.array(data)
    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]
    N = data.shape[1]
    for ii in range(data.shape[1]):
        data[:, ii] = data[:, ii] - np.mean(data[:, ii])
    Cov = data.dot((data).T) / N
    d=np.diag(1/np.sqrt(np.diag(Cov)))
    Corr=d.dot(Cov).dot(d)
    plt.imshow(Corr, cmap='seismic', clim=(-np.amax(np.abs(Corr)), np.amax(np.abs(Corr))))
    plt.xlabel("Data 1")
    plt.ylabel("Data 1")
    plt.title("Corr(|D1|,|D1|)")
    plt.colorbar()
    plt.show()

def pl_mucov(data1,data2):
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
    plt.imshow(np.cov(np.abs(data1), np.abs(data2))[0:D, D:2 * D], cmap='seismic')
    plt.xlabel("Data 1")
    plt.ylabel("Data 2")
    plt.title("cov(|D1|,|D2|)")
    plt.colorbar()
    plt.show()

def plot_mat(data,start=0,range=1000):
    data=np.array(data)
    assert len(data.shape) == 2
    img=data[:,start:start+range]
    plt.imshow(img, cmap='seismic')
    plt.colorbar()
    plt.show()

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

def cal_entropy(data):
    """
    Cal entropy of a vector
    :param data:
    :return:
    """
    data=np.array(data)
    data=data/np.sum(data)
    assert len(data.shape) == 1
    ent=-np.sum(data*np.log(data))
    return ent

def cal_kldiv(p,q):
    """
    Cal KL divergence of p over q
    :param data:
    :return:
    """
    p = np.array(p)
    q = np.array(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    assert len(q.shape) == 1
    assert len(p.shape) == 1
    assert p.shape[0] == q.shape[0]
    print(p,q)
    kld=np.sum(p*np.log(p/q))
    return kld

def genDist(N):
    """
    Generate 1D distribution vector of N entry, sum(V)=1
    :param N:
    :return:
    """
    vec=np.random.random(N)
    return vec/np.sum(vec)
