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
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def cluster(data,n_clusters,mode="kmeans",sample_weight=None, ii_iter=1000):
    """
    Do clustering
    :param data: data to be clustered
    :param n_clusters: number of clusters
    :param mode: "kmeans"
    :return:
    """
    startt = time.time()
    if mode == "kmeans":
        if np.std(data)>0:
            kmeans = KMeans(n_clusters=n_clusters, init="random",max_iter=3000, tol=1e-5).fit(data,sample_weight=sample_weight)
            center=np.zeros((n_clusters,len(data[0])))
            clscounter=np.zeros(n_clusters)
            for iid in range(len(data)):
                ncls=kmeans.labels_[iid]
                center[ncls]=center[ncls]+data[iid]
                clscounter[ncls]=clscounter[ncls]+1
            for iic in range(n_clusters):
                if clscounter[iic] != 0:
                    center[iic]=center[iic]/clscounter[iic]
            res = (kmeans.labels_, center)
        else:
            rndlabels=np.floor(np.random.rand(len(data)) * n_clusters)
            center=[data[0] for ii in range(n_clusters)]
            res = (rndlabels, center)
    elif mode == "b-kmeans": # Balanced k-means
        N, dim = data.shape
        cen = np.random.random((n_clusters,dim))
        for ii_step in range(1):
            print("A")
            ## Calculate cost matrix
            diffmat=data.reshape(N,1,dim)-cen.reshape(1,n_clusters,dim)# N,n_cluster,dim
            print("B")
            costmat=np.sum(diffmat*diffmat,axis=2)# N,n_cluster
            nrep=np.ceil(N/n_clusters)
            costmat=np.repeat(costmat, nrep, axis=1)
            print("C")
            rid, cid = linear_sum_assignment(costmat) ## Toooooo slow
    endt = time.time()
    print("Time used in training:", endt - startt)
    return res

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
    startt = time.time()
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
    endt = time.time()
    print("Time used in ica:", endt - startt)
    plt.plot(ltab)
    plt.show()
    return Xh,W

def ica_torch(data,LR=1e-2,step=1e4,show_step=1e2,cuda_device="cuda:0"):
    """
    Independent component analysis using Amari's learning rule
    :param data: dataset of signal mixing
    :param LR: learning rate
    :param step: learning step
    :return: independent source
    """
    startt = time.time()

    gpuavail = torch.cuda.is_available()
    print("Using Amari's ICA rule to do independent component analysis. Don't forget PCA projection beforehand.")
    data = torch.from_numpy(np.array(data))

    assert len(data.shape) == 2
    assert data.shape[1] >= data.shape[0]

    D = data.shape[0]
    N = data.shape[1]
    ltab = []

    W = torch.eye(D)  # initial weights
    EYE = torch.eye(D)  # initial weights
    if gpuavail:
        data=data.to(cuda_device)
        W = W.to(cuda_device)
        EYE = EYE.to(cuda_device)

    def g(x):
        res = torch.sign(x) * (1 - torch.exp(-np.sqrt(2) * torch.abs(x))) / 2  # nonlinear function for Laplace distribution
        return res

    for k in range(int(step)):
        if k%show_step==0:
            print("Step "+str(k)+" of "+str(step)+" total step.")
        Xh = torch.matmul(W,data)
        DW_R=torch.matmul(g(Xh),(torch.t(Xh) / N))
        DW = torch.matmul(EYE-DW_R,W)
        W2 = W + LR * DW # Amari's ICA rule
        W2 = W2 / (torch.norm(W2) / torch.norm(EYE))
        ltab.append(torch.norm(W2-W))
        W=W2
    Xh = torch.matmul(W,data)

    endt = time.time()
    print("Time used in ica:", endt - startt)

    plt.plot(ltab)
    plt.show()
    return Xh.cpu().numpy(),W.cpu().numpy()

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

def plot_corr(data,text=None,texty=None, ax=None,title=None):
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
    d=np.diag(1/(np.sqrt(np.diag(Cov))+1e-9))
    Corr=d.dot(Cov).dot(d)
    if ax is None:
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
    if title is None:
        plt.title("corr(|D1|,|D1|)")
    else:
        plt.title(title)
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

def pltfft(data):
    """
    Plot fft spectrum
    :param data:
    :return:
    """
    N = len(data)
    data=np.array(data).reshape(1,-1)
    fft = np.fft.rfft(data - np.mean(data),norm='ortho')
    x = np.array(list(range(len(fft[0])))) * 2 * np.pi / N
    y = abs(fft)[0]
    plt.plot(x, y)
    plt.show()

def pltsne(data,D=2,perp=1,labels=None):
    """
    Plot 2D w2v graph with tsne
    Referencing part of code from: Basic word2vec example tensorflow
    :param numpt: number of points
    :return: null
    """
    tsnetrainer = TSNE(perplexity=perp, n_components=D, init='pca', n_iter=5000, method='exact')
    tsne = tsnetrainer.fit_transform(data)
    plt.figure()
    for i in range(len(data)):
        x, y = tsne[i, :]
        plt.scatter(x, y)
        if labels is not None:
            label=labels[i]
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    return tsnetrainer

def cal_cosdist(v1,v2):
    """
    Calculate cosine distance between two word embedding vectors
    :param self:
    :param v1:
    :param v2:
    :return:
    """
    return np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)

def cal_hist(data,bins=10,range=None):
    """
    calculate a histogram of data
    :param data:
    :return:
    """
    if len(data.shape)==1: #1D case
        dist,bins=np.histogram(data, bins=bins, range=range, density=True)
        return dist / np.sum(dist), bins
    elif len(data.shape)==2: #2D case
        assert data.shape[1]==2
        dist, binsx,  binsy = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=range)
        return dist / np.sum(dist), binsx,  binsy

def cal_pdistance(data):
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

def cal_entropy(data,log_flag=False,byte_flag=False, torch_flag=False, cuda_device="cuda:1"):
    """
    Cal entropy of a vector
    :param data:
    :param log_flag: if input data is log probability
    :return:
    """
    adj = 1
    if byte_flag:
        adj = np.log(2)
    if not torch_flag:
        data=np.array(data)
        if log_flag:
            data=np.exp(data)
        # assert len(data.shape) == 1
        data_adj=np.zeros(data.shape)+data
        data_adj[data_adj==0]=1e-9
        data_adj = data_adj / np.sum(data_adj, axis=-1,keepdims=True)
        ent=-np.sum(data_adj*np.log(data_adj)/adj,axis=-1)
    else:
        data=data.to(cuda_device)
        if log_flag:
            data=torch.exp(data)
        data[data == 0] = 1e-9
        data = data / torch.sum(data, dim=-1, keepdim=True)
        ent = -torch.sum(data * torch.log(data) / adj, dim=-1)
    return ent

def cal_entropy_raw(data,data_discrete=True,data_bins=None):
    """
    Calculate entropy of raw data
    :param data: [Ndata of value]
    :param data_discrete: if data is discrete
    :return:
    """
    data=np.array(data)
    assert len(data.shape) == 1
    if data_discrete:
        if len(set(data))<=1:
            return 0
        np_data_bins=np.array(list(set(data)))
        data_bins_s = np.sort(np_data_bins)
        prec=data_bins_s[1]-data_bins_s[0]
        data_bins = np.concatenate((data_bins_s.reshape(-1,1),np.array(data_bins_s[-1]+prec).reshape(1,1)),axis=0)
        data_bins=data_bins.reshape(-1)
        data_bins=data_bins-prec/2
    pdata, _ = np.histogram(data, bins=data_bins)
    pdata=pdata/np.sum(pdata)
    return cal_entropy(pdata)

def cal_entropy_raw_ND_discrete(data):
    """
    Calculate entropy of raw data
    N dimensional discrete data
    :param data: [Ndata of D-dim value]
    :param data_discrete: if data is discrete
    :return:
    """
    datanp=np.array(data)
    if len(datanp.shape) == 1:
        datanp=datanp.reshape(-1,1)
    assert len(datanp.shape) == 2
    assert datanp.shape[0]>datanp.shape[1]

    # datatup = []
    # for iin in range(len(datanp)):
    #     datatup.append(tuple(datanp[iin]))
    # datatup= [tuple(datanp[iin]) for iin in range(len(datanp))] # Slow!!!

    projV=np.random.random(datanp.shape[1])
    datatup=datanp.dot(projV)

    itemsets = list(set(datatup))
    nsets = len(itemsets)

    hashtab = dict([])
    for ii in range(nsets):
        hashtab[itemsets[ii]] = 0

    for ii in range(len(datatup)):
        hashtab[datatup[ii]] = hashtab[datatup[ii]] + 1

    pvec = np.zeros(nsets)
    for ii, val in enumerate(hashtab.values()):
        pvec[ii] = val

    pvec = pvec / np.sum(pvec)

    return cal_entropy(pvec)

def cal_entropy_gauss(theta):
    """
    calculate the multi-variate entropy of theta
    :param theta:
    :return:
    """
    if type(theta) is float:
        ent = np.log(theta*np.sqrt(2*np.pi*np.e))
    elif type(theta) is list:
        ent=0
        for item in theta:
            ent=ent+np.log(item*np.sqrt(2*np.pi*np.e))
    return ent

def cal_entropy_gauss_gpu(theta,cuda_device="cuda:0"):
    """
    gpuversion of calculating gauss entropy average
    :param theta:
    :return:entropy
    """
    assert theta.shape[0]>theta.shape[1]
    pttheta=torch.FloatTensor(theta).to(cuda_device)
    res_ent=torch.log(pttheta * np.sqrt(2 * np.pi * np.e))
    res_ent=torch.sum(res_ent,dim=-1)
    res_ent=torch.mean(res_ent)
    return res_ent.cpu().item()

def cal_entropy_continous_raw(data,bins=10000):
    data = np.array(data)
    assert len(data.shape) == 1
    # entl=[]
    # for ii in tqdm(range(30)):
    #     nbins=int(1.2**(ii+55))
    #     pdata, bins = np.histogram(data, bins=nbins)
    #     pdata = pdata / np.sum(pdata)
    #     ent= cal_entropy(pdata)
    #
    #     entl.append(ent+np.log(dbin))
        # entl.append(ent)
    pdata, bins = np.histogram(data, bins=bins)
    dx = bins[1] - bins[0]
    pdata = pdata / np.sum(pdata)/dx
    ent=0
    for ii in range(len(pdata)):
        if pdata[ii]>0:
            ent=ent-pdata[ii]*np.log(pdata[ii])*dx
    return ent

def cal_muinfo(p,q,pq):
    """
    Calculate multual information
    :param p: marginal p
    :param q: marginal q
    :param pq: joint pq
    :return:
    """
    assert len(p)*len(q) == pq.shape[0]*pq.shape[1]
    ptq=p.reshape(-1,1)*q.reshape(1,-1)
    return cal_kldiv(pq,ptq)

def cal_muinfo_raw(x,y,x_discrete=True,y_discrete=True,x_bins=None,y_bins=None):
    """
    Calculation of mutual information between x,y from raw data (May have problem)!!!
    :param x: [Ndata of value]
    :param y: [Ndata of value]
    :param x_discrete: if x is discrete
    :param y_discrete: if y is discrete
    :param x_res: x resolution (None if discrete)
    :param y_res: y resolution (None if discrete)
    :return:
    """
    x=np.array(x)
    y = np.array(y)

    assert len(x) == len(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    if x_discrete:
        x_bins=len(set(x))
    px,_ = np.histogram(x, bins=x_bins)
    px = px/np.sum(px)

    if y_discrete:
        y_bins=len(set(y))
    py, _ = np.histogram(y, bins=y_bins)
    py = py / np.sum(py)

    pxy,_,_= np.histogram2d(x,y,bins=[x_bins,y_bins])
    pxy=pxy/np.sum(pxy)

    return cal_muinfo(px,py,pxy)

def cal_muinfo_raw_ND_discrete(X,Z):
    """
    Calculate multual information of N dimensional discrete data X and Z
    I(X;Z) = H(X) + H(Z) - H(X,Z)
    :param X: [Ndata of D-dim value]
    :param Z: [Ndata of D-dim value]
    :return:
    """
    Xnp = np.array(X)
    if len(Xnp.shape)==1:
        Xnp=Xnp.reshape(-1,1)
    assert len(Xnp.shape) == 2
    assert Xnp.shape[0] > Xnp.shape[1]

    Znp = np.array(Z)
    if len(Znp.shape)==1:
        Znp=Znp.reshape(-1,1)
    assert len(Znp.shape) == 2
    assert Znp.shape[0] > Znp.shape[1]

    XZnp = np.concatenate((Xnp,Znp),axis=1)

    Hx= cal_entropy_raw_ND_discrete(Xnp)

    Hz = cal_entropy_raw_ND_discrete(Znp)

    Hxz = cal_entropy_raw_ND_discrete(XZnp)

    return Hx+Hz-Hxz

def  cal_muinfo_continous_raw(x,y,bins=[301,299]):
    """
    Calculation of continous mutual information from definition (Gaussian Test passed)
    :param x: x data
    :param y:y data
    :param bins:
    :return:
    """
    x = np.array(x)
    y = np.array(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    pdataxy, xbins, ybins = np.histogram2d(x,y,bins=bins)
    dx = xbins[1] - xbins[0]
    dy = ybins[1] - ybins[0]
    # pdatax, _ = np.histogram(x, bins=bins[0])
    # pdatay, _ = np.histogram(y, bins=bins[1])
    pdatax = np.sum(pdataxy,axis=1)
    pdatax = pdatax/np.sum(pdatax)/dx
    pdatay = np.sum(pdataxy, axis=0)
    pdatay = pdatay / np.sum(pdatay) / dy
    pdataxy = pdataxy / np.sum(pdataxy) / dx / dy

    ent = 0
    for ii in range(len(pdatax)):
        for jj in range(len(pdatay)):
            if pdataxy[ii,jj] > 0:
                ent = ent+pdataxy[ii,jj]*np.log((pdataxy[ii,jj])/(pdatax[ii]*pdatay[jj]))*dx*dy
    return ent

def cal_muinfo_raw_ND_continous(X,Y,precision=0.01):
    """
    Estimate multidimensional continous mutual information via discretization (Gaussian Test passed)
    :param X:
    :param Y:
    :param precision: discretization precision
    :return:
    """
    X = np.array(X)
    Y = np.array(Y)
    X = np.floor(X/precision)
    Y = np.floor(Y/precision)
    return cal_muinfo_raw_ND_discrete(X,Y)

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
    assert p.shape == q.shape
    assert np.min(p)>0
    assert np.min(q)>0
    kld=np.sum(p*np.log(p/q))
    return kld

def cal_muinfo_hybrid_raw(xn,y,bins=100):
    """
    Calculate hybrid mutual information from discrete xn to continous y by raw data
    :param xn: discrete xn
    :param y: continuous
    :return:
    """
    xn = np.array(xn)
    y = np.array(y)
    assert len(xn.shape) == 1
    assert len(y.shape) == 1

    np_data_bins=np.array(list(set(xn)))
    data_bins_s = np.sort(np_data_bins)
    prec=data_bins_s[1]-data_bins_s[0]
    data_bins = np.concatenate((data_bins_s.reshape(-1,1),np.array(data_bins_s[-1]+prec).reshape(1,1)),axis=0)
    data_bins=data_bins.reshape(-1)
    data_bins=data_bins-prec/2
    pxn, _ = np.histogram(xn, bins=data_bins)
    pxn=pxn/np.sum(pxn)

    sx=set(xn)
    ynnx=[[] for nnx in sx]
    lsx = list(set(xn))
    for ii in range(len(xn)):
        ynnx[lsx.index(xn[ii])].append(y[ii])

    ent=0
    py, binsl = np.histogram(y, bins=bins)
    dy = binsl[1] - binsl[0]
    pydata = py / np.sum(py) / dy

    # plt.bar(binsl[:-1], py)
    # plt.show()
    # print(binsl[:-1], py)

    for iin in range(len(sx)):
        pxy, _ = np.histogram(ynnx[iin], bins=binsl)
        pxydata = pxy / np.sum(pxy) / dy
        for ii in range(len(pydata)):
            if pxydata[ii] > 0 and pydata[ii] > 0:
                ent = ent + pxn[iin] * pxydata[ii] * np.log(pxydata[ii]/pydata[ii]) * dy
    return ent

def cal_kldiv_torch(p,q):
    """
    Cal KL divergence of p over q
    :param data:
    :return:
    """
    p = p+1e-9
    q = q+1e-9
    p = p / torch.sum(p,dim=-1,keepdim=True)
    q = q / torch.sum(q,dim=-1,keepdim=True)
    assert p.shape == q.shape
    assert torch.min(p)>0
    assert torch.min(q)>0
    kld=torch.sum(p*torch.log(p/q),dim=-1)
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

def sample_id(prob, shape):
    """
    Sample id according to 1-d probability vec
    :param prob: normal probability vec
    :return: a torch LongTensor with shape
    """

    prob = np.array(prob)
    assert len(prob.shape) == 1
    prob = prob / np.sum(prob)
    idn = len(prob)

    smat = np.random.choice(idn, shape, p=prob)
    tsmat = torch.from_numpy(smat).type(torch.LongTensor)

    return tsmat

def sort_w_arg(datal,down_order=True,top_k=None):
    """
    Sort data list with arg index information
    :param datal: a list with data
    :return: list of tuple [(arg,data),...]
    """

    if top_k is None:
        l=len(datal)
        sorttp = zip(np.linspace(0, l - 1, l), datal)
        if down_order:
            count_pairs = sorted(sorttp, key=lambda x: -x[1])
        else:
            count_pairs = sorted(sorttp, key=lambda x: x[1])
        ids, data = list(zip(*count_pairs))
    else:
        res = []
        for ii in range(top_k):
            if down_order:
                res.append(("UNK", -1e99))
            else:
                res.append(("UNK", 1e99))
        for (ii, data) in enumerate(datal):
            if down_order:
                if data > res[top_k - 1][1]:
                    res[top_k - 1] = (ii,data)
                    res.sort(key=lambda tup: tup[1], reverse=True)
            else:
                if data < res[top_k - 1][1]:
                    res[top_k - 1] = (ii,data)
                    res.sort(key=lambda tup: tup[1])
        ids, data = list(zip(*res))
    return ids, data

def one_hot(ind_mat,n_digits,cuda_device="cuda:0"):
    """
    Generate one hot matrix with ind_mat
    :param mat: LongTensor [N1,N2,...]
    :return: LongTensor [N1,N2,...,vec_onehot]
    """
    rshape=list(ind_mat.shape)
    vshape = list(ind_mat.shape)
    rshape.append(n_digits)
    vshape.append(1)
    res_onehot = torch.zeros(rshape).to(cuda_device)
    res_onehot.scatter_(-1,ind_mat.view(vshape),1)
    return res_onehot

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

def softmax(x,dim=-1,torch_flag=False):
    if torch_flag:
        sfm=torch.nn.Softmax(dim=dim)
        res=sfm(x)
        if torch.isnan(res).any():
            raise Exception("NaN Error 1")
    else:
        res = np.exp(x)/(np.sum(np.exp(x),axis=-1,keepdims=True)+1e-9)
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

class HCObject(object):
    """
    A class storing HierachicalCluster object
    """
    def __init__(self,data,label,accN=1):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.data=data
        self.label=label
        self.accN=accN


class HierachicalCluster(object):
    """
    A class helping general hierachical clustering algorithm
    """
    def __init__(self,obj_list,Z,min_flag=True):
        self.obj_list=obj_list
        self.ini_N_obj=len(obj_list)
        self.obj_mask=[1 for ii in range(self.ini_N_obj)] # 1 means effective, 0 means masked out
        self.linkage=[] # id_cluster_L, id_cluster_R, dist, Accumulation Number
        self.Z = Z
        self.min_flag=min_flag
        if min_flag:
            self.Ldist=999999
        else:
            self.Ldist = -999999
        self.dist_mat=np.eye(self.ini_N_obj)*self.Ldist
        self.mearge_mat = np.eye(self.ini_N_obj) * self.Ldist

    def run_clustering(self):
        self.init_dist_mat()
        for ii in range(self.ini_N_obj-1):
            print("HC Step: ", ii)
            if self.min_flag:
                iix,jjy = np.unravel_index(self.mearge_mat.argmin(), self.dist_mat.shape)
            else:
                iix, jjy = np.unravel_index(self.mearge_mat.argmax(), self.dist_mat.shape)
            self.merge(iix,jjy,self.dist_mat[iix,jjy])
            self.expand_dist_mat()
        self.plot_clustering()

    def init_dist_mat(self):
        """
        Initialize the ini_N_obj*ini_N_obj distmat
        :return:
        """

        assert len(self.obj_list) == len(self.obj_mask)
        for ii in range(self.ini_N_obj):
            for jj in range(ii):
                if self.obj_mask[ii]*self.obj_mask[jj]==1:
                    I_merge, I_dist=self.cal_dist(self.obj_list[ii].data,self.obj_list[jj].data,self.Z)
                    self.dist_mat[ii, jj] = I_dist
                    self.dist_mat[jj, ii] = I_dist
                    self.mearge_mat[ii, jj] = I_merge
                    self.mearge_mat[jj, ii] = I_merge

    def expand_dist_mat(self):

        assert len(self.obj_list) == len(self.obj_mask)

        new_dist_mat=np.eye(len(self.obj_list))*self.Ldist
        new_mearge_mat=np.eye(len(self.obj_list))*self.Ldist
        for ii in range(len(self.dist_mat)):
            for jj in range(len(self.dist_mat[0])):
                new_dist_mat[ii,jj]=self.dist_mat[ii,jj]
                new_mearge_mat[ii, jj] = self.mearge_mat[ii, jj]

        new_dist_mat[:, np.array(self.obj_mask) == 0] = self.Ldist
        new_dist_mat[np.array(self.obj_mask) == 0, :] = self.Ldist
        new_mearge_mat[:, np.array(self.obj_mask) == 0] = self.Ldist
        new_mearge_mat[np.array(self.obj_mask) == 0, :] = self.Ldist

        ii=len(self.obj_list)-1
        for jj in range(ii):
            if self.obj_mask[ii] * self.obj_mask[jj] == 1:
                I_merge, I_dist=self.cal_dist(self.obj_list[ii].data,self.obj_list[jj].data,self.Z)
                new_dist_mat[ii, jj] = I_dist
                new_dist_mat[jj, ii] = I_dist
                new_mearge_mat[ii, jj] = I_merge
                new_mearge_mat[jj, ii] = I_merge

        self.mearge_mat=new_mearge_mat
        self.dist_mat=new_dist_mat

    def plot_clustering(self):
        def llf(id):
            return self.obj_list[id].label
        dgram = dendrogram(self.linkage, truncate_mode="level", leaf_label_func=llf, leaf_rotation=60)
        plt.tick_params(labelsize=10)
        plt.title("Hierachical Information Flow Graph")
        plt.xlabel("Feature ID")
        plt.ylabel("Predictive Mutual Infomation")
        plt.show()

    def cal_dist(self,X,Y,Z):
        """Distance defined as predictive mutual information"""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        XY=np.concatenate((X,Y),axis=1)
        Ixy_z=cal_muinfo_raw_ND_discrete(XY,Z)
        Ix_z = cal_muinfo_raw_ND_discrete(X, Z)
        Iy_z = cal_muinfo_raw_ND_discrete(Y, Z)
        # I_xy=cal_muinfo_raw_ND_discrete(X, Y)
        # Hx = cal_entropy_raw_ND_discrete(X)
        # Hy = cal_entropy_raw_ND_discrete(Y)
        Hxy = cal_entropy_raw_ND_discrete(XY)
        # I_merge=(2*Ixy_z-Ix_z-Iy_z)/Ixy_z
        I_merge = Ixy_z
        I_dist=Ixy_z
        return I_merge,I_dist

    def merge(self,id1,id2,dist):
        Obj1 = self.obj_list[id1]
        Obj2 = self.obj_list[id2]
        newData=np.concatenate((Obj1.data,Obj2.data),axis=1)
        newLabel = str(Obj1.label) + "," +str(Obj2.label)
        newaccN=Obj1.accN+Obj2.accN
        newObj=HCObject(newData,newLabel,accN=newaccN)

        self.obj_list.append(newObj)
        self.obj_mask[id1] = 0
        self.obj_mask[id2] = 0
        self.obj_mask.append(1)

        self.linkage.append([id1,id2,dist,newaccN])