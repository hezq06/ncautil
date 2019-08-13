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

def cluster(data,n_clusters,mode="kmeans"):
    """
    Do clustering
    :param data: data to be clustered
    :param n_clusters: number of clusters
    :param mode: "kmeans"
    :return:
    """
    startt = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init="random").fit(data)
    center=np.zeros((n_clusters,len(data[0])))
    clscounter=np.zeros(n_clusters)
    for iid in range(len(data)):
        ncls=kmeans.labels_[iid]
        center[ncls]=center[ncls]+data[iid]
        clscounter[ncls]=clscounter[ncls]+1
    for iic in range(n_clusters):
        if clscounter[iic] != 0:
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

def cal_entropy_raw(data,data_discrete,data_bins=None):
    """
    Calculate entropy of raw data
    :param data: [Ndata of value]
    :param data_discrete: if data is discrete
    :return:
    """
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
    :param data: [Ndata of value]
    :param data_discrete: if data is discrete
    :return:
    """
    datanp=np.array(data)
    assert len(datanp.shape) == 2
    assert datanp.shape[0]>datanp.shape[1]

    datatup=[]
    for iin in range(len(data)):
        datatup.append(tuple(data[iin]))

    itemsets=list(set(datatup))
    nsets=len(itemsets)

    hashtab = dict([])
    for ii in range(nsets):
        hashtab[itemsets[ii]] = 0
    for ii in range(len(datatup)):
        hashtab[datatup[ii]] = hashtab[datatup[ii]] + 1

    pvec=np.zeros(nsets)
    for ii,val in enumerate(hashtab.values()):
        pvec[ii]=val

    pvec = pvec / np.sum(pvec)

    return cal_entropy(pvec)

def cal_entropy(data,logit=False):
    """
    Cal entropy of a vector
    :param data:
    :param logit: if input data is logit mode
    :return:
    """
    # print(data)
    if logit:
        data=np.exp(data)
    data=data/np.sum(data)
    assert len(data.shape) == 1
    data_adj=np.zeros(data.shape)+data
    data_adj[data_adj==0]=1e-9
    ent=-np.sum(data*np.log(data_adj))
    return ent

def cal_mulinfo_raw(x,y,x_discrete,y_discrete,x_bins=None,y_bins=None):
    """
    Calculation of mutual information between x,y from raw data
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

    return cal_mulinfo(px,py,pxy)


def cal_mulinfo(p,q,pq):
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