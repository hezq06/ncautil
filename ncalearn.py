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
import time

import torch
import copy
from torch.autograd import Variable

from sklearn.cluster import KMeans
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

def pltsne(data):
    """
    Plot 2D w2v graph with tsne
    Referencing part of code from: Basic word2vec example tensorflow
    :param numpt: number of points
    :return: null
    """
    tsnetrainer = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, method='exact')
    tsne = tsnetrainer.fit_transform(data)
    for i in range(len(data)):
        x, y = tsne[i, :]
        plt.scatter(x, y)
    plt.show()


def pl_eig_pca(data):
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
    muCov=np.cov(np.abs(data1), np.abs(data2))[0:D, D:2 * D]
    plt.imshow(muCov, cmap='seismic', clim=(-np.amax(muCov), np.amax(muCov)))
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

def do_eval(dataset,lsize,rnn, id_2_vec=None):
    print("Start Evaluation ...")
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    datab=[]
    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec=np.array(id_2_vec[data])
            datab.append(datavec)
    databpt=torch.from_numpy(np.array(datab))
    databpt = databpt.type(torch.FloatTensor)
    hidden = rnn.initHidden_eval()
    outputl = []
    hiddenl = []

    for iis in range(len(databpt) - 1):
        x = databpt[iis, :].view(1, 1, lsize_in)
        output, hidden = rnn(x, hidden)
        outputl.append(output.view(-1).data.numpy())
        hiddenl.append(hidden)

    outputl = np.array(outputl)
    outputl = Variable(torch.from_numpy(outputl).contiguous())
    outputl = outputl.permute((1, 0))
    print(outputl.shape)
    # Generating output label
    yl = []
    for iiss in range(len(dataset) - 1):
        ylb = []
        wrd = dataset[iiss + 1]
        ylb.append(wrd)
        yl.append(np.array(ylb))
    outlab = torch.from_numpy(np.array(yl).T)
    outlab = outlab.type(torch.LongTensor)
    lossc = torch.nn.CrossEntropyLoss()
    loss = lossc(outputl.view(1, -1, len(dataset) - 1), outlab)
    print("Evaluation Perplexity: ", np.exp(loss.item()))
    return outputl, hiddenl, outlab.view(-1)

def do_eval_p(dataset,lsize,rnn, id_2_vec=None):
    """
    General evaluation function
    :param dataset:
    :param lsize:
    :param rnn:
    :return:
    """
    print("Start Evaluation ...")
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    datab=[]
    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec=np.array(id_2_vec[data])
            datab.append(datavec)
    databpt=torch.from_numpy(np.array(datab))
    databpt = databpt.type(torch.FloatTensor)
    hidden = rnn.initHidden_eval()

    perpls=[]
    for nn in range(len(databpt) - 1):
        x = databpt[nn]
        y = np.zeros(lsize_out)
        y[dataset[nn+1]]=1
        prd, hidden = rnn.forward(x.view(1, 1, lsize_in), hidden)
        prd = torch.exp(prd) / torch.sum(torch.exp(prd))
        perp = cal_kldiv(y, prd.view(-1).data.numpy())
        perpls.append(perp)
    avperp = np.mean(np.array(perpls))
    print("Calculated knowledge perplexity:", np.exp(avperp))
    return avperp


def run_training(dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, save=None,seqtrain=False,coop=None,coopseq=None, id_2_vec=None):
    """
    General rnn training funtion for one-hot training
    :param dataset:
    :param lsize:
    :param model:
    :param step:
    :param learning_rate:
    :param batch:
    :param window:
    :param save:
    :param seqtrain:
    :param coop: a cooperational rnn unit
    :param coopseq: a pre-calculated cooperational logit vec
    :return:
    """
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()
    datab=[]
    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec=np.array(id_2_vec[data])
            datab.append(datavec)
    databp=torch.from_numpy(np.array(datab))
    if coopseq is not None:
        coopseq=torch.from_numpy(np.array(coopseq))
        coopseq=coopseq.type(torch.FloatTensor)

    rnn.train()

    if coop is not None:
        coop.eval()

    def custom_KNWLoss(outputl, outlab, model, cstep):
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight+model.h2o.bias.view(-1)
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.Ws.weight.norm(1) + model.Wr.weight.norm(1)
        return loss1# + 0.001 * l1_reg * cstep / step + 0.005 * lossh2o * cstep / step  # +0.3*lossz+0.3*lossn #

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

    train_hist = []
    his = 0

    gpuavail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpuavail else "cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    if gpuavail:
        rnn.to(device)
        databp.to(device)

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)

        # Generating output label
        yl = []
        for iiss in range(window):
            ylb = []
            for iib in range(batch):
                wrd = dataset[(int(rstartv[iib]) + iiss + 1)]
                ylb.append(wrd)
            yl.append(np.array(ylb))
        outlab = Variable(torch.from_numpy(np.array(yl).T))
        outlab = outlab.type(torch.LongTensor)

        # step by step training
        if not seqtrain:
            outputl = None
            for iiss in range(window):
                vec1m = None
                vec2m = None
                if coopseq is not None:
                    veccoopm = None
                for iib in range(batch):
                    vec1 = databp[(int(rstartv[iib]) + iiss), :]
                    vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
                    if vec1m is None:
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                    if coopseq is not None:
                        veccoop=coopseq[(int(rstartv[iib]) + iiss +1), :]
                        if veccoopm is None:
                            veccoopm = veccoop.view(1, -1)
                        else:
                            veccoopm = torch.cat((veccoopm, veccoop.view(1, -1)), dim=0)
                # One by one guidance training ####### error can propagate due to hidden state
                x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    outlab = outlab.to(device)
                    x, y = x.to(device), y.to(device)
                if coop is not None:
                    outputc, hiddenc = coop(x, hidden=None,logitmode=True)
                    output, hidden = rnn(x, hidden, add_logit=outputc)
                elif coopseq is not None:
                    output, hidden = rnn(x, hidden, add_logit=veccoopm)
                else:
                    output, hidden = rnn(x, hidden, add_logit=None)
                if type(outputl) == type(None):
                    outputl = output.view(batch, lsize_out, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, lsize_out, -1), output.view(batch, lsize_out, 1)), dim=2)
            loss = custom_KNWLoss(outputl, outlab, rnn, iis)
        else:
            # LSTM/GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1 = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
                vec2 = databp[int(rstartv[iib])+1:int(rstartv[iib])+window+1, :]
                if type(vec1m) == type(None):
                    vec1m = vec1.view(window, 1, -1)
                    vec2m = vec2.view(window, 1, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                    vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
            x = Variable(vec1m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)  #
            y = Variable(vec2m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if gpuavail:
                outlab = outlab.to(device)
                x, y = x.to(device), y.to(device)
            output, hidden = rnn(x, hidden)
            loss = custom_KNWLoss(output.permute(1,2,0), outlab, rnn, iis)

        if int(iis / prtstep) != his:
            print("Perlexity: ", iis, np.exp(loss.item()))
            his = int(iis / prtstep)

        train_hist.append(np.exp(loss.item()))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    endt = time.time()
    print("Time used in training:", endt - startt)

    x = []
    for ii in range(len(train_hist)):
        x.append([ii, train_hist[ii]])
    x = np.array(x)
    try:
        plt.plot(x[:, 0], x[:, 1])
        if type(save) != type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()
    except:
        pass

    return rnn

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
