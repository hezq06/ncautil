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
import matplotlib.ticker as ticker

from wordcloud import WordCloud
import operator
from PIL import Image
from PIL import ImageDraw,ImageFont

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

def pltsne(data,D=2,perp=1):
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
    plt.show()
    return tsnetrainer


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

def pl_cov(data,text=None,texty=None):
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

def pl_corr(data,text=None,texty=None):
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
    plt.title("cov(|D1|,|D1|)")
    plt.colorbar(fig)
    plt.show()
    return Corr

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
    plt.imshow(img, cmap='seismic',clim=(-np.amax(data), np.amax(data)))
    plt.colorbar()
    plt.show()

def pl_conceptbubblecloud(id_to_con,id_to_word,prior,word_to_vec, pM=None):
    """
    Visualize concept using bubble word cloud
    :param id_to_con: [con for wrd0, con for wrd1, ...]
    :param id_to_word: dict[0:wrd0, 1:wrd1,...]
    :param prior:
    :return:
    """
    # step 1: data collection by concept : [[0,[wrds for con0],[priors for con0]]]
    dict_conwrdcol=dict([])
    dict_conprcol=dict([])
    dict_cen_wvec=dict([])
    dict_text_freq=dict([])
    dict_con_freq=dict([])
    list_cen_vec=[]
    con_freq_max=0
    for con_id in set(id_to_con):
        conwrdcol=[]
        conprcol=[]
        con_freq=0
        for iter_ii in range(len(id_to_con)):
            if id_to_con[iter_ii] == con_id:
                conwrdcol.append(id_to_word[iter_ii])
                conprcol.append(np.log(prior[iter_ii]))
                con_freq=con_freq+prior[iter_ii]
        # step 2: generate word cloud by concept
        text_freq = dict([])
        cen_vec=np.zeros(len(word_to_vec[conwrdcol[0]]))
        prbsum=np.sum(np.exp(np.array(conprcol)))
        for wrd_ii in range(len(conwrdcol)):
            text_freq[str(conwrdcol[wrd_ii])]=conprcol[wrd_ii]
            cen_vec=cen_vec+np.exp(conprcol[wrd_ii])/prbsum*word_to_vec[conwrdcol[wrd_ii]]
            # cen_vec = cen_vec + word_to_vec[conwrdcol[wrd_ii]]
        # cen_vec=cen_vec/len(conwrdcol)
        dict_conwrdcol[con_id]= conwrdcol
        dict_conprcol[con_id] = conprcol
        dict_cen_wvec[con_id] = cen_vec
        list_cen_vec.append(cen_vec)
        dict_text_freq[con_id]=text_freq
        dict_con_freq[con_id]=con_freq
        if con_freq>con_freq_max:
            con_freq_max=con_freq

    # Step 2: adjusting shift using force-field based method

    # PCA projection of cen_vec
    data=np.array(list_cen_vec).T
    # res,pM=pca_proj(data, 2)
    if pM is None:
        pM=np.random.random((2,len(cen_vec)))
    res=pM.dot(data)

    size_scalex,size_scaley=400,400

    dict_con_posi = dict([])
    dict_con_radius = dict([])
    for con_id in set(id_to_con):
        cen_vec = dict_cen_wvec[con_id]
        cen_posi = [pM.dot(cen_vec)[0] * size_scalex / 2, (pM.dot(cen_vec)[1]) * size_scaley / 2]
        dict_con_posi[con_id] = np.array(cen_posi)
        radius = min(size_scalex, size_scaley) / 2 * np.sqrt(dict_con_freq[con_id] / con_freq_max)
        dict_con_radius[con_id]=radius

    def cal_con_posi_stack(dict_con_radius,dist_margin=10):
        """
        Using random attaching algorithm
        :param dict_con_radius:
        :param dist_margin:
        :return:
        """
        res_con_posi = dict([])
        far_dist=100000
        for con_id_ii in set(id_to_con):
            if bool(res_con_posi):
                rnddir=np.random.random(2)-0.5
                rnddir=rnddir/np.linalg.norm(rnddir)
                start_vec=far_dist*rnddir
                min_dist=2*far_dist
                min_id=None
                for con_id_jj,posi in res_con_posi.items():
                    dist=np.linalg.norm(start_vec-posi)
                    if dist<min_dist:
                        min_dist=dist
                        min_id=con_id_jj
                cal_vec=start_vec-res_con_posi[min_id]
                cal_vec=cal_vec/np.linalg.norm(cal_vec)
                cal_vec=cal_vec*(dict_con_radius[min_id]+dict_con_radius[con_id_ii]+dist_margin)+res_con_posi[min_id]
                res_con_posi[con_id_ii]=cal_vec
            else:
                res_con_posi[con_id_ii] = np.zeros(2)

        ####### KEY BUG !!!!! center of circle is different from edge of square
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii]=res_con_posi[con_id_ii]-dict_con_radius[con_id_ii]

        return res_con_posi



    # def cal_con_posi_stack(dict_con_radius,dist_margin=10):
    #     """
    #     Using stacking method to calculate con_posi
    #     :param dict_con_radius:
    #     :return:
    #     """
    #     res_con_posi=dict([])
    #     BoundaryL=[]
    #     for con_id_ii in set(id_to_con):
    #         if not bool(res_con_posi) and not bool(BoundaryL): # All empty, step 1
    #             res_con_posi[id_to_con]=np.zeros(2)
    #             first_con_id=con_id_ii
    #         elif bool(res_con_posi) and not bool(BoundaryL): # Boundary empty, step 2
    #             BoundaryL.append((first_con_id,con_id_ii))
    #             res_con_posi[id_to_con] = np.array([dict_con_radius[first_con_id]+dict_con_radius[con_id_ii]+dist_margin,0])
    #         else:
    #             # Pick a random boundary
    #             pkid=int(np.ramdom.rand()*len(BoundaryL))
    #             bound=BoundaryL[pkid]
    #             cen1=res_con_posi[bound[0]]
    #             cen2 = res_con_posi[bound[1]]
    #
    #
    #         # Move to center
    #         tot_cen=np.zeros(2)
    #         totN=0
    #         for cen in res_con_posi.values():
    #             tot_cen=tot_cen+cen
    #             totN=totN+1
    #         tot_cen=tot_cen/totN
    #         for con in res_con_posi.keys():
    #             res_con_posi[con]=res_con_posi[con]-tot_cen


    def cal_con_posi(init_con_posi,dict_con_radius,attr=0.01,rep=2.0,dist_margin=10,fxoy=1.0,iter=10000):
        """
        Calculate updated concept position using graviy field
        :param init_con_posi:
        :param dict_con_radius:
        :param attr: attraction
        :param rep: repulsion
        :param dist_margin: distance margin between two concepts
        :param fxoy: x over y aspect ratio
        :return: cal_con_posi
        """
        res_con_posi=init_con_posi
        # Expand to remove overlap
        expansion_ratio=2.0
        # for con_id_ii in set(id_to_con):
        #     for con_id_jj in set(id_to_con):
        #         if con_id_jj != con_id_ii:
        #             # Global constant attraction
        #             dist = np.linalg.norm (init_con_posi[con_id_jj] - init_con_posi[con_id_ii])
        #             rsum=dict_con_radius[con_id_jj]+dict_con_radius[con_id_ii]
        #             test_ratio=rsum/dist
        #             if test_ratio>expansion_ratio:
        #                 expansion_ratio=test_ratio
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii]=res_con_posi[con_id_ii]*expansion_ratio*1.1

        # Move to center
        tot_cen = np.zeros(2)
        totN = 0
        for cen in res_con_posi.values():
            tot_cen = tot_cen + cen
            totN = totN + 1
        tot_cen = tot_cen / totN
        for con in res_con_posi.keys():
            res_con_posi[con] = res_con_posi[con] - tot_cen

        for ii_iter in range(iter):
            conlist=list(set(id_to_con))
            pick=int(np.random.rand()*len(conlist))
            con_id_ii=conlist[pick]

            for con_id_jj in set(id_to_con):
                if con_id_jj!=con_id_ii:

                    diffvec=init_con_posi[con_id_jj]- init_con_posi[con_id_ii]
                    dir_vec=diffvec/np.linalg.norm(diffvec)

                    rd=np.linalg.norm(diffvec)
                    rjj=dict_con_radius[con_id_jj]
                    rii=dict_con_radius[con_id_ii]
                    dist_val=rd-rjj-rii

                    res_con_posi[con_id_ii] = res_con_posi[con_id_ii] + attr * (dist_val-dist_margin) * dir_vec

                    if dist_val < dist_margin:
                        res_con_posi[con_id_ii] = res_con_posi[con_id_ii] - 1.1 *(dist_margin-dist_val)*diffvec/np.linalg.norm(diffvec)

        ####### KEY BUG !!!!! center of circle is different from edge of square
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii] = res_con_posi[con_id_ii] - dict_con_radius[con_id_ii]

        return res_con_posi


    res_con_posi=cal_con_posi(dict_con_posi, dict_con_radius)
    # res_con_posi = cal_con_posi_stack(dict_con_radius)
    # res_con_posi=dict_con_posi
    # Move to center
    # tot_cen = np.zeros(2)
    # totN = 0
    # for cen in res_con_posi.values():
    #     tot_cen = tot_cen + cen
    #     totN = totN + 1
    # tot_cen = tot_cen / totN
    # for con in res_con_posi.keys():
    #     res_con_posi[con] = res_con_posi[con] - tot_cen
    minx=0
    miny=0
    for con_id in set(id_to_con):
        if res_con_posi[con_id][0]<minx:
            minx=res_con_posi[con_id][0]
        if res_con_posi[con_id][1]<miny:
            miny=res_con_posi[con_id][1]

    # Step 3: actually draw the figure

    img1 = Image.new('RGBA', size=(1,1), color=(255, 255, 255, 255))

    for con_id in set(id_to_con):
        print(con_id)
        radius = dict_con_radius[con_id]
        circle = Image.new('RGBA', size=(int(2*radius),int(2*radius)), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(circle)
        draw.ellipse((0, 0, 2*radius, 2*radius), fill=(0, 0, 0, 255))
        circle2 = Image.new('RGBA', size=(int(2*radius),int(2*radius)), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(circle2)
        draw.ellipse((0, 0, 2*radius, 2*radius), fill=(0, 0, 0, 255))
        text_freq=dict_text_freq[con_id]
        wordcloud = WordCloud(mask=np.array(circle), background_color="white").generate_from_frequencies(text_freq)
        data2 = wordcloud.to_array()
        img2 = Image.fromarray(data2, 'RGB')
        img2 = img2.convert("RGBA")
        txt = Image.new('RGBA', img2.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeSerif.ttf", int(radius))
        w, h = font.getsize(str(con_id))
        draw.text((radius-w/2, radius-h/2), str(con_id), (0, 0, 0, 128), font=font)
        img2 = Image.alpha_composite(img2, txt)
        shift=tuple((res_con_posi[con_id]-np.array([minx,miny])).astype(int))
        nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)

        newimg = Image.new('RGBA', size=(nw, nh), color=(255, 255, 255, 255))
        newimg.paste(img1, (0, 0))
        newimg.paste(img2, shift, circle2)

        # blend with alpha=0.5
        # img1 = Image.blend(newimg1, newimg2, alpha=0.5)
        img1=newimg

        print(shift)

    plt.imshow(img1, interpolation='bilinear')
    plt.axis("on")
    plt.margins(x=0, y=0)
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

def wta_layer(l_input,schedule=1.0,wta_noise=0.0,upper_t = 0.3, schshift=0.2):

    concept_size = l_input.shape[-1]
    schedule=schedule+schshift
    if schedule>=1.0:
        schedule=1.0
    Nindr = (1.0 - np.sqrt(schedule)) * (concept_size - 2) * upper_t + 1  # Number of Nind largest number kept
    # Nindr = (1.0 - schedule) * (concept_size - 2) * upper_t + 1  # Number of Nind largest number kept
    smooth=Nindr-int(Nindr)
    Nind=int(Nindr)
    np_input=l_input.cpu().data.numpy()
    npargmax_i = np.argsort(-np_input, axis=-1)
    argmax_i = torch.from_numpy(npargmax_i).narrow(-1, 0, Nind)
    outer=torch.from_numpy(npargmax_i).narrow(-1, Nind, 1)
    concept_layer_i = torch.zeros(l_input.shape)
    concept_layer_i.scatter_(-1, argmax_i, 1.0)
    concept_layer_i.scatter_(-1, outer, smooth)
    concept_layer_i = concept_layer_i + wta_noise * torch.rand(concept_layer_i.shape)

    if l_input.is_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        concept_layer_i=concept_layer_i.to(device)

    ginput_masked = l_input * concept_layer_i
    ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)
    return ginput_masked

def free_gen(step,lsize,rnn, id_2_vec=None, id_to_word=None,prior=None):
    """
    Free generation
    :param step:
    :param lsize:
    :param rnn:
    :param id_2_vec:
    :return:
    """
    print("Start Evaluation ...")
    startt = time.time()
    if type(lsize) is list:
        lsize_in = lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize

    hidden = rnn.initHidden(1)
    x = torch.zeros(1, 1, lsize_in)
    outputl = []
    outwrdl=[]
    outpl=[]
    hiddenl=[]
    # clul=[]

    def logp(vec):
        """
        LogSoftmax function
        :param vec:
        :return:
        """
        vec = np.exp(vec)
        dwn = np.sum(vec)
        return vec / dwn

    for iis in range(step):
        x= x.type(torch.FloatTensor)
        output, hidden = rnn(x, hidden)    ############ rnn
        ynp = output.data.numpy().reshape(lsize_out)
        rndp = np.random.rand()
        pii = logp(ynp).reshape(-1)
        dig = 0
        for ii in range(len(pii)):
            rndp = rndp - pii[ii]
            if rndp < 0:
                dig = ii
                break
        xword = id_to_word[dig]
        outputl.append(dig)
        outwrdl.append(xword)
        # hiddenl.append(hout.cpu().data.numpy().reshape(-1))
        if prior is None:
            outpl.append(pii[0:200])
        else:
            outpl.append(pii[0:200]/prior[0:200])
        # clul.append(np.exp(clu.data.numpy().reshape(30)))
        xvec = id_2_vec[dig]
        x = torch.from_numpy(xvec.reshape((1, 1, lsize_in)))
    endt = time.time()
    print("Time used in generation:", endt - startt)
    return outputl,outwrdl,outpl,hiddenl #clul

def do_eval(dataset,lsize,rnn, id_2_vec=None, seqeval=False):
    print("Start Evaluation ...")
    startt = time.time()
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
    hidden = rnn.initHidden(1)
    hiddenl = []
    if not seqeval:
        outputl = []
        for iis in range(len(databpt) - 1):
            x = databpt[iis, :].view(1, 1, lsize_in)
            output, hidden = rnn(x, hidden)
            outputl.append(output.view(-1).data.numpy())
            hiddenl.append(hidden)
        outputl = np.array(outputl)
        outputl = Variable(torch.from_numpy(outputl).contiguous())
        outputl = outputl.permute((1, 0))
        print(outputl.shape)
    else:
        # LSTM/GRU provided whole sequence training
        dlen=len(databpt)
        x = databpt[0:dlen-1, :].view(dlen-1, 1, lsize_in)
        outputl, hidden = rnn(x, hidden)
        outputl=outputl.permute(1, 2, 0)

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
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return outputl, hiddenl, outlab.view(-1)

def do_eval_p(dataset,lsize,rnn, id_2_vec=None, prior=None):
    """
    General evaluation function
    :param dataset:
    :param lsize:
    :param rnn:
    :return:
    """
    print("Start Evaluation ...")
    startt = time.time()
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
    hidden = rnn.initHidden(1)

    perpls=[0.0]
    for nn in range(len(databpt) - 1):
        x = databpt[nn]
        y = np.zeros(lsize_out)
        y[dataset[nn+1]]=1
        if prior is not None:
            perp = cal_kldiv(y, prior)
        else:
            prd, hidden = rnn.forward(x.view(1, 1, lsize_in), hidden)
            prd = torch.exp(prd) / torch.sum(torch.exp(prd))
            perp = cal_kldiv(y, prd.view(-1).data.numpy())
        perpls.append(perp)
    avperp = np.mean(np.array(perpls))
    print("Calculated knowledge perplexity:", np.exp(avperp))
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return perpls

def do_eval_rnd(dataset,lsize, rnn, step, window=30, batch=20, id_2_vec=None, para=None):
    """
    General evaluation function using stochastic batch evaluation on GPU
    :param dataset:
    :param lsize:
    :param rnn:
    :param extend_mode: for concept to word extension mode, extend_mode[0] is extension dataset, extend_mode[1] is extension matrix.
    :return:
    ### Data definition
    perpl: 1-D list of calculated perplexity
    """
    print("Start Evaluation ...")
    startt = time.time()

    if para is None:
        para=dict([])
    seqeval = para.get("seqeval", False)
    extend_mode= para.get("extend_mode", None)
    supervise_mode = para.get("supervise_mode", False)
    pre_training = para.get("pre_training", False)

    if (type(dataset) is dict) != supervise_mode:
        raise Exception("Supervise mode Error.")

    if supervise_mode:
        label=dataset["label"]
        dataset=dataset["dataset"]

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

    gpuavail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpuavail else "cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    rnn=rnn.to(device)

    lossc = torch.nn.CrossEntropyLoss()
    perpl=[]
    outlabl=[]
    inputlabl = []
    conceptl=[]
    outputll=[]

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)


        if not supervise_mode:
            # Generating output label
            if extend_mode is None:
                extdataset = dataset
            else:
                extdataset = extend_mode[0]
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = extdataset[(int(rstartv[iib]) + iiss + 1)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)
        else:
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = label[(int(rstartv[iib]) + iiss)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)

        # LSTM/GRU provided whole sequence training
        if seqeval:
            vec1m = None
            inputlabsub=[]
            for iib in range(batch):
                vec1 = databpt[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                if type(vec1m) == type(None):
                    vec1m = vec1.view(window, 1, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                inputlab = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
                inputlabsub.append(inputlab)
            x = vec1m  #
            x = x.type(torch.FloatTensor)
            if gpuavail:
                outlab = outlab.to(device)
                x = x.to(device)
            if extend_mode is None:
                if pre_training:
                    outputl, hidden = rnn.pre_training(x, hidden, schedule=1.0)
                    conceptl.append(outputl.cpu().data.numpy())
                else:
                    outputl, hidden = rnn(x, hidden, schedule=1.0)
                    try:
                        conceptl.append(rnn.infer_pos.cpu().data.numpy())
                    except:
                        pass
            else:
                npM_ext=extend_mode[1]
                outputl, hidden = rnn.forward_concept_ext(x, hidden, npM_ext)

            outlabl.append(outlab.transpose(0,1).cpu().data.numpy())
            inputlabl.append(np.array(inputlabsub).T)
            # conceptl.append(rnn.concept_layer_i.cpu().data.numpy())
            # conceptl.append(rnn.hout2con_masked.cpu().data.numpy())
            outputll.append(outputl.cpu().data.numpy())
        else:
            outputl = None
            for iiss in range(window):
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databpt[(int(rstartv[iib]) + iiss), :]
                    vec2 = databpt[(int(rstartv[iib]) + iiss + 1), :]
                    if vec1m is None:
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                # One by one guidance training ####### error can propagate due to hidden state
                x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    outlab = outlab.to(device)
                    x, y = x.to(device), y.to(device)
                if pre_training:
                    output, hidden = rnn.pre_training(x, hidden, schedule=iis / step)
                else:
                    output, hidden = rnn(x, hidden, schedule=iis / step)
                if type(outputl) == type(None):
                    outputl = output.view(1, batch,lsize_out)
                else:
                    outputl = torch.cat((outputl.view(-1, batch, lsize_out), output.view(1, batch,lsize_out)), dim=0)

        loss = lossc(outputl.permute(1,2,0), outlab)
        perpl.append(loss.item())

    print("Evaluation Perplexity: ", np.exp(np.mean(np.array(perpl))))
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return perpl,outputll,np.array(conceptl),np.array(inputlabl),np.array(outlabl)

def lossf_rms(output, target):
    """
    Root mean square loss function
    :param input:
    :param output:
    :return:
    """
    lossc = torch.nn.MSELoss()
    # MSELoss is calculating dimension 1
    loss=lossc(torch.t(output), torch.t(target))
    return loss

def run_training_univ(dataset,lsize, model, lossf,step,learning_rate=1e-2, batch=20, save=None):
    """
    Trial of universal training function
    :param dataset:
    :param lsize:
    :param model:
    :param lossf: loss function
    :param step:
    :param learning_rate:
    :param batch:
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

    train_hist = []
    his = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    dataset=torch.from_numpy(dataset)
    dataset=dataset.type(torch.FloatTensor)

    for iis in range(step):
        rstartv = np.floor(np.random.rand(batch) * len(dataset))
        input=None
        for iib in range(batch):
            vec = dataset[int(rstartv[iib]), :]
            if input is None:
                input = vec.view(1, -1)
            else:
                input = torch.cat((input, vec.view(1, -1)), dim=0)

        output=model(input,iter=10)

        loss=lossf(output,input)

        train_hist.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if int(iis / prtstep) != his:
            print("Lost: ", iis, loss.item())
            his = int(iis / prtstep)

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

    return model


def run_training(dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, id_2_vec=None, para=None):
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
    if para is None:
        para=dict([])
    save= para.get("save",None)
    seqtrain = para.get("seqtrain", False)
    supervise_mode = para.get("supervise_mode", False)
    coop= para.get("coop", None)
    coopseq = para.get("coopseq", None)
    cuda_flag = para.get("cuda_flag", True)
    invec_noise=para.get("invec_noise", 0.0)
    pre_training=para.get("pre_training",False)
    loss_clip=para.get("loss_clip",0.0)

    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()
    datab=[]

    if (type(dataset) is dict) != supervise_mode:
        raise Exception("Supervise mode Error.")

    if supervise_mode:
        label=dataset["label"]
        dataset=dataset["dataset"]

    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1.0
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
        coop.eval() # Not ensured to work !!!

    def custom_KNWLoss(outputl, outlab, model, cstep):
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1 #+ 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0.0)
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    train_hist = []
    his = 0

    if cuda_flag:
        gpuavail = torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
    else:
        gpuavail=False
        device = torch.device("cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    if gpuavail:
        rnn.to(device)

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)

        if not supervise_mode:
            # Generating output label
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = dataset[(int(rstartv[iib]) + iiss + 1)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)
        else:
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = label[(int(rstartv[iib]) + iiss)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
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
                    # output, hidden = rnn(x, hidden,wta_noise=1.0+0.0*(1.0-iis/step))
                    output, hidden = rnn(x, hidden)
                    # output = rnn.pre_training(x)
                if type(outputl) == type(None):
                    outputl = output.view(batch, lsize_out, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, lsize_out, -1), output.view(batch, lsize_out, 1)), dim=2)
            loss = custom_KNWLoss(outputl, outlab, rnn, iis)
            # if gpuavail:
            #     del outputl,outlab
            #     torch.cuda.empty_cache()
        else:
            # LSTM/GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1_raw = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
                vec1_rnd=torch.rand(vec1_raw.shape)
                vec1_add=torch.mul((1.0-vec1_raw)*invec_noise,vec1_rnd.double())
                vec1=vec1_raw+vec1_add
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
            if pre_training:
                output, hidden = rnn.pre_training(x, hidden, schedule=iis / step)
            else:
                scheduled= iis / step
                output, hidden = rnn(x, hidden, schedule=scheduled)
            # output, hidden = rnn(x, hidden, wta_noise=0.2 * (1.0 - iis / step))
            loss = custom_KNWLoss(output.permute(1,2,0), outlab, rnn, iis)
            # if gpuavail:
            #     del x,y,outlab
            #     torch.cuda.empty_cache()


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
            if loss_clip>0:
                plt.ylim((0, loss_clip))
            plt.show()
    except:
        pass
    if gpuavail:
        torch.cuda.empty_cache()
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


def run_training_stack(dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, save=None, seqtrain=False,
                 coop=None, coopseq=None, id_2_vec=None):
    """
    General rnn training funtion for one-hot training
    Not working yet !!!!!!
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
        lsize_in = lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()
    datab = []
    if id_2_vec is None:  # No embedding, one-hot representation
        for data in dataset:
            datavec = np.zeros(lsize_in)
            datavec[data] = 1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec = np.array(id_2_vec[data])
            datab.append(datavec)
    databp = torch.from_numpy(np.array(datab))
    if coopseq is not None:
        coopseq = torch.from_numpy(np.array(coopseq))
        coopseq = coopseq.type(torch.FloatTensor)

    rnn.train()

    if coop is not None:
        coop.eval()  # Not ensured to work !!!

    def custom_KNWLoss(outputl, outlab, model, cstep):
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        logith2o = model.h2o.weight
        pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1  # + 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_StackLoss(perpl):
        """
        Special cost for stack FF mode
        :param perpl:
        :param stackl:
        :param input_size:
        :return:
        """
        loss = torch.mean(perpl)
        return loss

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

    train_hist = []
    his = 0

    gpuavail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpuavail else "cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    if gpuavail:
        rnn.to(device)

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
                # print("iiss",iiss)
            #     vec1m = None
            #     vec2m = None
            #     if coopseq is not None:
            #         veccoopm = None
            #     for iib in range(batch):
            #         vec1 = databp[(int(rstartv[iib]) + iiss), :]
            #         vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
            #         if vec1m is None:
            #             vec1m = vec1.view(1, -1)
            #             vec2m = vec2.view(1, -1)
            #         else:
            #             vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
            #             vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
            #         if coopseq is not None:
            #             veccoop = coopseq[(int(rstartv[iib]) + iiss + 1), :]
            #             if veccoopm is None:
            #                 veccoopm = veccoop.view(1, -1)
            #             else:
            #                 veccoopm = torch.cat((veccoopm, veccoop.view(1, -1)), dim=0)
            #     # One by one guidance training ####### error can propagate due to hidden state
            #     x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
            #     y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)

                vec1m = None
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1,window, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, window, -1)), dim=0)
                vec1m=vec1m.type(torch.FloatTensor)
                output, hidden = rnn(vec1m, hidden)
                if type(outputl) == type(None):
                    try:
                        outputl = output.view(batch, 1)
                    except:
                        pass
                elif output is not None:
                    outputl = torch.cat((outputl.view(-1), output.view(-1)), dim=-1)
            # loss = custom_KNWLoss(outputl, outlab, rnn, iis)

            # print("Before loss:",outputl)
            loss = custom_StackLoss(outputl)
            # if gpuavail:
            #     del outputl,outlab
            #     torch.cuda.empty_cache()
        else:
            # LSTM/GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1 = databp[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                vec2 = databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1, :]
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
            output, hidden = rnn(x, hidden, schedule=iis / step)
            # output, hidden = rnn(x, hidden, wta_noise=0.2 * (1.0 - iis / step))
            # loss = custom_KNWLoss(output.permute(1, 2, 0), outlab, rnn, iis)
            loss = output

            # if gpuavail:
            #     del x,y,outlab
            #     torch.cuda.empty_cache()

        if prtstep>0:
            if int(iis / prtstep) != his:
                print("Perlexity: ", iis, np.exp(loss.item()))
                his = int(iis / prtstep)
        else:
            print("Perlexity: ", iis, np.exp(loss.item()))

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
            # plt.ylim((0, 2000))
            plt.show()
    except:
        pass
    torch.cuda.empty_cache()
    return rnn
