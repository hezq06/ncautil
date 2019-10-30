"""
Utility for data processing and enhancement
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.animation import writers,ArtistAnimation
import os
import pickle
import time
import copy

import torch
from torch.autograd import Variable
from ncautil.ncalearn import *
import pickle

from tqdm import tqdm

from torchvision import datasets

# def save(item,path):
#     # torch.save(model.state_dict(), path))
# def load(model,path):
#     gru_gumbel = GRU_seq2seq(lsize_in, 100, lsize_in, outlen, para=gru_para)
#     model.load_state_dict(torch.load("./gru_myhsample_finetune"))
#     gru_gumbel = gru_gumbel.to(cuda_device)


def save_data(data,file):
    pickle.dump(data, open(file, "wb"))
    print("Data saved to ", file)

def load_data(file):
    data = pickle.load(open(file, "rb"))
    print("Data load from ", file)
    return data

def save_model(model,file):
    torch.save(model.state_dict(), file)
    print("Model saved to ", file)

def load_model(model,file):
    model.load_state_dict(torch.load(file))
    print("Model load from ", file)
    return model

def get_id_with_sample_vec(param,vec):
    """
    Working together with onehot_data_enhance_create_param
    get the id a certain sampled vec
    :param param:
    :param vec:
    :return:
    """
    lsize=param.shape[0]
    dim=param.shape[-1]-1
    delts = param[:, :dim] - vec
    dists = np.zeros(lsize)
    for dim_ii in range(dim):
        dists = dists + delts[:, dim_ii] ** 2
    dists = np.sqrt(dists)
    adjdists = dists / param[:, -1]
    argm = np.argmin(adjdists)
    return argm


def onehot_data_enhance_create_param(prior,dim=3,figure=False):
    """
    Create the Afinity scaled K-means like clustering parameter when given prior
    :param prior:
    :param dim:
    :return:
    """
    prior=np.array(prior)/np.sum(np.array(prior))
    lsize=len(prior)

    param = np.random.random((lsize, dim+1))
    lr = 0.05

    for iter in tqdm(range(100)):
        res = [[] for ii in range(lsize)]
        for ii in range(lsize):
            res[ii].append(param[ii, :dim])
        for ii in range(10000):
            dot = np.random.random(dim)
            argm=get_id_with_sample_vec(param,dot)
            res[argm].append(dot)
        caled_dist = np.zeros(lsize)
        for ii in range(lsize):
            caled_dist[ii] = len(res[ii])
        caled_dist = caled_dist / np.sum(caled_dist)
        delta = caled_dist - prior
        param[:, -1] = param[:, -1] - lr * delta

    if figure:
        for ii in range(lsize):
            plt.scatter(np.array(res[ii])[:, 0], np.array(res[ii])[:, 1])
        for ii in range(lsize):
            plt.scatter(param[ii, 0], param[ii, 1], marker='^')
        plt.show()

    return param

def onehot_data_enhance(dataset,prior,dim=3,param=None):
    """
    Enhancement of one-hot dataset [0,12,5,......] using Afinity scaled K-means like clustering
    input is one hot dataset, output is distributed data where clusters of data contains one-hot label information
    :param dataset_onehot:
    :param dim: enhancement dimention
    :return:
    """
    lsize=len(set(dataset))

    print("Building param ...")
    if param is None:
        param=onehot_data_enhance_create_param(prior,dim=dim)
    res_datavec=[]
    print("Enhancing data...")
    hitcnt=0
    misscnt=0
    for iiw in tqdm(range(len(dataset))):
        wid=dataset[iiw]
        hitlab=False
        sample = np.random.random(dim)
        for ii_trial in range(10):
            argm = get_id_with_sample_vec(param, sample)
            if argm==wid:
                hitlab=True
                res_datavec.append(sample)
                hitcnt=hitcnt+1
                break
            else:
                sample = (sample + param[wid, :dim]) / 2
        if ii_trial==9 and (not hitlab):
            res_datavec.append(param[wid,:dim])
            misscnt=misscnt+1
    print("Hit:",hitcnt,"Miss:",misscnt)
    return res_datavec,param

def data_padding(data,endp="#"):
    """
    End data padding with endp
    :param data:
    :param endp:
    :return:
    """
    lenlist=[len(sent) for sent in data]
    maxlen=np.max(np.array(lenlist))
    res=[]
    for sent in data:
        npad=maxlen-len(sent)
        for ii in range(npad+1):
            sent.append(endp)
        res.append(sent)
    return res

def plot_anim(mat_list,file=None,clim=None,interval=200):
    """
    Creat an animation clip from a list of matrix
    :param mat_list:
    :return:
    """
    fig2, ax = plt.subplots()
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Harry'), bitrate=1800)

    ims = []
    for ii in range(len(mat_list)):
        fig = plt.imshow(mat_list[ii], cmap='seismic', clim=clim)
        title = plt.text(0.5, 1.01, "Step " + str(ii), ha="center", va="bottom", transform=ax.transAxes,fontsize="large")
        ims.append((fig, title))

    plt.colorbar(fig)
    im_ani = ArtistAnimation(fig2, ims, interval=interval, repeat=False)
    if file is not None:
        im_ani.save('CorrEvl_hidden0_grad_permute.mp4', writer=writer)
    plt.show()
    return im_ani

class Plus_dataset(object):
    """
    A object creating NNN+NNN=NNN dataset for seq2seq model understanding
    """
    def __init__(self):
        self.num=0
        self.reset()


    def reset(self):
        self.dataset_raw = dict([])
        self.dataset_raw["dataset"] = []
        self.dataset_raw["label"] = []
        self.dataset_sup = dict([])
        self.dataset_sup["dataset"] = []
        self.dataset_sup["label"] = []
        self.digits = None

        self.memmat=None

    def create_dataset(self,num,digits=3,mode="normal",noise_level=None):
        """
        create dataset
        :param num: with number of example to be num
        :param max_range: and maximum digit less than max_range
        :return:
        """
        dataset=[]
        label=[]
        self.memmat=np.zeros((10**digits-1,10**digits-1))
        for ii in range(num):
            dig1=int(np.random.rand()*(10**digits-1))
            dig2 = int(np.random.rand() * (10**digits-1))
            while self.memmat[dig1,dig2]==1: # overlap detection
                dig1 = int(np.random.rand() * (10 ** digits - 1))
                dig2 = int(np.random.rand() * (10 ** digits - 1))
            self.memmat[dig1, dig2]=1
            if mode=="normal":
                dig_ans=int(dig1+dig2)
            elif mode=="random":
                dig_ans=int(np.random.rand() * (10**(digits+1)-1))
            elif mode=="noisy":
                assert noise_level is not None
                if np.random.rand()>noise_level:
                    dig_ans = int(dig1 + dig2)
                else:
                    dig_ans = int(np.random.rand() * (10 ** (digits + 1) - 1))
            else:
                raise Exception("Unknown mode")
            str1=str(dig1)
            # npad = digits - len(str1)
            # for ii in range(npad):
            #     str1 = "0"+str1
            str2 = str(dig2)
            # npad = digits - len(str2)
            # for ii in range(npad):
            #     str2 = "0" + str2
            strdata=str1+"+"+str2+"="
            dataset.append(list(strdata))
            ansdata=str(dig_ans)
            # npad = digits+ 1 - len(ansdata)
            # for ii in range(npad):
            #     ansdata = "0"+ansdata
            label.append(list(ansdata))
        self.num=self.num+num
        self.dataset_raw["dataset"]=data_padding(dataset)
        self.dataset_raw["label"]=data_padding(label)
        self.data_precess()
        self.digits=digits

    def create_dataset_simple(self,num,digits=3,mode="normal",noise_level=None):
        """
        create dataset
        :param num: with number of example to be num
        :param max_range: and maximum digit less than max_range
        :return:
        """
        dataset=[]
        label=[]
        self.memmat=np.zeros((10**digits-1,10**digits-1))
        for ii in range(num):
            dig1=int(np.random.rand()*(10**digits-1))
            dig2 = int(np.random.rand() * (10**digits-1))
            while self.memmat[dig1,dig2]==1: # overlap detection
                dig1 = int(np.random.rand() * (10 ** digits - 1))
                dig2 = int(np.random.rand() * (10 ** digits - 1))
            self.memmat[dig1, dig2]=1
            if mode=="normal":
                dig_ans=int(dig1+dig2)
            elif mode=="random":
                dig_ans=int(np.random.rand() * (10**(digits+1)-1))
            elif mode=="noisy":
                assert noise_level is not None
                if np.random.rand()>noise_level:
                    dig_ans = int(dig1 + dig2)
                else:
                    dig_ans = int(np.random.rand() * (10 ** (digits + 1) - 1))
            else:
                raise Exception("Unknown mode")
            str1=str(dig1)
            npad = digits - len(str1)
            for ii in range(npad):
                str1 = "0"+str1
            str2 = str(dig2)
            npad = digits - len(str2)
            for ii in range(npad):
                str2 = "0" + str2
            strdata=str1+str2
            dataset.append(list(strdata))
            ansdata=str(dig_ans)
            npad = digits+ 1 - len(ansdata)
            for ii in range(npad):
                ansdata = "0"+ansdata
            label.append(list(ansdata))
        self.num=self.num+num
        # self.dataset_raw["dataset"]=data_padding(dataset)
        # self.dataset_raw["label"]=data_padding(label)
        self.dataset_raw["dataset"]=dataset
        self.dataset_raw["label"]=label
        self.data_precess()
        self.digits=digits

    def data_precess(self):
        """
        Transfer data to digits
        :return:
        """
        wrd_2_id = dict([])
        for ii in range(10):
            wrd_2_id[str(ii)] = ii
        wrd_2_id["+"] = 10
        wrd_2_id["="] = 11
        wrd_2_id["#"] = 12

        for sent in self.dataset_raw["dataset"]:
            trans_sent=[]
            for chr in sent:
                trans_sent.append(wrd_2_id[chr])
            self.dataset_sup["dataset"].append(trans_sent)

        for sent in self.dataset_raw["label"]:
            trans_sent=[]
            for chr in sent:
                trans_sent.append(wrd_2_id[chr])
            self.dataset_sup["label"].append(trans_sent)

    # def data_precess_v2(self):
    #     """
    #     Transfer data to digits
    #     :return:
    #     """
    #     wrd_2_id = dict([])
    #     for ii in range(10):
    #         wrd_2_id[str(ii)] = ii
    #     wrd_2_id["+"] = 10
    #     wrd_2_id["="] = 11
    #     wrd_2_id["#"] = 12
    #
    #     for ii in range(len(self.dataset_raw["dataset"])):
    #         trans_sent=[]
    #         for chr in self.dataset_raw["dataset"][ii]:
    #             trans_sent.append(wrd_2_id[chr])
    #         for chr in self.dataset_raw["label"][ii]:
    #             trans_sent.append(wrd_2_id[chr])
    #         del trans_sent[-1]
    #         self.dataset_sup["dataset"].append(trans_sent)
    #
    #     for sent in self.dataset_raw["label"]:
    #         trans_sent=[]
    #         for chr in sent:
    #             trans_sent.append(wrd_2_id[chr])
    #         self.dataset_sup["label"].append(trans_sent)


    def print_example(self,num):
        """
        print num of examples
        :param num:
        :return:
        """
        print("Number of data is",self.num)
        print("digits is,", self.digits)
        for ii in range(num):
            idn=int(np.random.rand()*self.num)
            print("Q:",self.dataset_raw["dataset"][idn])
            print("A:", self.dataset_raw["label"][idn])

class MNIST_dataset(object):
    """
    pytorch mnist dataset
    """
    def __init__(self):

        data_train = datasets.MNIST(root="./data/", train=True)
        data_test = datasets.MNIST(root="./data/", train=False)

        self.dataset_sup=dict([])
        dshape=data_train.train_data.shape
        self.dataset_sup["dataset"] = data_train.train_data.reshape(dshape[0],-1).type(torch.FloatTensor) # 1D version
        self.dataset_sup["dataset"] = self.dataset_sup["dataset"] / 256.0
        self.dataset_sup["label"] = data_train.train_labels

        self.dataset_sup_test=dict([])
        dshape = data_test.test_data.shape
        self.dataset_sup_test["dataset"] = data_test.test_data.reshape(dshape[0],-1).type(torch.FloatTensor)
        self.dataset_sup_test["dataset"] = self.dataset_sup_test["dataset"] / 256.0
        self.dataset_sup_test["label"] = data_test.test_labels
