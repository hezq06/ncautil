"""
Utility for data processing and enhancement
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import copy

import torch
from torch.autograd import Variable
from ncautil.nlputil import NLPutil
from ncautil.ncalearn import *

from tqdm import tqdm

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

