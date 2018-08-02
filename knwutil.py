"""
Utility for Knowledge project development
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

from ncautil.ncalearn import cal_entropy,cal_kldiv

class KNW_OBJ(object):
    """
    Root node for knowledge
    """
    def __init__(self,lsize,prior=None):
        self.id_to_word=dict([])
        self.lsize = lsize
        # possibility prior distribution
        if prior is not None:
            self.prior=prior
        else:
            self.prior = np.ones(lsize)/lsize
        self.style=None
        self.description=None
        self.data=None
        self.knw_fingerp=None
        self.knw_utility=None
        self.knw_outmask = None
        self.knw_untilmask = None
        self.logith = 1.0 # Confidence of this knowledge
        self.eval_cnt=None # event cnt of this knowledge [total correct]

        self.knw_t=[1.0,0.2] # group thread
        self.theta_t=1.0

        self.rnn = None

    def distill_create(self):
        raise NotImplementedError

    def create(self, data):
        raise NotImplementedError

    def cal_knwU(self):
        raise NotImplementedError

    def build_rnn(self):
        raise NotImplementedError

    def eval_rate(self,y,fcnt=0):
        raise NotImplementedError

    def knw_distill(self):
        raise NotImplementedError

    def knwg_detect(self,knw_on):
        """
        Common function, detect an ON group
        :param vec:
        :return: list
        """
        knw_on_zip = list(zip(knw_on, range(len(knw_on))))
        knw_on_zip.sort(key=lambda x: x[0], reverse=True)
        maxv = knw_on_zip[0][0]
        maxvw = maxv - self.knw_t[1]
        maxvt = maxv - self.knw_t[0]
        resknw_N = []
        kstr = ""
        found = False
        for iip in range(len(knw_on_zip) - 1):
            if knw_on_zip[iip][0] > maxvw and knw_on_zip[iip + 1][0] < maxvt:
                for iik in range(iip + 1):
                    kid = knw_on_zip[iik][1]
                    resknw_N.append(kid)
                    kstr = kstr + str(kid) + ", "
                print("Expert for " + kstr + " detected!")
                found=True
                break
        if not found:
            print("Knowledge not found!")
        return resknw_N

    def thred_detect(self,theta_on):
        restheta_N=[]
        for ii in range(len(theta_on)):
            if theta_on[ii]>=self.theta_t:
                restheta_N.append(ii)
        return restheta_N

class KNW_IITNN(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(self.__class__, self).__init__(lsize,prior=prior)

    def distill_create(self):
        """
        Distill & create
        :return:
        """
        knwls = []
        items, expertInds = self.knw_distill()
        for item in items:
            for expertInd in expertInds:
                knw_obj = self.__class__(self.lsize)
                knw_obj.create([item,expertInd])
                knwls.append(knw_obj)
        return knwls

    def create(self, data):
        """
        Create a if item then next not knowledge
        :param data: data[0] is If item, data[1] is expertInd
        :return:
        """
        item=data[0]
        expertInd=data[1]
        self.style = "IITNN"  # if item then next not
        self.description = "If " + str(item) + " then next not " + str(expertInd)
        self.data = [item,expertInd]
        self.knw_fingerp = "IITNN" + "-" + str(item) + "-" + str(expertInd)
        self.knw_utility = self.cal_knwU()
        tmp = np.ones(self.lsize)
        tmp[expertInd] = 0
        self.knw_outmask = tmp

    def cal_knwU(self):
        tmp = self.prior.copy()
        tmp[self.data[-1]] = 0
        KnwU = cal_kldiv(tmp, self.prior) * (self.prior[self.data[0]] / np.sum(self.prior))
        return KnwU

    def build_rnn(self):
        plogits = self.prior
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        rnn = KNW_CELL(self.lsize,"IITNN",plogits)
        self.rnn=rnn
        return rnn

    def eval_rate(self,y,fcnt=0):
        mask = self.knw_outmask
        predvec = np.zeros(self.lsize)
        predvec[y] = 1
        mres = np.max(mask * predvec)
        self.eval_cnt[0] = self.eval_cnt[0] + 1
        if mres > 0:  # Compatible
            self.eval_cnt[1] = self.eval_cnt[1] + 1
        return 0

    def knw_distill(self):
        knw_b = self.rnn.h2o.bias.data.numpy()
        knw_w = self.rnn.h2o.weight.data.numpy()[:,0]
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()[0,:]
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        return [setId,expertInd]

class KNW_IITNI(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(self.__class__, self).__init__(lsize,prior=prior)

    def distill_create(self):
        """
        Distill & create
        :return:
        """
        knwls = []
        items, expertInds = self.knw_distill()
        for item in items:
            for expertInd in expertInds:
                knw_obj = self.__class__(self.lsize)
                knw_obj.create([item,expertInd])
                knwls.append(knw_obj)
        return knwls

    def create(self, data):
        """
        Create a if item then next is knowledge
        :param data: data[0] is If item, data[1] is expertInd
        :return:
        """
        item=data[0]
        expertInd=data[1]
        self.style = "IITNI"  # if item then next is
        self.description = "If " + str(item) + " then next is " + str(expertInd)
        self.data = [item,expertInd]
        self.knw_fingerp = "IITNI" + "-" + str(item) + "-" + str(expertInd)
        self.knw_utility = self.cal_knwU()
        tmp = np.zeros(self.lsize)
        tmp[expertInd] = 1
        self.knw_outmask = tmp

    def cal_knwU(self):
        tmp = self.prior.copy()
        tmp[self.data[-1]] = 0
        KnwU = cal_kldiv(tmp, self.prior) * (self.prior[self.data[0]] / np.sum(self.prior))
        return KnwU

    def build_rnn(self):
        plogits = self.prior
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        rnn=KNW_CELL(self.lsize,"IITNI",plogits)
        self.rnn=rnn
        return rnn

    def eval_rate(self,y,fcnt=0):
        mask = self.knw_outmask
        predvec = np.zeros(self.lsize)
        predvec[y] = 1
        mres = np.max(mask * predvec)
        self.eval_cnt[0] = self.eval_cnt[0] + 1
        if mres > 0:  # Compatible
            self.eval_cnt[1] = self.eval_cnt[1] + 1
        return 0

    def knw_distill(self):
        knw_b = self.rnn.h2o.bias.data.numpy()
        knw_w = self.rnn.h2o.weight.data.numpy()[:,0]
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()[0,:]
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        return [setId,expertInd]

class KNW_SETRESET(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(self.__class__, self).__init__(lsize,prior=prior)
        self.knw_ckeck = False

    def distill_create(self):
        """
        Distill & create
        :return:
        """
        knwls = []
        items, resetId, expertInds = self.knw_distill()
        for item in items:
            for expertInd in expertInds:
                if resetId:
                    knw_obj = self.__class__(self.lsize)
                    knw_obj.create([item,resetId,expertInd])
                    knwls.append(knw_obj)
        return knwls

    def create(self, data):
        """
        Create a setter resetter knowledge
        :param data: data[0] is If item, data[1] is list of resetter,data[2] is expertInd
        :return:
        """
        item=data[0]
        item_r=data[1]
        expertInd=data[2]
        self.rnn = self.rnn
        self.style = "SETRESET"
        self.description = "If " + str(item) + " then next is " + str(expertInd) + " until " + str(item_r)
        self.data = [item, item_r, expertInd]
        self.knw_fingerp = "SETRESET" + "-s-" + str(item) + "-r-" + str(item_r) + "-" + str(expertInd)
        self.knw_utility = self.cal_knwU()
        tmp = np.zeros(self.lsize)
        tmp[expertInd] = 1
        self.knw_outmask = tmp
        tmp2 = np.zeros(self.lsize)
        tmp2[item_r] = 1
        self.knw_untilmask = tmp2

    def cal_knwU(self):
        tmp = self.prior.copy()
        tmp[self.data[-1]] = 0
        KnwU = cal_kldiv(tmp, self.prior) * (self.prior[self.data[0]] / np.sum(self.prior))
        return KnwU

    def build_rnn(self):
        plogits = self.prior
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        rnn=KNW_CELL(self.lsize,"SETRESET",plogits)
        self.rnn = rnn
        return rnn

    def eval_rate(self,y,fcnt=0):
        """

        :param y:
        :param fcnt: first hit
        :return:
        """
        assert fcnt in [0,1]
        self.eval_cnt[0] = self.eval_cnt[0] + fcnt
        predvec = np.zeros(self.lsize)
        predvec[y] = 1
        masko = self.knw_outmask
        mres = np.max(masko * predvec)
        if mres > 0:  # Compatible
            self.knw_ckeck = True
        masku = self.knw_untilmask
        mresu = np.max(masku * predvec)
        if mresu > 0:  # Until hit
            if self.knw_ckeck: # Compatible
                self.eval_cnt[1] = self.eval_cnt[1] + 1
                self.knw_ckeck = False
            return -1
        else:
            return fcnt

    def knw_distill(self):
        knw_b = self.rnn.h2o.bias.data.numpy()
        knw_w = self.rnn.h2o.weight.data.numpy()[:,0]
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()[0,:]
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        Wr_b = self.rnn.Wr.bias.data.numpy()
        Wr_w = self.rnn.Wr.weight.data.numpy()[0,:]
        Wr = Wr_b + Wr_w
        resetId = self.thred_detect(Wr)
        return [setId,resetId,expertInd]

class KNW_SETRESETN(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(self.__class__, self).__init__(lsize,prior=prior)
        self.knw_ckeck = False

    def distill_create(self):
        """
        Distill & create
        :return:
        """
        knwls = []
        items, item_r, expertInds = self.knw_distill()
        for item in items:
            for expertInd in expertInds:
                if item_r:
                    knw_obj = self.__class__(self.lsize)
                    knw_obj.create([item,item_r,expertInd])
                    knwls.append(knw_obj)
        return knwls

    def create(self,data):
        """
         Create a setter resetter non knowledge
        :param data: data[0] is If item, data[1] is list of resetter,data[2] is expertInd
        :return:
        """
        item=data[0]
        item_r = data[1]
        expertInd=data[2]
        self.style = "SETRESETN"
        self.description = "If " + str(item) + " then next is non-" + str(expertInd) + " until " + str(item_r)
        self.data = [item, item_r, expertInd]
        self.knw_fingerp = "SETRESETN" + "-s-" + str(item) + "-r-" + str(item_r) + "-" + str(expertInd)
        self.knw_utility = self.cal_knwU()
        tmp = np.ones(self.lsize)
        tmp[expertInd] = 0
        self.knw_outmask = tmp
        tmp2 = np.zeros(self.lsize)
        tmp2[item_r] = 1
        self.knw_untilmask = tmp2

    def cal_knwU(self):
        tmp = self.prior.copy()
        tmp[self.data[-1]] = 0
        KnwU = cal_kldiv(tmp, self.prior) * (self.prior[self.data[0]] / np.sum(self.prior))
        return KnwU

    def build_rnn(self):
        plogits = self.prior
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        rnn=KNW_CELL(self.lsize, "SETRESETN", plogits)
        self.rnn = rnn
        return rnn

    def eval_rate(self, y, fcnt=0):
        """

        :param y:
        :param fcnt: first cnt
        :return:
        """
        assert fcnt in [0, 1]
        self.eval_cnt[0] = self.eval_cnt[0] + fcnt
        predvec = np.zeros(self.lsize)
        predvec[y] = 1
        masko = self.knw_outmask
        mres = np.max(masko * predvec)
        if mres == 0:  # Incompatible
            self.knw_ckeck = True
        masku = self.knw_untilmask
        mresu = np.max(masku * predvec)
        if mresu > 0:  # Until hit
            if not self.knw_ckeck:
                self.eval_cnt[1] = self.eval_cnt[1] + 1
            self.knw_ckeck = False
            return -1
        else:
            return fcnt

    def knw_distill(self):
        knw_b = self.rnn.h2o.bias.data.numpy()
        knw_w = self.rnn.h2o.weight.data.numpy()[:,0]
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()[0,:]
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        Wr_b = self.rnn.Wr.bias.data.numpy()
        Wr_w = self.rnn.Wr.weight.data.numpy()[0,:]
        Wr = Wr_b + Wr_w
        resetId = self.thred_detect(Wr)
        return [setId,resetId,expertInd]

class KNW_CELL(torch.nn.Module):
    """
    PyTorch knowledge cell for single piece of knowledge
    """
    def __init__(self, lsize, mode, prior=None):
        """

        :param lsize:
        :param mode:
        """
        super(KNW_CELL, self).__init__()
        # Setter or detector
        self.lsize=lsize
        self.Ws = torch.nn.Linear(lsize, 1)
        # Resetter
        self.Wr = torch.nn.Linear(lsize, 1)
        # Knowledge
        self.h2o = torch.nn.Linear(1, lsize, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if prior is not None:
            self.prior=prior
        else:
            self.prior = 0

        if mode=="IITNN":
            self.ctrl = [-1, 0]
        elif mode=="IITNI":
            self.ctrl = [1, 0]
        elif mode=="SETRESET":
            self.ctrl = [1, 1]
        elif mode=="SETRESETN":
            self.ctrl = [-1, 1]
        else:
            raise Exception("Mode not supported")

    def forward(self, input, hidden, add_prior=None):
        """
        Forward
        :param input:
        :param hidden:
        :param plogits: prior
        :return:
        """
        ht=self.sigmoid(self.Ws(input))+hidden
        output = self.ctrl[0]*self.h2o(ht) + self.prior
        output = self.softmax(output)
        if add_prior is not None:
            output=output+add_prior
        resetter = 1 - self.sigmoid(self.Wr(input))
        hidden=resetter*ht*self.ctrl[1]
        return output, hidden

    def initHidden(self,batch=1):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True)

class KNW_CELL_ORG(torch.nn.Module):
    """
    PyTorch knowledge cell for whole knowledge database
    """
    def __init__(self, lsize, knw_list, knw_gate_index, knw_list_fingerp):
        super(self.__class__, self).__init__()
        self.knw_size = len(knw_list)
        self.knw_para = torch.nn.Parameter(torch.ones(self.knw_size), requires_grad=True)
        # knowledge mask mat (knw_size,lsize)

        knw_maskmat = np.ones((self.knw_size, lsize))
        # knowledge act mat (lsize,knw_size)
        for iik in range(self.knw_size):
            knw_maskmat[iik,:]=knw_list[iik].knw_outmask
        self.knw_maskmat=torch.from_numpy(knw_maskmat)
        self.knw_maskmat=self.knw_maskmat.type(torch.FloatTensor)

        knw_actmat = np.zeros((lsize, self.knw_size))
        for iil in range(lsize):
            for iia in range(len(knw_gate_index[iil])):
                ida = knw_list_fingerp.index(knw_gate_index[iil][iia])
                knw_actmat[iil, ida] = 1
        self.knw_actmat = torch.from_numpy(knw_actmat)
        self.knw_actmat = self.knw_actmat.type(torch.FloatTensor)

        knw_resetmat = np.zeros((lsize, self.knw_size))
        for iik in range(len(knw_list)):
            kobj=knw_list[iik]
            if kobj.knw_untilmask is not None:
                knw_resetmat[:,iik]=1.0-kobj.knw_untilmask
        self.knw_resetmat = torch.from_numpy(knw_resetmat)
        self.knw_resetmat = self.knw_resetmat.type(torch.FloatTensor)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, plogits=None):
        """
        Forward
        :param input:
        :param hidden:
        :param plogits: prior
        :return:
        """
        knw_act=torch.matmul(input,self.knw_actmat)+hidden
        scaled_act=knw_act*self.knw_para
        knw_vec=torch.matmul(scaled_act,self.knw_maskmat)+plogits
        output=self.softmax(knw_vec)
        hiddenresetter=torch.matmul(input,self.knw_resetmat)
        hidden=knw_act*hiddenresetter
        return output,hidden

    def initHidden(self,batch=1):
        return Variable(torch.zeros(1, batch, self.knw_size), requires_grad=True)

class KNW_ORG(object):
    """
    Knowledge organizor
    """
    def __init__(self, dataset, lsize, prior=None):
        self.prior=prior
        self.dataset=dataset
        self.lsize=lsize
        self.knw_list=[]
        self.knw_list_fingerp=[]
        self.knw_gate_index = [[] for ii in range(self.lsize)]

        self.mlost = 1.0e9
        self.model = None
        self.knw_obj = None # current self.knw_obj

        self.knw_actmat = None
        self.knw_maskmat = None
        self.knw_resetmat = None
        self.knw_para = None

    def insert(self):
        if self.knw_obj is None:
            print("No ready knowledge found")
        else:
            knw_objs=self.knw_obj.distill_create()
            if not knw_objs:
                print("No knowledge extractable.")
            for knw_obj in knw_objs:
                if knw_obj.knw_fingerp not in self.knw_list_fingerp:
                    self.knw_list.append(knw_obj)
                    self.knw_list_fingerp.append(knw_obj.knw_fingerp)
                    self.knw_gate_index[knw_obj.data[0]].append(knw_obj.knw_fingerp)
                else:
                    print(knw_obj.knw_fingerp+" already exist.")

    def remove(self,knw_fingerp):
        """
        Delete a certain knowledge
        :param knw_fingerp: knowledge finger print.
        :return:
        """
        if knw_fingerp not in self.knw_list_fingerp:
            print("Knowledge not found.")
        else:
            ind=self.knw_list_fingerp.index(knw_fingerp)
            del self.knw_list[ind]
            del self.knw_list_fingerp[ind]
            self.knw_gate_index = [[] for ii in range(self.lsize)]
            for knwitem in self.knw_list:
                self.knw_gate_index[knwitem.data[0]].append(knwitem.knw_fingerp)

    def get(self,fingerp):
        """
        Get knw obj by fingerp
        :param fingerp:
        :return:
        """
        ind=self.knw_list_fingerp.index(fingerp)
        return self.knw_list[ind],ind

    def print(self):
        print("No. of knowledge: ",len(self.knw_list))
        for ii in range(len(self.knw_list)):
            print(self.knw_list_fingerp[ii], " : " , self.knw_list[ii].description, self.knw_list[ii].knw_utility, self.knw_list[ii].eval_cnt, self.knw_list[ii].logith)

    def save(self,name="knwlist.pickle"):
        """
        Saving key knowledge to a list
        :param name:
        :return:
        """
        knw_list_save = []
        print("No. of knowledge to be saved: ",len(self.knw_list))
        for knwobj in self.knw_list:
            knw_dict=dict([])
            knw_dict["style"]= knwobj.style
            knw_dict["data"] = knwobj.data
            knw_dict["logith"] = knwobj.logith
            knw_dict["eval_cnt"] = knwobj.eval_cnt
            knw_list_save.append(knw_dict)
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
        print(file)
        pickle.dump(knw_list_save, open(file, "wb"))
        print("Knowledge data list saved.")

    def load(self,name="knwlist.pickle"):
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
        print(file)
        knw_list = pickle.load(open(file, "rb"))
        self.knw_list = []
        self.knw_list_fingerp = []
        self.knw_gate_index = [[] for ii in range(self.lsize)]
        for knwd in knw_list:
            mode=knwd["style"]
            knw_item=None
            if mode=="IITNN":
                knw_item=KNW_IITNN(self.lsize)
            elif mode=="IITNI":
                knw_item=KNW_IITNI(self.lsize)
            elif mode=="SETRESET":
                knw_item=KNW_SETRESET(self.lsize)
            elif  mode=="SETRESETN":
                knw_item=KNW_SETRESETN(self.lsize)
            knw_item.create(knwd["data"])
            knw_item.logith = knwd["logith"]
            knw_item.eval_cnt = knwd["eval_cnt"]
            self.knw_list.append(knw_item)
            self.knw_list_fingerp.append(knw_item.knw_fingerp)
            self.knw_gate_index[knwd["data"][0]].append(knw_item.knw_fingerp)

    def eval_rate(self):
        """
        Knowledge evaluation
        :param dataset:
        :return:
        """
        dataset=self.dataset
        print("Knowledge evaluating ...")
        for knwobj in self.knw_list:
            knwobj.eval_cnt = [0, 0]
        ltknwcnt = np.zeros(len(self.knw_list)) # knowledge history counter
        for nn in range(len(dataset) - 1):
            x = dataset[nn]
            y = dataset[nn + 1]
            for knwid in self.knw_gate_index[x]:
                knwobj, ind = self.get(knwid)
                for iicnt in range(len(ltknwcnt)):
                    if ltknwcnt[iicnt]>0:
                        rescnt=self.knw_list[iicnt].eval_rate(y,fcnt=0)
                        if rescnt==-1:
                            ltknwcnt[iicnt]=0
                rescnt=knwobj.eval_rate(y,fcnt=1)
                ltknwcnt[ind] = ltknwcnt[ind] + rescnt
        self.print()

    def knw_plot(self):
        """
        Plot last learned knowledge
        :return:
        """
        plt.figure()
        vh2ob = self.model.h2o.bias.data.numpy()
        vh2ow = self.model.h2o.weight.data.numpy()[:, 0]
        plt.bar(np.array(range(len(self.model.h2o.bias))), vh2ob)
        plt.bar(np.array(range(len(self.model.h2o.bias))) + 0.3, vh2ow + vh2ob)
        plt.title("h2o plot")
        plt.show()
        plt.figure()
        vsb = self.model.Ws.bias.data.numpy()
        vsw = self.model.Ws.weight.data.numpy()[0, :]
        plt.bar(np.array(range(len(self.model.h2o.bias))), vsb)
        plt.bar(np.array(range(len(self.model.h2o.bias))) + 0.3, vsw + vsb)
        plt.title("Ws plot")
        plt.show()
        plt.figure()
        vrb = self.model.Wr.bias.data.numpy()
        vrw = self.model.Wr.weight.data.numpy()[0,:]
        plt.bar(np.array(range(len(self.model.h2o.bias))), vrb)
        plt.bar(np.array(range(len(self.model.h2o.bias))) + 0.3, vrw + vrb)
        plt.title("Wr plot")
        plt.show()

    def eval_perplexity(self):
        """
        Knowledge evaluation on dataset, perplexity
        :param dataset:
        :return:
        """
        print("Knowledge evaluating ...")
        dataset = self.dataset
        self.forward_init(1)
        perpls=[]
        knw_size = len(self.knw_list)
        hidden=torch.zeros(1, 1, knw_size)
        datab=[]
        for data in dataset:
            datavec=np.zeros(self.lsize)
            datavec[data]=1
            datab.append(datavec)
        datab=np.array(datab)
        databpt=torch.from_numpy(datab)
        databpt=databpt.type(torch.FloatTensor)
        for nn in range(len(databpt)-1):
            x = databpt[nn]
            y = databpt[nn+1]
            prd,hidden = self.forward(x.view(1,1,self.lsize),hidden)
            prd = torch.exp(prd) / torch.sum(np.exp(prd))
            perp=cal_kldiv(y,prd.view(-1))
            perpls.append(perp)
        avperp=np.mean(np.array(perpls))
        print("Calculated knowledge perplexity:",np.exp(avperp))
        return np.exp(avperp)

    def forward_init(self,batch):
        """
        Initialization for forward
        :return:
        """
        knw_size=len(self.knw_list)
        knw_maskmat = np.ones((knw_size, self.lsize))
        # knowledge act mat (lsize,knw_size)
        for iik in range(knw_size):
            knw_maskmat[iik, :] = self.knw_list[iik].knw_outmask
        self.knw_maskmat = torch.from_numpy(knw_maskmat)
        self.knw_maskmat = self.knw_maskmat.type(torch.FloatTensor)

        knw_actmat = np.zeros((self.lsize, knw_size))
        for iil in range(self.lsize):
            for iia in range(len(self.knw_gate_index[iil])):
                ida = self.knw_list_fingerp.index(self.knw_gate_index[iil][iia])
                knw_actmat[iil, ida] = 1
        self.knw_actmat = torch.from_numpy(knw_actmat)
        self.knw_actmat = self.knw_actmat.type(torch.FloatTensor)

        knw_resetmat = np.zeros((self.lsize, knw_size))
        for iik in range(len(self.knw_list)):
            kobj = self.knw_list[iik]
            if kobj.knw_untilmask is not None:
                knw_resetmat[:, iik] = 1.0 - kobj.knw_untilmask
        self.knw_resetmat = torch.from_numpy(knw_resetmat)
        self.knw_resetmat = self.knw_resetmat.type(torch.FloatTensor)

        self.knw_para = torch.zeros(knw_size)
        knw_para=np.zeros(knw_size)
        for iik in range(knw_size):
            knw_para[iik]=self.knw_list[iik].logith
        tLogith =torch.from_numpy(knw_para)
        tLogith = tLogith.type(torch.FloatTensor)
        self.knw_para=tLogith

        return torch.zeros(1, batch, knw_size)

    def forward(self,inputm, hidden, plogits=None):
        """
        Forwarding and calculate logit with knowledge
        assert shape(1,batch,lsize)
        :param logith: logit threshold for hard knowledge
        :return:
        """
        if self.knw_actmat is None:
            self.forward_init()

        assert inputm.shape[0] == 1
        assert inputm.shape[2] == self.lsize
        assert len(inputm.shape) == 3

        knw_act = torch.matmul(inputm, self.knw_actmat) + hidden
        scaled_act = knw_act * self.knw_para
        knw_vec = torch.matmul(scaled_act, self.knw_maskmat)
        if plogits is not None:
            knw_vec=knw_vec + plogits
        hiddenresetter = torch.matmul(inputm, self.knw_resetmat)
        hidden = knw_act * hiddenresetter
        return knw_vec,hidden

    def optimize_logith_torch(self,step,learning_rate=1e-2,batch=20, window=110):
        """
        Use pytorch batched version to do knowledge optimizing
        :param dataset:
        :param bper:
        :param step:
        :return:
        """
        prtstep=int(step/10)
        print("Knowledge para pytorch optimizing ...")
        dataset = self.dataset
        if self.prior is not None:
            plogits = np.log(self.prior)
        else:
            plogits = np.zeros(self.lsize)
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        datab=[]
        for data in dataset:
            datavec=np.zeros(self.lsize)
            datavec[data]=1
            datab.append(datavec)
        datab=np.array(datab)
        databp = torch.from_numpy(datab)

        rnn=KNW_CELL_ORG(self.lsize,self.knw_list,self.knw_gate_index,self.knw_list_fingerp)
        lossc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        train_hist = []
        his = 0

        for iis in range(step):

            hidden = rnn.initHidden(batch)

            rstartv = np.floor(np.random.rand(batch) * (len(self.dataset) - window - 1))
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
            outputl = None

            for iiss in range(window):
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databp[(int(rstartv[iib]) + iiss), :]
                    vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                # One by one guidance training
                x = Variable(vec1m.reshape(1, batch, self.lsize).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, self.lsize).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output,hidden = rnn(x,hidden,plogits)
                if type(outputl) == type(None):
                    outputl = output.view(batch, self.lsize, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, self.lsize, -1), output.view(batch, self.lsize, 1)), dim=2)
            loss = lossc(outputl, outlab)

            if int(iis / prtstep) != his:
                print("Perlexity: ",iis, np.exp(loss.item()))
                his=int(iis / prtstep)

            train_hist.append(np.exp(loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        res=rnn.knw_para.data.numpy()
        for iit in range(len(res)):
            self.knw_list[iit].logith = res[iit]

    def run_training(self, step, mode, knw=True, learning_rate=1e-2, batch=20, window=110, save=None):
        """
        Main training entrance
        :param dataset: sequence of integer numbers
        :param step:
        :param mode: IITNN, IITNI, SETRESET, SETRESETN
        :param knw: If or not to substract already learned knowledge
        :param learning_rate:
        :param batch:
        :param window:
        :param save:
        :return:
        """
        if len(self.knw_list)==0:
            knw=False
        prtstep = int(step / 10)
        startt = time.time()
        lsize = self.lsize
        dataset = self.dataset
        datab=[]
        for data in dataset:
            datavec=np.zeros(lsize)
            datavec[data]=1
            datab.append(datavec)
        databp=torch.from_numpy(np.array(datab))

        # mode: IITNN, IITNI, SETRESET, SETRESETN
        if mode in ["IITNN", "IITNI", "SETRESET", "SETRESETN"]:
            if mode=="IITNN":
                knw_obj=KNW_IITNN(lsize)
            elif mode=="IITNI":
                knw_obj=KNW_IITNI(lsize)
            elif mode=="SETRESET":
                knw_obj=KNW_SETRESET(lsize)
            else: # mode=="SETRESETN":
                knw_obj=KNW_SETRESETN(lsize)
            rnn = knw_obj.build_rnn()
            self.model=rnn
            self.knw_obj=knw_obj
        elif mode=="CONTINUE" and self.model is not None:
            rnn=self.model
        else:
            raise Exception("Mode not supported")

        rnn.train()

        def custom_KNWLoss(outputl, outlab, model, cstep):
            lossc = torch.nn.CrossEntropyLoss()
            loss1 = lossc(outputl, outlab)
            logith2o = model.h2o.weight  # +model.h2o.bias.view(-1) size(47,cell)
            pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
            lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
            l1_reg = model.Ws.weight.norm(1) + model.Wr.weight.norm(1)
            return loss1 + 0.0005 * l1_reg * cstep / step + 0.005 * lossh2o * cstep / step  # +0.3*lossz+0.3*lossn #

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        train_hist = []
        his = 0

        for iis in range(step):

            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

            hidden = rnn.initHidden(batch)

            if knw:
                hidden_korg = self.forward_init(batch)

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
            outputl = None
            for iiss in range(window):
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databp[(int(rstartv[iib]) + iiss), :]
                    vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                # One by one guidance training ####### error can propagate due to hidden state
                x = Variable(vec1m.reshape(1, batch, lsize).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, lsize).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if knw:
                    add_tvec, hidden_korg = self.forward(x, hidden_korg)
                    output, hidden = rnn(x, hidden, add_prior=add_tvec)
                else:
                    output, hidden = rnn(x, hidden)
                if type(outputl) == type(None):
                    outputl = output.view(batch, lsize, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, lsize, -1), output.view(batch, lsize, 1)), dim=2)
            loss = custom_KNWLoss(outputl, outlab, rnn, iis)

            if int(iis / prtstep) != his:
                print("Perlexity: ", iis, np.exp(loss.item()))
                his = int(iis / prtstep)
                if loss.item() < self.mlost:
                    self.mlost = loss.item()
                    self.model = copy.deepcopy(rnn)

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


