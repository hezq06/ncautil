"""
Utility for Knowledge project development
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
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

    def create(self):
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
        super(KNW_IITNN, self).__init__(lsize,prior=prior)

    def create(self):
        """
        Create a if item then next not knowledge
        :param data: data[0] is If item, data[1] is expertInd
        :return:
        """
        items,expertInds = self.knw_distill()
        self.style = "IITNN"  # if item then next not
        self.description = "If " + str(items) + " then next not " + str(expertInds)
        self.data = [items,expertInds]
        self.knw_fingerp = "IITNN" + "-" + str(items) + "-" + str(expertInds)
        self.knw_utility = self.cal_knwU()
        tmp = np.ones(self.lsize)
        tmp[expertInds] = 0
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
        knw_w = self.rnn.h2o.weight.data.numpy()
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        return [setId,expertInd]

class KNW_IITNI(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(KNW_IITNI, self).__init__(lsize,prior=prior)

    def create(self):
        """
        Create a if item then next is knowledge
        :param data: data[0] is If item, data[1] is expertInd
        :return:
        """
        items, expertInds = self.knw_distill()
        self.style = "IITNI"  # if item then next is
        self.description = "If " + str(items) + " then next is " + str(expertInds)
        self.data = [items,expertInds]
        self.knw_fingerp = "IITNI" + "-" + str(items) + "-" + str(expertInds)
        self.knw_utility = self.cal_knwU()
        tmp = np.zeros(self.lsize)
        tmp[expertInds] = 1
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
        return KNW_CELL(self.lsize,"IITNI",plogits)

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
        knw_w = self.rnn.h2o.weight.data.numpy()
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        return [setId,expertInd]

class KNW_SETRESET(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(KNW_SETRESET, self).__init__(lsize,prior=prior)
        self.knw_ckeck = False

    def create(self):
        """
        Create a setter resetter knowledge
        :param data: data[0] is If item, data[1] is list of resetter,data[2] is expertInd
        :return:
        """
        item_s, item_r, expertInds = self.knw_distill()
        self.style = "SETRESET"
        self.description = "If " + str(item_s) + " then next is " + str(expertInds) + " until " + str(item_r)
        self.data = [item_s, item_r, expertInds]
        self.knw_fingerp = "SETRESET" + "-s-" + str(item_s) + "-r-" + str(item_r) + "-" + str(expertInds)
        self.knw_utility = self.cal_knwU()
        tmp = np.zeros(self.lsize)
        tmp[expertInds] = 1
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
        return KNW_CELL(self.lsize,"SETRESET",plogits)

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
        knw_w = self.rnn.h2o.weight.data.numpy()
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        Wr_b = self.rnn.Wr.bias.data.numpy()
        Wr_w = self.rnn.Wr.weight.data.numpy()
        resetId = Wr_b + Wr_w
        return [setId,resetId,expertInd]

class KNW_SETRESETN(KNW_OBJ):
    def __init__(self, lsize, prior=None):
        super(KNW_SETRESETN, self).__init__(lsize,prior=prior)
        self.knw_ckeck = False

    def create(self, data):
        """
         Create a setter resetter non knowledge
        :param data: data[0] is If item, data[1] is list of resetter,data[2] is expertInd
        :return:
        """
        item_s, item_r, expertInds = self.knw_distill()
        self.style = "SETRESETN"
        self.description = "If " + str(item_s) + " then next is non-" + str(expertInds) + " until " + str(item_r)
        self.data = [item_s, item_r, expertInds]
        self.knw_fingerp = "SETRESETN" + "-s-" + str(item_s) + "-r-" + str(item_r) + "-" + str(expertInds)
        self.knw_utility = self.cal_knwU()
        tmp = np.ones(self.lsize)
        tmp[expertInds] = 0
        self.knw_outmask = tmp
        tmp2 = np.zeros(self.lsize)
        tmp2[item_r] = 1
        self.knw_untilmask = tmp2

    def build_rnn(self):
        plogits = self.prior
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        return KNW_CELL(self.lsize,"SETRESETN",plogits)

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
        knw_w = self.rnn.h2o.weight.data.numpy()
        knw_on = knw_b + knw_w
        expertInd=self.knwg_detect(knw_on)
        Ws_b=self.rnn.Ws.bias.data.numpy()
        Ws_w = self.rnn.Ws.weight.data.numpy()
        Ws=Ws_b+Ws_w
        setId=self.thred_detect(Ws)
        Wr_b = self.rnn.Wr.bias.data.numpy()
        Wr_w = self.rnn.Wr.weight.data.numpy()
        resetId = Wr_b + Wr_w
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
        ht=self.sigmoid(self.Ws(input)+10*hidden)
        output = self.ctrl[0]*self.h2o(ht) + self.prior
        output = self.softmax(output)
        if add_prior is not None:
            output=output+add_prior
        resetter=self.Wr(input)
        hidden=resetter*ht*self.ctrl[1]
        return output, hidden

    def initHidden(self,batch=1):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True)

class KNW_CELL_ORG(torch.nn.Module):
    """
    PyTorch knowledge cell for whole knowledge database
    """
    def __init__(self, lsize, knw_list, knw_gate_index, knw_list_fingerp):
        super(KNW_CELL_ORG, self).__init__()
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
            if self.knw_obj.knw_fingerp not in self.knw_list_fingerp:
                self.knw_obj.knw_distill()
                self.knw_list.append(copy.deepcopy(self.knw_obj))
                self.knw_list_fingerp.append(self.knw_obj.knw_fingerp)
                self.knw_gate_index[self.knw_obj.data[0]].append(self.knw_obj.knw_fingerp)
            else:
                print(self.knw_obj.knw_fingerp+" already exist.")
            self.knw_obj=None

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
            print(self.knw_list_fingerp[ii]+" : " + self.knw_list[ii].description, self.knw_list[ii].knw_utility, self.knw_list[ii].eval_cnt, self.knw_list[ii].logith)

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
                for knwcnt in ltknwcnt:
                    if knwcnt>0:
                        rescnt=self.knw_list[knwcnt].eval_rate(y,fcnt=0)
                        ltknwcnt[knwcnt]=ltknwcnt[knwcnt]+rescnt
                rescnt=knwobj.eval_rate(y,fcnt=1)
                ltknwcnt[knwcnt] = ltknwcnt[knwcnt] + rescnt
        self.print()

    def eval_perplexity(self):
        """
        Knowledge evaluation on dataset, perplexity
        :param dataset:
        :return:
        """
        print("Knowledge evaluating ...")
        dataset = self.dataset
        if self.knw_actmat is None:
            self.forward_init()
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
        for nn in range(len(databpt)-1):
            x = dataset[nn]
            y = dataset[nn+1]
            prd,hidden = self.forward(x,hidden)
            yvec=np.zeros(self.lsize)
            yvec[y]=1
            perp=cal_kldiv(yvec,prd)
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
        print("Knowledge para pytorch optimizing ...")
        dataset = self.dataset
        plogits = np.log(self.prior)
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

            rstartv = np.floor(np.random.rand(batch) * (len(self.nlp.sub_corpus) - window - 1))
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

            if int(iis / 100) != his:
                print("Perlexity: ",iis, np.exp(loss.item()))
                his=int(iis / 100)

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
        prtstep = int(step / 10)
        startt = time.time()
        lsize = self.lsize
        dataset = self.dataset
        databp=[]
        for data in dataset:
            datavec=np.zeros(lsize)
            datavec[data]=1
            databp.append(datavec)
        databp=np.array(databp)

        # mode: IITNN, IITNI, SETRESET, SETRESETN
        if mode in ["IITNN", "IITNI", "SETRESET", "SETRESETN"]:
            if mode=="IITNN":
                knw_obj=KNW_IITNN(lsize)
            elif mode=="IITNI":
                knw_obj=KNW_IITNI(lsize)
            elif mode=="SETRESET":
                knw_obj=KNW_SETRESET(lsize)
            else: # mode=="SETRESETN":
                knw_obj=KNW_SETRESET(lsize)
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
            l1_reg = model.Wiz.weight.norm(1) + model.Win.weight.norm(1) + model.Whz.weight.norm(
                1) + model.Whn.weight.norm(1)
            return loss1 + 0.001 * l1_reg * cstep / step + 0.005 * lossh2o * cstep / step  # +0.3*lossz+0.3*lossn #

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        train_hist = []
        his = 0

        for iis in range(step):

            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

            hidden = rnn.initHidden(batch)

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


