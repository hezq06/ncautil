"""
Trial of Sequent multipole expansion.
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


def one_hot(num, lsize):
    if type(num) == type(1) or type(num) == np.int32:
        res = np.zeros(lsize)
        res[num] = 1
    else:
        res = []
        for nn in num:
            ytemp = np.zeros(lsize)
            ytemp[nn] = 1
            res.append(ytemp)
    return np.array(res)

def sample_onehot(prob):
    """
    Sample one hot vector
    :param prob: normal probability vec
    :return:
    """
    prob=np.array(prob)
    assert len(prob.shape)==1
    prob=prob/np.sum(prob)
    rndp = np.random.rand()
    dig=None
    for iin in range(len(prob)):
        rndp = rndp - prob[iin]
        if rndp < 0:
            dig = iin
            break
    xn=np.zeros(len(prob))
    xn[dig]=1
    xin = torch.from_numpy(xn)
    xin = xin.type(torch.FloatTensor)
    return xin,dig

def free_gen(step,model,lsize):
    """
    Free generation of one-hot style sequence given model
    :param model:
    :return:
    """

    def logp(vec):
        """
        Transfer LogSoftmax function to normal prob
        :param vec:
        :return:
        """
        vec = np.exp(vec)
        dwn = np.sum(vec)
        return vec / dwn

    res=[]
    hidden=model.initHidden(1)
    vec1=np.zeros(lsize)
    dig=int(np.random.rand()*lsize)
    vec1[dig]=1
    res.append(dig)
    x = torch.from_numpy(vec1)
    x = x.type(torch.FloatTensor)
    for ii in range(step):
        output, hidden = model(x.view(1,1,-1), hidden)
        output=logp(output.view(-1).data.numpy())
        x,dig=sample_onehot(output)
        res.append(dig)
    return res

def logit_gen(seq, model,lsize):
    """
    Gen logit sequnce based on a one-hot seq
    :param model:
    :return:
    """
    def logp(vec):
        """
        Transfer LogSoftmax function to normal prob
        :param vec:
        :return:
        """
        vec = np.exp(vec)
        dwn = np.sum(vec)
        return vec / dwn
    res=[]
    hidden = model.initHidden(1)
    for ii in range(len(seq)):
        vec1 = np.zeros(lsize)
        vec1[seq[ii]] = 1
        x = torch.from_numpy(vec1)
        x = x.type(torch.FloatTensor)
        output, hidden = model(x.view(1, 1, -1), hidden)
        res.append(logp(output.view(-1).data.numpy()))
    return np.array(res)

class Seq_Multip(object):
    """
    Sequent multipole expansion trial
    """
    def __init__(self, dataset, lsize):
        self.dataset = dataset
        self.lsize = lsize

        self.model=None
        self.mlost = 1.0e9

    def do_eval(self,coop=None):
        """
        Do evaluation
        :return:
        """
        print("Start Evaluation ...")
        rnn=self.model
        dataset = self.dataset
        lsize = self.lsize
        datab = []
        for data in dataset:
            datavec = np.zeros(self.lsize)
            datavec[data] = 1
            datab.append(datavec)
        datab = np.array(datab)
        databpt = torch.from_numpy(datab)
        databpt = databpt.type(torch.FloatTensor)
        outputl = []
        hiddenl = []
        hidden=rnn.initHidden(1)
        for iis in range(len(databpt) - 1):
            x = databpt[iis, :].view(1, 1, lsize)
            if coop is not None:
                outputc, hiddenc = coop(x, hidden, logitmode=True)
                output, hidden = rnn(x, hidden, add_logit=outputc)
            else:
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


    def run_training(self, step, learning_rate=1e-2, batch=20, window=30, save=None,seqtrain=False, coop=None):
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
        datab=[]
        for data in dataset:
            datavec=np.zeros(lsize)
            datavec[data]=1
            datab.append(datavec)
        databp=torch.from_numpy(np.array(datab))

        if self.model is None:
            rnn=GRU_NLP(self.lsize,30,self.lsize,num_layers=1)
            # rnn =PAIR_NET(self.lsize)
            # rnn = KNW_CELL(self.lsize,mode="SETRESET")
        else:
            rnn = self.model
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

        for iis in range(step):

            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

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
                    if coop is not None:
                        outputc, hiddenc = coop(x, hidden=None,logitmode=True)
                        output, hidden = rnn(x, hidden, add_logit=outputc)
                    else:
                        output, hidden = rnn(x, hidden, add_logit=None)
                    if type(outputl) == type(None):
                        outputl = output.view(batch, lsize, 1)
                    else:
                        outputl = torch.cat((outputl.view(batch, lsize, -1), output.view(batch, lsize, 1)), dim=2)
                loss = custom_KNWLoss(outputl, outlab, rnn, iis)
            # else:
            #     # LSTM/GRU provided whole sequence training
            #     vec1m = None
            #     vec2m = None
            #     outputl = None
            #     for iib in range(batch):
            #         vec1 = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
            #         vec2 = databp[int(rstartv[iib])+1:int(rstartv[iib])+window+1, :]
            #         if type(vec1m) == type(None):
            #             vec1m = vec1.view(window, 1, -1)
            #             vec2m = vec2.view(window, 1, -1)
            #         else:
            #             vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
            #             vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
            #     x = Variable(vec1m.reshape(window, batch, lsize).contiguous(), requires_grad=True)  #
            #     y = Variable(vec2m.reshape(window, batch, lsize).contiguous(), requires_grad=True)
            #     x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            #     output, hidden = rnn(x, hidden, batch=batch)
            #     loss = custom_KNWLoss(output.permute(1,2,0), outlab, rnn, iis)

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

class PAIR_NET(torch.nn.Module):
    """
    Pair wise network
    """
    def __init__(self,lsize):
        super(PAIR_NET, self).__init__()

        self.lsize=lsize
        self.i2o = torch.nn.Linear(lsize, lsize)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden=None, logitmode=False, add_logit=None):
        output = self.i2o(input)
        if add_logit is not None:
            output=output+add_logit
        if not logitmode:
            output = self.softmax(output)
        return output,None

    def initHidden(self,batch):
        return None

class GRU_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

    def initHidden_eval(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

class GRU_TWO(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU_TWO, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru1=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
        self.gru2 = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=1):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout1, hn1 = self.gru1(input.view(-1, batch, self.input_size),hidden[0])
        hout2, hn2 = self.gru2(input.view(-1, batch, self.input_size), hidden[1])
        hout=hout1+hout2
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        return output,[hn1,hn2]

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

class KNW_CELL(torch.nn.Module):
    """
    PyTorch knowledge cell for single piece of knowledge
    """
    def __init__(self, lsize, mode):
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

    def forward(self, input, hidden, add_logit=None):
        """
        Forward
        :param input:
        :param hidden:
        :param plogits: prior
        :return:
        """
        ht=self.sigmoid(self.Ws(input))+hidden
        output = self.ctrl[0] * self.h2o(ht)
        if add_logit is not None:
            output=output+add_logit
        output = self.softmax(output)
        resetter = 1 - self.sigmoid(self.Wr(input))
        hidden=resetter*ht*self.ctrl[1]
        return output, hidden

    def initHidden(self,batch=1):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True)