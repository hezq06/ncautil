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
from ncautil.ncalearn import *


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

class LSTM_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru=torch.nn.LSTM(input_size,hidden_size,num_layers=num_layers)
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
        return output,hn,hout.permute(1,0,2)

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)]

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

        # self.h2m = torch.nn.Linear(hidden_size, 150)
        # self.m2o = torch.nn.Linear(150, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")

        self.hout = None

        # dummy base
        # self.dummy = torch.nn.Parameter(torch.rand(1,output_size), requires_grad=True)
        # self.ones=torch.ones(50,30,1).to(self.device)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input,hidden1)
        # hout = self.cdrop(hout) # dropout layer
        output = self.h2o(hout)
        # outm=self.h2m(hout)
        # outm = self.cdrop(outm)
        # output = self.m2o(outm)

        self.hout=hout

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn
        # output=torch.matmul(self.ones,self.dummy)
        # output=self.softmax(output)
        # return output,hn

    def forward_concept_ext(self, input, hidden1, npM_ext,add_logit=None, logit_mode=False, schedule=None):
        """
        Forward function for concept extend to full output with possbility extension matrix M_ext
        (P_N= M_ext.dot(Pc)
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input,hidden1)
        # hout = self.cdrop(hout) # dropout layer
        output = self.h2o(hout)
        # outm=self.h2m(hout)
        # outm = self.cdrop(outm)
        # output = self.m2o(outm)

        self.hout=hout

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)

        p_output = torch.exp(output) / torch.sum(torch.exp(output),-1,keepdim=True)
        M_ext=torch.from_numpy(np.array(npM_ext))
        M_ext=M_ext.type(torch.FloatTensor)
        if torch.cuda.is_available():
            M_ext.to(self.device)
        N_p_output=torch.matmul(p_output.to(self.device), M_ext.to(self.device))
        output_ext=torch.log(N_p_output)
        return output_ext,hn
        # output=torch.matmul(self.ones,self.dummy)
        # output=self.softmax(output)
        # return output,hn


    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

    def initHidden_eval(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

class GRU_NLP_WTA(torch.nn.Module):
    """
    PyTorch GRU for NLP, with winner takes all output layer to form concept cluster
    """
    def __init__(self, input_size, hidden_size, concept_size, output_size, num_layers=1, block_mode=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.concept_size = concept_size
        self.input_size = input_size
        self.output_size=output_size
        self.num_layers = num_layers

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
        self.h2c = torch.nn.Linear(hidden_size, concept_size)
        self.c2o = torch.nn.Linear(concept_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        # mid result storing
        self.concept_layer = None
        self.hout2con_masked = None

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")
        self.gpuavail=gpuavail

        self.block_mode=block_mode

    # def forward(self, input, hidden1, add_logit=None, logit_mode=False,wta_noise=0.0, schedule=1.0):
    #     """
    #     Forward, GRU WTA path scheduling
    #     :param input:
    #     :param hidden:
    #     :return:
    #     """
    #     hout, hn = self.gru(input,hidden1)
    #     # hout = self.cdrop(hout) # dropout layer
    #     hout2con=self.h2c(hout)
    #     argmax = torch.argmax(hout2con, dim=-1, keepdim=True)
    #     self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
    #     self.concept_layer.scatter_(-1, argmax, 1.0)
    #     self.concept_layer=self.concept_layer+wta_noise*torch.rand(self.concept_layer.shape).to(self.device)
    #     if self.block_mode:
    #         hout2con_masked = self.concept_layer
    #     else:
    #         hout2con_masked=hout2con*self.concept_layer
    #         hout2con_masked=hout2con_masked/torch.norm(hout2con_masked,2,-1,keepdim=True)
    #         self.hout2con_masked=hout2con_masked
    #
    #     hout2con2 = hout2con / torch.norm(hout2con, 2, -1, keepdim=True)
    #
    #     output = self.c2o(schedule * hout2con_masked + (1 - schedule) * hout2con2)
    #
    #     if add_logit is not None:
    #         output=output+add_logit
    #     if not logit_mode:
    #         output=self.softmax(output)
    #     return output,hn

    def forward(self, input, hidden1, add_logit=None, logit_mode=False,wta_noise=0.0, schedule=1.0):
        """
        Forward, GRU WTA winner percentage scheduling
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input,hidden1)
        # hout = self.cdrop(hout) # dropout layer
        hout2con=self.h2c(hout)

        upper_t=0.3
        Nind=int((1.0-schedule)*(self.concept_size-2)*upper_t)+1 # Number of Nind largest number kept
        if self.gpuavail:
            nphout2con=hout2con.cpu().data.numpy()
        else:
            nphout2con = hout2con.data.numpy()
        argmax=np.argsort(-nphout2con,axis=-1)[:,:,0:Nind]
        argmax=torch.from_numpy(argmax).to(self.device)
        self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
        self.concept_layer.scatter_(-1, argmax, 1.0)
        self.concept_layer=self.concept_layer+wta_noise*torch.rand(self.concept_layer.shape).to(self.device)

        hout2con_masked=hout2con*self.concept_layer
        hout2con_masked=hout2con_masked/torch.norm(hout2con_masked,2,-1,keepdim=True)

        output = self.c2o(hout2con_masked)
        self.hout2con_masked = hout2con_masked

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


class GRU_NLP_WTA_2(torch.nn.Module):
    """
    PyTorch GRU for NLP, with winner takes all output layer to form concept cluster, both input and output WTA bottleneck added
    """

    def __init__(self, input_size, hidden_size, concept_size, output_size, num_layers=1, block_mode=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.concept_size = concept_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.i2g = torch.nn.Linear(input_size, concept_size)
        self.gru = torch.nn.GRU(concept_size, hidden_size, num_layers=num_layers)
        self.h2c = torch.nn.Linear(hidden_size, concept_size)
        self.c2o = torch.nn.Linear(concept_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        # mid result storing
        self.concept_layer_i = None
        self.concept_layer_o = None
        self.hout2con_masked = None

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")
        self.gpuavail = gpuavail

        self.block_mode = block_mode

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, wta_noise=0.0, schedule=1.0):
        """
        Forward, GRU WTA winner percentage scheduling
        :param input:
        :param hidden:
        :return:
        """
        upper_t = 0.3
        Nind = int((1.0 - schedule) * (self.concept_size - 2) * upper_t) + 1  # Number of Nind largest number kept

        # First WTA scheduling of input
        ginput = self.i2g(input)
        if self.gpuavail:
            npginput = ginput.cpu().data.numpy()
        else:
            npginput = ginput.data.numpy()
        argmax_i = np.argsort(-npginput, axis=-1)[:, :, 0:Nind]
        argmax_i = torch.from_numpy(argmax_i).to(self.device)
        self.concept_layer_i = torch.zeros(ginput.shape).to(self.device)
        self.concept_layer_i.scatter_(-1, argmax_i, 1.0)
        self.concept_layer_i = self.concept_layer_i + wta_noise * torch.rand(self.concept_layer_i.shape).to(self.device)

        ginput_masked = ginput * self.concept_layer_i
        ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)

        # GRU part

        hout, hn = self.gru(ginput_masked, hidden1)
        # hout = self.cdrop(hout) # dropout layer
        hout2con = self.h2c(hout)

        # # Second WTA scheduling of output
        # if self.gpuavail:
        #     nphout2con = hout2con.cpu().data.numpy()
        # else:
        #     nphout2con = hout2con.data.numpy()
        # argmax_o = np.argsort(-nphout2con, axis=-1)[:, :, 0:Nind]
        # argmax_o = torch.from_numpy(argmax_o).to(self.device)
        # self.concept_layer_o = torch.zeros(hout2con.shape).to(self.device)
        # self.concept_layer_o.scatter_(-1, argmax_o, 1.0)
        # self.concept_layer_o = self.concept_layer_o + wta_noise * torch.rand(self.concept_layer_o.shape).to(self.device)
        #
        # hout2con_masked = hout2con * self.concept_layer_o
        # hout2con_masked = hout2con_masked / torch.norm(hout2con_masked, 2, -1, keepdim=True)

        # output = self.c2o(hout2con_masked)
        # self.hout2con_masked = hout2con_masked

        output = self.c2o(hout2con)

        if add_logit is not None:
            output = output + add_logit
        if not logit_mode:
            output = self.softmax(output)
        return output, hn

    def initHidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

    def initHidden_eval(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def build_concept_vec(self,id_2_vec,prior):
        # Build concept mapping
        hidden = self.initHidden_cuda(self.device, 1)
        inputl = []
        dim = self.input_size
        for ii in range(self.output_size):
            vec = id_2_vec[ii]
            inputl.append(vec)
        inputx = torch.from_numpy(np.array(inputl)).view(-1, 1, dim)
        inputx = inputx.type(torch.FloatTensor)
        _ = self.forward(inputx.to(self.device), hidden)
        print(self.concept_layer_i.cpu().shape)
        np_con = self.concept_layer_i.cpu().data.numpy()
        id_2_con = []
        for ii in range(self.output_size):
            id_2_con.append(np.argmax(np_con[ii, 0, :]))
        # sort concept id according to its importance
        importance_dict=dict([])
        for conid in set(id_2_con):
            importance_dict[conid]=0.0
        for wrdid in range(len(id_2_con)):
            importance_dict[id_2_con[wrdid]]=importance_dict[id_2_con[wrdid]]+prior[wrdid]
        sorted_imp_l=sorted(importance_dict.items(), key=lambda x: (-x[1], x[0]))
        # swap concept id according to sorted important list
        swapdict=dict([])
        for iiord in range(len(sorted_imp_l)):
            swapdict[sorted_imp_l[iiord][0]]=iiord
        id_2_con_s=[]
        print(sorted_imp_l,swapdict)
        for iicon in id_2_con:
            id_2_con_s.append(swapdict[iicon])
        return id_2_con_s

class FF_NLP_WTA(torch.nn.Module):
    """
        PyTorch GRU for NLP, with winner takes all output layer to form concept cluster
        """

    def __init__(self, input_size, hidden_size, concept_size, output_size, block_mode=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.concept_size = concept_size
        self.input_size = input_size

        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.h2c = torch.nn.Linear(hidden_size, concept_size)
        self.c2o = torch.nn.Linear(concept_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        # mid result storing
        self.concept_layer = None
        self.hout2con_masked = None

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")

        self.block_mode = block_mode

    # def forward(self, input, hid=None, wta_noise=0.0):
    #     """
    #     Forward
    #     :param input:
    #     :param hid: no use
    #     :return:
    #     """
    #     hidden1 = self.i2h(input)
    #     hidden1 = self.relu(hidden1)
    #     hout2con = self.h2c(hidden1)
    #     argmax = torch.argmax(hout2con, dim=-1, keepdim=True)
    #     self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
    #     self.concept_layer.scatter_(-1, argmax, 1.0)
    #     self.concept_layer = self.concept_layer + wta_noise * torch.rand(self.concept_layer.shape).to(self.device)
    #     if self.block_mode:
    #         output = self.c2o(self.concept_layer)
    #     else:
    #         hout2con_masked = hout2con * self.concept_layer
    #         hout2con_masked = hout2con_masked / torch.norm(hout2con_masked, 2, -1, keepdim=True)
    #         self.hout2con_masked = hout2con_masked
    #         output = self.c2o(hout2con_masked)
    #
    #     output=self.softmax(output)
    #
    #     return output, hid

    def forward(self, input, hid=None, wta_noise=0.0, schedule=1.0):
        """
        Forward, path transfer scheduling
        :param input:
        :param hid: no use
        :return:
        """
        hidden1 = self.i2h(input)
        hidden1 = self.relu(hidden1)
        hout2con = self.h2c(hidden1)
        hout2con=self.relu(hout2con)

        argmax = torch.argmax(hout2con, dim=-1, keepdim=True)
        self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
        self.concept_layer.scatter_(-1, argmax, 1.0)
        self.concept_layer = self.concept_layer + wta_noise * torch.rand(self.concept_layer.shape).to(self.device)
        if self.block_mode:
            hout2con_masked=self.concept_layer
        else:
            hout2con_masked = hout2con * self.concept_layer
            hout2con_masked = hout2con_masked / torch.norm(hout2con_masked, 2, -1, keepdim=True)
        self.hout2con_masked = hout2con_masked

        hout2con2=hout2con/torch.norm(hout2con,2,-1,keepdim=True)

        output = self.c2o(schedule*hout2con_masked+(1-schedule)*hout2con2)
        output=self.softmax(output)

        return output, hid

    # def forward(self, input, hid=None, wta_noise=0.0):
    #     """
    #     Forward
    #     :param input:
    #     :param hid: no use
    #     :return:
    #     """
    #     hidden1 = self.i2h(input)
    #     hidden1 = self.relu(hidden1)
    #     hout2con = self.h2c(hidden1)
    #     hout2con=self.relu(hout2con)
    #     output=self.c2o(hout2con)
    #     output=self.softmax(output)
    #
    #     return output, hid

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None

    def initHidden_eval(self):
        return None


class GRU_NLP_CLU(torch.nn.Module):
    """
    PyTorch GRU for NLP, adding a SoftMax bottle neck for clustering
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(self.__class__, self).__init__()

        clusterNum=30

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
        self.h2c = torch.nn.Linear(hidden_size, clusterNum)
        self.c2o = torch.nn.Linear(clusterNum, output_size)

        # self.h2m = torch.nn.Linear(hidden_size, 150)
        # self.m2o = torch.nn.Linear(150, output_size)

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
        clusterl = self.h2c(hout)
        # clusterl = clusterl/torch.sum(clusterl, dim=-1,keepdim=True)
        clusterl=self.softmax(clusterl)
        output = self.c2o(clusterl)
        # outm=self.h2m(hout)
        # outm = self.cdrop(outm)
        # output = self.m2o(outm)

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn,hout.permute(1,0,2),clusterl

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

    def initHidden_eval(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

class GRU_TWO(torch.nn.Module):
    """
    PyTorch GRU for NLP, two GRU working together
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(self.__class__, self).__init__()

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

class GRU_KNW_CRYSTAL(torch.nn.Module):
    """
    Cystalized GRU for knowledge distilling
    """
    def __init__(self, input_size, hidden_size, output_size, prior_vec=None):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param prior_vec: logit of prior knowledge
        """
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # self.Wiz = torch.nn.Linear(input_size, hidden_size)
        # self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        # self.Win = torch.nn.Linear(input_size, hidden_size)
        # self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.in2ga = torch.nn.Linear(input_size, hidden_size)
        self.in2gb = torch.nn.Linear(input_size, hidden_size)
        self.in2gc = torch.nn.Linear(input_size, hidden_size)

        self.h2o = torch.nn.Linear(hidden_size, output_size,bias=True)

        if type(prior_vec)!=type(None):
            self.prior_vec=torch.from_numpy(np.array(prior_vec))
            self.prior_vec = self.prior_vec.type(torch.FloatTensor)
        else:
            self.prior_vec = torch.zeros(output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.gate_layer=None

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")
        self.gpuavail = gpuavail

    def forward(self, input, hidden, schedule=None):
        """
        Forward, GRU WTA path scheduling
        :param input:
        :param hidden:
        :return:
        """
        # hout, hn = self.gru(input,hidden1)
        # # hout = self.cdrop(hout) # dropout layer
        # hout2con=self.h2c(hout)
        # argmax = torch.argmax(hout2con, dim=-1, keepdim=True)
        # self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
        # self.concept_layer.scatter_(-1, argmax, 1.0)
        # self.concept_layer=self.concept_layer+wta_noise*torch.rand(self.concept_layer.shape).to(self.device)
        # if self.block_mode:
        #     hout2con_masked = self.concept_layer
        # else:
        #     hout2con_masked=hout2con*self.concept_layer
        #     hout2con_masked=hout2con_masked/torch.norm(hout2con_masked,2,-1,keepdim=True)
        #     self.hout2con_masked=hout2con_masked
        #
        # hout2con2 = hout2con / torch.norm(hout2con, 2, -1, keepdim=True)
        #
        # output = self.c2o(schedule * hout2con_masked + (1 - schedule) * hout2con2)
        #
        # if add_logit is not None:
        #     output=output+add_logit
        # if not logit_mode:
        #     output=self.softmax(output)
        # return output,hn

        ## WTA for gates
        gate_a = self.in2ga(input)
        gate_b = self.in2gb(input)
        gate_c = self.in2gc(input)
        shape=gate_a.shape
        gate = torch.cat((gate_a.view(shape[0],shape[1],shape[2],1),gate_b.view(shape[0],shape[1],shape[2],1),
                          gate_c.view(shape[0],shape[1],shape[2],1)),dim=-1)
        argmax = torch.argmax(gate, dim=-1, keepdim=True)
        self.gate_layer = torch.zeros(gate.shape).to(self.device)
        self.gate_layer.scatter_(-1, argmax, 1.0)
        gate_masked = gate * self.gate_layer
        gate_masked=gate_masked/torch.norm(gate_masked,2,-1,keepdim=True)
        gate_masked_c=gate_masked[:,:,:,2].view(shape[0],shape[1],shape[2])
        gate_masked_a = gate_masked[:, :, :, 0].view(shape[0], shape[1], shape[2])
        ht = hidden * gate_masked_c + (1-gate_masked_c)*gate_masked_a

        ## Scheduling WTA for hidden layer
        # upper_t = 0.3
        # Nind = int((1.0 - schedule) * (self.hidden_size - 2) * upper_t) + 1  # Number of Nind largest number kept
        #
        # # First WTA scheduling of input
        # argmax_i = np.argsort(-ht, axis=-1)[:, :, 0:Nind]
        # argmax_i = torch.from_numpy(argmax_i).to(self.device)
        # self.concept_layer_i = torch.zeros(ginput.shape).to(self.device)
        # self.concept_layer_i.scatter_(-1, argmax_i, 1.0)
        #
        # ginput_masked = ginput * self.concept_layer_i
        # ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)


        output = self.h2o(ht)
        output = self.softmax(output)

        return output, ht

    def forward_concept_ext(self, input, hidden, npM_ext, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward function for concept extend to full output with possbility extension matrix M_ext
        (P_N= M_ext.dot(Pc)
        :param input:
        :param hidden:
        :return:
        """

        output, ht = self.forward(input,hidden)

        p_output = torch.exp(output) / torch.sum(torch.exp(output),-1,keepdim=True)
        M_ext=torch.from_numpy(np.array(npM_ext))
        M_ext=M_ext.type(torch.FloatTensor)
        if torch.cuda.is_available():
            M_ext.to(self.device)
        N_p_output=torch.matmul(p_output.to(self.device), M_ext.to(self.device))
        output_ext=torch.log(N_p_output)
        return output_ext,ht

    # def forward(self, input, hidden, batch=None, add_prior=None):
    #     """
    #
    #     :param input: input
    #     :param hidden: hidden
    #     :param result:
    #     :return:
    #     """
    #     zt=self.sigmoid(self.Wiz(input)+self.Whz(hidden))
    #     nt = self.sigmoid(self.Win(input) +  self.Whn(hidden))
    #     ht = (1 - zt) * nt + zt * hidden
    #
    #     if add_prior is None:
    #         output = self.h2o(ht)+self.prior_vec
    #     else:
    #         output = self.h2o(ht) + self.prior_vec + add_prior
    #     output = self.softmax(output)
    #     return output, ht

    def knw_distill(self):
        """
        Distill knwledge from trained network
        KNW style KNW{"setter":[0,1,2],"resetter":[3,4,5],"knw_vec":vec}
        :return:
        """
        knw_list_res=[]
        hidden = self.initHidden_cuda(self.device)
        for knw_id in range(self.hidden_size):
            knw_dict=dict([])
            knw_dict["setter"]=[]
            knw_dict["resetter"] = []
            knw_dict["knw_vec"] = self.h2o.weight.cpu().data.numpy()[:,knw_id]
            knw_dict["knw_entropy"] = cal_entropy(knw_dict["knw_vec"],logit=True)
            for con_id in range(self.input_size):
                test_in = torch.zeros([1, 1, self.input_size]).to(self.device) # !!!!!!! test_in.to(self.device) takes no effect
                test_in[0,0,con_id]=1
                output, ht = self.forward(test_in,hidden)
                if self.gate_layer[0,0,knw_id,:][0]==1: # setter
                    knw_dict["setter"].append(con_id)
                elif self.gate_layer[0,0,knw_id,:][1]==1: # retter
                    knw_dict["resetter"].append(con_id)
            knw_list_res.append(knw_dict)
        return knw_list_res

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

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