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
from ncautil.ptfunction import *

# from torchnlp.nn import WeightDrop
from awd_lstm_lm.weight_drop import WeightDrop


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

def logp(vec):
    """
    Transfer LogSoftmax function to normal prob
    :param vec:
    :return:
    """
    vec = np.exp(vec)
    dwn = np.sum(vec)
    return vec / dwn

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

class Base_NLP(torch.nn.Module):
    """
    PyTorch Baseline with prior only
    """
    def __init__(self, output_size):
        super(self.__class__, self).__init__()

        self.output_size = output_size
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")

        self.prior = torch.nn.Parameter(torch.rand(1,output_size), requires_grad=True)

    def forward(self, input, hidden1=None, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        output=input*0+self.prior
        output=self.softmax(output)
        return output,None

    def initHidden(self,batch):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True).to(device)

class FF_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if gpuavail else "cpu")

        self.hout = None

        # dummy base
        # self.dummy = torch.nn.Parameter(torch.rand(1,output_size), requires_grad=True)
        # self.ones=torch.ones(50,30,1).to(self.device)

    def forward(self, input, hidden1=None, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hidden = self.i2h(input)
        hidden = self.tanh(hidden)
        output = self.h2o(hidden)

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,None

    def initHidden(self,batch):
        return Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device)

class RNN_NLP(torch.nn.Module):
    """
    PyTorch LSTM for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, cuda_flag=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers)

        # if weight_dropout>0:
        #     print("Be careful, only GPU works for now.")
        #     self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
        #     self.lstm = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.rnn(input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

class LSTM_NLP(torch.nn.Module):
    """
    PyTorch LSTM for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, cuda_flag=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.lstm(input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, cuda_flag=True, weight_dropout=0.0, gru_dropout=0.0):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers,dropout=gru_dropout)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.weight_dropout=weight_dropout
        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            # self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=weight_dropout)

        # self.h2m = torch.nn.Linear(hidden_size, 150)
        # self.m2o = torch.nn.Linear(150, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        if cuda_flag:
            gpuavail = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if gpuavail else "cpu")
        else:
            gpuavail = False
            self.device = torch.device("cpu")

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
        if len(input.shape)==2:
            input=input.view(1,input.shape[0],input.shape[1])
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

    def plot_layer_all(self):
        mat = self.h2o.cpu().weight.data.numpy()
        srow=2
        scol=7
        plt.subplot(srow,scol,1)
        plot_mat(mat, title="h2o", symmetric=True, tick_step=1, show=False)
        mat = self.h2o.cpu().bias.data.numpy()
        plt.subplot(srow,scol,1+scol)
        plot_mat(mat.reshape(1,-1), title="h2o_bias", symmetric=True, tick_step=1, show=False)
        subplotidx = 1
        if self.weight_dropout>0:
            weight_ih=self.gru.module.weight_ih_l0.cpu().data.numpy()
            weight_hh = self.gru.module.weight_hh_l0.cpu().data.numpy()
            bias_ih = self.gru.module.bias_ih_l0.cpu().data.numpy()
            bias_hh = self.gru.module.bias_hh_l0.cpu().data.numpy()
        else:
            weight_ih = self.gru.weight_ih_l0.cpu().data.numpy()
            weight_hh = self.gru.weight_hh_l0.cpu().data.numpy()
            bias_ih = self.gru.module.bias_ih_l0.cpu().data.numpy()
            bias_hh = self.gru.module.bias_hh_l0.cpu().data.numpy()
        allnameih = ["Wir", "Wiz", "Win"]
        for ii,nameitem in enumerate(allnameih):
            subplotidx = subplotidx + 1
            mat=weight_ih[ii*self.hidden_size:(ii+1)*self.hidden_size,:]
            plt.subplot(srow,scol,subplotidx)
            plot_mat(mat, title=nameitem, symmetric=True, tick_step=1, show=False)
            mat = bias_ih[ii * self.hidden_size:(ii + 1) * self.hidden_size]
            plt.subplot(srow,scol,subplotidx+scol)
            plot_mat(mat.reshape(1,-1), title=nameitem+"_bias", symmetric=True, tick_step=1, show=False)
        allnamehh = ["Whr", "Whz", "Whn"]
        for ii, nameitem in enumerate(allnamehh):
            subplotidx = subplotidx + 1
            mat = weight_hh[ii * self.hidden_size:(ii + 1) * self.hidden_size, :]
            plt.subplot(srow,scol,subplotidx)
            plot_mat(mat, title=nameitem, symmetric=True, tick_step=1, show=False)
            mat = bias_hh[ii * self.hidden_size:(ii + 1) * self.hidden_size]
            plt.subplot(srow,scol,subplotidx+scol)
            plot_mat(mat.reshape(1,-1), title=nameitem + "_bias", symmetric=True, tick_step=1, show=False)
        plt.show()


    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

    def initHidden_eval(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

class GRU_INPSEL_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP with selective input
    Using idea of Batchnorm to decide dynamic range of dropping
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, id_2_vec=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers)

        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(1, affine=False)

        self.ci2ch=torch.nn.Linear(input_size+hidden_size, 10)
        self.ch2co=torch.nn.Linear(10, 1)

        self.id_2_vec=id_2_vec

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpuavail else "cpu")

    def forward(self, inputvec, hidden1, add_logit=None, logit_mode=False, schedule=None, input_dropout=-10.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        assert inputvec.shape[0] == 1
        predvec=hidden1[1]
        predinputvec=torch.zeros(inputvec.shape)
        predvec_npcpu = predvec.cpu().data.numpy()
        for ii_batch in range(inputvec.shape[1]):
            prob = logp(predvec_npcpu[0,ii_batch,:])
            xin, dig = sample_onehot(prob)
            predinputvec[0,ii_batch,:]=torch.from_numpy(self.id_2_vec[dig])
        if self.gpuavail:
            predinputvec=predinputvec.to(self.device)

        cinvec=torch.cat((inputvec,hidden1),dim=-1)
        chd=self.ci2ch(cinvec)
        chd=self.tanh(chd)
        coutput=self.sigmoid(chd)
        coutput = self.bn(coutput)
        selgate = torch.zeros(inputvec.shape)
        for ii_batch in range(coutput.shape[1]):
            if coutput[0,ii_batch,0]>input_dropout: # keep
                selgate[0,ii_batch,:]=1

        combinputvec=selgate*inputvec+(1-selgate)*predinputvec

        hout, hn = self.gru(combinputvec,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,[hn,output]

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                torch.zeros(self.num_layers, batch, self.output_size)]


    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),
                torch.zeros(self.num_layers, batch, self.output_size).to(device)]


class GRU_TwoLayerCon_DataEnhanceVer(torch.nn.Module):
    """
    A trial of two layer training stracture for trial of layered inductive bias of Natural Language.
    Layer 1 is a pre-trained layer like GRU over POS which freezes.
    Layer 2 is a projecction perpendicular to layer 1
    Attention Gating is used to choose plitable information to two layers
    Data enhancement version is used.
    """
    def __init__(self, rnn, input_size, hidden_size, output_size, num_layers=1):
        """
        init
        :param gru_l1: gru_l1 [GRU, input_size, output_size]
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param num_layers:
        """
        super(self.__class__, self).__init__()
        self.rnn = rnn
        for param in self.rnn.parameters():
            param.requires_grad = False
        self.gru_input_size = self.rnn.input_size
        self.gru_output_size = self.rnn.output_size

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.i2s1 = torch.nn.Linear(input_size, hidden_size) # input to sigmoid
        self.i2s2 = torch.nn.Linear(hidden_size, hidden_size)  # input to sigmoid
        self.i2s3 = torch.nn.Linear(hidden_size, self.gru_input_size)  # input to sigmoid

        self.g2o = torch.nn.Linear(self.gru_output_size, output_size) # GRU output to output

        self.sigmoid = torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.infer_pos = None

        self.pre_trained = False

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        if self.pre_trained:
            for param in self.i2s.parameters():
                param.requires_grad = False
        sig_gru1=self.i2s1(input)
        sig_gru1 = self.relu(sig_gru1)
        sig_gru2 = self.i2s2(sig_gru1)
        sig_gru2 = self.relu(sig_gru2)
        sig_gru3 = self.i2s3(sig_gru2)
        self.infer_pos = self.sigmoid(sig_gru3)
        # sig_gru=self.sigmoid(sig_gru)
        # input_gru=self.s2g(sig_gru)
        # self.infer_pos =torch.exp(self.softmax(sig_gru))
        # self.infer_pos = wta_layer(sig_gru,schedule=schedule)
        hout, hn = self.rnn(self.infer_pos,hidden1,logit_mode=True)
        # hout=logit_sampling_layer(hout)
        output = self.g2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def pre_training(self,input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input:
        :param hidden1:
        :param add_logit:
        :param logit_mode:
        :param schedule:
        :return:
        """
        self.pre_trained = True
        sig_gru = self.i2s(input)
        output = self.softmax(sig_gru)
        return output, None

    def initHidden(self,batch):
        return self.rnn.initHidden(batch)

    def initHidden_cuda(self,device, batch):
        return self.rnn.initHidden_cuda(device, batch)

class GRU_TwoLayerCon_TwoStepVersion(torch.nn.Module):
    """
    A trial of two layer training stracture for trial of layered inductive bias of Natural Language.
    Layer 1 is a pre-trained layer like GRU over POS which freezes.
    Layer 2 is a projecction perpendicular to layer 1
    Attention Gating is used to choose plitable information to two layers
    Two step: 1, Normal training, 2, auto-encode aligning.
    """
    def __init__(self, rnn, input_size, hidden_size, output_size, num_layers=1):
        """
        init
        :param gru_l1: gru_l1 [GRU, input_size, output_size]
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param num_layers:
        """
        super(self.__class__, self).__init__()
        self.rnn = rnn
        for param in self.rnn.parameters():
            param.requires_grad = False
        self.gru_input_size = self.rnn.input_size
        self.gru_output_size = self.rnn.output_size

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.i2s1 = torch.nn.Linear(input_size, hidden_size)  # input to sigmoid
        self.i2s2 = torch.nn.Linear(hidden_size, hidden_size)  # input to sigmoid
        self.i2s3 = torch.nn.Linear(hidden_size, self.gru_input_size)  # input to sigmoid

        self.g2o = torch.nn.Parameter(torch.rand(self.gru_output_size,output_size),requires_grad=True)  # Parameter Matrix of concept to output

        self.sigmoid = torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.nsoftmax=torch.nn.Softmax(dim=-1)

        self.infer_pos = None

        self.pre_trained = False

    def auto_encode(self,input,schedule=None):
        """
        Parameter sharing auto-encoder
        :param input:
        :return:
        """
        sig_gru1 = self.i2s1(input)
        sig_gru1 = self.relu(sig_gru1)
        sig_gru2 = self.i2s2(sig_gru1)
        sig_gru2 = self.relu(sig_gru2)
        sig_gru3 = self.i2s3(sig_gru2)
        infer_pos = wta_layer(sig_gru3, schedule=schedule)
        output = torch.log(torch.matmul(infer_pos,self.nsoftmax(self.g2o))+1e-9)
        if torch.isnan(output).any():
            save_data(infer_pos, file="data_output1.pickle")
            data2=self.nsoftmax(self.g2o)
            save_data(data2, file="data_output2.pickle")
            raise Exception("NaN Error")
        return output

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        # if self.pre_trained:
        #     for param in self.i2s.parameters():
        #         param.requires_grad = False
        sig_gru1 = self.i2s1(input)
        sig_gru1 = self.relu(sig_gru1)
        sig_gru2 = self.i2s2(sig_gru1)
        sig_gru2 = self.relu(sig_gru2)
        sig_gru3 = self.i2s3(sig_gru2)
        # self.infer_pos = self.sigmoid(sig_gru3)
        # sig_gru=self.sigmoid(sig_gru)
        # input_gru=self.s2g(sig_gru)
        # self.infer_pos =torch.exp(self.softmax(sig_gru))
        self.infer_pos = wta_layer(sig_gru3,schedule=schedule)
        hout, hn = self.rnn(self.infer_pos,hidden1,logit_mode=True)
        output = torch.log(torch.matmul(torch.exp(hout), self.nsoftmax(self.g2o)))

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def pre_training(self,input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input:
        :param hidden1:
        :param add_logit:
        :param logit_mode:
        :param schedule:
        :return:
        """
        self.pre_trained = True
        sig_gru1 = self.i2s1(input)
        sig_gru1 = self.relu(sig_gru1)
        sig_gru2 = self.i2s2(sig_gru1)
        sig_gru2 = self.relu(sig_gru2)
        sig_gru3 = self.i2s3(sig_gru2)
        output = self.softmax(sig_gru3)
        return output, None

    def initHidden(self,batch):
        return self.rnn.initHidden(batch)

    def initHidden_cuda(self,device, batch):
        return self.rnn.initHidden_cuda(device, batch)

class GRU_SerialCon_SharedAssociation(torch.nn.Module):
    """
    A trial of two step training stracture for trial of layered inductive bias of Natural Language.
    Serial knowledge reusing is assumed.
    """
    def __init__(self, rnn, input_size, hidden_size, num_layers=1):
        """
        init
        :param gru_l1: gru_l1 [GRU, input_size, output_size]
        :param input_size:
        :param hidden_size:
        :param output_size: equal input_size
        :param num_layers:
        """
        super(self.__class__, self).__init__()
        self.rnn = rnn
        for param in self.rnn.parameters():
            param.requires_grad = False
        self.rnn.eval()
        self.gru_input_size = self.rnn.input_size
        self.gru_output_size = self.rnn.output_size

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.icmat=torch.nn.Parameter(torch.rand(input_size,self.gru_input_size), requires_grad=True)
        # self.icmat_bias1 = torch.nn.Parameter(torch.rand(self.gru_input_size), requires_grad=True)
        # self.icmat_bias2 = torch.nn.Parameter(torch.rand(input_size), requires_grad=True)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.nsoftmax = torch.nn.Softmax(dim=-1)

        self.infer_pos = None

        self.pre_trained = False

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        sig_gru=torch.matmul(input,self.icmat)#+self.icmat_bias1
        if hidden1[1] is not None:
            self.infer_pos = wta_layer(sig_gru*self.nsoftmax(hidden1[1]),schedule=schedule) # Containing inner feedforward loop, not effective
        else:
            self.infer_pos = wta_layer(sig_gru, schedule=schedule)
        # self.infer_pos = logit_sampling_layer(sig_gru)
        hout, hn = self.rnn(self.infer_pos,hidden1[0],logit_mode=True)
        # hout=logit_sampling_layer(hout)
        output = torch.matmul(hout, torch.t(self.icmat))#+self.icmat_bias2

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,[hn,None]

    def pre_training(self,input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input:
        :param hidden1:
        :param add_logit:
        :param logit_mode:
        :param schedule:
        :return:
        """
        sig_gru = torch.matmul(input, self.icmat)#+self.icmat_bias1
        output = self.softmax(sig_gru)
        return output, None

    def initHidden(self,batch):
        return [self.rnn.initHidden(batch),None]

    def initHidden_cuda(self,device, batch):
        return [self.rnn.initHidden_cuda(device, batch),None]

class GRU_TwoLayerCon_SharedAssociation(torch.nn.Module):
    """
    A trial of two layer training stracture for trial of layered inductive bias of Natural Language.
    Layer 1 is a pre-trained layer like GRU over POS which freezes.
    Layer 2 is a projecction perpendicular to layer 1
    Attention Gating is used to choose plitable information to two layers
    Shared association scheme is used to ensure item-concept alignment
    """
    def __init__(self, se, rnn, input_size, hidden_size, num_layers=1):
        """

        :param se: a pre_trained GRU_SerialCon_SharedAssociation
        :param rnn:
        :param input_size:
        :param hidden_size:
        :param num_layers:
        """
        super(self.__class__, self).__init__()
        self.rnn0 = se.rnn
        self.rnn1 = rnn

        self.gru_input_size0 = self.rnn0.input_size
        self.gru_output_size0 = self.rnn0.output_size

        self.gru_input_size1 = self.rnn1.input_size
        self.gru_output_size1 = self.rnn1.output_size

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.icmat0=se.icmat
        # self.icmat_bias1 = torch.nn.Parameter(torch.rand(self.gru_input_size), requires_grad=True)
        # self.icmat_bias2 = torch.nn.Parameter(torch.rand(input_size), requires_grad=True)
        self.icmat1 = torch.nn.Parameter(torch.rand(input_size, self.gru_input_size1), requires_grad=True)

        for param in self.rnn0.parameters():
            param.requires_grad = False
        self.icmat0.requires_grad = False
        self.rnn0.eval()

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.infer_pos0 = None
        self.infer_pos1 = None

        self.pre_trained = False

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        sig_gru0=torch.matmul(input,self.icmat0)#+self.icmat_bias1
        self.infer_pos0 = wta_layer(sig_gru0,schedule=schedule)
        hout0, hn0 = self.rnn0(self.infer_pos0,hidden1[0],logit_mode=True)
        output0 = torch.matmul(hout0, torch.t(self.icmat0))#+self.icmat_bias2

        sig_gru1 = torch.matmul(input, self.icmat1)  # +self.icmat_bias1
        self.infer_pos1 = wta_layer(sig_gru1, schedule=schedule)
        hout1, hn1 = self.rnn1(self.infer_pos1, hidden1[1], logit_mode=True)
        output1 = torch.matmul(hout1, torch.t(self.icmat1))  # +self.icmat_bias2

        output=self.softmax(output0+output1)

        return output,[hn0,hn1]

    def forward0(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward using only layer0
        :param input:
        :param hidden:
        :return:
        """
        sig_gru0=torch.matmul(input,self.icmat0)#+self.icmat_bias1
        self.infer_pos0 = wta_layer(sig_gru0,schedule=schedule)
        hout0, hn0 = self.rnn0(self.infer_pos0,hidden1[0],logit_mode=True)
        output0 = torch.matmul(hout0, torch.t(self.icmat0))#+self.icmat_bias2

        output=self.softmax(output0)

        return output,[hn0,None]

    def forward1(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward using only layer1
        :param input:
        :param hidden:
        :return:
        """

        sig_gru1 = torch.matmul(input, self.icmat1)  # +self.icmat_bias1
        self.infer_pos1 = wta_layer(sig_gru1, schedule=schedule)
        hout1, hn1 = self.rnn1(self.infer_pos1, hidden1[1], logit_mode=True)
        output1 = torch.matmul(hout1, torch.t(self.icmat1))  # +self.icmat_bias2

        output = self.softmax(output1)

        return output,[None,hn1]

    def build_concept_map(self,num,prior,device,switch=1):
        # Build concept mapping from word id to concept id with importance
        # num, number of words to calculate
        hidden = self.initHidden_cuda(device, 1)
        dim = self.input_size
        inputt = torch.zeros((num,dim))
        for ii in range(num):
            inputt[ii,ii]=1.0
        inputt = inputt.view(-1, 1, dim)
        inputt = inputt.type(torch.FloatTensor)
        _ = self.forward(inputt.to(device), hidden,schedule=1.0)
        if switch==0:
            np_con = self.infer_pos0.cpu().data.numpy()
            print(self.infer_pos0.cpu().shape)
        elif switch==1:
            np_con = self.infer_pos1.cpu().data.numpy()
            print(self.infer_pos1.cpu().shape)
        id_2_con = []
        for ii in range(num):
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
        for iicon in id_2_con:
            id_2_con_s.append(swapdict[iicon])
        return id_2_con_s

    def initHidden(self,batch):
        return [self.rnn0.initHidden(batch),self.rnn1.initHidden(batch)]

    def initHidden_cuda(self,device, batch):
        return [self.rnn0.initHidden_cuda(device, batch),self.rnn1.initHidden_cuda(device, batch)]

class GRU_TwoLayerCon(torch.nn.Module):
    """
    A trial of two layer training stracture for trial of layered inductive bias of Natural Language.
    Layer 1 is a pre-trained layer like GRU over POS which freezes.
    Layer 2 is a projecction perpendicular to layer 1
    Attention Gating is used to choose plitable information to two layers
    """
    def __init__(self, rnn, input_size, hidden_size, output_size, num_layers=1):
        """
        init
        :param gru_l1: gru_l1 [GRU, input_size, output_size]
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param num_layers:
        """
        super(self.__class__, self).__init__()
        self.rnn = rnn
        for param in self.rnn.parameters():
            param.requires_grad = False
        # self.rnn.eval()
        self.gru_input_size = self.rnn.input_size
        self.gru_output_size = self.rnn.output_size

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.i2s = torch.nn.Linear(input_size, self.gru_input_size) # input to sigmoid
        self.s2g= torch.nn.Linear(self.gru_input_size, self.gru_input_size) # sigmoid to GRU input
        self.g2o = torch.nn.Linear(self.gru_output_size, output_size) # GRU output to output

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.infer_pos = None

        self.pre_trained = False

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        if self.pre_trained:
            for param in self.i2s.parameters():
                param.requires_grad = False
        sig_gru=self.i2s(input)
        # sig_gru=self.sigmoid(sig_gru)
        # input_gru=self.s2g(sig_gru)
        # self.infer_pos =torch.exp(self.softmax(sig_gru))
        self.infer_pos = wta_layer(sig_gru,schedule=schedule)
        hout, hn = self.rnn(self.infer_pos,hidden1,logit_mode=True)
        hout=logit_sampling_layer(hout)
        output = self.g2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def pre_training(self,input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input:
        :param hidden1:
        :param add_logit:
        :param logit_mode:
        :param schedule:
        :return:
        """
        self.pre_trained = True
        sig_gru = self.i2s(input)
        output = self.softmax(sig_gru)
        return output, None

    def initHidden(self,batch):
        return self.rnn.initHidden(batch)

    def initHidden_cuda(self,device, batch):
        return self.rnn.initHidden_cuda(device, batch)

class LSTM_AdvConNet(torch.nn.Module):
    """
    A trial of adversarial two step training for finding information partition.
    A self attention gate G1 is used to partition Wordvec into two part with G1 and 1-G1 into wp1 and wp2
    wp1 does POS task and self-modeling task with perp p11 and p12
    wp2 does POS task and self-modeling task with perp p21 and p22
    Train step 1 minimize p11+p12+p21+p22 with G1 fixed
    Train step 2 minimize p11-p12-p21+p22 with ontly G1 changing

    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        init
        :param input_size:
        :param hidden_size: list
        :param output_size: list
        """
        super(self.__class__, self).__init__()
        self.input_size=input_size
        self.hidden_size_pos=hidden_size[0]
        self.hidden_size_auto = hidden_size[1]
        self.output_size_pos=output_size[0]
        self.output_size_auto = output_size[1]
        self.num_layers=num_layers

        # self.rnn = rnn
        # for param in self.rnn.parameters():
        #     param.requires_grad = False
        # self.rnn.eval()

        self.context_id=0 # Training step controlling flexible part of network


        self.i2g = torch.nn.Linear(input_size, self.input_size) # self-attention

        self.lstm11 = torch.nn.LSTM(input_size, self.hidden_size_pos, num_layers=num_layers)
        self.h2o11 = torch.nn.Linear(self.hidden_size_pos, self.output_size_pos)

        self.lstm12 = torch.nn.LSTM(input_size, self.hidden_size_auto, num_layers=num_layers)
        self.h2o12 = torch.nn.Linear(self.hidden_size_auto, self.output_size_auto)

        self.lstm21 = torch.nn.LSTM(input_size, self.hidden_size_pos, num_layers=num_layers)
        self.h2o21 = torch.nn.Linear(self.hidden_size_pos, self.output_size_pos)

        self.lstm22 = torch.nn.LSTM(input_size, self.hidden_size_auto, num_layers=num_layers)
        self.h2o22 = torch.nn.Linear(self.hidden_size_auto, self.output_size_auto)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.infer_pos = None

        self.pre_trained = False

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        if self.context_id==0: # fix self attention
            for param in self.parameters():
                param.requires_grad = True
            for param in self.i2g.parameters():
                param.requires_grad = False
        elif self.context_id==1: # train self attention
            for param in self.parameters():
                param.requires_grad = False
            for param in self.i2g.parameters():
                param.requires_grad = True
        else:
            raise Exception("Self.train_step not known")

        self_att=self.i2g(input)
        self_att=self.sigmoid(self_att)
        wp1=self_att*input
        wp2 = (1-self_att) * input
        hidden11=hidden1[0]
        hout11, hn11 = self.lstm11(wp1,hidden11)
        output11 = self.h2o11(hout11)
        hidden12 = hidden1[1]
        hout12, hn12 = self.lstm12(wp1, hidden12)
        output12 = self.h2o12(hout12)
        hidden21 = hidden1[2]
        hout21, hn21 = self.lstm21(wp2, hidden21)
        output21 = self.h2o21(hout21)
        hidden22 = hidden1[3]
        hout22, hn22 = self.lstm22(wp2, hidden22)
        output22 = self.h2o22(hout22)

        output=[output11,output12,output21,output22]
        hn=[hn11,hn12,hn21,hn22]

        return output,hn

    def initHidden(self,batch):
        hd11=[Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True),
              Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True)]
        hd12 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True)]
        hd21 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True)]
        hd22 =  [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True)]
        return [hd11,hd12,hd21,hd22]

    def initHidden_cuda(self, device, batch):
        hd11 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True).to(device)]
        hd12 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True).to(device)]
        hd21 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_pos), requires_grad=True).to(device)]
        hd22 = [Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size_auto), requires_grad=True).to(device)]
        return [hd11, hd12, hd21, hd22]

class LSTM_PJ_NLP(torch.nn.Module):
    """
    PyTorch LSTM for NLP with rotation and projection
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, cuda_flag=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rotate = torch.nn.Linear(input_size, input_size)
        self.proj_size = 200
        self.conj_mode = False

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.lstm = torch.nn.LSTM(self.proj_size, hidden_size, num_layers=num_layers)
        self.lstm_conj = torch.nn.LSTM(input_size-self.proj_size, hidden_size, num_layers=num_layers)

        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=weight_dropout)
            self.lstm_conj = WeightDrop(self.lstm_conj, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        rot_input = self.rotate(input)
        if self.conj_mode:
            proj_input = rot_input[:, :, self.proj_size:]
            for param in self.rotate.parameters():
                param.requires_grad = False
            hout, hn = self.lstm_conj(proj_input, hidden1)
        else:
            proj_input = rot_input[:,:,:self.proj_size]
            hout, hn = self.lstm(proj_input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)]

class Binary_W2V(torch.nn.Module):
    """
    PyTorch binary word to vector trial
    """
    def __init__(self, input_size, interaction_w, window_size):
        """

        :param lsize:
        :param mode:
        """
        super(self.__class__, self).__init__()
        # Setter or detector
        self.input_size=input_size
        self.interaction_w=interaction_w
        self.window_size=window_size

        self.w2v_bin=torch.nn.Linear(self.input_size, 3)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.sinorm = None

        self.theta=torch.zeros((3,3))
        self.theta[0,0]=1
        self.theta[2, 2] = 1
        self.theta[0, 2] = -1
        self.theta[2, 0] = -1

        self.multip=torch.zeros(3)
        self.multip[0]=1
        self.multip[2]=-1

        self.reg=torch.zeros(3)
        self.reg[0] = 1
        self.reg[2] = 1

        self.interaction_m=torch.zeros((self.window_size,self.window_size))
        for ii in range(self.window_size):
            for jj in range(self.window_size):
                if ii<jj and ii>=jj-self.interaction_w:
                    self.interaction_m[ii,jj]=1

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpuavail else "cpu")

        if self.gpuavail:
            self.theta=self.theta.to(self.device)
            self.interaction_m = self.interaction_m.to(self.device)
            self.multip = self.multip.to(self.device)
            self.reg= self.reg.to(self.device)

        self.lamda=0.01

    def forward(self, input, hidden,schedule=None):
        """
        Forward
        :param input: [window, batch, lsize]
        :param hidden:
        :param plogits: prior
        :return:
        """
        input=input.permute(1,0,2) # [batch, window, lsize]
        si=self.w2v_bin(input) # [batch, window, 3]
        sinorm = self.softmax(si)
        self.sinorm=sinorm
        sit=torch.matmul(sinorm,self.theta) # [batch, window, 3]
        si2=sinorm.permute(0,2,1) # [batch, 3, window]
        sits1=torch.matmul(sit,si2) # [batch, window, window]
        sits1_mask=self.interaction_m*sits1
        nitem=(2*self.window_size-self.interaction_w-1)*self.interaction_w/2
        batch_size=input.shape[0]
        E1=torch.sum(sits1_mask)/batch_size/nitem
        meanF=torch.sum(sinorm*self.multip.view(1,-1))/self.window_size/batch_size
        E2=meanF*meanF

        regF=torch.sum(sinorm*self.reg.view(1,-1))/self.window_size/batch_size
        E3=regF

        return -E1+E2+self.lamda*E3, None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None

class LSTM_DIMGatting_NLP(torch.nn.Module):
    """
    PyTorch LSTM for NLP with input dimemsion gating.
    Basic hypothesis, more understandable means tighter information bottleneck
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, cuda_flag=True, noise=0.1):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        self.gate = torch.nn.Parameter(torch.rand(input_size), requires_grad=True)
        self.noise = noise

        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.siggate = self.sigmoid(self.gate)

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpuavail else "cpu")

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        self.siggate=self.sigmoid(self.gate)
        input=input*self.siggate
        if self.gpuavail:
            input=input+self.noise*(2*torch.rand(input.shape).to(self.device)-1)
        else:
            input = input  + self.noise * (2 * torch.rand(input.shape) - 1)
        hout, hn = self.lstm(input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)]


class LSTM_DIMProbGatting_NLP(torch.nn.Module):
    """
    PyTorch LSTM for NLP with input dimemsion gating with a probablistic 0/1 gate, two step training
    Basic hypothesis, more understandable means tighter information bottleneck
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, weight_dropout=0.0, cuda_flag=True):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        # self.gate_para = torch.ones(input_size)

        self.gate =  torch.nn.Parameter(torch.rand(input_size), requires_grad=True)

        if weight_dropout>0:
            print("Be careful, only GPU works for now.")
            self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.siggate = self.sigmoid(self.gate)

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpuavail else "cpu")

        # self.mysampler = MySampler.apply

    # def sample_gate(self,shape):
    #     gate=torch.rand(shape)
    #     zeros=torch.zeros(shape)
    #     if self.gpuavail:
    #         gate=gate.to(self.device)
    #         zeros=zeros.to(self.device)
    #     gate=gate-self.siggate
    #     gate[gate == zeros] = 1e-8
    #     gate=(gate/torch.abs(gate)+1.0)/2
    #     if torch.isnan(gate).any():
    #         print(self.siggate,torch.sum(torch.isnan(gate)))
    #         raise Exception("NaN Error")
    #     return gate

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        self.siggate = self.sigmoid(self.gate)

        # torch.autograd.set_detect_anomaly(True)
        # probgate=self.mysampler(self.siggate)
        mysampler=None
        probgate = mysampler(self.siggate)
        input=input*probgate
        hout, hn = self.lstm(input,hidden1)
        output = self.h2o(hout)
        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return [Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)]


class RECURSIVE_AUTOENCODER(torch.nn.Module):
    """
    Recursive auto-encoder
    """
    def __init__(self,input_size):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.ff = torch.nn.Linear(input_size, input_size)
        self.ffknw_in = torch.nn.Linear(2 * input_size, input_size)
        self.ffknw_outl = torch.nn.Linear(input_size, input_size)  # Auto-encoder kind
        self.ffknw_outr = torch.nn.Linear(input_size, input_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def pre_training(self,input):
        predvec = self.ff(input)
        predvec = self.softmax(predvec)
        return predvec

    def fknw(self,vecl,vecr,schedule=1.0):
        vecm = self.ffknw_in(torch.cat((vecl, vecr), dim=-1))
        vecm = torch.exp(self.softmax(vecm))
        # vecm = self.wta_layer(vecm, schedule=schedule)
        return vecm

    def fknw_test(self,vecl,vecr,schedule=1.0):
        vecm = self.ffknw_in(torch.cat((vecl, vecr), dim=-1))
        vecm = torch.exp(self.softmax(vecm))

        np_input = vecm.data.numpy()
        argmax_i = np.argsort(-np_input, axis=-1)[0]
        argmax_i = torch.from_numpy(np.array(argmax_i))
        concept_layer_i = torch.zeros(vecm.shape)
        concept_layer_i.scatter_(-1, argmax_i, 1.0)
        concept_layer_i = concept_layer_i

        ginput_masked = vecm * concept_layer_i
        ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)

        return ginput_masked

    def wta_layer(self,l_input,schedule=1,wta_noise=0.0):
        upper_t = 0.3
        Nind = int((1.0 - schedule) * (self.input_size - 2) * upper_t) + 1
        np_input=l_input.data.numpy()
        argmax_i = np.argsort(-np_input, axis=-1)[ :, 0:Nind]
        argmax_i = torch.from_numpy(argmax_i)
        concept_layer_i = torch.zeros(l_input.shape)
        concept_layer_i.scatter_(-1, argmax_i, 1.0)
        concept_layer_i = concept_layer_i + wta_noise * torch.rand(concept_layer_i.shape)

        ginput_masked = l_input * concept_layer_i
        ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)
        return ginput_masked


    def forward(self, input, hidden=None, schedule=1.0):
        """
        One step All to All forward input[Seq,batch,l_size]
        :param input:
        :return:
        """
        # def cal_perp_torch(predvec,invec):
        #     perp = -torch.sum(invec * predvec, dim=-1) + torch.log(torch.sum(torch.exp(predvec), dim=-1))
        #     return perp

        def cal_kldiv_torch(p, q):
            """
            Cal KL divergence of p over q
            :param data:
            :return:
            """
            p = p + 1e-9
            q = q + 1e-9
            p = p / torch.sum(p, dim=-1, keepdim=True)
            q = q / torch.sum(q, dim=-1, keepdim=True)
            kld = torch.sum(p * torch.log(p / q), dim=-1)
            return kld

        def cal_autoencode(vecl,vecr):
            vecm = self.fknw(vecl, vecr, schedule=schedule)
            recvecl = self.ffknw_outl(vecm)
            recvecl = torch.exp(self.softmax(recvecl))
            recvecr = self.ffknw_outr(vecm)
            recvecr = torch.exp(self.softmax(recvecr))
            perpl = cal_kldiv_torch(recvecl, vecl)
            perpr = cal_kldiv_torch(recvecr, vecr)
            resperp = perpl + perpr
            return vecm,resperp

        def step_forward(input,argmax_i):
            seql=[]
            seqr = []
            for ii in range(len(argmax_i)):
                seql.append(input[argmax_i[ii],ii,:])
                seqr.append(input[argmax_i[ii]+1,ii,:])
            vecl = torch.stack(seql,0)
            vecr = torch.stack(seqr, 0)
            vecm, resperp = cal_autoencode(vecl,vecr)
            resout=[]
            length=len(input)
            for ii in range(len(argmax_i)):
                ind=argmax_i[ii]
                if ind>0 and ind<length-2:
                    resout.append(torch.cat((input[:ind,ii,:], vecm[ii,:].view(1,-1),input[ind+2:,ii,:]), dim=0))
                elif ind==0 and ind<length-2:
                    resout.append(torch.cat((vecm[ii, :].view(1, -1), input[ind + 2:, ii, :]),dim=0))
                elif ind>0 and ind==length-2:
                    resout.append(torch.cat((input[:ind, ii, :], vecm[ii, :].view(1, -1))))
                else:
                    resout.append(vecm[ii, :].view(1, -1))
            resout=torch.stack(resout, 0)
            resout=resout.permute((1,0,2))
            return resout,resperp/2

        length=input.shape[0]
        batch=input.shape[1]

        tot_perp=torch.zeros(1)
        for ii in range(length-1): # Looping over tree buidling
            vecm, perp = cal_autoencode(input[:-1,:,:],input[1:,:,:])
            argmax_i = torch.argmax(perp,dim=0)
            input,perp_rec=step_forward(input,argmax_i)
            tot_perp=tot_perp+torch.sum(perp_rec)/batch
        return tot_perp/length, None

    def initHidden(self,batch):
        return None


class STACK_FF_NLP(torch.nn.Module):
    """
    Stack augmented FF network for handling explanatory hierachical binary sequence
    Not working yet !!!!!!!!!!
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.ff = torch.nn.Linear(input_size, input_size)
        self.ffgate = torch.nn.Linear(1, 2) # Gate should link with perp
        # self.ffgate = torch.nn.Linear(2 * input_size , 2)
        self.ffknw_in = torch.nn.Linear(2 * input_size, input_size)
        self.ffknw_outh = torch.nn.Linear(input_size, input_size) # Auto-encoder kind
        self.ffknw_outi = torch.nn.Linear(input_size, input_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.max_depth=6 # 5+1
        self.window_size=10

    def wta_layer(self,l_input,Nind=1,wta_noise=0.0):
        np_input=l_input.data.numpy()
        argmax_i = np.argsort(-np_input, axis=-1)[:, :, 0:Nind]
        argmax_i = torch.from_numpy(argmax_i)
        concept_layer_i = torch.zeros(l_input.shape)
        concept_layer_i.scatter_(-1, argmax_i, 1.0)
        concept_layer_i = concept_layer_i + wta_noise * torch.rand(concept_layer_i.shape)

        ginput_masked = l_input * concept_layer_i
        ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)
        return ginput_masked

    def pre_training(self,input):
        predvec = self.ff(input)
        predvec = self.softmax(predvec)
        return predvec

    def fknw(self,invec,hvec):
        inp_cat = torch.cat((invec.view(1, 1, -1), hvec.view(1, 1, -1)), dim=-1)
        new_inp = self.ffknw_in(inp_cat)
        new_inp = torch.exp(self.softmax(new_inp))
        # new_inp = self.wta_layer(new_inp)
        return new_inp

    def forward(self, input, stackl):

        """
        Forward
        :param input:
        :param stackl: [stack,pointer]
        :return:
        """
        # for param in self.ff.parameters():
        #     param.requires_grad = False

        stack=stackl[0]
        stack_pointer = stackl[1]
        input_pointer = stackl[2]
        # print("init,",input_pointer)
        rprepush = 1.0 # total push prob until now
        resperp = torch.tensor(0.0)

        def push(vec,stack,pointer):
            """
            Quantum push
            :param vec:
            :param stack: a mixed quantum stack
            :param pointer: a quantum pointer
            :return:
            """
            stack=stack*(1-pointer).view(pointer.shape[0],pointer.shape[1],1)+torch.matmul(pointer.view(pointer.shape[0],pointer.shape[1],1),vec)
            pointer_new = torch.cat((torch.tensor(0.0).view(pointer.shape[0], 1),pointer[:, :-1]),dim=-1)
            return stack, pointer_new

        def pop(stack,pointer):
            """
            Quantum pop
            :param stack:
            :param pointer:
            :return:
            """
            pointer_new = torch.cat((pointer[:,1:], torch.tensor(0.0).view(pointer.shape[0],1)),dim=-1)
            if torch.sum(pointer_new, -1) > 0:
                outvec = torch.matmul(pointer_new/torch.sum(pointer_new, -1),stack)
            else:
                outvec = torch.matmul(pointer_new, stack)
            stack_new=(1-pointer_new).view(pointer_new.shape[0],pointer_new.shape[1],1)*stack
            addvec=torch.zeros(pointer_new.shape)
            addvec[:,0]=1
            pointer_new=pointer_new+addvec*pointer[:,0] # pointer[:,0] is invalid for pop
            return outvec, stack_new, pointer_new

        def read(stack,pointer):
            """
            Quantum read
            :param stack:
            :param pointer:
            :return:
            """
            outvec = torch.matmul(pointer,stack)
            return outvec

        def get(input,pointer):
            """
            Quantum get
            :param input:
            :param pointer:
            :return:
            """
            outvec = torch.matmul(pointer[:,:-1],input)
            pointer_new = torch.cat((torch.tensor(0.0).view(pointer.shape[0], 1), pointer[:, :-1]), dim=-1)
            pointer_new[:,-1]=pointer_new[:,-1]+pointer[:,-1]
            return outvec, pointer_new

        def cal_kldiv_torch(p, q):
            """
            Cal KL divergence of p over q
            :param data:
            :return:
            """
            p = p + 1e-9
            q = q + 1e-9
            p = p / torch.sum(p)
            q = q / torch.sum(q)
            kld = torch.sum(p * torch.log(p / q))
            return kld

        def calforward(invec,hvec):

            # print("calforward 1,", invec, hvec)
            predvec = self.ff(hvec)
            predvec = self.softmax(predvec)
            # perp_pred = -torch.sum(invec * predvec) + torch.log(torch.sum(torch.exp(predvec)))
            perp_pred=cal_kldiv_torch(predvec, invec)

            new_inp = self.fknw(invec,hvec)
            proj_hvec = self.ffknw_outh(new_inp)
            proj_hvec=torch.exp(self.softmax(proj_hvec))
            proj_invec = self.ffknw_outi(new_inp)
            proj_invec=torch.exp(self.softmax(proj_invec))

            perp_invec = cal_kldiv_torch(proj_invec, invec)
            perp_hvec = cal_kldiv_torch(proj_hvec, hvec)
            perp=perp_pred+perp_invec/2+perp_hvec/2

            # inp_cat = torch.cat((invec.view(1, 1, -1), hvec.view(1, 1, -1)), dim=-1)
            # gatevec = self.ffgate(inp_cat)
            gatevec = self.ffgate(perp_pred.view(-1))
            gatevec = torch.exp(self.softmax(gatevec))  # gatevec [PUSP prop, POP prob]
            gatevec=gatevec.view(-1)

            if torch.sum(hvec, -1) == 0: # No hvec means PUSH
                gatevec=torch.from_numpy(np.array([1.0,0.0]))
                gatevec=gatevec.type(torch.FloatTensor)
            elif torch.sum(invec, -1) == 0: # No invec means POP
                # print("POP")
                gatevec=torch.from_numpy(np.array([0.0,1.0]))
                gatevec=gatevec.type(torch.FloatTensor)
            # print("calforward 2,", gatevec, perp, invec)
            return gatevec,perp,new_inp

        new_input_pointer = torch.zeros(self.window_size+1)
        new_stack_pointer = torch.zeros(self.hidden_size)
        new_stack = torch.zeros(stack.shape)
        stackmem=[]
        stack_pointermem=[]
        input_pointermem=[]
        wldlprobmem=torch.zeros(self.max_depth)
        perpmem=torch.zeros(self.max_depth)
        for ii_wrdl in range(self.max_depth): # different world line
            if ii_wrdl == 0: # POP POPC PUSH
                invec,stack_new_1, stack_pointer_1 = pop(stack, stack_pointer)
                hvec, stack_new_1, stack_pointer_1 = pop(stack_new_1, stack_pointer_1)
                gatevec, perp , new_inp= calforward(invec,hvec)
                perpmem[ii_wrdl]=perpmem[ii_wrdl]+perp
                stack_new_1,stack_pointer_1=push(new_inp,stack_new_1,stack_pointer_1)
                stackmem.append(stack_new_1)
                stack_pointermem.append(stack_pointer_1)
                input_pointermem.append(input_pointer)
                wldlprobmem[ii_wrdl]=wldlprobmem[ii_wrdl]+gatevec[1]
                # print("cal ii_wrdl1 ", ii_wrdl, gatevec, resperp, rprepush,stack_pointer_1,new_stack_pointer)
            elif ii_wrdl >= 1:
                # print("ii_wrdl:", ii_wrdl)
                input_pointer_2=input_pointer
                stack_new_2=stack
                stack_pointer_2=stack_pointer
                for ii_push in range(ii_wrdl-1):
                    invec, input_pointer_2 = get(input,input_pointer_2)
                    hvec = read(stack,stack_pointer_2)
                    stack_new_2, stack_pointer_2 = push(invec, stack_new_2, stack_pointer_2)
                invec, input_pointer_2 = get(input, input_pointer_2)
                hvec, stack_new_2, stack_pointer_2 = pop(stack_new_2, stack_pointer_2)
                gatevec, perp, new_inp = calforward(invec, hvec)
                stack_new_2, stack_pointer_2 = push(new_inp, stack_new_2, stack_pointer_2)
                stackmem.append(stack_new_2)
                stack_pointermem.append(stack_pointer_2)
                input_pointermem.append(input_pointer_2)
                input_avail=1-input_pointer_2[:,-1]
                wldlprobmem[ii_wrdl] = wldlprobmem[ii_wrdl] + (1.0-wldlprobmem[ii_wrdl-1]) * gatevec[1]*input_avail
                perpmem[ii_wrdl] = perpmem[ii_wrdl] + perp

        wldlprobmem=wldlprobmem/torch.sum(wldlprobmem)
        # print("perpmem",perpmem)
        resperp=torch.sum(wldlprobmem*perpmem)
        # print("resperp", resperp)
        for ii_wrdl in range(self.max_depth):
            new_input_pointer=new_input_pointer+wldlprobmem[ii_wrdl]*input_pointermem[ii_wrdl]
            new_stack_pointer=new_stack_pointer+wldlprobmem[ii_wrdl]*stack_pointermem[ii_wrdl]
            new_stack=new_stack+wldlprobmem[ii_wrdl]*stackmem[ii_wrdl]
        if torch.sum(new_stack_pointer, -1)>0:
            new_stack_pointer = new_stack_pointer / torch.sum(new_stack_pointer, -1, keepdim=True)
        if torch.sum(new_input_pointer, -1)>0:
            new_input_pointer = new_input_pointer / torch.sum(new_input_pointer, -1, keepdim=True)
        # print("End,",new_stack_pointer,new_input_pointer)
        return resperp,[new_stack, new_stack_pointer, new_input_pointer]

    def initHidden(self,batch):
        # Initialization of stack,stack_pointer,input_pointer
        stack_pointer=torch.zeros(batch, self.hidden_size)
        stack_pointer[:,0]=1
        # Adding an extra space of "running out of input" for input_pointer
        input_pointer=torch.zeros(batch, self.window_size+1)
        input_pointer[:, 0] = 1
        return [Variable(torch.zeros(batch, self.hidden_size,self.input_size), requires_grad=True),stack_pointer,input_pointer]

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

        # First WTA scheduling of input
        ginput = self.i2g(input)

        # if self.gpuavail:
        #     npginput = ginput.cpu().data.numpy()
        # else:
        #     npginput = ginput.data.numpy()
        #
        # upper_t = 0.3
        # Nind = int((1.0 - schedule) * (self.concept_size - 2) * upper_t) + 1  # Number of Nind largest number kept
        # argmax_i = np.argsort(-npginput, axis=-1)[:, :, 0:Nind]
        # argmax_i = torch.from_numpy(argmax_i).to(self.device)
        # self.concept_layer_i = torch.zeros(ginput.shape).to(self.device)
        # self.concept_layer_i.scatter_(-1, argmax_i, 1.0)
        # self.concept_layer_i = self.concept_layer_i + wta_noise * torch.rand(self.concept_layer_i.shape).to(self.device)
        #
        # ginput_masked = ginput * self.concept_layer_i
        # ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)

        ginput_masked =  wta_layer(ginput,schedule=schedule)

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

    def __init__(self, input_size, hidden_size, concept_size, output_size, block_mode=False):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.concept_size = concept_size
        self.input_size = input_size

        self.i2m = torch.nn.Linear(input_size, hidden_size)
        self.m2h = torch.nn.Linear(hidden_size, concept_size)
        self.h2o = torch.nn.Linear(concept_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        # dropout
        # self.cdrop = torch.nn.Dropout(p=0.5)

        # mid result storing
        self.concept_layer = None
        self.hout2con_masked = None

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpuavail else "cpu")

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

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, wta_noise=0.0, schedule=1.0):
        """
        Forward, GRU WTA winner percentage scheduling
        :param input:
        :param hidden:
        :return:
        """
        upper_t = 0.3
        Nind = int((1.0 - schedule) * (self.concept_size - 2) * upper_t) + 1  # Number of Nind largest number kept

        hidden1 = self.i2m(input)
        hidden1 = self.relu(hidden1)
        hout2con = self.m2h(hidden1)
        hout2con=self.relu(hout2con)

        if self.gpuavail:
            npginput = hout2con.cpu().data.numpy()
        else:
            npginput = hout2con.data.numpy()
        argmax_i = np.argsort(-npginput, axis=-1)[:, :, 0:Nind]
        argmax_i = torch.from_numpy(argmax_i).to(self.device)
        self.concept_layer_i = torch.zeros(hout2con.shape).to(self.device)
        self.concept_layer_i.scatter_(-1, argmax_i, 1.0)
        self.concept_layer_i = self.concept_layer_i + wta_noise * torch.rand(self.concept_layer_i.shape).to(self.device)

        hout2con_masked = hout2con * self.concept_layer_i
        hout2con_masked = hout2con_masked / torch.norm(hout2con_masked, 2, -1, keepdim=True)

        # GRU part

        output = self.h2o(hout2con_masked)

        if add_logit is not None:
            output = output + add_logit
        if not logit_mode:
            output = self.softmax(output)
        return output, None

    # def forward(self, input, hid=None, wta_noise=0.0, schedule=1.0):
    #     """
    #     Forward, path transfer scheduling
    #     :param input:
    #     :param hid: no use
    #     :return:
    #     """
    #     hidden1 = self.i2m(input)
    #     hidden1 = self.relu(hidden1)
    #     hout2con = self.m2h(hidden1)
    #     hout2con=self.relu(hout2con)
    #
    #     argmax = torch.argmax(hout2con, dim=-1, keepdim=True)
    #     self.concept_layer = torch.zeros(hout2con.shape).to(self.device)
    #     self.concept_layer.scatter_(-1, argmax, 1.0)
    #     self.concept_layer = self.concept_layer + wta_noise * torch.rand(self.concept_layer.shape).to(self.device)
    #     if self.block_mode:
    #         hout2con_masked=self.concept_layer
    #     else:
    #         hout2con_masked = hout2con * self.concept_layer
    #         hout2con_masked = hout2con_masked / torch.norm(hout2con_masked, 2, -1, keepdim=True)
    #     self.hout2con_masked = hout2con_masked
    #
    #     hout2con2=hout2con/torch.norm(hout2con,2,-1,keepdim=True)
    #
    #     output = self.h2o(schedule*hout2con_masked+(1-schedule)*hout2con2)
    #     output=self.softmax(output)
    #
    #     return output, None

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

class DisSpsMemNet(torch.nn.Module):
    """
    2019-4-2 Discrete Sparse Memory Net
    """
    def __init__(self, input_size, abstract_size, memory_size, output_size, cuda_flag=True, coop=None):
        """

        :param input_size:
        :param abstract_size: abstract layer
        :param memory_size:
        :param output_size: working number of output knowledge detector
        :param cuda_flag:
        """
        super(self.__class__, self).__init__()

        self.input_size = input_size
        self.abstract_size = abstract_size
        self.memory_size = memory_size
        self.output_size = output_size

        self.i2a = torch.nn.Linear(input_size, abstract_size)
        self.a2keepz = torch.nn.Linear(abstract_size, memory_size)
        self.a2sreth = torch.nn.Linear(abstract_size, memory_size)
        self.h2o = torch.nn.Linear(abstract_size+memory_size, output_size)
        self.o2l= torch.nn.Linear(output_size, input_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.outnrn = None
        self.mem = None

        self.coop = None
        if coop is not None:
            self.coop=coop
            for param in self.coop.parameters():
                param.requires_grad = False

    def plot_layer(self,lname="all"):
        if lname=="i2a":
            mat=self.i2a.cpu().weight.data.numpy()
        elif lname=="a2keepz":
            mat=self.a2keepz.cpu().weight.data.numpy()
        elif lname=="a2sreth":
            mat=self.a2sreth.cpu().weight.data.numpy()
        elif lname=="h2o":
            mat=self.h2o.cpu().weight.data.numpy()
        elif lname=="o2l":
            mat=self.o2l.cpu().weight.data.numpy()
        elif lname=="all":
            allname=["i2a","a2keepz","a2sreth","h2o","o2l"]
            for nameitem in allname:
                self.plot_layer(nameitem)
        if lname != "all":
            plot_mat(mat,title=lname,symmetric=True,tick_step=1)


    def plot_layer_all(self):
        allname = ["i2a", "a2keepz", "a2sreth", "h2o", "o2l"]
        subplotidx=230
        for nameitem in allname:
            mat = getattr(self, nameitem).cpu().weight.data.numpy()
            subplotidx=subplotidx+1
            plt.subplot(subplotidx)
            plot_mat(mat,title=nameitem,symmetric=True,tick_step=1,show=False)
        plt.show()

    def forward(self, input, hidden1, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        abs_layer=self.i2a(input)
        # abs_layer=(mysign(abs_layer)+1)/2
        abs_layer = torch.sigmoid(abs_layer)
        abs_layer = mysampler(abs_layer)
        ## Set reset mem
        keep_layer=self.a2keepz(abs_layer)
        # keep_layer = (mysign(keep_layer) + 1) / 2
        keep_layer = torch.sigmoid(keep_layer)
        keep_layer=mysampler(keep_layer)
        sret_layer=self.a2sreth(abs_layer)
        # sret_layer = (mysign(sret_layer) + 1) / 2
        sret_layer = torch.sigmoid(sret_layer)
        sret_layer = mysampler(sret_layer)
        mem=hidden1*keep_layer+(1-keep_layer)*sret_layer
        # mem = mem.clamp(min=0)
        self.mem=mem

        catabsmem=torch.cat((abs_layer,mem),dim=-1)
        outnrn=self.h2o(catabsmem)
        # outnrn = (mysign(outnrn) + 1) / 2
        outnrn = torch.sigmoid(outnrn)
        outnrn = mysampler(outnrn)

        self.outnrn=outnrn

        output=self.o2l(outnrn)

        if self.coop is not None:
            coop_logit , _ = self.coop(input, hidden1, logit_mode=True)
            output=output+coop_logit

        output=self.softmax(output)

        return output,mem

    def initHidden(self,batch):
        return Variable(torch.zeros(batch, self.memory_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(batch, self.memory_size), requires_grad=True).to(device)

class GRU_Cell_Maskout(torch.nn.Module):
    """
    PyTorch GRU with pathway stochastic mask
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # self.Wir = torch.nn.Linear(input_size, hidden_size)
        # self.Whr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)
        self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = Gumbel_Sigmoid()
        # self.sigmoid = MyHardSig.apply # clamp input
        self.tanh = torch.nn.Tanh()
        # self.tanh = Gumbel_Tanh()
        # self.tanh = MySign.apply
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.zt=None
        self.nt=None

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """
        # rt=self.sigmoid(self.Wir(input)+self.Whr(hidden))
        # rt = self.sigmoid(self.Wir(input) + self.Whr(hidden),temperature=1.1-schedule)
        # temp1=torch.clamp(self.Wir(input)+self.Whr(hidden),-1,1)
        # rt=self.sigmoid(temp1)
        zt=self.sigmoid(self.Wiz(input)+self.Whz(hidden))
        # zt = self.sigmoid(self.Wiz(input) + self.Whz(hidden),temperature=1.1-schedule)
        # temp2 = torch.clamp(self.Wiz(input)+self.Whz(hidden), -1, 1)
        # zt = self.sigmoid(temp2)
        # nt=self.tanh(self.Win(input)+rt*self.Whn(hidden),temperature=1.1-schedule)
        nt = self.tanh(self.Win(input) * self.Whn(hidden))
        # nt = self.tanh(self.Win(input) * self.Whn(hidden), temperature=1.1 - schedule)
        # temp3 = torch.clamp(self.Win(input)+rt*self.Whn(hidden), -1, 1)
        # nt = self.tanh(temp3)
        ht = (1 - zt) * nt + zt * hidden
        output = self.h2o(ht)
        output = self.softmax(output)

        self.zt=zt
        self.nt=nt

        return output, ht

    # def plot_layer_all(self):
    #     # allname = ["h2o", "Wir",  "Wiz", "Win", "Whr", "Whz" ,"Whn"]
    #     allname = ["h2o", "Wiz", "Win", "Whz", "Whn"]
    #     subplotidx=330
    #     for nameitem in allname:
    #         mat = getattr(self, nameitem).cpu().weight.data.numpy()
    #         subplotidx=subplotidx+1
    #         plt.subplot(subplotidx)
    #         plot_mat(mat,title=nameitem,symmetric=True,tick_step=1,show=False)
    #     plt.show()

    def plot_layer_all(self):
        srow=2
        scol=5
        allname = ["h2o", "Wiz", "Win", "Whz", "Whn"]
        for nn,nameitem in enumerate(allname):
            mat = getattr(self, nameitem).cpu().weight.data.numpy()
            plt.subplot(srow, scol, 1+nn)
            plot_mat(mat, title=nameitem, symmetric=True, tick_step=1, show=False)
            plt.subplot(srow, scol, 1+nn + scol)
            mat = getattr(self, nameitem).cpu().bias.data.numpy()
            plot_mat(mat.reshape(1, -1), title=nameitem+"_bias", symmetric=True, tick_step=1, show=False)
        plt.show()

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

class GRU_Cell_DropConnect(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0, zoneout_rate=0.0, switch="normal"):
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.zoneout_rate=zoneout_rate # seems not that useful
        self.dropout_rate = dropout_rate

        # self.Wir = torch.nn.Linear(input_size, hidden_size)
        # self.Whr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)
        self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.switch=switch

        if switch=="normal":
            self.sigmoid = torch.nn.Sigmoid()
            self.tanh = torch.nn.Tanh()
        elif switch=="st": # Straight through estimator
            self.sigmoid = MyHardSig.apply  # clamp input
            self.tanh = MySign.apply
        elif switch=="gumbel": # Gumbel sigmoid
            self.sigmoid = Gumbel_Sigmoid()
            self.tanh = Gumbel_Tanh()

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.zt = None
        self.nt = None

    def para_copy(self,gru):
        """
        Copy parameter from another gru
        :param gru:
        :return:
        """
        allname = ["h2o", "Wiz", "Win", "Whz", "Whn"]
        for nn,nameitem in enumerate(allname):
            rnn_w = getattr(self, nameitem).weight
            rnn_w.data.copy_(getattr(gru, nameitem).weight.data)
            rnn_b = getattr(self, nameitem).bias
            rnn_b.data.copy_(getattr(gru, nameitem).bias.data)

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """

        if self.dropout_rate>0.0: # Not very correct
            mask = self.Whz.weight.new_ones(self.Whz.weight.size())
            mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
            # self.Whz.weight= torch.nn.Parameter(mask*self.Whz.weight)
            self.Whz.weight.data.mul_(mask)
            mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
            # self.Whn.weight = torch.nn.Parameter(mask * self.Whn.weight)
            self.Whn.weight.data.mul_(mask)

        if self.switch=="gumbel":
            zt = self.sigmoid(self.Wiz(input) + self.Whz(hidden), temperature=1.1 - schedule)
            nt = self.tanh(self.Win(input) + self.Whn(hidden), temperature=1.1 - schedule)
        else:
            zt = self.sigmoid(self.Wiz(input) + self.Whz(hidden))
            nt = self.tanh(self.Win(input) + self.Whn(hidden))

        if self.training:
            mask=(np.sign(np.random.random(list(zt.shape))-self.zoneout_rate)+1)/2
            mask = torch.from_numpy(mask)
            mask = mask.type(torch.FloatTensor)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                mask=mask.to(device)
            zt=1-(1-zt)*mask
            ht=(1-zt)*nt+zt*hidden
        else:
            ht = (1 - zt) * nt + zt * hidden
        output = self.h2o(ht)
        output = self.softmax(output)

        self.zt = zt
        self.nt = nt

        return output, ht

    # def plot_layer_all(self):
    #     allname = ["h2o", "Wir",  "Wiz","Win", "Whr","Whz", "Whn"]
    #     subplotidx=330
    #     for nameitem in allname:
    #         mat = getattr(self, nameitem).cpu().weight.data.numpy()
    #         subplotidx=subplotidx+1
    #         plt.subplot(subplotidx)
    #         plot_mat(mat,title=nameitem,symmetric=True,tick_step=1,show=False)
    #     plt.show()

    def plot_layer_all(self):
        srow=2
        scol=5
        allname = ["h2o", "Wiz", "Win", "Whz", "Whn"]
        for nn,nameitem in enumerate(allname):
            mat = getattr(self, nameitem).cpu().weight.data.numpy()
            plt.subplot(srow, scol, 1+nn)
            plot_mat(mat, title=nameitem, symmetric=True, tick_step=1, show=False)
            plt.subplot(srow, scol, 1+nn + scol)
            mat = getattr(self, nameitem).cpu().bias.data.numpy()
            plot_mat(mat.reshape(1, -1), title=nameitem+"_bias", symmetric=True, tick_step=1, show=False)
        plt.show()

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

class GRU_seq2seq(torch.nn.Module):
    """
    A GRU based seq2seq model
    """

    def __init__(self, input_size, hidden_size, output_size, output_len, num_layers=1, dropout_rate=0.0,
                 zoneout_rate=0.0, switch="normal", pruning=False, cuda_device="cuda:0"):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len=output_len
        self.num_layers = num_layers

        # self.gru_enc = torch.nn.GRU(input_size, hidden_size)
        # self.gru_dec = torch.nn.GRU(input_size, hidden_size)

        self.hgate = torch.nn.Parameter(torch.rand(hidden_size)+1.0, requires_grad=True)
        self.gru_enc = GRU_Cell_Seq(input_size, hidden_size,dropout_rate=dropout_rate)
        self.gru_dec = GRU_Cell_Seq(input_size, hidden_size,dropout_rate=dropout_rate)

        # self.gru_enc = WeightDrop(torch.nn.GRU(input_size, hidden_size), ['weight_hh_l0'], dropout=dropout_rate)
        # self.gru_dec = WeightDrop(torch.nn.GRU(input_size, hidden_size), ['weight_hh_l0'], dropout=dropout_rate)

        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.gsigmoid = Gumbel_Sigmoid(cuda_device=cuda_device)
        self.mysampler=mysampler
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.gpuavail else "cpu")

        self.pruning=pruning
        self.pruning_mask=None

    def para_copy(self,gru):
        """
        Copy parameter from another gru
        :param gru:
        :return:
        """
        self.gru_enc.para_copy(gru.gru_enc)
        self.gru_dec.para_copy(gru.gru_dec)
        self.h2o.para_copy(gru.h2o)

        self.h2o.weight.data.copy_(gru.h2o.weight.data)
        self.h2o.bias.data.copy_(gru.h2o.bias.data)

        self.hgate.data.copy_(gru.hgate.data)

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
        """
        Seqence mode only
        :param input:
        :param hidden:
        :param add_logit:
        :param logit_mode:
        :param schedule:
        :return:
        """
        # Sequence mode Encoding
        # assert len(input) == 2

        # input_in,input_dec = input
        input_in = input

        if self.pruning:
            exphgate=self.hgate.expand_as(hidden)
            siggate=self.sigmoid(exphgate)
            # pruning_mask = self.gsigmoid(self.hgate, temperature=1.01 - schedule)
            pruning_mask = self.mysampler(siggate,cuda_device=self.device)
        else:
            pruning_mask=None
        self.pruning_mask = pruning_mask

        hout, hn = self.gru_enc(input_in, hidden, schedule=schedule, pruning_mask=pruning_mask)

        # Sequence mode Decoding
        input_dec=torch.zeros((self.output_len,input.shape[1],input.shape[2]))
        if self.gpuavail:
            input_dec=input_dec.to(self.device)
        hout, hn = self.gru_dec(input_dec, hn, schedule=schedule,pruning_mask=pruning_mask)
        output = self.h2o(hout)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

# class GRU_seq2seq_v2(torch.nn.Module):
#     """
#     A GRU based seq2seq model, version 2, where encoder decoder combined, input output combined , to provide output information for training
#     """
#
#     def __init__(self, input_size, hidden_size, output_size, output_len, num_layers=1, dropout_rate=0.0,
#                  zoneout_rate=0.0, switch="normal", pruning=False):
#         super(self.__class__, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.output_len=output_len
#         self.num_layers = num_layers
#
#         # self.gru_enc = torch.nn.GRU(input_size, hidden_size)
#         # self.gru_dec = torch.nn.GRU(input_size, hidden_size)
#
#         self.hgate = torch.nn.Parameter(torch.rand(hidden_size), requires_grad=True)
#         self.gru_enc_dec = GRU_Cell_Seq(input_size, hidden_size,dropout_rate=dropout_rate)
#
#         self.h2o = torch.nn.Linear(hidden_size, output_size)
#
#         self.sigmoid = torch.nn.Sigmoid()
#         self.gsigmoid = Gumbel_Sigmoid()
#         self.tanh = torch.nn.Tanh()
#         self.softmax = torch.nn.LogSoftmax(dim=-1)
#         self.gpuavail = torch.cuda.is_available()
#         self.device = torch.device("cuda:0" if self.gpuavail else "cpu")
#
#         self.pruning=pruning
#         self.pruning_mask=None
#
#     def para_copy(self,gru):
#         """
#         Copy parameter from another gru
#         :param gru:
#         :return:
#         """
#         self.gru_enc_dec.para_copy(gru.gru_enc)
#
#         self.h2o.weight.data.copy_(gru.h2o.weight.data)
#         self.h2o.bias.data.copy_(gru.h2o.bias.data)
#
#         self.hgate.data.copy_(gru.hgate.data)
#
#     def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
#         """
#         Seqence mode only
#         :param input:
#         :param hidden:
#         :param add_logit:
#         :param logit_mode:
#         :param schedule:
#         :return:
#         """
#         # Sequence mode Encoding
#
#         if self.pruning:
#             pruning_mask = self.gsigmoid(self.hgate, temperature=1.01 - schedule)
#         else:
#             pruning_mask=None
#         self.pruning_mask = pruning_mask
#
#         hout, hn = self.gru_enc_dec(input, hidden,schedule=schedule,pruning_mask=pruning_mask)
#         output = self.h2o(hout)
#
#         return output,hn
#
#     def initHidden(self,batch):
#         return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)
#
#     def initHidden_cuda(self,device, batch):
#         return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

class GRU_Cell_Seq(torch.nn.Module):
    """
    PyTorch self-coded GRU with sequence mode
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.0, zoneout_rate=0.0, switch="normal"):
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.zoneout_rate=zoneout_rate # seems not that useful
        self.dropout_rate = dropout_rate

        # self.Wir = torch.nn.Linear(input_size, hidden_size)
        # self.Whr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)
        self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.switch=switch

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # if switch=="normal":
        #     self.sigmoid = torch.nn.Sigmoid()
        #     self.tanh = torch.nn.Tanh()
        # elif switch=="st": # Straight through estimator
        #     self.sigmoid = MyHardSig.apply  # clamp input
        #     self.tanh = MySign.apply
        # elif switch=="gumbel": # Gumbel sigmoid
        #     self.sigmoid = Gumbel_Sigmoid()
        #     self.tanh = Gumbel_Tanh()

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.zt = None
        self.nt = None

    def para_copy(self,gru):
        """
        Copy parameter from another gru
        :param gru:
        :return:
        """
        allname = ["Wiz", "Win", "Whz", "Whn"]
        for nn,nameitem in enumerate(allname):
            rnn_w = getattr(self, nameitem).weight
            rnn_w.data.copy_(getattr(gru, nameitem).weight.data)
            rnn_b = getattr(self, nameitem).bias
            rnn_b.data.copy_(getattr(gru, nameitem).bias.data)

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None, pruning_mask=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """
        assert len(input.shape)==3
        batch=input.shape[1]
        ht=hidden
        output=None

        # if self.dropout_rate>0.0: # Not working
        #     mask = self.Whz.weight.new_ones(self.Whz.weight.size())
        #     mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
        #     # self.Whz.weight= torch.nn.Parameter(mask*self.Whz.weight)
        #     self.Whz.weight.data.mul_(mask)
        #     mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
        #     # self.Whn.weight = torch.nn.Parameter(mask * self.Whn.weight)
        #     self.Whn.weight.data.mul_(mask)

        for iis in range(input.shape[0]):
            input_ii=input[iis,:,:]
            if pruning_mask is not None:
                ht = ht * pruning_mask
            zt = self.sigmoid(self.Wiz(input_ii) + self.Whz(ht))
            nt = self.tanh(self.Win(input_ii) + self.Whn(ht))
            ht = (1 - zt) * nt + zt * ht
            if output is None:
                output = nt.view(1, batch, self.hidden_size)
            else:
                output = torch.cat(
                    (output.view(-1, batch, self.hidden_size), nt.view(1, batch, self.hidden_size)), dim=0)


        # if self.dropout_rate>0.0:
        #     mask = self.Whz.weight.new_ones(self.Whz.weight.size())
        #     mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
        #     # self.Whz.weight= torch.nn.Parameter(mask*self.Whz.weight)
        #     self.Whz.weight.data.mul_(mask)
        #     mask = torch.nn.functional.dropout(mask, p=self.dropout_rate, training=True)
        #     # self.Whn.weight = torch.nn.Parameter(mask * self.Whn.weight)
        #     self.Whn.weight.data.mul_(mask)
        #
        # if self.switch=="gumbel":
        #     zt = self.sigmoid(self.Wiz(input) + self.Whz(hidden), temperature=1.1 - schedule)
        #     nt = self.tanh(self.Win(input) + self.Whn(hidden), temperature=1.1 - schedule)
        # else:
        #     zt = self.sigmoid(self.Wiz(input) + self.Whz(hidden))
        #     nt = self.tanh(self.Win(input) + self.Whn(hidden))
        #
        # if self.training:
        #     mask=(np.sign(np.random.random(list(zt.shape))-self.zoneout_rate)+1)/2
        #     mask = torch.from_numpy(mask)
        #     mask = mask.type(torch.FloatTensor)
        #     if torch.cuda.is_available():
        #         device = torch.device("cuda:0")
        #         mask=mask.to(device)
        #     zt=1-(1-zt)*mask
        #     ht=(1-zt)*nt+zt*hidden
        # else:
        #     ht = (1 - zt) * nt + zt * hidden
        # output = self.h2o(ht)
        # output = self.softmax(output)
        #
        # self.zt = zt
        # self.nt = nt

        return output, ht

    # def plot_layer_all(self):
    #     allname = ["h2o", "Wir",  "Wiz","Win", "Whr","Whz", "Whn"]
    #     subplotidx=330
    #     for nameitem in allname:
    #         mat = getattr(self, nameitem).cpu().weight.data.numpy()
    #         subplotidx=subplotidx+1
    #         plt.subplot(subplotidx)
    #         plot_mat(mat,title=nameitem,symmetric=True,tick_step=1,show=False)
    #     plt.show()

    def plot_layer_all(self):
        srow=2
        scol=5
        allname = ["Wiz", "Win", "Whz", "Whn"]
        for nn,nameitem in enumerate(allname):
            mat = getattr(self, nameitem).cpu().weight.data.numpy()
            plt.subplot(srow, scol, 1+nn)
            plot_mat(mat, title=nameitem, symmetric=True, tick_step=1, show=False)
            plt.subplot(srow, scol, 1+nn + scol)
            mat = getattr(self, nameitem).cpu().bias.data.numpy()
            plot_mat(mat.reshape(1, -1), title=nameitem+"_bias", symmetric=True, tick_step=1, show=False)
        plt.show()

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)