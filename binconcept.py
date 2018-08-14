"""
Trial of exciting idea of binary iterationary concept formation algorithm.
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

class Bin_Concept(object):
    """
    Main class of binary concept formation
    """
    def __init__(self,dataset, lsize):
        self.dataset = dataset
        self.lsize = lsize

        self.model = None
        self.mlost = 1.0e9

        self.submodel = None

    def proj(self):
        """
        Class projection
        :return:
        """
        proj=[]
        for ii in range(self.lsize):
            vecn = np.zeros(self.lsize)
            vecn[ii] = 1
            vect = torch.from_numpy(vecn)
            vect = vect.type(torch.FloatTensor)
            prj = self.model.sigmoid(self.model.bin(vect))
            prjn=prj.data.numpy()[0]
            proj.append(prjn)
        return proj

    def plt_proj(self):
        """
        Plot bin projection
        :return:
        """
        prjv=[]
        for ii in range(self.lsize):
            vecn=np.zeros(self.lsize)
            vecn[ii]=1
            vect=torch.from_numpy(vecn)
            vect=vect.type(torch.FloatTensor)
            prj=self.model.sigmoid(self.model.bin(vect))
            prjv.append(prj.data.numpy()[0])
        plt.figure()
        plt.bar(np.array(range(self.lsize)), prjv)
        plt.title("prj plot")
        plt.show()

    def do_eval(self):
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
            output, hidden = rnn(x, hidden)
            outputl.append(output.view(-1).data.numpy())
            hiddenl.append(hidden)

        outputl = np.array(outputl)
        outputl = Variable(torch.from_numpy(outputl).contiguous())
        outputl = outputl.permute((1, 0))
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

    def run_training_sub(self, step, learning_rate=1e-2, batch=20, window=30, save=None,seqtrain=False):
        """
        run sub class training
        :param step:
        :param learning_rate:
        :param batch:
        :param window:
        :param save:
        :param seqtrain:
        :return:
        """
        lsize=self.lsize
        proj=self.proj()

        prtstep = int(step / 10)
        startt = time.time()

        dataset = self.dataset
        datasetsub=[]
        for data in dataset:
            datasetsub.append([proj[data]])
        databp = torch.from_numpy(np.array(datasetsub))

        rnn = GRU_Sub(self.lsize)
        rnn.train()

        lossc = torch.nn.CrossEntropyLoss()

        def custom_KNWLoss(outputl, outlab, model, cstep):
            loss1 = lossc(outputl, outlab)
            # logith2o = model.h2o.weight+model.h2o.bias.view(-1)
            # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
            # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
            # l1_reg = model.Ws.weight.norm(1) + model.Wr.weight.norm(1)
            # l1_invreg = model.bin.weight.norm(2)  # regulerization for sharper bin
            return loss1
            # return loss1# + 0.001 * l1_reg * cstep / step + 0.005 * lossh2o * cstep / step  # +0.3*lossz+0.3*lossn #

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
                    output, hidden = rnn(x, hidden, batch=batch)
                    if type(outputl) == type(None):
                        outputl = output.view(batch, lsize, 1)
                    else:
                        outputl = torch.cat((outputl.view(batch, lsize, -1), output.view(batch, lsize, 1)), dim=2)
                loss = custom_KNWLoss(outputl, outlab, rnn, iis)
            else:
                # LSTM/GRU provided whole sequence training
                vec1m = None
                vec2m = None
                outputl = None
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                    vec2 = databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1, :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(window, 1, -1)
                        vec2m = vec2.view(window, 1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                        vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
                x = Variable(vec1m.reshape(window, batch, 1).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(window, batch, 1).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden, batch=batch)
                loss = custom_KNWLoss(output.permute(1, 2, 0), outlab, rnn, iis)

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


    def run_training(self, step, learning_rate=1e-2, batch=20, window=30, save=None,seqtrain=False):
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
            rnn=GRU_Proj(self.lsize)
        else:
            rnn = self.model
        rnn.train()

        lossc = torch.nn.CrossEntropyLoss()
        def custom_KNWLoss(outputl, outlab, model, cstep):
            loss1 = lossc(outputl, outlab)
            # logith2o = model.h2o.weight+model.h2o.bias.view(-1)
            # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
            # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
            # l1_reg = model.Ws.weight.norm(1) + model.Wr.weight.norm(1)
            # l1_invreg=model.bin.weight.norm(2) # regulerization for sharper bin
            return loss1#-0.002*l1_invreg
            # return loss1# + 0.001 * l1_reg * cstep / step + 0.005 * lossh2o * cstep / step  # +0.3*lossz+0.3*lossn #

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
                    output, hidden = rnn(x, hidden,batch=batch)
                    if type(outputl) == type(None):
                        outputl = output.view(batch, lsize, 1)
                    else:
                        outputl = torch.cat((outputl.view(batch, lsize, -1), output.view(batch, lsize, 1)), dim=2)
                loss = custom_KNWLoss(outputl, outlab, rnn, iis)
            else:
                # LSTM/GRU provided whole sequence training
                vec1m = None
                vec2m = None
                outputl = None
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
                    vec2 = databp[int(rstartv[iib])+1:int(rstartv[iib])+window+1, :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(window, 1, -1)
                        vec2m = vec2.view(window, 1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                        vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
                x = Variable(vec1m.reshape(window, batch, lsize).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(window, batch, lsize).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden, batch=batch)
                loss = custom_KNWLoss(output.permute(1,2,0), outlab, rnn, iis)

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

class GRU_Sub(torch.nn.Module):
    """
    Sub GRU to train model when class is given
    """
    def __init__(self, lsize, hidden_size=30):
        super(GRU_Sub, self).__init__()

        self.hidden_size = hidden_size
        self.lsize = lsize

        self.gru=torch.nn.GRU(1,hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, lsize)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=1):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input,hidden)
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(1, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)


class GRU_Proj(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, lsize, hidden_size=1):
        super(GRU_Proj, self).__init__()

        self.hidden_size = hidden_size
        self.lsize = lsize

        self.bin=torch.nn.Linear(lsize,hidden_size)
        self.gru=torch.nn.GRU(hidden_size,hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, lsize)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=1):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        binin=self.bin(input.view(-1, batch, self.lsize))
        hout, hn = self.gru(binin,hidden)
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(1, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)
