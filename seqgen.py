"""
Utility for symbolic sequence generation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np

import matplotlib.pyplot as plt

from ncautil.tfnlp import TFNet
import tensorflow as tf
import torch
from torch.autograd import Variable

__author__ = "Harry He"

class SeqGen(object):
    """
        Class for sequence generation wrong
        """

    def __init__(self):
        self.vocab = dict([])

    def gen_ABseq(self, length):
        """
        Generate ABseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15]
        self.vocab["cA"] = cA
        self.vocab["cB"] = cB

        nA = len(cA)
        nB = len(cB)
        res = []
        for ii in range(length):
            if ii % 2 == 0:
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
            else:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
        return res

    def gen_AABBseq(self, length):
        """
        Generate AABBseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15]
        self.vocab["cA"] = cA
        self.vocab["cB"] = cB

        nA = len(cA)
        nB = len(cB)
        res = []
        for ii in range(length):
            if ii % 4 in [0,1] :
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
            else:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
        return res

    def gen_AABBCCseq(self, length):
        """
        Generate AABBseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [0, 1, 4, 8, 9]
        cC = [6, 10, 12, 14, 15]
        self.vocab["cA"] = cA
        self.vocab["cB"] = cB
        self.vocab["cC"] = cC

        nA = len(cA)
        nB = len(cB)
        nC = len(cC)
        res = []
        for ii in range(length):
            if ii % 6 in [0,1] :
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
            elif ii % 6 in [2,3] :
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
            else:
                # pick from class C
                id = int(np.floor(np.random.rand() * nC))
                pknum = cC[id]
                res.append(pknum)
        return res

    def gen_ABCseq(self, length):
        """
        Generate ABseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [0, 1, 4, 8, 9]
        cC = [6, 10, 12, 14, 15]
        self.vocab["cA"] = cA
        self.vocab["cB"] = cB
        self.vocab["cC"] = cC

        nA = len(cA)
        nB = len(cB)
        nC = len(cC)
        res = []
        for ii in range(length):
            if ii%3==0:
                # pick from class A
                id=int(np.floor(np.random.rand()*nA))
                pknum=cA[id]
                res.append(pknum)
            elif ii%3==1:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
            else:
                # pick from class C
                id = int(np.floor(np.random.rand() * nC))
                pknum = cC[id]
                res.append(pknum)
        return res

    def one_hot(self,num):
        ytemp = np.zeros(16)
        ytemp[num] = 1
        return ytemp

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size))

class PT_RNN(object):
    """
    PyTorch Recurrent Net
    """
    def __init__(self):
        self.seqgen = SeqGen()
        self.model = None

    def do_eval(self,step,mode="AABBCC"):
        id1 = int(np.floor(np.random.rand() * 4))
        if id1 in [0,1]:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cA"])))
            nninit = self.seqgen.vocab["cA"][id2]
        elif id1 in [2,3]:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cB"])))
            nninit = self.seqgen.vocab["cB"][id2]
        # else:
        #     id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cC"])))
        #     invec_init = self.seqgen.numtobin(self.seqgen.vocab["cC"][id2])
        seqres = []
        seqres.append(nninit)
        xin=Variable(torch.from_numpy(np.array(self.seqgen.one_hot(nninit)).reshape(-1,16)), requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn=np.sum(vec)
            return vec/dwn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec=np.exp(vec)
            dwn=np.sum(vec)
            return vec/dwn

        hidden = self.model.initHidden()
        for ii in range(step):
            y_pred,hidden = self.model(xin,hidden)
            ynp=y_pred.data.numpy().reshape(16)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig=0
            for ii in range(len(pii)):
                rndp=rndp-pii[ii]
                if rndp<0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig)).reshape(-1,16)), requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            seqres.append(dig)

        tot=0
        right=0
        if mode =="AABB":
            vA = self.seqgen.vocab["cA"]
            vB = self.seqgen.vocab["cB"]
            for ii in range(len(seqres)-3):
                tot=tot+1
                int1=seqres[ii]
                int2 = seqres[ii+1]
                int3 = seqres[ii+2]
                int4 = seqres[ii+3]
                if ((int1 in vA) and (int2 in vA) and(int3 in vB) and (int4 in vB))\
                        or ((int1 in vA) and (int2 in vB) and(int3 in vB) and (int4 in vA))\
                        or ((int1 in vB) and (int2 in vB) and(int3 in vA) and (int4 in vA))\
                        or ((int1 in vB) and (int2 in vA) and(int3 in vA) and (int4 in vB)):
                    right=right+1
                print(int1,int2,int3,int4)
            print("True rate is: "+str(right/tot))
        elif mode =="AABBCC":
            vA = self.seqgen.vocab["cA"]
            vB = self.seqgen.vocab["cB"]
            vC = self.seqgen.vocab["cC"]
            for ii in range(len(seqres)-5):
                tot=tot+1
                int1=seqres[ii]
                int2 = seqres[ii+1]
                int3 = seqres[ii+2]
                int4 = seqres[ii+3]
                int5 = seqres[ii + 4]
                int6 = seqres[ii + 5]
                if ((int1 in vA) and (int2 in vA) and(int3 in vB) and (int4 in vB) and(int5 in vC) and (int6 in vC))\
                        or ((int1 in vA) and (int2 in vB) and (int3 in vB) and (int4 in vC) and(int5 in vC) and (int6 in vA))\
                        or ((int1 in vB) and (int2 in vB) and (int3 in vC) and (int4 in vC) and(int5 in vA) and (int6 in vA))\
                        or ((int1 in vB) and (int2 in vC) and (int3 in vC) and (int4 in vA) and(int5 in vA) and (int6 in vB)) \
                        or ((int1 in vC) and (int2 in vC) and (int3 in vA) and (int4 in vA) and (int5 in vB) and (int6 in vB)) \
                        or ((int1 in vC) and (int2 in vA) and (int3 in vA) and (int4 in vB) and (int5 in vB) and (int6 in vC)):
                    right=right+1
                print(int1,int2,int3,int4,int5,int6)
            print("True rate is: "+str(right/tot))

    def run(self,step,learning_rate=5e-3,mode="AABBCC",window=30):
        if mode == "AABB":
            seqs = self.seqgen.gen_AABBseq(step)
        elif mode == "AABBCC":
            seqs = self.seqgen.gen_AABBCCseq(step)
        else:
            raise Exception("Mode not recognize.")

        #def __init__(self, input_size, hidden_size, output_size)
        rnn=RNN(16, 6, 16)

        # rnn.zero_grad()

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
            return loss+0.005*l2_reg

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        his = 0
        for ii in range(0,step-window,window):

            outputl=[]
            yl=[]
            hidden = rnn.initHidden()

            for nn in range(window):
                num1=seqs[ii+nn]
                num2 = seqs[ii+nn+1]
                np1= np.array(self.seqgen.one_hot(num1))
                np2 = np.array(self.seqgen.one_hot(num2))
                x = Variable(torch.from_numpy(np1.reshape(-1, 16)), requires_grad=True)
                y = Variable(torch.from_numpy(np2.reshape(-1, 16)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden)
                outputl.append(output)
                yl.append(y)

            loss = customized_loss(outputl, yl, rnn)

            if int(ii / 10000) != his:
                print(ii, loss.data[0])
                his=int(ii / 10000)

            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        self.model=rnn


class PT_FFN(object):
    """
    PyTorch Feedforward Net
    """
    def __init__(self):
        self.seqgen = SeqGen()
        self.batch_size=10
        self.trainN=100000
        self.model = None

    def get_data(self,seqs):
        resin=[]
        resout=[]
        nS=len(seqs)-1
        for ii in range(self.batch_size):
            id = int(np.floor(np.random.rand() * nS))
            resin.append(seqs[id])
            resout.append(seqs[id+1])
        return np.array(resin),np.array(resout)

    def do_eval(self,step,mode="AB"):
        id1 = int(np.floor(np.random.rand() * 2))
        if id1 == 0:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cA"])))
            nninit = self.seqgen.vocab["cA"][id2]
        elif id1 == 1:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cB"])))
            nninit = self.seqgen.vocab["cB"][id2]
        # else:
        #     id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cC"])))
        #     invec_init = self.seqgen.numtobin(self.seqgen.vocab["cC"][id2])
        seqres = []
        seqres.append(nninit)
        xin=Variable(torch.from_numpy(np.array(self.seqgen.one_hot(nninit)).reshape(-1,16)), requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn=np.sum(vec)
            return vec/dwn

        for ii in range(step):
            y_pred = self.model(xin)
            ynp=y_pred.data.numpy().reshape(16)
            rndp = np.random.rand()
            pii = linp(ynp)
            # print(ynp)
            # print(pii)
            for ii in range(len(pii)):
                rndp=rndp-pii[ii]
                if rndp<0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig)).reshape(-1,16)), requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            seqres.append(dig)

        tot=0
        right=0
        if mode =="AB":
            for ii in range(len(seqres)-1):
                tot=tot+1
                int1=seqres[ii]
                int2 = seqres[ii+1]
                if not(bool(int1 in self.seqgen.vocab["cA"]) ^ bool(int2 in self.seqgen.vocab["cB"])):
                    right=right+1
                print(int1,int2)
            print("True rate is: "+str(right/tot))
        elif mode=="ABC":
            for ii in range(len(seqres)-1):
                tot=tot+1
                int1=seqres[ii]
                int2 = seqres[ii+1]
                if (int1 in self.seqgen.vocab["cA"] and int2 in self.seqgen.vocab["cB"])\
                        or (int1 in self.seqgen.vocab["cB"] and int2 in self.seqgen.vocab["cC"])\
                        or (int1 in self.seqgen.vocab["cC"] and int2 in self.seqgen.vocab["cA"]):
                    right=right+1
                print(int1,int2)
            print("True rate is: "+str(right/tot))


    def run(self,step,learning_rate=5e-3,mode="AB"):

        if mode == "AB":
            seqs = self.seqgen.gen_ABseq(self.trainN)
        elif mode == "ABC":
            seqs = self.seqgen.gen_ABCseq(self.trainN)
        else:
            raise Exception("Unknown mode.")

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H ,D_out = self.batch_size, 16, 16 ,16

        # Use the nn package to define our model and loss function.
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.Softmax(),
        )

        def customized_loss(x, y, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)
            loss = -torch.sqrt((torch.sum(x*y)))+0.005*l2_reg
            return loss

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Variables it should update.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        for t in range(step):
            # Forward pass: compute predicted y by passing x to the model.
            xnp,ynp = self.get_data(seqs)
            xin=[]
            for num in xnp:
                xin.append(list(self.seqgen.one_hot(num)))
            xin=np.array(xin)
            yout=[]
            for num in ynp:
                yout.append(self.seqgen.one_hot(num))
            yout=np.array(yout)
            x = Variable(torch.from_numpy(xin.reshape(-1,D_in)))
            y = Variable(torch.from_numpy(yout), requires_grad=False)

            # x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

            y_pred = model(x)

            # Compute and print loss.
            loss = customized_loss(y_pred, y, model)

            if t%1000==1:
                print(t, loss.data[0])

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
        self.model=model
















class SeqGen_wrong(object):
    """
    Class for sequence generation wrong
    """
    def __init__(self):
        self.vocab=dict([])

    def numtobin(self,nn):
        """
        Change an integer to binary
        :param num: 5
        :return: (0101)
        """
        bstr = bin(nn)[2:]
        blist = [int(s) for s in bstr]
        blzeros = []
        if len(blist) < 4:
            blzeros = [0 for s in range(4 - len(blist))]
        return tuple(blzeros + blist)

    def bintonum(self,bin):
        """
        binary to number
        :param bin: binary tuple
        :return:
        """
        strg=""
        for dig in bin:
            strg=strg+str(int(dig))
        return int(strg,2)

    def gen_ABseq_onehot(self,length):
        """
        Generate ABseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [1, 4, 6, 8, 9, 10, 12, 14, 15]
        self.vocab["cA"] = cA
        self.vocab["cB"] = cB

        nA = len(cA)
        nB = len(cB)
        res = []
        for ii in range(length):
            if ii % 2 == 0:
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
            else:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
        return res

    def gen_ABseq(self,length):
        """
        Exp 1: ABABABABABABABA...
        A: binary form of 2,3,5,7,11,13
        B: others
        which number to pick is random among class
        :return: [(0010),(0001)]
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [1, 4, 6, 8, 9, 10, 12, 14, 15]
        self.vocab["cA"]=cA
        self.vocab["cB"] = cB

        nA=len(cA)
        nB=len(cB)
        res=[]
        for ii in range(length):
            if ii%2==0:
                # pick from class A
                id=int(np.floor(np.random.rand()*nA))
                pknum=cA[id]
                tpnum=self.numtobin(pknum)
                res.append(tpnum)
            else:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                tpnum = self.numtobin(pknum)
                res.append(tpnum)
        return res

    def gen_ABCseq(self,length):
        """
        Exp 1: ABCABCABCABC...
        A: binary form of 2,3,5,7,11,13
        B: others
        which number to pick is random among class
        :return: [(0010),(0001)]
        """
        cA = [2, 3, 5, 7, 11, 13]
        cB = [1, 4, 8, 9]
        cC = [6, 10, 12, 14, 15]
        self.vocab["cA"]=cA
        self.vocab["cB"] = cB
        self.vocab["cC"] = cC

        nA=len(cA)
        nB=len(cB)
        nC=len(cC)
        res=[]
        for ii in range(length):
            if ii%3==0:
                # pick from class A
                id=int(np.floor(np.random.rand()*nA))
                pknum=cA[id]
                tpnum=self.numtobin(pknum)
                res.append(tpnum)
            elif ii%3==1:
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                tpnum = self.numtobin(pknum)
                res.append(tpnum)
            else:
                # pick from class C
                id = int(np.floor(np.random.rand() * nC))
                pknum = cC[id]
                tpnum = self.numtobin(pknum)
                res.append(tpnum)
        return res

class PT_FFN_wrong(object):
    """
    PyTorch Feedforward Net
    """
    def __init__(self):
        self.seqgen = SeqGen()
        self.batch_size=10
        self.trainN=100000
        self.model = None

    def get_data(self,seqs,mode="one_hot"):
        resin=[]
        resout=[]
        nS=len(seqs)-1
        for ii in range(self.batch_size):
            id = int(np.floor(np.random.rand() * nS))
            resin.append(seqs[id])
            resout.append(seqs[id+1])
        if mode=="one_hot":
            tmp=[]
            for item in resin:
                ytemp = np.zeros(15)
                ytemp[item - 1] = 1
                tmp.append(ytemp)
            resin=tmp
        return np.array(resin),np.array(resout)

    def do_eval(self,step,mode="one_hot"):
        id1 = int(np.floor(np.random.rand() * 2))
        if id1 == 0:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cA"])))
            invec_init = self.seqgen.numtobin(self.seqgen.vocab["cA"][id2])
        elif id1 == 1:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cB"])))
            invec_init = self.seqgen.numtobin(self.seqgen.vocab["cB"][id2])
        # else:
        #     id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cC"])))
        #     invec_init = self.seqgen.numtobin(self.seqgen.vocab["cC"][id2])
        seqres = []
        seqres.append(np.array(invec_init))
        shape2=4
        if mode=="one_hot":
            int_ini = self.seqgen.bintonum(invec_init)
            seqres = []
            ytemp = np.zeros(15)
            ytemp[int_ini - 1] = 1
            seqres.append(int_ini)
            shape2=15
        xin=Variable(torch.from_numpy(np.array(invec_init).reshape(-1,shape2)), requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn=np.sum(vec)
            return vec/dwn

        for ii in range(step):
            y_pred = self.model(xin)
            ynp=y_pred.data.numpy().reshape(15)
            rndp = np.random.rand()
            pii = linp(ynp)
            # print(ynp)
            # print(pii)
            for ii in range(len(pii)):
                rndp=rndp-pii[ii]
                if rndp<0:
                    dig = ii
                    break
            xtemp=self.seqgen.numtobin(dig+1)
            xin = Variable(torch.from_numpy(np.array(xtemp).reshape(-1,4)), requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            if mode == "one_hot":
                ytemp = np.zeros(15)
                ytemp[int_ini - 1] = 1
                xin = Variable(torch.from_numpy(np.array(xtemp).reshape(-1, 4)), requires_grad=False)
                xin = xin.type(torch.FloatTensor)
            seqres.append(np.array(xtemp).reshape(4))

        tot=0
        right=0
        for ii in range(len(seqres)-1):
            tot=tot+1
            int1=self.seqgen.bintonum(seqres[ii])
            int2 = self.seqgen.bintonum(seqres[ii+1])
            if not(bool(int1 in self.seqgen.vocab["cA"]) ^ bool(int2 in self.seqgen.vocab["cB"])):
                right=right+1
            print(int1,int2)
        print("True rate is: "+str(right/tot))


    def run(self,step,learning_rate=5e-3):

        seqs = self.seqgen.gen_ABseq_onehot(self.trainN)

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H ,D_out = self.batch_size, 15, 10 ,15

        # Use the nn package to define our model and loss function.
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out,bias=False),
            # torch.nn.ReLU(),
            # torch.nn.Linear(H, D_out),
            torch.nn.Softmax(),
        )
        loss_fn = torch.nn.MSELoss(size_average=False)

        def sftmax(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            vecnp=vec.data.numpy()
            dwn=np.sum(np.exp(vecnp))
            res=np.exp(vecnp)/dwn
            return Variable(torch.from_numpy(res))

        def customized_loss(x, y, model):
            # print(x,y)
            Mpara = []
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)
            loss = -torch.sqrt((torch.sum(x*y)))+0.01*l2_reg
            return loss

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Variables it should update.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        for t in range(step):
            # Forward pass: compute predicted y by passing x to the model.
            xnp,ynp = self.get_data(seqs)
            yout=[]
            for item in ynp:
                assert len(item)==D_in
                dig=self.seqgen.bintonum(item)
                # yout.append(dig-1)
                ytemp=np.zeros(15)
                ytemp[dig-1]=1
                yout.append(ytemp)
            yout=np.array(yout)
            x = Variable(torch.from_numpy(xnp.reshape(-1,4)))
            y = Variable(torch.from_numpy(yout), requires_grad=False)

            # x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

            y_pred = model(x)

            # Compute and print loss.
            loss = customized_loss(y_pred, y, model)

            # l1_crit = torch.nn.L1Loss(size_average=False)
            # reg_loss = 0
            # for param in model.parameters():
            #     reg_loss += l1_crit(param,0)
            #
            # factor = 0.0005
            # loss += factor * reg_loss

            if t%1000==1:
                print(t, loss.data[0])

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
        self.model=model


class TF_FFN(TFNet):
    """
    Fensorflow Feedforward Net
    """
    def __init__(self,seqgen,option=None):
        if type(option)==type(None):
            option=dict([])
        super(TF_FFN, self).__init__(option=option)
        self.seqgen=seqgen
        self.trainN=10000
        self.testN=2000

    def inference(self,vec_in):
        with tf.name_scope("hidden1"):
            M1 = tf.Variable(tf.random_uniform([4,4],-1.0,1.0,name="M1"))
            M1b = tf.Variable(tf.zeros([1,4],name="M1b"))
            nhu1=tf.nn.sigmoid(tf.matmul(vec_in,M1)+M1b)
        # with tf.name_scope("hidden2"):
        #     M2 = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0, name="M2"))
        #     M2b = tf.Variable(tf.zeros([1,self.Ntags], name="M2b"))
        #     nhu2 = tf.matmul(nhu1,M2)+M2b
        return nhu1,[]

    def loss(self,infvec,result):
        distvec=result-infvec
        loss=tf.norm(distvec)
        return loss,[]

    def evaluation(self, infvec, result):
        infres=tf.sign(infvec-0.5)
        distvec = infres - result
        return tf.norm(distvec)

    def get_trainingDataSet(self):
        seqs=self.seqgen.gen_ABseq(self.trainN)
        return seqs

    def do_eval(self,
                sess,
                eval_correct,
                invec,
                outvec):
        id1 = int(np.floor(np.random.rand() * 2))
        if id1 == 0:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cA"])))
            invec_init=self.seqgen.numtobin(self.seqgen.vocab["cA"][id2])
        else:
            id2 = int(np.floor(np.random.rand() * len(self.seqgen.vocab["cB"])))
            invec_init = self.seqgen.numtobin(self.seqgen.vocab["cB"][id2])
        feed_dict = {
            invec: np.array(invec_init).reshape(1,-1),
            outvec: np.array(invec_init).reshape(1,-1)
        }
        seqres=[]
        seqres.append(invec_init)
        for ii in range(self.testN):
            # nhu1,_=self.inference(invec)
            infres = tf.sign(- tf.constant([-0.5,-0.5,-0.5,-0.5]))
            res = sess.run([infres], feed_dict=feed_dict)
            feed_dict = {
                invec: np.array(res).reshape(1,-1),
                outvec: np.array(res).reshape(1,-1)
            }
            seqres.append(res)
        print(len(res))


    def fill_feed_dict(self,datain,dataout,seqs):
        resin=[]
        resout=[]
        nS=len(seqs)-1
        for ii in range(self.batch_size):
            id = int(np.floor(np.random.rand() * nS))
            resin.append(seqs[id])
            resout.append(seqs[id+1])
        feed_dict = {
            datain: np.array(resin).reshape((self.batch_size,-1)),
            dataout: np.array(resout).reshape((self.batch_size,-1))
        }
        return feed_dict

    def run(self,save_dir=None,mode=None):
        with tf.Graph().as_default(), tf.Session() as sess:
            datain = tf.placeholder(dtype=tf.float32, shape=(None, 4))
            dataout = tf.placeholder(dtype=tf.float32, shape=(None,4))
            if type(save_dir)!=type(None):
                self.resume_training(sess,save_dir,datain,dataout)
            else:
                self.run_training(sess,datain,dataout,mode=None)


