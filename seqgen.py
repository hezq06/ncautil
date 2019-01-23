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
from random import random

from ncautil.tfnlp import TFNet
from ncautil.ncalearn import *
from ncautil.ncamath import *
from ncautil.nlputil import NLPutil,GRU_Cell_Zoneout


import tensorflow as tf
import torch
import copy
from torch.autograd import Variable
from torch.nn import Parameter
from scipy.optimize import minimize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

__author__ = "Harry He"

class SeqGen(object):
    """
        Class for sequence generation wrong
        """

    def __init__(self):
        self.vocab = dict([])

    def gen_hierachical_tree_seq(self, length=50,depth_p=0.8,depth_max=3, mode_num=True):
        """
        Generate symbolic seqence using a set of hierachical tree rule.
        :param length: length of top "S"
        :param depth_p: possibility of extending a next tree
        :param depth_max: maximum depth
        :return:
        """
        wrd_2_id={
            "S":0,
            "A":1,
            "V":2,
            "D":3,
            "J":4,
            "B":5,
            "P":6
        }
        rules={
            # "Parent Node":["Leaf Node 1","Leaf Node 2", Strength]
            "S": [["S", "S", 1],["A","V",1]],
            "A":[["D","A",1],["A","A",1],["J","A",1]],
            "V":[["B","V",1]],
            "J":[["P","A",1]]
        }
        # rules = {
        #     # "Parent Node":["Leaf Node 1","Leaf Node 2", Strength]
        #     "S": [["A", "V", 1]],
        #     "A": [["D", "A", 1]],
        #     "V": [["B", "V", 1]],
        #     "J": [["P", "A", 1]]
        # }
        seqres=["S"]*length

        # def extend(IN,depth):
        #     if rules.get(IN, None) is None:
        #         OUT = IN
        #     else:
        #         extcheck=np.random.rand()
        #         if extcheck>depth_p:
        #             OUT = IN
        #         elif depth>depth_max:
        #             OUT = IN
        #         else:
        #             # We should extend syntax tree here
        #             pickrules=rules[IN]
        #             tot_str=np.sum(np.array([itemr[2] for itemr in pickrules]))
        #             pick_rule_rnd=np.random.rand()
        #             for ii_rule in pickrules:
        #                 pick_rule_rnd=pick_rule_rnd-ii_rule[2]/tot_str
        #                 if pick_rule_rnd<=0:
        #                     # Do ii_rule
        #                     OUT=[extend(ii_rule[0],depth+1),extend(ii_rule[1],depth+1)]
        #                     break
        #     return OUT
        #
        # seqout=[]
        # for iit in range(len(seqres)):
        #     seqout.append(extend(seqres[iit],0))


        for extii in range(depth_max):
            for iit in range(len(seqres)):
                item = seqres[iit]
                if rules.get(item,None) is None:
                    pass
                else:
                    extcheck=np.random.rand()
                    if extcheck>depth_p:
                        pass
                    else:
                        # We should extend syntax tree here
                        pickrules=rules[item]
                        tot_str=np.sum(np.array([itemr[2] for itemr in pickrules]))
                        pick_rule_rnd=np.random.rand()
                        for ii_rule in pickrules:
                            pick_rule_rnd=pick_rule_rnd-ii_rule[2]/tot_str
                            if pick_rule_rnd<=0:
                                # Do ii_rule
                                seqres[iit]=[ii_rule[0],ii_rule[1]]
                                break
            # Flatten seqres
            seqres_flat = [item for sublist in seqres for item in sublist]
            # print("Layer:",extii,seqres)
            seqres=seqres_flat
        if mode_num:
            seqres=[wrd_2_id[wrd] for wrd in seqres]

        return seqres

    def gen_freeseq(self,length,propl = [0.7, 0.1, 0.1, 0.1],onehot=False):
        """
        Free style stochastic seq
        dig=[0,1,2] possible digit
        prop=[0.8,0.1,0.1] possibility of each digit
        :param length:
        :param onehot:
        :return:
        """
        reseq=[]
        digl = [0, 1, 2, 3]
        propl=np.array(propl)
        propl=propl/np.sum(propl)
        assert len(digl)==len(propl)
        for iil in range(length):
            rndp = np.random.rand()
            for ii in range(len(propl)):
                rndp = rndp - propl[ii]
                if rndp < 0:
                    dig = digl[ii]
                    reseq.append(dig)
                    break
        if onehot:
            reseq=self.one_hot(reseq,length=len(digl))
            reseq = np.array(reseq)
        return reseq

    def gen_higherorderdep(self,length,onehot=False):
        """
        Artifitial high order dependency seq
        [if A+A->B, B+B->C, C+C->A else random]
        :param length:
        :param onehot:
        :return:
        """
        sq1=int(9 * np.random.rand())
        sq2 = int(9 * np.random.rand())
        cA = [0, 1, 2]
        cB = [3, 4, 5]
        cC = [6, 7, 8]
        reseq = [sq1,sq2]
        for iil in range(length):
            if reseq[-2] in cA and reseq[-1] in cA:
                sqn = int(3 * np.random.rand()) + 3
            elif reseq[-2] in cB and reseq[-1] in cB:
                sqn = int(3 * np.random.rand()) + 6
            elif reseq[-2] in cC and reseq[-1] in cC:
                sqn = int(3 * np.random.rand())
            else:
                sqn = int(9 * np.random.rand())
            reseq.append(sqn)
        if onehot:
            reseq=self.one_hot(reseq,length=9)
            reseq = np.array(reseq)
        return reseq


    def gen_softrule(self,length,prob=0.9,onehot=False):
        """
        Soft rule sequence A->B->C->A is prob, A->C->B->A is 1-prob.
        [A1: 0, A2: 1, A3: 2, B1: 3, B2: 4, B3: 5, C1: 6, C2: 7, C3: 8]
        :param length:
        :param onehot:
        :return:
        """
        sq1 = int(3 * np.random.rand())
        reseq = [sq1]
        # statem=[0,1,2]
        for iil in range(length):
            lstat=int(reseq[-1]/3)
            if np.random.rand()-prob<0: # forward
                nstat=(lstat+1)%3
            else:
                nstat = (lstat - 1) % 3
            dig=nstat*3+int(3 * np.random.rand())
            reseq.append(dig)
        if onehot:
            reseq=self.one_hot(reseq,length=9)
            reseq = np.array(reseq)
        return reseq


    def gen_longtermdep(self,length,onehot=False):
        """
        Artifitial long term dependency seq
        [A1 B3 C2 A23 B23 C13 ...]
        [A1: 0, A2: 1, A3: 2, B1: 3, B2: 4, B3: 5, C1: 6, C2: 7, C3: 8]
        :param length:
        :return:
        """
        sq1 = int(3 * np.random.rand())
        sq2 = int(3 * np.random.rand()) + 3
        sq3 = int(3 * np.random.rand()) + 6
        reseq=[sq1,sq2,sq3]
        for iil in range(length):
            cA=[0,1,2]
            cA.remove(reseq[-3])
            aA=cA[int(2*np.random.rand())]
            cB = [3, 4, 5]
            cB.remove(reseq[-2])
            aB = cB[int(2 * np.random.rand())]
            cC = [6, 7, 8]
            cC.remove(reseq[-1])
            aC = cC[int(2 * np.random.rand())]
            reseq=reseq+[aA,aB,aC]
        if onehot:
            reseq=self.one_hot(reseq,length=9)
            reseq = np.array(reseq)
        return reseq


    def gen_cellauto(self,size,length,period,delta=0.5):
        """
        Generation of cellular automaton
        :param size: world size
        :param length: sequence length
        :return: res=[]
        """
        rule301={
            "111":0, "110":0, "101":0, "100":1,"011":1,"010":1,"001":1,"000":0,
        }
        rule1101={
            "111":0,"110":1,"101":1,"100":0,"011":1,"010":1,"001":1,"000":0,
        }
        res=[]
        init=[0]*size
        init[0]=1
        res.append(init)
        context=[]

        def cellauto(input,rule):
            """
            use rull to convert input vec to output vec, periodic boundary condition
            :param input:
            :param rule:
            :return: output
            """
            assert len(input)>=3
            output=input[:]
            for ii in range(len(input)):
                item1=input[(ii-1)%len(input)]
                item2=input[ii]
                item3=input[(ii+1)%len(input)]
                ikey=str(item1)+str(item2)+str(item3)
                output[ii]=rule[ikey]
            if np.max(np.array(output))==0:
                output[0]=1
            return output

        for ii_l in range(length):
            p1=int((1+delta*(np.random.rand()-0.5)*2)*period)
            for ii_p1 in range(p1):
                nitem=cellauto(res[-1],rule301)
                res.append(nitem)
                context.append(-1)
            p2 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p2 in range(p2):
                nitem = cellauto(res[-1], rule1101)
                res.append(nitem)
                context.append(1)
        return res,context

    def gen_contextseq(self,length, period, delta=0.5):
        """
        Generate context varying sequence
        :param length: length of certain context
        :param delta: varying of period
        :return: res=[]
        """
        res=[]
        context1=[0,0,1]
        # context1 = [1,1,1,0,0]
        context2=[1,1,1,0,0]
        context3 = [1,1,0]
        for ii_l in range(length):
            p1=int((1+delta*(np.random.rand()-0.5)*2)*period)
            for ii_p1 in range(p1):
                res=res+context1
            p2 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p2 in range(p2):
                res=res+context2
            p3 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p3 in range(p3):
                res = res + context3
        return res

    def gen_contextseq_ABC(self,length, period, delta=0.5):
        """
        Generate context varying sequence
        :param length: length of certain context
        :param delta: varying of period
        :return: res=[]
        """
        tres=[]
        res=[]
        cA = [2, 3, 5, 7, 11, 13]
        cB = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15]
        nA = len(cA)
        nB = len(cB)
        context1=[0,0,1]
        contextt_1 = [-1, -1, -1]
        context2 = [0, 0, 1, 1]
        contextt_2 = [0.5, 0.5, 0.5, 0.5]
        context3 = [0, 0, 1, 1, 1]
        contextt_3 = [1, 1, 1, 1, 1]
        contextt_tracker=[]
        for ii_l in range(length):
            p1=int((1+delta*(np.random.rand()-0.5)*2)*period)
            for ii_p1 in range(p1):
                tres=tres+context1
                contextt_tracker=contextt_tracker+contextt_1
            p2 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p2 in range(p2):
                tres=tres+context2
                contextt_tracker = contextt_tracker + contextt_2
            p3 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p3 in range(p3):
                tres = tres + context3
                contextt_tracker = contextt_tracker + contextt_3
        for item in tres:
            if item==0:
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
            elif item==1:
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
            res.append(pknum)
        return res,contextt_tracker

    def gen_contextseq_two(self, lsize, length, period, delta=0.5):
        """
        Two independent context detection test
        :param length:
        :param period:
        :param delta:
        :return:
        """
        res=[]
        contextt_tracker=[]
        cUpS = [ii for ii in range(lsize)]
        contextUpS = [1 for ii in range(lsize)]
        cDnS = [lsize - ii - 1 for ii in range(lsize)]
        contextDnS = [-1 for ii in range(lsize)]
        cUpL=[]
        contextUpL=[]
        for item in cUpS:
            cUpL.append(item)
            cUpL.append(item)
            contextUpL.append(2)
            contextUpL.append(2)
        cDnL = []
        contextDnL = []
        for item in cDnS:
            cDnL.append(item)
            cDnL.append(item)
            contextDnL.append(-2)
            contextDnL.append(-2)

        for ii_l in range(length):
            pp = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            c1 = int(2 * np.random.rand())  # short / long context
            c2 = int(2 * np.random.rand())  # up / down context
            for ii_p1 in range(pp):
                if c1==0 and c2==0: #short up
                    res=res+cUpS
                    contextt_tracker=contextt_tracker+contextUpS
                if c1 == 0 and c2 == 1:  # short down
                    res = res + cDnS
                    contextt_tracker = contextt_tracker + contextDnS
                if c1 == 1 and c2 == 0:  # Long up
                    res = res + cUpL
                    contextt_tracker = contextt_tracker + contextUpL
                if c1 == 1 and c2 == 1:  # Long down
                    res = res + cDnL
                    contextt_tracker = contextt_tracker + contextDnL
        return res,contextt_tracker


    def gen_contextseq_h(self,length, period1, period2, delta=0.5):
        """
        Generate context varying sequence with 2 level hierachy
        :param length: length of certain context
        :param delta: varying of period
        :return: res=[]
        """
        res=[]
        context1=[0,0,1]
        contextb_t1=[0,0,0]
        context2 = [0, 0, 1, 1]
        contextb_t2 = [1, 1, 1, 1]
        context3=[0,0,1,1,1]
        contextb_t3 = [2,2,2,2,2]
        contextt_tracker = []
        contextb_tracker = []
        for ii_l in range(length):
            p1=int((1+delta*(np.random.rand()-0.5)*2)*period1)
            for ii_p1 in range(p1):
                p11=int((1+delta*(np.random.rand()-0.5)*2)*period2)
                for ii_p1 in range(p11):
                    res=res+context1
                    contextb_tracker=contextb_tracker+contextb_t1
                    contextt_tracker=contextt_tracker+[-1]*len(context1)
                p21 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period2)
                for ii_p2 in range(p21):
                    res=res+context2
                    contextb_tracker = contextb_tracker + contextb_t2
                    contextt_tracker = contextt_tracker + [-1] * len(context2)
                p31 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period2)
                for ii_p3 in range(p31):
                    res = res + context3
                    contextb_tracker = contextb_tracker + contextb_t3
                    contextt_tracker = contextt_tracker + [-1] * len(context3)
            p2 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period1)
            for ii_p2 in range(p2):
                p32 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period2)
                for ii_p3 in range(p32):
                    res = res + context3
                    contextb_tracker = contextb_tracker + contextb_t3
                    contextt_tracker = contextt_tracker + [1] * len(context3)
                p22 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period2)
                for ii_p2 in range(p22):
                    res = res + context2
                    contextb_tracker = contextb_tracker + contextb_t2
                    contextt_tracker = contextt_tracker + [1] * len(context2)
                p12 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period2)
                for ii_p1 in range(p12):
                    res = res + context1
                    contextb_tracker = contextb_tracker + contextb_t1
                    contextt_tracker = contextt_tracker + [1] * len(context1)
        return res,contextt_tracker,contextb_tracker

    def gen_cantorseq(self,length,depth=2):
        """
        Generate cantor factal sequence
        :param length:
        :return:
        """
        def cantor(inl):
            """
            Cantor operation on a list, 1->101,0->000
            :param inl:
            :return: outlist
            """
            res=[]
            for num in inl:
                if num==0:
                    res=res+[0,0,0]
                elif num==1:
                    res = res + [1, 0, 1]
                else:
                    raise Exception("Number not supported!")
            return res

        resseq=[]
        for iin in range(length):
            dp=int(np.random.rand()*depth)+1
            # For test
            dp=depth
            subseq=[1]
            for iid in range(dp):
                subseq=cantor(subseq)
            resseq=resseq+subseq
        return resseq

    def gen_123321seq(self,lsize,length,period,delta=0.5):
        """
        Generate 123 (clockwise) and 321 (counter-clockwise) sequences
        :param length:
        :return:
        """
        cA=[ii for ii in range(lsize)]
        contextA = [-1 for ii in range(lsize)]
        cB=[lsize-ii-1 for ii in range(lsize)]
        contextB = [1 for ii in range(lsize)]

        resseq=[]
        context_tracker = []

        for ii in range(length):
            p1 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p in range(p1):
                resseq=resseq+cA
                context_tracker=context_tracker+contextA
            p2 = int((1 + delta * (np.random.rand() - 0.5) * 2) * period)
            for ii_p in range(p2):
                resseq=resseq+cB
                context_tracker = context_tracker + contextB
        return resseq,context_tracker

    def gen_simplegrammar(self,length):
        """
        Generation of simple grammar
        :param length:
        :return:
        """
        rules={
            "A":[("A",0.4),("B",0.3),("C",0.3)],
            "B":[("B",0.5),("D",0.5)],
            "C":[("C",0.3),("D",0.7)],
            "D":[("D",0.5),("END",0.5)]
        }
        vocab={
            "A":0,
            "B":1,
            "C":2,
            "D":3,
            "END":4
        }
        reshl=[]
        for ii in range(length):
            subres=[]
            pointer="A"
            subres.append(vocab[pointer])
            while pointer!="END":
                rset=rules[pointer]
                rseed=np.random.random()
                for rl in rset:
                    rseed=rseed-rl[1]
                    if rseed<0.0:
                        pointer=rl[0]
                        subres.append(vocab[pointer])
                        break
            reshl.append(subres)

        cA = [2, 3, 5]
        cB = [7, 11, 13]
        cC = [0, 1, 4, 8, 9]
        cD = [6, 10, 12, 14]
        cEND=[15]
        comb=[cA,cB,cC,cD,cEND]

        res=[]
        for seq in reshl:
            seq2=[]
            for elem in seq:
                pickc=comb[elem]
                dig=pickc[int(np.random.random()*len(pickc))]
                seq2.append(dig)
            res.append(seq2)
        return res,reshl

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

    def gen_AAABBBseq(self, length):
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
            if ii % 6 in [0,1,2] :
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
        hlres=[]
        for ii in range(length):
            if ii % 6 in [0,1] :
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
                hlres.append(0)
            elif ii % 6 in [2,3] :
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
                hlres.append(1)
            else:
                # pick from class C
                id = int(np.floor(np.random.rand() * nC))
                pknum = cC[id]
                res.append(pknum)
                hlres.append(2)
        return res,hlres

    def gen_ABBCCCCseq(self, length, seg=False):
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
        hlres=[]
        if seg:
            res_seg=[]
            hlres_seg=[]
        for iil in range(length):
            for ii in range(7):
                if ii % 7 in [0] :
                    # pick from class A
                    id = int(np.floor(np.random.rand() * nA))
                    pknum = cA[id]
                    res.append(pknum)
                    hlres.append(0)
                elif ii % 7 in [1,2] :
                    # pick from class B
                    id = int(np.floor(np.random.rand() * nB))
                    pknum = cB[id]
                    res.append(pknum)
                    hlres.append(1)
                else:
                    # pick from class C
                    id = int(np.floor(np.random.rand() * nC))
                    pknum = cC[id]
                    res.append(pknum)
                    hlres.append(2)
            if seg:
                res_seg.append(res)
                hlres_seg.append(hlres)
                res = []
                hlres = []
        if seg:
            res=res_seg
            hlres=hlres_seg
        return res,hlres

    def gen_ABBACCCseq(self, length, seg=False):
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
        hlres=[]
        if seg:
            res_seg=[]
            hlres_seg=[]
        for iil in range(length):
            for ii in range(7):
                if ii % 7 in [0,3] :
                    # pick from class A
                    id = int(np.floor(np.random.rand() * nA))
                    pknum = cA[id]
                    res.append(pknum)
                    hlres.append(0)
                elif ii % 7 in [1,2] :
                    # pick from class B
                    id = int(np.floor(np.random.rand() * nB))
                    pknum = cB[id]
                    res.append(pknum)
                    hlres.append(1)
                else:
                    # pick from class C
                    id = int(np.floor(np.random.rand() * nC))
                    pknum = cC[id]
                    res.append(pknum)
                    hlres.append(2)
            if seg:
                res_seg.append(res)
                hlres_seg.append(hlres)
                res = []
                hlres = []
        if seg:
            res=res_seg
            hlres=hlres_seg
        return res,hlres

    def gen_ABBCCCDDDDseq(self, length):
        """
        Generate AABBseq one hot
        :param length:
        :return:
        """
        cA = [2, 3, 5 ]
        cB = [7, 11, 13]
        cC = [0, 1, 4, 8, 9]
        cD = [6, 10, 12, 14, 15]

        nA = len(cA)
        nB = len(cB)
        nC = len(cC)
        nD = len(cD)
        res = []
        hlres=[]
        for ii in range(length):
            if ii % 10 in [0] :
                # pick from class A
                id = int(np.floor(np.random.rand() * nA))
                pknum = cA[id]
                res.append(pknum)
                hlres.append(0)
            elif ii % 10 in [1,2] :
                # pick from class B
                id = int(np.floor(np.random.rand() * nB))
                pknum = cB[id]
                res.append(pknum)
                hlres.append(1)
            elif ii % 10 in [3,4,5] :
                # pick from class B
                id = int(np.floor(np.random.rand() * nC))
                pknum = cC[id]
                res.append(pknum)
                hlres.append(2)
            else:
                # pick from class C
                id = int(np.floor(np.random.rand() * nD))
                pknum = cD[id]
                res.append(pknum)
                hlres.append(3)
        return res,hlres

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

    def gen_ABCoverDEF(self,length):
        """
        Generate two layer dynamics
        :param length:
        :return:
        """
        rulesA = {
            "A": [("A", 0.2), ("B", 0.8)],
            "B": [("B", 0.3), ("C", 0.7)],
            "C": [("C", 0.4), ("A", 0.6)],
        }
        rulesB={
            "D": [("D", 0.5), ("E", 0.5)],
            "E": [("E", 0.6), ("F", 0.4)],
            "F": [("F", 0.7), ("D", 0.3)],
        }
        combindex=["AD","AE","AF","BD","BE","BF","CD","CE","CF"]

        def sample_next(rules,wrd):
            rule=rules[wrd]
            prbs=[]
            for itm in rule:
                prbs.append(itm[1])
            assert np.sum(np.array(prbs))==1
            rseed = np.random.random()
            for rl in rule:
                rseed = rseed - rl[1]
                if rseed < 0.0:
                    nitem = rl[0]
                    break
            return nitem

        resA = ["A"]
        resB = ["D"]
        ptA=resA[-1]
        ptB=resB[-1]
        res = []
        res.append(combindex.index(ptA+ptB))

        for ii in range(length):
            ptA=sample_next(rulesA,ptA)
            resA.append(ptA)
            ptB = sample_next(rulesB, ptB)
            resB.append(ptB)
            res.append(combindex.index(ptA + ptB))

        return res,resA,resB


    def gen_sinseq(self,length,k=1,dig=16,noise=0.0):
        """
        Generate sin data
        :param length:
        :return:
        """
        res = []
        phi=0
        for ii in range(length):
            phi=phi+k*(1+noise*(2*np.random.rand()-1))
            outd=(np.sin(phi)+1)/2*(dig-1)
            outl=np.floor(outd)
            assert outl+1<16
            if np.random.rand()>outd-outl:
                res.append(int(outl))
            else:
                res.append(int(outl)+1)
        return res

    def one_hot(self,num,length=16):
        if type(num) == type(1) or type(num) == np.int32:
            res = np.zeros(length)
            res[num] = 1
        else:
            res=[]
            for nn in num:
                ytemp = np.zeros(length)
                ytemp[nn] = 1
                res.append(ytemp)
        return res

    def pltsne(self,context,downsample=10):
        """
        Plot tsne clustering of context
        :param data: (length,dim)
        :return:
        """
        context=np.array(context)
        assert len(context.shape)==2
        assert len(context)>len(context[0])
        tsnetrainer = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000, method='exact')
        ebmpick = context[0:-1:downsample, :]
        print(ebmpick.shape)
        last_tsne = tsnetrainer.fit_transform(ebmpick)
        plt.figure(figsize=(18, 18))  # in inches
        for i in range(len(ebmpick)):
            x, y = last_tsne[i, :]
            plt.scatter(x, y)
        plt.show()

    def pltpca(self,context,dim=2,downsample=1,cluster=None,save=None,rnd=0.0):
        """
        Plot PCA of context
        :param context:
        :param dim:
        :param downsample:
        :return:
        """
        context = np.array(context)
        assert len(context.shape) == 2
        assert len(context) > len(context[0])
        ebmpick = context[0:-1:downsample, :]
        if type(cluster)!=type(None):
            cluster = cluster[0:-1:downsample]
        print(ebmpick.shape)
        res,pM=pca_proj(ebmpick.T, dim)
        plt.figure(figsize=(18, 18))  # in inches
        print("Plotting result...")
        if type(cluster)==type(None):
            plt.plot(res.T[0, 0] + rnd * random(), res.T[0, 1] + rnd * random(), '-ro')
            # plt.plot(res.T[:, 0], res.T[:, 1],'y')
            # for i in range(len(ebmpick)):
            #     x, y = res.T[i, :]
            #     # plt.scatter(x, y,marker='+')
            #     plt.plot(x+rnd*random(), y+rnd*random(),'y+')
            plt.plot(res.T[:, 0] + rnd * random(), res.T[:, 1] + rnd * random(), '-y+')
            plt.plot(res.T[-1, 0] + rnd * random(), res.T[-1, 1] + rnd * random(), '-bx')
        else:
            # plt.plot(res.T[:, 0], res.T[:, 1],'y')
            clist=["b","g","r","y","c","m"]
            for i in range(len(ebmpick)):
                x, y = res.T[i, :]
                c=cluster[i]
                # plt.scatter(x, y,c=clist[c],marker='+')
                plt.plot(x+rnd*random(), y+rnd*random(), str(clist[int(c)])+'+')
        # plt.show()
        plt.hist2d(res.T[:,0],res.T[:,1],bins=30)
        plt.colorbar()
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()
        return pM,res

    def plt3d(self,context,downsample=1,cluster=None):
        """
        Plot 3D of context
        :param context:
        :param dim:
        :param downsample:
        :param cluster:
        :return:
        """
        context = np.array(context)
        assert len(context.shape) == 2
        assert len(context) > len(context[0])
        ebmpick = context[0:-1:downsample, :]
        print(ebmpick.shape)
        res, pM = pca_proj(ebmpick.T, 3)
        plt.figure(figsize=(18, 18))  # in inches
        print("Plotting result...")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if type(cluster) == type(None):
            for i in range(len(res.T)):
                x, y, z = res.T[i, :]
                ax.scatter(x, y, z)
        else:
            clist = ["b", "g", "r", "y", "c", "m"]
            for i in range(len(res.T)):
                x, y, z = res.T[i, :]
                c = cluster[i]
                ax.scatter(x, y, z, c=clist[c])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def pltsne(self,context,downsample=1,cluster=None,save=None,rnd=0.0):
        """
        Plot 2D w2v graph with tsne
        Referencing part of code from: Basic word2vec example tensorflow
        :param numpt: number of points
        :return: null
        """
        assert len(context.shape) == 2
        assert len(context) > len(context[0])
        tsnetrainer = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000, method='exact')
        ebmpick = context[0:-1:downsample, :]
        if type(cluster)!=type(None):
            cluster = cluster[0:-1:downsample]
        print(ebmpick.shape)
        res = tsnetrainer.fit_transform(ebmpick)
        if type(cluster)==type(None):
            plt.plot(res[:, 0] + rnd * random(), res[:, 1] + rnd * random(), 'y+')
        else:
            clist=["b","g","r","y","c","m"]
            for i in range(len(ebmpick)):
                x, y = res[i, :]
                c=int(cluster[i])
                plt.plot(x+rnd*random(), y+rnd*random(), str(clist[c])+'+')
        plt.hist2d(res[:,0],res[:,1],bins=30)
        plt.colorbar()
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()

class RNN_PDC(torch.nn.Module):
    """
    RNN+RNN Context Layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_PDC, self).__init__()

        self.hidden_size = hidden_size
        self.r1i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r1i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.r2i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r2i2o = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden, result):
        combined1 = torch.cat((input, hidden[0]), 1)
        hidden1 = self.r1i2h(combined1)
        output = self.r1i2o(combined1)
        output = self.softmax(output)
        errin=result-torch.exp(output)
        combined2=torch.cat((errin, hidden[1]), 1)
        hidden2 = self.r2i2h(combined2)
        hadj=self.r2i2o(combined2)
        hidden1=hidden1+hadj
        return output, [hidden1,hidden2]

    def initHidden(self):
        return [Variable(torch.zeros(1,self.hidden_size), requires_grad=True),Variable(torch.zeros(1,self.hidden_size), requires_grad=True)]

class RNN_PDC2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_PDC2, self).__init__()

        self.hidden_size = hidden_size
        self.r1i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r1i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size, hidden_size,bias=False)
        self.c2r1h = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden, result):
        combined1 = torch.cat((input, hidden[0]), 1)
        hidden1 = self.r1i2h(combined1)
        output = self.r1i2o(combined1)
        output = self.softmax(output)
        errin=result-torch.exp(output)
        context=self.sigmoid(hidden[1])
        context=context+self.sigmoid(self.err2c(errin))
        hidden1=hidden1+context
        return output, [hidden1,context]

    def initHidden(self):
        return [Variable(torch.zeros(1,self.hidden_size), requires_grad=True),Variable(torch.zeros(1,self.hidden_size), requires_grad=True)]

class RNN_PDC3(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pipe_size, output_size):
        super(RNN_PDC3, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.r1i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r1i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,hidden_size, bias=False)
        self.c2r1h = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden, result):
        combined1 = torch.cat((input, hidden[0]), 1)
        hidden1 = self.r1i2h(combined1)
        output = self.r1i2o(combined1)
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = 2*self.sigmoid(hidden[2])-1
        context = context + (2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)
        hidden1 = hidden1 + context
        return output, [hidden1, errpipe, context]

    def initHidden(self):
        return [Variable(torch.zeros(1, self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.hidden_size), requires_grad=True)]

class RNN_PDC_HTANH(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_HTANH, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.r1i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r1i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.c2c = torch.nn.Linear(context_size, context_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh = torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [h0, errpipe ,context]
        :param result:
        :return:
        """
        combined1 = torch.cat((input, hidden[0]), 1)
        hidden1 = self.r1i2h(combined1)
        output = self.r1i2o(combined1)
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = hidden[2]
        context = self.hardtanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1))
        # context = self.hardtanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1) + (
        #         2 * self.sigmoid(self.c2c(context)) - 1))
        hidden1 = hidden1 + self.c2r1h(context)
        return output, [hidden1, errpipe, context]

    def initHidden(self):
        return [Variable(torch.zeros(1, self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_HTANH_WTA(torch.nn.Module):
    """
    Adding competitive learning layer by winner-takes-all process
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, concept_size,output_size):
        super(RNN_PDC_HTANH_WTA, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.concept_size = concept_size
        self.r1i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.r1i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(concept_size, hidden_size)
        # self.c2c = torch.nn.Linear(context_size, context_size)
        self.wta = torch.nn.Linear(context_size, concept_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh = torch.nn.Hardtanh()
        self.relu=torch.nn.ReLU()
        V=np.zeros((concept_size,concept_size))-1/concept_size
        for ii in range(concept_size):
            V[ii,ii]=1
        V = V / np.linalg.det(V)
        self.V=Variable(torch.from_numpy(V), requires_grad=True)
        self.V = self.V.type(torch.FloatTensor)

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [h0, errpipe ,context, concept]
        :param result:
        :return:
        """
        combined1 = torch.cat((input, hidden[0]), 1)
        hidden1 = self.r1i2h(combined1)
        output = self.r1i2o(combined1)
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = hidden[2]
        context = self.hardtanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1))
        # context = self.hardtanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1) + (
        #         2 * self.sigmoid(self.c2c(context)) - 1))
        wtalayer=self.wta(context)
        for ii in range(5):
            wtalayer = torch.matmul(self.V,wtalayer.view(-1,1))
            wtalayer = self.relu(wtalayer)
        if (wtalayer.norm(2)>0).data.numpy():
            wtalayer=wtalayer/wtalayer.norm(2)
        hidden1 = hidden1 + self.c2r1h(wtalayer.view(1,-1))
        return output, [hidden1, errpipe, context, wtalayer]

    def initHidden(self):
        return [Variable(torch.zeros(1, self.hidden_size), requires_grad=True),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True),
                Variable(torch.zeros(1, self.concept_size), requires_grad=True)]

class RNN_PDC_LSTMP2_L(torch.nn.Module):
    """
    Two phase PDC model, learning phase
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size,W=None):
        super(RNN_PDC_LSTMP2_L, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax()
        # self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()

        # High level context self-connection
        if type(W) != type(None):
            W = np.array(W)
        else:
            W=np.zeros((self.context_size,self.context_size))
        Wt = Variable(torch.from_numpy(W), requires_grad=True)
        Wt = Wt.type(torch.FloatTensor)
        self.Wt=Wt

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context=hidden[2]
        # print(context)
        # context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+0.1*(2*self.sigmoid(self.c2c(context))-1))
        context = self.hardtanh(context+self.tanh(self.err2c(errpipe.view(1,-1)))+self.tanh(torch.matmul(context,self.Wt)))
        # print(context)
        hidden1 = hidden1 * self.c2r1h(context)
        # hidden1 = hidden1 + self.c2r1h(context)
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTMP2_C(torch.nn.Module):
    """
    Two phase PDC model, consolidation phase
    """
    def __init__(self, context_size,Wt):
        super(RNN_PDC_LSTMP2_C, self).__init__()

        self.context_size = context_size
        # self.c2c = torch.nn.Linear(context_size, context_size,bias=False)
        # self.c2c.weight.data.copy_(Wt.data)
        self.c2cw = Parameter(torch.rand(context_size, context_size),requires_grad=True)
        # self.c2cw = Variable(torch.rand(context_size, context_size), requires_grad=True)
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        # wt=torch.mm()
        # wt = torch.t(self.c2cw)
        # wt = torch.matmul(self.c2cw, self.c2cw)
        wt=torch.t(self.c2cw)+self.c2cw
        output = self.tanh(torch.matmul(wt,input))
        return output


class RNN_PDC_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.c2r2h = torch.nn.Linear(context_size, hidden_size)
        # self.c2r1h = torch.nn.Linear(context_size, hidden_size, bias=False)
        # self.c2c = torch.nn.Linear(context_size, context_size)
        # self.c2c = torch.nn.Linear(context_size, context_size, bias=False)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, hidden, result, cps=1.0, gen=0.0):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context=hidden[2]
        # context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+0.1*(2*self.sigmoid(self.c2c(context))-1))
        # context = self.tanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1) + (
        #         2 * self.sigmoid(self.c2c(context)) - 1))
        context = self.hardtanh(context + (1.0-gen)*self.tanh(self.err2c(errpipe.view(1, -1))))
        hidden1 = hidden1 * self.c2r1h(context)
        # hidden1 = hidden1 + cps*self.c2r1h(context)
        # hidden1 = hidden1*self.c2r2h(context)+ cps*self.c2r1h(context)
        # hidden1 = hidden1
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTM_CELL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTM_CELL, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.c2cell = torch.nn.Linear(context_size, hidden_size)
        self.c2c = torch.nn.Linear(context_size, context_size)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context=hidden[2]
        context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+(2*self.sigmoid(self.c2c(context))-1))
        # hidden1 = hidden1 * self.c2r1h(context)
        hidden1 = hidden1 + self.c2r1h(context)
        c1 = c1 + self.c2cell(context)
        # hidden1 = hidden1
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTM_DERIV(torch.nn.Module):
    """
    Only derivative of context is sent
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTM_DERIV, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.c2c = torch.nn.Linear(context_size, context_size)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context=hidden[2]
        context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+(2*self.sigmoid(self.c2c(context))-1))
        dcontext=context-hidden[2]
        # hidden1 = hidden1 * self.c2r1h(context)
        hidden1 = hidden1 + self.c2r1h(dcontext)
        # hidden1 = hidden1
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTM_WTA(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pipe_size, context_size, concept_size, output_size):
        super(RNN_PDC_LSTM_WTA, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.concept_size = concept_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        # self.c2c = torch.nn.Linear(context_size, context_size)
        self.wta = torch.nn.Linear(context_size, concept_size)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.relu = torch.nn.ReLU()
        V = np.zeros((concept_size, concept_size)) - 1 / concept_size
        for ii in range(concept_size):
            V[ii, ii] = 1
        V = V / np.linalg.det(V)
        self.V = Variable(torch.from_numpy(V), requires_grad=True)
        self.V = self.V.type(torch.FloatTensor)

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context=hidden[2]
        context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1))
                                #+(2*self.sigmoid(self.c2c(context))-1))
        wtalayer = self.wta(context)
        for ii in range(5):
            wtalayer = torch.matmul(self.V, wtalayer.view(-1, 1))
            wtalayer = self.relu(wtalayer)
        if (wtalayer.norm(2) > 0).data.numpy(): ### ?????????
            wtalayer = wtalayer / wtalayer.norm(2)
        hidden1 = hidden1 * self.c2r1h(wtalayer.view(1,-1))
        # hidden1 = hidden1 + self.c2r1h(wtalayer.view(1,-1))
        # hidden1 = hidden1
        return output, [(hidden1,c1), errpipe, context, wtalayer]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True),
                Variable(torch.zeros(1, self.concept_size), requires_grad=True)]

class RNN_PDC_LSTMV2(torch.nn.Module):
    """
    Use signal * error gate to drive context swtich instead of error itself
    ### But not working ...
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTMV2, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errgate = torch.abs(result - torch.exp(output))
        errin = errgate*result
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context = context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)
        hidden1 = hidden1 * self.c2r1h(context)
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTM_L2(torch.nn.Module):
    """
    Two level nested PDC
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size,output_size):
        super(RNN_PDC_LSTM_L2, self).__init__()
        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.pdcl1=RNN_PDC_LSTM(input_size, hidden_size, pipe_size, context_size,output_size)
        self.pdcl2 = RNN_PDC_LSTM(context_size, hidden_size, pipe_size, context_size, context_size)

    def forward(self, input, hidden, result):
        # hidden: [(lstm h0, c0),(errpipe),context]
        output1, hidden1 = self.pdcl1(input, hidden[0], result)
        output2, hidden2 = self.pdcl2(hidden[0][2], hidden[1],hidden1[2])
        return output1,[hidden1,hidden2]

    def initHidden(self):
        hidden1=[(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]
        hidden2=[(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.context_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]
        return [hidden1,hidden2]


class RNN_PDC_LSTM_CA(torch.nn.Module):
    """
    For cellular automaton
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size,output_size):
        super(RNN_PDC_LSTM_CA, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        errin = result - torch.exp(output)/torch.max(torch.exp(output))
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context = context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)
        hidden1 = hidden1 + self.c2r1h(context)
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN_PDC_LSTM_DR(torch.nn.Module):
    """
    For distributed representation
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size,output_size):
        super(RNN_PDC_LSTM_DR, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        # self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()

    def forward(self, input, hidden, result):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, 1, self.hidden_size)
        c0 = hidden[0][1].view(1, 1, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, 1, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(self.hidden_size))
        output = 2*self.sigmoid(output)-1
        output = output/output.norm(2)
        errin = result - output
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,1:], errin.view(self.input_size,-1)),1)
        context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context = context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)
        hidden1 = hidden1 + self.c2r1h(context)
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self):
        return [(Variable(torch.zeros(1, self.hidden_size), requires_grad=True),Variable(torch.zeros(1, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, self.context_size), requires_grad=True)]

class RNN1(torch.nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(RNN1, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, y):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size), requires_grad=True)

class RNN2(torch.nn.Module):
    """
    Not very good
    """
    def __init__(self, input_size,hidden_size, mid_size,output_size):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size
        self.i2m = torch.nn.Linear(input_size + hidden_size, mid_size)
        self.m2h = torch.nn.Linear(mid_size, hidden_size)
        self.m2o = torch.nn.Linear(mid_size, output_size)
        self.sigmoid = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        mid = self.i2m(combined)
        mid=self.sigmoid(mid)
        hidden = self.m2h(mid)
        output = self.m2o(mid)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size))

class RNNR2(torch.nn.Module):
    def __init__(self, input_size,hidden1_size, hidden2_size,output_size):
        super(RNNR2, self).__init__()

        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.i2h1 = torch.nn.Linear(input_size + hidden1_size, hidden1_size)
        self.i2o = torch.nn.Linear(input_size + hidden1_size, output_size)
        self.h12h1 = torch.nn.Linear(hidden1_size + hidden2_size, hidden1_size)
        self.h12h2 = torch.nn.Linear(hidden1_size + hidden2_size, hidden2_size)
        # self.h12h1.weight = torch.nn.Parameter(torch.cat((torch.eye(hidden1_size,hidden1_size), torch.zeros(hidden1_size,hidden2_size)), 1))
        # self.h12h2.weight = torch.nn.Parameter(torch.cat((torch.zeros(hidden2_size,hidden1_size),torch.eye(hidden2_size,hidden2_size)), 1))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden, y=None):
        hidden1=hidden[0]
        hidden2=hidden[1]
        hidden1N=self.sigmoid(hidden1)
        combined2 = torch.cat((hidden1N, hidden2), 1)
        hidden2 = self.h12h2(combined2)
        hidden1 = self.h12h1(combined2)
        combined1 = torch.cat((input, hidden1), 1)
        hidden1=self.i2h1(combined1)
        output = self.i2o(combined1)
        output = self.softmax(output)
        return output, [hidden1, hidden2]

    def initHidden(self):
        return [Variable(torch.zeros(1,self.hidden1_size), requires_grad=True),Variable(torch.zeros(1,self.hidden2_size), requires_grad=True)]

class RNNR3(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNR3, self).__init__()

        self.hidden_size = hidden_size
        self.l32h3 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.l32h2 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.l22h2 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.l22h1 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.l12h1 = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.l12o = torch.nn.Linear(input_size + hidden_size, output_size)

        self.nonlinear = torch.nn.Sigmoid()
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden):
        hidden1 = hidden[0]
        hidden2 = hidden[1]
        hidden3 = hidden[2]
        hidden2N = self.nonlinear(hidden2)
        combined3 = torch.cat((hidden3, hidden2N), 1)
        hidden3 = self.l32h3(combined3)
        hidden2 = self.l32h2(combined3)

        hidden1N = self.nonlinear(hidden1)
        combined2 = torch.cat((hidden2, hidden1N), 1)
        hidden2 = self.l22h2(combined2)
        hidden1 = self.l22h1(combined2)

        combined1 = torch.cat((hidden1, input), 1)
        hidden1 = self.l12h1(combined1)
        output = self.l12o(combined1)
        output = self.softmax(output)
        return output, [hidden1, hidden2, hidden3]

    def initHidden(self):
        return [Variable(torch.zeros(1, self.hidden_size), requires_grad=True),
                Variable(torch.zeros(1, self.hidden_size), requires_grad=True),
                Variable(torch.zeros(1, self.hidden_size), requires_grad=True)]

class FRNN(torch.nn.Module):
    """
    Fully self-connected RNN for chaotic motion study
    """
    def __init__(self, n_size):
        super(FRNN, self).__init__()
        self.n_size=n_size
        self.io=torch.nn.Linear(n_size, n_size, bias=False)
        self.tanh = torch.nn.Hardtanh()

    def forward(self, state, lamda):
        state = self.tanh(state+lamda*self.io(state))
        return state

    def initState(self):
        inits=np.random.rand(self.n_size)
        state=Variable(torch.from_numpy(inits), requires_grad=True)
        state = state.type(torch.FloatTensor)
        return state


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden, y=None):
        hout, hidden = self.lstm(input.view(1, 1, self.input_size), hidden)
        output=self.h2o(hout.view(self.hidden_size))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),Variable(torch.zeros(1, 1, self.hidden_size)))

# class CNNA(torch.nn.Module):
#     def __init__(self, vec_size, kwid):
#         """
#         CNN for fractal detection
#         :param vec_size: input vector element size
#         :param kwid: convolutional window
#         """
#         super(CNNA, self).__init__()
#         self.kwid=kwid
#         self.vec_size = vec_size
#         self.kns=[]
#         for ii in range(vec_size):
#             self.kns.append(torch.nn.Linear(kwid, vec_size))
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, input):
#         for ii_step in range(len(input)-self.kwid+1):
#             for ii_conv in range(len(self.kns))
#                 convl1=
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output
#
#     def initHidden(self):
#         return Variable(torch.zeros(1,self.hidden_size))

class CNN(torch.nn.Module):
    def __init__(self, vec_size, kwid):
        """
        CNN for fractal detection
        :param vec_size: input vector element size
        :param kwid: convolutional window
        """
        super(CNN, self).__init__()
        self.kwid=kwid
        self.vec_size = vec_size
        # class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv=torch.nn.Conv1d(vec_size,vec_size,kwid,bias=False)
        # class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(kwid)
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self, input):
        conv1=self.conv(input)
        # sigm1=self.relu(conv1)
        pool1 = self.pool(conv1)
        output = self.softmax(pool1)
        return output

class PT_LSTM(object):
    """
    simple pytorch LSTM sequence learner
    """
    def __init__(self):
        self.model=None
        self.seqgen = SeqGen()
        self.seqpara=None
        self.ct=None

    def free_gen(self, length, hidden=None, init=None):
        if init==None:
            init=[0]
        seqres = []
        hiddenres = []
        lsize = self.seqpara[0]

        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init[0], length=lsize)).reshape(-1, lsize)),
                       requires_grad=False)
        seqres.append(init[0])
        xin = xin.type(torch.FloatTensor)

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        hiddenres.append(hidden)
        for ii_l in range(length):
            y_pred, hidden = self.model(xin, hidden)
            ynp = y_pred.data.numpy().reshape(lsize)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for iin in range(len(pii)):
                rndp = rndp - pii[iin]
                if rndp < 0:
                    dig = iin
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig, length=lsize)).reshape(-1, lsize)),
                           requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            if ii_l<len(init)-1:
                xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init[ii_l + 1], length=lsize)).reshape(-1, lsize)),
                             requires_grad=False)
                xin = xin.type(torch.FloatTensor)
                seqres.append(init[ii_l + 1])
            else:
                seqres.append(dig)
            hiddenres.append(hidden)
        return seqres, hiddenres


    def do_eval(self, length, hidden=None):
        seqres = []
        # self.seqpara = [lsize, period , delta]
        lsize=self.seqpara[0]
        seqs,ct = self.seqgen.gen_123321seq(lsize, length, self.seqpara[1], delta=self.seqpara[2])
        self.ct=ct

        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(seqs[0], length=lsize)).reshape(-1, lsize)),
                       requires_grad=False)
        seqres.append(seqs[0])
        xin = xin.type(torch.FloatTensor)

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        hiddenres = []
        hiddenres.append(hidden)
        for ii in range(len(seqs)-1):
            y = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(seqs[ii+1], length=lsize)).reshape(-1, lsize)),
                         requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = self.model(xin, hidden)
            ynp = y_pred.data.numpy().reshape(lsize)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig, length=lsize)).reshape(-1, lsize)),
                           requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            xin=y
            seqres.append(dig)
            hiddenres.append(hidden)

        return seqres, seqs, hiddenres

    def run(self,lsize,length,period, delta=0.5,learning_rate=1e-2,window=10,save=None):

        self.seqpara = [lsize, period , delta]

        print("Sequence generating...")

        seqs = self.seqgen.gen_123321seq(lsize, length, period, delta=delta)
        self.seqs = seqs
        # def __init__(self, input_size, concept_size, hidden_size, output_size):
        print("Learning preparation...")

        #LSTM(self, input_size, hidden_size, output_size):

        rnn = LSTM(lsize, 6, lsize)

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                # loss = loss - (torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss #+ 0.00 * l2_reg

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step = len(seqs)
        train_data = []
        for ii in range(0, step - window, window):
            hidden = rnn.initHidden()
            outputl = []
            yl = []
            xnp = np.array(self.seqgen.one_hot(seqs[ii], length=lsize))
            x = Variable(torch.from_numpy(xnp.reshape(-1, lsize)), requires_grad=True)
            output=None
            for nn in range(window):
                num2 = seqs[ii + nn + 1]
                np2 = np.array(self.seqgen.one_hot(num2, length=lsize))
                y = Variable(torch.from_numpy(np2.reshape(-1, lsize)), requires_grad=False)
                # if type(output)!=type(None):
                #     x = torch.exp(output)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden)
                outputl.append(output)
                yl.append(y)
                x = y # soft training
            loss = customized_loss(outputl, yl, rnn)

            if int(ii / 5000) != his:
                print(ii, loss.data[0])
                his = int(ii / 5000)
            train_data.append(loss.data[0])

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        x = []
        for ii in range(len(train_data)):
            x.append([ii, train_data[ii]])
        x = np.array(x)
        plt.plot(x[:, 0], x[:, 1])
        if type(save) != type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()
        self.model = rnn

class PT_2step(object):
    """
    2 step learning extension of pytorch
    """
    def __init__(self):
        self.model=None
        self.seqgen = SeqGen()

    def do_eval(self, step, hidden=None, init=0):
        seqres = []
        seqres.append(init)
        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init, length=2)).reshape(-1, 2)),
                       requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn = np.sum(vec)
            return vec / dwn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        for ii in range(step):
            y = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(self.seqs[ii], length=2)).reshape(-1, 2)),
                         requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = self.model(xin, hidden, y)
            ynp = y_pred.data.numpy().reshape(2)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig, length=2)).reshape(-1, 2)),
                           requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            seqres.append(dig)

        return seqres, self.seqs

    def run(self,length,period,delta=0.5,learning_rate=1e-2,window=10):
        print("Sequence generating...")
        seqs = self.seqgen.gen_contextseq(length, period, delta=delta)
        self.seqs = seqs
        # def __init__(self, input_size, concept_size, hidden_size, output_size):
        print("Learning preparation...")

        # rnn=RNN1(2, 5, 2)
        rnn = RNN_PDC2(2, 5, 2)

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                # loss = loss - (torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss #+ 0.00 * l2_reg

        def rnn_ffc(ii,hidden):
            """
            rnn feedforward chain
            :return:
            """
            outputl = []
            yl = []
            for nn in range(window):
                num1 = self.seqs[ii + nn]
                num2 = self.seqs[ii + nn + 1]
                np1 = np.array(self.seqgen.one_hot(num1, length=2))
                np2 = np.array(self.seqgen.one_hot(num2, length=2))
                x = Variable(torch.from_numpy(np1.reshape(-1, 2)), requires_grad=True)
                y = Variable(torch.from_numpy(np2.reshape(-1, 2)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden, y)
                outputl.append(output)
                yl.append(y)
            loss = customized_loss(outputl, yl, rnn)
            return loss

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step = len(seqs)
        train_data = []
        hidden = rnn.initHidden()
        for ii in range(0, step - window, window):

            # ### Here comes in step 1, hidden optimization (scipy minimize approach)
            #
            # def fun(para):
            #     hidden=Variable(torch.from_numpy(para.reshape(1,-1)), requires_grad=False)
            #     hidden = hidden.type(torch.FloatTensor)
            #     loss = rnn_ffc(hidden)
            #     return loss.data.numpy()[0]
            # x=hidden.data.numpy()
            # res=minimize(fun, x, method='SLSQP')
            # print(res)
            # hidden=res.x

            optimizerH = torch.optim.Adam(hidden, lr=learning_rate, weight_decay=0)
            for ii_step in range(10):
                loss = rnn_ffc(ii,hidden)
                optimizerH.zero_grad()
                loss.backward()
                optimizerH.step()

            ### Then comes step 2, synapse potimization
            loss = rnn_ffc(ii,hidden)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if int(ii / 10) != his:
                print(ii, loss.data[0])
                his = int(ii / 10)
            train_data.append(loss.data[0])


        plt.plot(train_data)
        plt.show()

        self.model = rnn

class PT_RNN_PDC(object):
    """
    RNN predictive coding testing
    """
    def __init__(self):
        self.seqgen = SeqGen()
        # self.seqs = None
        self.ct1=None
        self.ct2 = None
        self.model = None
        self.seqpara = None
        self.mlost = 1.0e9

    def ctrl_gen(self,length, ctxmat,labels=None, pick=0,lsize = 2, init=0,n_clusters=3):
        """
        Context controlled generation of sequence
        :param ctxmat:
        :param pick:
        :return:
        """
        assert len(ctxmat.shape) == 2
        assert len(ctxmat) > len(ctxmat[0])

        # Get out mean value of picked concept
        if labels==None:
            kmeans = KMeans(n_clusters=n_clusters, init="random", ).fit(ctxmat)
            labels=kmeans.labels_
        mean=np.zeros(ctxmat[0].shape)
        totn=0
        for ii in range(len(labels)):
            if labels[ii]==pick:
                totn=totn+1
                mean=mean+ctxmat[ii]
        mean=mean/totn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        # sequence generation
        hidden = self.model.initHidden()
        hidden[2][0].data.copy_(torch.from_numpy(np.array(mean)))
        seqres=[]
        seqres.append(init)
        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init, length=lsize)).reshape(-1, lsize)),
                       requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        for ii_s in range(length):
            # print(ii,len(seqs)-1)
            y = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init, length=lsize)).reshape(-1, lsize)),
                requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = self.model(xin, hidden, y, gen=1.0)
            ynp = y_pred.data.numpy().reshape(lsize)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig, length=lsize)).reshape(-1, lsize)),
                           requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            seqres.append(dig)

        nlp = NLPutil()
        nlp.plot_txtmat(np.array(seqres).reshape(1,-1))


    def do_eval(self,seqs,lsize = 2, hidden=None,init=None, cps=1.0):

        seqres = []
        init=seqs[0]
        seqres.append(init)
        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init, length=lsize)).reshape(-1, lsize)),
                       requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn = np.sum(vec)
            return vec / dwn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        hiddenres = []
        hiddenres.append(hidden)

        rnn=self.model
        rnn.eval()

        for ii_s in range(len(seqs)-1):
            # print(ii,len(seqs)-1)
            y = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(seqs[ii_s+1], length=lsize)).reshape(-1, lsize)),
                           requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = rnn(xin, hidden)
            ynp = y_pred.data.numpy().reshape(lsize)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig, length=lsize)).reshape(-1, lsize)),
                           requires_grad=False)
            # xin = xin.type(torch.FloatTensor)
            xin = y
            seqres.append(dig)
            # seqres.append(np.exp(ynp))
            hiddenres.append(hidden)

        return seqres,hiddenres

    def run(self,seqs,lsize=2,learning_rate=1e-2,window=10,save=None,guide=1.0, cps=1.0):
        # self.mlost = 1.0e99
        if type(self.model)==type(None):
            # def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
            # rnn = RNN_PDC_LSTM(lsize, 30, 3, 30, lsize)
            rnn = GRU_Cell_Zoneout(lsize,10,lsize,zoneout_rate=0.2)
        else:
            rnn=self.model
        rnn.train()

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(1)
            loss=0
            for ii in range(len(xl)):
                # loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                loss = loss - torch.sqrt(torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss #+ 0.01 * l2_reg

        def errcell_loss(errl):
            loss = 0
            for ii in range(len(errl)):
                # loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                loss = loss+torch.sum(errl[ii][:,-1]*errl[ii][:,-1])
            return loss


        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step=len(seqs)
        train_data=[]
        # mgf=Variable(torch.FloatTensor(1.0-guide), requires_grad=False)
        # gf = Variable(torch.FloatTensor(guide), requires_grad=False)
        for ii in range(0,step-window,window):
            hidden = rnn.initHidden()
            outputl=[]
            yl=[]
            errl=[]
            output=np.array(self.seqgen.one_hot(seqs[ii],length=lsize))
            output=Variable(torch.from_numpy(output.reshape(-1, lsize)), requires_grad=True)
            for nn in range(window):
                num1 = seqs[ii + nn]
                num2 = seqs[ii+nn+1]
                np1= np.array(self.seqgen.one_hot(num1,length=lsize))
                np2 = np.array(self.seqgen.one_hot(num2,length=lsize))
                x = Variable(torch.from_numpy(np1.reshape(-1, lsize)), requires_grad=True)
                # x=torch.exp(output)
                x= (1.0-guide)*torch.exp(output).type(torch.FloatTensor)+(guide*x).type(torch.FloatTensor)
                y = Variable(torch.from_numpy(np2.reshape(-1, lsize)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                # output, hidden = rnn(x, hidden, y, cps=cps)
                output, hidden = rnn(x, hidden)
                outputl.append(output)
                # hidden: [(lstm h0, c0),(errpipe),context],[(lstm h0, c0),(errpipe),context]
                # errl.append(hidden[0][1])
                # errl.append(hidden[1][1])
                yl.append(y)
            loss = customized_loss(outputl, yl, rnn)
            # loss = errcell_loss(errl)

            if int(ii / 5000) != his:
                print(ii, loss.item())
                his=int(ii / 5000)
            train_data.append(loss.item())

            if loss.item()<self.mlost:
                self.mlost=loss.item()
                self.model = copy.deepcopy(rnn)

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
        x=[]
        for ii in range(len(train_data)):
            x.append([ii,train_data[ii]])
        x=np.array(x)
        plt.plot(x[:,0],x[:,1])
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()

        # self.model = rnn


    def run_p2(self,length,period1,period2,delta=0.5,learning_rate=1e-2,conslp=200,window=10,seqs=None,lsize=2,save=None):
        """
        Trying 2 phase learning-consolication
        conslp: consolidation period
        consstp: consolidation step
        """
        print("Sequence generating...")
        self.seqpara=[period1,period2,delta]
        if type(seqs)==type(None):
            seqs, c1, c2 = self.seqgen.gen_contextseq_h(length, period1, period2, delta=delta)
            self.ct1 = c1
            self.ct2 = c2
        else:
            seqs=seqs
        print("Learning preparation...")

        csize=24
        rnnl = RNN_PDC_LSTMP2_L(lsize, 30, 3, csize, lsize)

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss - torch.sqrt(torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss #+ 0.00 * l2_reg

        def customized_loss2(x,y):
            loss = (x-y).norm(2)
            return loss


        optimizerl = torch.optim.Adam(rnnl.parameters(), lr=learning_rate, weight_decay=0)

        his = 0
        step=len(seqs)
        train_data=[]
        csdcnt=conslp
        hiddenl=[]

        for ii in range(0,step-window,window):
            hidden = rnnl.initHidden()
            outputl=[]
            yl=[]
            output=np.array(self.seqgen.one_hot(seqs[ii],length=lsize))
            output=Variable(torch.from_numpy(output.reshape(-1, lsize)), requires_grad=True)
            for nn in range(window):
                # num1 = seqs[ii + nn]
                num2 = seqs[ii+nn+1]
                # np1= np.array(self.seqgen.one_hot(num1,length=lsize))
                np2 = np.array(self.seqgen.one_hot(num2,length=lsize))
                # x = Variable(torch.from_numpy(output.reshape(-1, lsize)), requires_grad=True)
                x=torch.exp(output)
                y = Variable(torch.from_numpy(np2.reshape(-1, lsize)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnnl(x, hidden, y)
                outputl.append(output)
                # hidden: [(lstm h0, c0),(errpipe),context]
                hiddenl.append(hidden[2][0].data.numpy())
                yl.append(y)
            lossl = customized_loss(outputl, yl, rnnl)
            # loss = errcell_loss(errl)

            if int(ii / 5000) != his:
                print(ii, lossl.data[0])
                his=int(ii / 5000)
            train_data.append(lossl.data[0])

            optimizerl.zero_grad()
            lossl.backward()
            optimizerl.step()

            csdcnt=csdcnt-1
            if csdcnt<0:
                csdcnt=conslp
                print("Consolidating...")
                # start consolidation
                Wt = rnnl.Wt
                rnnc = RNN_PDC_LSTMP2_C(csize,Wt)
                optimizerc = torch.optim.Adam(rnnc.parameters(), lr=learning_rate, weight_decay=0)
                for iit in range(len(hiddenl)):
                    input= Variable(torch.from_numpy(hiddenl[iit]), requires_grad=True)
                    output=rnnc(input)
                    lossc = customized_loss2(input, output)
                    optimizerc.zero_grad()
                    lossc.backward()
                    optimizerc.step()
                hiddenl = []
                rnnl.Wt=rnnc.c2c.weight

        x=[]
        for ii in range(len(train_data)):
            x.append([ii,train_data[ii]])
        x=np.array(x)
        plt.plot(x[:,0],x[:,1])
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()
        self.model=rnnl

    def do_eval2(self,step,seqs):

        seqres = []
        lsize=len(seqs[0])
        xin = Variable(torch.from_numpy(seqs[0].reshape(-1, lsize)),requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        hidden = self.model.initHidden()
        hiddenres = []
        hiddenres.append(hidden)
        for ii in range(step):
            y = Variable(torch.from_numpy(seqs[ii+1].reshape(-1, lsize)),requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = self.model(xin, hidden, y)
            xin = y_pred
            # xin = xin.type(torch.FloatTensor)
            xin = y
            seqres.append(y_pred.data.numpy())
            # seqres.append(np.exp(ynp))
            hiddenres.append(hidden)

        return seqres,seqs,hiddenres

    def run2(self,seqs,learning_rate=1e-2,window=10,save=None):
        """
        Not learning one-hot but distributed representation
        :param seqs:
        :param learning_rate:
        :param window:
        :param save:
        :return:
        """
        seqs=np.array(seqs)
        assert len(seqs.shape)==2
        assert len(seqs)>len(seqs[0])

        lsize=len(seqs[0])
        print("Learning preparation...")
        rnn = RNN_PDC_LSTM_DR(lsize, 12, 6, 2, lsize)

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                # loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                loss = loss - torch.sum(xl[ii] * yl[ii])
            return loss #+ 0.00 * l2_reg


        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step=len(seqs)
        train_data=[]
        for ii in range(0,step-window,window):
            outputl=[]
            yl=[]
            hidden = rnn.initHidden()
            output=seqs[ii]
            output=Variable(torch.from_numpy(output.reshape(-1, lsize)), requires_grad=True)
            for nn in range(window):
                np2 = seqs[ii+nn+1]
                x=torch.exp(output)
                y = Variable(torch.from_numpy(np2.reshape(-1, lsize)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden, y)
                outputl.append(output)
                yl.append(y)
            loss = customized_loss(outputl, yl, rnn)
            # loss = errcell_loss(errl)

            if int(ii / 5000) != his:
                print(ii, loss.data[0])
                his=int(ii / 5000)
            train_data.append(loss.data[0])

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        plt.plot(train_data)
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()
        self.model=rnn

    def context_postproc(self,context,conceptN,threshold=5,ct=None):
        """
        Post processing of context for hieracical PDC
        :param context:
        :return: shrink
        """
        context=np.array(context)
        assert len(context.shape)==2
        assert context.shape[0]>context.shape[1]
        # if type(ct)!=type(None): # not actually equal
        #     assert len(context)==len(ct)
        kmeans = KMeans(n_clusters=conceptN, random_state=0).fit(context)
        # self.seqgen.pltpca(context[5000:], cluster=kmeans.labels_[5000:])
        shrink=[]
        ctsh=[]
        clab=kmeans.labels_[0]
        clablast=None
        cnt=0
        cntc=0
        for ii in range(len(context)):
            if kmeans.labels_[ii]==clab and kmeans.labels_[ii]!=clablast:
                cnt=cnt+1
                if cnt>threshold:
                    cnt=0
                    shrink.append(clab)
                    ctsh.append(ct[ii])
                    clablast=clab
            elif kmeans.labels_[ii]!=clab:
                cntc=cntc+1
                if cntc>threshold:
                    cntc=0
                    clab=kmeans.labels_[ii]
        print(shrink)
        return shrink,ctsh

    def context_postproc_downsample(self,context,downsampling=17):
        """
        Post processing of cpntext for hierachical PDC, down sampling
        :param context:
        :return: shrink
        """
        context = np.array(context)
        assert len(context.shape) == 2
        assert context.shape[0] > context.shape[1]

        shrink=[]
        for ii in range(0,len(context),downsampling):
            if np.linalg.norm(context[ii],ord=2)>0:
                shrink.append(context[ii]/np.linalg.norm(context[ii],ord=2))
            else:
                shrink.append(context[ii])
        return np.array(shrink)


class PT_RNN_PDC_CA(object):
    """
    RNN predictive coding testing for cellular automaton
    """
    def __init__(self):
        self.seqgen = SeqGen()
        self.seqs = None
        self.model = None
        self.context=None

    def do_eval(self,step,init,hidden=None):
        seqres = []
        seqres.append(init)
        size=len(init)
        xin = Variable(torch.from_numpy(np.array(init).reshape(-1, size)),
                       requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn = np.sum(vec)
            return vec / dwn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        hiddenres = []
        hiddenres.append(hidden)
        for ii in range(step):
            y = Variable(torch.from_numpy(np.array(self.seqs[ii+1]).reshape(-1, size)),requires_grad=False)
            y = y.type(torch.FloatTensor)
            y_pred, hidden = self.model(xin, hidden, y)
            ynp = torch.exp(y_pred).data.numpy().reshape(size)
            ynp=ynp/np.max(ynp)
            # for ii in range(len(ynp)):
            #     if ynp[ii]>= 0.5:
            #         ynp[ii]=1
            #     else:
            #         ynp[ii] = 0
            # xin = Variable(torch.from_numpy(ynp.reshape(-1, size)),
            #                requires_grad=False)
            # xin = xin.type(torch.FloatTensor)
            xin = y
            seqres.append(ynp)
            hiddenres.append(hidden)

        return seqres,self.seqs,hiddenres

    def run(self,size,length,period,delta=0.5,learning_rate=1e-2,window=10):
        print("Sequence generating...")
        seqs,context = self.seqgen.gen_cellauto(size,length,period,delta=delta)
        self.seqs = seqs
        self.context=context
        # def __init__(self, input_size, concept_size, hidden_size, output_size):
        print("Learning preparation...")
        # rnn=RNN(2, 5, 2)

        # rnn = RNN_PDC3(2, 5, 6, 2)

        #def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        rnn = RNN_PDC_LSTM_CA(size, 16, 3, 2, size)

        # rnn = RNN2(2, 10, 10, 2)

        # rnn = RNNR2(2, 10, 10, 2)

        # rnn = RNNR3(2, 10, 2)

        # rnn = LSTM(16, 16, 16)

        # rnn.zero_grad()

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss-torch.sum(torch.exp(xl[ii])/torch.exp(xl[ii]).norm(2) * yl[ii]/yl[ii].norm(2))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                # loss = loss - (torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss #+ 0.00 * l2_reg


        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step=len(seqs)
        train_data=[]
        num1 = seqs[ii + nn]
        np1 = np.array(num1)
        x = Variable(torch.from_numpy(np1.reshape(-1, size)), requires_grad=True)
        for ii in range(0,step-window,window):
            outputl=[]
            yl=[]
            hidden = rnn.initHidden()
            for nn in range(window):
                # num1=seqs[ii+nn]
                num2 = seqs[ii+nn+1]
                # np1= np.array(num1)
                np2 = np.array(num2)
                # x = Variable(torch.from_numpy(np1.reshape(-1, size)), requires_grad=True)
                x = torch.exp(output)
                y = Variable(torch.from_numpy(np2.reshape(-1, size)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden, y)
                outputl.append(output)
                yl.append(y)
            loss = customized_loss(outputl, yl, rnn)

            if int(ii / 1000) != his:
                print(ii, loss.data[0])
                his=int(ii / 1000)
            train_data.append(loss.data[0])

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        plt.plot(train_data)
        plt.show()

        self.model=rnn

class PT_CNN_FRAC(object):
    """
    CNN for fractal detection
    """
    def __init__(self):
        self.seqgen = SeqGen()
        self.seqs = None
        self.window=None
        self.model=None
        self.batch_size=None

    def do_eval(self):
        nn_strat = int(np.random.rand() * len(self.seqs) - self.window)
        data_train = np.array(self.seqs)[nn_strat:nn_strat + self.window]
        data_train = np.array(self.seqgen.one_hot(data_train, length=2))
        pt_data = Variable(torch.from_numpy(data_train.T.reshape(1, 2, -1)), requires_grad=True)
        pt_data=pt_data.type(torch.FloatTensor)
        output = self.model(pt_data)
        return output.data.numpy()

    def run(self, clength, step_train,batch_size=10,learning_rate=1e-2,window=500,kwid=3):
        self.batch_size=batch_size
        self.seqs = self.seqgen.gen_cantorseq(clength)
        self.window=window

        cnn = CNN(2, kwid)

        # rnn.zero_grad()

        def customized_loss(input, data_train, model):
            maxcov = Variable(torch.from_numpy(np.zeros(self.batch_size)), requires_grad=True)
            maxcov = maxcov.type(torch.FloatTensor)
            for ii_step in range(len(data_train[0][0])-len(input[0].t())+1):
                for ii_batch in range(self.batch_size):
                    npdatapck = Variable(torch.from_numpy(data_train[ii_batch][:,ii_step:ii_step+len(input[0].t())].T), requires_grad=True)
                    npdatapck = npdatapck.type(torch.FloatTensor)
                    conv=torch.sum(torch.mul(npdatapck,input[ii_batch].t()))
                    if (conv[0]>maxcov[ii_batch]).data.numpy():
                        maxcov[ii_batch]=conv
            loss=-maxcov
            return loss.sum()

        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

        # random data feeding of CNN for step_train
        his=0
        loss_tab=[]
        for ii_train in range(step_train):
            data_train_batch=[]
            for ii_batch in range(batch_size):
                nn_strat=int(np.random.rand()*(len(self.seqs)-window))
                data_train=np.array(self.seqs)[nn_strat:nn_strat+window]
                data_train = np.array(self.seqgen.one_hot(data_train, length=2))
                data_train_batch.append(data_train.T)
            pt_data=Variable(torch.from_numpy(np.array(data_train_batch)))
            pt_data = pt_data.type(torch.FloatTensor)
            output=cnn(pt_data)
            loss=customized_loss(output,data_train_batch,cnn)

            if int(ii_train / 300) != his:
                print(ii_train, loss.data[0])
                his=int(ii_train / 300)

            loss_tab.append(loss.data[0])

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
        plt.plot(loss_tab)
        plt.show()
        self.model = cnn


class PT_RNN_Cantor(object):
    def __init__(self):
        self.seqgen = SeqGen()
        self.model = None

    def do_eval(self,step,hidden=None,init=1):
        seqres = []
        seqres.append(init)
        xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(init,length=2)).reshape(-1, 2)), requires_grad=False)
        xin = xin.type(torch.FloatTensor)

        def linp(vec):
            """
            Softmax function
            :param vec:
            :return:
            """
            dwn = np.sum(vec)
            return vec / dwn

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn
        if type(hidden) == type(None):
            hidden = self.model.initHidden()
        for ii in range(step):
            y_pred, hidden = self.model(xin, hidden)
            ynp = y_pred.data.numpy().reshape(2)
            rndp = np.random.rand()
            pii = logp(ynp)
            # print(ynp)
            # print(pii)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig,length=2)).reshape(-1, 2)), requires_grad=False)
            xin = xin.type(torch.FloatTensor)
            seqres.append(dig)

        return seqres


    def run(self,clength,learning_rate=1e-2,window=10):
        seqs = self.seqgen.gen_cantorseq(clength)
        # def __init__(self, input_size, concept_size, hidden_size, output_size):
        rnn=RNN1(2, 5, 2)

        # rnn = RNN2(2, 10, 10, 2)

        # rnn = RNNR2(2, 10, 10, 2)

        # rnn = RNNR3(2, 10, 2)

        # rnn = LSTM(2, 10, 2)

        # rnn.zero_grad()

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for ii,W in enumerate(model.parameters()):
                l2_reg = l2_reg + W.norm(2)
            loss=0
            for ii in range(len(xl)):
                loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                # loss = loss - (torch.sum(torch.exp(xl[ii]) * yl[ii]))
            return loss + 0.00 * l2_reg


        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        his = 0
        step=len(seqs)
        for ii in range(0,step-window,window):
            outputl=[]
            yl=[]
            hidden = rnn.initHidden()
            for nn in range(window):
                num1=seqs[ii+nn]
                num2 = seqs[ii+nn+1]
                np1= np.array(self.seqgen.one_hot(num1,length=2))
                np2 = np.array(self.seqgen.one_hot(num2,length=2))
                x = Variable(torch.from_numpy(np1.reshape(-1, 2)), requires_grad=True)
                y = Variable(torch.from_numpy(np2.reshape(-1, 2)), requires_grad=False)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output, hidden = rnn(x, hidden)
                outputl.append(output)
                yl.append(y)
            loss = customized_loss(outputl, yl, rnn)

            if int(ii / 5000) != his:
                print(ii, loss.data[0])
                his=int(ii / 5000)

            optimizer.zero_grad()

            # for para in rnn.parameters():
            #     print(para)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        self.model=rnn

        return hidden



class RNN(torch.nn.Module):
    def __init__(self, input_size, concept_size,hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2c = torch.nn.Linear(input_size , concept_size)
        self.c2h = torch.nn.Linear(concept_size + hidden_size, hidden_size)
        self.c2o = torch.nn.Linear(concept_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, input, hidden):
        concept = self.i2c(input)
        combined = torch.cat((concept, hidden), 1)
        hidden = self.c2h(combined)
        output = self.c2o(combined)
        output = self.softmax(output)
        return output, hidden, concept

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
        recorder=[]
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
            rec = np.concatenate((y_pred.data.numpy(), hidden.data.numpy()), axis=1)
            recorder.append(rec.reshape(-1))
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
            # xin = Variable(torch.from_numpy(np.array(self.seqgen.one_hot(dig)).reshape(-1,16)), requires_grad=False)
            # xin = xin.type(torch.FloatTensor)
            xin = torch.exp(y_pred)
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
        elif mode =="AAABBB":
            vA = self.seqgen.vocab["cA"]
            vB = self.seqgen.vocab["cB"]
            for ii in range(len(seqres) - 5):
                tot = tot + 1
                int1 = seqres[ii]
                int2 = seqres[ii + 1]
                int3 = seqres[ii + 2]
                int4 = seqres[ii + 3]
                int5 = seqres[ii + 4]
                int6 = seqres[ii + 5]
                if ((int1 in vA) and (int2 in vA) and (int3 in vA) and (int4 in vB) and (int5 in vB) and (int6 in vB)) \
                        or ((int1 in vA) and (int2 in vA) and (int3 in vB) and (int4 in vB) and (int5 in vB) and (
                        int6 in vA)) \
                        or ((int1 in vA) and (int2 in vB) and (int3 in vB) and (int4 in vB) and (int5 in vA) and (
                        int6 in vA)) \
                        or ((int1 in vB) and (int2 in vB) and (int3 in vB) and (int4 in vA) and (int5 in vA) and (
                        int6 in vA)) \
                        or ((int1 in vB) and (int2 in vB) and (int3 in vA) and (int4 in vA) and (int5 in vA) and (
                        int6 in vB)) \
                        or ((int1 in vB) and (int2 in vA) and (int3 in vA) and (int4 in vA) and (int5 in vB) and (
                        int6 in vB)):
                    right = right + 1
                print(int1, int2, int3, int4, int5, int6)
            print("True rate is: " + str(right / tot))

        return np.array(recorder).T

    def run(self,step,learning_rate=5e-3,mode="AABBCC",window=30):
        if mode == "AABB":
            seqs = self.seqgen.gen_AABBseq(step)
        elif mode == "AABBCC":
            seqs = self.seqgen.gen_AABBCCseq(step)
        elif mode == "AAABBB":
            seqs = self.seqgen.gen_AAABBBseq(step)
        else:
            raise Exception("Mode not recognize.")

        # def __init__(self, input_size, concept_size, hidden_size, output_size):
        rnn=RNN1(16, 2, 16)

        # rnn.zero_grad()

        def customized_loss(xl, yl, model):
            # print(x,y)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(1)
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

