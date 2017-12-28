"""
Simple RNN study utility
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

__author__ = "Harry He"

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

class LSTMcell(object):
    def __init__(self,size,opt=dict([])):
        self.size=size
        initlab=opt.get("init",0)
        if initlab==0:
            self.ht=np.zeros(size)
            self.yt = np.zeros(size)
            self.Wf=np.zeros((size,size))
            self.Rf = np.zeros((size, size))
            self.Bf = np.zeros(size)
            self.Wh = np.zeros((size, size))
            self.Rh = np.zeros((size, size))
            self.Bh = np.zeros(size)
            self.Wu = np.zeros((size, size))
            self.Ru = np.zeros((size, size))
            self.Bu = np.zeros(size)
            self.Wo = np.zeros((size, size))
            self.Ro = np.zeros((size, size))
            self.Bo = np.zeros(size)
        else:
            self.ht = np.random.random(5)
        self.init=False

    def init_tab(self):
        self.yttab = []
        self.httab = []
        self.yttab.append(self.yt)
        self.httab.append(self.ht)
        self.init=True
        self.thftab = []
        self.httltab = []
        self.thutab = []
        self.thotab = []

    def run(self,xt):
        if not self.init:
            self.init_tab()
        thft=sigmoid(self.Wf.dot(xt)+self.Rf.dot(self.yt)+self.Bf)
        htt=np.tanh(self.Wh.dot(xt)+self.Rh.dot(self.yt)+self.Bh)
        thut=sigmoid(self.Wu.dot(xt)+self.Ru.dot(self.yt)+self.Bu)
        self.ht=thut*htt+thft*self.ht
        thot=sigmoid(self.Wo.dot(xt)+self.Ro.dot(self.yt)+self.Bo)
        self.yt=thot*np.tanh(self.ht)
        self.yttab.append(self.yt)
        self.httab.append(self.ht)

        self.thftab.append(thft)
        self.httltab.append(htt)
        self.thutab.append(thut)
        self.thotab.append(thot)

        return self.yt,self.ht

    def pltseq(self,start=0):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey='row',figsize=(10, 5))
        for ii in range(self.size):
            ax1.plot(np.array(self.yttab)[start:,ii])
        ax1.set_title("Yt_tab")
        for ii in range(self.size):
            ax2.plot(np.array(self.httab)[start:,ii])
        ax2.set_title("Ht_tab")
        plt.show()

    def plthide(self):
        f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, sharex='col', sharey='row',figsize=(10, 10))
        for ii in range(self.size):
            ax1.plot(np.array(self.thftab)[:,ii])
        ax1.set_title("Forget Gate")
        for ii in range(self.size):
            ax2.plot(np.array(self.httltab)[:,ii])
        ax2.set_title("Candidate State")
        for ii in range(self.size):
            ax3.plot(np.array(self.thutab)[:,ii])
        ax3.set_title("Update Gate")
        for ii in range(self.size):
            ax4.plot(np.array(self.thotab)[:,ii])
        ax4.set_title("Output Gate")
        plt.show()

    def show(self):
        print("Forgot Gate:"+str(self.Wf)+str(self.Rf)+str(self.Bf))
        print("Candidate State:" + str(self.Wh) + str(self.Rh) + str(self.Bh))
        print("Update Gate:" + str(self.Wu) + str(self.Ru) + str(self.Bu))
        print("Output Gate:" + str(self.Wo) + str(self.Ro) + str(self.Bo))