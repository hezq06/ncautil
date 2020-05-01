"""
Word to vector utility
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import OrderedDict
from sklearn.manifold import TSNE

import torch
from torch.autograd import Variable

__author__ = "Harry He"

class W2vUtil(object):

    def __init__(self,dir="data"):
        self._dir = dir

        self.last_w2vtab = None
        self.last_tsne = None

    def build_w2vtab(self,w2v_skg,Nvocab,nlp):
        """

        :param w2v_skg: W2V_SkipGram object
        :param Nvocab: Number of vocabulary
        :param nlp: NLPutil object
        :return:
        """
        w2v_skg = w2v_skg.cpu()
        w2vtab = dict([])
        for ii in range(Nvocab):
            wrd = nlp.id_to_word[ii]
            vec = w2v_skg.w2v(torch.tensor(ii)).detach().numpy()
            w2vtab[wrd] = vec

    def flwritew2v(self,w2vtab,ofile="w2vtab.pickle"):
        """
        Write word to vector data to file
        :param w2vtab: ordered dictionary containing w2v data
        :param ofile: data file
        :return: null
        """
        assert type(w2vtab) == OrderedDict or type(w2vtab) == dict
        if not os.path.exists(self._dir):
            os.mkdir(self._dir)
        pickle.dump(w2vtab, open(os.path.join(self._dir, ofile), "wb"))
        self.last_w2vtab = w2vtab

    def flloadw2v(self,ofile="w2vtab.pickle"):
        """
        Load word to vector data via pickle
        :param ofile: data file
        :return: w2vtab, ordered dictionary containing w2v data
        """
        w2vtab=pickle.load(open(os.path.join(self._dir, ofile), "rb"))
        self.last_w2vtab=w2vtab
        return w2vtab

    def pltsne(self,w2vtab,numpt=500,start=0, perplexity=20, n_iter=5000):
        """
        Plot 2D w2v graph with tsne
        Referencing part of code from: Basic word2vec example tensorflow
        :param numpt: number of points
        :return: null
        """
        tsnetrainer = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, method='exact')
        ebmpick=np.array(list(w2vtab.values()))[start : start+numpt,:]
        print(ebmpick.shape)
        last_tsne = tsnetrainer.fit_transform(ebmpick)
        labels = [list(w2vtab.keys())[start+ii] for ii in range(numpt)]
        self.plot_with_labels(labels,last_tsne)
        self.last_tsne=last_tsne

    def plot_with_labels(self,labels,last_tsne):
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = last_tsne[i, :]
            plt.scatter(x, y)
            plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
        plt.show()

class W2v_COBW(object):
    def __init__(self,seqs,Ndim,window=2):
        """
        Using COBW method to do word embedding
        :param seqs: [a,b,c,d,a,c,...]
        :param Ndim: vector dimension
        :return: dic[a:[0.2,0.3,0.4,0.1],b:[...],...]
        """
        self.seqs=seqs
        self.Ndim=Ndim
        self.window=window
        self.word_to_id=None
        self.id_to_word=None
        self.w2v_dict=None

    def build_vocab(self):
        """
        Building vocabulary
        Referencing part of code from: Basic word2vec example tensorflow, reader.py
        :return:
        """
        print("Building vocabulary...")
        counter = collections.Counter(self.seqs)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, counts = list(zip(*count_pairs))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return len(words)

    def run(self,step,learning_rate=5e-3):
        Nw=self.build_vocab()
        self.w2v_dict = dict([])

        W = Variable(torch.from_numpy(np.random.random((self.Ndim,Nw))), requires_grad=True)

        def customized_loss(x_pred, x, xn):
            """
            Loss
            :param x_pred: x predict
            :param x: true x
            :param xn: x noise
            :return:
            """
            if x_pred.norm(2).data.numpy()!=0:
                x_pred=x_pred/x_pred.norm(2)
            if x.norm(2).data.numpy()!=0:
                x = x / x.norm(2)
            if xn.norm(2).data.numpy()!=0:
                xn = xn / xn.norm(2)
            distT=torch.sum(x_pred*x)
            distTN = torch.sum(x_pred *xn)
            return distTN-distT

        optimizer = torch.optim.Adam([W], lr=learning_rate, weight_decay=0)
        for iistep in range(step):
            idp=int(np.random.rand()*(len(self.seqs)-2*self.window))+self.window

            xnp1h=np.zeros(Nw)
            xnp1h[int(self.word_to_id[self.seqs[idp]])]=1
            xnp = W.data.numpy().dot(xnp1h)
            x=Variable(torch.from_numpy(xnp.reshape(self.Ndim,-1)))

            idn=int(np.random.rand()*len(self.seqs))
            xnnp1h=np.zeros(Nw)
            xnnp1h[self.word_to_id[self.seqs[idn]]] = 1
            xnnp = W.data.numpy().dot(xnnp1h)
            xn = Variable(torch.from_numpy(xnnp.reshape(self.Ndim,-1)))

            x_pred = Variable(torch.from_numpy(np.zeros(self.Ndim).reshape(self.Ndim,-1)), requires_grad=True)
            for ii in range(self.window):
                xpnp = np.zeros(Nw)
                xpnp[self.word_to_id[self.seqs[idp+ii]]] = 1
                xp = Variable(torch.from_numpy(xpnp.reshape(Nw,-1)), requires_grad=True)
                x_pred = x_pred + torch.mm(W,xp)
                xpnm = np.zeros(Nw)
                xpnm[self.word_to_id[self.seqs[idp - ii]]] = 1
                xm = Variable(torch.from_numpy(xpnm.reshape(Nw,-1)), requires_grad=True)
                x_pred = x_pred + torch.mm(W , xm)

            loss = customized_loss(x_pred, x, xn)

            if iistep%10000==1:
                print(iistep, loss.data[0])

            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        for k,v in self.word_to_id.items():
            xw = np.zeros(Nw)
            xw[v] = 1
            self.w2v_dict[k]=W.data.numpy().dot(xw)

        return self.w2v_dict


# class TFW2v


if __name__ == "__main__":

    w2vtab=OrderedDict({'a':[0,0],'b':[0,1]})
    w2vtester=W2vUtil()
    w2vtester.flwritew2v(w2vtab)
    vecres=OrderedDict()
    vecres=w2vtester.flloadw2v()
    print(vecres)



