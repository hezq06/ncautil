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
from collections import OrderedDict
from sklearn.manifold import TSNE

__author__ = "Harry He"

class W2vUtil(object):

    def __init__(self,dir="tmp"):
        self._dir = dir

        self.last_w2vtab = None
        self.last_tsne = None

    def flwritew2v(self,w2vtab,ofile="w2vtab.pickle"):
        """
        Write word to vector data to file
        :param w2vtab: ordered dictionary containing w2v data
        :param ofile: data file
        :return: null
        """
        assert type(w2vtab) == OrderedDict
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

    def pltsne(self,w2vtab,numpt=500,start=0):
        """
        Plot 2D w2v graph with tsne
        Referencing part of code from: Basic word2vec example tensorflow
        :param numpt: number of points
        :return: null
        """
        tsnetrainer = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
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



if __name__ == "__main__":

    w2vtab=OrderedDict({'a':[0,0],'b':[0,1]})
    w2vtester=W2vUtil()
    w2vtester.flwritew2v(w2vtab)
    vecres=OrderedDict()
    vecres=w2vtester.flloadw2v()
    print(vecres)



