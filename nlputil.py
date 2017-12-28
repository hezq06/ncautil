"""
Utility for NLP development
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from ncautil.w2vutil import W2vUtil

__author__ = "Harry He"

class NLPutil(object):
    def __init__(self):
        print("Initializing NLPUtil")
        self.dir="tmp/"
        self.corpus=None
        self.sub_size = 100000
        self.sub_corpus = None
        self.word_to_id=None
        self.id_to_word=None
        self.w2v_dict=None
        self.test_text=None

    def get_data(self,corpus,type=0):
        """
        Get corpus data
        :param corpus: "brown"
        :param type: 0
        :return:
        """
        print("Getting corpus data...")
        if corpus=="brown" and type==0:
            tmpcorp=brown.words(categories=brown.categories())
            self.corpus = []
            for item in tmpcorp:
                self.corpus.append(item.lower())
                self.sub_corpus = self.corpus[:self.sub_size]
        else:
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/'+str(corpus))
            f = open(file)
            raw = f.read()
            self.corpus = word_tokenize(raw)
            self.corpus = [w.lower() for w in self.corpus]
            self.sub_corpus = self.corpus[:self.sub_size]
        print("Length of corpus: "+str(len(self.corpus)))
        print("Vocabulary of corpus: " + str(len(set(self.corpus))))

    def init_test(self,ttext):
        self.test_text=word_tokenize(ttext)
        self.test_text = [w.lower() for w in self.test_text]
        print("Length of test text: " + str(len(self.test_text)))

    def print_txt(self,start=0,length=100,switch="sub_corpus"):
        """
        Print a passage
        :param start: start offset
        :param length: length
        :return: null
        """
        string=""
        for ii in range(length):
            string=string+self.sub_corpus[start+ii] +" "
        print(string)

    def build_vocab(self):
        """
        Building vocabulary
        Referencing part of code from: Basic word2vec example tensorflow, reader.py
        :return:
        """
        print("Building vocabulary...")
        counter = collections.Counter(self.corpus)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="pickle",args=dict([])):
        """
        Build word to vector lookup table
        :param mode: "pickle"
        :return:
        """
        print("Building word to vector lookup table...")
        if mode=="pickle":
            w2vu = W2vUtil()
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/w2vtab_opt.pickle')
            w2vtemp = w2vu.flloadw2v(ofile=file)
            self.w2v_dict = dict((key.decode("utf-8"), val.astype(float)) for (key, val) in w2vtemp.items())
        return self.w2v_dict

    def build_textmat(self,text):
        """
        Build sequecing vector of sub_corpus
        :return:
        """
        print("Building sequecing vector of text...")
        txtmat = []
        unkvec=self.w2v_dict["UNK"]
        for word in text:
            wvec=self.w2v_dict.get(word,unkvec)
            txtmat.append(wvec)
        txtmat=np.array(txtmat).T
        self.plot_txtmat(txtmat)
        return txtmat

    def plot_txtmat(self,data,start=0,length=1000,text=None):
        data=np.array(data)
        assert len(data.shape) == 2
        fig,ax=plt.subplots()
        img=data[:,start:start+length]
        fig=ax.imshow(img, cmap='seismic',clim=(-15,15))
        st,end=ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(st+0.5,end+0.5,1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        labels=[item.get_text() for item in ax.get_xticklabels()]
        for ii in range(len(labels)):
            labels[ii]=str(text[ii+int(start)])
        ax.set_xticklabels(labels,rotation=70)
        plt.colorbar(fig)
        plt.show()

    def cal_v2w(self,vec,numW,dist="cos"):
        """
        Calculate leading numW nearest using cosine distance definition
        :param vec: input vector
        :param numW: number of words returned
        :return: (word,dist) list, length is numW
        """
        res=[]
        # def cosdist(v,vec):
        #     v=np.array(v)
        #     vec=np.array(vec)
        #     dist=v.dot(vec)
        #     return dist

        for ii in range(numW):
            res.append(("NULL",-1e10))
        for (k,v) in self.w2v_dict.items():
            dist=np.dot(v,vec)/np.linalg.norm(v)/np.linalg.norm(vec)
            #dist=-np.linalg.norm(v-vec)
            if dist>=res[numW-1][1]:
                res[numW-1]=(k,dist)
                res.sort(key=lambda tup:tup[1],reverse=True)
        return res


    def write_data(self,ofile='text.txt'):
        with open(ofile,'w') as fp:
            for item in self.corpus:
                fp.write(str(item))
                fp.write(" ")

