"""
Utility for NLP development
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import collections
from nltk.corpus import brown
from w2vutil import W2vUtil

__author__ = "Harry He"

class NLPutil(object):
    def __init__(self):
        self.dir="tmp/"
        self.corpus=None
        self.word_to_id=None
        self.id_to_word=None

    def get_data(self,corpus,type=0):
        """
        Get corpus data
        :param corpus: "brown"
        :param type: 0
        :return:
        """
        if corpus=="brown" and type==0:
            self.corpus=brown.words(categories=brown.categories())
        print("Length of corpus: "+str(len(self.corpus)))
        print("Vocabulary of corpus: " + str(len(set(self.corpus))))

    def build_vocab(self):
        """
        Building vocabulary
        Referencing part of code from: Basic word2vec example tensorflow, reader.py
        :return:
        """
        counter = collections.Counter(self.corpus)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.id_to_word = {v: k for k, v in my_map.items()}
        print(self.word_to_id)
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="pickle",args=dict([])):
        """
        Build world to vector lookup table
        :param mode: "pickle"
        :return:
        """
        if mode=="pickle":
            pass


