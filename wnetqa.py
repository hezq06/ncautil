"""
WordMet based Q&A system
Stanford NLP parser is also used
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

from nltk.corpus import wordnet as wn
from nltk.parse import stanford
from ncautil.nlputil import SQuADutil

__author__ = "Harry He"

class WnetQAutil(object):
    """
    Main class for WordMet based Q&A system
    """
    def __init__(self):
        self.parser=stanford.StanfordParser()
        self.squad=SQuADutil()

    def parse(self,sents):
        """
        Use Stanford NLP parser to get syntax parse tree
        :return: sentences
        """
        return self.parser.raw_parse_sents(sents)