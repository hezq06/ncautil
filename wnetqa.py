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
        self.neglist=['``','\'\'',',','.','?']
        self.NN=["NN","NNS","NNP","NNPS"]
        self.V = ["VB", "VBG", "VBN", "VBP","VBZ"]

    def parse(self,sents,draw=False):
        """
        Use Stanford NLP parser to get syntax parse tree
        :return: sentences
        """
        psents=self.parser.raw_parse_sents([sents])
        if draw==True:
            # GUI
            for line in psents:
                for sentence in line:
                    sentence.draw()
        return psents

    def get_BagConcepts(self,sents,switch="NN"):
        psents = self.parser.raw_parse_sents([sents])
        a = []
        # GUI
        for line in psents:
            for sentence in line:
                a.append(sentence)

        ptree = a[0]
        pos=ptree.pos()
        NNcons=[]
        singnn = []
        ii=0
        if switch=="NN":
            plist=self.NN
        elif switch=="V":
            plist = self.V
        while ii<len(pos):
            if pos[ii][1] in plist:
                singnn.append(pos[ii][0])
            elif len(singnn)>0:
                NNcons.append(singnn)
                singnn = []
            ii = ii + 1
        if len(singnn)>0:
            NNcons.append(singnn)
        return NNcons

    def build_ConceptFlow(self,sents):
        psents=self.parser.raw_parse_sents([sents])
        a = []
        # GUI
        for line in psents:
            for sentence in line:
                a.append(sentence)

        ptree=a[0]
        # Start concept flow building
        ileaves=ptree.leaves()
        leaves=[]
        attmat=[]
        lmon=[]
        for ii in range(len(ileaves)):
            attmat.append(list(ptree.leaf_treeposition(ii)))
            leaves.append(Concept(ileaves[ii]))
            lmon.append(len(attmat[ii]))
        print(attmat)
        print(leaves)

        def find_sibling(attmat,max_index):
            lab=False
            inds=[]
            targ=list(attmat[max_index])
            targ[-1]=0
            for ii,item in enumerate(attmat):
                item[-1]=0
                if targ==item:
                    lab=True
                    inds.append(ii)
            return lab,inds

        def contract_tree(attmat, lmon, max_index):
            tlist=list(attmat[max_index])
            tlist.pop()
            attmat[max_index]=tlist
            lmon[max_index]=lmon[max_index]-1
            return attmat, lmon


        while len(attmat)>1:
            assert len(attmat)==len(leaves)
            max_value = max(lmon)
            max_index = lmon.index(max_value)
            lab,inds=find_sibling(attmat,max_index)
            if lab:
                flist=list(attmat[inds[0]])
                flist.pop()
                subtree=ptree.__getitem__(flist)
                attmat, leaves, lmon = self.merge_concept(subtree, attmat,leaves,lmon,inds)
            else:
                attmat, lmon = contract_tree(attmat, lmon, max_index)
                leaves[max_index].pos.append(ptree.__getitem__(attmat[max_index]).label())

        return leaves[0]

    def merge_concept(self, subtree, attmat,leaves,lmon,inds):
        """
        Main method to merge sibling concepts into complex concepts
        :param ptree:
        :param attmat:
        :param leaves:
        :param lmon:
        :param inds:
        :return:
        """
        flist = list(attmat[inds[0]])
        flist.pop()
        fcon=Concept()
        text=""
        for item in subtree.leaves():
            text=text+" "+item
        fcon.text=text
        fcon.pos=subtree.label()
        for ii in inds:
            leaves[ii].output.append(fcon)
            fcon.input.append(leaves[ii])
        rattmat=[]
        rleaves=[]
        rlmon=[]

        for ii in range(len(attmat)):
            if ii==min(inds):
                rattmat.append(flist)
                rleaves.append(fcon)
                rlmon.append(len(flist))
            elif ii!=min(inds) and (ii in inds):
                pass
            else:
                rattmat.append(attmat[ii])
                rleaves.append(leaves[ii])
                rlmon.append(lmon[ii])

        return rattmat,rleaves,rlmon



class Concept(object):
    """
    Concept class
    """
    def __init__(self,text=None):
        self.text=text
        self.pos=[]
        self.input=[]
        self.output=[]

    def __repr__(self):
        return "Concept "+self.text

class Flow(object):
    """
    Flow class
    """
    def __init__(self,text,pos):
        self.text=text
        self.pos=pos
        self.fcon=None
        self.tocon=None