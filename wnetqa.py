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
import nltk
from ncautil.nlputil import SQuADutil
from nltk.tokenize import word_tokenize

__author__ = "Harry He"

class BaseBOCQA(object):
    """
    A baseline for bag of concept QA system
    """
    def __init__(self):
        self.parser = stanford.StanfordParser()
        self.qa=None
        self.sep=["IN","TO",',']
        self.NN = ["NN", "NNS", "NNP", "NNPS"]
        self.V = ["VB", "VBD", "VBP", "VBZ"]
        self.WH1=["who","when","why","where","how","how much","how many","how long"]
        self.WH2=["what","which"]

    def run(self,qa,wfdict):
        """
        Main entrance point for base BOC QA system.
        :param qa: SquadQAutil
        :return:
        """
        qcons=self.boc_seg(qa["question"])
        cntxsents=self.sent_seg(qa["context"])
        ccons=[]
        for sent in cntxsents:
            ccon=self.boc_seg(sent)
            ccons.append(ccon)
        nn=self.cal_sentatt(qcons,ccons,wfdict)
        print("Answer sentence is:")
        print(cntxsents[nn])
        f1, p, r = self.cal_scaledf1(qa['answers'][0], cntxsents[nn], wfdict)
        print(f1, p, r)

    def cal_sentatt(self,qcons,ccons,wfdict):
        """
        Calculate the attention for each sentence and return the most relevant one
        :param qcons:
        :param ccons:
        :param wfdict:
        :return:
        """
        pts=np.zeros(len(ccons))
        for qcon in qcons:
            for ii in range(len(ccons)):
                pt=0
                for testcon in ccons[ii]:
                    cpt,p,r=self.cal_scaledf1(qcon,testcon,wfdict)
                    if cpt>pt:
                        pt=cpt
                pts[ii]=pts[ii]+pt
        print(pts)
        return np.argmax(pts)

    def sent_seg(self,sent):
        sents = nltk.sent_tokenize(sent)
        return sents

    def qwh_kw(self,sent):
        """
        WH key wording of question
        :param sent:
        :return:
        """
        res=None
        type=None
        print(sent)
        sent = sent.lower()
        pos=self.pos_tagger(sent)
        for ii,item in enumerate(pos):
            if item[0] in self.WH1:
                res=item[0]
                if ii+1<len(pos):
                    test=pos[ii][0]+" "+pos[ii+1][0]
                    if test in self.WH1:
                        res=test
                break
            elif item[0] in self.WH2:
                res = item[0]
                while ii+1<len(pos):
                    ii=ii+1
                    if pos[ii][1] in self.NN:
                        type=pos[ii][0]
                        break
                break
        return res,type

    def pos_tagger(self,sent):
        """
        Calculate the part of speech tag
        :return: [(word,NN),...]
        """

        psents = self.parser.raw_parse_sents([sent])
        a = []
        # GUI
        for line in psents:
            for sentence in line:
                a.append(sentence)

        ptree = a[0]
        pos = ptree.pos()
        return pos

    def boc_seg(self,sent):
        """
        Seperate bag of concepts
        :return:
        """
        pos = self.pos_tagger(sent)
        res=[]
        bag=[]
        for item in pos:
            if item[1] in self.sep or item[1] in self.V:
                if len(bag)>0:
                    res.append(bag)
                    bag = []
                bag.append(item)
            else:
                bag.append(item)
        if len(bag)>0:
            res.append(bag)
        return res

    def cal_f1(self,base,comp):
        """
        Calculate the F1 score of two list of tokens
        F1= 2 * (precision * recall) / (precision + recall)
        :param base: ["base",...]
        :param comp: ["comp",...]
        :return: F1
        """
        if type(base)==type("string"):
            base=word_tokenize(base)
            base = [w.lower() for w in base]
        if type(comp)==type("string"):
            comp=word_tokenize(comp)
            comp = [w.lower() for w in comp]
        precision=0
        for item in comp:
            if item in base:
                precision=precision+1
        precision=precision/len(comp)

        recall=0
        for item in base:
            if item in comp:
                recall=recall+1
        recall=recall/len(base)

        F1=2 * (precision * recall) / (precision + recall)

        return F1,precision,recall

    def cal_scaledf1(self,base,comp,wfdict):
        """
        Calculate the word frequency scaled F1 score of two list of tokens
        F1= 2 * (precision * recall) / (precision + recall)
        :param base: ["base",...]
        :param comp: ["comp",...]
        :param wfdict: word frequency dictionary
        :return: F1
        """
        def attscale(word,wfdict,cut=100):
            """
            Scale word attention
            :param word: word
            :param wfdict: dict
            :param cut: threshold
            :return: att
            """
            att=np.sqrt(wfdict.get(word,cut*cut))
            return att

        if type(base)==type("string"):
            base=word_tokenize(base)
            base = [w.lower() for w in base]
        if type(comp)==type("string"):
            comp=word_tokenize(comp)
            comp = [w.lower() for w in comp]
        precision=0
        tot=0
        for item in comp:
            att=attscale(item,wfdict)
            tot=tot+att
            if item in base:
                precision=precision+att
        precision=precision/tot

        recall=0
        tot = 0
        for item in base:
            att = attscale(item, wfdict)
            tot = tot + att
            if item in comp:
                recall=recall+att
        recall=recall/tot

        F1=2 * (precision * recall) / (precision + recall)

        return F1,precision,recall

    def train_whtypedoc_prepare(self,squad):
        """
        Prepare a database to to type matching
        :param wqa: SQUAD qa utility
        :return: dict["what NN"]: dict(word-frequency)
        """
        allqa=squad.get_all()
        # for exmp in allqa:


    def train_contagger(self,boc_context,ans,threshold=0.95):
        """
        Tag a bag of concept to be answer or not
        :param boc_context: list of context sentense boc
        :param ans: answer str
        :return: [0 0 1 1 0 0 ...]
        """
        res = np.zeros(len(boc_context))
        for ii in range(len(res)):
            F1, precision, recall=self.cal_f1(ans,boc_context)
            if max(precision,recall)>threshold:
                res[ii]=1
        if max(res)==0:
            print("Tag not found")
        return res


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