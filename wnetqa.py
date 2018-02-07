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
import pickle
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
        self.wfdict=None
        self.sep=["IN","TO",',','CC']
        self.NN = ["NN", "NNS", "NNP", "NNPS"]
        self.V = ["VB", "VBD", "VBP", "VBZ"]
        self.WH1=["who","whom","whose","when","why","where","how","how much","how many","how long"]
        self.WH2=["what","which"]

    def run(self,qa):
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
        nn=self.cal_sentatt(qcons,ccons)
        print("Answer sentence is:")
        print(cntxsents[nn])
        f1, p, r = self.cal_scaledf1(qa['answers'][0], cntxsents[nn])
        print(f1, p, r)

    def cal_sentatt(self,qcons,ccons):
        """
        Calculate the attention for each sentence and return the most relevant one
        :param qcons:[[con],[con]]
        :param ccons:[[[con],[con]]--sent,[[con],[con]]--sent]
        :param wfdict:
        :return:
        """
        pts=np.zeros(len(ccons))
        for qcon in qcons:
            for ii in range(len(ccons)):
                pt=0
                for testcon in ccons[ii]:
                    cpt,p,r=self.cal_scaledf1(qcon,testcon)
                    if cpt>pt:
                        pt=cpt
                pts[ii]=pts[ii]+pt
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
        typ=None
        if type(sent)==type("string"):
            sent = sent.lower()
            pos=self.pos_tagger(sent)
        elif type(sent)==list:
            pos=[(w[0].lower(),w[1]) for w in sent]
        else:
            raise Exception("Type(sent) not supported")

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
                        typ=pos[ii][0]
                        break
                break
        # if res==None:
            # print("WH not found!!!")
            # print(sent)
        return res,typ

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
        if type(sent) == type("string"):
            sent = sent.lower()
            pos = self.pos_tagger(sent)
        elif type(sent) == list:
            pos = [(w[0].lower(),w[1]) for w in sent]
        else:
            raise Exception("Type(sent) not supported")

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
        else:
            base = [w.lower() for w in base]
        if type(comp)==type("string"):
            comp=word_tokenize(comp)
            comp = [w.lower() for w in comp]
        else:
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

        try:
            F1=2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            F1=0

        return F1,precision,recall


    def cal_scaledf1(self,base,comp):
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
            att=attscale(item,self.wfdict)
            tot=tot+att
            if item in base:
                precision=precision+att
        precision=precision/tot

        recall=0
        tot = 0
        for item in base:
            att = attscale(item, self.wfdict)
            tot = tot + att
            if item in comp:
                recall=recall+att
        recall=recall/tot

        if precision + recall != 0:
            F1=2 * (precision * recall) / (precision + recall)
        else:
            F1=0

        return F1,precision,recall

    def pos_remove(self,input):
        res=[]
        if type(input[0])==tuple:
            for item in input:
                res.append(item[0])
        elif type(input[0][0])==tuple:
            for litem in input:
                lres = []
                for item in litem:
                    lres.append(item[0])
                res.append(lres)
        else:
            print(input)
            raise Exception("Input format not recognized")
        return res

    def train_whtypedoc_prepare(self,squad):
        """
        Prepare a database to to type matching
        :param wqa: SQUAD qa utility
        :return: dict["what NN"]: dict(word-frequency)
        """
        def strprint(tlist):
            strg=""
            try:
                print(strg+tlist)
            except:
                for item in tlist:
                    try:
                        strg=strg+" "+item[0]
                    except:
                        for subitem in item:
                            strg = strg + " " + subitem[0]
                print(strg)

        res=dict([])
        typeflag=self.WH1+self.WH2+["UNK"]
        for entry in typeflag:
            res[entry]=[] # list at beginning
        allqa=squad.get_all()
        for ii_doc in range(len(allqa)):
            if ii_doc%1000==1:
                print("Working doc#" + str(ii_doc) + " in total " + str(len(allqa)))
        # for ii_doc in range(27,28): # for text purpose
        #     strprint(allqa[ii_doc]["context_pos"])
        #     strprint(allqa[ii_doc]["question_pos"])
        #     strprint(allqa[ii_doc]["answers"][0])
            qu=allqa[ii_doc]["question_pos"]
            wh,ty=self.qwh_kw(qu)

            # Pick up sentence if interest
            qcons = self.boc_seg(qu)
            cntxsents = allqa[ii_doc]["context_pos"]
            ccons = []
            for sent in cntxsents:
                ccon = self.boc_seg(sent)
                ccons.append(ccon)
            nn = self.cal_sentatt(qcons, ccons)
            sent=cntxsents[nn]
            bocseg_sent=self.boc_seg(sent)
            bocseg_sent=self.pos_remove(bocseg_sent)
            bocseg_ans=self.boc_seg(allqa[ii_doc]["answers_pos"][0])
            # print(allqa[ii_doc]["answers_pos"][0])
            bocseg_ans = self.pos_remove(bocseg_ans)
            ctag=self.train_contagger(bocseg_sent,bocseg_ans)
            for ii in range(len(ctag)):
                if ctag[ii]==1:
                    if type(ty)!=type(None):
                        for n, item in enumerate(bocseg_sent[ii]):
                            if item.lower() == ty.lower():
                                bocseg_sent[ii][n] = "NN"
                    try:
                        res[wh]=res[wh]+bocseg_sent[ii]
                    except KeyError:
                        res["UNK"] = res["UNK"] + bocseg_sent[ii]
        for k,v in res.items():
            counter = collections.Counter(v)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            words, counts = list(zip(*count_pairs))
            res[k] = dict(zip(words, counts))

        pickle.dump(res, open("whtype_traindata.pickle", "wb"))

        return res

    def train_contagger(self,boc_context,ans,threshold=0.95):
        """
        Tag a bag of concept to be answer or not
        :param boc_context: list of context sentense boc
        :param ans: answer str
        :return: [0 0 1 1 0 0 ...]
        """
        assert type(boc_context[0][0])==type("string")
        assert type(ans[0][0]) == type("string")
        res = np.zeros(len(boc_context))
        for ii_boc in range(len(res)):
            for ii_ans in range(len(ans)):
                F1, precision, recall=self.cal_f1(ans[ii_ans],boc_context[ii_boc])
                # print(F1, precision, recall,ans,boc_context[ii_boc] )
                if max(precision,recall)>threshold:
                    res[ii_boc]=1
        # if max(res)==0:
        #     print("Tag not found")
        return res

    def train_dataforcontagger(self,squad):
        """
        Producing training data for SVM concept tagger
        [([F1wQ,+1F1wQ,-1F1wQ,Tyscore],0/1)]
        :return:
        """
        res=[]
        whtypescoredict=pickle.load(open("whtype_traindata.pickle", "rb"))
        WFDtotv = 0
        for v in self.wfdict.values():
            WFDtotv = WFDtotv + v

        def cal_whtypescore(comp,wh,ty):
            """
            Cal wh type tf-idf like score
            :param comp:[w1 w2 w3 ...]
            :return:
            """
            for nn, item in enumerate(comp):
                assert type(item) == type("string")
                if item == ty:
                    comp[nn] = "NN"
            if type(wh)==type(None):
                wh="UNK"

            tydict = whtypescoredict[wh]
            totv = 0
            for v in tydict.values():
                totv = totv + v

            def caltfidf(item,lowcut=1e-6):
                ff=tydict.get(item,0)
                ffs=ff/totv
                ffc=self.wfdict.get(item,0)/WFDtotv
                restfidf=0
                if ffs*ffc!=0:
                    restfidf=ffs*np.log(ffs/ffc)
                elif ffs==0:
                    restfidf=0
                elif ffc==0:
                    restfidf = ffs * np.log(ffs / lowcut)
                return restfidf

            resscore=0
            for item in comp:
                sc = caltfidf(item)
                resscore=resscore+sc

            return resscore

        allqa = squad.get_all()
        for ii_doc in range(len(allqa)):
            if ii_doc%1000==1:
                print("Working doc#" + str(ii_doc) + " in total " + str(len(allqa)))
            qu = allqa[ii_doc]["question_pos"]
            wh, ty = self.qwh_kw(qu)
            # Pick up sentence if interest
            qcons = self.boc_seg(qu)
            cntxsents = allqa[ii_doc]["context_pos"]
            ccons = []
            for sent in cntxsents:
                ccon = self.boc_seg(sent)
                ccons.append(ccon)
            nn = self.cal_sentatt(qcons, ccons)
            sent = cntxsents[nn]
            bocseg_sent = self.boc_seg(sent)
            bocseg_sent = self.pos_remove(bocseg_sent)
            bocseg_ans = self.boc_seg(allqa[ii_doc]["answers_pos"][0])
            # print(allqa[ii_doc]["answers_pos"][0])
            bocseg_ans = self.pos_remove(bocseg_ans)
            ctag = self.train_contagger(bocseg_sent, bocseg_ans)
            qu=self.pos_remove(qu)
            for ii_boc in range(len(bocseg_sent)):
                cpt,p,r=self.cal_scaledf1(qu,bocseg_sent[ii_boc])
                F1wQ = cpt
                Fm1wQ = 0
                Fp1wQ = 0
                if ii_boc>0:
                    cpt, p, r = self.cal_scaledf1(qu, bocseg_sent[ii_boc-1])
                    Fm1wQ = cpt
                if ii_boc<len(bocseg_sent)-1:
                    cpt, p, r = self.cal_scaledf1(qu, bocseg_sent[ii_boc + 1])
                    Fp1wQ = cpt
                tytfidf=cal_whtypescore(bocseg_sent[ii_boc],wh,ty)
                res.append(([F1wQ,Fm1wQ,Fp1wQ,tytfidf],ctag[ii_boc]))
        return res



class WnetQAutil(object):
    """
    Main class for WordMet based Q&A system
    """
    def __init__(self):
        self.parser=stanford.StanfordParser()
        self.squad=SQuADutil()
        self.squad.get_data(mode="pickle")
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