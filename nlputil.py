"""
Utility for NLP development
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import collections
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
import nltk
from nltk.corpus import brown,treebank
from nltk.tokenize import word_tokenize
import gensim
from ncautil.w2vutil import W2vUtil

__author__ = "Harry He"

class NLPutil(object):
    def __init__(self):
        print("Initializing NLPUtil")
        self.dir="tmp/"
        self.corpus=None
        self.tagged_sents = None
        self.sub_size = 100000
        self.sub_corpus = None
        self.word_to_id=None
        self.id_to_word=None
        self.word_to_cnt=None
        self.w2v_dict=None
        self.test_text=None
        self.labels=None
        self.synmat=SyntaxMat()

    def get_data(self,corpus,type=0):
        """
        Get corpus data
        :param corpus: "brown","ptb"
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
        elif corpus=="ptb":
            tmpcorp=treebank.words()
            self.corpus = []
            for item in tmpcorp:
                self.corpus.append(item.lower())
            self.sub_corpus = self.corpus[:self.sub_size]
            _,lablist=zip(*treebank.tagged_words())
            self.labels = set(lablist)
            counter = collections.Counter(lablist)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            self.labels, _ = list(zip(*count_pairs))

            self.tagged_sents = treebank.tagged_sents()
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
        words, counts = list(zip(*count_pairs))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.word_to_cnt = dict(zip(words, counts))
        bv=self.word_to_cnt[words[0]]
        for k,v in self.word_to_cnt.items():
            self.word_to_cnt[k]=1/(self.word_to_cnt[k]/bv)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="gensim",Nvac=49800):
        """
        Build word to vector lookup table
        :param mode: "pickle"
        :return:
        """
        print("Building word to vector lookup table...")
        if mode=="pickle":
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/w2vtab_opt.pickle')
            w2vu = W2vUtil()
            w2vtemp = w2vu.flloadw2v(ofile=file)
            try:
                self.w2v_dict = dict((key.decode("utf-8"), val.astype(float)) for (key, val) in w2vtemp.items())
            except AttributeError:
                self.w2v_dict = dict((key, val.astype(float)) for (key, val) in w2vtemp.items())
        elif mode=="gensim":
            assert type(self.id_to_word)!=type(None)
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/GoogleNews-vectors-negative300.bin')
            model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
            self.w2v_dict = dict([])
            skip=[]
            for ii in range(Nvac):
                try:
                    word=self.id_to_word[ii]
                    vec=model[word]
                    self.w2v_dict[word]=vec
                except:
                    skip.append(word)
            print("Except list: length "+str(len(skip)))
            print(skip)
            return self.w2v_dict

    def proj_w2v(self,w2v_dict,pM):
        """
        Project w2v dict using pM
        :param w2v_dict: dictionary of w2v
        :param pM: projection matrix
        :return: w2v_proj: dictionary of projected w2v
        """
        w2v_proj=dict([])
        assert type(w2v_dict) == OrderedDict or type(w2v_dict) == dict
        for k,v in w2v_dict.items():
            vecp=pM.dot(np.array(v))
            w2v_proj[k]=vecp
        return w2v_proj

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

    def plot_txtmat(self,data,start=0,length=1000,text=None,texty=None):
        data=np.array(data)
        assert len(data.shape) == 2
        fig,ax=plt.subplots()
        img=data[:,start:start+length]
        fig=ax.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)),np.amax(np.abs(data))))
        if text!=None:
            st,end=ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(st+0.5,end+0.5,1))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            labels=[item.get_text() for item in ax.get_xticklabels()]
            for ii in range(len(labels)):
                labels[ii]=str(text[ii+int(start)])
            ax.set_xticklabels(labels,rotation=70)
        if texty!=None:
            st, end = ax.get_ylim()
            if st<end:
                ax.yaxis.set_ticks(np.arange(st + 0.5, end + 0.5, 1))
            else:
                ax.yaxis.set_ticks(np.arange(end + 0.5, st + 0.5, 1))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            labels = [item.get_text() for item in ax.get_yticklabels()]
            for ii in range(len(labels)):
                labels[ii] = str(texty[ii + int(start)])
            ax.set_yticklabels(labels, rotation=0)
        plt.colorbar(fig)
        plt.show()

    def cal_cosdist(self,v1,v2):
        return np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)

    def cal_v2w(self,vec,numW=10,dist="cos"):
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
            dist=self.cal_cosdist(v,vec)
            # np.dot(v,vec)/np.linalg.norm(v)/np.linalg.norm(vec)
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

    def pltsne(self,numpt=500,start=0):
        w2vu = W2vUtil()
        w2vu.pltsne(self.w2v_dict,numpt=numpt,start=start)



class NLPdatap(object):
    """
    Advanced NLP data processing object
    """
    def __init__(self,NLPutilC):
        """
        A NLP Data processing class should link to a NLPutil object when initialization
        :param NLPutilC:
        """
        self.NLPutilC=NLPutilC

    def pl_conceptmap(self,nx,ny,Nwords,pM=None,mode="default"):
        """
        Plot a 2D concept map with word vectors
        :param nx: concept No. for axis x
        :param ny: concept No. for axis y
        :param Nwords: number of words shown
        :param pM: projection matrix
        :param word sampling mode: mode: default/...
        :return: null
        """
        vocabmat = []  # size (Nwords,dim)
        labels = [] # size (Nwords)
        if mode == "default":
            for ii in range(Nwords):
                wrd=self.NLPutilC.id_to_word[ii]
                vec=self.NLPutilC.w2v_dict[wrd]
                vocabmat.append(vec)
                labels.append(wrd)
        else:
            raise Exception("Other mode not yet supported")

        vocabmat=np.array(vocabmat)
        if type(pM) != type(None): # Projection matrix existing
            vocabmat=pM.dot(vocabmat.T).T

        D2mat=vocabmat[:,[nx,ny]]
        self.plot_with_labels(labels,D2mat,"Concept Dimension No. "+str(nx+1), "Concept Dimension No. "+str(ny+1))


    def plot_with_labels(self, labels, points,strx,stry):
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = points[i]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.show()

class SyntaxMat(object):
    """
    A class encoding PTB POS to COMPACT POS
    """
    def __init__(self):
        """
        PTB POS Tags
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        self.PTBPOS=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS',
                     'PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG',
                     'VBN','VBP','VBZ','WDT','WP','WP$','WRB','-NONE-']
        self.CMPPOS_base=['NN','VB','MD','JJ','RB','PRP','IN','DT','PDT','CC','CD','UH','RP','-NONE-']#14
        #'plural','proper','possessive','past','present','participle','non-3rd person singular present',
        # '3rd person singular present','comparative','superlative','wh'
        self.CMPPOS_atri=['SS','PRO','POS','PST','PTP','V12','V3','ER','EST','WH']#10
        self.PTBCMP_dict=dict([])
        self.initdic()

    def get_labvec(self,label):
        vecout=np.zeros(len(self.CMPPOS_base)+len(self.CMPPOS_atri))
        atrlst=self.PTBCMP_dict.get(label,None)
        if type(atrlst)==type(None):
            Warning("Label not found, return '-NONE-' instead")
            vecout[self.CMPPOS_base.index('-NONE-')]=1
        else:
            for item in atrlst:
                try:
                    nn=self.CMPPOS_base.index(item)
                    vecout[nn]=1
                except ValueError:
                    nn = self.CMPPOS_atri.index(item)
                    vecout[nn+len(self.CMPPOS_base)] = 1
        return vecout

        
    def initdic(self):
        """
        Initialization of transfer dict
        :return: dict
        """
        self.PTBCMP_dict['CC']=['CC']
        self.PTBCMP_dict['CD'] = ['CD']
        self.PTBCMP_dict['DT'] = ['DT']
        self.PTBCMP_dict['EX'] = ['PRP']
        self.PTBCMP_dict['FW'] = ['-NONE-']
        self.PTBCMP_dict['IN'] = ['IN']
        self.PTBCMP_dict['JJ'] = ['JJ']
        self.PTBCMP_dict['JJR'] = ['JJ','ER']
        self.PTBCMP_dict['JJS'] = ['JJ', 'EST']
        self.PTBCMP_dict['LS'] = ['CD']
        self.PTBCMP_dict['MD'] = ['MD']
        self.PTBCMP_dict['NN'] = ['NN']
        self.PTBCMP_dict['NNP'] = ['NN','PRO']
        self.PTBCMP_dict['NNPS'] = ['NN', 'PRO','SS']
        self.PTBCMP_dict['NNS'] = ['NN', 'SS']
        self.PTBCMP_dict['PDT'] = ['PDT']
        self.PTBCMP_dict['POS'] = ['POS']
        self.PTBCMP_dict['PRP'] = ['PRP']
        self.PTBCMP_dict['PRP$'] = ['PRP','POS']
        self.PTBCMP_dict['RB'] = ['RB']
        self.PTBCMP_dict['RBR'] = ['RB','ER']
        self.PTBCMP_dict['RBS'] = ['RB', 'EST']
        self.PTBCMP_dict['RP'] = ['RP']
        self.PTBCMP_dict['SYM'] = ['-NONE-']
        self.PTBCMP_dict['TO'] = ['PRP']
        self.PTBCMP_dict['UH'] = ['UH']
        self.PTBCMP_dict['VB'] = ['VB']
        self.PTBCMP_dict['VBD'] = ['VB','PST']
        self.PTBCMP_dict['VBG'] = ['VB', 'PTP']
        self.PTBCMP_dict['VBN'] = ['VB', 'PST','PTP']
        self.PTBCMP_dict['VBP'] = ['VB', 'V12']
        self.PTBCMP_dict['VBZ'] = ['VB', 'V3']
        self.PTBCMP_dict['WDT'] = ['DT', 'WH']
        self.PTBCMP_dict['WP'] = ['PRP', 'WH']
        self.PTBCMP_dict['WP$'] = ['PRP', 'WH','POS']
        self.PTBCMP_dict['WRB'] = ['RB', 'WH']
        self.PTBCMP_dict['-NONE-'] = ['-NONE-']

class SQuADutil(object):
    """
    A class assisting SQuAD task handling
    Data structure
    "version" 1.1
    "data" [list]
        "title" str
        "paragraphs" [list]
            "context" str
            "qas"  [list]
                'answers' [list]
                    'answer_start' int
                    'text' str
                'id' str
                'question' str
    """
    def __init__(self):
        self.data_train=None
        self.data_dev = None
        self.get_data()

    def get_data(self):
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_train-v1.1.json')
        json_data = open(file).read()
        self.data_train = json.loads(json_data)
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_dev-v1.1.json')
        json_data2 = open(file).read()
        self.data_dev = json.loads(json_data2)
        self.ndoc_t=len(self.data_train["data"])

    def get_rnd(self,set="train"):
        """
        Get a random example
        :return: dict["title":str,"context":str,'question':str,'answers':[str]]
        """
        res=dict([])

        if set=="dev":
            data=self.data_dev["data"]
        else:
            data=self.data_train["data"]

        # Get random example from train
        rd=np.random.random()
        ndocs=len(data)
        pdoc=int(np.floor(rd*ndocs))
        doc=self.data_train["data"][pdoc]
        res["title"]=doc["title"]

        rd2 = np.random.random()
        nparas=len(doc["paragraphs"])
        ppara=int(np.floor(rd2*nparas))
        para=doc["paragraphs"][ppara]
        res["context"] = para["context"]

        rd3 = np.random.random()
        nqas=len(para["qas"])
        pqas = int(np.floor(rd3 * nqas))
        qas=para["qas"][pqas]
        res["question"] = qas["question"]
        anslist=[]
        for item in qas['answers']:
            anslist.append(item['text'])
        res["answers"] = anslist

        return res
























