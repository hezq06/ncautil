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
from matplotlib import cm
from collections import OrderedDict
import nltk
nltk.internals.config_java(options='-Xmx2048m')
import pickle
from nltk.corpus import brown,treebank
from nltk.tokenize import word_tokenize
from nltk.parse import stanford
import time, copy
import gensim
from ncautil.w2vutil import W2vUtil
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from scipy.optimize import minimize

import torch
from torch.autograd import Variable

from ncautil.ncalearn import pca_proj,cal_entropy,cal_kldiv

__author__ = "Harry He"

class NLPutil(object):
    def __init__(self):
        print("Initializing NLPUtil")
        self.dir="tmp/"
        self.corpus=None
        self.tagged_sents = None
        self.sub_size = 100000
        self.sub_corpus = None
        self.sub_mat=None
        self.word_to_id=None
        self.id_to_word=None
        self.word_to_cnt=None
        self.w2v_dict=None
        self.test_text=None
        self.labels=None
        self.synmat=SyntaxMat()

    def get_data(self,corpus,type=0,data=None):
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
        elif corpus=="selfgen" and data is not None:
            self.corpus=data
            self.sub_corpus=data
        else:
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/'+str(corpus))
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

    def build_vocab(self,corpus=None):
        """
        Building vocabulary
        Referencing part of code from: Basic word2vec example tensorflow, reader.py
        :return:
        """
        if type(corpus)==type(None):
            corpus=self.corpus
        else:
            self.corpus=corpus
        print("Building vocabulary...")
        counter = collections.Counter(corpus)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, counts = list(zip(*count_pairs))
        self.word_to_id = dict(zip(words, range(1,1+len(words))))
        self.word_to_id["UNK"] = 0
        self.word_to_cnt = dict(zip(words, counts))
        bv = self.word_to_cnt[words[0]]
        for k,v in self.word_to_cnt.items():
            self.word_to_cnt[k]=1/(self.word_to_cnt[k]/bv)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        print(self.id_to_word)
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="gensim",Nvac=10000):
        """
        Build word to vector lookup table
        :param mode: "pickle"
        :return:
        """
        print("Building word to vector lookup table...")
        if mode != "onehot":
            if mode=="pickle":
                file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/w2vtab_opt.pickle')
                w2vu = W2vUtil()
                w2vtemp = w2vu.flloadw2v(ofile=file)
                try:
                    model = dict((key.decode("utf-8"), val.astype(float)) for (key, val) in w2vtemp.items())
                except AttributeError:
                    model = dict((key, val.astype(float)) for (key, val) in w2vtemp.items())
            elif mode=="gensim": # Google pretrained w2v_tab
                assert type(self.id_to_word)!=type(None)
                file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/GoogleNews-vectors-negative300.bin')
                model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
            elif mode=="gensim_raw":
                file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/GoogleNews-vectors-negative300.bin')
                model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
                return model
            self.w2v_dict = dict([])
            skip = []
            for ii in range(Nvac):
                try:
                    word = self.id_to_word[ii]
                    vec = model[word]
                    self.w2v_dict[word] = vec
                except:
                    skip.append(word)
            print("Except list: length " + str(len(skip)))
            print(skip)
            self.w2v_dict["UNK"] = np.zeros(len(self.w2v_dict[self.id_to_word[10]]))
        else:
            if len(self.id_to_word)>200:
                raise Exception("Too much words for onehot representation.")
            self.w2v_dict=dict([])
            for ii in range(len(self.id_to_word)):
                vec=np.zeros(len(self.id_to_word))
                vec[ii]=1.0
                self.w2v_dict[self.id_to_word[ii]]=vec
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
        unkvec=self.w2v_dict.get("UNK",None)
        for word in text:
            wvec=self.w2v_dict.get(word,unkvec)
            txtmat.append(wvec)
        txtmat=np.array(txtmat)
        # self.plot_txtmat(txtmat.T)
        return txtmat

    def plot_txtmat(self,data,start=0,length=1000,text=None,texty=None,title=None,save=None,origin='lower'):
        data=np.array(data)
        assert len(data.shape) == 2
        fig,ax=plt.subplots()
        img=data[:,start:start+length]
        fig=ax.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)),np.amax(np.abs(data))), origin=origin)
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
        if type(title) != type(None):
            plt.title(title)
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()

    def plot_textshade(self,text,data,start=0,length=100,cper_line=100,gamma=0.3):
        data = np.array(data)
        assert len(data.shape) == 1
        # assert len(data) == len(text)
        datap = data[start:start + length]
        textp = text[start:start + length]
        shift=0
        ccnt=0
        cmax=np.max(datap)
        cmin = np.min(datap)
        # cmin = -cmax
        cmax=1.0
        cmin=0.0

        for iiw in range(len(textp)):
            wrdp=textp[iiw]
            clp=datap[iiw]
            cpck=cm.seismic(int((clp-cmin)/(cmax-cmin)*256))
            nc=len(str(wrdp))+2
            plt.text(1/cper_line*ccnt, 1.0-shift*0.1, wrdp, size=10,
                     ha="left", va="center",
                     bbox=dict(ec=(1., 1., 1.),
                               fc=(cpck[0],cpck[1],cpck[2],gamma),
                               )
                     )
            ccnt=ccnt+nc
            if ccnt>cper_line:
                ccnt=0
                shift=shift+1
        plt.draw()
        plt.axis('off')
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
        # self.get_data()
        try:
            self.parser = stanford.StanfordParser()
        except:
            print("Stanford parser not working!!!")
        path = '/Users/zhengqihe/HezMain/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models'
        os.environ['STANFORD_MODELS'] = path
        self.pos = StanfordPOSTagger('english-caseless-left3words-distsim.tagger',
                                path_to_jar=path + '/../stanford-postagger.jar')

    def get_data(self,mode="pickle"):
        if mode=="json":
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_train-v1.1.json')
            json_data = open(file).read()
            self.data_train = json.loads(json_data)
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_dev-v1.1.json')
            json_data2 = open(file).read()
            self.data_dev = json.loads(json_data2)
        elif mode=="pickle":
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_train-v1.1.pickle')
            self.data_train = pickle.load(open(file, "rb"))
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_dev-v1.1.pickle')
            self.data_dev = pickle.load(open(file, "rb"))

    def save_data(self):
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_train-v1.1.pickle')
        pickle.dump(self.data_train, open(file, "wb"))
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/SQuAD_dev-v1.1.pickle')
        pickle.dump(self.data_dev, open(file, "wb"))

    def get_rnd(self,set="train"):
        """
        Get a random example
        :param set: "train","dev"
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
        try:
            res["context_pos"] = para["context_pos"]
        except:
            print("No context_pos found")

        rd3 = np.random.random()
        nqas=len(para["qas"])
        pqas = int(np.floor(rd3 * nqas))
        qas=para["qas"][pqas]
        res["question"] = qas["question"]
        try:
            res["question_pos"] = qas["question_pos"]
        except:
            print("No question_pos found")
        anslist=[]
        for item in qas['answers']:
            anslist.append(item['text'])
        res["answers"] = anslist
        try:
            res["answers_pos"] = qas["answers_pos"]
        except:
            print("No answers_pos found")

        return res

    def get_all(self,set="train"):
        """
        Get out all qa examples into a list
        :param set: "train","dev"
        :return:[res]
        """
        resall = []

        if set == "dev":
            data = self.data_dev["data"]
        else:
            data = self.data_train["data"]

        ndocs = len(data)

        for ii_docs in range(ndocs):
            # print("Docs "+str(ii_docs)+" in total "+str(ndocs)+".")
            doc = self.data_train["data"][ii_docs]
            title = doc["title"]
            nparas = len(doc["paragraphs"])
            for ii_para in range(nparas):
                para = doc["paragraphs"][ii_para]
                try:
                    context=para["context_pos"]
                except:
                    context=para["context"]
                    print(ii_docs)
                    raise Exception("Uncomplete doc #"+str(ii_docs))
                nqas = len(para["qas"])
                for ii_qas in range(nqas):
                    qa = para["qas"][ii_qas]
                    res=dict([])
                    res["title"]=title
                    res["context_pos"] = context
                    try:
                        res["question_pos"] = qa["question_pos"]
                    except:
                        res["question_pos"] = qa["question"]
                    try:
                        res["answers_pos"] = qa['answers_pos']
                    except:
                        res["answers_pos"] = qa['answers']
                    anslist = []
                    for item in qa['answers']:
                        anslist.append(item['text'])
                    res["answers"] = anslist
                    resall.append(res)
        return resall

    def preprocess_format(self):
        """
        Adjusting
        :return:
        """
        pass

    def preprocess_pos(self):
        """
        Doing pos preprocessiong
        :return:
        """
        data = self.data_train["data"]
        ndocs = len(data)
        print("Pre POSing training data.")
        for ii_docs in range(ndocs):
            print("Posing doc#" + str(ii_docs) + " in total " + str(ndocs))
            doc = self.data_train["data"][ii_docs]
            nparas = len(doc["paragraphs"])
            for ii_para in range(nparas):
                print("    For doc#" + str(ii_docs) + " POSing para#" + str(ii_para) + " in total " + str(nparas))
                para = doc["paragraphs"][ii_para]
                context = para["context"]
                sents = self.sent_seg(context)
                context_pos = []
                for sent in sents:
                    pos = self.pos_tagger(sent)
                    context_pos.append(pos)
                self.data_train["data"][ii_docs]["paragraphs"][ii_para]["context_pos"] = context_pos
                nqas = len(para["qas"])
                for ii_qas in range(nqas):
                    qas = para["qas"][ii_qas]
                    pos = self.pos_tagger(qas["question"])
                    self.data_train["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["question_pos"] = pos
                    self.data_train["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"] = []
                    for item in qas['answers']:
                        pos = self.pos_tagger(item['text'])
                        self.data_train["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"].append(
                            pos)

        data = self.data_dev["data"]
        ndocs = len(data)
        print("Pre POSing developmet data.")
        for ii_docs in range(ndocs):
            print("Posing doc#" + str(ii_docs) + " in total " + str(ndocs))
            doc = self.data_dev["data"][ii_docs]
            nparas = len(doc["paragraphs"])
            for ii_para in range(nparas):
                print("    For doc#" + str(ii_docs) + " POSing para#" + str(ii_para) + " in total " + str(nparas))
                para = doc["paragraphs"][ii_para]
                context = para["context"]
                sents = self.sent_seg(context)
                context_pos = []
                for sent in sents:
                    pos = self.pos_tagger(sent)
                    context_pos.append(pos)
                self.data_dev["data"][ii_docs]["paragraphs"][ii_para]["context_pos"] = context_pos
                nqas = len(para["qas"])
                for ii_qas in range(nqas):
                    qas = para["qas"][ii_qas]
                    pos = self.pos_tagger(qas["question"])
                    self.data_dev["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["question_pos"] = pos
                    self.data_dev["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"] = []
                    for item in qas['answers']:
                        pos = self.pos_tagger(item['text'])
                        self.data_dev["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"].append(pos)


    def preprocess_pos_fast(self,chunck_size=5000,doc_start=None,doc_end=None,switch="train"):
        """
        Doing pos preprocessiong
        :return:
        """
        ########### Training set

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        mismatch=[]
        print("Start collecting data in "+switch+" set! " )
        data = self.data_train
        if switch=="development":
            data = self.data_dev
        ndocs = len(data["data"])
        if doc_start==None:
            doc_start=0
        if doc_end==None:
            doc_end=ndocs-1
        if doc_start>ndocs:
            doc_start=ndocs
        if doc_end>ndocs:
            doc_end=ndocs-1
        for ii_docs in range(doc_start,doc_end+1):
            print("Working doc#" + str(ii_docs) + " in total " + str(doc_start)+"~"+str(doc_end))

            wlall = []
            doc = data["data"][ii_docs]
            nparas = len(doc["paragraphs"])
            for ii_para in range(nparas):
                para = doc["paragraphs"][ii_para]
                context=para["context"]
                sents=self.sent_seg(context)
                context_pos=[]
                for sent in sents:
                    pos=word_tokenize(sent)
                    context_pos.append(pos)
                    wlall=wlall+pos
                data["data"][ii_docs]["paragraphs"][ii_para]["context_pos"]=context_pos
                nqas = len(para["qas"])
                for ii_qas in range(nqas):
                    qas = para["qas"][ii_qas]
                    pos = word_tokenize(qas["question"])
                    wlall = wlall + pos
                    data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["question_pos"] = pos
                    data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"] = []
                    for item in qas['answers']:
                        pos = word_tokenize(item['text'])
                        wlall = wlall + pos
                        data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"].append(pos)

            print("Total words to parse in training set: " + str(len(wlall)))
            print("Parsing......")
            if len(wlall)>chunck_size:
                wlall_chuncks=chunks(wlall,chunck_size)
                posall=[]
                for wlall_chunck in wlall_chuncks:
                    posall_chunck = self.pos_tagger_fast(wlall_chunck)
                    posall=posall+posall_chunck
            else:
                posall=self.pos_tagger_fast(wlall)
            print("Answer writing.")

            ptall=0

            print("Answering doc#" + str(ii_docs) + " in total " + str(doc_start)+"~"+str(doc_end))
            doc = data["data"][ii_docs]
            nparas = len(doc["paragraphs"])
            for ii_para in range(nparas):
                para = doc["paragraphs"][ii_para]
                context=para["context"]
                sents=self.sent_seg(context)
                context_pos=[]
                for sent in sents:
                    pos=word_tokenize(sent)
                    sent_pos=[]
                    for ii_w,wrd in enumerate(pos):
                        sent_pos.append(posall[ptall])
                        if posall[ptall][0] != wrd:
                            print("Mismatch: "+str(posall[ptall][0])+", "+str(wrd))
                            mismatch.append((posall[ptall][0],wrd))
                            if ii_w + 1 < len(pos):
                                if posall[ptall+1][0] == pos[ii_w+1]:
                                    pass
                                elif posall[ptall][0] == pos[ii_w+1]:
                                    ptall = ptall - 1
                                elif posall[ptall+2][0] == pos[ii_w+1]:
                                    ptall = ptall + 1
                                else:
                                    raise Exception("Adjustment failed")
                        ptall = ptall + 1
                    context_pos.append(sent_pos)
                    data["data"][ii_docs]["paragraphs"][ii_para]["context_pos"]=context_pos
                nqas = len(para["qas"])
                for ii_qas in range(nqas):
                    qas = para["qas"][ii_qas]
                    pos = word_tokenize(qas["question"])
                    sent_pos = []
                    for ii_w, wrd in enumerate(pos):
                        sent_pos.append(posall[ptall])
                        if posall[ptall][0] != wrd:
                            print("Mismatch: " + str(posall[ptall][0]) + ", " + str(wrd))
                            mismatch.append((posall[ptall][0], wrd))
                            if ii_w + 1 < len(pos):
                                if posall[ptall+1][0] == pos[ii_w+1]:
                                    pass
                                elif posall[ptall][0] == pos[ii_w+1]:
                                    ptall = ptall - 1
                                elif posall[ptall+2][0] == pos[ii_w+1]:
                                    ptall = ptall + 1
                                else:
                                    raise Exception("Adjustment failed")
                        ptall = ptall + 1
                    data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["question_pos"] = sent_pos
                    data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"] = []
                    for item in qas['answers']:
                        pos = word_tokenize(item['text'])
                        sent_pos = []
                        for ii_w, wrd in enumerate(pos):
                            sent_pos.append(posall[ptall])
                            if posall[ptall][0] != wrd:
                                print("Mismatch: " + str(posall[ptall][0]) + ", " + str(wrd))
                                mismatch.append((posall[ptall][0], wrd))
                                if ii_w + 1<len(pos):
                                    if posall[ptall + 1][0] == pos[ii_w + 1]:
                                        pass
                                    elif posall[ptall][0] == pos[ii_w + 1]:
                                        ptall = ptall - 1
                                    elif posall[ptall + 2][0] == pos[ii_w + 1]:
                                        ptall = ptall + 1
                                    else:
                                        raise Exception("Adjustment failed")
                            ptall = ptall + 1
                        data["data"][ii_docs]["paragraphs"][ii_para]["qas"][ii_qas]["answers_pos"].append(sent_pos)

        print("Mismatch summary:" +str(len(mismatch)))
        print(mismatch)

    def sent_seg(self,sent):
        sents = nltk.sent_tokenize(sent)
        return sents

    def pos_tagger(self,sent):
        """
        Calculate the part of speech tag
        :return: [(word,NN),...]
        """
        if type(sent)==list:
            sent=""
            for w in sent:
                sent=sent+" "+w
        psents = self.parser.raw_parse_sents([sent])
        a = []
        # GUI
        for line in psents:
            for sentence in line:
                a.append(sentence)

        ptree = a[0]
        pos = ptree.pos()
        return pos

    def pos_tagger_fast(self,sent):
        if type(sent)==type("string"):
            sent=word_tokenize(sent)
            sent = [w.lower() for w in sent]
        def ini_pos():
            path = '/Users/zhengqihe/HezMain/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models'
            os.environ['STANFORD_MODELS'] = path
            pos = StanfordPOSTagger('english-bidirectional-distsim.tagger',
                                         path_to_jar=path + '/../stanford-postagger.jar')
            return pos
        # pos=ini_pos()
        res=self.pos.tag(sent)
        return res

class Sememe_NLP(object):
    """
    Unsupervised learning of word sememe
    """
    def __init__(self,nlp):
        """
        PDC NLP
        """
        self.nlp=nlp
        self.mlost = 1.0e9
        self.model = None
        self.lsize = 10
        self.nvac=5000

    def do_eval(self,txtseqs):
        pass

    def free_gen(self,step):
        pass

    def run_training(self,step,learning_rate=1e-2,batch=20, window=110, save=None):
        """
        Entrance for sememe training
        :param step:
        :param learning_rate:
        :param batch:
        :param window:
        :return:
        """
        startt = time.time()
        self.mlost = 1.0e9
        lsize = self.lsize

        vac_size = self.nvac

        if type(self.model)==type(None):
            # def __init__(self, vac_size, sememe_size, hidden_size, num_layers=1):
            rnn = GRU_Sememe(vac_size, lsize, 20, num_layers=1)
        else:
            rnn=self.model
        rnn.train()

        gpuavail = torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(device)
        if gpuavail:
            rnn.to(device)

        # lossc=torch.nn.KLDivLoss()

        # def custom_KLDivLoss(x,y):
        #     return 0

        def customized_loss(xl, yl, model=None):
            # print(x,y)
            # l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            # for ii,W in enumerate(model.parameters()):
            #     l2_reg = l2_reg + W.norm(1)
            loss=0
            for ii in range(len(xl)):
                # loss = loss-torch.sqrt((torch.sum(torch.exp(xl[ii]) * yl[ii])))
                # loss = loss - (torch.sum(xl[ii] * yl[ii]))
                loss = loss + torch.sqrt(torch.sum((xl[ii] - yl[ii])*(xl[ii] - yl[ii])))
            return loss #+ 0.01 * l2_reg

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        train_hist = []
        his = 0

        def get_onehottext(textseq,start,word_to_id,window=window):
            """
            Get one hot rep starting from start with length window from textseq using word_to_id
            :param textseq:
            :param start:
            :param window:
            :return: shape [window,vac_size]
            """
            res=[]
            for iiw in range(window):
                wrd=textseq[start+iiw]
                wid=word_to_id.get(wrd,word_to_id["UNK"])
                item=np.zeros(vac_size)
                if wid<vac_size:
                    item[wid] = 1
                else:
                    item[word_to_id["UNK"]]=1
                res.append(item)
            pres = torch.from_numpy(np.array(res))
            return pres


        for iis in range(step):

            if gpuavail:
                hidden = rnn.initHidden_cuda(device, batch)
            else:
                hidden = rnn.initHidden(batch)

            rstartv=np.floor(np.random.rand(batch)*(len(self.nlp.sub_corpus)-window-1))

            # GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1 = get_onehottext(self.nlp.sub_corpus,int(rstartv[iib]),self.nlp.word_to_id)
                vec2 = get_onehottext(self.nlp.sub_corpus, int(rstartv[iib])+1, self.nlp.word_to_id)
                # (batch,seq,lsize)
                if type(vec1m) == type(None):
                    vec1m = vec1.view(1, window, -1)
                    vec2m = vec2.view(1, window, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(1, window, -1)), dim=0)
                    vec2m = torch.cat((vec2m, vec2.view(1, window, -1)), dim=0)
            # GRU order (seql,batch,lsize)
                x = Variable(vec1m.permute(1,0,2), requires_grad=True)
                y = Variable(vec2m.permute(1,0,2), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if gpuavail:
                x, y = x.to(device), y.to(device)
            # output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
            output, hidden = rnn(x, hidden, batch=batch)
            target=rnn.calsememe(y)

            loss = customized_loss(output, target)

            if int(iis / 10) != his:
                print("MSE Err: ",iis, loss.item())
                his=int(iis / 10)
                if loss.item() < self.mlost:
                    self.mlost = loss.item()
                    self.model = copy.deepcopy(rnn)

            train_hist.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        endt = time.time()
        print("Time used in training:", endt - startt)

        x = []
        for ii in range(len(train_hist)):
            x.append([ii, train_hist[ii]])
        x = np.array(x)
        plt.plot(x[:, 0], x[:, 1])
        if type(save) != type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()


class PDC_NLP(object):
    """
    Main class for NLP PDC modeling
    """
    def __init__(self,nlp):
        """
        PDC NLP
        """
        self.nlp=nlp
        self.mlost = 1.0e9
        self.model = None
        self.lsize = None

        self.lout = None
        self.pcaPM = None
        self.mode = None

        self.prior= None

    def do_eval_baseline(self,txtseqs,mode="random"):
        """
        Calculate some simple baseline
        :param txtseqs:
        :param mode:
        :return:
        """

        datab = self.nlp.build_textmat(txtseqs)
        databp = np.matmul(self.pcaPM, datab.T)
        databp = databp.T
        databp = np.array(databp)

        probs=np.zeros(self.lsize)
        for iic in range(self.lsize):
            try:
                probs[iic]=1/self.nlp.word_to_cnt[self.nlp.id_to_word[iic]]
            except:
                probs[iic] = 1e-9
        probs=probs/np.sum(probs)

        outputl = []
        for iis in range(len(databp) - 1):
            if mode=="random":
                output=np.log(np.ones(self.lsize)/self.lsize)
                outputl.append(output)
            elif mode=="prior":
                output=np.log(probs)
                outputl.append(output)
        print("Prediction Entropy:",cal_entropy(np.exp(output)))

        outputl=torch.from_numpy(np.array(outputl))
        outputl=outputl.type(torch.FloatTensor)
        outputl=outputl.t()
        print(outputl.shape)

        # Generating output label
        yl = []
        for iiss in range(len(txtseqs) - 1):
            ylb = []
            wrd = txtseqs[iiss + 1]
            try:
                vec = self.nlp.w2v_dict[wrd]
                ydg = self.nlp.word_to_id[wrd]
            except:
                ydg = self.nlp.word_to_id["UNK"]
            ylb.append(ydg)
            yl.append(np.array(ylb))
        outlab = Variable(torch.from_numpy(np.array(yl).T))
        outlab = outlab.type(torch.LongTensor)
        lossc = torch.nn.CrossEntropyLoss()
        # (minibatch, C, d1, d2, ..., dK)
        loss = lossc(outputl.view(1, -1, len(databp) - 1), outlab)
        print("Evaluation Perplexity: ", np.exp(loss.item()))

    def do_eval(self,txtseqs,knw_org=None):
        """

        :param seqs: sequence for evaluation
        :return:
        """
        lsize = self.lsize
        datab = self.nlp.build_textmat(txtseqs)
        try:
            databp = np.matmul(self.pcaPM,datab.T)
        except:
            pass
        databp = databp.T
        databp=np.array(databp)
        rnn = self.model
        gpuavail = torch.cuda.is_available()
        if gpuavail:
            device = torch.device("cpu")
            rnn.to(device)
        rnn.eval()
        print(databp.shape)
        assert databp.shape[1] == lsize
        hidden = rnn.initHidden(1)
        outputl = []
        hiddenl=[]

        for iis in range(len(databp)-1):
            x = Variable(torch.from_numpy(databp[iis, :].reshape(1, 1, lsize)).contiguous(), requires_grad=True)
            y = Variable(torch.from_numpy(databp[iis+1, :].reshape(1, 1, lsize)).contiguous(), requires_grad=True)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if self.mode=="LSTM":
                output, hidden = rnn(x, hidden, y)
            elif self.mode=="GRU":
                if knw_org is None:
                    output, hidden = rnn(x, hidden)
                else:
                    add_vec = np.log(knw_org.forward_m(x.data.numpy()))
                    add_tvec = torch.from_numpy(add_vec)
                    add_tvec = add_tvec.type(torch.FloatTensor)
                    output, hidden = rnn(x, hidden, add_prior=add_tvec)
            # if type(outputl) == type(None):
            #     outputl = output.view(1, -1)
            # else:
            #     outputl = torch.cat((outputl, output.view(1, -1)), dim=0)
            # print(outputl.reshape(1,-1).shape,output.data.numpy().reshape(1,-1).shape)
            # outputl=np.stack((outputl.reshape(1,-1),output.data.numpy().reshape(1,-1)))
            outputl.append(output.view(-1).data.numpy())
            hiddenl.append(hidden)

        outputl=np.array(outputl)
        outputl=Variable(torch.from_numpy(outputl).contiguous())
        outputl=outputl.permute((1,0))
        print(outputl.shape)
        # Generating output label
        yl = []
        for iiss in range(len(txtseqs)-1):
            ylb = []
            wrd = txtseqs[iiss+1]
            try:
                vec = self.nlp.w2v_dict[wrd]
                ydg = self.nlp.word_to_id[wrd]
            except:
                ydg = self.nlp.word_to_id["UNK"]
            ylb.append(ydg)
            yl.append(np.array(ylb))
        outlab = Variable(torch.from_numpy(np.array(yl).T))
        outlab = outlab.type(torch.LongTensor)
        lossc = torch.nn.CrossEntropyLoss()
        #(minibatch, C, d1, d2, ..., dK)
        loss = lossc(outputl.view(1,-1,len(databp)-1), outlab)
        print("Evaluation Perplexity: ",np.exp(loss.item()))
        return outputl,hiddenl,outlab.view(-1)

    def free_gen(self,step):
        """
        Training evaluation
        :return:
        """
        rnn=self.model
        gpuavail = torch.cuda.is_available()
        if gpuavail:
            device = torch.device("cpu")
            rnn.to(device)
        rnn.eval()
        lsize=self.lsize

        hidden = rnn.initHidden(1)
        x = Variable(torch.zeros(1,1,lsize), requires_grad=True)
        y = Variable(torch.zeros(1,1,lsize), requires_grad=True)
        outputl=[]

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        for iis in range(step):
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if self.mode=="LSTM":
                y_pred, hidden = rnn(x, hidden, y, cps=0.0, gen=1.0)
            elif self.mode=="GRU":
                y_pred, hidden = rnn(x, hidden)
            ynp = y_pred.data.numpy().reshape(self.lout)
            rndp = np.random.rand()
            pii = logp(ynp).reshape(-1)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            xword=self.nlp.id_to_word[dig]
            outputl.append(xword)
            xvec=self.nlp.w2v_dict[xword]
            xvec=np.matmul(self.pcaPM,xvec)
            x= Variable(torch.from_numpy(xvec.reshape((1,1,lsize))).contiguous(), requires_grad=True)
            y=x
        return outputl


    def run_training(self,step,learning_rate=1e-2,batch=20, window=110, save=None, seqtrain=False, mode="GRU", pcalsize=0, knw_org=None, cellnum=1):
        """
        Entrance for training
        :param learning_rate:
                seqtrain: If whole sequence training is used or not
        Assert tensorshape interface: [batch,x,y,z,lsize]
        :return:
        """
        prtstep=int(step/10)

        startt=time.time()

        self.mlost = 1.0e9
        if pcalsize>0:
            self.lsize=pcalsize
        else:
            self.lsize=len(self.nlp.w2v_dict[self.nlp.id_to_word[0]])
        lsize=self.lsize
        lout=len(self.nlp.w2v_dict)
        self.lout=lout
        self.mode=mode

        ### Calculate prior vector
        probs = np.zeros(self.lsize)
        for iic in range(self.lsize):
            try:
                probs[iic] = 1 / self.nlp.word_to_cnt[self.nlp.id_to_word[iic]]
            except:
                probs[iic] = 1e-9
        probs = probs / np.sum(probs)
        plogits=np.log(probs)
        self.prior=plogits
        self.prior = np.zeros(lout)


        if type(self.model)==type(None):
            # def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
            if mode=="LSTM":
                rnn = RNN_PDC_LSTM_NLP(lsize, 100, 3, 24, lout)
            elif mode=="GRU":
                # rnn = GRU_Cell_Zoneout(lsize, 30, lout, zoneout_rate=0.2)
                # rnn=GRU_NLP(lsize, 30, lout, num_layers=1)
                rnn = GRU_KNW_L(lsize, cellnum, lout ,prior_vec=self.prior)
            # rnn = LSTM_AU(lsize, 12, lsize)
        else:
            rnn=self.model
        self.model = copy.deepcopy(rnn)
        rnn.train()


        gpuavail=torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(device)
        if gpuavail:
            rnn.to(device)

        lossc = torch.nn.CrossEntropyLoss()

        def custom_KNWLoss(outputl, outlab, model, cstep):
            loss1 = lossc(outputl, outlab)
            logith2o = model.h2o.weight#+model.h2o.bias.view(-1) size(47,cell)
            pih2o=torch.exp(logith2o)/torch.sum(torch.exp(logith2o),dim=0)
            lossh2o = -torch.mean(torch.sum(pih2o*torch.log(pih2o),dim=0))

            # # logitz=model.Wiz.weight.view(-1)
            # logitz = model.Wiz.view(-1)
            # piz = torch.exp(logitz) / torch.sum(torch.exp(logitz))
            # lossz= -torch.sum(piz * torch.log(piz))
            #
            # # logitn = model.Win.weight.view(-1)
            # logitn = model.Win.view(-1)
            # pin = torch.exp(logitn) / torch.sum(torch.exp(logitn))
            # lossn = -torch.sum(pin * torch.log(pin))

            l1_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            # l1_reg = l1_reg + model.Wiz.weight.norm(1)+model.Win.weight.norm(1)
            l1_reg = model.Wiz.weight.norm(1) + model.Win.weight.norm(1)

            return loss1 +0.002*l1_reg*cstep/step +0.01*lossh2o*cstep/step   #+0.3*lossz+0.3*lossn #

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        train_hist=[]
        his = 0

        datab=self.nlp.build_textmat(self.nlp.sub_corpus)
        if pcalsize>0:
            databp,pm=pca_proj(datab.T,lsize)
            databp=databp.T
            self.pcaPM=pm
        else:
            self.pcaPM=np.eye(self.lsize)
            databp=datab
        assert databp[0].shape[0] == lsize
        databp = torch.from_numpy(databp)
        if gpuavail:
            databp = databp.to(device)

        for iis in range(step):

            rstartv=np.floor(np.random.rand(batch)*(len(self.nlp.sub_corpus)-window-1))

            if gpuavail:
                hidden = rnn.initHidden_cuda(device, batch)
            else:
                hidden = rnn.initHidden(batch)

            # Generating output label
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = self.nlp.sub_corpus[(int(rstartv[iib]) + iiss + 1)]
                    ydg=self.nlp.word_to_id.get(wrd,self.nlp.word_to_id.get("UNK",None))
                    ylb.append(ydg)
                yl.append(np.array(ylb))
            outlab = Variable(torch.from_numpy(np.array(yl).T))
            outlab = outlab.type(torch.LongTensor)

            if not seqtrain:
                # step by step training
                outputl = None
                # vec1 = rdata_b[:, :, 0]
                # x = Variable(torch.from_numpy(vec1.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                for iiss in range(window):
                    vec1m=None
                    vec2m=None
                    for iib in range(batch):
                        vec1 = databp[(int(rstartv[iib])+iiss),:]
                        vec2 = databp[(int(rstartv[iib])+iiss + 1), :]
                        if type(vec1m) == type(None):
                            vec1m = vec1.view(1, -1)
                            vec2m = vec2.view(1, -1)
                        else:
                            vec1m = torch.cat((vec1m, vec1.view(1,-1)), dim=0)
                            vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                    # One by one guidance training ####### error can propagate due to hidden state
                    x = Variable(vec1m.reshape(1, batch, lsize).contiguous(), requires_grad=True) #
                    y = Variable(vec2m.reshape(1, batch, lsize).contiguous(), requires_grad=True)
                    x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                    if gpuavail:
                        outlab = outlab.to(device)
                        x, y = x.to(device), y.to(device)
                    if mode == "LSTM":
                        output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                    elif mode=="GRU":
                        if knw_org is None:
                            output, hidden = rnn(x, hidden)
                        else:
                            add_vec=np.log(knw_org.forward_m(x.data.numpy()))
                            add_tvec=torch.from_numpy(add_vec)
                            add_tvec=add_tvec.type(torch.FloatTensor)
                            output, hidden = rnn(x, hidden, add_prior=add_tvec)
                    if type(outputl)==type(None):
                        outputl=output.view(batch,lout,1)
                    else:
                        outputl=torch.cat((outputl.view(batch,lout,-1),output.view(batch,lout,1)),dim=2)
                # loss = lossc(outputl, outlab)
                loss=custom_KNWLoss(outputl, outlab, rnn, iis)

            else:
                # LSTM provided whole sequence training
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]) : int(rstartv[iib]) + window, :]
                    vec2 = databp[int(rstartv[iib]) + 1 : int(rstartv[iib]) + 1 + window, :]
                    # (batch,seq,lsize)
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1, window, -1)
                        vec2m = vec2.view(1, window, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, window, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, window, -1)), dim=0)
                # LSTM order (seql,batch,lsize)
                    x = Variable(vec1m.permute(1,0,2), requires_grad=True)
                    y = Variable(vec2m.permute(1,0,2), requires_grad=True)
                    x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    outlab = outlab.to(device)
                    x, y = x.to(device), y.to(device)
                # output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                output, hidden = rnn(x, hidden, batch=batch)
                output=output.permute(1,2,0)

                loss = lossc(output, outlab)

            if int(iis / prtstep) != his:
                print("Perlexity: ",iis, np.exp(loss.item()))
                his=int(iis / prtstep)
                if loss.item() < self.mlost:
                    self.mlost = loss.item()
                    self.model = copy.deepcopy(rnn)

            train_hist.append(np.exp(loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        endt = time.time()
        print("Time used in training:", endt - startt)

        x = []
        for ii in range(len(train_hist)):
            x.append([ii, train_hist[ii]])
        x = np.array(x)
        try:
            plt.plot(x[:, 0], x[:, 1])
            if type(save) != type(None):
                plt.savefig(save)
                plt.gcf().clear()
            else:
                plt.show()
        except:
            pass


class RNN_PDC_LSTM_NLP(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size,num_layers=1):
        super(RNN_PDC_LSTM_NLP, self).__init__()

        self.num_layers=num_layers

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers = self.num_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.h2o2 = torch.nn.Linear(output_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        # self.c2r2h = torch.nn.Linear(context_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()
        self.relu=torch.nn.ReLU()
        self.t0=torch.nn.Parameter(torch.rand(output_size),requires_grad=True)
        self.softmax = torch.nn.LogSoftmax(dim=2)

        self.cdrop = torch.nn.Dropout(p=0.5)

    def forward(self, input, hidden, result, cps=1.0, gen=0.0, batch=1):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(self.num_layers, batch, self.hidden_size)
        c0 = hidden[0][1].view(self.num_layers, batch, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(-1, batch, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output=self.softmax(output)
        # errin = self.relu(result - output-self.t0)-self.relu(-result + output-self.t0)
        # errin = result - output
        # errpipe=hidden[1]
        # errpipe=torch.cat((errpipe[:,:,1:], errin.view(self.input_size,batch,-1)),2)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        # context=hidden[2]
        # context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+0.1*(2*self.sigmoid(self.c2c(context))-1))
        # context = self.tanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1) + (
        #         2 * self.sigmoid(self.c2c(context)) - 1))
        # context = self.hardtanh(context + (1.0-gen)*self.tanh(self.err2c(errpipe.view(1, batch, -1))))
        # context = self.cdrop(context) # trial of dropout
        # c1 = c1 * self.c2r1h(context)
        return output, [(hidden1,c1), [], []]

    def initHidden(self,batch):
        return [(Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True),Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(self.num_layers, batch, self.context_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [(Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device),Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True).to(device),
                Variable(torch.zeros(self.num_layers, batch, self.context_size), requires_grad=True).to(device)]

class GRU_Sememe(torch.nn.Module):
    """
    Pytorch module for Sememe learning
    """
    def __init__(self, vac_size, sememe_size, hidden_size, num_layers=1):
        """
        Init
        :param vac_size: vacubulary size
        :param sememe_size: number of sememe
        :param hidden_size: GRU hidden
        """
        super(GRU_Sememe, self).__init__()

        self.sememe_size = sememe_size
        self.hidden_size = hidden_size
        self.vac_size = vac_size
        self.num_layers=num_layers

        self.Wsem = torch.nn.Linear(vac_size, sememe_size)

        self.gru = torch.nn.GRU(sememe_size, hidden_size, num_layers=num_layers)
        self.h2o = torch.nn.Linear(hidden_size, sememe_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, hidden, batch=1):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        sememe = self.Wsem(input)
        sememe=self.sigmoid(sememe)
        hout, hn = self.gru(sememe.view(-1, batch, self.sememe_size),hidden)
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output = self.sigmoid(output)
        return output,hn

    def calsememe(self, input):
        """
        Calculate sememe embeding sharing Wsem
        :param input:
        :return:
        """
        sememe = self.Wsem(input)
        sememe = self.sigmoid(sememe)
        return sememe

    def initHidden(self,batch=1):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch=1):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)



class GRU_Cell_Zoneout(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, output_size, zoneout_rate=0.2):
        super(GRU_Cell_Zoneout, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.zoneout_rate=zoneout_rate

        self.Wir = torch.nn.Linear(input_size, hidden_size)
        self.Whr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)
        self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """
        rt=self.sigmoid(self.Wir(input)+self.Whr(hidden))
        zt=self.sigmoid(self.Wiz(input)+self.Whz(hidden))
        nt=self.tanh(self.Win(input)+rt*self.Whn(hidden))
        if self.training:
            mask=(np.sign(np.random.random(list(zt.shape))-self.zoneout_rate)+1)/2
            mask = Variable(torch.from_numpy(mask))
            mask = mask.type(torch.FloatTensor)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                mask=mask.to(device)
            zt=1-(1-zt)*mask
            ht=(1-zt)*nt+zt*hidden
        else:
            ht = (1 - zt) * nt + zt * hidden
        output = self.h2o(ht)
        output = self.softmax(output)

        return output, ht

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

class GRU_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU_NLP, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=1):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        hout, hn = self.gru(input.view(-1, batch, self.input_size),hidden)
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output = self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

# class GRU_KNW_P(torch.nn.Module):
#     """
#     PyTorch GRU
#     """
#     def __init__(self, input_size, hidden_size, output_size, prior_vec,batch=20,seq=110,num_layers=1):
#         super(GRU_KNW_P, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#         self.prior_vec = np.array(prior_vec)
#         self.batch=batch
#         self.seq=seq
#         assert len(prior_vec) == output_size
#
#         self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers)
#         self.knowledge = torch.Parameter(torch.from_numpy(np.zeros(hidden_size,output_size)),requires_grad=True)
#
#         tprior=torch.from_numpy(np.array(prior_vec))
#         batchcnt=np.ones(batch)
#         seqcnt=np.ones(seq)
#         batchext=torch.matmul(seqcnt.view(-1,1),tprior.view(1,-1))
#         self.prior=torch.matmul(batchcnt.view(-1,1),batchext.view(seq,1,output_size))
#
#         self.sigmoid = torch.nn.Sigmoid()
#         self.tanh = torch.nn.Tanh()
#         self.softmax = torch.nn.LogSoftmax(dim=-1)
#
#     def forward(self, input, hidden):
#         """
#         Forward
#         :param input:
#         :param hidden:
#         :return:
#         """
#         hout, hn = self.gru(input.view(-1, self.batch, self.input_size),hidden)
#         thetak=(hout+1)/2 # [seq,batch,hidden]
#         self.knowledge=self.knowledge-torch.matmul(torch.mean(self.knowledge,dim=-1, keepdim=True),torch.ones((1,self.output_size)))
#         # [hidden,output]
#         gatedk=torch.matmul(thetak,self.knowledge) # [seq,batch,output]
#         # self.prior [seq,batch,output]
#         output = (self.htanh(self.prior+gatedk)+1)/2
#         output=output/torch.matmul(torch.sum(output,dim=-1, keepdim=True),torch.ones((1,self.output_size)))
#         return torch.log(output)
#
#     def initHidden(self,batch):
#         return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True)
#
#     def initHidden_cuda(self,device, batch):
#         return Variable(torch.zeros(self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)


class GRU_KNW_L(torch.nn.Module):
    """
    PyTorch GRU adjuested for Logit based knowledge learning
    """
    def __init__(self, input_size, hidden_size, output_size,prior_vec=None):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param prior_vec: logit of prior knowledge
        """
        super(GRU_KNW_L, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.Wir = torch.nn.Linear(input_size, hidden_size)
        self.Whr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Whz = torch.nn.Linear(hidden_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)
        self.Whn = torch.nn.Linear(hidden_size, hidden_size)

        self.h2o = torch.nn.Linear(hidden_size, output_size,bias=True)
        # self.h2o.weight=torch.nn.Parameter(torch.zeros(output_size,hidden_size))
        # self.h2o.bias = torch.nn.Parameter(torch.zeros(output_size))

        # self.Wshare = torch.nn.Parameter(torch.rand(input_size, hidden_size), requires_grad=True)
        # self.Wshare_bz = torch.nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        # self.Wshare_bn = torch.nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        # self.Wshare_bo = torch.nn.Parameter(torch.rand(output_size), requires_grad=True)

        if type(prior_vec)!=type(None):
            self.prior_vec=torch.from_numpy(np.array(prior_vec))
            self.prior_vec = self.prior_vec.type(torch.FloatTensor)
        else:
            self.prior_vec = torch.zeros(output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, batch=None, add_prior=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """
        # rt=self.sigmoid(self.Wir(input)+self.Whr(hidden))
        zt=self.sigmoid(self.Wiz(input)+self.Whz(hidden))
        # nt=self.sigmoid(self.Win(input)+rt*self.Whn(hidden))
        nt = self.sigmoid(self.Win(input) +  self.Whn(hidden))
        ht = (1 - zt) * nt + zt * hidden
        if add_prior is None:
            output = self.h2o(ht)+self.prior_vec
        else:
            output = self.h2o(ht) + self.prior_vec + add_prior
        output = self.softmax(output)
        # zt = self.sigmoid(torch.matmul(input,self.Wshare)+ self.Wshare_bz + self.Whz(hidden))
        # nt = self.sigmoid(torch.matmul(input,self.Wshare)+ self.Wshare_bn + self.Whn(hidden))
        # ht = (1 - zt) * nt + zt * hidden
        # output = torch.matmul(ht,self.Wshare.t()) + self.Wshare_bo + self.prior_vec
        # output = self.softmax(output)
        return output, ht

    def knw_distill(self,knw_t,theta_t,id_to_word):
        """
        Distilling knowledge from result
        :param knw_t: significant threshold for knowledge, [th,win] selected group has window win and diff th
        :param theta_t: significant threshold for sigmoid
        :param id_to_word: id to word dictionary
        :return:
        """
        resKNWls=[]
        try:
            knw_b = self.h2o.bias.data.numpy()
        except:
            knw_b = 0
        Nknw=self.h2o.weight.shape[1] #size(47,cell) for input, size(cell,cell) for hidden
        for iic in range(Nknw):
            # Getting out first order knowledge
            knw_w = self.h2o.weight.data.numpy()[:,iic]
            knw_on=knw_b+knw_w
            # logitM=np.max(knw_on)
            # nM=np.argmax(knw_on)
            # knw_on[nM]=np.min(knw_on)
            # logitM2 = np.max(knw_on)
            # resknw_N=None
            # if logitM-logitM2>knw_t:
            #     print("Expert for "+str(id_to_word[nM])+" detected!")
            #     resknw_N=nM
            # else:
            #     print("Knowledge not found!")
            #     continue
            knw_on_zip=list(zip(knw_on,range(len(knw_on))))
            knw_on_zip.sort(key=lambda x: x[0],reverse=True)
            maxv=knw_on_zip[0][0]
            maxvw=maxv-knw_t[1]
            maxvt=maxv-knw_t[0]
            resknw_N=[]
            kstr=""
            found=False
            for iip in range(len(knw_on_zip)-1):
                if knw_on_zip[iip][0]>maxvw and knw_on_zip[iip+1][0]<maxvt:
                    for iik in range(iip+1):
                        kid=knw_on_zip[iik][1]
                        resknw_N.append(kid)
                        kstr=kstr+str(id_to_word[kid])+", "
                    print("Expert for " + kstr + " detected!")
                    found=True
                    break
            if not found:
                print("Knowledge not found!")
                continue

            ## Para hidden for sigmoid
            thz_w=self.Whz.weight.data.numpy()[iic,iic]
            thz_b=self.Whz.bias.data.numpy()[iic]
            thn_w = self.Whn.weight.data.numpy()[iic,iic]
            thn_b = self.Whn.bias.data.numpy()[iic]

            ## Para input for sigmoid
            tiz_w = self.Wiz.weight.data.numpy()[iic,:]
            tiz_b = self.Wiz.bias.data.numpy()[iic]
            tin_w = self.Win.weight.data.numpy()[iic,:]
            tin_b = self.Win.bias.data.numpy()[iic]
            ## Assume ht-1==0
            # Z
            posi_z_t=[]
            posi_z_tvec=thz_b+tiz_b+tiz_w-theta_t
            ## if ii_in in knw set, ht-1==1
            for ii_in in resknw_N:
                posi_z_tvec[ii_in]=posi_z_tvec[ii_in]+thz_w
            for iin in range(len(posi_z_tvec)):
                if posi_z_tvec[iin]>0:
                    posi_z_t.append(iin)
            print("ht-1==0: Posi Z found:",posi_z_t)
            neg_z_t = []
            neg_z_tvec = posi_z_tvec + 2*theta_t
            for iin in range(len(neg_z_tvec)):
                if neg_z_tvec[iin] < 0:
                    neg_z_t.append(iin)
            print("ht-1==0: Neg Z found:", neg_z_t)
            # N
            posi_n_t = []
            posi_n_tvec = thn_b + tin_b + tin_w - theta_t
            ## if ii_in in knw set, ht-1==1
            for ii_in in resknw_N:
                posi_n_tvec[ii_in] = posi_n_tvec[ii_in] + thn_w
            for iin in range(len(posi_n_tvec)):
                if posi_n_tvec[iin] > 0:
                    posi_n_t.append(iin)
            print("ht-1==0: Posi N found:", posi_n_t)
            neg_n_t = []
            neg_n_tvec = posi_n_tvec + 2*theta_t
            for iin in range(len(neg_n_tvec)):
                if neg_n_tvec[iin] < 0:
                    neg_n_t.append(iin)
            print("ht-1==0: Neg N found:", neg_n_t)
            resKNWls.append([resknw_N,posi_z_t,neg_z_t,posi_n_t,neg_n_t])

        return resKNWls

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

class GRU_KNW_CL(torch.nn.Module):
    """
    PyTorch GRU adjuested for Clean Logit based knowledge learning
    """
    def __init__(self, input_size, hidden_size, output_size,prior_vec=None):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param prior_vec: logit of prior knowledge
        """
        super(GRU_KNW_CL, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.Wiz = torch.nn.Parameter(torch.rand(input_size, hidden_size), requires_grad=True)
        self.Win = torch.nn.Parameter(torch.rand(input_size, hidden_size), requires_grad=True)

        self.h2o = torch.nn.Linear(hidden_size, output_size,bias=True)


        if type(prior_vec)!=type(None):
            self.prior_vec=torch.from_numpy(np.array(prior_vec))
            self.prior_vec = self.prior_vec.type(torch.FloatTensor)
        else:
            self.prior_vec = torch.zeros(output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.Lsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, input, hidden,batch=None):
        """

        :param input: input
        :param hidden: hidden
        :param result:
        :return:
        """
        zt = torch.matmul(input,self.softmax(self.Wiz))
        nt = torch.matmul(input,self.softmax(self.Win))
        ht = (1 - zt) * nt + zt * hidden
        output = self.h2o(ht)+self.prior_vec
        output = self.Lsoftmax(output)
        return output, ht

    def initHidden(self, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch=1):
        return Variable(torch.zeros( batch, self.hidden_size), requires_grad=True).to(device)

class KNW_OBJ(object):
    """
    A piece of knowledge
    """
    def __init__(self,id_to_word,prior):
        self.id_to_word=id_to_word
        self.prior=prior
        self.lsize=len(id_to_word)
        self.style=None
        self.description=None
        self.data=None
        self.knw_fingerp=None
        self.knw_utility=None
        self.knw_outmask = None
        self.logith = 1.0 # Confidence of this knowledge
        self.eval_cnt=None # event cnt of this knowledge [total correct]

    def create(self,style,data):
        item=data[0]
        expertInd=data[1]
        if style=="IITNN":
            self.style = "IITNN"  # if item then next not
            self.description = "If " + str(self.id_to_word[item]) + " then next not " + str(self.id_to_word[expertInd])
            self.data = data
            self.knw_fingerp = "IITNN" + "-" + str(item) + "-" + str(expertInd)
            self.knw_utility = self.cal_knwU(self.style,self.data)
            tmp=np.ones(self.lsize)
            tmp[expertInd]=0
            self.knw_outmask=tmp
        elif style=="IITNI":
            self.style = "IITNI"  # if item then next is
            self.description = "If " + str(self.id_to_word[item]) + " then next is " + str(self.id_to_word[expertInd])
            self.data = data
            self.knw_fingerp = "IITNI" + "-" + str(item) + "-" + str(expertInd)
            self.knw_utility = self.cal_knwU(self.style, self.data)
            tmp = np.zeros(self.lsize)
            tmp[expertInd] = 1
            self.knw_outmask = tmp

    def cal_knwU(self, style, data):
        """
        Calculate knowledge utility
        :param switch:
        :return:
        """

        Ntot = self.lsize
        KnwU = None
        if style == "IITNN":
            tmp = self.prior.copy()
            tmp[data[1]] = 0
            KnwU = cal_kldiv(tmp, self.prior) * (self.prior[data[0]] / np.sum(self.prior))
        elif style == "IITNI":
            tmp = np.zeros(Ntot)
            tmp[data[1]] = 1
            KnwU = cal_kldiv(tmp, self.prior) * (self.prior[data[0]] / np.sum(self.prior))
        return KnwU


class KNW_ORG(object):
    """
    Knowledge organizor
    """
    def __init__(self,nlp,prior):
        self.nlp=nlp
        self.prior=prior
        self.lsize=len(nlp.id_to_word)
        self.knw_list=[]
        self.knw_list_fingerp=[]
        self.knw_gate_index = [[] for ii in range(self.lsize)]

    def test(self):
        # Generating output label
        yl = []
        xl = []
        kl=[]
        for iiss in range(len(self.nlp.sub_corpus) - 1):
            xl.append(np.log(self.prior))
            wrd=self.nlp.word_to_id[self.nlp.sub_corpus[iiss+1]]
            yl.append(wrd)
            yvec=np.zeros(self.lsize)
            yvec[wrd]=1
            kld=cal_kldiv(yvec,self.prior)
            kl.append(kld)
        outlab = torch.from_numpy(np.array(yl))
        lossc = torch.nn.CrossEntropyLoss()
        # (minibatch, C, d1, d2, ..., dK)
        xlt=torch.from_numpy(np.array(xl).T)
        loss = lossc(xlt.view(1, -1, len(self.nlp.sub_corpus) - 1), outlab.view(1,-1))
        pkl=np.exp(np.mean(np.array(kl)))
        print("Evaluation Perplexity: ", np.exp(loss.item()),"  v.s.  ", pkl)

    def get(self,fingerp):
        """
        Get knw obj by fingerp
        :param fingerp:
        :return:
        """
        ind=self.knw_list_fingerp.index(fingerp)
        return self.knw_list[ind],ind

    def insert(self,knw_ls):
        """
        Insert knowledge
        :return:
        """
        for knw in knw_ls:
            expertIndL, posiZ, negZ, posiN, negN = knw
            ### Z0 N0 style knowledge
            for expertInd in expertIndL:
                if len(negN)>0:
                    for item in negN:
                        if item in negZ:
                            # if item then next not expertInd style knowledge
                            knw_fingerp = "IITNN" + "-" + str(item) + "-" + str(expertInd)
                            if knw_fingerp not in self.knw_list_fingerp:
                                knw_item=KNW_OBJ(self.nlp.id_to_word,self.prior)
                                knw_item.create("IITNN",[item,expertInd])
                                self.knw_list.append(knw_item)
                                self.knw_list_fingerp.append(knw_item.knw_fingerp)
                                self.knw_gate_index[item].append(knw_item.knw_fingerp)
                ### Z0 N1 style knowledge
                if len(posiN)>0:
                    for item in posiN:
                        if item in negZ:
                            # if item then next is expertInd style knowledge
                            knw_fingerp = "IITNI" + "-" + str(item) + "-" + str(expertInd)
                            if knw_fingerp not in self.knw_list_fingerp:
                                knw_item = KNW_OBJ(self.nlp.id_to_word,self.prior)
                                knw_item.create("IITNI", [item, expertInd])
                                self.knw_list.append(knw_item)
                                self.knw_list_fingerp.append(knw_item.knw_fingerp)
                                self.knw_gate_index[item].append(knw_item.knw_fingerp)
        print(len(self.knw_list))

    def clean(self):
        """
        Clean up knowledge base
        :return:
        """
        # Criteria 1: too low hit rate
        evalhith=1e-4
        deletels=[]
        for item in self.knw_list:
            if item.eval_cnt[1]<evalhith*item.eval_cnt[0]:
                deletels.append(item.knw_fingerp)
        # Criteria 2: too small logith
        for item in self.knw_list:
            if item.logith < 0.1:
                deletels.append(item.knw_fingerp)
        for dstr in deletels:
            self.remove(dstr)
        self.print()

    def remove(self,knw_fingerp):
        """
        Delete a certain knowledge
        :param knw_fingerp: knowledge finger print.
        :return:
        """
        if knw_fingerp not in self.knw_list_fingerp:
            print("Knowledge not found.")
        else:
            ind=self.knw_list_fingerp.index(knw_fingerp)
            del self.knw_list[ind]
            del self.knw_list_fingerp[ind]
        self.knw_gate_index = [[] for ii in range(self.lsize)]
        for knwitem in self.knw_list:
            self.knw_gate_index[knwitem.data[0]].append(knwitem.knw_fingerp)

    def print(self):
        print("No. of knowledge: ",len(self.knw_list))
        for ii in range(len(self.knw_list)):
            print(self.knw_list_fingerp[ii]+" : "+self.knw_list[ii].description, self.knw_list[ii].knw_utility, self.knw_list[ii].eval_cnt, self.knw_list[ii].logith)

    def update(self):
        """
        Updating its knowledge database
        :return:
        """



    def eval_rate(self):
        """
        Knowledge evaluation on dataset, correct rate
        :param dataset:
        :return:
        """
        word_to_id =  {v: k for k, v in self.nlp.id_to_word.items()}
        print("Knowledge evaluating ...")
        for knwobj in self.knw_list:
            knwobj.eval_cnt = [0, 0]
        for nn in range(len(self.nlp.sub_corpus)-1):
            x = word_to_id[self.nlp.sub_corpus[nn]]
            y = word_to_id[self.nlp.sub_corpus[nn+1]]
            for knwid in self.knw_gate_index[x]:
                knwobj,_=self.get(knwid)
                mask=knwobj.knw_outmask
                predvec=np.zeros(self.lsize)
                predvec[y]=1
                mres=np.max(mask*predvec)
                knwobj.eval_cnt[0] = knwobj.eval_cnt[0] + 1
                if mres>0: # Compatible
                    knwobj.eval_cnt[1]=knwobj.eval_cnt[1]+1

    def assign(self,data,switch="logith"):
        """
        Assign a data to knowledge property
        :param data:
        :param switch:
        :return:
        """
        assert len(data)==len(self.knw_list)
        print(data)
        for knwii in range(len(self.knw_list)):
            if switch=="logith":
                self.knw_list[knwii].logith=data[knwii]
        self.print()

    def eval_perplexity(self):
        """
        Knowledge evaluation on dataset, perplexity
        :param dataset:
        :return:
        """
        word_to_id = {v: k for k, v in self.nlp.id_to_word.items()}
        print("Knowledge evaluating ...")
        perpls=[]
        for nn in range(len(self.nlp.sub_corpus)-1):
            x = word_to_id[self.nlp.sub_corpus[nn]]
            y = word_to_id[self.nlp.sub_corpus[nn+1]]
            prd = self.forward_s(x)
            yvec=np.zeros(self.lsize)
            yvec[y]=1
            perp=cal_kldiv(yvec,prd)
            perpls.append(perp)
        avperp=np.mean(np.array(perpls))
        print("Calculated knowledge perplexity:",np.exp(avperp))
        return np.exp(avperp)


    def forward_s(self,inputd):
        """
        Forwarding and calculate logit with knowledge, inputd is a digit
        :param logith: logit threshold for hard knowledge
        :return:
        """
        plogits = np.log(self.prior)
        for knwid in self.knw_gate_index[inputd]:
            knwobj, _ = self.get(knwid)
            plogits=plogits+knwobj.knw_outmask*knwobj.logith
        postr=np.exp(plogits)/np.sum(np.exp(plogits))
        return postr

    def forward_m(self,inputm):
        """
        Forwarding and calculate logit with knowledge
        assert shape(1,batch,lsize)
        :param logith: logit threshold for hard knowledge
        :return:
        """
        assert inputm.shape[0] == 1
        assert inputm.shape[2] == self.lsize
        assert len(inputm.shape) == 3
        plogits = np.log(self.prior)
        postr=np.zeros(inputm.shape)
        for batchii in range(inputm.shape[1]):
            inputd=list(inputm[0,batchii,:]).index(1)
            clogits = plogits
            try:
                for knwid in self.knw_gate_index[inputd]:
                    knwobj,_ = self.get(knwid)
                    clogits=clogits+knwobj.knw_outmask*knwobj.logith
                postr[0,batchii,:]=np.exp(clogits)/np.sum(np.exp(clogits))
            except:
                pass
        return postr

    def optimize_logith(self):
        """
        Optimize logith using scipy optimizor
        :param dataset:
        :return:
        """
        print("Knowledge optimizing ...")

        def eval(para,dataset):
            perpls = []
            for nn in range(len(dataset) - 1):
                x = self.nlp.word_to_id[dataset[nn]]
                y = self.nlp.word_to_id[dataset[nn + 1]]
                plogits = np.log(self.prior)
                for knwid in self.knw_gate_index[x]:
                    knwobj,ind = self.get(knwid)
                    plogits = plogits + knwobj.knw_outmask * para[ind]
                postr = np.exp(plogits) / np.sum(np.exp(plogits))
                yvec = np.zeros(self.lsize)
                yvec[y] = 1
                perp = cal_kldiv(yvec, postr)
                perpls.append(perp)
            avperp = np.mean(np.array(perpls))
            print("Calculated knowledge perplexity:", np.exp(avperp))
            return avperp

        x=np.ones(len(self.knw_list))
        res=minimize(eval, x ,self.nlp.sub_corpus, method='SLSQP')
        print(res)
        for iit in range(len(res.x)):
            self.knw_list[iit].logith=res.x[iit]

    def optimize_logith_batch(self,bper=0.02,step=100):
        """
        Use mini batched version to do knowledge optimizing
        :param dataset:
        :param bper:
        :return:
        """
        print("Knowledge mini-batched optimizing ...")

        def eval(para,dataset):
            perpls = []
            for nn in range(len(dataset) - 1):
                x = self.nlp.word_to_id[dataset[nn]]
                y = self.nlp.word_to_id[dataset[nn + 1]]
                plogits = np.log(self.prior)
                for knwid in self.knw_gate_index[x]:
                    knwobj,ind = self.get(knwid)
                    plogits = plogits + knwobj.knw_outmask * para[ind]
                postr = np.exp(plogits) / np.sum(np.exp(plogits))
                yvec = np.zeros(self.lsize)
                yvec[y] = 1
                perp = cal_kldiv(yvec, postr)
                perpls.append(perp)
            avperp = np.mean(np.array(perpls))
            return avperp

        x = np.ones(len(self.knw_list))
        for iis in range(step):
            startp=np.random.rand()*(1-bper)*len(self.nlp.sub_corpus)
            mbdataset=self.nlp.sub_corpus[int(startp):int(startp+bper*len(self.nlp.sub_corpus))]
            res = minimize(eval, x, mbdataset, method='SLSQP',options={"maxiter":3})
            avperp=eval(res.x,mbdataset)
            print("Step "+str(iis)+", calculated perplexity:", np.exp(avperp))
            x=res.x
        avperp = eval(x, self.nlp.sub_corpus)
        print("Final evaluation, calculated perplexity:", np.exp(avperp))
        print(res)
        for iit in range(len(res.x)):
            self.knw_list[iit].logith = res.x[iit]

    def optimize_logith_torch(self,step,learning_rate=1e-2,batch=20, window=110):
        """
        Use pytorch batched version to do knowledge optimizing
        :param dataset:
        :param bper:
        :param step:
        :return:
        """
        print("Knowledge para pytorch optimizing ...")
        plogits = np.log(self.prior)
        plogits = torch.from_numpy(plogits)
        plogits = plogits.type(torch.FloatTensor)
        datap = self.nlp.build_textmat(self.nlp.sub_corpus)
        assert datap[0].shape[0] == self.lsize
        databp = torch.from_numpy(datap)

        rnn=KNW_CELL(self.lsize,self.knw_list,self.knw_gate_index,self.knw_list_fingerp)
        lossc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

        train_hist = []
        his = 0

        for iis in range(step):
            rstartv = np.floor(np.random.rand(batch) * (len(self.nlp.sub_corpus) - window - 1))
            # Generating output label
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = self.nlp.sub_corpus[(int(rstartv[iib]) + iiss + 1)]
                    ydg = self.nlp.word_to_id.get(wrd, self.nlp.word_to_id.get("UNK",None))
                    ylb.append(ydg)
                yl.append(np.array(ylb))
            outlab = Variable(torch.from_numpy(np.array(yl).T))
            outlab = outlab.type(torch.LongTensor)


            outputl = None

            for iiss in range(window):
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databp[(int(rstartv[iib]) + iiss), :]
                    vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                # One by one guidance training
                x = Variable(vec1m.reshape(1, batch, self.lsize).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, self.lsize).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                output = rnn(x,plogits)
                if type(outputl) == type(None):
                    outputl = output.view(batch, self.lsize, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, self.lsize, -1), output.view(batch, self.lsize, 1)), dim=2)
            loss = lossc(outputl, outlab)

            if int(iis / 100) != his:
                print("Perlexity: ",iis, np.exp(loss.item()))
                his=int(iis / 100)

            train_hist.append(np.exp(loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        def eval(para,dataset):
            perpls = []
            for nn in range(len(dataset) - 1):
                x = self.nlp.word_to_id[dataset[nn]]
                y = self.nlp.word_to_id[dataset[nn + 1]]
                plogits = np.log(self.prior)
                for knwid in self.knw_gate_index[x]:
                    knwobj,ind = self.get(knwid)
                    plogits = plogits + knwobj.knw_outmask * para[ind]
                postr = np.exp(plogits) / np.sum(np.exp(plogits))
                yvec = np.zeros(self.lsize)
                yvec[y] = 1
                perp = cal_kldiv(yvec, postr)
                perpls.append(perp)
            avperp = np.mean(np.array(perpls))
            return avperp

        res=rnn.knw_para.data.numpy()
        avperp = eval(res, self.nlp.sub_corpus)
        print("Final evaluation, calculated perplexity:", np.exp(avperp))
        for iit in range(len(res)):
            self.knw_list[iit].logith = res[iit]


    def save(self,name="knwlist.pickle"):
        """
        Saving key knowledge to a list
        :param name:
        :return:
        """
        knw_list_save = []
        print("No. of knowledge to be saved: ",len(self.knw_list))
        for knwobj in self.knw_list:
            knw_dict=dict([])
            knw_dict["style"]= knwobj.style
            knw_dict["data"] = knwobj.data
            knw_dict["knw_outmask"] = knwobj.knw_outmask
            knw_dict["logith"] = knwobj.logith
            knw_dict["eval_cnt"] = knwobj.eval_cnt
            knw_list_save.append(knw_dict)
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
        print(file)
        pickle.dump(knw_list_save, open(file, "wb"))
        print("Knowledge data list saved.")

    def load(self,name="knwlist.pickle"):
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
        print(file)
        knw_list = pickle.load(open(file, "rb"))
        self.knw_list = []
        self.knw_list_fingerp = []
        self.knw_gate_index = [[] for ii in range(self.lsize)]
        for knwd in knw_list:
            knw_item = KNW_OBJ(self.nlp.id_to_word,self.prior)
            knw_item.create(knwd["style"],knwd["data"])
            knw_item.knw_outmask=knwd["knw_outmask"]
            knw_item.logith = knwd["logith"]
            knw_item.eval_cnt = knwd["eval_cnt"]
            self.knw_list.append(knw_item)
            self.knw_list_fingerp.append(knw_item.knw_fingerp)
            self.knw_gate_index[knwd["data"][0]].append(knw_item.knw_fingerp)

class KNW_CELL(torch.nn.Module):
    """
    PyTorch knowledge cell
    """
    def __init__(self, lsize, knw_list, knw_gate_index,knw_list_fingerp):
        super(KNW_CELL, self).__init__()
        knw_size = len(knw_list)
        self.knw_para = torch.nn.Parameter(torch.ones(knw_size), requires_grad=True)
        # knowledge mask mat (knw_size,lsize)
        knw_maskmat=np.ones((knw_size,lsize))
        # knowledge act mat (lsize,knw_size)
        knw_actmat = np.zeros((lsize,knw_size))
        for iik in range(knw_size):
            knw_maskmat[iik,:]=knw_list[iik].knw_outmask
        for iil in range(lsize):
            for iia in range(len(knw_gate_index[iil])):
                ida=knw_list_fingerp.index(knw_gate_index[iil][iia])
                knw_actmat[iil,ida] = 1

        self.knw_maskmat=torch.from_numpy(knw_maskmat)
        self.knw_maskmat=self.knw_maskmat.type(torch.FloatTensor)

        self.knw_actmat = torch.from_numpy(knw_actmat)
        self.knw_actmat = self.knw_actmat.type(torch.FloatTensor)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input,plogits):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        knw_act=torch.matmul(input,self.knw_actmat)
        scaled_act=knw_act*self.knw_para
        knw_vec=torch.matmul(scaled_act,self.knw_maskmat)+plogits
        output=self.softmax(knw_vec)
        return output









