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

import torch
from torch.autograd import Variable

from ncautil.ncalearn import pca_proj

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
        self.word_to_id = dict(zip(words, range(1,1+len(words))))
        self.word_to_id["UNK"] = 0
        self.word_to_cnt = dict(zip(words, counts))
        bv = self.word_to_cnt[words[0]]
        for k,v in self.word_to_cnt.items():
            self.word_to_cnt[k]=1/(self.word_to_cnt[k]/bv)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="gensim",Nvac=10000):
        """
        Build word to vector lookup table
        :param mode: "pickle"
        :return:
        """
        print("Building word to vector lookup table...")
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


class PDC_NLP(object):
    """
    Main class for NLP PDC modeling
    """
    def __init__(self,nlp):
        """
        PDC NLP
        """
        self.nlp=nlp
        self.mlost = 1.0e99
        self.model = None
        self.lsize = 100

        self.lout = None
        self.pcaPM = None

    def do_eval(self,txtseqs):
        """

        :param seqs: sequence for evaluation
        :return:
        """
        lsize = self.lsize
        datab = self.nlp.build_textmat(txtseqs)
        databp, pm = pca_proj(datab.T, lsize)
        databp = databp.T
        databp=np.array(databp)
        rnn = self.model
        rnn.eval()
        print(databp.shape)
        assert databp.shape[1] == lsize
        hidden = rnn.initHidden(1)
        outputl = []
        hiddenl=[]
        outputl.append(databp[0, :].reshape(-1, ))
        hiddenl.append(hidden)
        for iis in range(len(databp)-1):
            x = Variable(torch.from_numpy(databp[iis, :].reshape(1, 1, lsize)).contiguous(), requires_grad=True)
            y = Variable(torch.from_numpy(databp[iis+1, :].reshape(1, 1, lsize)).contiguous(), requires_grad=True)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            output, hidden = rnn(x, hidden, y)
            # print(outputl.reshape(1,-1).shape,output.data.numpy().reshape(1,-1).shape)
            # outputl=np.stack((outputl.reshape(1,-1),output.data.numpy().reshape(1,-1)))
            outputl.append(output.data.numpy().reshape(-1, ))
            hiddenl.append(hidden)
        return outputl,hiddenl

    def free_gen(self,step):
        """
        Training evaluation
        :return:
        """
        rnn=self.model
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
            y_pred, hidden = rnn(x, hidden, y, cps=0.0, gen=1.0)
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
            x= Variable(torch.from_numpy(xvec).contiguous(), requires_grad=True)
            y=x
        return outputl


    def run_training(self,step,learning_rate=1e-2,batch=20, window=100, save=None, seqtrain=False):
        """
        Entrance for training
        :param learning_rate:
                seqtrain: If whole sequence training is used or not
        :return:
        """
        startt=time.time()

        self.mlost = 1.0e9
        lsize=self.lsize
        lout=len(self.nlp.w2v_dict)
        self.lout=lout


        if type(self.model)==type(None):
            # def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
            rnn = RNN_PDC_LSTM_NLP(lsize, 100, 3, 24, lout)
            # rnn = LSTM_AU(lsize, 12, lsize)
        else:
            rnn=self.model


        gpuavail=torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(device)
        if gpuavail:
            rnn.to(device)

        lossc = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        train_hist=[]
        his = 0

        datab=self.nlp.build_textmat(self.nlp.sub_corpus)
        databp,pm=pca_proj(datab.T,lsize)
        databp=databp.T
        self.pcaPM=pm

        for iis in range(step):

            rstartv=np.floor(np.random.rand(batch)*(len(self.nlp.sub_corpus)-window-1))

            if gpuavail:
                hidden = rnn.initHidden_cuda(device, batch)
            else:
                hidden = rnn.initHidden(batch)

            assert databp[0].shape[0]==lsize

            # Generating output label
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = self.nlp.sub_corpus[(int(rstartv[iib]) + iiss + 1)]
                    try:
                        vec = self.nlp.w2v_dict[wrd]
                        ydg = self.nlp.word_to_id[wrd]
                    except:
                        ydg = self.nlp.word_to_id["UNK"]
                    ylb.append(ydg)
                yl.append(np.array(ylb))
            outlab = Variable(torch.from_numpy(np.array(yl).T))
            outlab = outlab.type(torch.LongTensor)

            if not seqtrain:
                # step by step training
                if gpuavail:
                    databp = torch.from_numpy(databp)
                    databp = databp.to(device)
                outputl = None
                # vec1 = rdata_b[:, :, 0]
                # x = Variable(torch.from_numpy(vec1.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                for iiss in range(window):
                    vec1m=[]
                    vec2m=[]
                    for iib in range(batch):
                        vec1 = databp[(int(rstartv[iib])+iiss),:]
                        vec2 = databp[(int(rstartv[iib])+iiss + 1), :]
                        vec1m.append(vec1)
                        vec2m.append(vec2)
                    vec1m=np.array(vec1m)
                    vec2m = np.array(vec2m)
                    if gpuavail:
                        # One by one guidance training ####### error can propagate due to hidden state
                        x = Variable(vec1m.reshape(1, batch, lsize).contiguous(), requires_grad=True) #
                        y = Variable(vec2m.reshape(1, batch, lsize).contiguous(), requires_grad=True)
                    else:
                        # One by one guidance training #######
                        x = Variable(torch.from_numpy(vec1m.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                        y = Variable(torch.from_numpy(vec2m.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                    output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                    if type(outputl)==type(None):
                        outputl=output.view(batch,lout,1)
                    else:
                        outputl=torch.cat((outputl.view(batch,lout,-1),output.view(batch,lout,1)),dim=2)
                loss = lossc(outputl, outlab)

            else:
                # LSTM provided whole sequence training
                vec1m = []
                vec2m = []
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]) + iiss : int(rstartv[iib]) + iiss + window, :]
                    vec2 = databp[int(rstartv[iib]) + iiss + 1 : int(rstartv[iib]) + iiss + 1 + window, :]
                    vec1m.append(vec1)
                    vec2m.append(vec2) # (batch,seq,lsize)
                vec1m = np.array(vec1m)
                vec2m = np.array(vec2m)
                # LSTM order (seql,batch,lsize)
                x = Variable(torch.from_numpy(np.transpose(vec1m, (1,0,2))).contiguous(), requires_grad=True)
                y = Variable(torch.from_numpy(np.transpose(vec2m, (1,0,2))).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    x, y = x.to(device), y.to(device)
                    outlab=outlab.to(device)
                output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                output=output.permute(1,2,0)

                loss = lossc(output, outlab)

            if int(iis / 10) != his:
                print("Perlexity: ",iis, np.exp(loss.data[0]))
                his=int(iis / 10)
                if loss.data[0] < self.mlost:
                    self.mlost = loss.data[0]
                    self.model = copy.deepcopy(rnn)

            train_hist.append(np.exp(loss.data[0]))

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

class RNN_PDC_LSTM_NLP(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTM_NLP, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
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
        hidden0=hidden[0][0].view(1, batch, self.hidden_size)
        c0 = hidden[0][1].view(1, batch, self.hidden_size)
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
        return [(Variable(torch.zeros(1, batch,self.hidden_size), requires_grad=True),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, batch, self.context_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [(Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True).to(device),
                Variable(torch.zeros(1, batch, self.context_size), requires_grad=True).to(device)]



















