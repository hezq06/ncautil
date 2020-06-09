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
from matplotlib import cm
from collections import OrderedDict
import nltk
nltk.internals.config_java(options='-Xmx2048m')
import pickle
from nltk.corpus import brown,treebank
from nltk.parse import stanford
import gensim
from ncautil.w2vutil import W2vUtil
from ncautil.datautil import *
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize

import torch
from torch.autograd import Variable

from torchnlp.word_to_vector import GloVe
from torchnlp.datasets import simple_qa_dataset
from torchnlp.datasets import wikitext_2_dataset

from ncautil.ncalearn import *
from ncautil.ncamath import *
from ncautil.datautil import save_data

from tqdm import tqdm,tqdm_notebook

__author__ = "Harry He"

class NLPDataSet(torch.utils.data.Dataset):

    def __init__(self):
        print("Initializing NLPDataSet")

class NLPutil(object):
    def __init__(self):
        print("Initializing NLPUtil")
        self.dir="tmp/"
        self.corpus=None
        self.tagged_sents = None
        self.sub_size = 1000000
        self.sub_corpus = None
        self.sub_mat=None
        self.word_to_id=None
        self.id_to_word=None
        self.word_to_cnt=None
        self.prior=None
        self.w2v_dict=None
        self.test_text=None
        self.labels=None
        self.w2v_mat=None
        self.pt_emb=None

        self.id2v_mat=None
        self.nVcab=None
        self.lsize = None

        # Wiki parameter
        self.file_idx=0 # Current file index
        self.tot_filenum = 1 # total available file (for wiki data full)
        self.wiki_text_path = "/storage/hezq17/wikitext/wki_text_process"

        # self.synmat=SyntaxMat()

    def get_data(self,corpus,file_idx=0,data=None,format="txt"):
        """
        Get corpus data
        :param corpus: "brown","ptb"
        :param file_idx: file idx
        :return:
        """
        print("Getting corpus data...")
        if corpus=="brown":
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
            self.labels = lablist
            counter = collections.Counter(lablist)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            self.setlabels, _ = list(zip(*count_pairs))
            self.tagged_sents = treebank.tagged_sents()
        elif corpus=="selfgen" and data is not None:
            self.corpus=data
            self.sub_corpus=data
        elif corpus=="simple_qa_dataset":
            self.corpus = simple_qa_dataset(train=True)
            self.sub_corpus = self.corpus
        elif corpus=="wiki":
            tmpcorp = wikitext_2_dataset(train=True)
            self.corpus = []
            for item in tmpcorp:
                self.corpus.append(item.lower())
            self.sub_corpus = self.corpus
        elif corpus=="wiki_full":
            self.file_idx=file_idx
            self.tot_filenum=134
            file = os.path.join(self.wiki_text_path, "wiki_"+str(file_idx))
            f = open(file)
            raw = f.read()
            self.corpus = word_tokenize(raw)
            self.corpus = [w.lower() for w in self.corpus]
            self.sub_corpus = self.corpus[:self.sub_size]
        elif corpus=="imdb":
            imdb = IMDbSentimentUtil()
            imdb.get_corpus()
            imdb.corpus_clean()
            self.corpus = imdb.corpus_train
            self.corpus_test = imdb.corpus_test
            self.sub_corpus = self.corpus[:self.sub_size]
        else:
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/' + str(corpus))
            if format=="txt":
                f = open(file)
                raw = f.read()
                self.corpus = word_tokenize(raw)
                self.corpus = [w.lower() for w in self.corpus]
                self.sub_corpus = self.corpus[:self.sub_size]
            elif format=="pickle":
                self.corpus = load_data(file)
                self.sub_corpus = self.corpus[:self.sub_size]
        print("Length of corpus: "+str(len(self.corpus)))
        print("Vocabulary of corpus: " + str(len(set(self.corpus))))

    def fall_back(self,fallback_dicts):
        print("Falling back...")
        new_corpus=[]
        for wrd in tqdm_notebook(self.corpus):
            for idict in fallback_dicts:
                wrd=idict.get(wrd,wrd)
            new_corpus.append(wrd)
        self.corpus=new_corpus

    def init_test(self,ttext):
        self.test_text=word_tokenize(ttext)
        self.test_text = [w.lower() for w in self.test_text]
        print("Length of test text: " + str(len(self.test_text)))

    def print_txt(self,start=0,length=100,switch="sub_corpus",data=None,mode="simple"):
        """
        Print a passage
        :param start: start offset
        :param length: length
        :return: null
        """
        if data is None:
            string=""
            for ii in range(length):
                string=string+self.sub_corpus[start+ii] +" "
            print(string)
        else:
            string = ""
            if mode=="full":
                for ii in range(len(data)):
                    string = string + self.id_to_word[data[ii].item()] + "("+str(ii) +"," + str(data[ii].item())+")" + " "
                print(string)
            elif mode=="simple":
                for ii in range(len(data)):
                    string = string + self.id_to_word[data[ii].item()] + " "
                print(string)

    def set_corpus(self,corpus):
        """
        Set new corpus
        :param corpus:
        :return:
        """
        self.corpus=corpus
        print("Corpus updated.")

    def build_vocab(self,corpus=None,Vsize=None):
        """
        Building vocabulary
        Referencing part of code from: Basic word2vec example tensorflow, reader.py
        :param corpus: corpus
        :param Vsize: vocabulary size
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
        if Vsize is not None and len(words)>Vsize:
            words = list(words)
            counts = list(counts)
            print(len(words),len(counts))
            unkc=0
            clenw = len(words)
            for iiex in range(clenw-Vsize):
                del words[clenw-iiex-1]
                unkc=unkc+counts[clenw - iiex - 1]
                del counts[clenw - iiex - 1]
            words.append("UNK")
            counts.append(unkc)
        self.word_to_id = dict(zip(words, range(len(words))))
        self.word_to_cnt = dict(zip(words, counts))
        bv = self.word_to_cnt[words[0]]
        for k,v in self.word_to_cnt.items():
            self.word_to_cnt[k]=1/(self.word_to_cnt[k]/bv)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.prior=np.zeros(len(self.id_to_word))
        for id in range(len(self.id_to_word)):
            self.prior[id]=1/self.word_to_cnt[self.id_to_word[id]]
        self.prior=self.prior/np.sum(self.prior)
        print(self.id_to_word)
        self.nVcab=len(self.word_to_id.items())
        print("Collected vocabulary size:",self.nVcab)
        return self.word_to_id, self.id_to_word

    def build_w2v(self,mode="onehot",Nvac=10000):
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
            elif mode=="torchnlp":
                hdir=os.getenv("HOME")
                cache_path=os.path.join(hdir,"MyWorkSpace/NLConLayer/.word_vectors_cache")
                model = GloVe(cache=cache_path)
            self.w2v_dict = dict([])
            skip = []
            for ii in range(Nvac):
                try:
                    word = self.id_to_word[ii]
                    if mode=="torchnlp":
                        vec = model[word].data.numpy()
                    else:
                        vec = model[word]
                    self.w2v_dict[word] = vec
                    if np.std(vec)==0:
                        skip.append(word)
                except:
                    vec = np.zeros(len(model[self.id_to_word[10]]))
                    self.w2v_dict[word] = vec
                    skip.append(word)
            print("Except list: length " + str(len(skip)))
            print(skip)
            self.w2v_dict["UNK"] = np.zeros(len(self.w2v_dict[self.id_to_word[10]]))
            # self.w2v_dict["UNK"] = np.random.random(len(self.w2v_dict[self.id_to_word[10]])) !!!!!! A BIG BUG !!! Source of unpredictable behavior when using model reload
        else:
            if len(self.id_to_word)>200:
                raise Exception("Too much words for onehot representation.")
            self.w2v_dict=dict([])
            for ii in range(len(self.id_to_word)):
                vec=np.zeros(len(self.id_to_word))
                vec[ii]=1.0
                self.w2v_dict[self.id_to_word[ii]]=vec
        self.lsize=len(list(self.w2v_dict.values())[0])
        print("Dimension of embedding:",self.lsize)
        return self.w2v_dict

    def build_pt_emb(self):
        Ndim=len(self.w2v_dict[self.id_to_word[0]])
        weight=torch.zeros((self.nVcab,Ndim))
        for ii in range(self.nVcab):
            weight[ii,:]=torch.FloatTensor(self.w2v_dict[self.id_to_word[ii]])
        self.pt_emb=torch.nn.Embedding.from_pretrained(weight, freeze=True)

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

    def build_w2v_from_emb(self,emb):
        """
        Build w2v_dict using a torch.nn.Embeddings
        :param embedding:
        :return:
        """
        print("Building w2v_dict from torch embedding module")
        emb_cpu=emb.cpu()
        wrdl = torch.range(0, self.nVcab-1).type(torch.LongTensor)
        wrd_mat = emb_cpu(wrdl).detach().numpy()
        self.w2v_dict=dict([])
        for ii in range(self.nVcab):
            self.w2v_dict[self.id_to_word[ii]]=wrd_mat[ii]

    def build_textmat(self,text,w2v_dict_temp=None):
        """
        Build sequecing vector of sub_corpus
        :return:
        """
        print("Building sequecing vector of text...")
        txtmat = []
        if w2v_dict_temp is not None:
            w2v_dict=w2v_dict_temp
        else:
            w2v_dict=self.w2v_dict
        unkvec=w2v_dict.get("UNK",None)
        for word in text:
            wvec=w2v_dict.get(word,unkvec)
            txtmat.append(wvec)
        txtmat=np.array(txtmat)
        # self.plot_txtmat(txtmat.T)
        return txtmat

    def plot_txtmat(self,data,start=0,length=1000,text=None,texty=None,title=None,save=None,origin='lower',xlabel=None, ylabel=None):
        data=np.array(data)
        assert len(data.shape) == 2
        fig,ax=plt.subplots()
        img=data[:,start:start+length]
        fig=ax.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)),np.amax(np.abs(data))), origin=origin)
        if text is not None:
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
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if type(title) != type(None):
            plt.title(title)
        if type(save)!=type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()

    def plot_textshade(self,text,data,start=0,length=100,cper_line=100,gamma=0.3):
        """
        Plot text with shaded back ground of data
        :param text:
        :param data:
        :param start:
        :param length:
        :param cper_line:
        :param gamma:
        :return:
        """
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

    def cal_v2w(self,vec,numW=10):
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
            res.append(("UNK",-1e10))
        for (k,v) in self.w2v_dict.items():
            dist=self.cal_cosdist(v,vec)
            # np.dot(v,vec)/np.linalg.norm(v)/np.linalg.norm(vec)
            #dist=-np.linalg.norm(v-vec)
            if dist>=res[numW-1][1]:
                res[numW-1]=(k,dist)
                res.sort(key=lambda tup:tup[1],reverse=True)
        return res

    def cal_v2w_gpu(self,vec,numW=10,cuda_device="cuda:0"):
        """
        Calculate leading numW nearest using cosine distance definition
        :param vec: input vector
        :param numW: number of words returned
        :return: (word,dist) list, length is numW
        """
        res=[]

        assert torch.cuda.is_available()

        if self.id2v_mat is None:
            self.id2v_mat = torch.zeros((self.nVcab,self.lsize))
            for k, v in self.w2v_dict.items():
                self.id2v_mat[self.word_to_id[k]] = torch.from_numpy(v).type(torch.FloatTensor)
            self.id2v_mat=self.id2v_mat.to(cuda_device)

        tvec=torch.from_numpy(vec).type(torch.FloatTensor).to(cuda_device)

        matmm=torch.matmul(self.id2v_mat,tvec)
        cosdistvec=matmm/torch.norm(self.id2v_mat,2,dim=1)/torch.norm(tvec,2)

        cosdistvecnp=cosdistvec.cpu().numpy()


        for ii in range(numW):
            res.append(("UNK",-1e10))
        for ii in range(len(cosdistvecnp)):
            dist=cosdistvecnp[ii]
            if dist>=res[numW-1][1]:
                res[numW-1]=(self.id_to_word[ii],dist)
                res.sort(key=lambda tup:tup[1],reverse=True)
        return res

    def cal_v2w_full(self,vec,numW=10,dist="cos",reverse=True):
        """
        Calculate leading numW nearest using cosine distance definition
        :param vec: input vector
        :param numW: number of words returned
        :return: (word,dist) list, length is numW
        """
        res=[]
        for (k,v) in self.w2v_dict.items():
            dist=self.cal_cosdist(v,vec)
            res.append((k,dist))
        res.sort(key=lambda tup:tup[1],reverse=reverse)
        return res[:numW]


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
        # path = '/Users/zhengqihe/HezMain/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models'
        path = "/home/hezq17/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models"
        os.environ['STANFORD_MODELS'] = path
        model = 'english-caseless-left3words-distsim.tagger'
        # model="english-bidirectional-distsim.tagger"
        self.pos = StanfordPOSTagger(model,
                                path_to_jar=path + '/../stanford-postagger.jar',java_options='-mx16g')

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

    def pos_tagger(self,sent): # Suprisingly low accuracy
        """
        Calculate the part of speech tag
        :sent: list of word ["Hello","world","!"]
        :return: [(word,NN),...]
        """
        # if type(sent)==list:
        #     sent=""
        #     for w in sent:
        #         sent=sent+" "+w
        psents = self.parser.raw_parse_sents(sent)
        a = []
        # GUI
        for line in psents:
            for sentence in line:
                a.append(sentence)

        posls=[]
        for ptree in a:
            pos = ptree.pos()
            posls.append(pos[0])

        return posls

    def pos_tagger_fast(self,sent,tag_lenghth = 100000): # 90% accuracy for PTB
        """

        :param sent: list of words
        :param tag_lenghth: if sent very long, seperate to tag_lenghth and tag would speed up
        :return:
        """
        if type(sent)==type("string"):
            sent=word_tokenize(sent)
            sent = [w.lower() for w in sent]
        # def ini_pos():
            # path = '/Users/zhengqihe/HezMain/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models'
            # path="/data03/home/hezq17/MySourceCode/stanfordparser/stanford-postagger-full-2017-06-09/models"
            # os.environ['STANFORD_MODELS'] = path
            # pos = StanfordPOSTagger('english-bidirectional-distsim.tagger',
            #                              path_to_jar=path + '/../stanford-postagger.jar')
            # return pos
        # pos=ini_pos()

        pos_tot_l = []
        N_split = int(np.ceil(len(sent) / tag_lenghth))
        for ii in tqdm_notebook(range(N_split)):
            sent_seg = sent[ii * tag_lenghth:(ii + 1) * tag_lenghth]
            pos_sent = self.pos.tag(sent_seg)
            for item in pos_sent:
                pos_tot_l.append(item)
        return pos_tot_l


class FallBack(object):
    def __init__(self, leaftag, parenttag, pos_corpus, nlp, similar_thrd=0.7, cnt_thrd=5, balance_para=10):
        self.leaftag = leaftag
        self.parenttag = parenttag

        self.LF2PR_dict = dict([])

        self.similar_thrd = similar_thrd
        self.cnt_thrd = cnt_thrd
        self.balance_para = balance_para

        self.pos_corpus=pos_corpus
        self.nlp=nlp

    def run(self):

        LF_cnt_dict = dict([])
        PR_cnt_dict = dict([])

        sorted_LF_cnt = None
        sorted_PR_cnt = None
        ls_sorted_LF_cnt = None
        ls_sorted_PR_cnt = None

        for item in tqdm_notebook(self.pos_corpus):
            if item[1] == self.leaftag:
                if item[0] in LF_cnt_dict.keys():
                    LF_cnt_dict[item[0]] = LF_cnt_dict[item[0]] + 1
                else:
                    LF_cnt_dict[item[0]] = 1
            elif item[1] == self.parenttag:
                if item[0] in PR_cnt_dict.keys():
                    PR_cnt_dict[item[0]] = PR_cnt_dict[item[0]] + 1
                else:
                    PR_cnt_dict[item[0]] = 1

        sorted_LF_cnt = sorted(list(LF_cnt_dict.items()), key=lambda x: -x[1])
        sorted_PR_cnt = sorted(list(PR_cnt_dict.items()), key=lambda x: -x[1])
        ls_sorted_LF_cnt = list(zip(*sorted_LF_cnt))
        ls_sorted_PR_cnt = list(zip(*sorted_PR_cnt))
        print(len(sorted_LF_cnt), len(sorted_PR_cnt))

        for item in tqdm_notebook(sorted_LF_cnt):
            if item[1] > self.cnt_thrd and item[0] in self.nlp.w2v_dict.keys():
                vecs = self.nlp.cal_v2w_gpu(self.nlp.w2v_dict[item[0]])
                for vec_item in vecs:
                    if vec_item[1] > self.similar_thrd and vec_item[0] in ls_sorted_PR_cnt[0] \
                            and PR_cnt_dict[vec_item[0]] > LF_cnt_dict[item[0]] / self.balance_para:
                        self.LF2PR_dict[item[0]] = vec_item[0]
                        break

        LF2PR_dict_2 = dict([])
        for k, v in self.LF2PR_dict.items():
            if k != v:
                LF2PR_dict_2[k] = v
        self.LF2PR_dict = LF2PR_dict_2
        print(self.LF2PR_dict, len(self.LF2PR_dict))

    def save(self, title):
        save_data(self.LF2PR_dict, title)

class WikiTextProcess(object):
    """
    Advanced NLP data processing object
    """
    def __init__(self,path_to_raw_data,path_to_workspace):
        """
        A NLP Data processing class should link to a NLPutil object when initialization
        :param NLPutilC:
        """
        self.path_to_raw_data=path_to_raw_data
        self.path_to_workspace=path_to_workspace

    def file_iterator(self):
        for ii in range(2):
            print("File wiki_"+str(ii))
            file_r=open(os.path.join(self.path_to_raw_data,"wiki_"+str(ii)),"r")
            file_w=open(os.path.join(self.path_to_workspace,"wiki_"+str(ii)),"w+")
            self.line_iterator(file_r,file_w)
            file_r.close()
            file_w.close()

    def line_iterator(self,file_r,file_w):
        lines=file_r.readlines()
        for line in lines:
            line=self.remove_url(line)
            line = self.addspace(line)
            line = self.remove_empty_line(line)
            file_w.write(line)

    def remove_url(self,line):
        if line.startswith("<") and line.endswith(">\n"):
            return ""
        else:
            return line

    def remove_empty_line(self,line):
        if line=="\n":
            return ""
        else:
            return line

    def addspace(self,line):
        line=line.replace("-"," - ")
        line=line.replace(", ", " , ")
        line=line.replace(". ", " . ")
        line = line.replace(".\n", " .\n")
        line = line.replace("–", " - ")
        line = line.replace("—", " - ")
        line = line.replace(" (", " ( ")
        line = line.replace(") ", " ) ")
        line = line.replace("\" ", " \" ")
        line = line.replace(" \"", " \" ")
        line = line.replace(": ", " : ")
        line = line.replace(":\n", " :\n")
        line = line.replace("! ", " ! ")
        line = line.replace("? ", " ? ")
        line = line.replace("; ", " ; ")
        line = line.replace("\'s ", " \'s ")
        return line

class StanfordSentimentUtil(object):
    """
    Advanced NLP data processing object
    """
    def __init__(self,path_to_raw_data="/home/hezq17/MySourceCode/data/SentimentTreeBank/stanfordSentimentTreebank"):
        """
        A NLP Data processing class should link to a NLPutil object when initialization
        :param NLPutilC:
        """
        self.path_to_raw_data=path_to_raw_data
        self.dict_phrases2senti=dict([])
        self.dict_phrases2senti_discrete = dict([])

    def dataset_extract(self):
        """
        Extract the data into (word, senti_class) form
        senti_class is [--,-,0,+,++] 5 classes
        :return:
        """

        ## Getting dictionary.txt data and sentiment_labels.txt data
        dict_file=os.path.join(self.path_to_raw_data,"dictionary.txt")
        dict_phrases=dict([])
        with open(dict_file,"r") as dict_f:
            lines=dict_f.readlines()
            for line in lines:
                line_splt=line.split("|")
                line_splt[1]=line_splt[1].replace("\n","")
                assert len(line_splt)==2
                dict_phrases[line_splt[0]]=line_splt[1]

        senti_l_file=os.path.join(self.path_to_raw_data,"sentiment_labels.txt")
        dict_sentilab = dict([])
        with open(senti_l_file,"r") as senti_l_f:
            lines=senti_l_f.readlines()
            for line in lines:
                line_splt=line.split("|")
                line_splt[1] = line_splt[1].replace("\n", "")
                assert len(line_splt)==2
                dict_sentilab[line_splt[0]]=line_splt[1]

        ## Combine the two dictionaries
        for key,value in dict_phrases.items():
            self.dict_phrases2senti[key]=dict_sentilab[value]

        for item in self.dict_phrases2senti.items():
            self.dict_phrases2senti_discrete[item[0]] = int(4.9*float(item[1]))

        ## Get data sentences and add tag

        data_text = os.path.join(self.path_to_raw_data, "datasetSentences.txt")
        text_sentiment_l=[]
        with open(data_text, "r") as text_f:
            lines = text_f.readlines()
            for line in lines:
                line_splt = line.split("\t")
                assert len(line_splt)==2
                line_splt[1] = line_splt[1].replace("\n", "")
                wrd_splt=line_splt[1].split(" ")
                for wrd in wrd_splt:
                    senti_point=self.dict_phrases2senti.get(wrd,0.5)
                    senti_point=int(4.9*float(senti_point))
                    text_sentiment_l.append((wrd,senti_point))

        return text_sentiment_l

class IMDbSentimentUtil(object):
    """
    Object handling IMDb database
    """
    def __init__(self,path_to_raw_data="/home/hezq17/MySourceCode/data/aclImdb"):
        """
        A NLP Data processing class should link to a NLPutil object when initialization
        :param NLPutilC:
        """
        self.path_to_raw_data=path_to_raw_data
        self.senti_dict=None
        self.corpus_train=None
        self.corpus_test=None

    def gen_wrd2senti_dict(self):
        """
        Generate word to sentiment rating from vocab and vocabEr
        :return:
        """
        senti_vocab_file=os.path.join(self.path_to_raw_data,"imdb.vocab")
        with open(senti_vocab_file,"r") as senti_v_f:
            lines=senti_v_f.readlines()
            vocab_l=[strg.replace("\n", "") for strg in lines]
            print(len(vocab_l))

        senti_rate_file = os.path.join(self.path_to_raw_data, "imdbEr.txt")
        with open(senti_rate_file,"r") as senti_r_f:
            lines=senti_r_f.readlines()
            rate_l=[float(rate.replace("\n", "")) for rate in lines]
            print(len(rate_l))

        assert len(vocab_l) == len(rate_l)

        self.senti_dict=dict([])
        for ii,wrd in enumerate(vocab_l):
            self.senti_dict[wrd] = rate_l[ii]

        return vocab_l,rate_l

    def get_corpus(self):
        """
        Getting train, test, unsupervised corpus
        :return:
        """
        train_pos = os.path.join(self.path_to_raw_data,"train/pos/")
        train_neg = os.path.join(self.path_to_raw_data, "train/neg/")
        train_unsup = os.path.join(self.path_to_raw_data, "train/unsup/")
        self.corpus_train = []
        for pathit in [train_pos,train_neg,train_unsup]:
            print(pathit)
            for filename in tqdm_notebook(os.listdir(pathit)):
                filename_f = os.path.join(pathit, filename)
                with open(filename_f, "r") as file:
                    lines = file.read()
                    tokens = word_tokenize(lines)
                    tokensl = [w.lower() for w in tokens]
                    for wrd in tokensl:
                        self.corpus_train .append(wrd)
        print(len(self.corpus_train))

        test_pos = os.path.join(self.path_to_raw_data, "test/pos/")
        test_neg = os.path.join(self.path_to_raw_data, "test/neg/")

        self.corpus_test = []
        for pathit in [test_pos, test_neg]:
            print(pathit)
            for filename in tqdm_notebook(os.listdir(pathit)):
                filename_f = os.path.join(pathit, filename)
                with open(filename_f, "r") as file:
                    lines = file.read()
                    tokens = word_tokenize(lines)
                    tokensl = [w.lower() for w in tokens]
                    for wrd in tokensl:
                        self.corpus_test.append(wrd)
        print(len(self.corpus_test))

    def corpus_clean(self):
        # clean_list=[",",".","/","<",">","br","!","?",":","...","(",")",";","-","\'","\"","--",'``',"\'\'"]
        clean_list = [ "/", "<", ">", "br"]

        cleaned_train=[]
        print("Cleaning Corpus Train ...")
        for wrd in tqdm_notebook(self.corpus_train):
            if wrd not in clean_list:
                cleaned_train.append(wrd)
        self.corpus_train = cleaned_train

        cleaned_test = []
        print("Cleaning Corpus Test ...")
        for wrd in tqdm_notebook(self.corpus_test):
            if wrd not in clean_list:
                cleaned_test.append(wrd)
        self.corpus_test = cleaned_test








