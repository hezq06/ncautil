"""
A collection of examples
Author: Harry He
"""
### Example of ipython notebook start
# %reload_ext autoreload
# %autoreload 2
# import matplotlib
# # matplotlib.use('gtk')
# %matplotlib

### Example of cuda debug
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

### Import example
import matplotlib
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from ncautil.seqmultip2 import *
from ncautil.ncalearn import *
from ncautil.datautil import *
from ncautil.ncamath import *
from ncautil.nlputil import NLPutil,SQuADutil,StanfordSentimentUtil,POS_SIMP
import torch
import nltk
from tqdm import tqdm_notebook
import collections

### Example of Corpus preparation
def Example_of_Corpus_preparation():
    nVcab=30000
    nlp=NLPutil()
    nlp.get_data("IMDb_corpus.data",format="pickle")
    # nlp.get_data("imdb")
    w2id,id2w=nlp.build_vocab(Vsize=nVcab)
    w2v_dict=nlp.build_w2v(mode="torchnlp",Nvac=nVcab)
    nlp.build_pt_emb()

    nlpt=NLPutil()
    nlpt.get_data("IMDb_corpus_test.data",format="pickle")

    pos_corpus=load_data("IMDb_corpus_pos.data")
    pos_corpusb=load_data("IMDb_corpus_test_pos.data")

    dataset = [w2id.get(item[0], nVcab) for item in pos_corpus]
    datasetb = [w2id.get(item[0], nVcab) for item in pos_corpusb]

    pos_only=[item[1] for item in pos_corpus]
    nlp_pos=NLPutil()
    nlp_pos.set_corpus(pos_only)
    w2id_pos,id2w_pos=nlp_pos.build_vocab()
    pos_only=[w2id_pos.get(item[1],0) for item in pos_corpus]
    pos_onlyb=[w2id_pos.get(item[1],0) for item in pos_corpusb]
    vocab_out=nlp_pos.nVcab

    senti_corpus = load_data("imdb_senti_train.data")
    senti_corpusb = load_data("imdb_senti_test.data")

    dataset_senti = [w2id.get(item[0], nVcab) for item in senti_corpus]
    dataset_sentib = [w2id.get(item[0], nVcab) for item in senti_corpusb]
    senti_only = [item[1] for item in senti_corpus]
    senti_onlyb = [item[1] for item in senti_corpusb]

### Example of Wordnet
### https://www.nltk.org/howto/wordnet.html
from nltk.corpus import wordnet as wn
def Example_of_Wordnet():
    print(wn.synsets("travel"))
    print(wn.morphy('ran', wn.VERB))
    print(wn.synsets("travel")[6].hyponyms())
    print(wn.synsets("travel")[6].hypernyms())
    w1 = wn.synset('ship.n.01')
    w2 = wn.synset('boat.n.01')
    print(w1.wup_similarity(w2))
    print(wn.synset('dog.n.01').pos())
    hypo = lambda s: s.hyponyms()
    hyper = lambda s: s.hypernyms()
    dog = wn.synset('dog.n.01')
    print(list(dog.closure(hypo)))
    print(dog.root_hypernyms())

### Example of word sense disabiguiation
### http://www.nltk.org/howto/wsd.html
from nltk.wsd import lesk
def Example_of_Word_Sense_Disabiguiation():
    sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']
    print(lesk(sent, 'bank', 'n'))
    # Output Synset('savings_bank.n.02')
    # running time ~32 us

### Example of POS Tagging
def Example_of_POS_Tagging(nlp):
    squ = SQuADutil()
    posl=squ.pos_tagger_fast(nlp.corpus)
    print(posl)

### Shell conda environment setup
# $ conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit==9.2 -c pytorch

### Pandas Example
import pandas as pd
def Example_of_pandas():
    expdict={
        "aaa":12,
        "bbb":15,
        "ccc":5
    }
    df = pd.DataFrame({
        "word":list(expdict.keys()),
        "cnt":list(expdict.values())
    })
    df=df.sort_values(by='cnt',ascending=False)
    print(df)
    df.to_excel("test.xlsx", index=False)
    tot = df.sum(axis=0)[1]
    df["normcnt"] = df["cnt"] / tot
    df.loc[df["word"] == "aaa"]
    for index, row in df.iterrows():
        print(row['aaa'], row['bbb'])

### Semantic role labeling
from allennlp.predictors.predictor import Predictor
def Example_of_SRL():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
    predictor.predict(
      sentence="Did Uriah honestly think he could beat the game in under three hours?"
    )

### Example of Python Profiling
import cProfile
import pstats
def Example_of_Python_Profiling(run_something):
    cProfile.run('run_something()', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs()
    p.sort_stats('cumulative')
    p.print_stats()

### ipython useful example
import os
def Example_of_iPython_Notebook():
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"

### example of numba
from numba import njit
@ njit(parallel=True)
def Example_of_numba(input_data):
    for iia in range(1000):
        for iib in range(1000):
            pass

### simple parallel example
import multiprocessing

def Example_of_Parallel(procnum, return_dict):
    """worker function"""
    print(str(procnum) + " represent!")
    return_dict[procnum] = procnum

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict() # very slow if return_dict is big
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=Example_of_Parallel, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())