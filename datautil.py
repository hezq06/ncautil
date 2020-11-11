"""
Utility for data processing and enhancement
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.animation import writers,ArtistAnimation
import os
import pickle, json
import time
import copy

import torch
from torch.autograd import Variable
import pickle

from tqdm import tqdm,tqdm_notebook

from torchvision import datasets
from PIL import Image
from pycocotools import mask as coco_mask
from ncautil.ncamath import check_shape
from ncautil.nlputil import NLPutil

from ncautil.seqmultip3 import BiGRU_NLP,GSVIB_InfoBottleNeck,ABS_CNN_SEQ,CAL_LOSS,Softmax_Sample,ABS_CNN_COOP,Multitube_FF_MLP,Gsoftmax_Sample
from ncautil.seqmultip3 import Multitube_BiGRU_MLP
from ncautil.seqmultip2 import FF_MLP


def save_data(data,file,large_data=False):
    if not large_data:
        pickle.dump(data, open(file, "wb"))
        print("Data saved to ", file)
    else:
        pickle.dump(data, open(file, "wb"), protocol=4)
        print("Large Protocal 4 Data saved to ", file)

def load_data(file):
    data = pickle.load(open(file, "rb"))
    print("Data load from ", file)
    return data

def save_model(model,file):
    torch.save(model.state_dict(), file)
    print("Model saved to ", file)
    if hasattr(model,"save_para"):
        if model.save_para is not None:
            file_p=file+str(".para")
            # save_para={
            #     "model_para":model.model_para,
            #     "type":type(model)
            #
            # }
            # model.model_para["type"]=type(model)
            # model.model_para["misc_para"] = model.misc_para
            save_data(model.save_para,file_p)
        else:
            print("Attribute save_para is None.")
    else:
        print("Attribute save_para not found.")

def load_model(model,file,map_location=None,except_list=[]):
    try:
        if map_location is None:
            state_dict=torch.load(file)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(file,map_location=map_location))
        print("Model load from ", file)
    except Exception as inst:
        print(inst)
        pretrained_dict = torch.load(file)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k not in except_list)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def build_model_from_para(save_para, leaf_map_dict=None):
    print(save_para["type"])
    trial_list = ["ABS_CNN_COOP", "ABS_CNN_SEQ", "Gsoftmax_Sample", "Multitube_FF_MLP", "Softmax_Sample", "FF_MLP",
                  "BiGRU_NLP"]
    if leaf_map_dict is None:
        leaf_map_dict={"leaf_cnt":0, "id_map":dict([])}

    typestr = save_para["type"]
    if type(save_para["model_para"]) is not list:
        hit_flag = False
        for ii in range(len(trial_list)):
            if trial_list[ii] in typestr:
                hit_flag = True
                typeclass = eval(trial_list[ii])
                mid = save_para["id"]
                if leaf_map_dict["id_map"].get(mid,None) is None: # not yet initialized
                    pt_class = typeclass(save_para["model_para"], para=save_para["misc_para"])
                    leaf_name=trial_list[ii]+"_"+str(leaf_map_dict["leaf_cnt"])
                    leaf_map_dict["leaf_cnt"] += 1
                    leaf_map_dict[leaf_name]=pt_class
                    leaf_map_dict["id_map"][mid] = leaf_name
                else:
                    leaf_name = leaf_map_dict["id_map"][mid]
                    pt_class = leaf_map_dict[leaf_name]
                return pt_class,leaf_map_dict
        if not hit_flag:
            raise Exception("%s isn't known" % typestr)
    else:
        assert len(save_para["model_para"]) == 2
        pt_class0,leaf_map_dict = build_model_from_para(save_para["model_para"][0],leaf_map_dict)
        pt_class1,leaf_map_dict = build_model_from_para(save_para["model_para"][1],leaf_map_dict)
        hit_flag = False
        for ii in range(len(trial_list)):
            if trial_list[ii] in typestr:
                hit_flag = True
                typeclass = eval(trial_list[ii])
                pt_class = typeclass(pt_class0, pt_class1, para=save_para["misc_para"])
                return pt_class,leaf_map_dict
        if not hit_flag:
            raise Exception("%s isn't known" % typestr)

def create_dataset_sup(train_data, train_label, valid_data, valid_label, test_data, test_label):

    ### IMDb original data

    # Training set
    assert len(train_data) == len(train_label)
    dataset_train_sup = dict([])
    dataset_train_sup["dataset"] = train_data
    dataset_train_sup["label"] = train_label

    # Valid set
    assert len(valid_data) == len(valid_label)
    dataset_valid_sup = dict([])
    dataset_valid_sup["dataset"] = valid_data
    dataset_valid_sup["label"] = valid_label

    # Test set
    assert len(test_data) == len(test_label)
    dataset_test_sup = dict([])
    dataset_test_sup["dataset"] = test_data
    dataset_test_sup["label"] = test_label

    dataset_dict = {"data_train": dataset_train_sup, "data_valid": dataset_valid_sup, "data_test": dataset_test_sup}

    return dataset_dict

def get_id_with_sample_vec(param,vec):
    """
    Working together with onehot_data_enhance_create_param
    get the id a certain sampled vec
    :param param:
    :param vec:
    :return:
    """
    lsize=param.shape[0]
    dim=param.shape[-1]-1
    delts = param[:, :dim] - vec
    dists = np.zeros(lsize)
    for dim_ii in range(dim):
        dists = dists + delts[:, dim_ii] ** 2
    dists = np.sqrt(dists)
    adjdists = dists / param[:, -1]
    argm = np.argmin(adjdists)
    return argm


def onehot_data_enhance_create_param(prior,dim=3,figure=False):
    """
    Create the Afinity scaled K-means like clustering parameter when given prior
    :param prior:
    :param dim:
    :return:
    """
    prior=np.array(prior)/np.sum(np.array(prior))
    lsize=len(prior)

    param = np.random.random((lsize, dim+1))
    lr = 0.05

    for iter in tqdm(range(100)):
        res = [[] for ii in range(lsize)]
        for ii in range(lsize):
            res[ii].append(param[ii, :dim])
        for ii in range(10000):
            dot = np.random.random(dim)
            argm=get_id_with_sample_vec(param,dot)
            res[argm].append(dot)
        caled_dist = np.zeros(lsize)
        for ii in range(lsize):
            caled_dist[ii] = len(res[ii])
        caled_dist = caled_dist / np.sum(caled_dist)
        delta = caled_dist - prior
        param[:, -1] = param[:, -1] - lr * delta

    if figure:
        for ii in range(lsize):
            plt.scatter(np.array(res[ii])[:, 0], np.array(res[ii])[:, 1])
        for ii in range(lsize):
            plt.scatter(param[ii, 0], param[ii, 1], marker='^')
        plt.show()

    return param

def onehot_data_enhance(dataset,prior,dim=3,param=None):
    """
    Enhancement of one-hot dataset [0,12,5,......] using Afinity scaled K-means like clustering
    input is one hot dataset, output is distributed data where clusters of data contains one-hot label information
    :param dataset_onehot:
    :param dim: enhancement dimention
    :return:
    """
    lsize=len(set(dataset))

    print("Building param ...")
    if param is None:
        param=onehot_data_enhance_create_param(prior,dim=dim)
    res_datavec=[]
    print("Enhancing data...")
    hitcnt=0
    misscnt=0
    for iiw in tqdm(range(len(dataset))):
        wid=dataset[iiw]
        hitlab=False
        sample = np.random.random(dim)
        for ii_trial in range(10):
            argm = get_id_with_sample_vec(param, sample)
            if argm==wid:
                hitlab=True
                res_datavec.append(sample)
                hitcnt=hitcnt+1
                break
            else:
                sample = (sample + param[wid, :dim]) / 2
        if ii_trial==9 and (not hitlab):
            res_datavec.append(param[wid,:dim])
            misscnt=misscnt+1
    print("Hit:",hitcnt,"Miss:",misscnt)
    return res_datavec,param

def data_padding(data,endp="#"):
    """
    End data padding with endp
    :param data:
    :param endp:
    :return:
    """
    lenlist=[len(sent) for sent in data]
    maxlen=np.max(np.array(lenlist))
    res=[]
    for sent in data:
        npad=maxlen-len(sent)
        for ii in range(npad+1):
            sent.append(endp)
        res.append(sent)
    return res

def plot_anim(mat_list,file=None,clim=None,interval=200):
    """
    Creat an animation clip from a list of matrix
    :param mat_list:
    :return:
    """
    fig2, ax = plt.subplots()
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Harry'), bitrate=1800)

    ims = []
    for ii in range(len(mat_list)):
        fig = plt.imshow(mat_list[ii], cmap='seismic', clim=clim)
        title = plt.text(0.5, 1.01, "Step " + str(ii), ha="center", va="bottom", transform=ax.transAxes,fontsize="large")
        ims.append((fig, title))

    plt.colorbar(fig)
    im_ani = ArtistAnimation(fig2, ims, interval=interval, repeat=False)
    if file is not None:
        im_ani.save('CorrEvl_hidden0_grad_permute.mp4', writer=writer)
    plt.show()
    return im_ani

class Plus_dataset(object):
    """
    A object creating NNN+NNN=NNN dataset for seq2seq model understanding
    """
    def __init__(self):
        self.num=0
        self.reset()


    def reset(self):
        self.dataset_raw = dict([])
        self.dataset_raw["dataset"] = []
        self.dataset_raw["label"] = []
        self.dataset_sup = dict([])
        self.dataset_sup["dataset"] = []
        self.dataset_sup["label"] = []
        self.digits = None

        self.memmat=None

    def create_dataset(self,num,digits=3,mode="normal",noise_level=None):
        """
        create dataset
        :param num: with number of example to be num
        :param max_range: and maximum digit less than max_range
        :return:
        """
        dataset=[]
        label=[]
        self.memmat=np.zeros((10**digits-1,10**digits-1))
        for ii in range(num):
            dig1=int(np.random.rand()*(10**digits-1))
            dig2 = int(np.random.rand() * (10**digits-1))
            while self.memmat[dig1,dig2]==1: # overlap detection
                dig1 = int(np.random.rand() * (10 ** digits - 1))
                dig2 = int(np.random.rand() * (10 ** digits - 1))
            self.memmat[dig1, dig2]=1
            if mode=="normal":
                dig_ans=int(dig1+dig2)
            elif mode=="random":
                dig_ans=int(np.random.rand() * (10**(digits+1)-1))
            elif mode=="noisy":
                assert noise_level is not None
                if np.random.rand()>noise_level:
                    dig_ans = int(dig1 + dig2)
                else:
                    dig_ans = int(np.random.rand() * (10 ** (digits + 1) - 1))
            else:
                raise Exception("Unknown mode")
            str1=str(dig1)
            # npad = digits - len(str1)
            # for ii in range(npad):
            #     str1 = "0"+str1
            str2 = str(dig2)
            # npad = digits - len(str2)
            # for ii in range(npad):
            #     str2 = "0" + str2
            strdata=str1+"+"+str2+"="
            dataset.append(list(strdata))
            ansdata=str(dig_ans)
            # npad = digits+ 1 - len(ansdata)
            # for ii in range(npad):
            #     ansdata = "0"+ansdata
            label.append(list(ansdata))
        self.num=self.num+num
        self.dataset_raw["dataset"]=data_padding(dataset)
        self.dataset_raw["label"]=data_padding(label)
        self.data_precess()
        self.digits=digits

    def create_dataset_simple(self,num,digits=3,mode="normal",noise_level=None):
        """
        create dataset
        :param num: with number of example to be num
        :param max_range: and maximum digit less than max_range
        :return:
        """
        dataset=[]
        label=[]
        self.memmat=np.zeros((10**digits-1,10**digits-1))
        for ii in range(num):
            dig1=int(np.random.rand()*(10**digits-1))
            dig2 = int(np.random.rand() * (10**digits-1))
            while self.memmat[dig1,dig2]==1: # overlap detection
                dig1 = int(np.random.rand() * (10 ** digits - 1))
                dig2 = int(np.random.rand() * (10 ** digits - 1))
            self.memmat[dig1, dig2]=1
            if mode=="normal":
                dig_ans=int(dig1+dig2)
            elif mode=="random":
                dig_ans=int(np.random.rand() * (10**(digits+1)-1))
            elif mode=="noisy":
                assert noise_level is not None
                if np.random.rand()>noise_level:
                    dig_ans = int(dig1 + dig2)
                else:
                    dig_ans = int(np.random.rand() * (10 ** (digits + 1) - 1))
            else:
                raise Exception("Unknown mode")
            str1=str(dig1)
            npad = digits - len(str1)
            for ii in range(npad):
                str1 = "0"+str1
            str2 = str(dig2)
            npad = digits - len(str2)
            for ii in range(npad):
                str2 = "0" + str2
            strdata=str1+str2
            dataset.append(list(strdata))
            ansdata=str(dig_ans)
            npad = digits+ 1 - len(ansdata)
            for ii in range(npad):
                ansdata = "0"+ansdata
            label.append(list(ansdata))
        self.num=self.num+num
        # self.dataset_raw["dataset"]=data_padding(dataset)
        # self.dataset_raw["label"]=data_padding(label)
        self.dataset_raw["dataset"]=dataset
        self.dataset_raw["label"]=label
        self.data_precess()
        self.digits=digits

    def data_precess(self):
        """
        Transfer data to digits
        :return:
        """
        wrd_2_id = dict([])
        for ii in range(10):
            wrd_2_id[str(ii)] = ii
        wrd_2_id["+"] = 10
        wrd_2_id["="] = 11
        wrd_2_id["#"] = 12

        for sent in self.dataset_raw["dataset"]:
            trans_sent=[]
            for chr in sent:
                trans_sent.append(wrd_2_id[chr])
            self.dataset_sup["dataset"].append(trans_sent)

        for sent in self.dataset_raw["label"]:
            trans_sent=[]
            for chr in sent:
                trans_sent.append(wrd_2_id[chr])
            self.dataset_sup["label"].append(trans_sent)

    # def data_precess_v2(self):
    #     """
    #     Transfer data to digits
    #     :return:
    #     """
    #     wrd_2_id = dict([])
    #     for ii in range(10):
    #         wrd_2_id[str(ii)] = ii
    #     wrd_2_id["+"] = 10
    #     wrd_2_id["="] = 11
    #     wrd_2_id["#"] = 12
    #
    #     for ii in range(len(self.dataset_raw["dataset"])):
    #         trans_sent=[]
    #         for chr in self.dataset_raw["dataset"][ii]:
    #             trans_sent.append(wrd_2_id[chr])
    #         for chr in self.dataset_raw["label"][ii]:
    #             trans_sent.append(wrd_2_id[chr])
    #         del trans_sent[-1]
    #         self.dataset_sup["dataset"].append(trans_sent)
    #
    #     for sent in self.dataset_raw["label"]:
    #         trans_sent=[]
    #         for chr in sent:
    #             trans_sent.append(wrd_2_id[chr])
    #         self.dataset_sup["label"].append(trans_sent)


    def print_example(self,num):
        """
        print num of examples
        :param num:
        :return:
        """
        print("Number of data is",self.num)
        print("digits is,", self.digits)
        for ii in range(num):
            idn=int(np.random.rand()*self.num)
            print("Q:",self.dataset_raw["dataset"][idn])
            print("A:", self.dataset_raw["label"][idn])

class MNIST_dataset(object):
    """
    pytorch mnist dataset
    """
    def __init__(self,download=False):

        data_train = datasets.MNIST(root="./data/", train=True, download=download)
        data_test = datasets.MNIST(root="./data/", train=False, download=download)

        self.dataset_sup=dict([])
        dshape=data_train.train_data.shape
        self.dataset_sup["dataset"] = data_train.train_data.reshape(dshape[0],-1).type(torch.FloatTensor) # 1D version
        self.dataset_sup["dataset"] = self.dataset_sup["dataset"] / 256.0
        self.dataset_sup["label"] = data_train.train_labels

        self.dataset_sup_test=dict([])
        dshape = data_test.test_data.shape
        self.dataset_sup_test["dataset"] = data_test.test_data.reshape(dshape[0],-1).type(torch.FloatTensor)
        self.dataset_sup_test["dataset"] = self.dataset_sup_test["dataset"] / 256.0
        self.dataset_sup_test["label"] = data_test.test_labels

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_files, window = 128, get_mode="POS", nVcab=30000, nlputil_ls=None):
        """
        IMDbDataset util
        :param data_path: /storage/hezq17/IMDbDataset
        :param data_files: train: IMDb_corpus_pos.data, imdb_senti_train.data,
                            test: IMDb_corpus_test_pos.data, imdb_senti_test.data
        :param get_mode:
        """
        self.window = window
        self.get_mode = get_mode

        print("Loading IMDb dataset...")
        path_pos = os.path.join(data_path, data_files[0])
        pos_corpus_tup = load_data(path_pos)

        path_senti = os.path.join(data_path, data_files[1])
        senti_corpus_tup = load_data(path_senti)

        self.sanity_check(pos_corpus_tup,senti_corpus_tup)

        corpus, pos_corpus = zip(*pos_corpus_tup)
        corpus, senti = zip(*senti_corpus_tup)

        self.nVcab = nVcab
        if nlputil_ls is None:
            self.nlp_text = NLPutil()
            self.nlp_text.set_corpus(corpus)
            w2id, id2w = self.nlp_text.build_vocab(nVcab=nVcab)
            w2v_dict = self.nlp_text.build_w2v(mode="torchnlp")
            self.nlp_text.build_pt_emb()
        else:
            print("Shared NLPUtil detected.")
            self.nlp_text = nlputil_ls[0]
            w2id = self.nlp_text.word_to_id
            id2w = self.nlp_text.id_to_word

        self.dataset = [w2id.get(item, nVcab) for item in corpus]

        if nlputil_ls is None:
            self.nlp_pos = NLPutil()
            self.nlp_pos.set_corpus(pos_corpus)
            w2id_pos, id2w_pos = self.nlp_pos.build_vocab()
        else:
            self.nlp_pos = nlputil_ls[1]
            w2id_pos = self.nlp_pos.word_to_id
            id2w_pos = self.nlp_pos.id_to_word
        self.pos_only = [w2id_pos.get(item, 0) for item in pos_corpus]

        self.senti = senti

        self.id2w = id2w
        self.id2w_pos = id2w_pos

    def sanity_check(self,pos_corpus_tup,senti_corpus_tup):
        print("Doing sanity check...")
        assert len(pos_corpus_tup) == len(senti_corpus_tup)
        for ii in range(100):
            iip = int(np.random.rand()*len(pos_corpus_tup))
            assert pos_corpus_tup[iip][0] == senti_corpus_tup[iip][0]

    def __getitem__(self, idx):
        """
        Get block idx
        :param idx:
        :return: [window, l_size]
        """
        startp = self.window * idx
        endp = self.window * (idx+1)

        ptdata = self.dataset[int(startp):int(endp)]
        textblock_vecs = self.nlp_text.pt_emb(torch.LongTensor(ptdata))

        if self.get_mode=="POS":
            pos_p = self.pos_only[int(startp):int(endp)]
            pt_pos = torch.LongTensor(pos_p)
            return textblock_vecs,pt_pos
        elif self.get_mode=="Senti":
            senti_p = self.senti[int(startp):int(endp)]
            pt_senti = torch.LongTensor(senti_p)
            return textblock_vecs, pt_senti
        elif self.get_mode=="Autoencode":
            word_p = self.dataset[int(startp):int(endp)]
            pt_word = torch.LongTensor(word_p)
            return textblock_vecs, pt_word
        elif self.get_mode=="senti_auto":
            senti_p = self.senti[int(startp):int(endp)]
            pt_senti = torch.LongTensor(senti_p)
            word_p = self.dataset[int(startp):int(endp)]
            pt_word = torch.LongTensor(word_p)
            senti_word = torch.stack([pt_senti,pt_word],dim=-1)
            return textblock_vecs, senti_word
        elif self.get_mode=="wordlabel_pos":
            pos_p = self.pos_only[int(startp):int(endp)]
            return ptdata, pos_p
        elif self.get_mode=="AutoencodeWV":
            return textblock_vecs, textblock_vecs

    def __len__(self):
        """
        number of text
        :return:
        """
        numblocks = np.floor(len(self.dataset)/self.window)
        return int(numblocks)


class ClevrDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_path, get_mode="maskedimage_posishapematerial"):
        """
        Clevr Dataset Util
        :param path: pictures path
        :param json_path: path for json file
        :param img_seg_path: pretrain CLEVR segmentation model
        """
        # json_path = ""
        # data_path_scene_train = "/storage/hezq17/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
        # data_path_scene_val = "/storage/hezq17/CLEVR_v1.0/scenes/CLEVR_val_scenes.json"
        # data_path_images_train = "/storage/hezq17/CLEVR_v1.0/images/train"
        # data_path_images_val = "/storage/hezq17/CLEVR_v1.0/images/val"
        # img_seg_path = "/storage/hezq17/CLEVR_v1.0/seg_model/Mask_RCNN_ClevrMiniTrained.model"
        self.image_path = image_path

        # with open(json_path) as f:
        #     self.json_clevr = json.load(f)
        self.json_clevr = load_data(json_path)
        self.img_size = [320, 480] # [H, W]

        # self.seg_model = None
        # if img_seg_path is not None:
        #     self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        #     # get the model using our helper function
        #     self.seg_model = self.get_instance_segmentation_model()
        #     # move model to the right device
        #     load_model(self.seg_model, img_seg_path)
        #     self.seg_model.to(self.device)
        #     self.seg_model.eval()

        self.masks = []
        self.imgs = []
        print("Parsing dataset ...")
        for iis in range(len(self.json_clevr["scenes"])):
            self.imgs.append(self.json_clevr["scenes"][iis]["image_filename"])

        self.get_mode = get_mode

        self.color_map={
            "gray":0, "blue":1, "brown":2, "yellow":3, "red":4, "green":5, "purple":6, "cyan":7
        }
        self.shape_map = {
            "cube": 0, "cylinder": 1, "sphere": 2
        }
        self.material_map = {
            "metal":0, "rubber":1
        }

    def __getitem__(self, idx):
        if self.get_mode == "full":
            return self.getitem_full(idx)
        elif self.get_mode in ["maskedimage_colorshape","maskedimage_posicolorshape","maskedimage_posicolormaterial","auto_encode","maskedimage_posishapematerial"
                               ,"maskedimage_shape","maskedimage_color", "maskedimage_posi","auto_encode_focused"]:
            return self.getitem_mask_posicolorshape(idx)
        elif self.get_mode == "whole_pic_auto":
            return self.getitem_mask_wholepicauto(idx)
        elif self.get_mode == "hidden_only_no_pic_posi":
            return self.hidden_only_no_pic_posi(idx)

    def getitem_full(self, idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # mask_t = np.zeros(self.img_size)
        # if self.seg_model is not None:
        #     imgt = torch.from_numpy(img / 255).type(torch.FloatTensor).permute(2, 0, 1)
        #     with torch.no_grad():
        #         prediction = self.seg_model([imgt.to(self.device)])
        #     Nobj = len(prediction[0]['masks'][:, 0])
        #     for iio in range(Nobj):
        #         mask = prediction[0]['masks'][iio, 0].mul(255).cpu().numpy() * (iio + 1)
        #         mask_t = mask_t + mask

        json_scene = self.json_clevr["scenes"][idx]

        mask_t = np.zeros(self.img_size)
        for iio in range(len(json_scene["objects"])):
            rle = json_scene["objects"][iio]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            mask = mask*(iio + 1)
            mask_t = mask_t+mask

        return img, mask_t, json_scene

    def getitem_mask_posicolorshape(self,idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0

        json_scene = self.json_clevr["scenes"][idx]

        # masked_imgl=[]
        # colorsl=[]
        # for iio in range(len(json_scene["objects"])):
        #     rle = json_scene["objects"][iio]["mask"]
        #     compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
        #     mask = coco_mask.decode(compressed_rle)
        #     masked_img = img*mask.reshape(self.img_size+[1])
        #     masked_imgl.append(masked_img)
        #     color = json_scene["objects"][iio]["color"]
        #     colorsl.append(self.color_map[color])
        # print("Len",len(masked_imgl),len(colorsl))
        # return masked_imgl,colorsl

        Nobj = len(json_scene["objects"])
        objp = int(np.random.rand()*Nobj)
        rle = json_scene["objects"][objp]["mask"]
        compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
        mask = coco_mask.decode(compressed_rle)
        self.img = img
        self.mask = mask
        masked_img = img * mask.reshape(self.img_size + [1])
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
        posix = json_scene["objects"][objp]["pixel_coords"][0] / self.img_size[1]
        posiy = json_scene["objects"][objp]["pixel_coords"][1] / self.img_size[0]
        color = self.color_map[json_scene["objects"][objp]["color"]]
        shape = self.shape_map[json_scene["objects"][objp]["shape"]]
        material = self.material_map[json_scene["objects"][objp]["material"]]

        if self.get_mode =="maskedimage_posicolorshape":
            return masked_img, np.array([float(posix),float(posiy),color,shape])

        elif self.get_mode =="maskedimage_colorshape":
            return masked_img, np.array([color,shape])

        elif self.get_mode == "maskedimage_posicolormaterial":
            return masked_img, np.array([float(posix),float(posiy),color, material])

        elif self.get_mode == "maskedimage_posishapematerial":
            return masked_img, np.array([float(posix), float(posiy), shape, material])

        elif self.get_mode == "maskedimage_shape":
            return masked_img, shape

        elif self.get_mode == "maskedimage_color":
            return masked_img, color

        elif self.get_mode == "maskedimage_posi":
            if posix>2/3:
                posi_right=1
            else:
                posi_right=0
            return masked_img, posi_right

        elif self.get_mode == "auto_encode":
            return masked_img, masked_img

        elif self.get_mode == "auto_encode_focused":
            xrange = 75
            yrange = 75
            pixposix = json_scene["objects"][objp]["pixel_coords"][0]
            pixposiy = json_scene["objects"][objp]["pixel_coords"][1]
            xcropstart = pixposix-xrange
            xcropend = pixposix + xrange
            ycropstart = pixposiy - yrange
            ycropend = pixposiy + yrange
            # Boundary handling
            xlowshift=0
            if xcropstart<0:
                xlowshift = 0-xcropstart
                xcropstart=0
            xhighshift=0
            if xcropend>self.img_size[1]:
                xhighshift=xcropend-self.img_size[1]
                xcropend = self.img_size[1]
            ylowshift=0
            if ycropstart<0:
                ylowshift = 0-ycropstart
                ycropstart = 0
            yhighshift=0
            if ycropend>self.img_size[0]:
                yhighshift = ycropend-self.img_size[0]
                ycropend = self.img_size[0]
            focused_masked_img = torch.zeros((3, 2*xrange, 2*yrange))
            focused_masked_img[:,ylowshift:2*yrange-yhighshift,xlowshift:2*xrange-xhighshift] = masked_img[:, ycropstart:ycropend, xcropstart:xcropend]
            return masked_img, focused_masked_img

    def getitem_mask_wholepicauto(self,idx):
        assert self.get_mode == "whole_pic_auto"
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        masked_imgl = torch.zeros([10, 3] + self.img_size).type(torch.FloatTensor)
        for iio in range(Nobj):

            rle = json_scene["objects"][iio]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            masked_img = img * mask.reshape(self.img_size + [1])
            masked_img = np.transpose(masked_img, (2, 0, 1))
            masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
            masked_imgl[iio,:,:,:] = masked_img

        return masked_imgl, masked_imgl

    def hidden_only_no_pic_posi(self,idx):

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        objp = int(np.random.rand() * Nobj)
        pAutocode = torch.from_numpy(json_scene["objects"][objp]["auto_code"]).type(torch.FloatTensor)
        posix = json_scene["objects"][objp]["pixel_coords"][0] / self.img_size[1]
        if posix > 2 / 3:
            posi_right = 1
        else:
            posi_right = 0
        return pAutocode, posi_right

    def __len__(self):
        return len(self.imgs)

    def get_instance_segmentation_model(self, num_classes=2):
        """
        https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=mTgWtixZTs3X
        :param num_classes: number of classes
        :return:
        """
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model


class MultipleChoiceClevrDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_path, epoch_len = 10000, question_mode="exist_colorshape"):
        """
        Clevr Dataset Util
        :param path: pictures path
        :param json_path: path for json file
        :param img_seg_path: pretrain CLEVR segmentation model
        """

        self.image_path = image_path
        self.json_clevr = load_data(json_path)
        self.epoch_len = epoch_len
        self.img_size = [320, 480] # [H, W]
        self.question_mode = question_mode
        self.checkmode = False
        self.singlecheckmode = False

        self.imgs = []
        print("Parsing dataset ...")
        for iis in range(len(self.json_clevr["scenes"])):
            self.imgs.append(self.json_clevr["scenes"][iis]["image_filename"])

        self.color_map={"gray":0, "blue":1, "brown":2, "yellow":3, "red":4, "green":5, "purple":6, "cyan":7}
        self.shape_map = {"cube": 0, "cylinder": 1, "sphere": 2}
        self.size_map = {"small": 0, "large": 1}
        self.material_map = {"rubber": 0, "metal": 1}

    def __getitem__(self, idx_temp):
        """
        Get 4 pictures and an answaer in 0,1,2,3
        :param idx:
        :return:
        """
        ## get a random sets of imgs of a question and an answer
        if self.question_mode == "exist_colorshape":
            idxl, answ = self.question_exist_colorshapesize(color="red", shape = "cube", size = None, material = None)
        elif self.question_mode == "exist_shapesize":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "sphere", size = "large", material = None)
        elif self.question_mode == "exist_shapesize2":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "cylinder", size = "large", material = None)
        elif self.question_mode == "exist_sizematerial":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = None, size = "small", material = "rubber")
        elif self.question_mode == "exist_sizecolor":
            idxl, answ = self.question_exist_colorshapesize(color="yellow", shape = None, size = "small", material = None)
        elif self.question_mode == "exist_colormaterial":
            idxl, answ = self.question_exist_colorshapesize(color="green", shape = None, size = None, material = "metal")
        elif self.question_mode == "posimost_property":
            idxl, answ = self.question_posimost_colorshapesize(posimost = "rightmost", color=None, shape = "cylinder", size = None, material = None)
        elif self.question_mode == "posiside_property":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = None, size = None, material = "rubber", posi_side = "left")
        elif self.question_mode == "posiside_property2":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "cylinder", size = None, material = None, posi_side = "right")
        elif self.question_mode == "posiside_property3":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "sphere", size = None, material = None, posi_side = "right")
        elif self.question_mode == "posimost_property2":
            idxl, answ = self.question_posimost_colorshapesize(posimost = "rightmost", color=None, shape = None, size = "small", material = None)
        elif self.question_mode == "number_shape":
            idxl, answ = self.question_number_colorshapesize(key_aim = ["shape","cube"], num = 3)
        else:
            raise Exception("Not implemented")
        dataxl = []
        self.imgl=[]
        self.mask_tl=[]
        Nobjl = []
        singecheckl=[]
        for idx in idxl:
            # dataxl.append(img.unsqueeze(0))
            autocodel, Nobj = self.getitem_autocodeperobj(idx)
            dataxl.append(autocodel.unsqueeze(0))
            Nobjv = torch.zeros(10)
            Nobjv[:Nobj]=1
            Nobjl.append(Nobjv)
            if self.checkmode:
                print("Image getting ...")
                img = self.getitem_maskedperobj(idx)
                self.imgl.append(self.img)
                self.mask_tl.append(self.mask_t)
                self.idxl = idxl
            if self.singlecheckmode:
                # print(self.singlecheck.shape) # torch.Size([4, 10])
                singecheckl.append(self.singlecheck)

        datax = torch.cat(dataxl,dim=0)
        Nobjl = torch.stack(Nobjl)
        if self.singlecheckmode:
            singecheckmat = torch.stack(singecheckl)
            return datax, Nobjl, answ, singecheckmat
        else:
            return datax, Nobjl, answ

    def collate_fn(self, batchdata):
        datax = [item[0].view([1]+list(item[0].shape)) for item in batchdata]
        Nobjl = [item[1].view([1]+list(item[1].shape)) for item in batchdata]
        answ = [item[2] for item in batchdata]
        datax = torch.cat(datax,dim=0)
        Nobjl = torch.cat(Nobjl, dim=0)
        answ = torch.LongTensor(answ)
        if self.singlecheckmode:
            batchcheckmat = [item[3].view([1] + list(item[3].shape)) for item in batchdata]
            batchcheckmat = torch.cat(batchcheckmat, dim=0)
            return (datax, Nobjl), answ, batchcheckmat
        else:
            return (datax, Nobjl), answ

    def question_exist_colorshapesize(self, color=None, shape = "cube", size = "large", material = None, posi_side = None):

        checklist=[]
        if color is not None:
            checklist.append(["color",color])
        if shape is not None:
            checklist.append(["shape",shape])
        if size is not None:
            checklist.append(["size",size])
        if material is not None:
            checklist.append(["material",material])

        def checkposi(xy, posi_flag):
            if posi_flag == "left":
                if xy[0]/self.img_size[1]<1/3:
                    return True
                else:
                    return False
            elif posi_flag == "right":
                if xy[0]/self.img_size[1]>2/3:
                    return True
                else:
                    return False
            else:
                raise Exception("Checkposi not implemented")


        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            exist_flag = False
            for obj in json_scene[idp_exist]["objects"]:
                andFlag = True
                for itemd in checklist:
                    andFlag = (andFlag and obj[itemd[0]]==itemd[1])
                if posi_side is not None:
                    andFlag = (andFlag and checkposi(obj["pixel_coords"],posi_side))
                if andFlag:
                    exist_flag=True
                    break
            if exist_flag:
                break
        ## get 3 non-exist scene
        idp_Nexistl=[]
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            exist_flag = False
            for obj in json_scene[idp_Nexist]["objects"]:
                andFlag = True
                for itemd in checklist:
                    andFlag = (andFlag and obj[itemd[0]] == itemd[1])
                if posi_side is not None:
                    andFlag = (andFlag and checkposi(obj["pixel_coords"],posi_side))
                if andFlag:
                    exist_flag = True
                    break
            if not exist_flag:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break
        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def question_posimost_colorshapesize(self, posimost = "rightmost", color=None, shape = None, size = None, material = "metal"):

        checklist=[]
        if color is not None:
            checklist.append(["color",color])
        if shape is not None:
            checklist.append(["shape",shape])
        if size is not None:
            checklist.append(["size",size])
        if material is not None:
            checklist.append(["material",material])

        posimost_dict = {
            "leftmost":0,
            "rightmost":1,
            "farmost":2,
            "nearmost":3
        }

        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            exist_flag = False
            posimost_id = [-1, 1, -1, -1] # [leftmost, rightmost, farmost, nearmost] ids
            posimost_record = [1.0, 0.0, 1.0, 0.0]
            for iid, obj in enumerate(json_scene[idp_exist]["objects"]):
                posix = obj["pixel_coords"][0] / self.img_size[1]
                posiy = obj["pixel_coords"][1] / self.img_size[0]
                if posix<posimost_record[0]:
                    posimost_id[0] = iid
                    posimost_record[0] = posix
                if posix>posimost_record[1]:
                    posimost_id[1] = iid
                    posimost_record[1] = posix
                if posiy<posimost_record[2]:
                    posimost_id[2] = iid
                    posimost_record[2] = posix
                if posiy>posimost_record[3]:
                    posimost_id[3] = iid
                    posimost_record[3] = posix
            andFlag = True
            objpick = json_scene[idp_exist]["objects"][posimost_id[posimost_dict[posimost]]]
            for itemd in checklist:
                andFlag = (andFlag and objpick[itemd[0]]==itemd[1])
            if andFlag:
                exist_flag=True
            if exist_flag:
                break
        ## get 3 non-exist scene
        idp_Nexistl = []
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            exist_flag = False
            posimost_id = [-1, -1, -1, -1] # [leftmost, rightmost, farmost, nearmost] ids
            posimost_record = [1.0, 0.0, 1.0, 0.0]
            for iid, obj in enumerate(json_scene[idp_Nexist]["objects"]):
                posix = obj["pixel_coords"][0] / self.img_size[1]
                posiy = obj["pixel_coords"][1] / self.img_size[0]
                if posix<posimost_record[0]:
                    posimost_id[0] = iid
                    posimost_record[0] = posix
                if posix>posimost_record[1]:
                    posimost_id[1] = iid
                    posimost_record[1] = posix
                if posiy<posimost_record[2]:
                    posimost_id[2] = iid
                    posimost_record[2] = posiy
                if posiy>posimost_record[3]:
                    posimost_id[3] = iid
                    posimost_record[3] = posiy
            andFlag = True
            objpick = json_scene[idp_Nexist]["objects"][posimost_id[posimost_dict[posimost]]]
            for itemd in checklist:
                andFlag = (andFlag and objpick[itemd[0]]==itemd[1])
            if andFlag:
                exist_flag=True

            if not exist_flag:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break

        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def question_number_colorshapesize(self, key_aim = ["shape","cube"], num = 3):

        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            cnt = 0
            for obj in json_scene[idp_exist]["objects"]:
                if obj[key_aim[0]] == key_aim[1]:
                    cnt += 1
            if cnt == num:
                break
        ## get 3 non-exist scene
        idp_Nexistl=[]
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            cnt = 0
            for obj in json_scene[idp_Nexist]["objects"]:
                if obj[key_aim[0]] == key_aim[1]:
                    cnt += 1
            if cnt != num:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break
        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def getitem_maskedperobj(self,idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0
        self.img = img

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        masked_imgl = torch.zeros([10, 3]+self.img_size).type(torch.FloatTensor)

        mask_t = np.zeros(self.img_size)
        for objp in range(Nobj):
            rle = json_scene["objects"][objp]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            masked_img = img * mask.reshape(self.img_size + [1])
            masked_img = np.transpose(masked_img, (2, 0, 1))
            masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
            masked_imgl[objp,:,:,:]=masked_img
            maskid = mask * (objp + 1)
            mask_t = mask_t + maskid
        self.mask_t=mask_t

        return masked_imgl # [10, 3, 320, 480]

    def getitem_autocodeperobj(self, idx):

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        codelen = len(json_scene["objects"][0]["auto_code"])
        autocodel = torch.zeros([10, codelen]).type(torch.FloatTensor)
        for objp in range(Nobj):
            autocodel[objp,:] = torch.from_numpy(json_scene["objects"][objp]["auto_code"]).type(torch.FloatTensor)

        self.singlecheck = torch.zeros((10,4))

        posixl=[]
        for iio in range(Nobj):
            posixl.append(self.json_clevr["scenes"][idx]['objects'][iio]["pixel_coords"][0])

        iio_xmax = np.argmax(posixl)
        self.singlecheck[iio_xmax, 0] = 1

        for iio in range(Nobj):
            # if self.json_clevr["scenes"][idx]['objects'][iio]["color"]=="yellow":
            #     self.singlecheck[iio, 1]=1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["size"]=="small":
            #     self.singlecheck[iio, 3] = 1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["color"]=="red":
            #     self.singlecheck[iio, 1]=1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["shape"]=="cube":
            #     self.singlecheck[iio, 3] = 1
            if self.json_clevr["scenes"][idx]['objects'][iio]["shape"]=="cylinder":
                self.singlecheck[iio, 3] = 1
            posix = posixl[iio]
            if posix/self.img_size[1]>2/3: # right side
                self.singlecheck[iio, 0] = 1


        return autocodel, Nobj

    def __len__(self):
        return self.epoch_len