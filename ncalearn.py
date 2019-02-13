"""
Python package for NCA learning algorithm
Algorithm:  Takuya Isomura, Taro Toyoizumi
Developer: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time, os, pickle

import torch
import copy
from torch.autograd import Variable

from sklearn.manifold import TSNE
import matplotlib.ticker as ticker

from wordcloud import WordCloud
import operator
from PIL import Image
from PIL import ImageDraw,ImageFont

from tqdm import tqdm

def save_data(data,file):
    pickle.dump(data, open(file, "wb"))
    print("Data saved to ", file)

def load_data(file):
    data = pickle.load(open(file, "rb"))
    print("Data load from ", file)
    return data

def pltfft(data):
    """
    Plot fft spectrum
    :param data:
    :return:
    """
    N = len(data)
    data=np.array(data).reshape(1,-1)
    fft = np.fft.rfft(data - np.mean(data),norm='ortho')
    x = np.array(list(range(len(fft[0])))) * 2 * np.pi / N
    y = abs(fft)[0]
    plt.plot(x, y)
    plt.show()

def pltsne(data,D=2,perp=1):
    """
    Plot 2D w2v graph with tsne
    Referencing part of code from: Basic word2vec example tensorflow
    :param numpt: number of points
    :return: null
    """
    tsnetrainer = TSNE(perplexity=perp, n_components=D, init='pca', n_iter=5000, method='exact')
    tsne = tsnetrainer.fit_transform(data)
    plt.figure()
    for i in range(len(data)):
        x, y = tsne[i, :]
        plt.scatter(x, y)
    plt.show()
    return tsnetrainer


def plot_mat(data,start=0,lim=1000,symmetric=False):
    data=np.array(data)
    assert len(data.shape) == 2
    img=data[:,start:start+lim]
    if symmetric:
        plt.imshow(img, cmap='seismic',clim=(-np.amax(data), np.amax(data)))
    else:
        plt.imshow(img, cmap='seismic')
    plt.colorbar()
    plt.show()

def plot_mat_sub(datal,start=0,lim=1000,symmetric=True):
    """
    Subplot a list of mat
    :param data:
    :param start:
    :param range:
    :param symmetric:
    :return:
    """
    N=len(datal)
    assert N<6
    fig, axes = plt.subplots(nrows=N, ncols=1)
    for ii in range(len(datal)):
        ax=axes.flat[ii]
        data=datal[ii]
        data = np.array(data)
        assert len(data.shape) == 2
        img = data[:, start:start + lim]
        if symmetric:
            im=ax.imshow(img, cmap='seismic', clim=(-np.amax(data), np.amax(data)))
        else:
            im=ax.imshow(img, cmap='seismic')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def pl_conceptbubblecloud(id_to_con,id_to_word,prior,word_to_vec, pM=None):
    """
    Visualize concept using bubble word cloud
    :param id_to_con: [con for wrd0, con for wrd1, ...]
    :param id_to_word: dict[0:wrd0, 1:wrd1,...]
    :param prior:
    :return:
    """
    # step 1: data collection by concept : [[0,[wrds for con0],[priors for con0]]]
    dict_conwrdcol=dict([])
    dict_conprcol=dict([])
    dict_cen_wvec=dict([])
    dict_text_freq=dict([])
    dict_con_freq=dict([])
    list_cen_vec=[]
    con_freq_max=0
    for con_id in set(id_to_con):
        conwrdcol=[]
        conprcol=[]
        con_freq=0
        for iter_ii in range(len(id_to_con)):
            if id_to_con[iter_ii] == con_id:
                conwrdcol.append(id_to_word[iter_ii])
                conprcol.append(np.log(prior[iter_ii]))
                con_freq=con_freq+prior[iter_ii]
        # step 2: generate word cloud by concept
        text_freq = dict([])
        cen_vec=np.zeros(len(word_to_vec[conwrdcol[0]]))
        prbsum=np.sum(np.exp(np.array(conprcol)))
        for wrd_ii in range(len(conwrdcol)):
            text_freq[str(conwrdcol[wrd_ii])]=conprcol[wrd_ii]
            cen_vec=cen_vec+np.exp(conprcol[wrd_ii])/prbsum*word_to_vec[conwrdcol[wrd_ii]]
            # cen_vec = cen_vec + word_to_vec[conwrdcol[wrd_ii]]
        # cen_vec=cen_vec/len(conwrdcol)
        dict_conwrdcol[con_id]= conwrdcol
        dict_conprcol[con_id] = conprcol
        dict_cen_wvec[con_id] = cen_vec
        list_cen_vec.append(cen_vec)
        dict_text_freq[con_id]=text_freq
        dict_con_freq[con_id]=con_freq
        if con_freq>con_freq_max:
            con_freq_max=con_freq

    # Step 2: adjusting shift using force-field based method

    # PCA projection of cen_vec
    data=np.array(list_cen_vec).T
    # res,pM=pca_proj(data, 2)
    if pM is None:
        pM=np.random.random((2,len(cen_vec)))
    res=pM.dot(data)

    size_scalex,size_scaley=400,400

    dict_con_posi = dict([])
    dict_con_radius = dict([])
    for con_id in set(id_to_con):
        cen_vec = dict_cen_wvec[con_id]
        cen_posi = [pM.dot(cen_vec)[0] * size_scalex / 2, (pM.dot(cen_vec)[1]) * size_scaley / 2]
        dict_con_posi[con_id] = np.array(cen_posi)
        radius = min(size_scalex, size_scaley) / 2 * np.sqrt(dict_con_freq[con_id] / con_freq_max)
        dict_con_radius[con_id]=radius

    def cal_con_posi_stack(dict_con_radius,dist_margin=10):
        """
        Using random attaching algorithm
        :param dict_con_radius:
        :param dist_margin:
        :return:
        """
        res_con_posi = dict([])
        far_dist=100000
        for con_id_ii in set(id_to_con):
            if bool(res_con_posi):
                rnddir=np.random.random(2)-0.5
                rnddir=rnddir/np.linalg.norm(rnddir)
                start_vec=far_dist*rnddir
                min_dist=2*far_dist
                min_id=None
                for con_id_jj,posi in res_con_posi.items():
                    dist=np.linalg.norm(start_vec-posi)
                    if dist<min_dist:
                        min_dist=dist
                        min_id=con_id_jj
                cal_vec=start_vec-res_con_posi[min_id]
                cal_vec=cal_vec/np.linalg.norm(cal_vec)
                cal_vec=cal_vec*(dict_con_radius[min_id]+dict_con_radius[con_id_ii]+dist_margin)+res_con_posi[min_id]
                res_con_posi[con_id_ii]=cal_vec
            else:
                res_con_posi[con_id_ii] = np.zeros(2)

        ####### KEY BUG !!!!! center of circle is different from edge of square
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii]=res_con_posi[con_id_ii]-dict_con_radius[con_id_ii]

        return res_con_posi



    # def cal_con_posi_stack(dict_con_radius,dist_margin=10):
    #     """
    #     Using stacking method to calculate con_posi
    #     :param dict_con_radius:
    #     :return:
    #     """
    #     res_con_posi=dict([])
    #     BoundaryL=[]
    #     for con_id_ii in set(id_to_con):
    #         if not bool(res_con_posi) and not bool(BoundaryL): # All empty, step 1
    #             res_con_posi[id_to_con]=np.zeros(2)
    #             first_con_id=con_id_ii
    #         elif bool(res_con_posi) and not bool(BoundaryL): # Boundary empty, step 2
    #             BoundaryL.append((first_con_id,con_id_ii))
    #             res_con_posi[id_to_con] = np.array([dict_con_radius[first_con_id]+dict_con_radius[con_id_ii]+dist_margin,0])
    #         else:
    #             # Pick a random boundary
    #             pkid=int(np.ramdom.rand()*len(BoundaryL))
    #             bound=BoundaryL[pkid]
    #             cen1=res_con_posi[bound[0]]
    #             cen2 = res_con_posi[bound[1]]
    #
    #
    #         # Move to center
    #         tot_cen=np.zeros(2)
    #         totN=0
    #         for cen in res_con_posi.values():
    #             tot_cen=tot_cen+cen
    #             totN=totN+1
    #         tot_cen=tot_cen/totN
    #         for con in res_con_posi.keys():
    #             res_con_posi[con]=res_con_posi[con]-tot_cen


    def cal_con_posi(init_con_posi,dict_con_radius,attr=0.01,rep=2.0,dist_margin=10,fxoy=1.0,iter=10000):
        """
        Calculate updated concept position using graviy field
        :param init_con_posi:
        :param dict_con_radius:
        :param attr: attraction
        :param rep: repulsion
        :param dist_margin: distance margin between two concepts
        :param fxoy: x over y aspect ratio
        :return: cal_con_posi
        """
        res_con_posi=init_con_posi
        # Expand to remove overlap
        expansion_ratio=2.0
        # for con_id_ii in set(id_to_con):
        #     for con_id_jj in set(id_to_con):
        #         if con_id_jj != con_id_ii:
        #             # Global constant attraction
        #             dist = np.linalg.norm (init_con_posi[con_id_jj] - init_con_posi[con_id_ii])
        #             rsum=dict_con_radius[con_id_jj]+dict_con_radius[con_id_ii]
        #             test_ratio=rsum/dist
        #             if test_ratio>expansion_ratio:
        #                 expansion_ratio=test_ratio
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii]=res_con_posi[con_id_ii]*expansion_ratio*1.1

        # Move to center
        tot_cen = np.zeros(2)
        totN = 0
        for cen in res_con_posi.values():
            tot_cen = tot_cen + cen
            totN = totN + 1
        tot_cen = tot_cen / totN
        for con in res_con_posi.keys():
            res_con_posi[con] = res_con_posi[con] - tot_cen

        for ii_iter in range(iter):
            conlist=list(set(id_to_con))
            pick=int(np.random.rand()*len(conlist))
            con_id_ii=conlist[pick]

            for con_id_jj in set(id_to_con):
                if con_id_jj!=con_id_ii:

                    diffvec=init_con_posi[con_id_jj]- init_con_posi[con_id_ii]
                    dir_vec=diffvec/np.linalg.norm(diffvec)

                    rd=np.linalg.norm(diffvec)
                    rjj=dict_con_radius[con_id_jj]
                    rii=dict_con_radius[con_id_ii]
                    dist_val=rd-rjj-rii

                    res_con_posi[con_id_ii] = res_con_posi[con_id_ii] + attr * (dist_val-dist_margin) * dir_vec

                    if dist_val < dist_margin:
                        res_con_posi[con_id_ii] = res_con_posi[con_id_ii] - 1.1 *(dist_margin-dist_val)*diffvec/np.linalg.norm(diffvec)

        ####### KEY BUG !!!!! center of circle is different from edge of square
        for con_id_ii in set(id_to_con):
            res_con_posi[con_id_ii] = res_con_posi[con_id_ii] - dict_con_radius[con_id_ii]

        return res_con_posi


    res_con_posi=cal_con_posi(dict_con_posi, dict_con_radius)
    # res_con_posi = cal_con_posi_stack(dict_con_radius)
    # res_con_posi=dict_con_posi
    # Move to center
    # tot_cen = np.zeros(2)
    # totN = 0
    # for cen in res_con_posi.values():
    #     tot_cen = tot_cen + cen
    #     totN = totN + 1
    # tot_cen = tot_cen / totN
    # for con in res_con_posi.keys():
    #     res_con_posi[con] = res_con_posi[con] - tot_cen
    minx=0
    miny=0
    for con_id in set(id_to_con):
        if res_con_posi[con_id][0]<minx:
            minx=res_con_posi[con_id][0]
        if res_con_posi[con_id][1]<miny:
            miny=res_con_posi[con_id][1]

    # Step 3: actually draw the figure

    img1 = Image.new('RGBA', size=(1,1), color=(255, 255, 255, 255))

    for con_id in set(id_to_con):
        print(con_id)
        radius = dict_con_radius[con_id]
        circle = Image.new('RGBA', size=(int(2*radius),int(2*radius)), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(circle)
        draw.ellipse((0, 0, 2*radius, 2*radius), fill=(0, 0, 0, 255))
        circle2 = Image.new('RGBA', size=(int(2*radius),int(2*radius)), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(circle2)
        draw.ellipse((0, 0, 2*radius, 2*radius), fill=(0, 0, 0, 255))
        text_freq=dict_text_freq[con_id]
        wordcloud = WordCloud(mask=np.array(circle), background_color="white").generate_from_frequencies(text_freq)
        data2 = wordcloud.to_array()
        img2 = Image.fromarray(data2, 'RGB')
        img2 = img2.convert("RGBA")
        txt = Image.new('RGBA', img2.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("/usr/share/fonts/gnu-free/FreeSerif.ttf", int(radius))
        w, h = font.getsize(str(con_id))
        draw.text((radius-w/2, radius-h/2), str(con_id), (0, 0, 0, 128), font=font)
        img2 = Image.alpha_composite(img2, txt)
        shift=tuple((res_con_posi[con_id]-np.array([minx,miny])).astype(int))
        nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)

        newimg = Image.new('RGBA', size=(nw, nh), color=(255, 255, 255, 255))
        newimg.paste(img1, (0, 0))
        newimg.paste(img2, shift, circle2)

        # blend with alpha=0.5
        # img1 = Image.blend(newimg1, newimg2, alpha=0.5)
        img1=newimg

        print(shift)

    plt.imshow(img1, interpolation='bilinear')
    plt.axis("on")
    plt.margins(x=0, y=0)
    plt.show()

def wta_layer(l_input,schedule=1.0,wta_noise=0.0,upper_t = 0.3, schshift=0.2):

    concept_size = l_input.shape[-1]
    schedule=schedule+schshift
    if schedule>=1.0:
        schedule=1.0
    Nindr = (1.0 - np.sqrt(schedule)) * (concept_size - 2) * upper_t + 1  # Number of Nind largest number kept
    # Nindr = (1.0 - schedule) * (concept_size - 2) * upper_t + 1  # Number of Nind largest number kept
    smooth=Nindr-int(Nindr)
    Nind=int(Nindr)
    np_input=l_input.cpu().data.numpy()
    npargmax_i = np.argsort(-np_input, axis=-1)
    argmax_i = torch.from_numpy(npargmax_i).narrow(-1, 0, Nind)
    outer=torch.from_numpy(npargmax_i).narrow(-1, Nind, 1)
    concept_layer_i = torch.zeros(l_input.shape)
    concept_layer_i.scatter_(-1, argmax_i, 1.0)
    concept_layer_i.scatter_(-1, outer, smooth)
    concept_layer_i = concept_layer_i + wta_noise * torch.rand(concept_layer_i.shape)

    if l_input.is_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        concept_layer_i=concept_layer_i.to(device)

    ginput_masked = l_input * concept_layer_i
    # ginput_masked = ginput_masked / torch.norm(ginput_masked, 2, -1, keepdim=True)
    # ginput_masked=softmax(ginput_masked)
    return ginput_masked

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def logit_sampling_layer(l_input): ### Not seems to be working
    """
    A sampling layer over logit input
    Input logit layer, output possibility based 1-hot sampling
    :param l_input:
    :return:
    """
    raise Exception("Not working!")
    concept_size = l_input.shape[-1]
    prob=torch.exp(l_input)
    matint=torch.triu(torch.ones((concept_size,concept_size))) # Upper triangular matrix for integration purpose
    if l_input.is_cuda:
        cpu=torch.device("cpu")
        prob=prob.to(cpu)
    int_layer=torch.matmul(prob,matint)
    randm=torch.rand(prob.shape[:-1])
    # print(int_layer,randm)
    adj_layer=int_layer-randm.view(list(randm.shape)+[1])
    absplus=torch.abs(adj_layer)+adj_layer
    lastonevec = torch.zeros(concept_size)
    lastonevec[-1] = 1.0
    endadj = torch.zeros(prob.shape) + lastonevec
    absminus=torch.abs(adj_layer)-adj_layer+endadj
    rollabsminus=roll(absminus,1,-1)
    pickmat=rollabsminus*absplus
    maxind=torch.argmax(pickmat,dim=-1,keepdim=True)
    pick_layer_i = torch.zeros(prob.shape)
    pick_layer_i.scatter_(-1, maxind, 1.0)
    pick_layer_i=pick_layer_i
    if l_input.is_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pick_layer_i=pick_layer_i.to(device)
    l_output=l_input*pick_layer_i
    # l_output_masked = l_output / torch.norm(l_output, 2, -1, keepdim=True)
    return l_output

class PyTrain(object):
    """
    A class trying to wrap all possible training practice nicely
    """
    def __init__(self, dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, para=None):
        """

        :param dataset:
        :param lsize:
        :param rnn:
        :param step:
        :param learning_rate:
        :param batch:
        :param window:
        :param id_2_vec:
        :param para:
        """
        if para is None:
            para = dict([])
        self.save = para.get("save", None)
        self.seqtrain = para.get("seqtrain", False)
        self.id_2_vec = para.get("id_2_vec", None)
        self.supervise_mode = para.get("supervise_mode", False)
        self.coop = para.get("coop", None)
        self.coopseq = para.get("coopseq", None)
        self.cuda_flag = para.get("cuda_flag", True)
        self.invec_noise = para.get("invec_noise", 0.0)
        self.pre_training = para.get("pre_training", False)
        self.loss_clip = para.get("loss_clip", 0.0)
        self.digit_input = para.get("digit_input", True)
        self.two_step_training = para.get("two_step_training", False)
        self.length_sorted=False
        self.SYM_PAD = para.get("SYM_PAD", None)
        self.PAD_VEC = None

        if type(lsize) is list:
            self.lsize_in = lsize[0]
            self.lsize_out = lsize[1]
        else:
            self.lsize_in = lsize
            self.lsize_out = lsize

        self.rnn=rnn
        self.step=step
        self.batch=batch
        self.window=window

        # profiler
        self.prtstep = int(step / 10)
        self.train_hist = []
        self.his = 0

        # optimizer
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=learning_rate, weight_decay=0.0)
        # self.optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        # CUDA
        if self.cuda_flag:
            self.gpuavail = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if self.gpuavail else "cpu")
        else:
            self.gpuavail = False
            self.device = torch.device("cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(self.device)

        self.data_init = True
        self.data(dataset)

        # Eval data mem
        self.inputlabl = []
        self.conceptl = []
        self.outputll = []

        # self.lossc=torch.nn.CrossEntropyLoss()

    def data(self,dataset):
        """
        Swap dataset
        :param dataset:
        :return:
        """
        if (type(dataset) is dict) != self.supervise_mode:
            raise Exception("Supervise mode Error.")
        if self.supervise_mode:
            self.label = dataset["label"]
            self.dataset = dataset["dataset"]
        else:
            self.dataset = dataset
        self.__init_data()
        self.length_sorted = False

    def para(self,para):
        """
        Update parameter
        :param para:
        :return:
        """
        if para is None:
            para = dict([])
        self.save = para.get("save", None)
        self.seqtrain = para.get("seqtrain", False)
        self.id_2_vec = para.get("id_2_vec", None)
        self.supervise_mode = para.get("supervise_mode", False)
        self.coop = para.get("coop", None)
        self.coopseq = para.get("coopseq", None)
        self.cuda_flag = para.get("cuda_flag", True)
        self.invec_noise = para.get("invec_noise", 0.0)
        self.pre_training = para.get("pre_training", False)
        self.loss_clip = para.get("loss_clip", 0.0)
        self.digit_input = para.get("digit_input", True)
        self.two_step_training = para.get("two_step_training", False)
        self.length_sorted=False
        self.SYM_PAD = para.get("SYM_PAD", None)
        self.PAD_VEC = None

    def __profiler(self,iis,loss):
        if int(iis / self.prtstep) != self.his:
            print("Perlexity: ", iis, np.exp(loss.item()))
            self.his = int(iis / self.prtstep)
        self.train_hist.append(np.exp(loss.item()))

    def __postscript(self):
        x = []
        for ii in range(len(self.train_hist)):
            x.append([ii, self.train_hist[ii]])
        x = np.array(x)
        try:
            plt.plot(x[:, 0], x[:, 1])
            if type(self.save) != type(None):
                plt.savefig(self.save)
                plt.gcf().clear()
            else:
                if self.loss_clip > 0:
                    plt.ylim((0, self.loss_clip))
                plt.show()
        except:
            pass

    def __init_data(self,limit=1e9):
        if len(self.dataset)*self.lsize_in<limit:
            datab = []
            if self.digit_input:
                if self.id_2_vec is None: # No embedding, one-hot representation
                    self.PAD_VEC=np.zeros(self.lsize_in, dtype=np.float32)
                    self.PAD_VEC[self.SYM_PAD] = 1.0
            if type(self.dataset[0]) != list:
                if self.digit_input:
                    if self.id_2_vec is None:  # No embedding, one-hot representation
                        for data in self.dataset:
                            datavec = np.zeros(self.lsize_in)
                            datavec[data] = 1.0
                            datab.append(datavec)
                    else:
                        for data in self.dataset:
                            datavec = np.array(self.id_2_vec[data])
                            datab.append(datavec)
                else:  # if not digit input, raw data_set is used
                    datab = self.dataset
                self.databp = torch.from_numpy(np.array(datab))
                self.databp = self.databp.type(torch.FloatTensor)
            else: # we assume sentence structure
                self.databp=[]
                if self.digit_input:
                    if self.id_2_vec is None:  # No embedding, one-hot representation
                        for sent in self.dataset:
                            datab_sent=[]
                            for data in sent:
                                datavec = np.zeros(self.lsize_in)
                                datavec[data] = 1.0
                                datab_sent.append(datavec)
                            datab_sent = torch.from_numpy(np.array(datab_sent))
                            datab_sent = datab_sent.type(torch.FloatTensor)
                            self.databp.append(datab_sent)
                    else:
                        for sent in self.dataset:
                            datab_sent = []
                            for data in sent:
                                datavec = np.array(self.id_2_vec[data])
                                datab_sent.append(datavec)
                            datab_sent=torch.from_numpy(np.array(datab_sent))
                            datab_sent = datab_sent.type(torch.FloatTensor)
                            self.databp.append(datab_sent)
                else:  # if not digit input, raw data_set is used
                    for sent in self.dataset:
                        datab_sent = torch.from_numpy(np.array(sent))
                        datab_sent = datab_sent.type(torch.FloatTensor)
                        self.databp.append(datab_sent)
            self.data_init = True
        else:
            print("Warning, large dataset, not pre-processed.")
            self.databp=None
            self.data_init=False

    def __build_databp(self,inlab):
        """
        Build databp from inlab (when dataset too large)
        :param inlab:
        :return:
        """
        if self.digit_input and self.id_2_vec is None and not self.data_init:
            datab=np.zeros((len(inlab),self.lsize_in))
            for ii_b in range(len(inlab)):
                datab[ii_b,inlab[ii_b]]=1.0
            databp = torch.from_numpy(np.array(datab))
            # databp = databp.type(torch.FloatTensor)
        else:
            raise Exception("Not Implemented")
        return databp


    def __get_data_continous(self):

        rstartv = np.floor(np.random.rand(self.batch) * (len(self.dataset) - self.window - 1))

        if not self.supervise_mode:
            # Generating output label
            yl = np.zeros((self.batch,self.window))
            xl = np.zeros((self.batch,self.window))
            for iib in range(self.batch):
                xl[iib,:]=self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window]
                yl[iib,:]=self.dataset[int(rstartv[iib])+1:int(rstartv[iib]) + self.window+1]
            inlab = torch.from_numpy(xl)
            inlab = inlab.type(torch.LongTensor)
            outlab = torch.from_numpy(yl)
            outlab = outlab.type(torch.LongTensor)

        else:
            xl = np.zeros((self.batch, self.window))
            for iib in range(self.batch):
                xl[iib, :] = self.label[int(rstartv[iib]):int(rstartv[iib]) + self.window]
            inlab = torch.from_numpy(xl)
            inlab = inlab.type(torch.LongTensor)
            outlab = inlab

        vec1m = torch.zeros(self.window, self.batch, self.lsize_in)
        # vec2m = torch.zeros(self.window, self.batch, self.lsize_in)
        for iib in range(self.batch):
            # vec1_raw = self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            # vec1_rnd = torch.rand(vec1_raw.shape)
            # vec1_add = torch.mul((1.0 - vec1_raw) * self.invec_noise, vec1_rnd.double())
            # vec1 = vec1_raw + vec1_add
            if self.data_init:
                vec1=self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            else:
                vec1=self.__build_databp(self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window])
            # vec2 = self.databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.window + 1, :]
            vec1m[:,iib,:]=vec1
            # vec2m[:, iib, :] = vec2
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor) #
        # y = Variable(vec2m, requires_grad=True)

        if self.gpuavail:
            inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            x = x.to(self.device)

        return x, None, inlab, outlab

    def __get_data_bias(self,bias_num):
        """
        A data getting subroutine biased towards num
        :param num:
        :return:
        """

        rstartv = np.zeros(self.batch)
        for iir in range(self.batch):
            rrstart = np.floor(np.random.rand() * (len(self.dataset) - self.window - 1))
            while bias_num not in set(self.dataset[int(rrstart):int(rrstart) + self.window]): # resample until ii_f found
                rrstart = np.floor(np.random.rand() * (len(self.dataset) - self.window - 1))
            rstartv[iir]=rrstart

        if not self.supervise_mode:
            # Generating output label
            yl = np.zeros((self.batch,self.window))
            xl = np.zeros((self.batch,self.window))
            for iib in range(self.batch):
                xl[iib,:]=np.array(self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window])
                yl[iib,:]=np.array(self.dataset[int(rstartv[iib])+1:int(rstartv[iib]) + self.window+1])
            inlab = torch.from_numpy(xl)
            inlab = inlab.type(torch.LongTensor)
            outlab = torch.from_numpy(yl)
            outlab = outlab.type(torch.LongTensor)

        else:
            raise Exception("Non-implemented")

        vec1m = torch.zeros(self.window, self.batch, self.lsize_in)
        # vec2m = torch.zeros(self.window, self.batch, self.lsize_in)
        for iib in range(self.batch):
            if self.data_init:
                vec1=self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            else:
                vec1=self.__build_databp(np.array(self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window]))
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor) #
        # y = Variable(vec2m, requires_grad=True)

        if self.gpuavail:
            inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            x = x.to(self.device)

        return x, None, inlab, outlab


    def __get_data_sentence(self,length_cluster=True):

        if length_cluster:
            rstartv=np.floor(np.random.rand() * (len(self.dataset)-self.batch))
            rstartv=rstartv+np.array(range(self.batch))
            if not self.length_sorted:
                self.dataset.sort(key=lambda elem:len(elem))
                self.data(self.dataset)
                self.length_sorted=True
        else:
            rstartv = np.floor(np.random.rand(self.batch) * (len(self.dataset)))

        if self.supervise_mode:
            raise Exception("Not supported")

        else:
            # Generating output label
            yl = []
            xl = []
            maxl=0
            for iib in range(self.batch):
                sentN = int(rstartv[iib])
                xl.append(self.dataset[sentN][:])
                yl.append(self.dataset[sentN][1:]+[self.SYM_PAD])
                if len(self.dataset[sentN])>maxl:
                    maxl=len(self.dataset[sentN])
            #STM_END padding
            xlp=[]
            for datal in xl:
                datal=datal+[self.SYM_PAD]*(maxl-len(datal))
                xlp.append(datal)
            ylp=[]
            for datal in yl:
                datal=datal+[self.SYM_PAD]*(maxl-len(datal))
                ylp.append(datal)

            inlab = torch.from_numpy(np.array(xlp))
            inlab = inlab.type(torch.LongTensor)
            outlab = torch.from_numpy(np.array(ylp))
            outlab = outlab.type(torch.LongTensor)

        vec1m = None
        vec2m = None
        length= maxl
        for iib in range(self.batch):
            sentN = int(rstartv[iib])
            vec1 = self.databp[sentN]
            assert maxl >= len(vec1)
            if maxl>len(vec1):
                padding=torch.zeros((maxl-len(vec1),len(vec1[0])))+torch.from_numpy(self.PAD_VEC)
                vec1=torch.cat((vec1,padding),dim=0)
                assert len(vec1)==maxl
            if self.invec_noise>0:
                raise Exception("Not supported")
            # vec1 = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
            vec2 = self.databp[sentN][1:]
            assert maxl >= len(vec2)
            if maxl > len(vec2):
                padding = torch.zeros((maxl - len(vec2), len(vec2[0])))+torch.from_numpy(self.PAD_VEC)
                vec2 = torch.cat((vec2, padding), dim=0)
                assert len(vec2) == maxl
            if type(vec1m) == type(None):
                vec1m = vec1.view(length, 1, -1)
                vec2m = vec2.view(length, 1, -1)
            else:
                raise Exception("Do not do cat.")
                vec1m = torch.cat((vec1m, vec1.view(length, 1, -1)), dim=1)
                vec2m = torch.cat((vec2m, vec2.view(length, 1, -1)), dim=1)
        x = Variable(vec1m.reshape(length, self.batch, self.lsize_in).contiguous(), requires_grad=True)  #
        y = Variable(vec2m.reshape(length, self.batch, self.lsize_in).contiguous(), requires_grad=True)
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

        if self.gpuavail:
            outlab = outlab.to(self.device)
            x, y = x.to(self.device), y.to(self.device)
        return x, y, inlab, outlab

    def __get_data(self,bias_num=None):

        if bias_num is None:
            if type(self.dataset[0]) != list:
                x, y, inlab, outlab=self.__get_data_continous()
            else:
                x, y, inlab, outlab=self.__get_data_sentence()
        else:
            x, y, inlab, outlab = self.__get_data_bias(bias_num)
        return x,y,inlab, outlab

    def custom_KNWLoss(self, outputl, outlab, model=None, cstep=None):
        lossc=torch.nn.CrossEntropyLoss()
        # loss1 = self.lossc(outputl, outlab)
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg


    def run_training(self,step=None):

        if step is not None:
            self.step=step

        startt = time.time()
        self.rnn.train()
        if self.gpuavail:
            self.rnn.to(self.device)
        for iis in range(self.step):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x,y,inlab,outlab=self.__get_data()
            if self.pre_training:
                output, hidden = self.rnn.pre_training(x, hidden, schedule=iis / self.step)
            else:
                output, hidden = self.rnn(x, hidden, schedule=iis / self.step)
            if self.two_step_training:
                output_twostep = self.rnn.auto_encode(y, schedule=iis / self.step)
            loss = self.custom_KNWLoss(output.permute(1, 2, 0), outlab, self.rnn, iis)
            if self.two_step_training:
                loss_twostep = self.custom_KNWLoss(output_twostep.permute(1, 2, 0), outlab, self.rnn, iis)
                loss = 0.8 * loss + 0.2 * loss_twostep
            self.__profiler(iis,loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        endt = time.time()
        print("Time used in training:", endt - startt)
        self.__postscript()
        if self.gpuavail:
            torch.cuda.empty_cache()

    def do_eval(self,step_eval=300,layer_sep_mode=None):

        self.inputlabl = []
        self.conceptl = []
        self.conceptl0 = []
        self.conceptl1 = []
        self.outputll = []

        print("Start Evaluation ...")
        if layer_sep_mode == 0:
            print("Evaluate layer 0 only ...")
        elif layer_sep_mode==1:
            print("Evaluate layer 1 only ...")
        startt = time.time()
        self.rnn.eval()
        perpl=[]
        for iis in range(step_eval):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, y, inlab, outlab = self.__get_data()
            self.inputlabl.append(inlab.cpu().data.numpy().T)
            if self.pre_training:
                outputl, hidden = self.rnn.pre_training(x, hidden, schedule=1.0)
            elif layer_sep_mode==0:
                outputl, hidden = self.rnn.forward0(x, hidden, schedule=1.0)
            elif layer_sep_mode==1:
                outputl, hidden = self.rnn.forward1(x, hidden, schedule=1.0)
            else:
                outputl, hidden = self.rnn(x, hidden, schedule=1.0)

            try:
                self.conceptl0.append(self.rnn.infer_pos0.cpu().data.numpy())
            except:
                pass
            try:
                self.conceptl1.append(self.rnn.infer_pos1.cpu().data.numpy())
            except:
                pass
            try:
                self.conceptl.append(self.rnn.infer_pos.cpu().data.numpy())
            except:
                pass

            loss = self.custom_KNWLoss(outputl.permute(1, 2, 0), outlab, self.rnn, iis)
            perpl.append(loss.item())
        print("Evaluation Perplexity: ", np.exp(np.mean(np.array(perpl))))
        endt = time.time()
        print("Time used in evaluation:", endt - startt)

        self.inputlabl=np.array(self.inputlabl)
        self.conceptl=np.array(self.conceptl)

    def do_eval_conditioned(self,step_eval=300,layer_sep_mode=0):
        """
        Instead of calculating total average perplexiy, calculate perplexity conditioned over layer/word
        outputl_shape->(seql,batch,lsize)
        outlab_shape->(batch,seql)
        :param step_eval:
        :param layer_sep_mode:
        :return:
        """
        print("Start Conditional Evaluation ...")
        if layer_sep_mode == 0:
            print("Evaluate layer 0 only ...")
        elif layer_sep_mode == 1:
            print("Evaluate layer 1 only ...")
        startt = time.time()
        self.rnn.eval()

        perp_list_tot = np.zeros(self.lsize_out)
        perp_list_cnt = np.zeros(self.lsize_out)

        for iis in range(step_eval):
            perp_array_store = [[] for ii in range(self.lsize_out)]
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, y, inlab, outlab = self.__get_data()
            if self.pre_training:
                outputl, hidden = self.rnn.pre_training(x, hidden, schedule=1.0)
            elif layer_sep_mode==0:
                outputl, hidden = self.rnn.forward0(x, hidden, schedule=1.0)
            elif layer_sep_mode==1:
                outputl, hidden = self.rnn.forward1(x, hidden, schedule=1.0)
            else:
                outputl, hidden = self.rnn(x, hidden, schedule=1.0)

            for ii_l in range(len(outputl[0])):
                for ii_b in range(len(outputl[1])):
                    perp_array_store[int(outlab[ii_b,ii_l])].append(outputl[ii_l,ii_b])

            for ii_wrd in range(len(perp_array_store)):
                if perp_array_store[ii_wrd]: # Not empty
                    Nhit=len(perp_array_store[ii_wrd])
                    perp_list_cnt[ii_wrd]=perp_list_cnt[ii_wrd]+Nhit
                    calcrossEnt_in=torch.zeros((Nhit,self.lsize_out))
                    for iin in range(Nhit):
                        calcrossEnt_in[iin,:]=perp_array_store[ii_wrd][iin]
                    calcrossEnt_C = torch.from_numpy(np.array([ii_wrd]*Nhit))
                    calcrossEnt_C = calcrossEnt_C.type(torch.LongTensor)
                    loss = self.custom_KNWLoss(calcrossEnt_in, calcrossEnt_C)
                    perp_list_tot[ii_wrd]=perp_list_tot[ii_wrd]+loss.item()*Nhit

        endt = time.time()
        print("Time used in evaluation:", endt - startt)

        return perp_list_tot/perp_list_cnt,perp_list_cnt

    def do_eval_conditioned_ave(self,min_sh=25, max_sh=1000,layer_sep_mode=0):
        """
        Instead of calculating total average perplexiy, calculate perplexity conditioned over layer/word
        a minimum encounter per word is set to remove noise
        outputl_shape->(seql,batch,lsize)
        outlab_shape->(batch,seql)
        Each word has a minimum appear number >=min
        :param step_eval:
        :param layer_sep_mode:
        :return:
        """
        print("Start Conditional Evaluation ...")
        if layer_sep_mode == 0:
            print("Evaluate layer 0 only ...")
        elif layer_sep_mode == 1:
            print("Evaluate layer 1 only ...")
        elif layer_sep_mode is None:
            print("Evaluate both layers ...")
        startt = time.time()
        self.rnn.eval()

        perp_list_tot = np.zeros(self.lsize_out)
        perp_list_cnt = np.zeros(self.lsize_out)

        step_cnt=0
        while np.min(perp_list_cnt)<min_sh:
            bias_num=np.argmin(perp_list_cnt)
            step_cnt=step_cnt+1
            print("No. of step: ",step_cnt,"Focusing on: ",bias_num, "Count: ",perp_list_cnt[bias_num])
            perp_array_store = dict([])
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, y, inlab, outlab = self.__get_data(bias_num=bias_num)

            # if bias_num==197:
            #     print(outlab[0])
            #     a=input("Wait:")

            if self.pre_training:
                outputl, hidden = self.rnn.pre_training(x, hidden, schedule=1.0)
            elif layer_sep_mode==0:
                outputl, hidden = self.rnn.forward0(x, hidden, schedule=1.0)
            elif layer_sep_mode==1:
                outputl, hidden = self.rnn.forward1(x, hidden, schedule=1.0)
            else:
                outputl, hidden = self.rnn(x, hidden, schedule=1.0)

            for ii_l in range(outputl.shape[0]):
                for ii_b in range(outputl.shape[1]):
                    if perp_array_store.get(int(outlab[ii_b,ii_l]),None) is None:
                        perp_array_store[int(outlab[ii_b, ii_l])]=[]
                        perp_array_store[int(outlab[ii_b, ii_l])].append(outputl[ii_l, ii_b])
                    else:
                        perp_array_store[int(outlab[ii_b,ii_l])].append(outputl[ii_l,ii_b])

            for ii_wrd in range(self.lsize_out):
                if perp_array_store.get(ii_wrd,None) and perp_list_cnt[ii_wrd]<max_sh: # Not empty and not too much
                    Nhit=len(perp_array_store[ii_wrd])
                    perp_list_cnt[ii_wrd]=perp_list_cnt[ii_wrd]+Nhit
                    calcrossEnt_in=torch.zeros((Nhit,self.lsize_out))
                    for iin in range(Nhit):
                        calcrossEnt_in[iin,:]=perp_array_store[ii_wrd][iin]
                    calcrossEnt_C = torch.from_numpy(np.array([ii_wrd]*Nhit))
                    calcrossEnt_C = calcrossEnt_C.type(torch.LongTensor)
                    loss = self.custom_KNWLoss(calcrossEnt_in, calcrossEnt_C)
                    perp_list_tot[ii_wrd]=perp_list_tot[ii_wrd]+loss.item()*Nhit
            print("After Count: ", perp_list_cnt[bias_num])

        endt = time.time()
        print("Time used in evaluation:", endt - startt)

        return perp_list_tot/perp_list_cnt,perp_list_cnt

    def free_gen(self, step_gen=1000,noise=0.0):
        """
        Free generation
        :param step:
        :param lsize:
        :param rnn:
        :param id_2_vec:
        :return:
        """
        print("Start free generation ...")
        startt = time.time()

        x = torch.zeros(1, 1, self.lsize_in)
        x[0, 0, 0] = 1.0
        x=x+noise*(torch.rand((1, 1, self.lsize_in))-0.5)
        x = x.type(torch.FloatTensor)
        if self.gpuavail:
            hidden = self.rnn.initHidden_cuda(self.device, 1)
            x=x.to(self.device)
        else:
            hidden = self.rnn.initHidden(1)

        outputl = []

        # clul=[]

        def logp(vec):
            """
            LogSoftmax function
            :param vec:
            :return:
            """
            vec = np.exp(vec)
            dwn = np.sum(vec)
            return vec / dwn

        for iis in range(step_gen):
            output, hidden = self.rnn(x, hidden, schedule=1.0)  ############ rnn
            ynp = output.cpu().data.numpy().reshape(self.lsize_out)
            rndp = np.random.rand()
            pii = logp(ynp).reshape(-1)
            dig = 0
            for ii in range(len(pii)):
                rndp = rndp - pii[ii]
                if rndp < 0:
                    dig = ii
                    break
            outputl.append(dig)
            x = torch.zeros(1, 1, self.lsize_in)
            x[0, 0, dig] = 1.0
            x = x + noise*(torch.rand((1, 1, self.lsize_in))-0.5)
            x = x.type(torch.FloatTensor)
            if self.gpuavail:
                x = x.to(self.device)
        endt = time.time()
        print("Time used in generation:", endt - startt)
        return outputl

