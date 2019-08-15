"""
Python package for NCA learning algorithm
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

import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

from wordcloud import WordCloud
import operator
from PIL import Image
from PIL import ImageDraw,ImageFont

from tqdm import tqdm
import datetime

def pltscatter(data,dim=(0,1),labels=None,title=None,xlabel=None,ylabel=None):
    assert data.shape[0]>data.shape[1]
    for i in range(len(data)):
        x,y=data[i,dim[0]], data[i,dim[1]]
        plt.scatter(x,y)
        if labels is not None:
            label=labels[i]
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def pltscatter3D(data,labels=None,title=None,xlabel=None,ylabel=None):
    assert data.shape[0]>data.shape[1]
    assert data.shape[1]==3

    fig = plt.figure()
    ax = Axes3D(fig)

    for ii in range(len(data)):  # plot each point + it's index as text above
        ax.scatter(data[ii, 0], data[ii, 1], data[ii, 2], color='b')
        ax.text(data[ii, 0], data[ii, 1], data[ii, 2], str(labels[ii]),color='k')

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def plot_mat(data,start=0,lim=1000,symmetric=False,title=None,tick_step=None,show=True,xlabel=None,ylabel=None):
    if show:
        plt.figure()
    data=np.array(data)
    if len(data.shape) != 2:
        data=data.reshape(1,-1)
    img=data[:,start:start+lim]
    if symmetric:
        plt.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)), np.amax(np.abs(data))))
    else:
        plt.imshow(img, cmap='seismic')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if tick_step is not None:
        plt.xticks(np.arange(0, len(img[0]), tick_step))
        plt.yticks(np.arange(0, len(img), tick_step))
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show:
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

def plot_mat_ax(ax, img, symmetric=False, title=None, tick_step=None, clim=None,xlabel=None,ylabel=None):
    img = np.array(img)
    assert len(img.shape) == 2
    if clim is None:
        if symmetric:
            clim=(-np.amax(np.abs(img)), np.amax(np.abs(img)))
        else:
            clim = (np.min(img), np.max(img))
    pltimg = ax.imshow(img, cmap='seismic', clim=clim,aspect="auto")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if tick_step is not None:
        ax.set_xticks(np.arange(0, len(img[0]), tick_step))
        ax.set_yticks(np.arange(0, len(img), tick_step))
    return pltimg

def plot_activity_example(self,datalist,titlelist,climlist=None,sysmlist=None,mode="predict"):
    """
    Plot a single example of prediction
    :return:
    """
    assert type(datalist) is list

    if mode=="predict":
        for ii in range(len(datalist)):
            datalist[ii]=datalist[ii].cpu().data.numpy()
        num_plot=len(datalist)
        f, axes = plt.subplots(num_plot, 1, sharex=False)
        for ii in range(len(datalist)):
            symlab=True
            clim=None
            if sysmlist is not None:
                symlab=sysmlist[ii]
            if climlist is not None:
                clim=climlist[ii]
            plt_t =plot_mat_ax(axes[ii],datalist[ii].T, title=titlelist[ii], symmetric=symlab, tick_step=1,clim=clim)
            plt.colorbar(plt_t, ax=axes[ii])
        # [ax.grid(True, linestyle='-.') for ax in (ax1, ax2, ax3, ax4, ax5)]
        plt.show()
    elif mode=="seq2seq":
        assert type(datalist[0]) is list
        assert len(datalist[0])==2
        for ii in range(len(datalist)):
            datalist[ii][0]=datalist[ii][0].cpu().data.numpy()
            datalist[ii][1] = datalist[ii][1].cpu().data.numpy()
        num_plot = len(datalist)
        f, axes = plt.subplots(num_plot, 2, sharex=False)
        if num_plot==1:
            axes=[axes] # if only 1 plot, make it support indexing
        for ii in range(len(datalist)):
            symlab = True
            clim = None
            if sysmlist is not None:
                symlab = sysmlist[ii]
            if climlist is not None:
                clim0 = climlist[ii][0]
                clim1 = climlist[ii][1]
            plt_t = plot_mat_ax(axes[ii][0], datalist[ii][0].T, title=titlelist[ii]+"_enc", symmetric=symlab, tick_step=1,clim=clim0)
            plt.colorbar(plt_t, ax=axes[ii][0])
            plt_t = plot_mat_ax(axes[ii][1], datalist[ii][1].T, title=titlelist[ii]+"_dec", symmetric=symlab, tick_step=1,clim=clim1)
            plt.colorbar(plt_t, ax=axes[ii][1])
        # [ax.grid(True, linestyle='-.') for ax in (ax1, ax2, ax3, ax4, ax5)]
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
    l_output_masked = l_output / torch.norm(l_output, 2, -1, keepdim=True)
    return l_output_masked


class PyTrain_Lite(object):
    """
    A class trying to wrap all possible training practice nicely and litely
    """

    def __init__(self, dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, para=None):
        """
        A lite version of pytrain
        :param lsize:
        :param rnn:
        :param step:
        :param custom:
        :param learning_rate:
        :param batch:
        :param window:
        :param para:
        """

        self.rnn = rnn
        self.step = step
        self.dataset = dataset
        self.batch = batch
        self.window = window

        if para is None:
            para = dict([])
        self.para(para)

        if type(lsize) is list:
            self.lsize_in = lsize[0]
            self.lsize_out = lsize[1]
        else:
            self.lsize_in = lsize
            self.lsize_out = lsize

        self.lossf = None
        self.lossf_eval = None
        self.loss = None

        # profiler
        self.prtstep = int(step / 20)
        self.train_hist = []
        self.his = 0

        # optimizer
        if self.optimizer_label=="adam":
            print("Using adam optimizer")
            self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=learning_rate, weight_decay=0.0)
        else:
            print("Using SGD optimizer")
            self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=learning_rate)

        # CUDA
        if self.cuda_flag:
            self.gpuavail = torch.cuda.is_available()
            self.device = torch.device(self.cuda_device if self.gpuavail else "cpu")
            self.rnn=self.rnn.to(self.device)
        else:
            self.gpuavail = False
            self.device = torch.device("cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(self.device)

        currentDT = datetime.datetime.now()
        self.log="Time of creation: "+str(currentDT)+"\n"

        # Evaluation memory
        self.evalmem = None
        self.data_col_mem = None
        self.training_data_mem=None

    def para(self,para):
        self.save_fig = para.get("save_fig", None)
        self.cuda_flag = para.get("cuda_flag", True)
        self.seqtrain = para.get("seqtrain", True)
        self.id_2_vec = para.get("id_2_vec", None)
        self.supervise_mode = para.get("supervise_mode", False)
        self.coop = para.get("coop", None)
        self.coopseq = para.get("coopseq", None)
        self.invec_noise = para.get("invec_noise", 0.0)
        self.pre_training = para.get("pre_training", False)
        self.loss_clip = para.get("loss_clip", 0.0)
        self.digit_input = para.get("digit_input", True)
        self.two_step_training = para.get("two_step_training", False)
        self.context_total = para.get("context_total", 1)
        self.context_switch_step = para.get("context_switch_step", 10)
        self.reg_lamda = para.get("reg_lamda", 0.0)
        self.optimizer_label = para.get("optimizer_label", "adam")
        self.length_sorted = False
        self.cuda_flag = para.get("cuda_flag", True)
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.figure_plot = para.get("figure_plot", True)
        self.custom_interface = para.get("custom_interface", None)


    def run_training(self,step=None,lr=None,optimizer_label="adam"):

        if step is not None:
            self.step=step

        if lr is not None:
            if optimizer_label == "adam":
                print("Using adam optimizer")
                self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr, weight_decay=0.0)
            else:
                print("Using SGD optimizer")
                self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=lr)

        startt = time.time()
        self.rnn.train()
        if self.gpuavail:
            self.rnn.to(self.device)
        for iis in range(self.step):
        # for iis in tqdm(range(self.step)):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            if self.seqtrain:
                x, label, _ = self.get_data()
                outputl, hidden = self.rnn(x, hidden, schedule=iis / self.step)
            else:
                outputl=None
                x, label, _ = self.get_data()
                for iiw in range(self.window):
                    output, hidden = self.rnn(x[iiw,:,:], hidden, schedule=iis / self.step)
                    if outputl is None:
                        outputl = output.view(1, self.batch, self.lsize_out)
                    else:
                        outputl = torch.cat((outputl.view(-1, self.batch, self.lsize_out), output.view(1, self.batch, self.lsize_out)), dim=0)
            self.loss = self.lossf(outputl, label, self.rnn, iis)
            self._profiler(iis, self.loss)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            self.while_training(iis)

        endt = time.time()
        print("Time used in training:", endt - startt)
        self.log = self.log + "Time used in training: " + str(endt - startt) + "\n"
        self._postscript()
        if self.gpuavail:
            torch.cuda.empty_cache()

    def eval_mem(self,*args):
        pass

    def do_eval(self,step_eval=300,schedule=1.0,posi_ctrl=None,allordermode=False):
        startt = time.time()
        self.rnn.eval()
        if self.gpuavail:
            self.rnn.to(self.device)
        perpl=[]
        if allordermode:
            step_eval=int(self.dataset_length/self.batch)
            print(step_eval)
        for iis in range(step_eval):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            if allordermode:
                rstartv=iis*self.batch+np.linspace(0,self.batch-1,self.batch)
                x, label, _ = self.get_data(rstartv=rstartv)
            else:
                x, label, _ = self.get_data()
            if self.seqtrain:
                output, hidden = self.rnn(x, hidden, schedule=schedule)
                if posi_ctrl is None:
                    loss = self.lossf_eval(output, label, self.rnn, None)
                else:
                    loss = self.lossf_eval(output[posi_ctrl,:,:].view(1,output.shape[1],output.shape[2]), label[:,posi_ctrl].view(label.shape[0],1), self.rnn, None)
                self.eval_mem(x, label, self.rnn)

            else:
                outputl=None
                hiddenl=None
                for iiw in range(self.window):
                    output, hidden = self.rnn(x[iiw, :, :], hidden, schedule=schedule)
                    if outputl is None:
                        outputl = output.view(1, self.batch, self.lsize_out)
                        hiddenl = hidden.view(1, self.batch, self.rnn.hidden_size)
                    else:
                        outputl = torch.cat(
                            (outputl.view(-1, self.batch, self.lsize_out), output.view(1, self.batch, self.lsize_out)),
                            dim=0)
                        hiddenl = torch.cat(
                            (hiddenl.view(-1, self.batch, self.rnn.hidden_size), hidden.view(1, self.batch, self.rnn.hidden_size)),
                            dim=0)
                loss = self.lossf_eval(outputl, label, self.rnn, iis)
                self.eval_mem(outputl, hiddenl)
            perpl.append(loss.cpu().item())
        # print("Evaluation Perplexity: ", np.mean(np.array(perpl)))
        perp=np.exp(np.mean(np.array(perpl)))
        print("Evaluation Perplexity: ", perp)
        endt = time.time()
        print("Time used in evaluation:", endt - startt)
        if self.gpuavail:
            torch.cuda.empty_cache()

        currentDT = datetime.datetime.now()
        self.log = self.log + "Time at evaluation: " + str(currentDT) + "\n"
        self.log = self.log + "Evaluation Perplexity: "+ str(np.exp(np.mean(np.array(perpl)))) + "\n"
        self.log = self.log + "Time used in evaluation:"+ str(endt - startt) + "\n"
        return perp

    def do_test(self,step_test=300,schedule=1.0,posi_ctrl=None):
        """
        Calculate correct rate
        :param step_test:
        :return:
        """
        startt = time.time()
        self.rnn.eval()
        if self.gpuavail:
            self.rnn.to(self.device)
        total=0
        correct=0
        correct_ratel=[]
        for iis in range(step_test):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, label, _ = self.get_data()
            output, hidden = self.rnn(x, hidden, schedule=schedule)
            output = output.permute(1, 2, 0)
            _,predicted = torch.max(output,1)
            if posi_ctrl is None:
                total += label.size(0)*label.size(1)
                correct += (predicted == label).sum().item()
            else:
                total += label.size(0)
                correct += (predicted[:,posi_ctrl] == label[:,posi_ctrl]).sum().item()
            correct_ratel.append(correct/total)
            # self.eval_mem(output)
        # print("Evaluation Perplexity: ", np.mean(np.array(perpl)))
        crate=np.mean(np.array(correct_ratel))
        print("Correct rate: ", crate)
        endt = time.time()
        print("Time used in test:", endt - startt)
        if self.gpuavail:
            torch.cuda.empty_cache()

        currentDT = datetime.datetime.now()
        self.log = self.log + "Time of test: " + str(currentDT) + "\n"
        self.log = self.log + "Correct rate: " + str(np.mean(np.array(correct_ratel))) + "\n"
        self.log = self.log + "Time used in test:" + str(endt - startt) + "\n"

        return crate

    def example_data_collection(self,*args,**kwargs):
        raise Exception("NotImplementedException")

    def plot_example(self,items="all"):
        """
        Plot a single example of prediction
        :return:
        """
        print("Plot a single example of predcition.")
        self.data_col_mem = None
        self.rnn.eval()
        if self.gpuavail:
            hidden = self.rnn.initHidden_cuda(self.device, 1)
            self.rnn.to(self.device)
        else:
            hidden = self.rnn.initHidden(1)
        x, label, _ = self.get_data(batch=1)

        if self.seqtrain:
            output, hidden = self.rnn(x, hidden, schedule=1.0)
            self.example_data_collection(x, output, hidden, label,items=items)
        else:
            for iiw in range(self.window):
                output, hidden = self.rnn(x[iiw, :, :], hidden, schedule=1.0)
                self.example_data_collection(x[iiw, :, :], output, hidden, label,items="all")

        # datalist=[xnp.T,outmat.T,hidmat.T,ztmat.T,ntmat.T]
        # datalist = [[xnp.T,xnp.T], [outmat.T,outmat.T], [hidmat.T,hidmat.T], [ztmat.T,ztmat.T], [ntmat.T,ntmat.T]]
        # titlelist=["input","predict","hidden","zt keep gate","nt set gate"]
        # sysmlist=[True,False,True,True,True]

        datalist=self.data_col_mem["datalist"]
        titlelist = self.data_col_mem["titlelist"]
        climlist = self.data_col_mem["climlist"]
        sysmlist = self.data_col_mem["sysmlist"]
        mode = self.data_col_mem["mode"]

        # plot_activity_example(self, datalist, titlelist, sysmlist=sysmlist,mode="predict")

        plot_activity_example(self, datalist, titlelist, climlist=climlist,sysmlist=sysmlist, mode=mode)


        if self.gpuavail:
            torch.cuda.empty_cache()


    def get_data(self):
        raise Exception("NotImplementedException")

    def lossf(self):
        raise Exception("NotImplementedException")

    def while_training(self,iis):
        raise Exception("NotImplementedException")

    def _profiler(self, iis, loss):

        if int(iis / self.prtstep) != self.his:
            print("Perlexity: ", iis, np.exp(loss.item()))
            self.log = self.log + "Perlexity: " + str(iis)+" "+ str(np.exp(loss.item())) + "\n"
            self.his = int(iis / self.prtstep)
        self.train_hist.append(np.exp(loss.item()))

        # if int(iis / self.prtstep) != self.his:
        #     print("Loss: ", iis, loss.item())
        #     self.his = int(iis / self.prtstep)
        # self.train_hist.append(loss.item())

    def _postscript(self):
        x = []
        for ii in range(len(self.train_hist)):
            x.append([ii, self.train_hist[ii]])
        x = np.array(x)
        if self.figure_plot:
            try:
                plt.plot(x[:, 0], x[:, 1])
                if type(self.save_fig) != type(None):
                    plt.savefig(self.save_fig)
                    self.log = self.log + "Figure saved: " + str(self.save_fig) + "\n"
                    plt.gcf().clear()
                else:
                    if self.loss_clip > 0:
                        # plt.ylim((-self.loss_clip, self.loss_clip))
                        plt.ylim((0, self.loss_clip))
                    plt.show()
            except:
                pass

class PyTrain_Custom(PyTrain_Lite):
    """
    A pytrain custom object aiding PyTrain_Lite
    """
    def __init__(self, dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, para=None):
        """
        PyTrain custom
        :param para:
        """
        super(self.__class__, self).__init__(dataset, lsize, rnn, step, learning_rate=learning_rate, batch=batch, window=window, para=para)

        self.data_init = None
        self.databp = None
        self.databp_lab = None
        self.dataset_length = None

        # Interface 1
        self._init_data = getattr(self, self.custom_interface["init_data"])
        # self._evalmem = getattr(self, self.custom_interface["evalmem"])
        self._evalmem = getattr(self, self.custom_interface.get("evalmem", "evalmem_default"))
        self._example_data_collection = getattr(self, self.custom_interface.get("example_data_collection","example_data_collection_default"))
        self._while_training = getattr(self, self.custom_interface.get("while_training","while_training_default"))


        self.get_data = None

        # context controller
        self.context_id=0

        # Last step
        self.data(dataset)
        self.context_switch(0)

    def context_switch(self,context_id):
        # assert context_id<self.context_total
        # self.rnn.context_id=context_id
        # if context_id==0:
        #     # self.lossf=self.custom_loss_pos_auto_0
        #     # self.get_data=self.custom_get_data_pos_auto
        #     # Interface 2
        #     self.lossf = self.raw_loss
        #     self.get_data = self.get_data_continous
            # self.__init_data =
        # elif context_id==1:
        #     self.lossf=self.custom_loss_pos_auto_1
        #     self.get_data=self.custom_get_data_pos_auto

        if context_id==0:
            self.lossf = getattr(self, self.custom_interface["lossf"])
            # Interface
            # self.lossf = self.KNWLoss
            # self.lossf = self.KNWLoss_GateReg_hgate
            # self.lossf = self.KNWLoss_WeightReg
            # self.lossf = self.KNWLoss_WeightReg_GRU
            self.lossf_eval = getattr(self, self.custom_interface["lossf_eval"])
            # self.lossf_eval = self.KNWLoss
            self.get_data = getattr(self, self.custom_interface["get_data"])
            # self.get_data = self.get_data_continous
            # self.get_data = self.get_data_sent_sup
            # self.get_data = self.get_data_seq2seq not working
        # elif context_id==1:
        #     # Interface
        #     # self.lossf = self.KNWLoss
        #     self.lossf = self.KNWLoss_GateReg_hgate
        #     # self.lossf = self.KNWLoss_WeightReg
        #     # self.lossf = self.KNWLoss_WeightReg_GRU
        #     self.lossf_eval = self.KNWLoss
        #     self.get_data = self.get_data_sent_sup
        #     # self.get_data = self.get_data_seq2seq
        else:
            raise Exception("Unknown context")


    def data(self,dataset):
        """
        Swap dataset
        :param dataset:
        :return:
        """
        limit = 1e9
        self.dataset = dataset
        if type(dataset) is list:
            print("Data symbol size: ",len(self.dataset) * self.lsize_in)
        elif type(dataset) is dict:
            print("Data symbol size: ", len(self.dataset["dataset"]) * self.lsize_in)
        if len(self.dataset) * self.lsize_in < limit:
            self.data_init = True
            self._init_data()
        else:
            self.data_init = False
            print("Warning, large dataset, not pre-processed.")

        if (type(dataset) is dict) != self.supervise_mode:
            raise Exception("Supervise mode Error.")

    def eval_mem(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """

        # return self.custom_eval_mem_tdff(x, label, rnn)
        # return self.custom_eval_mem_attn(x, label, rnn)
        return self._evalmem(x, label, rnn)


    def while_training(self,iis):
        # self.context_monitor(iis)
        return self._while_training(iis)

    def context_monitor(self,iis):
        """Monitor progress"""
        pass
        # self.context_id=int(iis/100)%self.context_total
        # self.context_switch(self.context_id)

    def _build_databp(self,inlabs):
        """
        Build databp from inlab (when dataset too large)
        :param inlab:
        :return:
        """
        if self.digit_input and self.id_2_vec is None and not self.data_init:
            datab=np.zeros((len(inlabs),self.lsize_in))
            for ii_b in range(len(inlabs)):
                datab[ii_b,inlabs[ii_b]]=1.0
            databp = torch.from_numpy(np.array(datab))
            # databp = databp.type(torch.FloatTensor)
        else:
            raise Exception("Not Implemented")
        return databp

    def get_data_sent_sup(self,batch=None, rstartv=None):
        assert self.supervise_mode
        assert type(self.dataset["dataset"][0]) == list  # we assume sentence structure
        assert self.data_init

        if batch is None:
            batch=self.batch

        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.dataset["dataset"]) - 1))
        else:
            assert len(rstartv)==batch

        qlen = len(self.dataset["dataset"][0])
        anslen=len(self.dataset["label"][0])
        xl = np.zeros((batch, qlen))
        outl = np.zeros((batch, anslen))
        for iib in range(batch):
            xl[iib, :] = np.array(self.dataset["dataset"][int(rstartv[iib])])
            outl[iib, :] = np.array(self.dataset["label"][int(rstartv[iib])])
        inlab = torch.from_numpy(xl)
        inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(outl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(self.window, batch, self.lsize_in)
        for iib in range(batch):
            vec1=self.databp[int(rstartv[iib])]
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if self.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.device)
            x = x.to(self.device)

        return x, outlab, inlab


    def custom_do_test(self,step_test=300,schedule=1.0):
        """
        Calculate correct rate
        :param step_test:
        :return:
        """
        startt = time.time()
        self.rnn.eval()
        total=0
        correct=0
        correct_ratel=[]
        total_ans = 0
        correct_ans=0
        correct_ratel_ans = []
        for iis in range(step_test):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, label, inlab = self.get_data()
            output, hidden = self.rnn(x, hidden, schedule=schedule)
            output = output.permute(1, 2, 0)
            _,predicted = torch.max(output,1)
            # Calculate digit level correct rate
            total += label.size(0)*label.size(1)
            correct += (predicted == label).sum().item()
            correct_ratel.append(correct/total)
            # Calculate answer level correct rate predicted.shape [batch,outlen]
            total_ans += label.size(0)
            for iib in range(predicted.shape[0]):
                if torch.all(torch.eq(predicted[iib,:], label[iib,:])):
                    correct_ans=correct_ans+1
                else:
                    # print("Question", inlab[iib].cpu().data.numpy(), "Label:", label[iib].cpu().data.numpy(), "Predicted:",
                    #       predicted[iib].cpu().data.numpy())
                    pass
            correct_ratel.append(correct / total)
            correct_ratel_ans.append(correct_ans / total_ans)
        # print("Evaluation Perplexity: ", np.mean(np.array(perpl)))
        print("Digit level Correct rate: ", np.mean(np.array(correct_ratel)))
        print("Ans level Correct rate: ", np.mean(np.array(correct_ratel_ans)))
        endt = time.time()
        print("Time used in test:", endt - startt)

        currentDT = datetime.datetime.now()
        self.log = self.log + "Time of custom test: " + str(currentDT) + "\n"
        self.log = self.log + "Digit level Correct rate: " + str(np.mean(np.array(correct_ratel))) + "\n"
        self.log = self.log + "Ans level Correct rate: " + str(np.mean(np.array(correct_ratel_ans))) + "\n"
        if self.gpuavail:
            torch.cuda.empty_cache()

    def custom_get_data_pos_auto(self):
        """
        Customed data get subroutine for both pos tag and self tag
        :return:
        """
        assert self.supervise_mode
        label=np.array(self.dataset["label"])
        dataset = np.array(self.dataset["dataset"])
        rstartv = np.floor(np.random.rand(self.batch) * (len(dataset) - self.window - 1))

        autol = np.zeros((self.batch, self.window))
        labell = np.zeros((self.batch, self.window))
        for iib in range(self.batch):
            labell[iib, :] = label[int(rstartv[iib]):int(rstartv[iib]) + self.window]
            autol[iib, :] = dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window]
        poslab = torch.from_numpy(labell)
        poslab = poslab.type(torch.LongTensor)
        autolab = torch.from_numpy(autol)
        autolab = autolab.type(torch.LongTensor)

        vec1m = torch.zeros(self.window, self.batch, self.lsize_in)
        for iib in range(self.batch):
            if self.data_init:
                vec1 = self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            else:
                vec1 = self.__build_databp(dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window])
            vec1m[:, iib, :] = vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if self.gpuavail:
            x = x.to(self.device)
            poslab = poslab.to(self.device)
            autolab = autolab.to(self.device)

        return x, [poslab,autolab]

    def KNWLoss(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg


    def example_data_collection(self, x, output, hidden, label,items="all"):
        # return self.custom_example_data_collection_continuous(x, output, hidden)
        # return self.custom_example_data_collection_seq2seq(x, output, hidden, label,items=items)
        # return self.custom_example_data_collection_attention(x, output, hidden, label)
        # return self.custom_example_data_collection_tdff(x, output, hidden, label)
        return self._example_data_collection(x, output, hidden, label)

class PyTrain_Interface_Default(object):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def evalmem(self,*kwargs,**args):
        """
        Defalt evalmem
        called in do_eval
        usage self.eval_mem(x, label, self.rnn)
        :param kwargs:
        :param args:
        :return:
        """
        pass

    def while_training(self,*kwargs,**args):
        """
        Default while_training function
        called while training
        :param kwargs:
        :param args:
        :return:
        """
        pass

    def example_data_collection(self,*kwargs,**args):
        """
        Default example_data_collection function
        called by plot_example
        :param kwargs:
        :param args:
        :return:
        """
        pass

class PyTrain_Interface_others(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def _init_data_sent_sup(self,limit=1e9):
        assert self.supervise_mode
        assert type(self.dataset["dataset"][0]) == list # we assume sentence structure
        assert self.digit_input
        assert self.id_2_vec is None # No embedding, one-hot representation

        self.dataset_length=len(self.dataset["dataset"])
        print("Dataset length ",self.dataset_length)

        if len(self.dataset)*self.lsize_in<limit:
            self.databp=[]
            for sent in self.dataset["dataset"]:
                datab_sent=[]
                for data in sent:
                    datavec = np.zeros(self.lsize_in)
                    datavec[data] = 1.0
                    datab_sent.append(datavec)
                datab_sent = torch.from_numpy(np.array(datab_sent))
                datab_sent = datab_sent.type(torch.FloatTensor)
                self.databp.append(datab_sent)
            self.data_init = True
        else:
            print("Warning, large dataset, not pre-processed.")
            self.databp=None
            self.data_init=False

    def _init_data_all(self,limit=1e9):
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

    def _init_data_sup(self):

        assert self.digit_input
        assert self.id_2_vec is not None
        assert self.supervise_mode

        dataset=self.dataset["dataset"]
        datab=np.zeros((len(dataset),len(self.id_2_vec[0])))
        for ii in range(len(dataset)):
            datavec = np.array(self.id_2_vec[dataset[ii]])
            datab[ii]=datavec
        self.databp = torch.from_numpy(np.array(datab))
        self.databp = self.databp.type(torch.FloatTensor)
        self.data_init = True

    # def __build_databp(self,inlabs):
    #     """
    #     Build databp from inlab (when dataset too large)
    #     :param inlab:
    #     :return:
    #     """
    #     assert self.digit_input
    #     assert self.id_2_vec is not None
    #     assert self.supervise_mode
    #
    #     datab=np.zeros((len(inlabs),self.lsize_in))
    #     for ii_b in range(len(inlabs)):
    #         datab[ii_b,inlabs[ii_b]]=np.array(self.id_2_vec[inlabs[ii_b]])
    #     databp = torch.from_numpy(np.array(datab))
    #     return databp

    def KNWLoss_GateReg_hgate(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # if outputl.shape[-1]==outlab.shape[-1]:
        #     loss1 = lossc(outputl, outlab)
        # else:
        #     loss1 = lossc(outputl[:,:,(outputl.shape[-1]-outlab.shape[-1]):], outlab)

        loss_gate = model.sigmoid(model.hgate)


        # allname = ["Wiz", "Whz", "Win", "Whn"]
        # wnorm1 = 0
        # for namep in allname:
        #     wattr = getattr(self.rnn.gru_enc, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        #     wattr = getattr(self.rnn.gru_dec, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1+self.reg_lamda*torch.mean(loss_gate)#+self.reg_lamda*wnorm1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_WeightReg(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in", "W_out", "W_hd"]
        wnorm1=0
        for namep in allname:
                wattr = getattr(self.rnn.gru_enc, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
                try:
                    wattr = getattr(self.rnn.gru_dec, namep)
                    wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
                except:
                    pass
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/(len(allname))
        return loss1+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_ExpertReg(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["Wiz", "Whz", "Win", "Whn"]
        wnorm1=0
        for namep in allname:
                wattr = getattr(self.rnn.gru_enc, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
                try:
                    wattr = getattr(self.rnn.gru_dec, namep)
                    wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
                except:
                    pass
        wattr = getattr(self.rnn, "h2o")
        logith2o = wattr.weight# + wattr.bias.view(-1)
        pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        # wnorm1=wnorm1/(2*len(allname))
        return loss1+ self.reg_lamda * lossh2o # + self.reg_lamda *wnorm1 # + 0.001 * l1_reg #  + 0.01 * l1_reg



    def KNWLoss_WeightReg_GRU(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        # loss1 = self.lossc(outputl, outlab)
        loss1 = lossc(outputl, outlab)
        wnorm1=0
        if self.rnn.weight_dropout>0:
            weight_ih=self.rnn.gru.module.weight_ih_l0
            weight_hh = self.rnn.gru.module.weight_hh_l0
        else:
            weight_ih = self.rnn.gru.weight_ih_l0
            weight_hh = self.rnn.gru.weight_hh_l0
        wnorm1 = wnorm1 + torch.mean(torch.abs(weight_ih)) + torch.mean(torch.abs(weight_hh))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/6
        return loss1+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    # def raw_loss(self, output, label, rnn, iis):
    #     return output

    def custom_loss_pos_auto_0(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for pos_auto task context0
        outputl=[output11,output12,output21,output22]
        outlab=[poslab,autolab]
        """
        loss11=self.KNWLoss(outputl[0], outlab[0])
        loss12 = self.KNWLoss(outputl[1], outlab[1])
        loss21 = self.KNWLoss(outputl[2], outlab[0])
        loss22 = self.KNWLoss(outputl[3], outlab[1])
        return loss11+loss12+loss21+loss22

    def custom_loss_pos_auto_1(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for pos_auto task context1
        outputl=[output11,output12,output21,output22]
        outlab=[poslab,autolab]
        """
        loss11=self.KNWLoss(outputl[0], outlab[0])
        loss12 = self.KNWLoss(outputl[1], outlab[1])
        loss21 = self.KNWLoss(outputl[2], outlab[0])
        loss22 = self.KNWLoss(outputl[3], outlab[1])
        return loss11-loss12-loss21+loss22

    def custom_loss_pos_auto_eval(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for pos_auto task context1
        outputl=[output11,output12,output21,output22]
        outlab=[poslab,autolab]
        """
        loss11=self.KNWLoss(outputl[0], outlab[0])
        loss12 = self.KNWLoss(outputl[1], outlab[1])
        loss21 = self.KNWLoss(outputl[2], outlab[0])
        loss22 = self.KNWLoss(outputl[3], outlab[1])
        return loss11,loss12,loss21,loss22

    def do_eval_custom_pos_auto(self,step_eval=300):

        startt = time.time()
        self.rnn.eval()
        perpl=[[],[],[],[]]
        for iis in range(step_eval):
            if self.gpuavail:
                hidden = self.rnn.initHidden_cuda(self.device, self.batch)
            else:
                hidden = self.rnn.initHidden(self.batch)
            x, label = self.get_data()
            output, hidden = self.rnn(x, hidden, schedule=1.0)
            loss11, loss12, loss21, loss22 = self.custom_loss_pos_auto_eval(output, label, self.rnn, iis)
            perpl[0].append(loss11.cpu().item())
            perpl[1].append(loss12.cpu().item())
            perpl[2].append(loss21.cpu().item())
            perpl[3].append(loss22.cpu().item())
            # self.eval_mem(output)
        print("Evaluation Perplexity 11: ", np.mean(np.array(perpl[0])))
        print("Evaluation Perplexity 12: ", np.mean(np.array(perpl[1])))
        print("Evaluation Perplexity 21: ", np.mean(np.array(perpl[2])))
        print("Evaluation Perplexity 22: ", np.mean(np.array(perpl[3])))
        endt = time.time()
        print("Time used in evaluation:", endt - startt)
        if self.gpuavail:
            torch.cuda.empty_cache()



class PyTrain_Interface_continous(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def _init_data_continous(self,limit=1e9):
        assert self.digit_input
        assert not self.supervise_mode
        datab = []
        for data in self.dataset:
            if self.id_2_vec is None:  # No embedding, one-hot representation
                datavec = np.zeros(self.lsize_in)
                datavec[data] = 1.0
            else:
                datavec = np.array(self.id_2_vec[data])
            datab.append(datavec)
        self.databp = torch.from_numpy(np.array(datab))
        self.databp = self.databp.type(torch.FloatTensor)

    def get_data_continous(self,batch=None):

        if batch is None:
            batch=self.batch

        if not self.supervise_mode:
            # Generating output label
            rstartv = np.floor(np.random.rand(batch) * (len(self.dataset) - self.window - 1))
            yl = np.zeros((batch,self.window))
            xl = np.zeros((batch,self.window))
            for iib in range(batch):
                xl[iib,:]=self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window]
                yl[iib,:]=self.dataset[int(rstartv[iib])+1:int(rstartv[iib]) + self.window+1]
            # inlab = torch.from_numpy(xl)
            # inlab = inlab.type(torch.LongTensor)
            outlab = torch.from_numpy(yl)
            outlab = outlab.type(torch.LongTensor)

        else:
            rstartv = np.floor(np.random.rand(batch) * (len(self.dataset["dataset"]) - self.window - 1))
            xl = np.zeros((batch, self.window))
            for iib in range(batch):
                xl[iib, :] = np.array(self.dataset["label"][int(rstartv[iib]):int(rstartv[iib]) + self.window])
            outlab = torch.from_numpy(xl)
            outlab = outlab.type(torch.LongTensor)
            # inlab = outlab

        vec1m = torch.zeros(self.window, batch, self.lsize_in)
        # vec2m = torch.zeros(self.window, batch, self.lsize_in)
        for iib in range(batch):
            # vec1_raw = self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            # vec1_rnd = torch.rand(vec1_raw.shape)
            # vec1_add = torch.mul((1.0 - vec1_raw) * self.invec_noise, vec1_rnd.double())
            # vec1 = vec1_raw + vec1_add
            if self.data_init:
                vec1=self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
            else:
                vec1=self._build_databp(self.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window])
            # vec2 = self.databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.window + 1, :]
            vec1m[:,iib,:]=vec1
            # vec2m[:, iib, :] = vec2
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor) #
        # y = Variable(vec2m, requires_grad=True)

        if self.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.device)
            x = x.to(self.device)

        return x, outlab, None

    def custom_example_data_collection_continuous(self, x, output, hidden, label):

        if self.data_col_mem is None:
            # Step 1 of example_data_collection
            self.data_col_mem=dict([])
            self.data_col_mem["titlelist"]=["input","predict","hidden","zt keep gate","nt set gate"]
            self.data_col_mem["sysmlist"]=[True,False,True,True,True]
            self.data_col_mem["mode"]="predict"
            self.data_col_mem["datalist"] = [None,None,None,None,None]

        if self.data_col_mem["datalist"][0] is None:
            self.data_col_mem["datalist"][0] = x.view(1, -1)
            self.data_col_mem["datalist"][1] = output.view(1, -1)
            self.data_col_mem["datalist"][2] = hidden.view(1, -1)
            self.data_col_mem["datalist"][3] = self.rnn.zt.view(1, -1)
            self.data_col_mem["datalist"][4] = self.rnn.nt.view(1, -1)
        else:
            self.data_col_mem["datalist"][0] = torch.cat((self.data_col_mem["datalist"][0], x.view(1, -1)), dim=0)
            self.data_col_mem["datalist"][1] = torch.cat((self.data_col_mem["datalist"][1], output.view(1, -1)), dim=0)
            self.data_col_mem["datalist"][2] = torch.cat((self.data_col_mem["datalist"][2], hidden.view(1, -1)), dim=0)
            self.data_col_mem["datalist"][3] = torch.cat((self.data_col_mem["datalist"][3], self.rnn.zt.view(1, -1)), dim=0)
            self.data_col_mem["datalist"][4] = torch.cat((self.data_col_mem["datalist"][4], self.rnn.nt.view(1, -1)), dim=0)

class PyTrain_Interface_encdec(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def custom_eval_mem_enc_dec(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.evalmem is None:
            self.evalmem = [[] for ii in range(8)]  # x,label,enc(ht,zt,nt),dec(ht,zt,nt)
        else:
            try:
                self.evalmem[0].append(x.cpu().data.numpy())
                self.evalmem[1].append(label.cpu().data.numpy())
                self.evalmem[2].append(rnn.gru_enc.ht.cpu().data.numpy())
                self.evalmem[3].append(rnn.gru_enc.zt.cpu().data.numpy())
                self.evalmem[4].append(rnn.gru_enc.nt.cpu().data.numpy())
                self.evalmem[5].append(rnn.gru_dec.ht.cpu().data.numpy())
                self.evalmem[6].append(rnn.gru_dec.zt.cpu().data.numpy())
                self.evalmem[7].append(rnn.gru_dec.nt.cpu().data.numpy())
            except:
                # print("eval_mem failed")
                pass

    def KNWLoss_GateReg_encdec(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # if outputl.shape[-1]==outlab.shape[-1]:
        #     loss1 = lossc(outputl, outlab)
        # else:
        #     loss1 = lossc(outputl[:,:,(outputl.shape[-1]-outlab.shape[-1]):], outlab)
        # loss_gate = model.siggate
        # loss_gate = model.sigmoid(model.hgate)

        loss_gate_enc = (model.sigmoid(model.gru_enc.Whz_mask)+model.sigmoid(model.gru_enc.Whn_mask))/2
        loss_gate_dec = (model.sigmoid(model.gru_dec.Whz_mask)+model.sigmoid(model.gru_dec.Whn_mask))/2

        # allname = ["Wiz", "Whz", "Win", "Whn"]
        # wnorm1 = 0
        # for namep in allname:
        #     wattr = getattr(self.rnn.gru_enc, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        #     wattr = getattr(self.rnn.gru_dec, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1+self.reg_lamda*torch.mean((loss_gate_enc+loss_gate_dec)/2) # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_GateReg_encdec_L1(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        loss_gate_enc = (model.sigmoid(model.gru_enc.Whz_mask)+model.sigmoid(model.gru_enc.Whn_mask))/2
        loss_gate_dec = (model.sigmoid(model.gru_dec.Whz_mask)+model.sigmoid(model.gru_dec.Whn_mask))/2

        allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        wnorm1 = 0
        for namep in allname:
                wattr = getattr(self.rnn.gru_enc, namep)
                wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
                wattr = getattr(self.rnn.gru_dec, namep)
                wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        return loss1+self.reg_lamda*torch.mean((loss_gate_enc+loss_gate_dec)/2)+self.reg_lamda*wnorm1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

class PyTrain_Interface_seq2seq(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def _init_data_seq2seq(self,limit=1e9):

        assert self.supervise_mode
        assert type(self.dataset["dataset"][0]) == list # we assume sentence structure
        assert self.digit_input
        assert self.id_2_vec is None # No embedding, one-hot representation

        if len(self.dataset)*self.lsize_in<limit:
            self.databp=[]
            for sent in self.dataset["dataset"]:
                datab_sent=[]
                for data in sent:
                    datavec = np.zeros(self.lsize_in)
                    datavec[data] = 1.0
                    datab_sent.append(datavec)
                datab_sent = torch.from_numpy(np.array(datab_sent))
                datab_sent = datab_sent.type(torch.FloatTensor)
                self.databp.append(datab_sent)
            self.databp_lab=[]
            for sent in self.dataset["label"]:
                datab_sent=[]
                datavec = np.zeros(self.lsize_in)
                datab_sent.append(datavec)
                for data in sent:
                    datavec = np.zeros(self.lsize_in)
                    datavec[data] = 1.0
                    datab_sent.append(datavec)
                del datab_sent[-1] # shift label right
                datab_sent = torch.from_numpy(np.array(datab_sent))
                datab_sent = datab_sent.type(torch.FloatTensor)
                self.databp_lab.append(datab_sent)
            self.data_init = True
        else:
            print("Warning, large dataset, not pre-processed.")
            self.databp=None
            self.data_init=False

    def get_data_seq2seq(self,batch=None):
        assert self.supervise_mode
        assert type(self.dataset["dataset"][0]) == list  # we assume sentence structure
        assert self.data_init

        if batch is None:
            batch=self.batch

        rstartv = np.floor(np.random.rand(batch) * (len(self.dataset["dataset"]) - 1))
        qlen = len(self.dataset["dataset"][0])
        anslen=len(self.dataset["label"][0])
        xl = np.zeros((batch, qlen))
        outl = np.zeros((batch, anslen))
        for iib in range(batch):
            xl[iib, :] = np.array(self.dataset["dataset"][int(rstartv[iib])])
            outl[iib, :] = np.array(self.dataset["label"][int(rstartv[iib])])
        inlab = torch.from_numpy(xl)
        inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(outl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(qlen, batch, self.lsize_in)
        for iib in range(batch):
            vec1=self.databp[int(rstartv[iib])]
            vec1m[:,iib,:]=vec1
        x_in = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        vec2m = torch.zeros(anslen, batch, self.lsize_in)
        for iib in range(batch):
            vec2 = self.databp_lab[int(rstartv[iib])]
            vec2m[:, iib, :] = vec2
        x_dec = Variable(vec2m, requires_grad=True).type(torch.FloatTensor)

        if self.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.device)
            x_in = x_in.to(self.device)
            x_dec = x_dec.to(self.device)

        return [x_in,x_dec], outlab, inlab

    def custom_example_data_collection_seq2seq(self, x, output, hidden, label, items=[3,4]):

        # datalist=[xnp.T,outmat.T,hidmat.T,ztmat.T,ntmat.T]
        # datalist = [[xnp.T,xnp.T], [outmat.T,outmat.T], [hidmat.T,hidmat.T], [ztmat.T,ztmat.T], [ntmat.T,ntmat.T]]
        # titlelist=
        # sysmlist=

        # temp_data_col_mem

        # if self.data_col_mem is None:
        # Step 1 of example_data_collection
        temp_data_col_mem = dict([])
        temp_data_col_mem["titlelist"] = ["input","output","hidden","zt forget gate","nt set gate"]
        temp_data_col_mem["sysmlist"] = [True,False,True,True,True]
        temp_data_col_mem["mode"] = "seq2seq"
        temp_data_col_mem["datalist"] = [None,None,None,None,None]
        temp_data_col_mem["climlist"] = [[None,None],[None,[1,-5]],[None,None],[None,None],[None,None]]

        length=x.shape[0]
        lsize=x.shape[-1]
        anslength=label.shape[-1]
        hdsize_enc=self.rnn.gru_enc.ht.shape[-1]
        hdsize_dec = self.rnn.gru_dec.ht.shape[-1]
        label_onehot=torch.zeros(anslength,lsize)
        for ii in range(label.shape[-1]):
            id=label[0,ii]
            label_onehot[ii,id]=1
        temp_data_col_mem["datalist"][0] = [x.view(length,lsize),label_onehot]
        temp_data_col_mem["datalist"][1] = [torch.zeros(length,lsize), output.view(anslength,lsize)]
        temp_data_col_mem["datalist"][2] = [self.rnn.gru_enc.ht.view(length, hdsize_enc),
                                            self.rnn.gru_dec.ht.view(anslength, hdsize_dec)]
        temp_data_col_mem["datalist"][3] = [self.rnn.gru_enc.zt.view(length, hdsize_enc),
                                            self.rnn.gru_dec.zt.view(anslength, hdsize_dec)]
        temp_data_col_mem["datalist"][4] = [self.rnn.gru_enc.nt.view(length, hdsize_enc),
                                            self.rnn.gru_dec.nt.view(anslength, hdsize_dec)]
        # self.data_col_mem["datalist"][1] = output.view(1, -1)
        # self.data_col_mem["datalist"][2] = hidden.view(1, -1)

        if items == "all":
            self.data_col_mem=temp_data_col_mem
        else:
            assert type(items) is list
            self.data_col_mem=dict([])
            self.data_col_mem["mode"] = "seq2seq"

            self.data_col_mem["titlelist"] = []
            self.data_col_mem["sysmlist"] = []
            self.data_col_mem["datalist"] = []
            self.data_col_mem["climlist"] = []
            for num in items:
                self.data_col_mem["titlelist"].append(temp_data_col_mem["titlelist"][num])
                self.data_col_mem["sysmlist"].append(temp_data_col_mem["sysmlist"][num])
                self.data_col_mem["datalist"].append(temp_data_col_mem["datalist"][num])
                self.data_col_mem["climlist"].append(temp_data_col_mem["climlist"][num])

class PyTrain_Interface_attn(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def KNWLoss_WeightReg_Attn(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in2V"]
        wnorm1=0
        for namep in allname:
            wattr = getattr(self.rnn.hd_attn1, namep)
            wnorm1= wnorm1+torch.mean(torch.abs(wattr.weight))
            # wattr = getattr(self.rnn.hd_attn2, namep)
            # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
            # wattr = getattr(self.rnn.hd_att_ff, namep)
            # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        allname = ["W_hff","W_ff"]
        for namep in allname:
            wattr = getattr(self.rnn, namep)
            wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/3
        return loss1+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_example_data_collection_attention(self, x, output, hidden, label):

        lsize = x.shape[-1]
        anslength = label.shape[-1]
        label_onehot = torch.zeros(anslength, lsize)
        for ii in range(label.shape[-1]):
            id = label[0, ii]
            label_onehot[ii, id] = 1

        if self.data_col_mem is None:
            # Step 1 of example_data_collection
            self.data_col_mem=dict([])
            self.data_col_mem["titlelist"]=["input","label","predict","attn1","hd0","hdout"]
            self.data_col_mem["sysmlist"]=[True,True,True,True,True,True]
            self.data_col_mem["mode"]="predict"
            self.data_col_mem["datalist"] = [None,None,None,None,None,None]
            maxPred = np.max(output.cpu().data.numpy())
            self.data_col_mem["climlist"] = [[None, None], [None, None], [None, None] ,[None, None], [None, None], [None, None]]

        if self.data_col_mem["datalist"][0] is None:
            print(x.shape,output.shape,self.rnn.attnM1.shape)
            self.data_col_mem["datalist"][0] = torch.squeeze(x)
            self.data_col_mem["datalist"][1] = torch.squeeze(label_onehot)
            self.data_col_mem["datalist"][2] = torch.squeeze(output)
            self.data_col_mem["datalist"][3] = torch.squeeze(self.rnn.attnM1).transpose(1, 0)
            self.data_col_mem["datalist"][4] = torch.squeeze(self.rnn.hd0)
            self.data_col_mem["datalist"][5] = self.rnn.hdout.view(-1,1)
            # self.data_col_mem["datalist"][5] = torch.squeeze(self.rnn.attnM3)

    def custom_eval_mem_attn(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.evalmem is None:
            # self.evalmem = [[] for ii in range(5)]  # x,label,attn1,attn2,attn_ff
            self.evalmem = [[] for ii in range(5)]  # x,label,attn1,hd0,hdout
        else:
            try:
                self.evalmem[0].append(x.cpu().data.numpy())
                self.evalmem[1].append(label.cpu().data.numpy())
                self.evalmem[2].append(rnn.attnM1.cpu().data.numpy())
                self.evalmem[3].append(rnn.hd0.cpu().data.numpy())
                # self.evalmem[3].append(rnn.attnM2.cpu().data.numpy())
                # self.evalmem[4].append(rnn.attnM3.cpu().data.numpy())
                self.evalmem[4].append(rnn.hdout.cpu().data.numpy())
            except:
                # print("eval_mem failed")
                pass



class PyTrain_Interface_tdff(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        pass

    def KNWLoss_WeightReg_TDFF(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in", "W_out","Whd0"]
        wnorm1=0
        for namep in allname:
            wattr = getattr(self.rnn, namep)
            wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        # for ii in range(self.rnn.num_layers):
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(self.rnn.hd_layer_stack[ii][1].weight))
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/len(allname)
        return loss1+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_WeightEnergyReg_TDFF(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in", "W_out","Whd0"]
        wnorm1=0
        for namep in allname:
            wattr = getattr(self.rnn, namep)
            wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
            # wnorm1 = wnorm1 + torch.mean(torch.mul(wattr.weight, wattr.weight))

        wnorm1 = wnorm1 / len(allname)

        energyloss=torch.mean(torch.mul(model.Wint,model.Wint))+torch.mean(torch.mul(model.hdt,model.hdt))

        # energyloss = torch.mean(torch.abs(model.Wint))+ torch.mean(torch.abs(model.hdt))

        # print(model.Wint.norm(2),model.hdt.norm(2))

        return loss1+self.reg_lamda*(energyloss+wnorm1) # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_GateReg_TDFF_L1(self, outputl, outlab, model=None, cstep=None, posi_ctrl=1):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        if posi_ctrl is None:
            loss1 = lossc(outputl, outlab)
        else:
            loss1 = lossc(outputl[:,:,posi_ctrl], outlab[:,posi_ctrl])

        # loss_gate = (torch.mean(model.sigmoid(model.W_in_mask))
        #              + torch.mean(model.sigmoid(model.W_out_mask))
        #              + torch.mean(model.sigmoid(model.Whd0_mask))) / 2

        loss_gate = torch.mean(model.sigmoid(model.input_gate))

        allname = ["W_in", "W_out","Whd0"]
        wnorm1=0
        for namep in allname:
                wattr = getattr(self.rnn, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1 + 0.01*wnorm1 + self.reg_lamda*loss_gate # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_eval_mem_tdff(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.evalmem is None:
            self.evalmem = [[] for ii in range(5)]  # x,label,hd_in,hd0, output
        else:
            try:
                self.evalmem[0].append(x.cpu().data.numpy())
                self.evalmem[1].append(label.cpu().data.numpy())
                self.evalmem[2].append(rnn.Wint.cpu().data.numpy())
                self.evalmem[3].append(rnn.hdt.cpu().data.numpy())
                self.evalmem[4].append(rnn.lgoutput.cpu().data.numpy())
            except:
                # print("eval_mem failed")
                pass

    def custom_whiletraining_gradcollect_tdff(self,iis):
        """
        Customed gradient information collection function for tdff
        :return:
        """
        if self.training_data_mem is None:
            self.training_data_mem=dict([])
            self.training_data_mem["gradInfo"]=dict([])
            for item in ["W_in","Whd0","W_out"]:
                self.training_data_mem["gradInfo"][item]=[]
        for item in ["W_in","Whd0","W_out"]:
            attr=getattr(self.rnn,item)
            grad=attr.weight.grad
            self.training_data_mem["gradInfo"][item].append(grad)

    def custom_example_data_collection_tdff(self, x, output, hidden, label):

        lsize = x.shape[-1]
        anslength = label.shape[-1]
        label_onehot = torch.zeros(anslength, lsize)
        for ii in range(label.shape[-1]):
            id = label[0, ii]
            label_onehot[ii, id] = 1

        if self.data_col_mem is None:
            # Step 1 of example_data_collection
            self.data_col_mem=dict([])
            self.data_col_mem["titlelist"]=["input","label","predict","Wint","hd1"]
            self.data_col_mem["sysmlist"]=[True,True,True,True,True]
            self.data_col_mem["mode"]="predict"
            self.data_col_mem["datalist"] = [None,None,None,None,None]
            self.data_col_mem["climlist"] = [[None, None], [None, None], [1, -20] ,[None, None], [None, None]]

        if self.data_col_mem["datalist"][0] is None:
            self.data_col_mem["datalist"][0] = torch.squeeze(x)
            self.data_col_mem["datalist"][1] = torch.squeeze(label_onehot)
            self.data_col_mem["datalist"][2] = torch.squeeze(output)
            self.data_col_mem["datalist"][3] = self.rnn.Wint.view(-1,1)
            self.data_col_mem["datalist"][4] = self.rnn.hdt[0].view(-1,1)
            # self.data_col_mem["datalist"][5] = self.rnn.hdt[1].view(-1,1)

    def custom_example_data_collection_tdffrnn(self, x, output, hidden, label):

        lsize = x.shape[-1]
        anslength = label.shape[-1]
        label_onehot = torch.zeros(anslength, lsize)
        for ii in range(label.shape[-1]):
            id = label[0, ii]
            label_onehot[ii, id] = 1

        if self.data_col_mem is None:
            # Step 1 of example_data_collection
            self.data_col_mem=dict([])
            self.data_col_mem["titlelist"]=["input","label","predict","hiddenl"]
            self.data_col_mem["sysmlist"]=[True,True,True,True]
            self.data_col_mem["mode"]="predict"
            self.data_col_mem["datalist"] = [None,None,None,None]
            self.data_col_mem["climlist"] = [[None, None], [None, None], [None, None], [None, None]]

        if self.data_col_mem["datalist"][0] is None:
            self.data_col_mem["datalist"][0] = torch.squeeze(x)
            self.data_col_mem["datalist"][1] = torch.squeeze(label_onehot)
            self.data_col_mem["datalist"][2] = torch.squeeze(output)
            self.data_col_mem["datalist"][3] = torch.squeeze(self.rnn.hdt).transpose(1,0)
            # self.data_col_mem["datalist"][5] = self.rnn.hdt[1].view(-1,1)


class PyTrain_Interface_backwardreverse(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """

    def __init__(self):
        pass

    def _init_data_sup_backwardreverse(self,limit=1e9):
        """
        _init_data_sup_backwardreverse 2019.8.13

        :param limit:
        :return:
        """
        assert self.supervise_mode
        assert len(self.dataset["dataset"]) == 2 # data_set,pvec_l
        assert self.digit_input
        assert self.id_2_vec is None # No embedding, one-hot representation

        self.dataset_length = len(self.dataset["dataset"][0])
        print("Dataset length ", self.dataset_length)

        self.databp=torch.zeros((len(self.dataset["dataset"][0]),self.lsize_in))
        for ii, data in enumerate(self.dataset["dataset"][0]):
            self.databp[ii,data] = 1.0
        self.data_init = True

    def get_data_sup_backwardreverse(self,batch=None, rstartv=None):
        assert self.supervise_mode
        assert len(self.dataset["dataset"]) == 2  # data_set,pvec_l
        assert self.data_init

        if batch is None:
            batch=self.batch

        if rstartv is None: # random mode
            rstartv = np.floor(np.random.rand(batch) * (len(self.dataset["dataset"][0]) - 1))
        else:
            assert len(rstartv)==batch

        xl = np.zeros(batch)
        outl = np.zeros(batch)
        for iib in range(batch):
            xl[iib] = self.dataset["dataset"][0][int(rstartv[iib])]
            outl[iib] = self.dataset["label"][int(rstartv[iib])]
        inlab = torch.from_numpy(xl)
        inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(outl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(batch, self.lsize_in)
        for iib in range(batch):
            vec1=self.databp[int(rstartv[iib])]
            vec1m[iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        pvec_mat = torch.zeros(batch, self.lsize_in)
        for iib in range(batch):
            vec1=self.dataset["dataset"][1][int(rstartv[iib])]
            pvec_mat[iib,:]=torch.from_numpy(vec1)
        pvec_matv = Variable(pvec_mat, requires_grad=True).type(torch.FloatTensor)

        if self.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.device)
            x = x.to(self.device)
            pvec_matv = pvec_matv.to(self.device)

        # print(x.shape,pvec_matv.shape)
        return (x, pvec_matv) , outlab, inlab

    def KNWLoss_backwardreverse(self, outputl, outlab, model=None, cstep=None):

        assert len(outputl)==2
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(model.softmax(outputl[0]+outputl[1]), outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        # allname = ["W_in", "W_out", "W_hd"]
        # wnorm1=0
        # for namep in allname:
        #         wattr = getattr(self.rnn.gru_enc, namep)
        #         wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        #         try:
        #             wattr = getattr(self.rnn.gru_dec, namep)
        #             wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        #         except:
        #             pass
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)

        loss2=torch.nn.functional.kl_div(model.nsoftmax(outputl[0]),model.nsoftmax(outputl[1]))

        return loss1-loss2 #+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_eval_mem_backwardreverse(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.evalmem is None:
            self.evalmem = [[] for ii in range(3)]  # label,p_vec, output
        else:
            try:
                self.evalmem[0].append(x[0].cpu().data.numpy())
                self.evalmem[1].append(x[1].cpu().data.numpy())
                self.evalmem[2].append(rnn.lgoutput.cpu().data.numpy())
            except:
                # print("eval_mem failed")
                pass

    def custom_example_data_collection_backwardreverse(self, x, output, hidden, label):

        print(output)

        if self.data_col_mem is None:
            # Step 1 of example_data_collection
            self.data_col_mem=dict([])
            self.data_col_mem["titlelist"]=["input","p_vec","sample_vec"]
            self.data_col_mem["sysmlist"]=[True,True,True]
            self.data_col_mem["mode"]="predict"
            self.data_col_mem["datalist"] = [None,None,None]
            self.data_col_mem["climlist"] = [[None, None], [None, None], [None, None]]

        if self.data_col_mem["datalist"][0] is None:
            self.data_col_mem["datalist"][0] = x[0]
            self.data_col_mem["datalist"][1] = x[1]
            self.data_col_mem["datalist"][2] = output[0]

