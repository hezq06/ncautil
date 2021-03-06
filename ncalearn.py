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
import shutil
import subprocess

import torch
import copy
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

# from wordcloud import WordCloud
import operator
from PIL import Image
from PIL import ImageDraw,ImageFont

from tqdm import tqdm
import datetime

from ncautil.ncamath import *
from ncautil.datautil import save_model, load_model,save_data,load_data

def pltscatter(data,dim=(0,1),labels=None,title=None,xlabel=None,ylabel=None,color=None):
    assert data.shape[0]>data.shape[1]
    for i in range(len(data)):
        x,y=data[i,dim[0]], data[i,dim[1]]
        if color is None:
            plt.scatter(x,y)
        else:
            plt.scatter(x, y, c=color[i])
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
        if labels is not None:
            ax.text(data[ii, 0], data[ii, 1], data[ii, 2], str(labels[ii]),color='k')

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def plot_mat(data,start=0,lim=1000,symmetric=False,title=None,tick_step=None,show=True,xlabel=None,ylabel=None, clim=None):
    if show:
        plt.figure()
    data=np.array(data)
    if len(data.shape) != 2:
        data=data.reshape(1,-1)
    img=data[:,start:start+lim]
    if symmetric:
        plt.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)), np.amax(np.abs(data))))
        # plt.imshow(img, cmap='seismic', clim=(-2,2))
    else:
        plt.imshow(img, cmap='seismic',clim=clim)
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

def plot_mat_sub(datal,start=0,lim=1000,symmetric=True, partition = (3,3), subtitles = None):
    """
    Subplot a list of mat
    :param data:
    :param start:
    :param range:
    :param symmetric:
    :return:
    """
    N=len(datal)
    # assert N<6
    nrows, ncols = partition
    assert N == nrows * ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    maxdata = np.amax(np.array(datal))
    mindata = np.amin(np.array(datal))

    for ii in range(len(datal)):
        ax=axes.flat[ii]
        data=datal[ii]
        data = np.array(data)
        assert len(data.shape) == 2
        img = data[:, start:start + lim]
        if symmetric:
            im=ax.imshow(img, cmap='seismic', clim=(-maxdata, maxdata))
        else:
            im=ax.imshow(img, cmap='seismic', clim=(mindata, maxdata))
        if subtitles is not None:
            ax.set_title(str(subtitles[ii]))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

def plot_mat_ax(ax, img, symmetric=False, title=None, tick_step=None, clim=None,xlabel=None,ylabel=None):
    img = np.array(img)
    assert len(img.shape) <= 2
    if len(img.shape)==1:
        img=img.reshape(1,-1)
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

def pl_wordcloud(text_freq):
    """
    plot a word cloud
    :param text_freq: "wrd":preq
    :return:
    """
    wordcloud = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(text_freq)
    data = wordcloud.to_array()
    img = Image.fromarray(data, 'RGB')
    img = img.convert("RGBA")

    plt.imshow(img)
    plt.axis("on")
    plt.margins(x=0, y=0)
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
    prior=prior/np.max(prior)
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
                # conprcol.append(np.log(prior[iter_ii]))
                conprcol.append(prior[iter_ii])
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

class PyTrain_Main(object):
    """
    A main class to run training
    """
    def __init__(self, loss_model, data_dict, device="cuda:0", para=None):

        self.data_dict = data_dict
        self.device = device

        if para is None:
            para = dict([])
        self.para(para)

        # Only for self defined model utility
        if self.dist_data_parallel:
            loss_model = loss_model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel( loss_model, device_ids=[device], output_device=device, dim=self.dist_data_parallel_dim,
                                                                 find_unused_parameters=False)
            self.pt_model = self.model.module
        else:
            self.model = loss_model
            self.model.model = self.model.model.to(device)
            self.pt_model = self.model

        currentDT = datetime.datetime.now()
        self.log = "Time of creation: " + str(currentDT) + "\n"
        self.train_hist = []
        self.eval_hist = []
        self.evalmem = None

    def para(self, para):
        self.loss_exp_flag = para.get("loss_exp_flag", False)
        self.figure_plot = para.get("figure_plot", True)
        self.loss_clip = para.get("loss_clip", 50.0)
        self.print_step = para.get("print_step", 200)
        self.dist_data_parallel = para.get("dist_data_parallel", False)
        self.dist_data_parallel_dim = para.get("dist_data_parallel_dim", 1)
        self.check_point_path = para.get("check_point_path", None)


    def run_training(self, epoch=2, lr=1e-3, optimizer_label="adam", weight_decay=0.0, warm_up_steps=10000, mix_precision=False, lr_linear_decay=False):

        currentDT = datetime.datetime.now()
        print("Time of training starting %s. "%str(currentDT))
        self.log = self.log + "Start training with epoch: %s, lr: %s, optimizer: %s, weight_decay: %s"%(epoch, lr, optimizer_label, weight_decay)

        if optimizer_label == "adam":
            print("Using adam optimizer")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=weight_decay)
        elif optimizer_label == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=weight_decay)
        else:
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        startt = time.time()

        if mix_precision:
            assert "cuda" in self.device
            scaler = torch.cuda.amp.GradScaler()
            print("Check @autocast() decorator of model for mix_precision.")

        self.model.train()

        checkpoint_pathname = None
        best_evalres = None

        step_per_epoch = len(self.data_dict["train"])
        if step_per_epoch < warm_up_steps:
            print("Warning, step_per_epoch<warm_up_steps.")

        for ii_epoch in range(epoch):
            print("Starting epoch %s." % str(ii_epoch))
            self.pt_model.loss_mode = "train"

            for iis,(datax, labels) in enumerate(self.data_dict["train"]):

                ii_tot = iis + ii_epoch * step_per_epoch
                cstep = ii_tot / (epoch * step_per_epoch)

                if ii_epoch==0 and warm_up_steps>0 and iis<warm_up_steps: # Warm up for first epoch
                    optimizer.param_groups[0]["lr"]=(iis/warm_up_steps)*lr
                elif lr_linear_decay:
                    lr_cstep = (ii_tot-warm_up_steps) / (epoch * step_per_epoch-warm_up_steps)
                    optimizer.param_groups[0]["lr"] = (1-lr_cstep)*lr
                else:
                    optimizer.param_groups[0]["lr"] = lr

                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = self.model(datax, labels.to(self.device), schedule=cstep)
                self._profiler(ii_tot, loss, optimizer.param_groups[0]["lr"],print_step=self.print_step)

                if mix_precision:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            midt = time.time()
            print("Time used till now:", midt - startt)
            print("Validation of epoch ", ii_epoch, ":")
            evalres = self.do_eval()
            ## Eval profile
            if self.check_point_path is not None:
                if checkpoint_pathname is None:
                    best_evalres = evalres
                    checkpoint_pathname = self.check_point_path+"epoch%s"%ii_epoch+"perp%.5s"%evalres
                    endt = time.time()
                    print("Time used util now:", endt - startt)
                    self.log = self.log + "Time used util now: " + str(endt - startt) + "\n"
                    self.save_session(checkpoint_pathname)
                elif evalres<best_evalres: # We get a better
                    try:
                        shutil.rmtree(checkpoint_pathname)
                    except:
                        pass
                    best_evalres = evalres
                    checkpoint_pathname = self.check_point_path + "epoch%s" % ii_epoch + "perp%.5s" % evalres
                    endt = time.time()
                    print("Time used util now:", endt - startt)
                    self.log = self.log + "Time used util now: " + str(endt - startt) + "\n"
                    self.save_session(checkpoint_pathname)
                else:
                    pass
        endt = time.time()
        print("Time used in training:", endt - startt)
        self.log = self.log + "Time used in training: " + str(endt - startt) + "\n"
        self._postscript()

    def do_eval(self, data_pt = "val", eval_mem_flag=False):
        self.model.eval()
        self.model.loss_mode="eval"
        # Validation
        lossl = []
        print("Start evaluation ...", len(self.data_dict[data_pt]))
        startt = time.time()
        for iis, (datax, labels) in tqdm(enumerate(self.data_dict[data_pt])):
            with torch.no_grad():
                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = self.model(datax, labels.to(self.device), schedule=1.0)
                lossl.append(loss.item())
                if eval_mem_flag:
                    self.eval_mem(datax, labels, self.model)
        if self.loss_exp_flag:
            evalres = np.exp(np.mean(np.array(lossl)))
            print("Evaluation Perplexity: ", evalres)
            self.eval_hist.append(np.exp(np.mean(np.array(lossl))))
        else:
            evalres = np.mean(np.array(lossl))
            print("Evaluation Perplexity: ", evalres)
            self.eval_hist.append(np.mean(np.array(lossl)))
        endt = time.time()
        print("Time used in evaluation: ", endt - startt)
        return evalres

    def do_test(self, data_pt = "val"):
        """
        Calculate correct rate
        :param step_test:
        :return:
        """
        print("Start testing ...")
        self.model.eval()
        total = 0
        correct = 0
        for iis, (datax, labels) in tqdm(enumerate(self.data_dict[data_pt])):
            with torch.no_grad():
                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = self.model(datax, labels.to(self.device), schedule=1.0)
                output = self.model.output
                _, predicted = torch.max(output, -1)
                predicted = predicted.cpu()
                labels = labels.squeeze()
                if len(predicted.shape)==len(labels.shape):
                    total += np.prod(labels.shape)
                    correct += (predicted == labels).sum().item()
                elif len(labels)==1:
                    labels = labels.view(labels.shape[0],1).expand(labels.shape[0],predicted.shape[1])
                    total += labels.shape[0]*labels.shape[1]
                    correct += (predicted == labels).sum().item()
                else:
                    raise Exception("Labels unknown shape")
        crate = correct / total
        print("Correct rate: ", correct / total)

        return crate

    def eval_mem(self, datax, labels, model):
        if self.evalmem is None:
            # self.evalmem = []
            self.evalmem = [[] for ii in range(12)]  # x,label,context
        if self.mem_eval_mode == "task2":
            self.evalmem[0].append(model.model.seq1_coop.output.detach().cpu().numpy())
            self.evalmem[1].append(model.model.seq1_coop.contextl.detach().cpu().numpy())
            self.evalmem[2].append(datax[1].cpu().numpy())
        elif self.mem_eval_mode == "cnn_sup":
            self.evalmem[0].append(model.model.context.detach().cpu().numpy())
        elif self.mem_eval_mode == "auto_encode":
            self.evalmem[0].append(model.model.out_seq1.detach().cpu().numpy())
            self.evalmem[1].append(model.model.seq1_coop.cooprer.context.detach().cpu().numpy())
        elif self.mem_eval_mode == "pos":
            softmax = torch.nn.Softmax(dim=-1)
            self.evalmem[0].append(softmax(model.model.output.detach().cpu()).numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            self.evalmem[2].append(torch.exp(model.model.seq2_train.context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.seq2_train.gssample.detach().cpu().numpy())
        elif self.mem_eval_mode == "auto_word":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent.cpu().data.numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            # cooprer
            self.evalmem[2].append(torch.exp(model.model.cooprer.seq1_coop.seq2_train.context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.cooprer.seq1_coop.seq2_train.gssample.detach().cpu().numpy())
            # trainer
            self.evalmem[4].append(torch.exp(model.model.trainer.seq2_train.context.detach().cpu()).numpy())
            self.evalmem[5].append(model.model.trainer.seq2_train.gssample.detach().cpu().numpy())

        elif self.mem_eval_mode == "self_pred_with_mask":
            # Output
            mask = model.model.mask_bigru.input_mask
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            entm = torch.sum(ent * (1 - mask)) / torch.sum(1 - mask)
            self.evalmem[0].append(entm.cpu().data.numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            # cooprer
            self.evalmem[2].append(torch.exp(model.model.infobn.context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.infobn.gssample.detach().cpu().numpy())
            # mask
            self.evalmem[4].append(mask.cpu().numpy())

        elif self.mem_eval_mode == "auto_word_att":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent.cpu().data.numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            # cooprer
            self.evalmem[2].append(torch.exp(model.model.sfms.context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.sfms.gssample.detach().cpu().numpy())
            # trainer
            self.evalmem[4].append(torch.exp(model.model.infobn_sem.context.detach().cpu()).numpy())
            self.evalmem[5].append(model.model.infobn_sem.gssample.detach().cpu().numpy())
            self.evalmem[6].append(model.model.infobn_sem.attentionmat.detach().cpu().numpy())

        elif self.mem_eval_mode == "auto_word_multihead":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent.cpu().data.numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            # cooprer
            self.evalmem[2].append(torch.exp(model.model.sfms.context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.sfms.gssample.detach().cpu().numpy())
            # trainer
            self.evalmem[4].append(torch.exp(model.model.infobn_sem.context.detach().cpu()).numpy())
            self.evalmem[5].append(model.model.infobn_sem.gssample.detach().cpu().numpy())

        elif self.mem_eval_mode == "senti":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent.cpu().data.numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            # cooprer POS
            self.evalmem[2].append(torch.exp(model.model.seq1_coop.seq2_train.infobnl[0].context.detach().cpu()).numpy())
            self.evalmem[3].append(model.model.seq1_coop.seq2_train.infobnl[0].gssample.detach().cpu().numpy())
            # trainer Sem
            self.evalmem[4].append(torch.exp(model.model.seq1_coop.seq2_train.infobnl[1].context.detach().cpu()).numpy())
            self.evalmem[5].append(model.model.seq1_coop.seq2_train.infobnl[1].gssample.detach().cpu().numpy())
            # Predected POS label
            pt_POS_label= self.find_evalmem_recursize(model.model, "POS_label")
            self.evalmem[6].append(pt_POS_label.gssample.detach().cpu().numpy())
            pt_SEM_label = self.find_evalmem_recursize(model.model, "SEM_label")
            self.evalmem[7].append(pt_SEM_label.gssample.detach().cpu().numpy())

        elif self.mem_eval_mode == "senti_auto":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output[...,:5].detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent.cpu().data.numpy())
            self.evalmem[1].append(labels[...,0].detach().cpu().numpy())

            output = lsoftmax(model.model.output[..., 5:].detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[2].append(ent.cpu().data.numpy())
            self.evalmem[3].append(labels[..., 1].detach().cpu().numpy())

            # cooprer POS
            pt_POS_cooprer = self.find_evalmem_recursize(model.model,"POS_cooprer")
            self.evalmem[4].append(torch.exp(pt_POS_cooprer.context.detach().cpu()).numpy())
            self.evalmem[5].append(pt_POS_cooprer.gssample.detach().cpu().numpy())
            # trainer Sem
            pt_SEM_trainer = self.find_evalmem_recursize(model.model, "SEM_trainer")
            self.evalmem[6].append(torch.exp(pt_SEM_trainer.context.detach().cpu()).numpy())
            self.evalmem[7].append(pt_SEM_trainer.gssample.detach().cpu().numpy())
            # cooprer POS->Senti
            pt_Multitube = self.find_evalmem_recursize(model.model, "Multitube")
            self.evalmem[8].append(torch.exp(pt_Multitube.infobnl[0].context.detach().cpu()).numpy())
            self.evalmem[9].append(pt_Multitube.infobnl[0].gssample.detach().cpu().numpy())
            # cooprer Sem->Senti
            self.evalmem[10].append(torch.exp(pt_Multitube.infobnl[1].context.detach().cpu()).numpy())
            self.evalmem[11].append(pt_Multitube.infobnl[1].gssample.detach().cpu().numpy())
        elif self.mem_eval_mode == "senti_only":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(ent)
            self.evalmem[1].append(labels.detach())
            # cooprer POS
            pt_POS_cooprer = self.find_evalmem_recursize(model.model, "POS_cooprer")
            self.evalmem[2].append(torch.exp(pt_POS_cooprer.context.detach()))
            self.evalmem[3].append(pt_POS_cooprer.gssample.detach())
            # trainer Senti
            pt_Senti_trainer = self.find_evalmem_recursize(model.model, "Senti_trainer")
            self.evalmem[4].append(torch.exp(pt_Senti_trainer.context.detach()))
            self.evalmem[5].append(pt_Senti_trainer.gssample.detach())

        elif self.mem_eval_mode == "senti2":
            # Output
            lsoftmax = torch.nn.LogSoftmax(dim=-1)
            output = lsoftmax(model.model.output.detach())
            ent = cal_entropy(output.data, log_flag=True, byte_flag=False, torch_flag=True)
            self.evalmem[0].append(torch.mean(ent).cpu().numpy())
            self.evalmem[1].append(labels.detach().cpu().numpy())
            self.evalmem[2].append(torch.exp(model.model.mltitubeff.infobnl[0].context.detach()).cpu().numpy())
            self.evalmem[3].append(model.model.mltitubeff.infobnl[0].gssample.detach().cpu().numpy())
            # trainer Senti
            self.evalmem[4].append(torch.exp(model.model.mltitubeff.infobnl[1].context.detach()).cpu().numpy())
            self.evalmem[5].append(model.model.mltitubeff.infobnl[1].gssample.detach().cpu().numpy())
            # POS_label
            self.evalmem[6].append(model.model.sfms.gssample.detach().cpu().numpy())
            # ELSE_label
            self.evalmem[7].append(model.model.gssample.gssample.detach().cpu().numpy())

    def find_evalmem_recursize(self, model, label):
        # print(model.__class__)
        if hasattr(model, "evalmem_label"):
            # print(model.evalmem_label)
            if model.evalmem_label == label:
                return model
        if hasattr(model, "submodels"):
            for submodel in model.submodels:
                model_pt = self.find_evalmem_recursize(submodel,label)
                if model_pt is not None:
                    return model_pt
        return None

    def save_session(self,file_name):

        def _save_session(model, file_name):
            os.mkdir(file_name)
            save_model(model, os.path.join(file_name, file_name + ".model"))
            session_log = dict([])
            session_log["log"] = self.log
            session_log["train_hist"] = self.train_hist
            session_log["eval_hist"] = self.eval_hist
            save_data(session_log, os.path.join(file_name, file_name + ".sessionlog"))

        _save_session(self.pt_model.model, file_name)

    def load_session(self,file_name):

        load_model(self.pt_model.model, os.path.join(file_name,file_name+".model"),map_location="cpu")
        session_log = load_data(os.path.join(file_name,file_name+".sessionlog"))
        self.log = session_log["log"]
        self.train_hist = session_log["train_hist"]
        self.eval_hist = session_log["eval_hist"]

    def _profiler(self, iis, loss, lr_current,print_step=200):

        if iis % print_step == 0:
            if self.loss_exp_flag:
                print("Perlexity: ", iis, np.exp(loss.item()))
                self.log = self.log + "Perlexity: " + str(iis)+" "+ str(np.exp(loss.item())) + "\n"
            else:
                print("Loss: ", iis, loss.item())
                self.log = self.log + "Loss: " + str(iis) + " " + str(loss.item()) + "\n"
            print("Current learning rate:", lr_current)

        if self.loss_exp_flag:
            self.train_hist.append(np.exp(loss.item()))
        else:
            self.train_hist.append(loss.item())


    def _postscript(self):
        x = []
        for ii in range(len(self.train_hist)):
            x.append([ii, self.train_hist[ii]])
        x = np.array(x)
        if self.figure_plot:
            try:
                plt.plot(x[:, 0], x[:, 1])
                if self.loss_clip > 0:
                    # plt.ylim((-self.loss_clip, self.loss_clip))
                    if self.loss_exp_flag:
                        low_b=1.0
                    else:
                        low_b=0.0
                    plt.ylim((low_b, self.loss_clip))
                if self.save_fig is not None:
                    filename=self.save_fig+str(self.cuda_device)+".png"
                    print(filename)
                    plt.savefig(filename)
                    self.log = self.log + "Figure saved: " + filename + "\n"
                    plt.gcf().clear()
                else:
                    plt.show()
            except:
                pass

class PyTrain_Lite(object):
    """
    A class trying to wrap all possible training practice nicely and litely
    """

    # def __init__(self, dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, para=None):
    def __init__(self, dataset, lsize, rnn, batch=20, window=30, para=None):
        """

        :param dataset: {"data_train":data_train,"data_valid":data_valid,"data_test":data_test}
        :param lsize:
        :param rnn:
        :param batch:
        :param window:
        :param para:
        """
        assert type(dataset)==dict

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

        if self.dist_data_parallel:
            rnn.to(self.cuda_device)
            self.rnn = torch.nn.parallel.DistributedDataParallel(rnn, device_ids=[self.cuda_device], output_device=self.cuda_device,dim=self.dist_data_parallel_dim,
                                                                 find_unused_parameters=False)
        elif self.data_parallel is not None:
            self.rnn = torch.nn.DataParallel(rnn, device_ids=self.data_parallel,dim=self.data_parallel_dim)
        else:
            self.rnn = rnn

        self.lossf = None
        self.lossf_eval = None
        self.loss = None

        # profiler
        # self.prtstep = int(step / 20)
        self.train_hist = []

        # optimizer
        # if self.optimizer_label=="adam":
        #     print("Using adam optimizer")
        #     self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=learning_rate, weight_decay=0.0)
        # else:
        #     print("Using SGD optimizer")
        #     self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=learning_rate)

        # CUDA
        if self.cuda_flag:
            self.gpuavail = torch.cuda.is_available()
            self.device = torch.device(self.cuda_device if self.gpuavail else "cpu")
            self.rnn=self.rnn.to(self.device)
        else:
            self.gpuavail = False
            self.device = torch.device("cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print("PyTrain Init, Using device:",self.device)

        currentDT = datetime.datetime.now()
        self.log="Time of creation: "+str(currentDT)+"\n"

        # Evaluation memory
        self.evalmem = None
        self.data_col_mem = None
        self.training_data_mem=None

    def para(self,para):
        self.save_fig = para.get("save_fig", None)
        self.cuda_flag = para.get("cuda_flag", True)
        self.pt_emb = para.get("pt_emb", None)
        self.supervise_mode = para.get("supervise_mode", False)
        self.coop = para.get("coop", None)
        self.coopseq = para.get("coopseq", None)
        self.invec_noise = para.get("invec_noise", 0.0)
        self.pre_training = para.get("pre_training", False)
        self.loss_clip = para.get("loss_clip", 20.0)
        self.digit_input = para.get("digit_input", True)
        self.two_step_training = para.get("two_step_training", False)
        self.context_total = para.get("context_total", 1)
        self.context_switch_step = para.get("context_switch_step", 10)
        self.reg_lamda = para.get("reg_lamda", 0.0)
        self.optimizer_label = para.get("optimizer_label", "adam")
        self.length_sorted = False
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.figure_plot = para.get("figure_plot", True)
        self.custom_interface = para.get("custom_interface", None)
        self.loss_exp_flag = para.get("loss_exp_flag", True)
        self.specialized_digit_eval = para.get("specialized_digit_eval", None)
        self.mem_limit = para.get("mem_limit", 1e12)
        self.lr_scheduler_flag = para.get("lr_scheduler_flag", True)
        self.sch_factor = para.get("sch_factor", 0.2)
        self.sch_patience = para.get("sch_patience", 2)
        self.sch_threshold = para.get("sch_threshold", 0.01)
        self.sch_cooldown = para.get("sch_cooldown", 2)
        self.beta_warmup = para.get("beta_warmup", 0.0) # measured in schedule
        self.data_parallel = para.get("data_parallel", None) # data parallel switch
        self.data_parallel_dim = para.get("data_parallel_dim", 1)  # data parallel switch
        self.dist_data_parallel = para.get("dist_data_parallel", False)  # distributed data parallel switch
        self.dist_data_parallel_dim = para.get("dist_data_parallel_dim", 1)  # distributed data parallel dimention
        self.scale_factor=para.get("scale_factor", 1.0)
        self.output_cluster=para.get("output_cluster", None)

    def run_training(self,epoch=2,step_per_epoch=2000,lr=1e-3,optimizer_label=None,print_step=200):

        self.rnn.train()

        # if step is not None:
        #     self.step=step
        #     self.prtstep=int(step/print_step)
        self.prtstep = print_step

        if lr is not None:
            if optimizer_label is None:
                optimizer_label=self.optimizer_label
            if optimizer_label == "adam":
                print("Using adam optimizer")
                self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr, weight_decay=0.0)
            elif optimizer_label == "adamw":
                self.optimizer = torch.optim.AdamW(self.rnn.parameters(), lr=lr, weight_decay=0.01)
            else:
                print("Using SGD optimizer")
                self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=lr)

        current_lr=lr

        if self.lr_scheduler_flag:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.sch_factor,
                                                                   patience=self.sch_patience, verbose=True,
                                                                   threshold=self.sch_threshold,
                                                                   cooldown=self.sch_cooldown)

        startt = time.time()

        warmup_notprintyet=True

        if self.dist_data_parallel:
            pt_rnn = self.rnn.module
        elif self.data_parallel is None:
            pt_rnn = self.rnn
        else:
            pt_rnn = self.rnn.module

        for ii_epoch in range(epoch):
            self.rnn.train()
            for iis in range(step_per_epoch):
                if self.gpuavail:
                    hidden = pt_rnn.initHidden_cuda(self.device, self.batch)
                else:
                    hidden = pt_rnn.initHidden(self.batch)

                x, label, loss_data = self.get_data(self.dataset["data_train"])

                ii_tot = iis + ii_epoch * step_per_epoch
                cstep=ii_tot / (epoch*step_per_epoch)
                outputl, hidden = self.rnn(x, hidden, schedule=cstep)
                # else:
                #     outputl=None
                #     x, label, _ = self.get_data()
                #     for iiw in range(self.window):
                #         output, hidden = self.rnn(x[iiw,:,:], hidden, schedule=iis / self.step)
                #         if outputl is None:
                #             outputl = output.view(1, self.batch, self.lsize_out)
                #         else:
                #             outputl = torch.cat((outputl.view(-1, self.batch, self.lsize_out), output.view(1, self.batch, self.lsize_out)), dim=0)
                self.loss = self.lossf(outputl, label, pt_rnn, ii_tot/(epoch*step_per_epoch), loss_data=loss_data)
                self._profiler(ii_tot, self.loss)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self.while_training(ii_tot)
            if self.lr_scheduler_flag:
                midt = time.time()
                print("Time used till now:", midt - startt)
                print("Validation of epoch ", ii_epoch, ":")
                loss_eval=self.do_eval(data_pt="data_valid", schedule=cstep)
                if cstep>self.beta_warmup:
                    if warmup_notprintyet:
                        print("Warm up finished!")
                        warmup_notprintyet=False
                    scheduler.step(loss_eval)
                    temp_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr != temp_lr:
                        self.log = self.log +"Learning rate change detected: "+str(current_lr)+" -> "+str(temp_lr)+ "\n"
                        current_lr=temp_lr

        endt = time.time()
        print("Time used in training:", endt - startt)
        self.log = self.log + "Time used in training: " + str(endt - startt) + "\n"
        self._postscript()

    def eval_mem(self, x, label, output, rnn):
        pass

    def do_eval(self,step_eval=600,schedule=1.0,batch=None,posi_ctrl=None,allordermode=False,eval_mem_flag=False,data_pt="data_valid",print_step=None,rstartv_l=None):
        """

        :param step_eval:
        :param schedule:
        :param posi_ctrl:
        :param allordermode:
        :param eval_mem_flag:
        :param data_pt: "data_valid" or "data_test"
        :return:
        """
        if self.dist_data_parallel:
            pt_rnn = self.rnn.module
        elif self.data_parallel is None:
            pt_rnn = self.rnn
        else:
            pt_rnn = self.rnn.module

        if batch is None:
            batch=self.batch

        startt = time.time()
        pt_rnn.eval()
        if self.gpuavail:
            pt_rnn.to(self.device)
        perpl=[]
        if allordermode:
            dataset_length=len(self.dataset[data_pt]["dataset"])
            step_eval=int(dataset_length/batch/self.window)
            print("All ordermode step eval:",step_eval)
            print("Data length:",dataset_length)
        for iis in range(step_eval):
            if print_step is not None:
                if iis%print_step==0:
                    print("Evaluation step ",iis)
            if self.gpuavail:
                hidden = pt_rnn.initHidden_cuda(self.device, batch)
            else:
                hidden = pt_rnn.initHidden(batch)
            if allordermode:
                rstartv=iis*batch*self.window+np.linspace(0,batch-1,batch)*self.window
                x, label, loss_data = self.get_data(self.dataset[data_pt],rstartv=rstartv,batch=batch)
            elif rstartv_l is not None:
                assert len(rstartv_l)==step_eval
                rstartv=rstartv_l[iis]
                x, label, loss_data = self.get_data(self.dataset[data_pt], rstartv=rstartv, batch=batch)
            else:
                x, label, loss_data = self.get_data(self.dataset[data_pt],batch=batch)

            output, hidden = pt_rnn(x, hidden, schedule=schedule)
            if posi_ctrl is None:
                loss = self.lossf_eval(output, label, pt_rnn, None, loss_data=loss_data)
            else:
                loss = self.lossf_eval(output[posi_ctrl,:,:].view(1,output.shape[1],output.shape[2]), label[:,posi_ctrl].view(label.shape[0],1), pt_rnn, None)
            if eval_mem_flag:
                self.eval_mem(x, label, output, pt_rnn)

            # else:
            #     outputl=None
            #     hiddenl=None
            #     for iiw in range(self.window):
            #         output, hidden = self.rnn(x[iiw, :, :], hidden, schedule=schedule)
            #         if outputl is None:
            #             outputl = output.view(1, self.batch, self.lsize_out)
            #             hiddenl = hidden.view(1, self.batch, self.rnn.hidden_size)
            #         else:
            #             outputl = torch.cat(
            #                 (outputl.view(-1, self.batch, self.lsize_out), output.view(1, self.batch, self.lsize_out)),
            #                 dim=0)
            #             hiddenl = torch.cat(
            #                 (hiddenl.view(-1, self.batch, self.rnn.hidden_size), hidden.view(1, self.batch, self.rnn.hidden_size)),
            #                 dim=0)
            #     loss = self.lossf_eval(outputl, label, self.rnn, iis)
            #     self.eval_mem(outputl, hiddenl)

            if type(loss) is not tuple:
                perpl.append(loss.cpu().item())
            else:
                subperpl=[]
                for item in loss:
                    subperpl.append(item.cpu().item())
                perpl.append(subperpl)
        # print("Evaluation Perplexity: ", np.mean(np.array(perpl)))
        if self.loss_exp_flag:
            perp=np.exp(np.mean(np.array(perpl),axis=0))
            print("Evaluation Perplexity: ", perp)
        else:
            perp = np.mean(np.array(perpl),axis=0)
            print("Evaluation Loss: ", perp)
        endt = time.time()
        print("Time used in evaluation:", endt - startt)
        if self.gpuavail:
            torch.cuda.empty_cache()

        currentDT = datetime.datetime.now()
        self.log = self.log + "Time at evaluation: " + str(currentDT) + "\n"
        self.log = self.log + "Evaluation Perplexity: "+ str(perp) + "\n"
        self.log = self.log + "Time used in evaluation:"+ str(endt - startt) + "\n"

        return perp

    def do_test(self,step_test=600,schedule=1.0,data_pt="data_valid",posi_ctrl=None, mask_comp=False):
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
            x, label, loss_data = self.get_data(self.dataset[data_pt])
            output, hidden = self.rnn(x, hidden, schedule=schedule)
            output = output.permute(1, 2, 0)
            _,predicted = torch.max(output,1)
            if posi_ctrl is not None:
                total += label.size(0)
                correct += (predicted[:, posi_ctrl] == label[:, posi_ctrl]).sum().item()
            elif mask_comp:
                loss_data = loss_data.permute(1, 0)
                total += torch.sum(1.0-loss_data.cpu())
                crrect_mat= (predicted == label).cpu().float()
                crrect_mat_masked=crrect_mat*(1.0-loss_data.cpu())
                correct += crrect_mat_masked.sum().item()
            else:
                total += label.size(0) * label.size(1)
                correct += (predicted == label).sum().item()

            correct_ratel.append(correct / total)
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

    def do_inference(self,batch=None,data_pt="data_train",print_step=None):
        """
        Use mode and run inference, default to allordermode
        :return: [(data,label),(data,label),...]
        """
        if self.dist_data_parallel:
            pt_rnn = self.rnn.module
        elif self.data_parallel is None:
            pt_rnn = self.rnn
        else:
            pt_rnn = self.rnn.module

        if batch is None:
            batch=self.batch

        dataset_length=len(self.dataset[data_pt]["dataset"])
        step_inf=int(dataset_length/batch/self.window)
        print("All ordermode step inference:",step_inf)
        print("Data length:",dataset_length)

        res_infer_mat=[]
        for iis in range(step_inf):
            if iis%100==0:
                print("Step: ",iis)
            if print_step is not None:
                if iis%print_step==0:
                    print("Evaluation step ",iis)
            if self.gpuavail:
                hidden = pt_rnn.initHidden_cuda(self.device, batch)
            else:
                hidden = pt_rnn.initHidden(batch)
            rstartv = iis * batch * self.window + np.linspace(0, batch - 1, batch) * self.window
            x, label, _ = self.get_data(self.dataset[data_pt], rstartv=rstartv, batch=batch)

            output, hidden = pt_rnn(x, hidden, schedule=1.0)
            output=output.detach().cpu().numpy()

            for iib in range(batch):
                ptdata = self.dataset[data_pt]["dataset"][int(rstartv[iib]):int(rstartv[iib]) + self.window]
                outputl=np.argmax(output[:,iib,:],axis=-1)
                res_infer_mat.append(list(zip(ptdata, outputl)))

        res_infer_mat_f = []
        for iteml in res_infer_mat:
            for item in iteml:
                res_infer_mat_f.append(item)

        return res_infer_mat_f


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
            self.example_data_collection(x, output, hidden, label)
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

        if iis % self.prtstep == 0:
            if self.loss_exp_flag:
                print("Perlexity: ", iis, np.exp(loss.item()))
                self.log = self.log + "Perlexity: " + str(iis)+" "+ str(np.exp(loss.item())) + "\n"
            else:
                print("Loss: ", iis, loss.item())
                self.log = self.log + "Loss: " + str(iis) + " " + str(loss.item()) + "\n"

        if self.loss_exp_flag:
            self.train_hist.append(np.exp(loss.item()))
        else:
            self.train_hist.append(loss.item())
        # elif mode=="valid":
        #     if self.loss_exp_flag:
        #         print("Validation Perlexity: ", iis, np.exp(loss.item()))
        #         self.log = self.log + "Validation Perlexity: " + str(iis) + " " + str(np.exp(loss.item())) + "\n"
        #     else:
        #         print("Validation Loss: ", iis, loss.item())
        #         self.log = self.log + "Validation Loss: " + str(iis) + " " + str(loss.item()) + "\n"

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
                if self.loss_clip > 0:
                    # plt.ylim((-self.loss_clip, self.loss_clip))
                    if self.loss_exp_flag:
                        low_b=1.0
                    else:
                        low_b=0.0
                    plt.ylim((low_b, self.loss_clip))
                if self.save_fig is not None:
                    filename=self.save_fig+str(self.cuda_device)+".png"
                    print(filename)
                    plt.savefig(filename)
                    self.log = self.log + "Figure saved: " + filename + "\n"
                    plt.gcf().clear()
                else:
                    plt.show()
            except:
                pass

class PyTrain_Custom(PyTrain_Lite):
    """
    A pytrain custom object aiding PyTrain_Lite
    """
    def __init__(self, dataset, lsize, rnn, interface_para, batch=20, window=30, para=None):
        """
        PyTrain custom
        :param para:
        """
        super(self.__class__, self).__init__(dataset, lsize, rnn, batch=batch, window=window, para=para)

        self.data_init = None
        self.databp = None
        self.databp_lab = None
        self.dataset_length = None

        self.init_interface(interface_para)

        # context controller
        self.context_id=0

        # Last step
        self.data(dataset)
        # self.context_switch(0)

    # def context_switch(self,context_id):
    #
    #     if context_id==0:
    #         self.lossf = getattr(self, self.custom_interface["lossf"])
    #         # Interface
    #         # self.lossf = self.KNWLoss
    #         # self.lossf = self.KNWLoss_GateReg_hgate
    #         # self.lossf = self.KNWLoss_WeightReg
    #         # self.lossf = self.KNWLoss_WeightReg_GRU
    #         self.lossf_eval = getattr(self, self.custom_interface["lossf_eval"])
    #         # self.lossf_eval = self.KNWLoss
    #         self.get_data = getattr(self, self.custom_interface["get_data"])
    #         # self.get_data = self.get_data_continous
    #         # self.get_data = self.get_data_sent_sup
    #         # self.get_data = self.get_data_seq2seq not working
    #     # elif context_id==1:
    #     #     # Interface
    #     #     # self.lossf = self.KNWLoss
    #     #     self.lossf = self.KNWLoss_GateReg_hgate
    #     #     # self.lossf = self.KNWLoss_WeightReg
    #     #     # self.lossf = self.KNWLoss_WeightReg_GRU
    #     #     self.lossf_eval = self.KNWLoss
    #     #     self.get_data = self.get_data_sent_sup
    #     #     # self.get_data = self.get_data_seq2seq
    #     else:
    #         raise Exception("Unknown context")

    def init_interface(self,interface_para):

        pfver = interface_para.get("version", "default")
        if interface_para["class"] == "PyTrain_Interface_continous":
            self.interface = PyTrain_Interface_continous(version=pfver)
        elif interface_para["class"] == "PyTrain_Interface_W2V":
            self.interface = PyTrain_Interface_W2V(interface_para.get("prior",None),interface_para.get("threshold",1e-5),version=pfver)
        elif interface_para["class"] == "PyTrain_Interface_sup":
            self.interface = PyTrain_Interface_sup(version=pfver)
        elif interface_para["class"] == "PyTrain_Interface_advsup":
            self.interface = PyTrain_Interface_advsup()
        else:
            raise Exception("Interface class unknown")

        self.interface.pt = self

        self.get_data = self.interface.get_data
        self.lossf = self.interface.lossf
        self.lossf_eval = self.interface.lossf_eval
        self.eval_mem = self.interface.eval_mem
        self.while_training = self.interface.while_training
        self.example_data_collection = self.interface.example_data_collection

    def data(self,dataset):
        """
        Swap dataset
        :param dataset:
        :return:
        """

        self.dataset = dataset

        # limit = self.mem_limit
        # if type(dataset) is list:
        #     print("Data symbol size: ",len(self.dataset) * self.lsize_in)
        # elif type(dataset) is dict:
        #     print("Data symbol size: ", len(self.dataset["dataset"]) * self.lsize_in)
        # if len(self.dataset) * self.lsize_in < limit:
        #     self.data_init = True
        #     self.interface.init_data()
        # else:
        #     self.data_init = False
        #     print("Warning, large dataset, not pre-processed.")

        if (type(dataset["data_train"]) is dict) != self.supervise_mode:
            raise Exception("Supervise mode Error.")

    # def eval_mem(self, x, label, rnn):
    #     """
    #     Archiving date
    #     :param output:
    #     :param hidden:
    #     :return:
    #     """
    #
    #     # return self.custom_eval_mem_tdff(x, label, rnn)
    #     # return self.custom_eval_mem_attn(x, label, rnn)
    #     return self.interface.evalmem(x, label, rnn)


    # def while_training(self,iis):
    #     # self.context_monitor(iis)
    #     return self.interface.while_training(iis)

    def context_monitor(self,iis):
        """Monitor progress"""
        pass
        # self.context_id=int(iis/100)%self.context_total
        # self.context_switch(self.context_id)

    # def _build_databp(self,inlabs):
    #     """
    #     Build databp from inlab (when dataset too large)
    #     :param inlab:
    #     :return:
    #     """
    #     if self.digit_input and self.id_2_vec is None and not self.data_init:
    #         datab=np.zeros((len(inlabs),self.lsize_in))
    #         for ii_b in range(len(inlabs)):
    #             datab[ii_b,inlabs[ii_b]]=1.0
    #         databp = torch.from_numpy(np.array(datab))
    #         databp = databp.type(torch.FloatTensor)
    #     elif self.digit_input and self.id_2_vec is not None and not self.data_init:
    #         datab = np.zeros((len(inlabs), self.lsize_in))
    #         for ii_ind in range(len(inlabs)):
    #             datab[ii_ind,:]=self.id_2_vec[inlabs[ii_ind]]
    #         databp = torch.from_numpy(np.array(datab))
    #         databp = databp.type(torch.FloatTensor)
    #     else:
    #         raise Exception("Not Implemented")
    #     return databp


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

    # def custom_get_data_pos_auto(self):
    #     """
    #     Customed data get subroutine for both pos tag and self tag
    #     :return:
    #     """
    #     assert self.supervise_mode
    #     label=np.array(self.dataset["label"])
    #     dataset = np.array(self.dataset["dataset"])
    #     rstartv = np.floor(np.random.rand(self.batch) * (len(dataset) - self.window - 1))
    #
    #     autol = np.zeros((self.batch, self.window))
    #     labell = np.zeros((self.batch, self.window))
    #     for iib in range(self.batch):
    #         labell[iib, :] = label[int(rstartv[iib]):int(rstartv[iib]) + self.window]
    #         autol[iib, :] = dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window]
    #     poslab = torch.from_numpy(labell)
    #     poslab = poslab.type(torch.LongTensor)
    #     autolab = torch.from_numpy(autol)
    #     autolab = autolab.type(torch.LongTensor)
    #
    #     vec1m = torch.zeros(self.window, self.batch, self.lsize_in)
    #     for iib in range(self.batch):
    #         if self.data_init:
    #             vec1 = self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.window, :]
    #         else:
    #             vec1 = self.__build_databp(dataset[int(rstartv[iib]):int(rstartv[iib]) + self.window])
    #         vec1m[:, iib, :] = vec1
    #     x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)
    #
    #     if self.gpuavail:
    #         x = x.to(self.device)
    #         poslab = poslab.to(self.device)
    #         autolab = autolab.to(self.device)
    #
    #     return x, [poslab,autolab]


    # def example_data_collection(self, x, output, hidden, label,items="all"):
    #
    #     return self.interface.example_data_collection(x, output, hidden, label)

class PyTrain_Interface_Default(object):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):

        self.pt=None
        self.test_print_interface=False

    def print_interface(self):
        """
        Test print interface
        :return:
        """
        self.test_print_interface=True
        # print("init_data: ")
        # self.init_data()
        print("get_data: ")
        self.get_data()
        print("lossf: ")
        self.lossf(None,None)
        print("lossf_eval: ")
        self.lossf_eval(None,None)
        print("eval_mem: ")
        self.eval_mem(None,None,None,None)
        print("while_training: ")
        self.while_training(None)
        print("example_data_collection: ")
        self.example_data_collection(None,None,None,None)
        self.test_print_interface = False

    # def init_data(self,*args,**kwargs):
    #     """
    #     Data initialization
    #
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     return self.init_data_sent_sup()

    def get_data(self, batch=None, rstartv=None):
        return self.get_data_sent_sup(batch=batch, rstartv=rstartv)

    def lossf(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        return self.KNWLoss(outputl, outlab, model=model, cstep=cstep, loss_data=loss_data)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        return self.KNWLoss(outputl, outlab, model=model, cstep=cstep, loss_data=loss_data)


    def eval_mem(self, x, label, output, rnn):
        """
        Defalt evalmem
        called in do_eval
        usage self.eval_mem(x, label, self.rnn)
        :param kwargs:
        :param args:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: default to nothing")
            return True

    def while_training(self,*args,**kwargs):
        """
        Default while_training function
        called while training
        :param kwargs:
        :param args:
        :return:
        """
        if self.test_print_interface:
            print("While training interface: default to nothing")
            return True

    def example_data_collection(self,*args,**kwargs):
        """
        Default example_data_collection function
        called by plot_example
        :param kwargs:
        :param args:
        :return:
        """
        if self.test_print_interface:
            print("Example data collection interface: default to nothing")
            return True

    def init_data_sent_sup(self,limit=1e9):
        if self.test_print_interface:
            print("Init data interface: init_data_sent_sup")
            return True

        assert self.pt.supervise_mode
        assert type(self.pt.dataset["dataset"][0]) == list # we assume sentence structure
        assert self.pt.digit_input
        assert self.pt.id_2_vec is None # No embedding, one-hot representation

        self.pt.dataset_length=len(self.pt.dataset["dataset"])
        print("Dataset length ",self.pt.dataset_length)

        if len(self.pt.dataset)*self.pt.lsize_in<limit:
            self.pt.databp=[]
            for sent in self.pt.dataset["dataset"]:
                datab_sent=[]
                for data in sent:
                    datavec = np.zeros(self.pt.lsize_in)
                    datavec[data] = 1.0
                    datab_sent.append(datavec)
                datab_sent = torch.from_numpy(np.array(datab_sent))
                datab_sent = datab_sent.type(torch.FloatTensor)
                self.pt.databp.append(datab_sent)
            self.pt.data_init = True
        else:
            print("Warning, large dataset, not pre-processed.")
            self.pt.databp=None
            self.pt.data_init=False

    def get_data_sent_sup(self, batch=None, rstartv=None):

        if self.test_print_interface:
            print("Get data interface: get_data_sent_sup")
            return True

        assert self.pt.supervise_mode
        assert type(self.pt.dataset["dataset"][0]) == list  # we assume sentence structure
        assert self.pt.data_init

        if batch is None:
            batch=self.pt.batch

        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset["dataset"]) - 1))
        else:
            assert len(rstartv)==batch

        qlen = len(self.pt.dataset["dataset"][0])
        anslen=len(self.pt.dataset["label"][0])
        xl = np.zeros((batch, qlen))
        outl = np.zeros((batch, anslen))
        for iib in range(batch):
            xl[iib, :] = np.array(self.pt.dataset["dataset"][int(rstartv[iib])])
            outl[iib, :] = np.array(self.pt.dataset["label"][int(rstartv[iib])])
        inlab = torch.from_numpy(xl)
        inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(outl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(self.pt.window, batch, self.pt.lsize_in)
        for iib in range(batch):
            vec1=self.pt.databp[int(rstartv[iib])]
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)

        return x, outlab, inlab


    def KNWLoss(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: default KNWLoss")
            return True

        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

class PyTrain_Interface_common(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self):
        super(self.__class__, self).__init__()

    def _init_data_all(self,limit=1e9):
        if len(self.pt.dataset)*self.pt.lsize_in<limit:
            datab = []
            if self.pt.digit_input:
                if self.pt.id_2_vec is None: # No embedding, one-hot representation
                    self.PAD_VEC=np.zeros(self.pt.lsize_in, dtype=np.float32)
                    self.PAD_VEC[self.pt.SYM_PAD] = 1.0
            if type(self.pt.dataset[0]) != list:
                if self.pt.digit_input:
                    if self.pt.id_2_vec is None:  # No embedding, one-hot representation
                        for data in self.pt.dataset:
                            datavec = np.zeros(self.pt.lsize_in)
                            datavec[data] = 1.0
                            datab.append(datavec)
                    else:
                        for data in self.pt.dataset:
                            datavec = np.array(self.pt.id_2_vec[data])
                            datab.append(datavec)
                else:  # if not digit input, raw data_set is used
                    datab = self.pt.dataset
                self.databp = torch.from_numpy(np.array(datab))
                self.databp = self.databp.type(torch.FloatTensor)
            else: # we assume sentence structure
                self.databp=[]
                if self.pt.digit_input:
                    if self.pt.id_2_vec is None:  # No embedding, one-hot representation
                        for sent in self.pt.dataset:
                            datab_sent=[]
                            for data in sent:
                                datavec = np.zeros(self.pt.lsize_in)
                                datavec[data] = 1.0
                                datab_sent.append(datavec)
                            datab_sent = torch.from_numpy(np.array(datab_sent))
                            datab_sent = datab_sent.type(torch.FloatTensor)
                            self.databp.append(datab_sent)
                    else:
                        for sent in self.pt.dataset:
                            datab_sent = []
                            for data in sent:
                                datavec = np.array(self.pt.id_2_vec[data])
                                datab_sent.append(datavec)
                            datab_sent=torch.from_numpy(np.array(datab_sent))
                            datab_sent = datab_sent.type(torch.FloatTensor)
                            self.databp.append(datab_sent)
                else:  # if not digit input, raw data_set is used
                    for sent in self.pt.dataset:
                        datab_sent = torch.from_numpy(np.array(sent))
                        datab_sent = datab_sent.type(torch.FloatTensor)
                        self.databp.append(datab_sent)
            self.data_init = True
        else:
            print("Warning, large dataset, not pre-processed.")
            self.databp=None
            self.data_init=False

    def _init_data_sup(self):

        assert self.pt.digit_input
        assert self.pt.id_2_vec is not None
        assert self.pt.supervise_mode

        dataset=self.pt.dataset["dataset"]
        datab=np.zeros((len(dataset),len(self.pt.id_2_vec[0])))
        for ii in range(len(dataset)):
            datavec = np.array(self.pt.id_2_vec[dataset[ii]])
            datab[ii]=datavec
        self.pt.databp = torch.from_numpy(np.array(datab))
        self.pt.databp = self.pt.databp.type(torch.FloatTensor)
        self.pt.data_init = True

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
        return loss1+self.pt.reg_lamda*torch.mean(loss_gate)#+self.reg_lamda*wnorm1  # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_WeightReg(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in", "W_out", "W_hd"]
        wnorm1=0
        for namep in allname:
                wattr = getattr(self.pt.rnn.gru_enc, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
                try:
                    wattr = getattr(self.pt.rnn.gru_dec, namep)
                    wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
                except:
                    pass
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/(len(allname))
        return loss1+self.pt.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_ExpertReg(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["Wiz", "Whz", "Win", "Whn"]
        wnorm1=0
        for namep in allname:
                wattr = getattr(self.pt.rnn.gru_enc, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
                try:
                    wattr = getattr(self.pt.rnn.gru_dec, namep)
                    wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
                except:
                    pass
        wattr = getattr(self.pt.rnn, "h2o")
        logith2o = wattr.weight# + wattr.bias.view(-1)
        pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        # wnorm1=wnorm1/(2*len(allname))
        return loss1+ self.pt.reg_lamda * lossh2o # + self.reg_lamda *wnorm1 # + 0.001 * l1_reg #  + 0.01 * l1_reg



    def KNWLoss_WeightReg_GRU(self, outputl, outlab, model=None, cstep=None):
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        # loss1 = self.lossc(outputl, outlab)
        loss1 = lossc(outputl, outlab)
        wnorm1=0
        if self.pt.rnn.weight_dropout>0:
            weight_ih=self.pt.rnn.gru.module.weight_ih_l0
            weight_hh = self.pt.rnn.gru.module.weight_hh_l0
        else:
            weight_ih = self.pt.rnn.gru.weight_ih_l0
            weight_hh = self.pt.rnn.gru.weight_hh_l0
        wnorm1 = wnorm1 + torch.mean(torch.abs(weight_ih)) + torch.mean(torch.abs(weight_hh))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/6
        return loss1+self.pt.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

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
        self.pt.rnn.eval()
        perpl=[[],[],[],[]]
        for iis in range(step_eval):
            if self.pt.gpuavail:
                hidden = self.pt.rnn.initHidden_cuda(self.pt.device, self.pt.batch)
            else:
                hidden = self.pt.rnn.initHidden(self.pt.batch)
            x, label = self.pt.get_data()
            output, hidden = self.rnn(x, hidden, schedule=1.0)
            loss11, loss12, loss21, loss22 = self.custom_loss_pos_auto_eval(output, label, self.pt.rnn, iis)
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
        if self.pt.gpuavail:
            torch.cuda.empty_cache()

class PyTrain_Interface_continous(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self,version=0):
        super(self.__class__, self).__init__()
        self.version = version
        self.allname=["h2c","c2o"]

    def init_data(self,*args,**kwargs):
        return self.init_data_continous()

    def get_data(self, dataset, batch=None, rstartv=None):
        if batch is None:
            batch=self.pt.batch
        data_shape=[self.pt.window, batch, self.pt.lsize_in]
        return MyDataFun.get_data_continous(dataset, data_shape, self.pt.pt_emb, self.pt, rstartv=None, shift=False)

    def lossf(self, outputl, outlab, model=None, cstep=None):
        if self.version == "vib":
            # return self.KNWLoss_EnergyReg(outputl, outlab, model=model, cstep=cstep)
            # return self.KNWLoss_VIB(outputl, outlab, model=model, cstep=cstep)
            return MyLossFun.KNWLoss_VIB(outputl, outlab, self.pt.reg_lamda, model)
        elif self.version == "ereg":
            # return self.KNWLoss_EnergyReg(outputl, outlab, model=model, cstep=cstep)
            return MyLossFun.KNWLoss_EnergyReg(outputl, outlab, self.pt.reg_lamda, model, balance_para=10)
        elif self.version == "mask_vib_coop":
            return MyLossFun.KNWLoss_withmask_VIB(outputl, outlab, self.pt.reg_lamda, model.trainer)
        else:
            return MyLossFun.KNWLoss(outputl, outlab)


    def lossf_eval(self, outputl, outlab, model=None, cstep=None):
        # return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        # if self.version in [0]:
        #     return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # elif self.version==1:
        #     return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        if self.version == "mask_vib_coop":
            # return MyLossFun.KNWLoss_withmask(outputl, outlab, model.trainer)
            return MyLossFun.KNWLoss(outputl, outlab)
        else:
            return MyLossFun.KNWLoss(outputl, outlab)

    def eval_mem(self, x, label, output, rnn):
        return self.custom_eval_mem_context(x, label, output, rnn)
        # pass

    def custom_eval_mem_context(self, x, label, output, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: custom_eval_mem_sup")
            return True


        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(3)]  # x,label,hd_middle

        try:
            self.pt.evalmem[0].append(x.cpu().data.numpy())
            self.pt.evalmem[1].append(label.cpu().data.numpy())
            self.pt.evalmem[2].append(rnn.trainer.context.cpu().data.numpy())
        except:
            # print("eval_mem failed")
            pass

    def eval_mem_Full_eval(self, x, label, output, rnn):
        if self.test_print_interface:
            print("eval_mem interface: eval_mem_Full_eval")
            return True
        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(1+len(output))]  # x,label,hd_middle

        for ii in range(len(output)):
            self.pt.evalmem[ii].append(output[ii].cpu().detach().data)
        self.pt.evalmem[-1].append(label.cpu().detach().data)

    def KNWLoss_HC(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        outputl1=outputl[0]
        outputl1=outputl1.permute(1, 2, 0)
        loss1 = lossc(outputl1, outlab)

        outputl2 = outputl[1]
        outputl2 = outputl2.permute(1, 2, 0)
        loss2 = lossc(outputl2, outlab)

        # wnorm1 = 0
        # for namep in self.allname:
        #     wattr = getattr(self.pt.rnn, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # wnorm1 = wnorm1 / len(self.allname)

        return loss1+loss2

    def KNWLoss_HC_Full(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        loss=0
        for output_item in outputl:
            loss=loss+lossc(output_item.permute(1, 2, 0), outlab)

        return loss

    def KNWLoss_HC_Int(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        loss=0
        # print("In loss")
        # print(outputl.shape,outlab.shape)
        for output_item in outputl:
            # print(output_item.shape)
            output_item = output_item.view(self.pt.window, self.pt.batch, -1)
            # print(output_item.shape)
            loss=loss+lossc(output_item.permute(1, 2, 0), outlab)
        # print("Out loss")
        return loss

    def KNWLoss_HC_Int_eval(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        loss=0
        for output_item in outputl:
            output_item=output_item.view(self.pt.window,self.pt.batch,-1)
            loss=loss+lossc(output_item.permute(1, 2, 0), outlab)

        return loss

    def KNWLoss_HC_Full_eval(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        lossl = []
        for output_item in outputl:
            lossl.append(lossc(output_item.permute(1, 2, 0), outlab))

        return tuple(lossl)

    def KNWLoss_HC_eval(self, outputl, outlab, model=None, cstep=None):
        """
        Loss function for hierachical de-clustering
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_HC")
            return True

        lossc = torch.nn.CrossEntropyLoss()

        outputl1=outputl[0]
        outputl1=outputl1.permute(1, 2, 0)
        loss1 = lossc(outputl1, outlab)

        outputl2 = outputl[1]
        outputl2 = outputl2.permute(1, 2, 0)
        loss2 = lossc(outputl2, outlab)

        # wnorm1 = 0
        # for namep in self.allname:
        #     wattr = getattr(self.pt.rnn, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # wnorm1 = wnorm1 / len(self.allname)

        return (loss1,loss2)

    # def init_data_continous(self,limit=1e9):
    #
    #     if self.test_print_interface:
    #         print("init_data interface: init_data_continous")
    #         return True
    #
    #     assert self.pt.digit_input
    #     assert not self.pt.supervise_mode
    #
    #     datab = []
    #     for data in self.pt.dataset:
    #         if self.pt.id_2_vec is None:  # No embedding, one-hot representation
    #             datavec = np.zeros(self.pt.lsize_in)
    #             datavec[data] = 1.0
    #         else:
    #             datavec = np.array(self.pt.id_2_vec[data])
    #         datab.append(datavec)
    #     self.databp = torch.from_numpy(np.array(datab))
    #     self.databp = self.databp.type(torch.FloatTensor)

    def get_data_continous(self, batch=None, rstartv=None, shift=False):

        if batch is None:
            batch=self.pt.batch

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset) - self.pt.window - 1))
        yl = np.zeros((batch,self.pt.window))
        # xl = np.zeros((batch,self.pt.window))
        for iib in range(batch):
            # xl[iib,:]=self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
            if shift:
                yl[iib,:]=self.pt.dataset[int(rstartv[iib])+1:int(rstartv[iib]) + self.pt.window+1]
            else:
                yl[iib, :] = self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
        # inlab = torch.from_numpy(xl)
        # inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)
        # inlab = outlab

        vec1m = torch.zeros(self.pt.window, batch, self.pt.lsize_in)
        for iib in range(batch):
            # if self.pt.data_init:
            #     vec1=self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window, :]
            # else:
            # vec1=self.pt._build_databp(self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window])
            ptdata = self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
            vec1 = self.pt.pt_emb(torch.LongTensor(ptdata))
            # vec2 = self.databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.window + 1, :]
            vec1m[:,iib,:]=vec1
            # vec2m[:, iib, :] = vec2
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor) #
        # y = Variable(vec2m, requires_grad=True)

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)

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
        super(self.__class__, self).__init__()

    def custom_eval_mem_enc_dec(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.evalmem is None:
            self.evalmem = [[] for ii in range(8)]  # x,label,enc(ht,zt,nt),dec(ht,zt,nt)

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
        super(self.__class__, self).__init__()

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
        super(self.__class__, self).__init__()

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
    def __init__(self,sub_version=0):
        super(self.__class__, self).__init__()
        self.sub_version=sub_version

        self.allname = ["W_i2m", "W_m2h1", "W_h12h2", "W_h2o"]

        if self.sub_version==0:
            self._get_data = self.get_data_sent_sup
            self._lossf =self.KNWLoss_GateReg_TDFF_L1
            self._lossf_eval = self.KNWLoss
        elif self.sub_version==1:
            self._get_data=self.get_data_sent_KLsup
            self._lossf=self.KNWLoss_GateReg_TDFF_KL
            self._lossf_eval = self.KNWLoss_GateReg_TDFF_KL

        self.print_interface()

    def get_data(self, batch=None, rstartv=None):
        # return self.get_data_sent_KLsup( batch=batch, rstartv=rstartv)
        return self._get_data(batch=batch, rstartv=rstartv)


    def lossf(self, outputl, outlab, model=None, cstep=None):
        # return self.KNWLoss_GateReg_TDFF_L1(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        return self._lossf(outputl, outlab, model=model, cstep=cstep)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None):
        # return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        return self._lossf_eval(outputl, outlab, model=model, cstep=cstep)


    def eval_mem(self, x, label, rnn):
        """
        Defalt evalmem
        called in do_eval
        usage self.eval_mem(x, label, self.rnn)
        :param kwargs:
        :param args:
        :return:
        """

        return self.custom_eval_mem_tdff(x, label, rnn)

    def example_data_collection(self, x, output, hidden, label):
        """
        Default example_data_collection function
        called by plot_example
        :param kwargs:
        :param args:
        :return:
        """
        return self.custom_example_data_collection_tdff(x, output, hidden, label)

    def get_data_sent_KLsup(self, batch=None, rstartv=None):

        if self.test_print_interface:
            print("Get data interface: get_data_sent_KLsup")
            return True

        assert self.pt.supervise_mode
        assert type(self.pt.dataset["dataset"][0]) == list  # we assume sentence structure
        assert self.pt.data_init

        if batch is None:
            batch=self.pt.batch

        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset["dataset"]) - 1))
        else:
            assert len(rstartv)==batch

        # qlen = len(self.pt.dataset["dataset"][0])
        anslen=self.pt.dataset["anslen"]

        vec1m = torch.zeros(self.pt.window, batch, self.pt.lsize_in)
        for iib in range(batch):
            vec1=self.pt.databp[int(rstartv[iib])]
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        vec_outm = torch.zeros(anslen, batch, self.pt.lsize_in)
        for iib in range(batch):
            vec_outm[:, iib, :] = torch.from_numpy(self.pt.dataset["target"][int(rstartv[iib])])


        y = Variable(vec_outm, requires_grad=True).type(torch.FloatTensor)

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            x = x.to(self.pt.device)
            y = y.to(self.pt.device)

        return x, y, None

    def KNWLoss_GateReg_TDFF_KL(self, outputl, outlab, model=None, cstep=None, posi_ctrl=1):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_GateReg_TDFF_KL")
            return True

        if posi_ctrl is None:
            # loss1 = torch.nn.functional.kl_div(model.nsoftmax(outputl), model.nsoftmax(outlab))
            loss1 = torch.mean(cal_kldiv_torch(model.nsoftmax(outputl), model.nsoftmax(outlab)))
        else:
            kl1 = outputl[posi_ctrl, :, :]
            kl2 = outlab[posi_ctrl, :, :]

            loss1 = torch.mean(cal_kldiv_torch(model.nsoftmax(kl1), model.nsoftmax(kl2)))

            # print(kl1[0, :], kl2[0, :], loss1)
            # a = input()

        loss_gate = torch.mean(model.sigmoid(model.input_gate))

        # allname = ["W_in", "W_out","Whd0"]
        # wnorm1=0
        # for namep in allname:
        #         wattr = getattr(self.pt.rnn, namep)
        #         wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        # return loss1 + self.pt.reg_lamda*loss_gate # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

        return self.pt.reg_lamda * loss_gate + loss1

    def KNWLoss_WeightReg_TDFF(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_WeightReg_TDFF")
            return True

        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        wnorm1=0
        for namep in self.allname:
            wattr = getattr(self.pt.rnn, namep)
            wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        # for ii in range(self.rnn.num_layers):
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(self.rnn.hd_layer_stack[ii][1].weight))
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        wnorm1=wnorm1/len(self.allname)
        return loss1+self.pt.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_WeightEnergyReg_TDFF(self, outputl, outlab, model=None, cstep=None):

        if self.pt.test_print_interface:
            print("KNWLoss interface: KNWLoss_WeightEnergyReg_TDFF")
            return True

        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # allname = ["Wiz", "Whz", "Win", "Whn","Wir", "Whr"]
        allname = ["W_in", "W_out","Whd0"]
        wnorm1=0
        for namep in allname:
            wattr = getattr(self.pt.rnn, namep)
            wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
            # wnorm1 = wnorm1 + torch.mean(torch.mul(wattr.weight, wattr.weight))

        wnorm1 = wnorm1 / len(allname)

        energyloss=torch.mean(torch.mul(model.Wint,model.Wint))+torch.mean(torch.mul(model.hdt,model.hdt))

        # energyloss = torch.mean(torch.abs(model.Wint))+ torch.mean(torch.abs(model.hdt))

        # print(model.Wint.norm(2),model.hdt.norm(2))

        return loss1+self.pt.reg_lamda*(energyloss+wnorm1) # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_GateReg_TDFF_L1(self, outputl, outlab, model=None, cstep=None, posi_ctrl=1):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_GateReg_TDFF_L1")
            return True

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

        wnorm1=0
        for namep in self.allname:
                wattr = getattr(self.pt.rnn, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))
        # wattr = getattr(self.rnn, "h2o")
        # wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1 + 0.01*wnorm1 + self.pt.reg_lamda*loss_gate # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_eval_mem_tdff(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: custom_eval_mem_tdff")
            return True


        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(3)]  # x,label,hd_middle

        try:
            self.pt.evalmem[0].append(x.cpu().data.numpy())
            self.pt.evalmem[1].append(label.cpu().data.numpy())
            self.pt.evalmem[2].append(rnn.mdl.cpu().data.numpy())
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
            for item in self.allname:
                self.training_data_mem["gradInfo"][item]=[]
        for item in self.allname:
            attr=getattr(self.pt.rnn,item)
            grad=attr.weight.grad
            self.training_data_mem["gradInfo"][item].append(grad)

    def custom_example_data_collection_tdff(self, x, output, hidden, label):

        if self.test_print_interface:
            print("example_data_collection interface: custom_example_data_collection_tdff")
            return True

        lsize = x.shape[-1]
        anslength = label.shape[-1]
        label_onehot = torch.zeros(anslength, lsize)
        for ii in range(label.shape[-1]):
            id = label[0, ii]
            label_onehot[ii, id] = 1

        if self.pt.data_col_mem is None:
            # Step 1 of example_data_collection
            self.pt.data_col_mem=dict([])
            self.pt.data_col_mem["titlelist"]=["input","label","predict"]
            self.pt.data_col_mem["sysmlist"]=[True,True,True]
            self.pt.data_col_mem["mode"]="predict"
            self.pt.data_col_mem["datalist"] = [None,None,None]
            self.pt.data_col_mem["climlist"] = [[None, None], [None, None] ,[None, None]]

        if self.pt.data_col_mem["datalist"][0] is None:
            self.pt.data_col_mem["datalist"][0] = torch.squeeze(x)
            self.pt.data_col_mem["datalist"][1] = torch.squeeze(label_onehot)
            self.pt.data_col_mem["datalist"][2] = torch.squeeze(output)
            # self.data_col_mem["datalist"][3] = self.pt.rnn.Wint.view(-1,1)
            # self.data_col_mem["datalist"][4] = self.pt.rnn.hdt[0].view(-1,1)
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
            self.data_col_mem["datalist"][3] = torch.squeeze(self.pt.rnn.hdt).transpose(1,0)
            # self.data_col_mem["datalist"][5] = self.rnn.hdt[1].view(-1,1)

class PyTrain_Interface_backwardreverse(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.print_interface()

    def init_data(self,*args,**kwargs):
        return self.init_data_sup_backwardreverse()

    def get_data(self, batch=None, rstartv=None):
        return self.get_data_sup_backwardreverse(batch=batch, rstartv=rstartv)

    def lossf(self, outputl, outlab, model=None, cstep=None):
        return self.KNWLoss_backwardreverse(outputl, outlab, model=model, cstep=cstep)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None):
        return self.KNWLoss_backwardreverse(outputl, outlab, model=model, cstep=cstep)

    def eval_mem(self, x, label, rnn):
        return self.custom_eval_mem_backwardreverse(x, label, rnn)

    def example_data_collection(self,x, output, hidden, label, items=None):
        return self.custom_example_data_collection_backwardreverse(x, output, hidden, label)

    def init_data_sup_backwardreverse(self,limit=1e9):
        """
        _init_data_sup_backwardreverse 2019.8.13

        :param limit:
        :return:
        """
        if self.test_print_interface:
            print("Init data: _init_data_sup_backwardreverse")
            return True

        assert self.pt.supervise_mode
        assert len(self.pt.dataset["dataset"]) == 2 # data_set,pvec_l
        assert self.pt.digit_input
        assert self.pt.id_2_vec is None # No embedding, one-hot representation

        self.pt.dataset_length = len(self.pt.dataset["dataset"][0])
        print("Dataset length ", self.pt.dataset_length)

        self.pt.databp=torch.zeros((len(self.pt.dataset["dataset"][0]),self.pt.lsize_in))
        for ii, data in enumerate(self.pt.dataset["dataset"][0]):
            self.pt.databp[ii,data] = 1.0
        self.pt.data_init = True

    def get_data_sup_backwardreverse(self,batch=None, rstartv=None):

        if self.test_print_interface:
            print("Get data: get_data_sup_backwardreverse")
            return True

        assert self.pt.supervise_mode
        assert len(self.pt.dataset["dataset"]) == 2  # data_set,pvec_l
        assert self.pt.data_init

        if batch is None:
            batch=self.pt.batch

        if rstartv is None: # random mode
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset["dataset"][0]) - 1))
        else:
            assert len(rstartv)==batch

        xl = np.zeros(batch)
        outl = np.zeros(batch)
        for iib in range(batch):
            xl[iib] = self.pt.dataset["dataset"][0][int(rstartv[iib])]
            outl[iib] = self.pt.dataset["label"][int(rstartv[iib])]
        inlab = torch.from_numpy(xl)
        inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(outl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(batch, self.pt.lsize_in)
        for iib in range(batch):
            vec1=self.pt.databp[int(rstartv[iib])]
            vec1m[iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        pvec_mat = torch.zeros(batch, self.pt.lsize_in)
        for iib in range(batch):
            vec1=self.pt.dataset["dataset"][1][int(rstartv[iib])]
            pvec_mat[iib,:]=torch.from_numpy(vec1)
        pvec_matv = Variable(pvec_mat, requires_grad=True).type(torch.FloatTensor)

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)
            pvec_matv = pvec_matv.to(self.pt.device)

        # print(x.shape,pvec_matv.shape)
        return (x, pvec_matv) , outlab, inlab

    def KNWLoss_backwardreverse(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLos: KNWLoss_backwardreverse")
            return True

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

        # loss2=torch.nn.functional.kl_div(model.nsoftmax(outputl[0]),model.nsoftmax(outputl[1]))
        # loss2 = torch.nn.functional.kl_div(outputl[0], outputl[1])
        loss2 = torch.mean(cal_kldiv_torch(model.nsoftmax(outputl[0]),model.nsoftmax(outputl[1])))

        # print(loss1,loss2)

        return loss1-0.1*loss2 #+self.reg_lamda*wnorm1 # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_eval_mem_backwardreverse(self, x, label, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("evalmem: custom_eval_mem_backwardreverse")
            return True

        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(3)]  # label,p_vec, output

        try:
            self.pt.evalmem[0].append(x[0].cpu().data.numpy())
            self.pt.evalmem[1].append(x[1].cpu().data.numpy())
            self.pt.evalmem[2].append(rnn.lgoutput.cpu().data.numpy())
        except:
            # print("eval_mem failed")
            pass

    def custom_example_data_collection_backwardreverse(self, x, output, hidden, label, items=None):

        if self.test_print_interface:
            print("example_data_collection: custom_example_data_collection_backwardreverse")
            return True

        if self.pt.data_col_mem is None:
            # Step 1 of example_data_collection
            self.pt.data_col_mem=dict([])
            self.pt.data_col_mem["titlelist"]=["input","p_vec","sample_vec"]
            self.pt.data_col_mem["sysmlist"]=[True,True,True]
            self.pt.data_col_mem["mode"]="predict"
            self.pt.data_col_mem["datalist"] = [None,None,None]
            self.pt.data_col_mem["climlist"] = [[None, None], [None, None], [None, None]]

        if self.pt.data_col_mem["datalist"][0] is None:
            self.pt.data_col_mem["datalist"][0] = x[0]
            self.pt.data_col_mem["datalist"][1] = x[1]
            self.pt.data_col_mem["datalist"][2] = output[0]

class PyTrain_Interface_mnist(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self,subversion=0):
        super(self.__class__, self).__init__()

        self.subversion=subversion # 0: normal, 1: autoencoder

        if self.subversion==0:
            self.allname = []
        elif self.subversion==1:
            # self.allname = ["i2h","h2m","m2o","h12h2"]
            self.allname = ["i2h","h2m","m2o"]
            # self.allname = ["m2o"]

        self.print_interface()

    def init_data(self,*args,**kwargs):
        if self.test_print_interface:
            print("init_data: do nothing")
            return True
        self.pt.data_init=True
        self.pt.dataset_length = len(self.pt.dataset["dataset"])
        print("Dataset length ", self.pt.dataset_length)

    def get_data(self, batch=None, rstartv=None):
        if self.subversion==0:
            return self.get_data_mnist(batch=batch, rstartv=rstartv)
        elif self.subversion==1:
            return self.get_data_mnist_autoencoder(batch=batch, rstartv=rstartv)

    def get_data_mnist(self, batch=None, rstartv=None):

        if self.test_print_interface:
            print("Get data interface: get_data_mnist")
            return True

        assert self.pt.supervise_mode
        assert self.pt.data_init

        if batch is None:
            batch=self.pt.batch

        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset["dataset"]) - 1))
        else:
            assert len(rstartv)==batch

        vec1m = self.pt.dataset["dataset"][rstartv,:]
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        outlab=self.pt.dataset["label"][rstartv]
        outlab = outlab.type(torch.LongTensor)

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)

        return x, outlab, None

    def get_data_mnist_autoencoder(self, batch=None, rstartv=None):

        if self.test_print_interface:
            print("Get data interface: get_data_mnist_autoencoder")
            return True

        assert self.pt.supervise_mode
        assert self.pt.data_init

        if batch is None:
            batch=self.pt.batch

        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(self.pt.dataset["dataset"]) - 1))
        else:
            assert len(rstartv)==batch

        vec1m = self.pt.dataset["dataset"][rstartv,:]
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        outlab = x

        if self.pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)

        return x, outlab, None

    def while_training(self,iis):
        if self.subversion == 0:
            pass
        elif self.subversion == 1:
            return self.while_training_non_neg_clamp(iis)
            # pass

    def while_training_non_neg_clamp(self,iis):
        if self.test_print_interface:
            print("While_training interface: while_training_non_neg_clamp")
            return True

        self.pt.rnn.m2o.weight.data.clamp_(0)

    def lossf(self, outputl, outlab, model=None, cstep=None):
        # return self.KNWLoss_GateReg_TDFF_L1(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        if self.subversion==0:
            # return self.KNWLoss_GateReg_MNIST(outputl, outlab, model=model, cstep=cstep)
            return self.KNWLoss_GateReg_MNIST_eval(outputl, outlab, model=model, cstep=cstep)
        elif self.subversion==1:
            return self.MSE_wta_autoencoder(outputl, outlab, model=model, cstep=cstep)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None):
        # return self.KNWLoss_GateReg_TDFF_L1(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        if self.subversion == 0:
            return self.KNWLoss_GateReg_MNIST_eval(outputl, outlab, model=model, cstep=cstep)
        elif self.subversion == 1:
            return self.MSE_wta_autoencoder_eval(outputl, outlab, model=model, cstep=cstep)

    def eval_mem(self, x, label, output,rnn):
        return self.custom_eval_mem_mnist_autoencoder(x, label, output, rnn)

    def custom_eval_mem_mnist_autoencoder(self, x, label, output, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: custom_eval_mem_mnist_autoencoder")
            return True


        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(2)]  # x,label,hd_middle

        try:
            self.pt.evalmem[0].append(x.cpu().data.numpy())
            self.pt.evalmem[1].append(rnn.mdl.cpu().data.numpy())
        except:
            # print("eval_mem failed")
            pass

    def KNWLoss_GateReg_MNIST(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_GateReg_MNIST")
            return True

        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        maxsig=torch.max(model.sigmoid(model.input_gate))
        loss_gate = torch.mean(model.sigmoid(model.input_gate)/maxsig)
        # loss_gate = torch.mean(model.sigmoid(model.input_gate))
        #
        wnorm1=0
        for namep in self.allname:
                wattr = getattr(self.pt.rnn, namep)
                wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))

        return loss1 + 0.01*wnorm1 + self.pt.reg_lamda*loss_gate # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def KNWLoss_GateReg_MNIST_eval(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_GateReg_MNIST_eval")
            return True

        lossc = torch.nn.CrossEntropyLoss()
        if self.pt.specialized_digit_eval is None:
            loss1 = lossc(outputl, outlab)
        else:
            outlab_sp=(outlab==self.pt.specialized_digit_eval)
            outlab_sp = outlab_sp.type(torch.LongTensor)
            if torch.cuda.is_available():
                outlab_sp=outlab_sp.to(self.pt.cuda_device)
            outputl_sum=torch.sum(torch.exp(outputl),dim=-1)
            outputl_notsp = torch.log(outputl_sum - torch.exp(outputl[:,self.pt.specialized_digit_eval]))
            outputl_sp=torch.cat((outputl_notsp.reshape(-1,1),outputl[:,self.pt.specialized_digit_eval].reshape(-1,1)),dim=-1)
            loss1 = lossc(outputl_sp, outlab_sp)

        # loss_gate = torch.mean(model.sigmoid(model.input_gate))
        # #
        # wnorm1=0
        # for namep in self.allname:
        #         wattr = getattr(self.pt.rnn, namep)
        #         wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))

        return loss1# + 0.01*wnorm1 + self.pt.reg_lamda*loss_gate # + 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    def MSE_wta_autoencoder(self, outputl, inputl, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: MSE_wta_autoencoder")
            return True

        # lossc=torch.nn.MSELoss(reduce=True,reduction="mean")
        # loss1 = lossc(outputl, inputl)

        loss1 = torch.mean((outputl-inputl)*(outputl-inputl))

        wnorm1=0
        for namep in self.allname:
            wattr = getattr(self.pt.rnn, namep)
            wnorm1=wnorm1+torch.mean(torch.abs(wattr.weight))

        # Wm2o=model.m2o.weight
        # Wm2o=Wm2o.reshape((28,28,-1))
        #
        # Wm2ox = (Wm2o * model.Tx).reshape(28*28,-1)
        #
        # Wm2ox_var=torch.var(Wm2ox,dim=0)
        #
        # Wm2oy = (Wm2o * model.Ty).reshape(28 * 28, -1)
        #
        # Wm2oy_var = torch.var(Wm2oy, dim=0)
        #
        # Wm2o_r =torch.mean(Wm2ox_var+Wm2oy_var)

        energyloss = torch.mean(torch.mul(model.mdl, model.mdl))

        # print(wnorm1,energyloss,Wm2o_r)

        return loss1 + self.pt.reg_lamda*(wnorm1+energyloss)

    def MSE_wta_autoencoder_eval(self, outputl, inputl, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: MSE_wta_autoencoder_eval")
            return True

        # lossc=torch.nn.MSELoss(reduce=True,reduction="mean")
        # loss1 = lossc(outputl, inputl)

        loss1 = torch.mean((outputl-inputl)*(outputl-inputl))

        return loss1

    def example_data_collection(self, x, output, hidden, label):
        if self.subversion==0:
            return self.custom_example_data_collection_mnist(x, output, hidden, label)
        elif self.subversion==1:
            return self.custom_example_data_collection_mnist_autoencoder(x, output, hidden, label)

    # def lossf_eval(self, outputl, outlab, model=None, cstep=None):
    #     # return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
    #     # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
    #     return self._lossf_eval(outputl, outlab, model=model, cstep=cstep)


    # def eval_mem(self, x, label, rnn):
    #     return self.custom_eval_mem_tdff(x, label, rnn)

    # def example_data_collection(self, x, output, hidden, label):
    #     return self.custom_example_data_collection_tdff(x, output, hidden, label)

    def custom_example_data_collection_mnist(self, x, output, hidden, label):

        if self.test_print_interface:
            print("example_data_collection interface: custom_example_data_collection_mnist")
            return True

        label_onehot = torch.zeros(10)
        label_onehot[label] = 1


        if self.pt.data_col_mem is None:
            # Step 1 of example_data_collection
            self.pt.data_col_mem=dict([])
            self.pt.data_col_mem["titlelist"]=["input","label","predict"]
            self.pt.data_col_mem["sysmlist"]=[True,True,True]
            self.pt.data_col_mem["mode"]="predict"
            self.pt.data_col_mem["datalist"] = [None,None,None]
            self.pt.data_col_mem["climlist"] = [[None, None], [None, None] ,[None, None]]

        if self.pt.data_col_mem["datalist"][0] is None:
            self.pt.data_col_mem["datalist"][0] = x.reshape((28,28)).transpose(1,0)
            # self.pt.data_col_mem["datalist"][0] = torch.squeeze(x)
            self.pt.data_col_mem["datalist"][1] = torch.squeeze(label_onehot)
            self.pt.data_col_mem["datalist"][2] = torch.squeeze(output)

    def custom_example_data_collection_mnist_autoencoder(self, x, output, hidden, label):

        if self.test_print_interface:
            print("example_data_collection interface: custom_example_data_collection_mnist")
            return True


        if self.pt.data_col_mem is None:
            # Step 1 of example_data_collection
            self.pt.data_col_mem=dict([])
            self.pt.data_col_mem["titlelist"]=["input","label","predict"]
            self.pt.data_col_mem["sysmlist"]=[True,True,True]
            self.pt.data_col_mem["mode"]="predict"
            self.pt.data_col_mem["datalist"] = [None,None,None]
            self.pt.data_col_mem["climlist"] = [[None, None], [None, None] ,[None, None]]

        if self.pt.data_col_mem["datalist"][0] is None:
            self.pt.data_col_mem["datalist"][0] = x.reshape((28,28)).transpose(1,0)
            self.pt.data_col_mem["datalist"][1] = torch.squeeze(self.pt.rnn.mdl)
            self.pt.data_col_mem["datalist"][2] = output.reshape((28,28)).transpose(1,0)

class PyTrain_Interface_W2V(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom
    """
    def __init__(self,prior,threshold=1e-5,version=0):
        super(self.__class__, self).__init__()
        self.version = version
        if version=="skipgram":
            self.prior=prior
            self.prior=self.prior/np.sum(self.prior)
            self.threshold=threshold
            self.dropmem=[]

    def init_data(self,*args,**kwargs):
        return self.init_data_empty()

    def get_data(self, dataset_dict, batch=None, rstartv=None):
        if batch is None:
            batch = self.pt.batch
        if self.version=="skipgram":
            return self.get_data_w2v(dataset_dict, batch=batch, rstartv=rstartv)
        elif self.version=="softmax":
            return self.get_data_w2v_sfm(dataset_dict, batch=batch, rstartv=rstartv)
        elif self.version=="coop_softmax":
            data_shape = [self.pt.window, batch, self.pt.lsize_in]
            return MyDataFun.get_data_sup(dataset_dict, data_shape, self.pt.pt_emb, self.pt, rstartv=None)

    def lossf(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        if self.version == "skipgram":
            return self.EmptyLoss(outputl, outlab, model=model, cstep=cstep)
        elif self.version == "softmax":
            return MyLossFun.KNWLoss(outputl, outlab)
        elif self.version == "coop_softmax":
            return self.KNWLoss_coop(outputl)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        if self.version == "skipgram":
            return self.EmptyLoss(outputl, outlab, model=model, cstep=cstep)
        elif self.version == "softmax":
            return MyLossFun.KNWLoss(outputl, outlab)
        elif self.version == "coop_softmax":
            return self.KNWLoss_coop(outputl)

    def eval_mem(self, x, label, output, rnn):
        # return self.eval_mem_Full_eval(x, label, output, rnn)
        pass

    def EmptyLoss(self, outputl, outlab, model=None, cstep=None):
        loss=outputl
        return loss

    def KNWLoss_coop(self, outputl):
        lossc = torch.nn.CrossEntropyLoss()
        outputl0 = outputl[0][0].permute(1, 2, 0)
        outlab0 = outputl[0][1] # batch, window_size
        loss0 = lossc(outputl0, outlab0)
        outputl1 = outputl[1][0].permute(1, 2, 0)
        outlab1 = outputl[1][1]  # batch, window_size
        loss1 = lossc(outputl1, outlab1)
        return loss0+loss1

    def init_data_empty(self,limit=1e9):
        pass

    def get_data_w2v(self, dataset_dict, batch=None, rstartv=None, freq_reg=False):

        dataset = dataset_dict["dataset"]

        if batch is None:
            batch=self.pt.batch

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - self.pt.window - 1))
        xl = np.zeros((self.pt.window,batch))
        for iib in range(batch):
            if freq_reg:
                dropped=0
                for iiw in range(self.pt.window):
                    if int(rstartv[iib])+iiw+dropped>=len(dataset):
                        dropped=dropped-self.pt.window
                    item=dataset[int(rstartv[iib])+iiw+dropped]
                    pdrop=1-np.sqrt(self.threshold/self.prior[item])
                    if np.random.rand()>pdrop:
                        xl[iiw,iib]=item
                    else:
                        dropped=dropped+1
                self.dropmem.append(dropped)
            else:
                xl[:, iib] = dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]

        xl=torch.from_numpy(xl).type(torch.LongTensor)

        if self.pt.gpuavail:
            xl = xl.to(self.pt.device)

        return xl, None, None

    def get_data_w2v_sfm(self, dataset_dict, batch=None, rstartv=None):

        dataset = dataset_dict["dataset"]
        window_size=self.pt.rnn.window_size
        window_size_ext = self.pt.window

        if batch is None:
            batch=self.pt.batch

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window_size_ext - 1))
        xl = np.zeros((window_size_ext,batch))
        for iib in range(batch):
            xl[:, iib] = dataset[int(rstartv[iib]):int(rstartv[iib]) + window_size_ext]

        xl=torch.from_numpy(xl).type(torch.LongTensor)

        outlab = torch.zeros((window_size_ext-window_size+1,batch)).type(torch.LongTensor)
        outlab[:,:] = xl[int((window_size+1)/2)-1:window_size_ext-int((window_size+1)/2)+1,:]
        outlab=outlab.permute(1,0)

        if self.pt.gpuavail:
            xl = xl.to(self.pt.device)
            outlab = outlab.to(self.pt.device)

        return xl, outlab, None

class PyTrain_Interface_sup(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom, supervised learning
    """
    def __init__(self,version=0):
        super(self.__class__, self).__init__()
        self.version = version
        self.allname=["h2c","c2o"]

    def get_data(self, dataset, batch=None, rstartv=None):
        if batch is None:
            batch = self.pt.batch
        if self.version == "gsvib_coop_special" or self.version == "gsvib_attcoop":
            data_shape = [[self.pt.window, batch, self.pt.lsize_in[0]],[self.pt.window, batch, self.pt.lsize_in[1]]]
            return MyDataFun.get_data_sup_coop_special(dataset, data_shape, self.pt.pt_emb, self.pt, rstartv=None, shift=False)
        elif self.version == "gsvib_task2":
            data_shape = [self.pt.window, batch, self.pt.lsize_in[0],self.pt.lsize_in[1]]
            return MyDataFun.get_data_sup_task2(dataset, data_shape, self.pt, rstartv=None)
            # return self.get_data_sup_task2(dataset, batch=batch, rstartv=rstartv)
        elif self.version == "with_mask_data_loss":
            data_shape = [self.pt.window, batch, self.pt.lsize_in]
            return MyDataFun.get_data_sup_with_mask(dataset, data_shape, self.pt, rstartv=None)
        elif self.version == "with_mask":
            data_shape = [self.pt.window, batch, self.pt.lsize_in]
            return MyDataFun.get_data_sup(dataset, data_shape, self.pt.pt_emb,self.pt, rstartv=None)
        elif self.version in ["gsvib_hsoftmax2","hsoftmax2"]:
            assert self.pt.output_cluster is not None
            data_shape = [self.pt.window, batch, self.pt.lsize_in, self.pt.output_cluster]
            return MyDataFun.get_data_sup_hsoftmax2(dataset, data_shape, self.pt.pt_emb, self.pt, rstartv=None, shift=False)
        else:
            return self.get_data_sup(dataset, batch=batch, rstartv=rstartv)

    def lossf(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        if self.version == "vib":
            return MyLossFun.KNWLoss_VIB(outputl, outlab, self.pt.reg_lamda, model,cstep, self.pt.beta_warmup, self.pt.scale_factor)
        elif self.version in ["gsvib" , "gsvib_coop_special" , "gsvib_task2","gsvib_hsoftmax2"]:
            return MyLossFun.KNWLoss_GSVIB(outputl, outlab, self.pt.reg_lamda, model, cstep, self.pt.beta_warmup, self.pt.scale_factor)
        elif self.version == "sigvib": # advanced version of sigmoid gsvib, each neuron is either 0 or 1
            return MyLossFun.KNWLoss_SIGVIB(outputl, outlab, self.pt.reg_lamda, model, self.pt.scale_factor)
        elif self.version == "gsvib_attcoop":
            return MyLossFun.KNWLoss_GSVIB_ATTCOOP(outputl, outlab, self.pt.reg_lamda, model, cstep, self.pt.beta_warmup, self.pt.scale_factor)
        elif self.version == "att_infofolk":
            return MyLossFun.KNWLoss_infofork(outputl, outlab, self.pt.reg_lamda, model, scale_factor=self.pt.scale_factor[0], scale_factor2=self.pt.scale_factor[1])
        elif self.version== "ereg":
            return MyLossFun.KNWLoss_EnergyReg(outputl, outlab, self.pt.reg_lamda, model, balance_para=10)
        elif self.version == "with_mask_data_loss":
            return MyLossFun.KNWLoss_withmask(outputl, outlab, model=model, cstep=cstep, data_loss_mask=loss_data)
        elif self.version == "with_mask":
            return MyLossFun.KNWLoss_withmask(outputl, outlab, model=model, cstep=cstep, data_loss_mask=None)
        elif self.version == "hsoftmax2":
            return MyLossFun.KNWLoss(outputl, outlab, model=model, hsoftmax_depth=2)
        elif self.version == "default":
            return MyLossFun.KNWLoss(outputl, outlab)
        else:
            raise Exception("Unsupported version.")

    def lossf_eval(self, outputl, outlab, model=None, cstep=None, loss_data=None):
        # return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # return self.KNWLoss_GateReg_TDFF_KL(outputl, outlab, model=model, cstep=cstep)
        # if self.version in [0]:
        #     return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # elif self.version==1:
        #     return self.KNWLoss(outputl, outlab, model=model, cstep=cstep)
        # return MyLossFun.KNWLoss(outputl, outlab)
        if self.version == "with_mask":
            return MyLossFun.KNWLoss_withmask(outputl, outlab, model=model, cstep=cstep, data_loss_mask=loss_data)
        elif self.version in ["gsvib_hsoftmax2","hsoftmax2"]:
            return MyLossFun.KNWLoss(outputl, outlab, model=model,hsoftmax_depth=2)
        else:
            # return MyLossFun.KNWLoss_VIB_EVAL(outputl, outlab, model)
            # return MyLossFun.KNWLoss_GSVIB(outputl, outlab, self.pt.reg_lamda, model, cstep, self.pt.beta_warmup,
            #                                self.pt.scale_factor)
            return MyLossFun.KNWLoss(outputl, outlab)

    def get_data_sup(self, dataset_dict, batch=None, rstartv=None, shift=False):
        """

        :param batch:
        :param rstartv:
        :param shift: default to true unless for transformer type network
        :return:
        """

        if self.test_print_interface:
            print("get_data interface: get_data_sup")
            return True

        assert self.pt.digit_input
        assert self.pt.supervise_mode

        if batch is None:
            batch=self.pt.batch

        labelset = dataset_dict["label"]
        dataset = dataset_dict["dataset"]

        # Generating output label # numpy random is necessary for distributed data parallel
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - self.pt.window - 1))
        yl = np.zeros((batch,self.pt.window))

        for iib in range(batch):
            if shift:
                yl[iib,:] = labelset[int(rstartv[iib])+1:int(rstartv[iib]) + self.pt.window+1]
            else:
                yl[iib, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(self.pt.window, batch, self.pt.lsize_in)
        for iib in range(batch):
            ptdata=dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
            vec1=self.pt.pt_emb(torch.LongTensor(ptdata))
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if self.pt.gpuavail:
            outlab = outlab.to(self.pt.device)
            x = x.to(self.pt.device)

        return x, outlab, None

    def eval_mem(self, x, label, output, rnn):
        return self.custom_eval_mem_sup(x, label, output, rnn)
        # return self.custom_eval_mem_sup_special(x, label, output, rnn)

    def custom_eval_mem_sup(self, x, label, output, rnn):
        """
        Archiving date
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: custom_eval_mem_sup")
            return True


        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(7)]  # x,label,context

        try:
            self.pt.evalmem[0].append(x.cpu().data.numpy())
        except:
            pass
        self.pt.evalmem[1].append(label.permute(1,0).cpu().data.numpy())
        # self.pt.evalmem[1].append(label.permute(1, 0, 2).cpu().data.numpy())
        self.pt.evalmem[2].append(rnn.context.cpu().data.numpy())
        # self.pt.evalmem[3].append(rnn.ctheta.cpu().data.numpy())
        # self.pt.evalmem[4].append(rnn.cmu.cpu().data.numpy())
        # self.pt.evalmem[5].append(output.cpu().data.numpy())
        self.pt.evalmem[3].append(rnn.gssample.cpu().data.numpy())
        # ent = cal_entropy(output.cpu().data, log_flag=True, byte_flag=False, torch_flag=True, cuda_device=self.pt.cuda_device)
        lsoftmax = torch.nn.LogSoftmax(dim=-1)
        if getattr(rnn,"hsoftmax_depth",0)>1:
            cluster_size=rnn.output_cluster #(wd, b, s, outc+outh)
            outputc= lsoftmax(output[:,:,:,:cluster_size])
            entc = cal_entropy(outputc.cpu().data, log_flag=True, byte_flag=False, torch_flag=True)
            outputh = lsoftmax(output[:, :, :, cluster_size:])
            enth = cal_entropy(outputh.cpu().data, log_flag=True, byte_flag=False, torch_flag=True)
            ent = entc + enth
        else:
            output = lsoftmax(output)
            ent = cal_entropy(output.cpu().data, log_flag=True, byte_flag=False, torch_flag=True)
        self.pt.evalmem[4].append(ent.cpu().data.numpy())
        # self.pt.evalmem[5].append(ent.cpu().data.numpy())
        # self.pt.evalmem[5].append(rnn.level1_coop.gssample.cpu().data.numpy())
        # self.pt.evalmem[6].append(rnn.level1_coop.gssample_coop.cpu().data.numpy())
        # self.pt.evalmem[2].append(rnn.seq1_coop.context.cpu().data.numpy())
        # self.pt.evalmem[3].append(rnn.seq1_coop.gssample.cpu().data.numpy())
        try:
            # pass
            self.pt.evalmem[5].append(rnn.context_coop.cpu().data.numpy())
            self.pt.evalmem[6].append(rnn.gssample_coop.cpu().data.numpy())
            # self.pt.evalmem[5].append(rnn.gs_mask.cpu().data.numpy())
            # self.pt.evalmem[5].append(rnn.coop_sample.cpu().data.numpy())
            # self.pt.evalmem[5].append(rnn.seq1_coop.gssample.cpu().data.numpy())
            # self.pt.evalmem[6].append(rnn.level1_coop.gssample_coop.cpu().data.numpy())
            # self.pt.evalmem[5].append(rnn.attention_prob.cpu().data.numpy())
            # self.pt.evalmem[5].append(rnn.seq1_coop.context_coop.cpu().data.numpy())
            # self.pt.evalmem[6].append(rnn.seq1_coop.gssample_coop.cpu().data.numpy())
        except:
            pass

    def custom_eval_mem_sup_special(self, x, label, output, rnn):
        """
        Archiving date
        x:[x_wrd,x_pos], outlab
        :param output:
        :param hidden:
        :return:
        """
        if self.test_print_interface:
            print("Eval mem interface: custom_eval_mem_sup")
            return True


        if self.pt.evalmem is None:
            self.pt.evalmem = [[] for ii in range(7)]  # x,label,context

        self.pt.evalmem[0].append(x[1].cpu().data.numpy()) # pos label
        self.pt.evalmem[1].append(label.cpu().permute(1,0).data.numpy()) # wrd label
        self.pt.evalmem[2].append(rnn.context.cpu().data.numpy())
        self.pt.evalmem[3].append(rnn.gssample.cpu().data.numpy())
        ent = cal_entropy(output.cpu().data, log_flag=True, byte_flag=False, torch_flag=True)
        self.pt.evalmem[4].append(ent.cpu().data.numpy())
        self.pt.evalmem[5].append(rnn.attention_prob.cpu().data.numpy())
        self.pt.evalmem[6].append(rnn.attention_sample.cpu().data.numpy())

class PyTrain_Interface_advsup(PyTrain_Interface_Default):
    """
    A pytrain interface object to plug into PyTrain_Custom, supervised learning with adversary labels (not working)
    """

    def __init__(self, version=0):
        super(self.__class__, self).__init__()
        self.version = version
        self.allname = ["h2c", "c2o","ac2o"]

    def get_data(self, batch=None, rstartv=None):
        return self.get_data_adsup(batch=batch, rstartv=rstartv)

    def lossf(self, outputl, outlab, model=None, cstep=None):
        return self.KNWLoss_WeightReg_adv(outputl, outlab, model=model, cstep=cstep)

    def lossf_eval(self, outputl, outlab, model=None, cstep=None):
        return self.KNWLoss_adv(outputl, outlab, model=model, cstep=cstep)

    def eval_mem(self, x, label, output, rnn):
        # return self.eval_mem_Full_eval(x, label, output, rnn)
        pass

    def KNWLoss_WeightReg_adv(self, outputl, outlab, model=None, cstep=None):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss_WeightReg")
            return True
        output=outputl[0]
        output = output.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(output, outlab[0])

        advoutput = outputl[1]
        advoutput = advoutput.permute(1, 2, 0)
        loss2 = lossc(advoutput, outlab[1])

        # wnorm1 = 0
        # for namep in self.allname:
        #     wattr = getattr(self.pt.rnn, namep)
        #     wnorm1 = wnorm1 + torch.mean(torch.abs(wattr.weight))
        # wnorm1 = wnorm1 / len(self.allname)

        return loss1-self.pt.reg_lamda*loss2 # + self.pt.reg_lamda * wnorm1

    def KNWLoss_adv(self, outputl, outlab, model=None, cstep=None, id=0):

        if self.test_print_interface:
            print("KNWLoss interface: KNWLoss")
            return True

        output = outputl[id].permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(output, outlab[id])
        return loss1

    def get_data_adsup(self, batch=None, rstartv=None, shift=False):
        """

        :param batch:
        :param rstartv:
        :param shift: default to true unless for transformer type network
        :return:
        """

        if self.test_print_interface:
            print("get_data interface: get_data_sup")
            return True

        assert self.pt.digit_input
        assert self.pt.supervise_mode

        if batch is None:
            batch = self.pt.batch


        dataset = self.pt.dataset["dataset"]
        labelset = self.pt.dataset["label"]
        advlabelset = self.pt.dataset["advlabel"]

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - self.pt.window - 1))

        yl = np.zeros((batch, self.pt.window))
        for iib in range(batch):
            if shift:
                yl[iib, :] = labelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.pt.window + 1]
            else:
                yl[iib, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        al = np.zeros((batch, self.pt.window))
        for iib in range(batch):
            if shift:
                al[iib, :] = advlabelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.pt.window + 1]
            else:
                al[iib, :] = advlabelset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
        adoutlab = torch.from_numpy(al)
        adoutlab = adoutlab.type(torch.LongTensor)

        vec1m = torch.zeros(self.pt.window, batch, self.pt.lsize_in)
        for iib in range(batch):
            ptdata = dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
            vec1 = self.pt.pt_emb(torch.LongTensor(ptdata))
            vec1m[:, iib, :] = vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if self.pt.gpuavail:
            outlab = outlab.to(self.pt.device)
            adoutlab = adoutlab.to(self.pt.device)
            x = x.to(self.pt.device)

        return x, [outlab,adoutlab], None

class MyDataFun(object):

    @staticmethod
    def get_data_sup(dataset_dict, data_shape, pt_emb, ptrain_pt, rstartv=None, shift=False):
        """

        :param dataset_dict:
        :param data_shape: shape like [window, batch, lsize_in]
        :param pt_emb: embedding
        :param ptrain_pt: pytrain interface
        :param rstartv: random start
        :param shift: data shift
        :return:
        """
        assert len(data_shape)==3

        window, batch, lsize_in = data_shape


        assert ptrain_pt.digit_input
        assert ptrain_pt.supervise_mode

        labelset = dataset_dict["label"]
        dataset = dataset_dict["dataset"]

        # Generating output label # Generating output label # numpy random is necessary for distributed data parallel
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))
        yl = np.zeros((batch,window))

        for iib in range(batch):
            if shift:
                yl[iib,:] = labelset[int(rstartv[iib])+1:int(rstartv[iib]) + window+1]
            else:
                yl[iib, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(window, batch,lsize_in)
        for iib in range(batch):
            ptdata=dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1=pt_emb(torch.LongTensor(ptdata))
            vec1m[:,iib,:]=vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if ptrain_pt.gpuavail:
            outlab = outlab.to(ptrain_pt.device)
            x = x.to(ptrain_pt.device)

        return x, outlab, None

    @staticmethod
    def get_data_sup_with_mask(dataset_dict, data_shape , ptrain_pt, rstartv=None, shift=False, mask_t=100):
        """
        :param dataset_dict:
        :param data_shape: shape like [window, batch, lsize_in]
        :param pt_emb: embedding
        :param ptrain_pt: pytrain interface
        :param rstartv: random start
        :param shift: data shift
        ### mask_t , a trial of masking high frequency word
        :return:
        """
        assert len(data_shape) == 3

        window, batch, lsize_in = data_shape

        assert ptrain_pt.digit_input
        assert ptrain_pt.supervise_mode

        labelset = dataset_dict["label"]
        dataset = dataset_dict["dataset"]

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        yl = np.zeros((batch, window))
        for iib in range(batch):
            if shift:
                yl[iib, :] = labelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1]
            else:
                yl[iib, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(window, batch, lsize_in)
        data_loss_mask = torch.zeros((window, batch))
        for iib in range(batch):
            ptdata = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1 = ptrain_pt.pt_emb(torch.LongTensor(ptdata))
            vec1m[:, iib, :] = vec1
            data_loss_mask[torch.LongTensor(ptdata)<100,iib]=1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if ptrain_pt.gpuavail:
            outlab = outlab.to(ptrain_pt.device)
            x = x.to(ptrain_pt.device)
            data_loss_mask = data_loss_mask.to(ptrain_pt.device)

        return x, outlab, data_loss_mask

    @staticmethod
    def get_data_sup_task2(dataset_dict, data_shape, ptrain_pt, rstartv=None,shift=False):
        """

        :param dataset_dict:
        :param data_shape: shape like [window, batch, lsize_in]
        :param pt_emb: embedding
        :param ptrain_pt: pytrain interface
        :param rstartv: random start
        :param shift: data shift
        :return:
        """
        assert len(data_shape) == 4

        window, batch, lsize_in_self, lsize_in_pos = data_shape

        assert ptrain_pt.digit_input
        assert ptrain_pt.supervise_mode

        labelset = dataset_dict["label"]
        dataset_dict = dataset_dict["dataset"]

        dataset = dataset_dict["words"]
        gssample_self = dataset_dict["gssample_self"]
        gssample_pos = dataset_dict["gssample_pos"]

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(gssample_self) - window - 1))
        yl = np.zeros((batch, window))

        for iib in range(batch):
            if shift:
                yl[iib, :] = labelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1]
            else:
                cplabelset=labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
                yl[iib, :] = cplabelset
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(window, batch, lsize_in_self)
        for iib in range(batch):
            ptdata = gssample_self[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1m[:, iib, :] = torch.FloatTensor(ptdata)
        x_self = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        vec1m = torch.zeros(window, batch, lsize_in_pos)
        for iib in range(batch):
            ptdata = gssample_pos[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1m[:, iib, :] = torch.FloatTensor(ptdata)
        x_pos = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if ptrain_pt.gpuavail:
            outlab = outlab.to(ptrain_pt.device)
            x_self = x_self.to(ptrain_pt.device)
            x_pos = x_pos.to(ptrain_pt.device)

        return [x_self,x_pos], outlab, None

    @staticmethod
    def get_data_sup_coop_special(dataset_dict, data_shapel, pt_embl, ptrain_pt, rstartv=None, shift=False):
        """
        A Special data getter for supervised cooperation mode
        :param dataset_dict:
        :param data_shapel: shape like [window, batch, lsize_in] arranged as [word_shape, pos_shape]
        :param pt_embl: embedding list [w2v pt_emb, onehot pt_emb_pos]
        :param ptrain_pt: pytrain interface
        :param rstartv: random start
        :param shift: data shift
        :return: [word,pos], word
        """
        assert len(data_shapel[0]) == 3
        assert len(data_shapel[1]) == 3

        window, batch, lsize_in_word = data_shapel[0]
        window, batch, lsize_in_pos = data_shapel[1]

        assert ptrain_pt.digit_input
        assert ptrain_pt.supervise_mode

        labelset = dataset_dict["label"] # POS
        dataset = dataset_dict["dataset"] # Word

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))
        yl = np.zeros((batch, window))

        for iib in range(batch):
            if shift:
                yl[iib, :] = dataset[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1]
            else:
                yl[iib, :] = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)

        vec1m = torch.zeros(window, batch, lsize_in_word)
        for iib in range(batch):
            ptdata = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1 = pt_embl[0](torch.LongTensor(ptdata))
            vec1m[:, iib, :] = vec1
        x_wrd = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        vec2m = torch.zeros(window, batch, lsize_in_pos)
        for iib in range(batch):
            ptdata = labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec2 = pt_embl[1](torch.LongTensor(ptdata))
            vec2m[:, iib, :] = vec2
        x_pos = Variable(vec2m, requires_grad=True).type(torch.FloatTensor)

        if ptrain_pt.gpuavail:
            outlab = outlab.to(ptrain_pt.device)
            x_wrd = x_wrd.to(ptrain_pt.device)
            x_pos = x_pos.to(ptrain_pt.device)

        return [x_wrd,x_pos], outlab, None

    @staticmethod
    def get_data_continous(dataset, data_shape, pt_emb, ptrain_pt, rstartv=None, shift=False):

        window, batch, lsize_in = data_shape

        # Generating output label
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))
        yl = np.zeros((batch,window))
        # xl = np.zeros((batch,self.pt.window))
        for iib in range(batch):
            # xl[iib,:]=self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window]
            if shift:
                yl[iib,:] = dataset[int(rstartv[iib])+1:int(rstartv[iib]) + window+1]
            else:
                yl[iib, :] = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
        # inlab = torch.from_numpy(xl)
        # inlab = inlab.type(torch.LongTensor)
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor)
        # inlab = outlab

        vec1m = torch.zeros(window, batch, lsize_in)
        for iib in range(batch):
            # if self.pt.data_init:
            #     vec1=self.databp[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window, :]
            # else:
            # vec1=self.pt._build_databp(self.pt.dataset[int(rstartv[iib]):int(rstartv[iib]) + self.pt.window])
            ptdata = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1 = pt_emb(torch.LongTensor(ptdata))
            # vec2 = self.databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + self.window + 1, :]
            vec1m[:,iib,:]=vec1
            # vec2m[:, iib, :] = vec2
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor) #
        # y = Variable(vec2m, requires_grad=True)

        if ptrain_pt.gpuavail:
            # inlab, outlab = inlab.to(self.device), outlab.to(self.device)
            outlab = outlab.to(ptrain_pt.device)
            x = x.to(ptrain_pt.device)

        return x, outlab, None

    @staticmethod
    def get_data_sup_hsoftmax2(dataset_dict, data_shape, pt_emb, ptrain_pt, rstartv=None, shift=False):
        """
        hsoftmax_depth is 2
        :param dataset_dict:
        :param data_shape: shape like [window, batch, lsize_in]
        :param pt_emb: embedding
        :param ptrain_pt: pytrain interface
        :param rstartv: random start
        :param shift: data shift
        :return:
        """
        assert len(data_shape) == 4

        window, batch, lsize_in, output_cluster = data_shape

        assert ptrain_pt.digit_input
        assert ptrain_pt.supervise_mode

        labelset = dataset_dict["label"]
        dataset = dataset_dict["dataset"]

        # Generating output label # Generating output label # numpy random is necessary for distributed data parallel
        if rstartv is None:
            rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        # yl = np.zeros((batch, window))
        # for iib in range(batch):
        #     if shift:
        #         yl[iib, :] = labelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1]
        #     else:
        #         yl[iib, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
        # ylp1 = yl % output_cluster
        # ylp2 = np.floor(yl / output_cluster)
        # yla=np.array([ylp1,ylp2]).transpose(1,2,0)
        # outlab = torch.from_numpy(yla)
        # outlab = outlab.type(torch.LongTensor)

        yl = np.zeros((batch, window, 2))
        for iib in range(batch):
            if shift:
                yl[iib, :, :] = labelset[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1]
            else:
                yl[iib, :, :] = labelset[int(rstartv[iib]):int(rstartv[iib]) + window]
        outlab = torch.from_numpy(yl)
        outlab = outlab.type(torch.LongTensor) # (batch, window, 2)

        vec1m = torch.zeros(window, batch, lsize_in)
        for iib in range(batch):
            ptdata = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
            vec1 = pt_emb(torch.LongTensor(ptdata))
            vec1m[:, iib, :] = vec1
        x = Variable(vec1m, requires_grad=True).type(torch.FloatTensor)

        if ptrain_pt.gpuavail:
            outlab = outlab.to(ptrain_pt.device)
            x = x.to(ptrain_pt.device)

        # return x, outlab, None
        return [x,outlab[:,:,0]], outlab, None


class MyLossFun(object):

    @staticmethod
    def KNWLoss(outputl, outlab, model=None ,hsoftmax_depth=1):
        # if len(outputl.shape) == 3:
        #     # wd, batch, l_size
        #     outputl = outputl.permute(1, 2, 0)
        # else:
        #     # wd, batch, sample, l_size
        #     wd, batch, sample, l_size = outputl.shape
        #     outputl = outputl.permute(1, 3, 0, 2)
        #     outlab = outlab.view(batch, wd, 1).expand(batch, wd, sample)
        # lossc=torch.nn.CrossEntropyLoss()
        # loss1 = lossc(outputl, outlab)
        lossc = torch.nn.CrossEntropyLoss()
        if len(outputl.shape) == 3:
            # wd, batch, l_size
            outputl = outputl.permute(1, 2, 0) # batch, l_size, wd
            if hsoftmax_depth == 1:
                loss1 = lossc(outputl, outlab)
            elif hsoftmax_depth == 2:
                loss1p1 = lossc(outputl[:, :model.output_cluster, :, :], outlab[:, :, :, 0])
                loss1p2 = lossc(outputl[:, model.output_cluster:, : , :], outlab[:, :, :, 1])
                loss1 = loss1p1 + loss1p2
        else:
            # wd, batch, sample, l_size
            wd, batch, sample, l_size = outputl.shape
            outputl = outputl.permute(1, 3, 0, 2) #batch,  l_size, wd, sample

            if hsoftmax_depth == 1:
                outlab = outlab.view(batch, wd, 1).expand(batch, wd, sample)
                loss1 = lossc(outputl, outlab)
            elif hsoftmax_depth == 2:
                outlab = outlab.view(batch, wd, 1, 2).expand(batch, wd, sample, 2)
                loss1p1 = lossc(outputl[:,:model.output_cluster,:,:], outlab[:,:,:,0])
                loss1p2 = lossc(outputl[:,model.output_cluster:,:,:], outlab[:,:,:,1])
                loss1 = loss1p1 + loss1p2

        return loss1

    @staticmethod
    def KNWLoss_WeightReg(outputl, outlab, reg_lamda, model):

        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        wnorm1 = 0
        for weight in model.loss_intf:
            wnorm1 = wnorm1 + torch.mean(torch.abs(weight))
        wnorm1 = wnorm1 / len(model.loss_intf)
        return loss1 + reg_lamda * wnorm1

    @staticmethod
    def KNWLoss_EnergyReg(outputl, outlab, reg_lamda, model, balance_para=10, gate_mode_flag=True):

        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        # model.loss_intf = [self.context, self.siggate]
        context,siggate=model.loss_intf

        energyloss = 0.5*torch.mean(torch.mul(context, context))

        if gate_mode_flag and siggate is not None:
            loss_gate = torch.mean(model.sigmoid(siggate))
        else:
            loss_gate = 0

        return loss1 + reg_lamda * energyloss + reg_lamda*loss_gate/balance_para

    # @staticmethod
    # def KNWLoss_VIB(outputl, outlab, reg_lamda, model):
    #     """
    #     Loss for variational information bottleneck
    #     :param outputl:
    #     :param outlab:
    #     :param model:
    #     :param cstep:
    #     :return:
    #     """
    #
    #     outputl=outputl.permute(1, 2, 0)
    #     lossc=torch.nn.CrossEntropyLoss()
    #     loss1 = lossc(outputl, outlab)
    #
    #     # self.loss_intf = [self.ctheta, self.cmu]
    #     ctheta,cmu=model.loss_intf
    #
    #     gausskl=0.5*torch.mean(torch.sum(ctheta**2+cmu**2-torch.log(ctheta**2)-1,dim=-1))
    #
    #     return loss1 + reg_lamda*gausskl

    @staticmethod
    def KNWLoss_VIB(outputl, outlab, reg_lamda, model, cstep, beta_warmup, scale_factor=0.1):
        """
        Loss for variational information bottleneck, Gauss Expanded
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """

        # if model.multi_sample_flag:
        #     # outputl = torch.exp(outputl)
        #     # outputl = torch.mean(outputl, dim=-2)
        #     # outputl[outputl<=0]=1e-9
        #     # outputl=torch.log(outputl)
        #     # if (outputl != outputl).any():
        #     #     raise Exception("NaN Error")
        #     w, b, s, l = outputl.shape
        #     outlab = outlab.view((b, w, 1)).expand((b, w, s))
        #     outputl = outputl.permute(1, 3, 0, 2)
        # else:
        # outputl=outputll[0]
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        # self.loss_intf = [self.ctheta, self.cmu]
        ctheta, cmu = model.loss_intf
        dw, batch, l_size = ctheta.shape
        # prior = prior.view(1,1,-1).expand(dw, batch, l_size)

        # ent_prior = torch.mean(torch.log(prior*np.sqrt(2*np.pi*np.e)))
        # ent_prior = torch.mean(prior * np.sqrt(2 * np.pi * np.e))

        gausskl = 0.5 * torch.mean(torch.sum(ctheta ** 2 + cmu ** 2 - torch.log(ctheta ** 2) - 1, dim=-1))
        assert gausskl>0

        # gausskl = 0.5 * torch.mean(torch.sum(ctheta ** 2 / prior**2 + cmu ** 2 / prior**2 - torch.log(ctheta ** 2 / prior**2) - 1, dim=-1))

        # beta_scaling = 1.0-np.exp(-cstep * 5)
        if cstep<beta_warmup:
            beta_scaling = cstep/beta_warmup
        else:
            beta_scaling = 1.0

        # return loss1 + beta_scaling * reg_lamda * (gausskl + scale_factor*ent_prior)
        return loss1 + beta_scaling * reg_lamda * gausskl

    @staticmethod
    def KNWLoss_GSVIB(outputl, outlab, reg_lamda, model, cstep, beta_warmup, scale_factor):
        """
        Loss for variational information bottleneck, Gauss Expanded
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        lossc = torch.nn.CrossEntropyLoss()
        if len(outputl.shape) == 3:
            # wd, batch, l_size
            outputl = outputl.permute(1, 2, 0)  # batch, l_size, wd
            if getattr(model,"hsoftmax_depth",1) == 1:
                loss1 = lossc(outputl, outlab)
            elif getattr(model,"hsoftmax_depth",1) == 2:
                loss1p1 = lossc(outputl[:,:model.output_cluster, :, : ], outlab[:, :, :, 0])
                loss1p2 = lossc(outputl[:,model.output_cluster:, :, :], outlab[:, :, :, 1])
                loss1 = loss1p1 + loss1p2
        else:
            # wd, batch, sample, l_size
            wd, batch, sample, l_size = outputl.shape
            outputl = outputl.permute(1, 3, 0, 2)  # batch,  l_size, wd, sample

            if getattr(model,"hsoftmax_depth",1) == 1:
                outlab = outlab.view(batch, wd, 1).expand(batch, wd, sample)
                loss1 = lossc(outputl, outlab)
            elif getattr(model,"hsoftmax_depth",1) == 2:
                outlab = outlab.view(batch, wd, 1, 2).expand(batch, wd, sample, 2)
                loss1p1 = lossc(outputl[:, :model.output_cluster , :, :], outlab[:, :, :, 0])
                loss1p2 = lossc(outputl[:, model.output_cluster:, :, :], outlab[:, :, :, 1])
                loss1 = loss1p1 + loss1p2

        # context,prior should be log probability
        context,prior = model.loss_intf
        gs_head, context_size = prior.shape

        ent_prior=-torch.mean(torch.sum(torch.exp(prior)*prior,dim=-1))

        prior=prior.view(1,1,gs_head,context_size).expand_as(context)

        flatkl = torch.mean(torch.sum(torch.exp(context)*(context-prior),dim=-1))

        # beta_scaling = 1.0-np.exp(-cstep * 5)
        # if cstep < beta_warmup:
        #     beta_scaling = cstep / beta_warmup
        # else:
        #     beta_scaling = 1.0
        beta_scaling = 1.0
        # print(loss1,beta_scaling,reg_lamda,flatkl,scale_factor,ent_prior)
        return loss1 + beta_scaling * reg_lamda * (flatkl + scale_factor*ent_prior)

    @staticmethod
    def KNWLoss_GSVIB_HSOFTMAX2(outputl, outlab, reg_lamda, model, cstep, beta_warmup, scale_factor=0.1):
        """
        Loss for variational information bottleneck, Hierarchical softmax
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        lossc = torch.nn.CrossEntropyLoss()
        if len(outputl.shape) == 3:
            # wd, batch, l_size
            outputl = outputl.permute(1, 2, 0)  # batch, l_size, wd
            loss1p1 = lossc(outputl[0], outlab[:, :, 0])
            loss1p2 = lossc(outputl[1], outlab[:, :, 1])
            loss1 = loss1p1 + loss1p2
        else:
            # wd, batch, sample, l_size

            wd, batch, sample, l_size = outputl.shape
            outputl = outputl.permute(1, 3, 0, 2)  # batch,  l_size, wd, sample

            outlab = outlab.view(batch, wd, 1, 2).expand(batch, wd, sample, 2)
            loss1p1 = lossc(outputl[0], outlab[:, :, :, 0])
            loss1p2 = lossc(outputl[1], outlab[:, :, :, 1])
            loss1 = loss1p1 + loss1p2

        # context,prior should be log probability
        context, prior = model.loss_intf
        gs_head, context_size = prior.shape

        ent_prior = -torch.mean(torch.sum(torch.exp(prior) * prior, dim=-1))

        prior = prior.view(1, 1, gs_head, context_size).expand_as(context)

        flatkl = torch.mean(torch.sum(torch.exp(context) * (context - prior), dim=-1))

        # beta_scaling = 1.0-np.exp(-cstep * 5)
        # if cstep < beta_warmup:
        #     beta_scaling = cstep / beta_warmup
        # else:
        #     beta_scaling = 1.0
        beta_scaling = 1.0

        return loss1 + beta_scaling * reg_lamda * (flatkl + scale_factor * ent_prior)

    @staticmethod
    def KNWLoss_GSVIB_ATTCOOP(outputl, outlab, reg_lamda, model, cstep, beta_warmup, scale_factor=0.1,scale_factor2=0.0):
        """
        Loss for variational information bottleneck, Gauss Expanded
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        # context,prior should be log probability
        context, p_prior, attention_prob = model.context, model.p_prior, model.attention_prob
        # context [w,b,gs_head_num,gs_head_dim]
        # p_prior [w,b,gs_head_num,gs_head_dim] expanded from [gs_head_num,gs_head_dim]
        # attention_prob [w, b, gs_head_num]
        gs_head_num, gs_head_dim = model.gs_head_num, model.gs_head_dim
        w, b, gs_head_num=attention_prob.shape

        # prior [gs_head_num,gs_head_dim]
        # gate_prob [win batch gs_head_num]

        # prior = p_prior.view(1, 1, gs_head_num, gs_head_dim).expand_as(context)
        ent_prior = -torch.mean(torch.sum(torch.exp(p_prior) * p_prior, dim=-1))
        # ent_prior = -torch.mean(attention_prob * torch.sum(torch.exp(p_prior) * p_prior, dim=-1))

        # attention_prob entropy
        # ent_attention_prob= -torch.mean(attention_prob*torch.log(attention_prob)+(1-attention_prob)*torch.log((1-attention_prob)))


        # flatkl = torch.mean(torch.sum(torch.exp(context) * (context - p_prior), dim=-1))
        flatkl = torch.mean(torch.sum(attention_prob.view(w, b, gs_head_num, 1) * torch.exp(context) * (context - p_prior), dim=-1))

        att_gate = torch.mean(attention_prob)

        # beta_scaling = 1.0-np.exp(-cstep * 5)
        # if cstep < beta_warmup:
        #     beta_scaling = cstep / beta_warmup
        # else:
        #     beta_scaling = 1.0
        beta_scaling = 1.0
        return loss1 + beta_scaling * reg_lamda * (flatkl + scale_factor * ent_prior + scale_factor2 * att_gate)

    @staticmethod
    def KNWLoss_SIGVIB(outputl, outlab, reg_lamda, model, scale_factor=0.1):
        """
        Loss for variational information bottleneck, 2 value neurons
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        # context,prior should be log probability
        context, prior = model.loss_intf
        coopout_size,gs_head_num = prior.shape

        eps = 1e-7
        ent_prior = -torch.mean( prior*torch.log(prior + eps) + (1.0-prior)*torch.log(1.0 - prior + eps))

        prior = prior.view(1, 1, coopout_size,gs_head_num).expand_as(context)

        flatkl = torch.mean( context * (torch.log(context+ eps) - torch.log(prior+ eps)) +
                            (1.0-context) * (torch.log(1.0-context+ eps) - torch.log(1.0-prior+ eps))
                            )

        if (prior != prior).any():
            raise Exception("NaN Error in prior")
        if (ent_prior != ent_prior).any():
            raise Exception("NaN Error in ent_prior")
        if (flatkl != flatkl).any():
            raise Exception("NaN Error in flatkl")

        return loss1 + reg_lamda * (flatkl + scale_factor * ent_prior)

    @staticmethod
    def KNWLoss_infofork(outputl, outlab, reg_lamda, model, scale_factor=0.1, scale_factor2=0.1):
        """
        Loss for variational information bottleneck, Gauss Expanded
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)

        # context,prior should be log probability
        # context, p_prior, attention_prob = model.context, model.p_prior, model.attention_prob
        context, attention_prob = model.context, model.attention_prob
        # context [w,b,gs_head_num,gs_head_dim]
        # p_prior [w,b,gs_head_num,gs_head_dim] expanded from [gs_head_num,gs_head_dim]
        # attention_prob [w, b, gs_head_num]
        gs_head_num, gs_head_dim = model.gs_head_num, model.gs_head_dim
        w, b, gs_head_num = attention_prob.shape

        # prior [gs_head_num,gs_head_dim]
        # gate_prob [win batch gs_head_num]

        # prior = p_prior.view(1, 1, gs_head_num, gs_head_dim).expand_as(context)
        # ent_prior = -torch.mean(torch.sum(torch.exp(p_prior) * p_prior, dim=-1))
        # ent_prior = -torch.mean(attention_prob * torch.sum(torch.exp(p_prior) * p_prior, dim=-1))

        # attention_prob entropy
        # ent_attention_prob= -torch.mean(attention_prob*torch.log(attention_prob)+(1-attention_prob)*torch.log((1-attention_prob)))

        # flatkl = torch.mean(torch.sum(torch.exp(context) * (context - p_prior), dim=-1))

        # p_prior = p_prior.view(1, 1, gs_head_num, gs_head_dim).expand_as(context)

        # flatkl = torch.mean( torch.sum(torch.exp(context) * (context - p_prior), dim=-1))

        att_gate = torch.mean(attention_prob)

        # beta_scaling = 1.0-np.exp(-cstep * 5)
        # if cstep < beta_warmup:
        #     beta_scaling = cstep / beta_warmup
        # else:
        #     beta_scaling = 1.0
        beta_scaling = 1.0

        # loss_res = loss1 + beta_scaling * reg_lamda * (flatkl + scale_factor * ent_prior + scale_factor2 * att_gate)

        loss_res = loss1 +  reg_lamda *  att_gate

        return loss_res

    @staticmethod
    def KNWLoss_VIB_EVAL(outputl, outlab, model):
        # outputl[w,b,s,l] outlab[b,w]
        # if model.multi_sample_flag:
        #     # outputl = torch.exp(outputl)
        #     # outputl = torch.mean(outputl, dim=-2)
        #     # outputl[outputl <= 0] = 1e-9
        #     # outputl = torch.log(outputl)
        #     # if (outputl != outputl).any():
        #     #     raise Exception("NaN Error")
        #     w, b, s, l = outputl.shape
        #     outlab=outlab.view((b, w, 1)).expand(( b, w, s))
        #     outputl = outputl.permute(1, 3, 0, 2)
        # else:
        # outputl = outputll[0]
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        return loss1

    @staticmethod
    def KNWLoss_withmask(outputl, outlab, model=None, cstep=None, data_loss_mask=None):
        """
        transformer style masked loss
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        outputl=outputl.permute(1, 2, 0)
        lossc=torch.nn.CrossEntropyLoss(reduce=False)
        loss1 = lossc(outputl, outlab)
        if data_loss_mask is not None:
            # comb_mask = model.input_mask * data_loss_mask
            comb_mask = model.input_mask + data_loss_mask
            comb_mask[comb_mask > 0.999] = 1
        else:
            comb_mask = model.input_mask
        loss=torch.sum(loss1*(1-comb_mask.permute(1,0)))/torch.sum(1-comb_mask.permute(1,0))
        # loss = torch.sum(loss1 * model.input_mask) / torch.sum(model.input_mask) # should be perfect information
        return loss

    @staticmethod
    def KNWLoss_withmask_VIB(outputl, outlab, reg_lamda, model, cstep=None):
        """
        transformer style masked loss, VIB version
        :param outputl:
        :param outlab:
        :param model:
        :param cstep:
        :return:
        """
        outputl = outputl.permute(1, 2, 0)
        lossc = torch.nn.CrossEntropyLoss(reduce=False)
        loss1 = lossc(outputl, outlab)
        loss1 = torch.sum(loss1 * (1 - model.input_mask)) / torch.sum(1 - model.input_mask)
        # loss = torch.sum(loss1 * model.input_mask) / torch.sum(model.input_mask) # should be perfect information

        gausskl = 0.5 * torch.mean(torch.sum(model.ctheta ** 2 + model.cmu ** 2 - torch.log(model.ctheta ** 2) - 1, dim=-1))

        return loss1 + reg_lamda * gausskl


### Distributed DataParallel

def dist_setup(rank, world_size, seed):
    """
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    :return:
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)

def dist_cleanup():
    dist.destroy_process_group()

# def demo_basic(rank, world_size):
#     setup(rank, world_size)
#
#     # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
#     # rank 2 uses GPUs [4, 5, 6, 7].
#     n = torch.cuda.device_count() // world_size
#     device_ids = list(range(rank * n, (rank + 1) * n))
#
#     # create model and move it to device_ids[0]
#     model = ToyModel().to(device_ids[0])
#     # output_device defaults to device_ids[0]
#     ddp_model = DDP(model, device_ids=device_ids)
#
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
#
#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(device_ids[0])
#     loss_fn(outputs, labels).backward()
#     optimizer.step()
#
#     cleanup()

def run_training_worker(rank, world_size, datafile, lsize, rnn, interface_para, batch, window, para, run_para, model_name):

    num_epoch, learning_rate, step_per_epoch, print_step, seed = run_para
    dist_setup(rank, world_size, seed)

    para["cuda_device"] = rank
    para["dist_data_parallel"] = True
    dataset = load_data(datafile)
    pt1 = PyTrain_Custom(dataset, lsize, rnn, interface_para, batch=batch, window=window, para=para)
    pt1.run_training(epoch = num_epoch, lr = learning_rate,step_per_epoch = step_per_epoch, print_step = print_step)
    log_name="log"+str(rank)+".txt"
    f = open(log_name, "a")
    f.write("\n\nLog for this run\n")
    f.write(pt1.log)
    f.close()

    if rank==0:
        # save_model(rnn,"bigru_pos"+str(rank)+".model")
        save_model(rnn, model_name)

    dist_cleanup()


def dist_run_training(device_ids, dataset, lsize, rnn, interface_para, batch=20, window=30, para=None,run_para=[30,1e-3,2000,500,12345],model_name="bigru.model"):
    world_size=len(device_ids)
    datafile="dataset_temp"
    try:
        f = open(datafile)
        f.close()
    except IOError:
        save_data(dataset, datafile, large_data=True)

    mp.spawn(run_training_worker,
             args=(world_size, datafile, lsize, rnn, interface_para, batch, window, para,run_para,model_name),
             nprocs=world_size,
             join=True)
    # subprocess.run(["rm", datafile])
