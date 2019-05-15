from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import time, os, pickle

import torch
import copy
from torch.autograd import Variable

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker

from wordcloud import WordCloud
import operator
from PIL import Image
from PIL import ImageDraw,ImageFont

from tqdm import tqdm

def free_gen(step,lsize,rnn, id_2_vec=None, id_to_word=None,prior=None):
    """
    Free generation
    :param step:
    :param lsize:
    :param rnn:
    :param id_2_vec:
    :return:
    """
    print("Start Evaluation ...")
    startt = time.time()
    if type(lsize) is list:
        lsize_in = lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize

    hidden = rnn.initHidden(1)
    x = torch.zeros(1, 1, lsize_in)
    outputl = []
    outwrdl=[]
    outpl=[]
    hiddenl=[]
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

    for iis in range(step):
        x= x.type(torch.FloatTensor)
        output, hidden = rnn(x, hidden)    ############ rnn
        ynp = output.data.numpy().reshape(lsize_out)
        rndp = np.random.rand()
        pii = logp(ynp).reshape(-1)
        dig = 0
        for ii in range(len(pii)):
            rndp = rndp - pii[ii]
            if rndp < 0:
                dig = ii
                break
        xword = id_to_word[dig]
        outputl.append(dig)
        outwrdl.append(xword)
        # hiddenl.append(hout.cpu().data.numpy().reshape(-1))
        if prior is None:
            outpl.append(pii[0:200])
        else:
            outpl.append(pii[0:200]/prior[0:200])
        # clul.append(np.exp(clu.data.numpy().reshape(30)))
        xvec = id_2_vec[dig]
        x = torch.from_numpy(xvec.reshape((1, 1, lsize_in)))
    endt = time.time()
    print("Time used in generation:", endt - startt)
    return outputl,outwrdl,outpl,hiddenl #clul

def do_eval(dataset,lsize,rnn, id_2_vec=None, seqeval=False):
    print("Start Evaluation ...")
    rnn.eval()
    startt = time.time()
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    datab=[]
    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec=np.array(id_2_vec[data])
            datab.append(datavec)
    databpt=torch.from_numpy(np.array(datab))
    databpt = databpt.type(torch.FloatTensor)
    hidden = rnn.initHidden(1)
    hiddenl = []
    if not seqeval:
        outputl = []
        for iis in range(len(databpt) - 1):
            x = databpt[iis, :].view(1, 1, lsize_in)
            output, hidden = rnn(x, hidden)
            outputl.append(output.view(-1).data.numpy())
            hiddenl.append(hidden)
        outputl = np.array(outputl)
        outputl = Variable(torch.from_numpy(outputl).contiguous())
        outputl = outputl.permute((1, 0))
        print(outputl.shape)
    else:
        # LSTM/GRU provided whole sequence training
        dlen=len(databpt)
        x = databpt[0:dlen-1, :].view(dlen-1, 1, lsize_in)
        outputl, hidden = rnn(x, hidden)
        outputl=outputl.permute(1, 2, 0)

    # Generating output label
    yl = []
    for iiss in range(len(dataset) - 1):
        ylb = []
        wrd = dataset[iiss + 1]
        ylb.append(wrd)
        yl.append(np.array(ylb))
    outlab = torch.from_numpy(np.array(yl).T)
    outlab = outlab.type(torch.LongTensor)
    lossc = torch.nn.CrossEntropyLoss()
    loss = lossc(outputl.view(1, -1, len(dataset) - 1), outlab)
    print("Evaluation Perplexity: ", np.exp(loss.item()))
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return outputl, hiddenl, outlab.view(-1)

def do_eval_p(dataset,lsize,rnn, id_2_vec=None, prior=None):
    """
    General evaluation function
    :param dataset:
    :param lsize:
    :param rnn:
    :return:
    """
    print("Start Evaluation ...")
    rnn.eval()
    startt = time.time()
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    datab=[]
    if id_2_vec is None: # No embedding, one-hot representation
        for data in dataset:
            datavec=np.zeros(lsize_in)
            datavec[data]=1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec=np.array(id_2_vec[data])
            datab.append(datavec)
    databpt=torch.from_numpy(np.array(datab))
    databpt = databpt.type(torch.FloatTensor)
    hidden = rnn.initHidden(1)

    perpls=[0.0]
    for nn in range(len(databpt) - 1):
        x = databpt[nn]
        y = np.zeros(lsize_out)
        y[dataset[nn+1]]=1
        if prior is not None:
            perp = cal_kldiv(y, prior)
        else:
            prd, hidden = rnn.forward(x.view(1, 1, lsize_in), hidden)
            prd = torch.exp(prd) / torch.sum(torch.exp(prd))
            perp = cal_kldiv(y, prd.view(-1).data.numpy())
        perpls.append(perp)
    avperp = np.mean(np.array(perpls))
    print("Calculated knowledge perplexity:", np.exp(avperp))
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return perpls

def do_eval_rnd(dataset,lsize, rnn, step, window=30, batch=20, id_2_vec=None, para=None):
    """
    General evaluation function using stochastic batch evaluation on GPU
    :param dataset:
    :param lsize:
    :param rnn:
    :param extend_mode: for concept to word extension mode, extend_mode[0] is extension dataset, extend_mode[1] is extension matrix.
    :return:
    ### Data definition
    perpl: 1-D list of calculated perplexity
    """
    print("Start Evaluation ...")
    startt = time.time()

    rnn.eval()

    if para is None:
        para=dict([])
    seqeval = para.get("seqeval", False)
    extend_mode= para.get("extend_mode", None)
    supervise_mode = para.get("supervise_mode", False)
    pre_training = para.get("pre_training", False)
    cuda_flag = para.get("cuda_flag", True)
    digit_input = para.get("digit_input", True)

    if (type(dataset) is dict) != supervise_mode:
        raise Exception("Supervise mode Error.")

    if supervise_mode:
        label=dataset["label"]
        dataset=dataset["dataset"]

    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    datab=[]
    if digit_input:
        if id_2_vec is None: # No embedding, one-hot representation
            for data in dataset:
                datavec=np.zeros(lsize_in)
                datavec[data]=1
                datab.append(datavec)
        else:
            for data in dataset:
                datavec=np.array(id_2_vec[data])
                datab.append(datavec)
    else:
        datab = dataset

    databpt=torch.from_numpy(np.array(datab))
    databpt = databpt.type(torch.FloatTensor)

    if cuda_flag:
        gpuavail = torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
    else:
        gpuavail=False
        device = torch.device("cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    rnn=rnn.to(device)

    lossc = torch.nn.CrossEntropyLoss()
    perpl=[]
    outlabl=[]
    inputlabl = []
    conceptl=[]
    outputll=[]

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)


        if not supervise_mode:
            # Generating output label
            if extend_mode is None:
                extdataset = dataset
            else:
                extdataset = extend_mode[0]
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = extdataset[(int(rstartv[iib]) + iiss + 1)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)
        else:
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = label[(int(rstartv[iib]) + iiss)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)

        # LSTM/GRU provided whole sequence training
        if seqeval:
            vec1m = None
            inputlabsub=[]
            for iib in range(batch):
                vec1 = databpt[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                if type(vec1m) == type(None):
                    vec1m = vec1.view(window, 1, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                inputlab = dataset[int(rstartv[iib]):int(rstartv[iib]) + window]
                inputlabsub.append(inputlab)
            x = vec1m  #
            x = x.type(torch.FloatTensor)
            if gpuavail:
                outlab = outlab.to(device)
                x = x.to(device)
            if extend_mode is None:
                if pre_training:
                    outputl, hidden = rnn.pre_training(x, hidden, schedule=1.0)
                    conceptl.append(outputl.cpu().data.numpy())
                else:
                    outputl, hidden = rnn(x, hidden, schedule=1.0)
                    try:
                        conceptl.append(rnn.infer_pos.cpu().data.numpy())
                    except:
                        pass
            else:
                npM_ext=extend_mode[1]
                outputl, hidden = rnn.forward_concept_ext(x, hidden, npM_ext)

            outlabl.append(outlab.transpose(0,1).cpu().data.numpy())
            inputlabl.append(np.array(inputlabsub).T)
            # conceptl.append(rnn.concept_layer_i.cpu().data.numpy())
            # conceptl.append(rnn.hout2con_masked.cpu().data.numpy())
            outputll.append(outputl.cpu().data.numpy())
        else:
            outputl = None
            for iiss in range(window):
                vec1m = None
                vec2m = None
                for iib in range(batch):
                    vec1 = databpt[(int(rstartv[iib]) + iiss), :]
                    vec2 = databpt[(int(rstartv[iib]) + iiss + 1), :]
                    if vec1m is None:
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                # One by one guidance training ####### error can propagate due to hidden state
                x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    outlab = outlab.to(device)
                    x, y = x.to(device), y.to(device)
                if pre_training:
                    output, hidden = rnn.pre_training(x, hidden, schedule=iis / step)
                else:
                    output, hidden = rnn(x, hidden, schedule=iis / step)
                if type(outputl) == type(None):
                    outputl = output.view(1, batch,lsize_out)
                else:
                    outputl = torch.cat((outputl.view(-1, batch, lsize_out), output.view(1, batch,lsize_out)), dim=0)

        loss = lossc(outputl.permute(1,2,0), outlab)
        perpl.append(loss.item())

    print("Evaluation Perplexity: ", np.exp(np.mean(np.array(perpl))))
    endt = time.time()
    print("Time used in evaluation:", endt - startt)
    return perpl,outputll,np.array(conceptl),np.array(inputlabl),np.array(outlabl)

def lossf_rms(output, target):
    """
    Root mean square loss function
    :param input:
    :param output:
    :return:
    """
    lossc = torch.nn.MSELoss()
    # MSELoss is calculating dimension 1
    loss=lossc(torch.t(output), torch.t(target))
    return loss

def run_training_univ(dataset,lsize, model, lossf,step,learning_rate=1e-2, batch=20, save=None):
    """
    Trial of universal training function
    :param dataset:
    :param lsize:
    :param model:
    :param lossf: loss function
    :param step:
    :param learning_rate:
    :param batch:
    :return:
    """
    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()

    train_hist = []
    his = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    dataset=torch.from_numpy(dataset)
    dataset=dataset.type(torch.FloatTensor)

    for iis in range(step):
        rstartv = np.floor(np.random.rand(batch) * len(dataset))
        input=None
        for iib in range(batch):
            vec = dataset[int(rstartv[iib]), :]
            if input is None:
                input = vec.view(1, -1)
            else:
                input = torch.cat((input, vec.view(1, -1)), dim=0)

        output=model(input,iter=10)

        loss=lossf(output,input)

        train_hist.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if int(iis / prtstep) != his:
            print("Lost: ", iis, loss.item())
            his = int(iis / prtstep)

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

    return model


def run_training(dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, id_2_vec=None, para=None):
    """
    General rnn training funtion for one-hot training
    :param dataset:
    :param lsize:
    :param model:
    :param step:
    :param learning_rate:
    :param batch:
    :param window:
    :param save:
    :param seqtrain:
    :param coop: a cooperational rnn unit
    :param coopseq: a pre-calculated cooperational logit vec
    :return:
    """
    if para is None:
        para=dict([])
    save= para.get("save",None)
    seqtrain = para.get("seqtrain", False)
    supervise_mode = para.get("supervise_mode", False)
    coop= para.get("coop", None)
    coopseq = para.get("coopseq", None)
    cuda_flag = para.get("cuda_flag", True)
    invec_noise=para.get("invec_noise", 0.0)
    pre_training=para.get("pre_training",False)
    loss_clip=para.get("loss_clip",0.0)
    digit_input=para.get("digit_input",True)
    two_step_training = para.get("two_step_training", False)

    if type(lsize) is list:
        lsize_in=lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()
    datab=[]

    if (type(dataset) is dict) != supervise_mode:
        raise Exception("Supervise mode Error.")

    if supervise_mode:
        label=dataset["label"]
        dataset=dataset["dataset"]

    if digit_input:
        if id_2_vec is None: # No embedding, one-hot representation
            for data in dataset:
                datavec=np.zeros(lsize_in)
                datavec[data]=1.0
                datab.append(datavec)
        else:
            for data in dataset:
                datavec=np.array(id_2_vec[data])
                datab.append(datavec)
    else: # if not digit input, raw data_set is used
        datab=dataset
    databp=torch.from_numpy(np.array(datab))
    if coopseq is not None:
        coopseq=torch.from_numpy(np.array(coopseq))
        coopseq=coopseq.type(torch.FloatTensor)

    rnn.train()

    if coop is not None:
        coop.eval() # Not ensured to work !!!

    def custom_KNWLoss(outputl, outlab, model, cstep):
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        # logith2o = model.h2o.weight
        # pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1 #+ 0.001 * l1_reg #+ 0.01 * lossh2o  + 0.01 * l1_reg

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0.0)
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    train_hist = []
    his = 0

    if cuda_flag:
        gpuavail = torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
    else:
        gpuavail=False
        device = torch.device("cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    if gpuavail:
        rnn.to(device)

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)

        if not supervise_mode:
            # Generating output label
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = dataset[(int(rstartv[iib]) + iiss + 1)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)
        else:
            yl = []
            for iiss in range(window):
                ylb = []
                for iib in range(batch):
                    wrd = label[(int(rstartv[iib]) + iiss)]
                    ylb.append(wrd)
                yl.append(np.array(ylb))
            outlab = torch.from_numpy(np.array(yl).T)
            outlab = outlab.type(torch.LongTensor)


        # step by step training
        if not seqtrain:
            outputl = None
            for iiss in range(window):
                vec1m = None
                vec2m = None
                if coopseq is not None:
                    veccoopm = None
                for iib in range(batch):
                    vec1 = databp[(int(rstartv[iib]) + iiss), :]
                    vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
                    if vec1m is None:
                        vec1m = vec1.view(1, -1)
                        vec2m = vec2.view(1, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
                        vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
                    if coopseq is not None:
                        veccoop=coopseq[(int(rstartv[iib]) + iiss +1), :]
                        if veccoopm is None:
                            veccoopm = veccoop.view(1, -1)
                        else:
                            veccoopm = torch.cat((veccoopm, veccoop.view(1, -1)), dim=0)
                # One by one guidance training ####### error can propagate due to hidden state
                x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
                y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    outlab = outlab.to(device)
                    x, y = x.to(device), y.to(device)
                if coop is not None:
                    outputc, hiddenc = coop(x, hidden=None,logitmode=True)
                    output, hidden = rnn(x, hidden, add_logit=outputc)
                elif coopseq is not None:
                    output, hidden = rnn(x, hidden, add_logit=veccoopm)
                else:
                    # output, hidden = rnn(x, hidden,wta_noise=1.0+0.0*(1.0-iis/step))
                    output, hidden = rnn(x, hidden)
                    # output = rnn.pre_training(x)
                if type(outputl) == type(None):
                    outputl = output.view(batch, lsize_out, 1)
                else:
                    outputl = torch.cat((outputl.view(batch, lsize_out, -1), output.view(batch, lsize_out, 1)), dim=2)
            loss = custom_KNWLoss(outputl, outlab, rnn, iis)
            # if gpuavail:
            #     del outputl,outlab
            #     torch.cuda.empty_cache()
        else:
            # LSTM/GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1_raw = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
                vec1_rnd=torch.rand(vec1_raw.shape)
                vec1_add=torch.mul((1.0-vec1_raw)*invec_noise,vec1_rnd.double())
                vec1=vec1_raw+vec1_add
                # vec1 = databp[int(rstartv[iib]):int(rstartv[iib])+window, :]
                vec2 = databp[int(rstartv[iib])+1:int(rstartv[iib])+window+1, :]
                if type(vec1m) == type(None):
                    vec1m = vec1.view(window, 1, -1)
                    vec2m = vec2.view(window, 1, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                    vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
            x = Variable(vec1m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)  #
            y = Variable(vec2m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if gpuavail:
                outlab = outlab.to(device)
                x, y = x.to(device), y.to(device)
            if pre_training:
                output, hidden = rnn.pre_training(x, hidden, schedule=iis / step)
            else:
                scheduled= iis / step
                output, hidden = rnn(x, hidden, schedule=scheduled)
            if two_step_training:
                output_twostep=rnn.auto_encode(y, schedule=scheduled)
            # output, hidden = rnn(x, hidden, wta_noise=0.2 * (1.0 - iis / step))
            loss = custom_KNWLoss(output.permute(1,2,0), outlab, rnn, iis)
            # if gpuavail:
            #     del x,y,outlab
            #     torch.cuda.empty_cache()
            if two_step_training:
                loss_twostep = custom_KNWLoss(output_twostep.permute(1, 2, 0), outlab, rnn, iis)
                loss=0.8*loss+0.2*loss_twostep


        if int(iis / prtstep) != his:
            print("Perlexity: ", iis, np.exp(loss.item()))
            his = int(iis / prtstep)

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
            if loss_clip>0:
                plt.ylim((0, loss_clip))
            plt.show()
    except:
        pass
    if gpuavail:
        torch.cuda.empty_cache()
    return rnn

def run_training_stack(dataset, lsize, rnn, step, learning_rate=1e-2, batch=20, window=30, save=None, seqtrain=False,
                 coop=None, coopseq=None, id_2_vec=None):
    """
    General rnn training funtion for one-hot training
    Not working yet !!!!!!
    :param dataset:
    :param lsize:
    :param model:
    :param step:
    :param learning_rate:
    :param batch:
    :param window:
    :param save:
    :param seqtrain:
    :param coop: a cooperational rnn unit
    :param coopseq: a pre-calculated cooperational logit vec
    :return:
    """
    rnn.train()
    if type(lsize) is list:
        lsize_in = lsize[0]
        lsize_out = lsize[1]
    else:
        lsize_in = lsize
        lsize_out = lsize
    prtstep = int(step / 10)
    startt = time.time()
    datab = []
    if id_2_vec is None:  # No embedding, one-hot representation
        for data in dataset:
            datavec = np.zeros(lsize_in)
            datavec[data] = 1
            datab.append(datavec)
    else:
        for data in dataset:
            datavec = np.array(id_2_vec[data])
            datab.append(datavec)
    databp = torch.from_numpy(np.array(datab))
    if coopseq is not None:
        coopseq = torch.from_numpy(np.array(coopseq))
        coopseq = coopseq.type(torch.FloatTensor)

    rnn.train()

    if coop is not None:
        coop.eval()  # Not ensured to work !!!

    def custom_KNWLoss(outputl, outlab, model, cstep):
        lossc = torch.nn.CrossEntropyLoss()
        loss1 = lossc(outputl, outlab)
        logith2o = model.h2o.weight
        pih2o = torch.exp(logith2o) / torch.sum(torch.exp(logith2o), dim=0)
        # lossh2o = -torch.mean(torch.sum(pih2o * torch.log(pih2o), dim=0))
        # l1_reg = model.h2o.weight.norm(2)
        return loss1  # + 0.01 * lossh2o  + 0.01 * l1_reg

    def custom_StackLoss(perpl):
        """
        Special cost for stack FF mode
        :param perpl:
        :param stackl:
        :param input_size:
        :return:
        """
        loss = torch.mean(perpl)
        return loss

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

    train_hist = []
    his = 0

    gpuavail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpuavail else "cpu")
    # If we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    if gpuavail:
        rnn.to(device)

    for iis in range(step):

        rstartv = np.floor(np.random.rand(batch) * (len(dataset) - window - 1))

        if gpuavail:
            hidden = rnn.initHidden_cuda(device, batch)
        else:
            hidden = rnn.initHidden(batch)

        # Generating output label
        yl = []
        for iiss in range(window):
            ylb = []
            for iib in range(batch):
                wrd = dataset[(int(rstartv[iib]) + iiss + 1)]
                ylb.append(wrd)
            yl.append(np.array(ylb))
        outlab = Variable(torch.from_numpy(np.array(yl).T))
        outlab = outlab.type(torch.LongTensor)

        # step by step training
        if not seqtrain:
            outputl = None
            for iiss in range(window):
                # print("iiss",iiss)
            #     vec1m = None
            #     vec2m = None
            #     if coopseq is not None:
            #         veccoopm = None
            #     for iib in range(batch):
            #         vec1 = databp[(int(rstartv[iib]) + iiss), :]
            #         vec2 = databp[(int(rstartv[iib]) + iiss + 1), :]
            #         if vec1m is None:
            #             vec1m = vec1.view(1, -1)
            #             vec2m = vec2.view(1, -1)
            #         else:
            #             vec1m = torch.cat((vec1m, vec1.view(1, -1)), dim=0)
            #             vec2m = torch.cat((vec2m, vec2.view(1, -1)), dim=0)
            #         if coopseq is not None:
            #             veccoop = coopseq[(int(rstartv[iib]) + iiss + 1), :]
            #             if veccoopm is None:
            #                 veccoopm = veccoop.view(1, -1)
            #             else:
            #                 veccoopm = torch.cat((veccoopm, veccoop.view(1, -1)), dim=0)
            #     # One by one guidance training ####### error can propagate due to hidden state
            #     x = Variable(vec1m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)  #
            #     y = Variable(vec2m.reshape(1, batch, lsize_in).contiguous(), requires_grad=True)

                vec1m = None
                for iib in range(batch):
                    vec1 = databp[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                    if type(vec1m) == type(None):
                        vec1m = vec1.view(1,window, -1)
                    else:
                        vec1m = torch.cat((vec1m, vec1.view(1, window, -1)), dim=0)
                vec1m=vec1m.type(torch.FloatTensor)
                output, hidden = rnn(vec1m, hidden)
                if type(outputl) == type(None):
                    try:
                        outputl = output.view(batch, 1)
                    except:
                        pass
                elif output is not None:
                    outputl = torch.cat((outputl.view(-1), output.view(-1)), dim=-1)
            # loss = custom_KNWLoss(outputl, outlab, rnn, iis)

            # print("Before loss:",outputl)
            loss = custom_StackLoss(outputl)
            # if gpuavail:
            #     del outputl,outlab
            #     torch.cuda.empty_cache()
        else:
            # LSTM/GRU provided whole sequence training
            vec1m = None
            vec2m = None
            for iib in range(batch):
                vec1 = databp[int(rstartv[iib]):int(rstartv[iib]) + window, :]
                vec2 = databp[int(rstartv[iib]) + 1:int(rstartv[iib]) + window + 1, :]
                if type(vec1m) == type(None):
                    vec1m = vec1.view(window, 1, -1)
                    vec2m = vec2.view(window, 1, -1)
                else:
                    vec1m = torch.cat((vec1m, vec1.view(window, 1, -1)), dim=1)
                    vec2m = torch.cat((vec2m, vec2.view(window, 1, -1)), dim=1)
            x = Variable(vec1m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)  #
            y = Variable(vec2m.reshape(window, batch, lsize_in).contiguous(), requires_grad=True)
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            if gpuavail:
                outlab = outlab.to(device)
                x, y = x.to(device), y.to(device)
            output, hidden = rnn(x, hidden, schedule=iis / step)
            # output, hidden = rnn(x, hidden, wta_noise=0.2 * (1.0 - iis / step))
            # loss = custom_KNWLoss(output.permute(1, 2, 0), outlab, rnn, iis)
            loss = output

            # if gpuavail:
            #     del x,y,outlab
            #     torch.cuda.empty_cache()

        if prtstep>0:
            if int(iis / prtstep) != his:
                print("Perlexity: ", iis, np.exp(loss.item()))
                his = int(iis / prtstep)
        else:
            print("Perlexity: ", iis, np.exp(loss.item()))

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
            # plt.ylim((0, 2000))
            plt.show()
    except:
        pass
    torch.cuda.empty_cache()
    return rnn

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
        self.seqtrain = para.get("seqtrain", True)
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

    def run_training_univ(self,step=None,lossf=None):
        """
        Iteration of two training subprocess
        :param step:
        :return:
        """
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
        if self.gpuavail:
            self.rnn.to(self.device)
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