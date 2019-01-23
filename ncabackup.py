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