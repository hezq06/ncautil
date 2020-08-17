"""
PyTorch Modules of All kinds vol2 starting 2019/10/18
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import copy

import torch
from torch.autograd import Variable
from ncautil.ncalearn import MyLossFun

from awd_lstm_lm.weight_drop import WeightDrop
from ncautil.seqmultip import WeightMask

def cnn_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin+2*padding-dilation*(kernel_size-1)-1)/stride+1
    HWout = np.floor(HWout)
    return HWout

def deconv_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin-1)*stride-2*padding+kernel_size
    return HWout

class ABS_CNN_SEQ(torch.nn.Module):
    """
    An abstract cooperation container for seqential model
    """

    def __init__(self, seq1_coop, seq2_train, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.seq1_coop = seq1_coop
        self.seq2_train = seq2_train

        # for param in self.seq1_coop.parameters():
        #     param.requires_grad = False
        self.seq1_coop.coop_mode = True

        if para is None:
            para = dict([])
        self.para(para)

        self.save_para = {
            "seq1_coop": seq1_coop.save_para,
            "seq2_train": seq2_train.save_para,
            "type": "ABS_CNN_SEQ"
        }

    def para(self,para):
        pass

    def forward(self, datax, labels, schedule=None):
        self.device = datax.device
        out_seq1 = self.seq1_coop(datax, None, schedule=1.0)
        loss = self.seq2_train(out_seq1, labels, schedule=schedule)
        return loss

class MultiHeadConvNet(torch.nn.Module):
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)
        if para is None:
            para = dict([])
        self.para(para)

        Hsize = self.input_H
        Wsize = self.input_W
        sizel=[]
        sizele = []
        for iic in range(len(self.conv_para)):
            Hsize = cnn_outszie(Hsize, self.conv_para[iic][2], stride=self.conv_para[iic][3])
            Wsize = cnn_outszie(Wsize, self.conv_para[iic][2], stride=self.conv_para[iic][3])
            sizel.append([int(Hsize),int(Wsize)])
            Hsize = cnn_outszie(Hsize, self.maxpool_para, stride=self.maxpool_para)
            Wsize = cnn_outszie(Wsize, self.maxpool_para, stride=self.maxpool_para)
            sizele.append([int(Hsize),int(Wsize)])

        self.conv_layer_stack = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(*self.conv_para[iic]), #[channel_in, channel_out, kernal, stride] * conv_num
                                torch.nn.LayerNorm(sizel[iic]),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(self.maxpool_para))
            for iic in range(len(self.conv_para))])
        if self.res_per_conv>0:
            self.res_conv_stack = torch.nn.ModuleList([
                torch.nn.Sequential(*[ResConvModule(
                    {
                            "input_H": sizele[iic][0],
                            "input_W": sizele[iic][1],
                            "channel": self.conv_para[iic][1],
                            "kernel": 5,
                            "conv_layer": 2
                    }
                ) for iires in range(self.res_per_conv)])
                for iic in range(len(self.conv_para))])

        conv_out = int(Hsize * Wsize * self.conv_para[-1][1])
        print("Calculated Hsize: %s, Wsize: %s, ConvOut:%s."%(Hsize,Wsize,conv_out))

        if len(self.mlp_para)>0:
            self.i2h = torch.nn.Linear(conv_out, self.mlp_para[0])
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_para[iil], self.mlp_para[iil+1], bias=True), torch.nn.LayerNorm(self.mlp_para[iil+1]),torch.nn.ReLU())
                for iil in range(len(self.mlp_para)-1)])
            self.h2o = torch.nn.Linear(self.mlp_para[-1], self.output_size)
        else:
            self.i2o = torch.nn.Linear(conv_out, self.output_size)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.save_para = {
            "model_para": model_para,
            "type": "MultiHeadConvNet",
            "misc_para": para
        }

    def para(self,para):
        self.misc_para = para
        self.coop_mode = para.get("coop_mode", False)

    def set_model_para(self,model_para):
        # model_para = {
        #     "input_H": 320,
        #     "input_W": 480,
        #     "output_size": 2 + 8 + 3,  # posix, posiy, color, shape
        #     "conv_para": [[3, 16, 7, 2], [16, 16, 7, 2], [16, 16, 7, 1]],
        #     "maxpool_para": 2,
        #     "mlp_para": [128, 64]
        # }
        self.model_para = model_para
        self.input_H = model_para["input_H"]
        self.input_W = model_para["input_W"]
        self.output_size = model_para["output_size"]
        self.conv_para = model_para["conv_para"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.maxpool_para = model_para.get("maxpool_para",2)
        self.mlp_para= model_para["mlp_para"] # [hidden1, hidden2, ...]
        self.res_per_conv = model_para.get("res_per_conv",0)

    def forward(self, datax, labels, schedule=None):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """
        batch = datax.shape[0]

        for iil, fwd in enumerate(self.conv_layer_stack):
            datax = fwd(datax)
            if self.res_per_conv > 0:
                datax = self.res_conv_stack[iil](datax)


        if len(self.mlp_para) > 0 :
            hidden = self.i2h(datax.view(batch,-1))
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
            output = self.h2o(hidden)
        else:
            output = self.i2o(datax)

        if self.coop_mode:
            return output

        ## Loss, posi
        device = labels.device
        loss_mse = torch.nn.MSELoss()
        lossposi = loss_mse(output[:,0:2],labels[:,0:2].type(torch.FloatTensor).to(device))
        ## Loss, color
        lossc = torch.nn.CrossEntropyLoss()
        losscolor = lossc(output[:,2:10],labels[:,2].type(torch.LongTensor).to(device))
        ## Loss, shape
        lossshape = lossc(output[:, 10:], labels[:, 3].type(torch.LongTensor).to(device))

        return lossposi+losscolor+lossshape

class DeConvNet(torch.nn.Module):
    """
    Deconvolutional Net for auto-encoding
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        Hsize = self.output_H
        Wsize = self.output_W

        for iic in range(len(self.deconv_para)):
            iip=len(self.deconv_para)-iic-1
            Hsize = cnn_outszie(Hsize, self.deconv_para[iip][2], stride=self.deconv_para[iip][3])
            Hsize = cnn_outszie(Hsize, self.maxunpool_para, stride=self.maxunpool_para)
            Wsize = cnn_outszie(Wsize, self.deconv_para[iip][2], stride=self.deconv_para[iip][3])
            Wsize = cnn_outszie(Wsize, self.maxunpool_para, stride=self.maxunpool_para)

        conv_in = int(Hsize * Wsize * self.deconv_para[0][0])
        self.Hsize_in = int(Hsize)
        self.Wsize_in = int(Wsize)
        print("Calculated Hsize: %s, Wsize: %s, ConvOut:%s." % (Hsize, Wsize, conv_in))

        ## Calculate actual H,W size for layernorm
        sizel = []
        sizele= []
        for iic in range(len(self.deconv_para)):
            sizele.append([int(Hsize), int(Wsize)])
            Hsize = deconv_outszie(Hsize, self.maxunpool_para, stride=self.maxunpool_para)
            Hsize = deconv_outszie(Hsize, self.deconv_para[iic][2], stride=self.deconv_para[iic][3])
            Wsize = deconv_outszie(Wsize, self.maxunpool_para, stride=self.maxunpool_para)
            Wsize = deconv_outszie(Wsize, self.deconv_para[iic][2], stride=self.deconv_para[iic][3])
            sizel.append([int(Hsize), int(Wsize)])
        print("Actual output shape (%s,%s)."%(str(Hsize),str(Wsize)))

        if len(self.mlp_para)>0:
            self.i2h = torch.nn.Linear(self.input_size, self.mlp_para[0])
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_para[iil], self.mlp_para[iil+1], bias=True), torch.nn.LayerNorm(self.mlp_para[iil+1]),torch.nn.ReLU())
                for iil in range(len(self.mlp_para)-1)])
            self.h2m = torch.nn.Linear(self.mlp_para[-1], conv_in)
        else:
            self.i2m = torch.nn.Linear(self.input_size, conv_in)

        if self.res_per_conv>0:
            self.res_conv_stack = torch.nn.ModuleList([
                torch.nn.Sequential(*[ResConvModule(
                    {
                            "input_H": sizele[iic][0],
                            "input_W": sizele[iic][1],
                            "channel": self.deconv_para[iic][0],
                            "kernel": 5,
                            "conv_layer": 2
                    }
                ) for iires in range(self.res_per_conv)])
                for iic in range(len(self.deconv_para))])

        self.conv_layer_stack = torch.nn.ModuleList([
            torch.nn.Sequential(
                                #torch.nn.MaxUnpool2d(self.maxunpool_para),
                                # Use deconvolution to replace MaxUnpool2d, no change in channel, same kernel and stride
                                torch.nn.ConvTranspose2d(self.deconv_para[iic][0],self.deconv_para[iic][0],self.maxunpool_para,self.maxunpool_para),
                                torch.nn.ConvTranspose2d(*self.deconv_para[iic]),
                                torch.nn.LayerNorm(sizel[iic]),
                                torch.nn.ReLU())
            for iic in range(len(self.deconv_para)-1)])
        self.conv_layer_stack.append(
            torch.nn.Sequential(
                # torch.nn.MaxUnpool2d(self.maxunpool_para),
                # Use deconvolution to replace MaxUnpool2d, no change in channel, same kernel and stride
                torch.nn.ConvTranspose2d(self.deconv_para[-1][0], self.deconv_para[-1][0], self.maxunpool_para,
                                         self.maxunpool_para),
                torch.nn.ConvTranspose2d(*self.deconv_para[-1]),
                torch.nn.ReLU())
        )

        self.save_para = {
            "model_para": model_para,
            "type": "DeConvNet",
            "misc_para": para
        }

    def set_model_para(self,model_para):
        # model_para = {"input_size": 300,
        #               "output_size": 7,
        #               "gs_head_dim": 2,
        #               "gs_head_num": 11,
        #               "mlp_num_layers": [3, 0],
        #               "mlp_hidden": 80,
        #               }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.output_H = model_para["output_H"]
        self.output_W = model_para["output_W"]
        self.deconv_para = model_para["deconv_para"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.maxunpool_para = model_para.get("maxunpool_para",2)
        self.mlp_para= model_para["mlp_para"] # [hidden1, hidden2, ...]
        self.res_per_conv = model_para.get("res_per_conv", 0)

    def forward(self, datax, labels, schedule=None):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """
        batch = datax.shape[0]

        if len(self.mlp_para) > 0 :
            hidden = self.i2h(datax)
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
            tconvin = self.h2m(hidden)
        else:
            tconvin = self.i2m(datax)

        tconvin = tconvin.view(batch, self.deconv_para[0][0], self.Hsize_in, self.Wsize_in)

        for iil, fwd in enumerate(self.conv_layer_stack):
            if self.res_per_conv > 0:
                tconvin = self.res_conv_stack[iil](tconvin)
            tconvin = fwd(tconvin)

        self.outimage=tconvin

        ## loss function
        loss_mse = torch.nn.MSELoss(reduction="sum")
        batch, channel, Hsize, Wsize = tconvin.shape
        loss = loss_mse(tconvin, labels[:,:,:Hsize,:Wsize])

        return loss

class ResConvModule(torch.nn.Module):
    """
    ResNet Module, dimension, channels are same
    """
    def __init__(self, model_para):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)
        padding=int((self.kernel-1)/2) # Note kernel should be odd
        self.conv = torch.nn.Conv2d(self.channel,self.channel,self.kernel,1,padding)
        self.lnorm = torch.nn.LayerNorm([self.input_H,self.input_W])
        self.relu = torch.nn.ReLU()

    def set_model_para(self,model_para):
        # model_para = {
        #     "input_H": 320,
        #     "input_W": 480,
        #     "channel": 16,
        #     "kernel": 5,
        #     "conv_layer": 2
        # }
        self.model_para = model_para
        self.input_H = model_para["input_H"]
        self.input_W = model_para["input_W"]
        self.channel = model_para["channel"]
        self.kernel = model_para["kernel"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.conv_layer = model_para.get("conv_layer",2)

    def forward(self, datax):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """

        datac=self.conv(datax)
        datac = self.lnorm(datac)
        datac = self.relu(datac)

        for iil in range(self.conv_layer-1):
            datac = self.conv(datac)
            datac = self.lnorm(datac)
            if iil<self.conv_layer-2:
                datac = self.relu(datac)

        datao = datac+datax
        datao = self.relu(datao)

        return datao