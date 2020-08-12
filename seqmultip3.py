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
from ncautil.ncalearn import *
from ncautil.ptfunction import *

from awd_lstm_lm.weight_drop import WeightDrop
from ncautil.seqmultip import WeightMask

def cnn_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = np.floor((HWin+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    return HWout

class ConvNet(torch.nn.Module):
    def __init__(self, model_para):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        self.conv_layer_stack = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(self.conv_para[iic]),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(self.maxpool_para))
            for iic in range(len(self.conv_para))])

        Hsize = self.input_H
        Wsize = self.input_W
        for iic in range(len(self.conv_para)):
            Hsize = cnn_outszie(Hsize, self.conv_para[2], stride=self.conv_para[iic][3])
            Wsize = cnn_outszie(Wsize, self.conv_para[2], stride=self.conv_para[iic][3])

        print("Calculated Hsize: %s, Wsize: %s."%(Hsize,Wsize))

        conv_out=Hsize*Wsize*self.conv_para[-1][1]
        if len(self.mlp_para)>0:
            self.i2h = torch.nn.Linear(conv_out, self.mlp_para[0])
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_para[iil], self.mlp_para[iil+1], bias=True), torch.nn.LayerNorm(self.mlp_para[iil+1]),torch.nn.ReLU())
                for iil in range(len(self.mlp_para)-1)])
            self.h2o = torch.nn.Linear(self.mlp_para[0], self.output_size)
        else:
            self.i2o = torch.nn.Linear(conv_out, self.output_size)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def set_model_para(self,model_para):
        # model_para = {"input_size": 300,
        #               "output_size": 7,
        #               "gs_head_dim": 2,
        #               "gs_head_num": 11,
        #               "mlp_num_layers": [3, 0],
        #               "mlp_hidden": 80,
        #               }
        self.model_para = model_para
        self.input_H = model_para["input_H"]
        self.input_W = model_para["input_W"]
        self.output_size = model_para["output_size"]
        self.conv_para = model_para["conv_para"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.maxpool_para = model_para.get("maxpool_para",2)
        self.mlp_para= model_para["mlp_para"] # [hidden1, hidden2, ...]

    def forward(self, xin):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """

        for fwd in self.conv_layer_stack:
            xin = fwd(xin)

        if len(self.mlp_para) > 0 :
            hidden = self.i2h(xin)
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
            output = self.h2o(hidden)
        else:
            output = self.i2o(xin)
        output = self.softmax(output)

        return output