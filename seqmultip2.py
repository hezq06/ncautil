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

class BiTRF_NLP(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """
    def __init__(self, model_size, hidden_size, output_size, window_size, para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.model_size = model_size
        self.window_size=window_size

        if para is None:
            para = dict([])
        self.para(para)

        # self.slf_attn = MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout)
        # self.pos_ffn = PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)

        self.posi_sinmat = self.get_sinusoid_encoding_table(window_size, model_size)
        # self.posi_mat, self.n_posiemb = self.posi_embed(self.window_size,
        #                                                 switch="log")  # (self.input_len,self.n_posiemb)

        self.trf_layers = torch.nn.ModuleList([
            torch.nn.ModuleList([MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
            for _ in range(self.num_layers)])

        self.h2o = torch.nn.Linear(model_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)

        # self.selfattn_mask=torch.eye(self.window_size)
        # self.selfattn_mask = self.selfattn_mask.type(torch.uint8)
        if torch.cuda.is_available():
            self.posi_sinmat = self.posi_sinmat.to(self.cuda_device)
            # self.selfattn_mask = self.selfattn_mask.to(self.cuda_device)

        self.input_mask=None

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.dropout = para.get("dropout", 0.1)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.n_head = para.get("n_head", 1)
        self.d_k = para.get("d_k", 99)
        self.d_v = para.get("d_v", 101)
        self.num_layers = para.get("num_layers", 1)

    def get_sinusoid_encoding_table(self, n_position, model_size):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / model_size)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(model_size)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def posi_embed(self,emb_len,switch="onehot"):
        if switch=="log":
            n_posiemb=int(np.log(emb_len)/np.log(2))+1
            posi_mat=torch.ones((emb_len,n_posiemb))
            for ii in range(n_posiemb):
                step = 2 ** (ii+1)
                for ii2 in range(2 ** ii):
                    posi_mat[ii2::step,ii]=0
        elif switch=="onehot":
            n_posiemb = emb_len
            posi_mat=torch.eye(emb_len)
        else:
            raise Exception("Switch not known")
        print(posi_mat)
        if torch.cuda.is_available():
            posi_mat = posi_mat.to(self.cuda_device)
        return posi_mat,n_posiemb

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        assert len(input.shape)==3
        length, batch, l_size = input.shape
        input=input.permute((1,0,2)) # Due to transformer, use [batch, seq_len, hd_size] convention
        rnd=torch.rand((input.shape[0],input.shape[1]),device=self.cuda_device)
        self.input_mask=torch.zeros((input.shape[0],input.shape[1]),device=self.cuda_device)
        self.input_mask[rnd>self.mask_rate]=1
        input=input*self.input_mask.view(input.shape[0],input.shape[1],1).expand_as(input)

        enc_output = self.layer_norm(input) + self.posi_sinmat.view(1, -1, self.model_size)
        ### or
        # enc_output = torch.cat(
        #     (self.posi_mat.view(1, length , self.n_posiemb).expand(batch, length, self.n_posiemb), input), dim=-1)

        for trf_l in self.trf_layers:
            enc_output, attn =trf_l[0](enc_output,enc_output,enc_output)
            enc_output = trf_l[1](enc_output)

        output=self.h2o(enc_output)

        output = self.softmax(output)

        output=output.permute((1,0,2))

        return output,None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        # attn = self.dropout(attn)
        # print(attn.shape,v.shape)
        # plot_mat(attn[0,:,:].cpu().detach())
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output+residual)
        # output = self.layer_norm(output)

        return output, attn

class PositionwiseFeedForward(torch.nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = torch.nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = torch.nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)
        return output