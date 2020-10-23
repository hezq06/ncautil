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

class ABS_NLP_SEQ(torch.nn.Module):
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

        for param in self.seq1_coop.parameters():
            param.requires_grad = False
        self.seq1_coop.coop_mode = True

        if para is None:
            para = dict([])
        self.para(para)

        # self.softmax = torch.nn.LogSoftmax(dim=-1)
        # self.sigmoid = torch.nn.Sigmoid()
        # self.gsigmoid = Gumbel_Sigmoid()

        # if type(seq2_train) == list:
        #     seq2_train_save_para=[seq2_train_item.save_para for seq2_train_item in seq2_train]
        # else:
        #     seq2_train_save_para =seq2_train.save_para

        self.save_para = {
            "seq1_coop": seq1_coop.save_para,
            "seq2_train": seq2_train.save_para,
            "type": "ABS_NLP_SEQ"
        }

    def para(self,para):
        self.coop_mode=para.get("coop_mode",False)

    def forward(self, inputl, hidden1, add_logit=None, logit_mode=False, schedule=None):

        # if not self.freeze_mode:
        #     temperature = np.exp(-schedule * 5)
        # else:
        #     temperature = np.exp(-5)

        self.cuda_device = inputl.device

        out_seq1, hn_seq1 = self.seq1_coop(inputl, hidden1[0], logit_mode=True, schedule=1.0)

        if self.coop_mode:
            self.seq2_train.coop_mode = True
        # print(out_seq1,hn_seq1)

        # if type(self.seq2_train) == list:
        #     assert len(out_seq1)==len(self.seq2_train)
        # else:
        out_seq2, hn_seq2 = self.seq2_train(out_seq1, hidden1[1], logit_mode=True, schedule=schedule)

        # self.loss_intf = self.seq2_train.loss_intf
        # self.context = self.seq2_train.context
        # self.gssample = self.seq2_train.gssample
        # self.gssample_coop = out_seq1
        # self.coop_sample = self.seq1_coop.coop_sample

        return out_seq2, [hn_seq1, hn_seq2]

    def initHidden(self,batch):
        return [self.seq1_coop.initHidden(batch),self.seq2_train.initHidden(batch)]

    def initHidden_cuda(self,device, batch):
        return [self.seq1_coop.initHidden_cuda(device, batch), self.seq2_train.initHidden_cuda(device, batch)]

class Hierachical_BiGRU_NLP_GSVIB(torch.nn.Module):
    """
    PyTorch BiGRU_GSVIB for NLP with Gumbel Softmax variational information bottleneck
    Hierachical Inductive Bias
    gs_dim is 2
    """
    def __init__(self, coop_bigru_nlp, maskmat, model_para, para=None, coop_bigru_nlp2 = None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        self.coop_bigru_nlp = coop_bigru_nlp
        for param in self.coop_bigru_nlp.parameters():
            param.requires_grad = False
        self.coop_bigru_nlp2 = coop_bigru_nlp2
        if self.coop_bigru_nlp2 is not None:
            for param in self.coop_bigru_nlp2.parameters():
                param.requires_grad = False

        self.maskmat = torch.from_numpy(maskmat).type(torch.FloatTensor)

        if para is None:
            para = dict([])
        self.para(para)

        self.gru = torch.nn.GRU(self.input_size,self.hidden_size,num_layers=self.num_layers,bidirectional=True)
        self.h2c = torch.nn.Linear(self.hidden_size * 2, self.gs_head_num*self.gs_head_dim)
        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num,self.gs_head_dim)) , requires_grad=True)
        if coop_bigru_nlp2 is not None:
            self.maskmat_t2 = torch.nn.Parameter(torch.zeros((self.gs_head_num,5)), requires_grad=True)

        if self.mlp_num_layers>0:
            self.c2h = torch.nn.Linear(self.gs_head_num*self.gs_head_dim, self.mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=True), torch.nn.LayerNorm(self.mlp_hidden),torch.nn.ReLU())
                for _ in range(self.mlp_num_layers-1)])
            self.h2o = torch.nn.Linear(self.mlp_hidden, self.output_size)
        else:
            self.c2o = torch.nn.Linear(self.gs_head_num*self.gs_head_dim, self.output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.zsoftmax = torch.nn.LogSoftmax(dim=0)
        self.gsoftmax = Gumbel_Softmax(sample_mode=False)
        self.layer_norm = torch.nn.LayerNorm(self.gs_head_num * self.gs_head_dim)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        if torch.cuda.is_available():
            self.maskmat=self.maskmat.to(self.cuda_device)

        self.save_para = {
            "cooprer": coop_bigru_nlp.save_para,
            "model_para":self.model_para,
            "type": "Hierachical_BiGRU_NLP_GSVIB",
            "misc_para": para
        }

    def set_model_para(self,model_para):
        # model_para_hgsbigru = {
        #     "input_size": 300,
        #     "output_size": 30001,
        #     "gs_head_num": 21,
        #     "gs_head_dim": 2,
        #     "hidden_size": 50,
        #     "num_layers": 3,
        #     "mlp_hidden": 160,
        #     "mlp_num_layers": 1,
        # }
        self.model_para = model_para
        self.hidden_size = model_para["hidden_size"]
        self.input_size = model_para["input_size"]
        self.output_size = model_para["output_size"]
        self.gs_head_num = model_para["gs_head_num"]
        self.gs_head_dim = model_para["gs_head_dim"]
        self.num_layers = model_para["num_layers"]
        self.mlp_num_layers = int(model_para.get("mlp_num_layers", 0))
        self.mlp_hidden = int(model_para.get("mlp_hidden", 80))

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.dropout_rate = para.get("dropout_rate", 0.2)
        self.coop_mode = para.get("coop_mode", False)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None, attention_sig=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        self.cuda_device=input.device
        dw, batch, l_size = input.shape

        hout, hn = self.gru(input,hidden1[0])
        context = self.h2c(hout)
        context = context.view(dw, batch, self.gs_head_num, self.gs_head_dim)
        context = self.softmax(context)
        gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        sfm_coop, hn_coop = self.coop_bigru_nlp(input, hidden1[1],schedule=1.0)
        coop_sample = self.gsoftmax(sfm_coop, temperature=temperature, cuda_device=self.cuda_device)
        gs_mask = coop_sample.matmul(self.maskmat)
        if self.coop_bigru_nlp2 is not None:
            sfm_coop1, hn_coop1 = self.coop_bigru_nlp2(input, hidden1[2], schedule=1.0)
            coop_sample1 = self.gsoftmax(sfm_coop1, temperature=temperature, cuda_device=self.cuda_device)
            maskmat_t2 = self.softmax(self.maskmat_t2)
            maskmat_t2 = self.gsoftmax(maskmat_t2, temperature=np.exp(-5), cuda_device=self.cuda_device)
            gs_mask1 = coop_sample1.matmul(maskmat_t2.transpose(1,0))
            gs_mask = gs_mask * gs_mask1

        gssample = gssample * gs_mask.view(dw, batch, self.gs_head_num, 1)

        p_prior = self.softmax(self.prior)
        self.context = context
        self.p_prior=p_prior
        self.gssample = gssample
        self.gs_mask = gs_mask
        self.loss_intf = [self.context, p_prior]
        self.coop_sample = coop_sample

        gssample = gssample.view(dw, batch, self.gs_head_num*self.gs_head_dim)
        gssample=self.layer_norm(gssample) ## Very important

        if self.coop_mode:
            return gssample, [hn , hn_coop]

        if self.mlp_num_layers > 0:
            hidden_c=self.c2h(gssample)
            hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            output=self.h2o(hidden_c)
        else:
            output = self.c2o(gssample)

        if not logit_mode:
            output = self.softmax(output)

        return output, [hn , hn_coop]

    def initHidden(self,batch):
        if self.coop_bigru_nlp2 is None:
            return [Variable(torch.zeros(self.num_layers*2, batch,self.hidden_size), requires_grad=True),
                    self.coop_bigru_nlp.initHidden(batch)]
        else:
            return [Variable(torch.zeros(self.num_layers * 2, batch, self.hidden_size), requires_grad=True),
                    self.coop_bigru_nlp.initHidden(batch),
                    self.coop_bigru_nlp2.initHidden(batch)]

    def initHidden_cuda(self, device, batch):
        if self.coop_bigru_nlp2 is None:
            return [Variable(torch.zeros(self.num_layers*2, batch, self.hidden_size), requires_grad=True).to(device),
                    self.coop_bigru_nlp.initHidden_cuda(device, batch)]
        else:
            return [Variable(torch.zeros(self.num_layers * 2, batch, self.hidden_size), requires_grad=True).to(device),
                    self.coop_bigru_nlp.initHidden_cuda(device, batch),
                    self.coop_bigru_nlp2.initHidden_cuda(device, batch)]

class COOP_W2V_Softmax(torch.nn.Module):
    """
    PyTorch Simple Softmax W2V calculation
    Generate output and label at one forward pass
    """
    def __init__(self, level1_coop, Ndiml, Nvocabl, window_sizel, wrd_to_idl, para=None):
        """

        :param Ndim: Embedding dimension
        :param Nvocab: # of vocab
        :param window_size: context size
        :param ns_size: negtive sampling size
        :param prior: prior distribution
        """
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.coop_mode = False

        self.level1_coop = level1_coop
        for param in self.level1_coop.parameters():
            param.requires_grad = False
        self.level1_coop.cooprer.coop_mode=True
        self.level1_coop.trainer.coop_mode = True
        self.level1_coop.coop_mode = True

        self.Ndim_pos = Ndiml[0]
        self.Nvocab_pos = Nvocabl[0]
        self.window_size_pos = window_sizel[0]

        self.Ndim_sem = Ndiml[1]
        self.Nvocab_sem = Nvocabl[1]
        self.window_size_sem = window_sizel[1]

        w2v_in_pos = self.level1_coop.cooprer.gs_head_num * level1_coop.cooprer.gs_head_dim
        self.w2v_pos = torch.nn.Linear(w2v_in_pos, self.Ndim_pos)

        w2v_in_sem = self.level1_coop.trainer.gs_head_num * level1_coop.trainer.gs_head_dim
        self.w2v_sem = torch.nn.Linear(w2v_in_sem, self.Ndim_sem)
        # self.w2v=torch.nn.Embedding(Nvocab,Ndim)

        self.conv1_pos = torch.nn.Conv1d(self.Ndim_pos, self.Ndim_pos, self.window_size_pos, stride=1, padding=0, bias=False)
        self.conv1_sem = torch.nn.Conv1d(self.Ndim_sem, self.Ndim_sem, self.window_size_sem, stride=1, padding=0, bias=False)
        ## A conv1d mimicking context sum
        # conv_tensor=torch.ones(Ndim, Ndim, window_size) / ((window_size-1))
        # conv_tensor[:,:,int((window_size-1)/2)]=0
        # self.conv1.weight = torch.nn.Parameter(conv_tensor, requires_grad=False)
        self.mask_pos = torch.ones(self.window_size_pos)
        self.mask_pos[int((self.window_size_pos - 1) / 2)] = 0
        self.mask_sem = torch.ones(self.window_size_sem)
        self.mask_sem[int((self.window_size_sem-1)/2)] = 0

        self.hout_pos=torch.nn.Linear(self.Ndim_pos, self.Nvocab_pos)
        self.hout_sem = torch.nn.Linear(self.Ndim_sem, self.Nvocab_sem)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if torch.cuda.is_available():
            self.mask_pos=self.mask_pos.to(self.cuda_device)
            self.mask_sem = self.mask_sem.to(self.cuda_device)

        self.masked_conv1_pos = WeightMask(self.conv1_pos, ["weight"], self.mask_pos.view(1, 1, -1))
        self.masked_conv1_sem = WeightMask(self.conv1_sem, ["weight"], self.mask_sem.view(1, 1, -1))

        ## for outlab generation
        vec_pos = np.array(range(self.level1_coop.cooprer.gs_head_num))
        self.vec_pos = level1_coop.cooprer.gs_head_dim ** vec_pos

        vec_sem = np.array(range(self.level1_coop.trainer.gs_head_num))
        self.vec_sem = level1_coop.trainer.gs_head_dim ** vec_sem

        self.wrd_to_id_pos = wrd_to_idl[0]
        self.wrd_to_id_sem = wrd_to_idl[1]

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")

    def forward(self, inputd, hidden, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: size[window_size_ext,batch]
        :return:
        """
        self.cuda_device = inputd.device

        window_size_ext,batch, _ = inputd.shape
        assert window_size_ext>=self.window_size_pos
        assert window_size_ext >= self.window_size_sem

        logit_level1, hn_level1 = self.level1_coop(inputd, hidden, logit_mode=True, schedule=1.0)
        gssample_lv1_t, gssample_lv1_s = logit_level1  # pos, sem, level 1 should return context

        input_mat_pos = self.w2v_pos(gssample_lv1_t.view(window_size_ext,batch,-1)) # window_size_ext,batch,Ndim
        input_mat_sem = self.w2v_sem(gssample_lv1_s.view(window_size_ext, batch, -1))  # window_size_ext,batch,Ndim
        if self.coop_mode:
            return [input_mat_pos, input_mat_sem], hn_level1

        input_mat_pos = input_mat_pos.permute(1, 2, 0) # batch,Ndim,window_size_ext
        # input_mat = input_mat * self.mask.view(1,1,-1)
        # conv_mat = self.conv1(input_mat)  # batch,Ndim,window_size_ext
        conv_mat_pos = self.masked_conv1_pos(input_mat_pos) # batch,Ndim,window_size_ext
        output_pos = self.hout_pos(conv_mat_pos.permute(2,0,1))
        output_pos=self.softmax(output_pos)


        input_mat_sem = input_mat_sem.permute(1, 2, 0)  # batch,Ndim,window_size_ext
        # input_mat = input_mat * self.mask.view(1,1,-1)
        # conv_mat = self.conv1(input_mat)  # batch,Ndim,window_size_ext
        conv_mat_sem = self.masked_conv1_sem(input_mat_sem)  # batch,Ndim,window_size_ext
        output_sem = self.hout_sem(conv_mat_sem.permute(2, 0, 1))
        output_sem = self.softmax(output_sem)

        ## outlab generation
        # POS
        outlab_pos_mat = gssample_lv1_t.cpu().detach().numpy()
        outlab_pos_mat = np.argmax(outlab_pos_mat, axis=-1)
        outlab_pos_mat = np.squeeze(outlab_pos_mat.dot(self.vec_pos.reshape(-1, 1)))
        outlab_pos_conv = outlab_pos_mat[int((self.window_size_pos+1)/2)-1:window_size_ext-int((self.window_size_pos+1)/2)+1,:]
        shape_pos = outlab_pos_conv.shape
        outlab_pos=[]
        for item in outlab_pos_conv.reshape(-1):
            outlab_pos.append(self.wrd_to_id_pos.get(item,len(self.wrd_to_id_pos)-1)) # -1 due to "UNK" token
        outlab_pos=np.array(outlab_pos).reshape(shape_pos)
        outlab_pos = torch.from_numpy(outlab_pos).type(torch.LongTensor).to(self.cuda_device)
        outlab_pos = outlab_pos.permute((1,0))
        # SEM
        outlab_sem_mat = gssample_lv1_s.cpu().detach().numpy()
        outlab_sem_mat = np.argmax(outlab_sem_mat, axis=-1)
        outlab_sem_mat = np.squeeze(outlab_sem_mat.dot(self.vec_sem.reshape(-1, 1)))
        outlab_sem_conv = outlab_sem_mat[int((self.window_size_sem + 1) / 2) - 1:window_size_ext - int((self.window_size_sem + 1) / 2) + 1, :]
        shape_sem = outlab_sem_conv.shape
        outlab_sem = []
        for item in outlab_sem_conv.reshape(-1):
            outlab_sem.append(self.wrd_to_id_sem.get(item, len(self.wrd_to_id_sem)-1))
        outlab_sem = np.array(outlab_sem).reshape(shape_sem)
        outlab_sem = torch.from_numpy(outlab_sem).type(torch.LongTensor).to(self.cuda_device)
        outlab_sem = outlab_sem.permute((1, 0))

        return [[output_pos,outlab_pos],[output_sem,outlab_sem]], hn_level1 # [window_size_ext-self.window_size+1,batch,Nvcab]

    def initHidden(self, batch):
        return self.level1_coop.initHidden(batch)

    def initHidden_cuda(self, device, batch):
        return self.level1_coop.initHidden_cuda(device, batch)

class LEVEL2_ATT_InfoFolk(torch.nn.Module):
    """
     step 2 of interpretation task with a attention based information folking machenism
    """

    def __init__(self, level1_coop, lsize_in, output_size, mlp_layersl, mlp_hidden ,para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.lsize_in = lsize_in
        self.output_size = output_size
        self.mlp_layersl = mlp_layersl
        self.mlp_hidden = mlp_hidden

        self.level1_coop = level1_coop
        for param in self.level1_coop.parameters():
            param.requires_grad = False
        self.level1_coop.cooprer.coop_mode=True
        self.level1_coop.trainer.coop_mode = True
        self.level1_coop.coop_mode = True

        if para is None:
            para = dict([])
        self.para(para)

        self.gs_head_num = level1_coop.cooprer.gs_head + level1_coop.trainer.gs_head
        self.gs_head_dim=level1_coop.cooprer.context_size

        # Trial of self attention
        self.att_infofolk = FF_MLP(lsize_in, mlp_hidden, self.gs_head_num, mlp_num_layers=self.mlp_layersl[0],para=para)
         # self.prior = torch.nn.Parameter(torch.zeros((gs_head_num, level1_coop.cooprer.context_size)), requires_grad=True)

        assert level1_coop.cooprer.context_size == level1_coop.trainer.context_size
        mlpout_in = self.gs_head_num * level1_coop.cooprer.context_size
        self.mlp_out = FF_MLP(mlpout_in, mlp_hidden, self.output_size, mlp_num_layers=self.mlp_layersl[1], para=para)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.gs_head_num*self.gs_head_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.gsoftmax = Gumbel_Softmax()
        self.gsigmoid = Gumbel_Sigmoid()

        self.prior_lv1 = torch.cat((self.level1_coop.cooprer.prior,self.level1_coop.trainer.prior),dim=-2)
        self.p_prior_lv1 = self.softmax(self.prior_lv1)

    def para(self,para):
        self.freeze_mode=para.get("freeze_mode",False)
        self.temp_scan_num = para.get("temp_scan_num", 1)

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None, context_set=None):

        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        self.cuda_device = input.device

        # hidden1 [level1_coop hidden, BiGRU1 hidden, BiGRU2 hidden]
        logit_level1, hn_level1 = self.level1_coop(input, hidden, logit_mode=True, schedule=1.0)
        context_lv1_t, context_lv1_s = logit_level1 # pos, sem, level 1 should return context
        context_lv1 = torch.cat((context_lv1_t, context_lv1_s), dim=-2)

        attention , _ = self.att_infofolk(input,None)
        attention_gsig = self.gsigmoid(attention, temperature=temperature, cuda_device=self.cuda_device)
        attention_prob = self.sigmoid(attention)

        dw, batch, gs_head_num = attention_gsig.shape
        p_prior = self.p_prior_lv1.expand(dw, batch, self.gs_head_num, self.gs_head_dim).to(self.cuda_device)

        att_context = context_lv1 * attention_gsig.view(dw, batch, self.gs_head_num, 1)
        att_context = att_context + p_prior * ( 1 - attention_gsig.view(dw, batch, self.gs_head_num, 1))

        pgssample = self.gsoftmax(att_context, temperature=temperature, cuda_device=self.cuda_device)

        # Sigmoid attention from POS
        # dw, batch, gs_head, gs_dim  = gssample_lv1_t.shape
        # attention_t, _ = self.att_infofolk(gssample_lv1_t.view(dw,batch,-1),None,logit_mode=True)
        # attention_sig_t = self.gsigmoid(attention_t, temperature=temperature, cuda_device=self.cuda_device)
        # attention_prob_t = self.sigmoid(attention_t)

        # p_prior = self.softmax(self.prior.expand(dw, batch, self.gs_head_num, self.gs_head_dim))
        # context = context_lv1 * attention_sig.view(dw, batch, self.gs_head_num, 1) + p_prior * (
        #             1 - attention_sig.view(dw, batch, self.gs_head_num, 1))
        # gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        # gssample_lv1_t = gssample_lv1_t * attention_sig_t.view(dw, batch, -1, 1)


        self.context = att_context
        self.gssample = pgssample
        self.attention_prob = attention_prob
        self.attention_sig = attention_gsig
        self.loss_intf = [self.attention_prob]

        output = self.layer_norm(pgssample.view(dw, batch, -1))
        output, _ = self.mlp_out(output, None, logit_mode=True, schedule=schedule)

        if add_logit is not None:
            output = output + add_logit
        if not logit_mode:
            output = self.softmax(output)

        return output, hn_level1

    def initHidden(self,batch):
        return self.level1_coop.initHidden(batch)

    def initHidden_cuda(self,device, batch):
        return self.level1_coop.initHidden_cuda(device, batch)

class LEVEL2_BiGRU_NLP_GSVIB(torch.nn.Module):
    """
    BiGRU_NLP_GSVIB equivalent for step 2 of interpretation task
    """

    # def __init__(self, level1_coop, lsize_in, hidden_sizel, gs_head_diml, gs_head_numl, mlp_num_layersl, mlp_hiddenl, output_size , para=None , w2v_flag=True):
    def __init__(self, level1_coop, model_para_gsvib1, model_para_gsvib2, model_para_ff, para=None , paral=None, w2v_flag=False):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        # self.lsize_in=lsize_in
        # self.hidden_sizel = hidden_sizel
        # self.gs_head_diml = gs_head_diml
        # self.mlp_num_layersl = mlp_num_layersl
        # self.mlp_hiddenl = mlp_hiddenl
        # self.output_size = output_size

        self.level1_coop = level1_coop
        for param in self.level1_coop.parameters():
            param.requires_grad = False

        self.level1_coop.cooprer.coop_mode = True
        self.level1_coop.trainer.coop_mode = True
        self.level1_coop.coop_mode = True

        if para is None:
            para = dict([])
        self.para(para)

        # hidden_dim_1=hidden_sizel[0]
        # gs_head_num_1=gs_head_numl[0]
        # gs_head_dim_1=gs_head_diml[0]
        # mlp_hidden_1=mlp_hiddenl[0]
        # mlpout_hd_1 = gs_head_num_1# * gs_head_dim_1
        # if not w2v_flag:
        #     lsize_in_1=level1_coop.cooprer.gs_head_num*level1_coop.cooprer.gs_head_dim # 1 cooprer POS
        #     lsize_in_2 = level1_coop.trainer.gs_head_num * level1_coop.trainer.gs_head_dim  # 2 trainer Semantics
        # else:
        #     lsize_in_1=level1_coop.Ndim_pos
        #     lsize_in_2 = level1_coop.Ndim_sem

        # input_size, hidden_size, gs_head_dim, gs_head_num, output_size, num_layers=1 ,para=None
        # self.level2_gsvib_1=BiGRU_NLP_GSVIB(lsize_in_1,hidden_dim_1,gs_head_dim_1,gs_head_num_1,
        #                 mlpout_hd_1,num_layers=mlp_num_layersl[0], para=para)
        # self.level2_gsvib_1 = BiGRU_NLP_VIB(lsize_in_1, hidden_dim_1, gs_head_num_1, mlp_hidden_1, mlpout_hd_1,
        #                                     num_layers=mlp_num_layersl[0], para=para)

        self.level2_gsvib_1 = FF_MLP_GSVIB(model_para_gsvib1, para=paral[0])

        # self.level2_gsvib_1 = BiGRU_NLP(lsize_in_1, hidden_dim_1, mlpout_hd_1, num_layers=self.num_layers, para=para)
        # self.level2_gsvib_1 = FF_MLP(lsize_in_1,mlp_hidden_1,mlpout_hd_1,mlp_num_layers=3,para=para)
        # para["coop_mode"] = True
        # self.level2_gsvib_1 = FF_MLP_GSVIB(lsize_in_1, mlp_hidden_1, context_dim_1, gs_head_1, mlpout_hd_1, mlp_num_layers=[mlp_num_layersl[0],0],
        #                              para=para)

        # hidden_dim_2 = hidden_sizel[1]
        # gs_head_num_2 = gs_head_numl[1]
        # gs_head_dim_2 = gs_head_diml[1]
        # mlp_hidden_2 = mlp_hiddenl[1]

        # mlpout_hd_2 = gs_head_num_2# * gs_head_dim_2
        # self.level2_gsvib_2 = BiGRU_NLP_GSVIB(lsize_in_2, hidden_dim_2, gs_head_dim_2, gs_head_num_2,
        #                                       mlpout_hd_2, num_layers=mlp_num_layersl[0], para=para)
        # self.level2_gsvib_2 = BiGRU_NLP_VIB(lsize_in_2, hidden_dim_2, gs_head_num_2, mlp_hidden_2, mlpout_hd_2,
        #                                     num_layers=mlp_num_layersl[0], para=para)

        self.level2_gsvib_2 = FF_MLP_GSVIB(model_para_gsvib2, para=paral[1])

        # self.level2_gsvib_2 = BiGRU_NLP(lsize_in_2, hidden_dim_2, mlpout_hd_2, num_layers=self.num_layers,
        #                                 para=para)
        # self.level2_gsvib_2 = FF_MLP(lsize_in_2, mlp_hidden_2, mlpout_hd_2, mlp_num_layers=3, para=para)
        # self.level2_gsvib_2 = FF_MLP_GSVIB(lsize_in_2, mlp_hidden_2, context_dim_2, gs_head_2, mlpout_hd_2,
        #                              mlp_num_layers=[mlp_num_layersl[0],0],para=para)

        self.mlp_out = FF_MLP(model_para_ff, para=paral[2])

        # self.level2_gsvib_3 = None
        ## Synergy Check
        # lsize_in_3 = lsize_in_1+lsize_in_2
        # self.level2_gsvib_3 = BiGRU_NLP_GSVIB(lsize_in_3, hidden_dim_2, context_dim_2, gs_head_2, mlp_hidden_2,
        #                                       self.output_size, num_layers=self.num_layers, para=para)
        # self.level2_gsvib_3 = FF_MLP(lsize_in_3, mlp_hidden_2, self.output_size, mlp_num_layers=3, para=para)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.save_para = {
            "lv1_coop": level1_coop.save_para,
            "level2_train_1": self.level2_gsvib_1.save_para,
            "level2_train_2": self.level2_gsvib_2.save_para,
            "type": "LEVEL2_BiGRU_NLP_GSVIB"
        }


    def para(self,para):
        self.freeze_mode=para.get("freeze_mode",False)
        self.semantic_only_mode=para.get("semantic_only_mode",False) # Semantic only, for next step when syntax is independent with sentiment is knowm

    def forward(self, inputl, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None):

        # if not self.freeze_mode:
        #     temperature = np.exp(-schedule * 5)
        # else:
        #     temperature = np.exp(-5)

        self.cuda_device = inputl[0].device

        # hidden1 [level1_coop hidden, BiGRU1 hidden, BiGRU2 hidden]
        logit_level1, hn_level1 = self.level1_coop(inputl, hidden1[0], logit_mode=True, schedule=1.0)
        gasample_lv1_t, gasample_lv1_s = logit_level1

        logit_level2_1, hn_level2_1 = self.level2_gsvib_1(gasample_lv1_t, hidden1[1],logit_mode=True,schedule=schedule)

        logit_level2_2, hn_level2_2 = self.level2_gsvib_2(gasample_lv1_s, hidden1[2], logit_mode=True, schedule=schedule)

        # if self.level2_gsvib_3 is not None:
        #     # assert self.semantic_only_mode
        #     gasample_lv1=torch.cat((gasample_lv1_t,gasample_lv1_s),dim=-1)
        #     logit_level2_3, hn_level2_3 = self.level2_gsvib_3(gasample_lv1, None, logit_mode=True, schedule=schedule)
        #     # self.context = torch.cat((self.level2_gsvib_1.context, self.level2_gsvib_2.context, self.level2_gsvib_3.context), dim=-2)
        #     # self.p_prior = torch.cat((self.level2_gsvib_1.p_prior, self.level2_gsvib_2.p_prior, self.level2_gsvib_3.p_prior), dim=-2)
        #     # self.gssample = torch.cat((self.level2_gsvib_1.gssample, self.level2_gsvib_2.gssample, self.level2_gsvib_3.gssample), dim=-2)
        #     # self.loss_intf = [self.context, self.p_prior]
        # #
        # # # gasample_lv1_a = torch.cat((gasample_lv1_t,gasample_lv1_s),dim=-1)
        # # # gasample_lv1_a = self.layer_norm3(gasample_lv1_a)
        # # # logit_level2_3, hn_level2_3 = self.level2_gsvib_3(gasample_lv1_a, hidden1[2], logit_mode=True, schedule=schedule)
        # else:
        #     # self.context = torch.cat((self.level2_gsvib_1.context,self.level2_gsvib_2.context),dim=-2)
        #     # self.p_prior = torch.cat((self.level2_gsvib_1.p_prior,self.level2_gsvib_2.p_prior),dim=-2)
        #     # self.gssample = torch.cat((self.level2_gsvib_1.gssample,self.level2_gsvib_2.gssample),dim=-2)
        #     self.context = self.level2_gsvib_2.context
        #     self.p_prior = self.level2_gsvib_2.p_prior
        #     self.gssample = self.level2_gsvib_2.gssample
        #     self.loss_intf = [self.context, self.p_prior]

        # if not self.semantic_only_mode:
        #     output = logit_level2_1 + logit_level2_2
        # elif self.level2_gsvib_3 is not None:
        #     output = logit_level2_2 + logit_level2_3
        # else:
        #     output = logit_level2_2

        self.context = torch.cat((self.level2_gsvib_1.context, self.level2_gsvib_2.context), dim=-2)
        self.p_prior = torch.cat((self.level2_gsvib_1.p_prior, self.level2_gsvib_2.p_prior), dim=-2)
        self.gssample = torch.cat((self.level2_gsvib_1.gssample, self.level2_gsvib_2.gssample), dim=-2)
        self.loss_intf = [self.context, self.p_prior]

        # self.cmu = torch.cat((self.level2_gsvib_1.cmu, self.level2_gsvib_2.cmu), dim=-1)
        # self.ctheta = torch.cat((self.level2_gsvib_1.ctheta, self.level2_gsvib_2.ctheta), dim=-1)
        # # self.prior = torch.cat((self.level2_gsvib_1.prior, self.level2_gsvib_2.prior), dim=-1)
        # self.loss_intf = [self.ctheta, self.cmu]
        # self.context = torch.cat((self.level2_gsvib_1.context, self.level2_gsvib_2.context), dim=-1)

        output = torch.cat((logit_level2_1,logit_level2_2),dim=-1)
        output, _ = self.mlp_out(output,None,logit_mode=True, schedule=None)

        if add_logit is not None:
            output = output + add_logit
        if not logit_mode:
            output = self.softmax(output)

        # return output,[hn_level2_1,hn_level2_2]

        return output, [hn_level1, hn_level2_1, hn_level2_2]
        # return output, [hn_level1, None]

    def initHidden(self,batch):
        return [self.level1_coop.initHidden(batch),self.level2_gsvib_1.initHidden(batch), self.level2_gsvib_2.initHidden(batch)]
        # return [self.level2_gsvib_1.initHidden(batch), self.level2_gsvib_2.initHidden(batch)]
        # return [self.level1_coop.initHidden(batch), None]

    def initHidden_cuda(self,device, batch):
        return [self.level1_coop.initHidden_cuda(device, batch),
            self.level2_gsvib_1.initHidden_cuda(device, batch),
            self.level2_gsvib_2.initHidden_cuda(device, batch)]
        # return [self.level2_gsvib_1.initHidden_cuda(device, batch),
        #         self.level2_gsvib_2.initHidden_cuda(device, batch)]
        # return [self.level1_coop.initHidden_cuda(device, batch), None]

class ABS_NLP_COOP_LOGIT(torch.nn.Module):
    """
    An abstrac cooperation container for logit cooperation
    """

    def __init__(self, trainer, cooprer, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.model_para = {"trainer":trainer.model_para,"cooprer":cooprer.model_para}
        self.misc_para = {"trainer":trainer.misc_para,"cooprer":cooprer.misc_para}

        self.coop_mode = False # For recursive cooreration

        self.trainer = trainer
        self.cooprer = cooprer
        self.l_size_tot = (self.trainer.gs_head_num + self.cooprer.gs_head_num) * self.trainer.gs_head_dim
        try:
            self.hsoftmax_depth = self.trainer.hsoftmax_depth
            self.output_cluster = self.trainer.output_cluster
        except:
            pass

        for param in self.cooprer.parameters():
            param.requires_grad = False

        if para is None:
            para = dict([])
        self.para(para)

        if self.cooprer_attn_mode:
            context_attn_size = self.cooprer.gs_head_num * self.cooprer.gs_head_dim
            self.cooprer_attn= torch.nn.Linear(context_attn_size, self.trainer.gs_head_num) # Attention from pos tag to representation, key to control hierachy/combinatory

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.gsigmoid = Gumbel_Sigmoid()

        self.logit_train=None

        self.save_para = {
            "trainer": trainer.save_para,
            "cooprer": cooprer.save_para,
            "type": "ABS_NLP_COOP_LOGIT",
            "misc_para": para
        }

    def para(self,para):
        self.cooprer_attn_mode=para.get("cooprer_attn_mode",False)
        self.freeze_mode=para.get("freeze_mode",False)
        self.coop_mode=para.get("coop_mode",False)

    def forward(self, inputl, hidden1, add_logit=None, logit_mode=True, schedule=None, coop_context_set=None):

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)
        try:
            self.cuda_device = inputl.device
            dw, batch, l_size = inputl.shape
        except:
            self.cuda_device = inputl[0].device
            dw, batch, l_size = inputl[0].shape

        logit_coop, hn_coop = self.cooprer(inputl, hidden1[1], logit_mode=True, schedule=1.0,
                                           context_set=coop_context_set)
        if self.cooprer_attn_mode:
            dw,dbatch,gs_head_num,gs_head_dim=torch.squeeze(self.cooprer.gssample).shape
            attention = self.cooprer_attn(self.cooprer.gssample.view(dw,dbatch,gs_head_num*gs_head_dim))
            attention_sig = self.gsigmoid(attention, temperature=temperature, cuda_device=self.cuda_device)
        else:
            attention_sig = None
        logit_train, hn_train = self.trainer(inputl, hidden1[0], logit_mode=True, schedule=schedule,
                                             attention_sig=attention_sig)

        # logit_train, hn_train = self.trainer(inputl[0], hidden1[0], logit_mode=True, schedule=schedule)
        # logit_coop, hn_coop = self.cooprer(inputl[1], hidden1[1], logit_mode=True, schedule=1.0,
        #                                    context_set=coop_context_set)
        self.loss_intf = self.trainer.loss_intf
        # print(self.loss_intf)
        self.context=self.trainer.context
        self.gssample=self.trainer.gssample
        # self.ctheta = self.trainer.ctheta
        # self.cmu = self.trainer.cmu
        self.context_coop = self.cooprer.context
        self.gssample_coop = self.cooprer.gssample

        if self.coop_mode:
            # return [logit_coop,logit_train], None # Coop is label, train is self
            gasample_lv1 = torch.cat((self.trainer.gssample, self.cooprer.gssample), dim=-2)
            return gasample_lv1.view(dw,batch,self.l_size_tot), None  # Coop is label, train is self

        # output=logit_coop[0]+logit_train[0]
        output = logit_coop + logit_train
        self.logit_train=logit_train

        if add_logit is not None:
            output = output + add_logit

        # return [output,logit_train[1]], [hn_train, hn_coop]
        return output, [hn_train, hn_coop]

    # def forward_comb(self,context_coop):
    #     """
    #     forward with substitue coop
    #     :param context_coop:
    #     :return:
    #     """
    #     logit_coop=self.cooprer.c2o(context_coop)
    #     output = logit_coop + self.logit_train
    #     output = self.softmax(output)
    #     return output

    def initHidden(self,batch):
        trainer_hd = None
        if self.trainer is not None:
            trainer_hd=self.trainer.initHidden(batch)
        return [trainer_hd,self.cooprer.initHidden(batch)]

    def initHidden_cuda(self,device, batch):
        trainer_hd = None
        if self.trainer is not None:
            trainer_hd = self.trainer.initHidden_cuda(device, batch)
        return [trainer_hd, self.cooprer.initHidden_cuda(device, batch)]

class ABS_NLP_COOP(torch.nn.Module):
    """
    An abstract cooperation container
    """

    def __init__(self, trainer, cooprer, output_size=None, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.coop_mode = False # For recursive cooreration

        self.trainer = trainer
        self.cooprer = cooprer
        self.cooprer.coop_mode = True

        if self.trainer is not None:
            self.context_size=self.cooprer.context_size+self.trainer.context_size
            self.trainer.coop_mode = True
            self.c2o = torch.nn.Linear(self.context_size, trainer.output_size)
        else:
            self.context_size = self.cooprer.context_size
            self.c2o = torch.nn.Linear(self.context_size, output_size)

        for param in self.cooprer.parameters():
            param.requires_grad = False

        if para is None:
            para = dict([])
        self.para(para)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

    def para(self,para):
        pass

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):

        context_coop, hn_coop = self.cooprer(input, hidden1[1], schedule=1.0)
        if self.trainer is not None:
            context_train, hn_train = self.trainer(input, hidden1[0], schedule=schedule)
            self.loss_intf = self.trainer.loss_intf
            context_cat = torch.cat((context_coop, context_train), dim=-1)
            self.context_coop=context_coop
            self.context_train=context_train
        else:
            context_cat = context_coop
            hn_train = None

        self.context=context_cat

        if self.coop_mode:
            return context_cat, [hn_train, hn_coop]

        output = self.c2o(context_cat)

        if add_logit is not None:
            output = output + add_logit
        if not logit_mode:
            output = self.softmax(output)
        return output, [hn_train, hn_coop]

    def forward_comb(self,context_coop):
        context_cat = torch.cat((context_coop, self.context_train), dim=-1)
        output = self.c2o(context_cat)
        output = self.softmax(output)
        return output

    def initHidden(self,batch):
        trainer_hd = None
        if self.trainer is not None:
            trainer_hd=self.trainer.initHidden(batch)
        return [trainer_hd,self.cooprer.initHidden(batch)]

    def initHidden_cuda(self,device, batch):
        trainer_hd = None
        if self.trainer is not None:
            trainer_hd = self.trainer.initHidden_cuda(device, batch)
        return [trainer_hd, self.cooprer.initHidden_cuda(device, batch)]


class BiGRU_NLP_COOP(torch.nn.Module):
    """
    PyTorch Bi-GRU for NLP cooperative version, for layer seperation of natural language
    """
    def __init__(self, input_size, hidden_size, context_size, output_size, bigru_coop, para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.context_size=context_size

        self.bigru_coop=bigru_coop
        self.bigru_coop.coop_mode=True
        for param in self.bigru_coop.parameters():
            param.requires_grad = False

        if para is None:
            para = dict([])
        self.para(para)

        self.gru=torch.nn.GRU(input_size,hidden_size,bidirectional=True)
        self.h2c = torch.nn.Linear(hidden_size * 2, context_size)
        self.h2g = torch.nn.Linear(hidden_size * 2, context_size)  # from hidden to gating
        self.c2o = torch.nn.Linear(context_size+self.bigru_coop.context_size, output_size)

        if self.weight_dropout>0:
            print("Be careful, only GPU works for now.")
            # self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=self.weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsigmoid = Gumbel_Sigmoid(cuda_device=self.cuda_device)

        self.context = None
        self.siggate = None

        self.batch_size = 1
        self.pad = torch.zeros((1, self.batch_size, self.input_size)).to(self.cuda_device)

    def para(self,para):
        self.weight_dropout = para.get("weight_dropout", 0.0)
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.self_include_flag=para.get("self_include_flag", False)
        self.precision = para.get("precision", 0.1)
        self.gate_mode_flag = para.get("gate_mode_flag", True)

    def para_copy(self,bigru):
        """
        Copy parameter from another bigru
        :param gru:
        :return:
        """
        print("Copying data from "+str(bigru))
        self.load_state_dict(bigru.state_dict())
        # allname = ["h2c", "c2o"]
        # for nn,nameitem in enumerate(allname):
        #     rnn_w = getattr(self, nameitem).weight
        #     rnn_w.data.copy_(getattr(bigru, nameitem).weight.data)
        #     rnn_b = getattr(self, nameitem).bias
        #     rnn_b.data.copy_(getattr(bigru, nameitem).bias.data)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        temperature = np.exp(-schedule * 5)
        if len(input.shape)==2:
            input=input.view(1,input.shape[0],input.shape[1])
        b_size = input.shape[1]
        if self.batch_size != b_size:
            self.batch_size = b_size
            self.pad = torch.zeros((1, b_size, self.input_size)).to(self.cuda_device)
        if not self.self_include_flag:
            input2=torch.cat((self.pad, input, self.pad), dim=0)
        else:
            input2=input
        hout, hn = self.gru(input2,hidden1[0])
        # if not self.self_include:
        #     hout_leftshf=torch.cat((hout[2:,:,self.hidden_size:],self.pad),dim=0)
        #     hout=torch.cat((hout[:,:,:self.hidden_size],hout_leftshf),dim=-1)
        if not self.self_include_flag:
            hout_forward = hout[:-2,:, :self.hidden_size]
            hout_backward = hout[2:, :, self.hidden_size:]
            hout=torch.cat((hout_forward,hout_backward),dim=-1)
        context = self.h2c(hout)
        context = self.relu(context)
        if self.gate_mode_flag:
            ## Adaptive Gating
            gatev=self.h2g(hout)
            siggate = self.gsigmoid(gatev, temperature=temperature)
            # siggate = self.sigmoid(gatev)
            self.siggate=siggate
            context=context*siggate
        context = mydiscrete(context, self.precision, cuda_device=self.cuda_device)
        self.context=context

        context_coop,hn_coop = self.bigru_coop(input,hidden1[1],schedule=1.0)
        context_cat=torch.cat((context_coop,context),dim=-1)

        output = self.c2o(context_cat)

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)
        return output,[hn,hn_coop]

    def forward_comb(self,context_coop):
        context_cat = torch.cat((context_coop, self.context), dim=-1)
        output = self.c2o(context_cat)
        output = self.softmax(output)
        return output


    def initHidden(self,batch):
        return [Variable(torch.zeros(2, batch,self.hidden_size), requires_grad=True),
                Variable(torch.zeros(2, batch, self.bigru_coop.hidden_size), requires_grad=False)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(2, batch, self.hidden_size), requires_grad=True).to(device),
                Variable(torch.zeros(2, batch, self.bigru_coop.hidden_size), requires_grad=False).to(device)]

class BiGRU_NLP(torch.nn.Module):
    """
    PyTorch GRU for NLP
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.gru=torch.nn.GRU(self.input_size,self.hidden_size,num_layers=self.num_layers,bidirectional=True)

        if self.mlp_num_layers>0:
            self.h2h = torch.nn.Linear(2*self.hidden_size, self.mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=True), torch.nn.LayerNorm(self.mlp_hidden),torch.nn.ReLU())
                for _ in range(self.mlp_num_layers-1)])
            self.h2o = torch.nn.Linear(self.mlp_hidden, self.output_size)
        else:
            self.h2o = torch.nn.Linear(2*self.hidden_size, self.output_size)

        if self.weight_dropout>0:
            print("Be careful, only GPU works for now.")
            # self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=self.weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.input_mask = None
        self.out_mask = None

        self.loss_intf=None

        self.save_para = {
            "model_para": model_para,
            "type": "BiGRU_NLP",
            "misc_para": para
        }

    def set_model_para(self,model_para):
        # model_para_bigrunlp = {
        #     "input_size": 300,
        #     "output_size": 43,
        #     "hidden_size": 30,
        #     "num_layers": 3,
        #     "mlp_hidden": 80,
        #     "mlp_num_layers": 0
        # }
        self.model_para = model_para
        self.hidden_size = model_para["hidden_size"]
        self.input_size = model_para["input_size"]
        self.output_size = model_para["output_size"]
        self.num_layers = model_para["num_layers"]
        self.mlp_hidden = int(model_para.get("mlp_hidden", 20))
        self.mlp_num_layers = int(model_para.get("mlp_num_layers", 1))

    def para(self,para):
        self.misc_para = para
        self.weight_dropout = para.get("weight_dropout", 0.0)
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.dropout_rate = para.get("dropout_rate", 0.2)
        self.input_mask_mode = para.get("input_mask_mode", False)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.self_unmask_rate = para.get("self_unmask_rate", 0.1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        # temperature = np.exp(-schedule * 5)
        # if len(input.shape)==2:
        #     input=input.view(1,input.shape[0],input.shape[1])
        self.cuda_device = input.device

        if self.input_mask_mode and self.training:
            rnd = torch.rand((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask = torch.zeros((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask[rnd > self.mask_rate] = 1
            # Self unmask sampling
            rnd2 = torch.rand((input.shape[0], input.shape[1]), device=self.cuda_device)
            self_unmask = torch.zeros((input.shape[0], input.shape[1]), device=self.cuda_device)
            self_unmask[rnd2 < self.self_unmask_rate] = 1
            comb_mask = self.input_mask + self_unmask
            comb_mask[comb_mask > 0.999] = 1
            input = input * comb_mask.view(input.shape[0], input.shape[1], 1).expand_as(input)

        hout, hn = self.gru(input,hidden1)
        hout = self.dropout(hout)

        if self.mlp_num_layers > 0:
            hout = self.h2h(hout)
            hout =self.relu(hout)
            for fmd in self.linear_layer_stack:
                hout = fmd(hout)

        output = self.h2o(hout)

        if add_logit is not None:
            output=output+add_logit
        if not logit_mode:
            output=self.softmax(output)

        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(2*self.num_layers, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(2*self.num_layers, batch, self.hidden_size), requires_grad=True).to(device)

class BiGRU_NLP_DIS(torch.nn.Module):
    """
    PyTorch GRU for NLP with discrete layer
    """
    def __init__(self, input_size, hidden_size, context_size, output_size, para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.context_size=context_size

        self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)

        self.gru=torch.nn.GRU(input_size,hidden_size,bidirectional=True)
        self.h2c = torch.nn.Linear(hidden_size*2, context_size)
        self.h2g = torch.nn.Linear(hidden_size * 2, context_size) # from hidden to gating
        self.c2o = torch.nn.Linear(context_size, output_size)

        if self.adv_mode_flag: # Adversary mode
            self.c2ao = torch.nn.Linear(context_size, self.adv_output_size)


        if self.weight_dropout>0:
            print("Be careful, only GPU works for now.")
            # self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=self.weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsigmoid = Gumbel_Sigmoid(cuda_device=self.cuda_device)

        self.context = None
        self.siggate = None
        self.loss_intf=[self.context,self.siggate]

        self.batch_size = 1
        self.pad = torch.zeros((1, self.batch_size, self.input_size)).to(self.cuda_device)

    def para(self,para):
        self.weight_dropout = para.get("weight_dropout", 0.0)
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.self_include_flag = para.get("self_include_flag", False)
        self.gate_mode_flag = para.get("gate_mode_flag", True)
        self.adv_mode_flag = para.get("adv_mode_flag", False)
        self.adv_output_size = para.get("adv_output_size", None)
        self.precision = para.get("precision", 0.1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        temperature = np.exp(-schedule * 5)
        if len(input.shape)==2:
            input=input.view(1,input.shape[0],input.shape[1])
        b_size = input.shape[1]
        if self.batch_size != b_size:
            self.batch_size = b_size
            self.pad = torch.zeros((1, b_size, self.input_size)).to(self.cuda_device)
        if not self.self_include_flag:
            input=torch.cat((self.pad, input, self.pad), dim=0)
        hout, hn = self.gru(input,hidden1)
        # if not self.self_include:
        #     hout_leftshf=torch.cat((hout[2:,:,self.hidden_size:],self.pad),dim=0)
        #     hout=torch.cat((hout[:,:,:self.hidden_size],hout_leftshf),dim=-1)
        if not self.self_include_flag:
            hout_forward = hout[:-2,:, :self.hidden_size]
            hout_backward = hout[2:, :, self.hidden_size:]
            hout=torch.cat((hout_forward,hout_backward),dim=-1)
        context = self.h2c(hout)
        context = self.relu(context)
        if self.gate_mode_flag:
            ## Adaptive Gating
            gatev=self.h2g(hout)
            siggate = self.gsigmoid(gatev, temperature=temperature)
            # siggate = self.sigmoid(gatev)
            self.siggate=siggate
            context=context*siggate
        context = mydiscrete(context, self.precision, cuda_device=self.cuda_device)
        self.context=context
        self.loss_intf = [self.context, self.siggate]
        if self.coop_mode:
            return context,hn
        output = self.c2o(context)
        output = self.softmax(output)

        if self.adv_mode_flag:
            adoutput = self.c2ao(context)
            adoutput = self.softmax(adoutput)
            return [output,adoutput], hn

        # if add_logit is not None:
        #     output=output+add_logit
        # if not logit_mode:
        #     output=self.softmax(output)
        return output,hn

    def initHidden(self,batch):
        return Variable(torch.zeros(2, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(2, batch, self.hidden_size), requires_grad=True).to(device)

class VariationalGauss(torch.nn.Module):
    """
    A gaussian noise module
    """
    def __init__(self, multi_sample_flag=False, sample_size=None):

        super(self.__class__, self).__init__()

        # self.noise = None
        self.sample_size=sample_size
        self.multi_sample_flag=multi_sample_flag

    def forward(self, mu, theta):
        assert mu.shape==theta.shape
        device = mu.device
        shape = np.array(mu.shape)
        if self.multi_sample_flag:
            shape=np.insert(shape,2,self.sample_size)
            mushape=copy.copy(shape)
            mushape[2]=1
            mu=mu.view(tuple(mushape))
            theta = theta.view(tuple(mushape))
        noise = torch.nn.init.normal_(torch.empty(tuple(shape))).to(device)
        # if self.noise is None:
        #     self.noise = torch.nn.init.normal_(torch.empty(tuple(shape))).to(device)
        # else:
        #     self.noise.data.normal_(0,std=1.0)

        if self.training:
            return mu + theta * noise
        else:
            return mu + theta * noise

class FF_VIB(torch.nn.Module):
    """
    PyTorch BiGRU for NLP with variational information bottleneck
    """
    def __init__(self, input_size, mlp_hidden, context_size, output_size, mlp_num_layers = [2, 2],para=None):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.mlp_hidden = mlp_hidden
        self.mlp_num_layers = mlp_num_layers

        if para is None:
            para = dict([])
        self.para(para)

        self.i2mu = torch.nn.Linear(input_size, context_size)
        self.i2t = torch.nn.Linear(input_size, context_size)
        self.c2h = torch.nn.Linear(context_size, mlp_hidden)
        self.h2o = torch.nn.Linear(mlp_hidden, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.softplus = torch.nn.Softplus()
        self.vagauss = VariationalGauss(cuda_device=self.cuda_device,multi_sample_flag=self.multi_sample_flag,
                                        sample_size=self.sample_size)

        self.context = None
        self.ctheta = None
        self.cmu = None

        self.loss_intf = [self.ctheta, self.cmu]

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.sample_size = para.get("sample_size", 16)
        self.multi_sample_flag = para.get("multi_sample_flag", True)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        # temperature = np.exp(-schedule * 5)

        cmu = self.i2mu(input)
        ctheta = self.i2t(input)
        ctheta = self.softplus(ctheta)
        context = self.vagauss(cmu,ctheta)
        self.cmu=cmu
        self.ctheta = ctheta
        self.context = context
        self.loss_intf = [self.ctheta, self.cmu]
        hidden = self.c2h(context)
        hidden = self.tanh(hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)

        return output, None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None

class BiGRU_NLP_VIB(torch.nn.Module):
    """
    PyTorch BiGRU for NLP with variational information bottleneck
    """
    def __init__(self, input_size, hidden_size, context_size, mlp_hidden, output_size, num_layers=1 ,para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.context_size=context_size
        self.mlp_hidden=mlp_hidden
        self.num_layers=num_layers

        # self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)

        if num_layers>1:
            assert self.self_include_flag

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers,bidirectional=True)
        self.h2mu = torch.nn.Linear(hidden_size * 2, context_size)
        self.h2t = torch.nn.Linear(hidden_size * 2, context_size) # from hidden to gating
        self.prior = torch.nn.Parameter(torch.ones(context_size), requires_grad=True)

        if self.mlp_num_layers>0:
            self.c2h = torch.nn.Linear(context_size, mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True), torch.nn.Tanh())
                for _ in range(self.mlp_num_layers)])
            self.h2o = torch.nn.Linear(mlp_hidden, output_size)
        else:
            self.c2o = torch.nn.Linear(context_size, output_size)

        if self.adv_mode_flag: # Adversary mode
            self.c2ao = torch.nn.Linear(context_size, self.adv_output_size)


        if self.weight_dropout>0:
            print("Be careful, only GPU works for now.")
            # self.h2o = WeightDrop(self.h2o, ['weight'], dropout=weight_dropout)
            self.gru = WeightDrop(self.gru, ['weight_hh_l0'], dropout=self.weight_dropout)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        # self.gsigmoid = Gumbel_Sigmoid(cuda_device=self.cuda_device)
        self.softplus = torch.nn.Softplus()
        self.vagauss = VariationalGauss(multi_sample_flag=self.multi_sample_flag,sample_size=self.sample_size)

        self.context = None
        self.ctheta = None
        self.cmu = None
        self.loss_intf = [self.ctheta, self.cmu]  # Mainly for ABS_COOP interface

        self.batch_size = 1
        # self.pad = torch.zeros((1, self.batch_size, self.input_size)).to(self.cuda_device)

        # self.cuda_device=None
        self.coop_mode = False

    def para(self,para):
        self.weight_dropout = para.get("weight_dropout", 0.0)
        # self.cuda_device = para.get("cuda_device", "cuda:0")
        self.self_include_flag = para.get("self_include_flag", True)
        self.adv_mode_flag = para.get("adv_mode_flag", False)
        self.adv_output_size = para.get("adv_output_size", None)
        # self.precision = para.get("precision", 0.1)
        self.rnd_input_mask = para.get("rnd_input_mask", False)
        self.mask_rate = para.get("mask_rate", 0.0)
        self.sample_size = para.get("sample_size", 16)
        self.multi_sample_flag = para.get("multi_sample_flag", False)
        self.mlp_num_layers = int(para.get("mlp_num_layers", 0))

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        # temperature = np.exp(-schedule * 5)

        # if self.cuda_device is None:
        self.cuda_device=input.device

        if len(input.shape)==2:
            input=input.view(1,input.shape[0],input.shape[1])
        b_size = input.shape[1]
        if self.batch_size != b_size:
            self.batch_size = b_size
            self.pad = torch.zeros((1, b_size, self.input_size)).to(self.cuda_device)
            self.vagauss.noise=None

        if self.rnd_input_mask:
            rnd = torch.rand((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask = torch.zeros((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask[rnd > self.mask_rate] = 1
            input = input * self.input_mask.view(input.shape[0], input.shape[1], 1).expand_as(input)

        if not self.self_include_flag:
            input=torch.cat((self.pad, input, self.pad), dim=0)
        hout, hn = self.gru(input,hidden1)
        # if not self.self_include:
        #     hout_leftshf=torch.cat((hout[2:,:,self.hidden_size:],self.pad),dim=0)
        #     hout=torch.cat((hout[:,:,:self.hidden_size],hout_leftshf),dim=-1)
        if not self.self_include_flag:
            hout_forward = hout[:-2,:, :self.hidden_size]
            hout_backward = hout[2:, :, self.hidden_size:]
            hout=torch.cat((hout_forward,hout_backward),dim=-1)
        cmu = self.h2mu(hout)
        ctheta = self.h2t(hout)
        ctheta = self.softplus(ctheta)
        context = self.vagauss(cmu,ctheta,cuda_device=self.cuda_device)
        self.cmu=cmu
        self.ctheta = ctheta
        self.loss_intf = [self.ctheta, self.cmu]
        self.context = context
        if self.coop_mode:
            return context,hn
        if self.mlp_num_layers > 0:
            hidden_c=self.c2h(context)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            output=self.h2o(hidden_c)
        else:
            output = self.c2o(context)
        output = self.softmax(output)

        if self.adv_mode_flag:
            adoutput = self.c2ao(context)
            adoutput = self.softmax(adoutput)
            return [output,adoutput], hn

        # if add_logit is not None:
        #     output=output+add_logit
        # if not logit_mode:
        #     output=self.softmax(output)
        # return [output,self.loss_intf],hn
        return output, hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers*2, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(self.num_layers*2, batch, self.hidden_size), requires_grad=True).to(device)

class Gumbel_Softmax(torch.nn.Module):
    """
    PyTorch Gumbel softmax function
    Categorical Reprarameterization with Gumbel-Softmax
    """
    def __init__(self,sample=1, sample_mode=False):
        super(self.__class__, self).__init__()
        self.sample=sample
        self.sample_mode=sample_mode

    def forward(self, inmat, temperature=1.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        # input must be log probability
        device = inmat.device
        if self.sample_mode:
            ishape=list(inmat.shape)
            vshape=list(inmat.shape)
            ishape.insert(2, self.sample) # dw, batch, sample, ...
            vshape.insert(2, self.sample)
            inmat=torch.unsqueeze(inmat, 2).expand(ishape)
        ui = torch.rand(inmat.shape)
        ui = ui.to(device)
        gi = -torch.log(-torch.log(ui))
        betax1 = (inmat + gi) / temperature
        self.betax1 = betax1
        betam,_=torch.max(betax1,-1,keepdim=True)
        betax = betax1 - betam
        self.betax=betax
        yi=torch.exp(betax)/torch.sum(torch.exp(betax),dim=-1,keepdim=True)
        return yi

class LinearProb(torch.nn.Module): ## Not useful
    """
    PyTorch linear probability calculation method.
    Aims at breaking softmax information bottleneck
    """
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, inmat):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        # inmat_shift = inmat - torch.min(inmat,dim=-1,keepdim=True)[0]
        inmat_shift = torch.relu(inmat)
        lprob = inmat_shift/torch.sum(inmat_shift,dim=-1,keepdim=True)
        return lprob

class HighRank_Softmax(torch.nn.Module):
    """
    PyTorch High Rank Softmax
    Breaking Softmax bottleneck ...
    Aims at breaking softmax information bottleneck

    """
    def __init__(self,input_size,output_size,num_sfm=3, bias=True):
        super(self.__class__, self).__init__()
        self.input_size=input_size
        self.output_size = output_size
        self.num_sfm=num_sfm
        self.psoftmax = torch.nn.Softmax(dim=-1)
        self.i2pi = torch.nn.Linear(input_size, num_sfm)
        self.i2sfms = torch.nn.Parameter(torch.Tensor(output_size, num_sfm, input_size), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.i2sfms, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.i2sfms)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_sfm, output_size), requires_grad=True)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, inmat):
        """
        Forward
        :param input:[w,b,s,input_size]
        :param hidden:
        :return:
        """
        pimat = self.i2pi(inmat)
        pimat = self.psoftmax(pimat) # [w,b,s,num_sfm]
        sfmout=torch.matmul(inmat.unsqueeze(-3),self.i2sfms.permute(1,2,0)).transpose(-3,-2) #[w,b,num_sfm,s,output_size]
        sfmout=sfmout+self.bias
        sfmout = self.psoftmax(sfmout)
        # print(pimat.shape,sfmout.shape)
        pi_sfmout = torch.matmul(pimat.unsqueeze(-2),sfmout).squeeze()
        pi_sfmout = torch.log(pi_sfmout+1e-9)
        return pi_sfmout

class FF_MLP(torch.nn.Module):
    """
    Feed forward multi-layer perceotron
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        # self.coop_mode=False
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        if len(self.mlp_layer_para)>0:
            mlp_layer_para0 = [self.input_size] + self.mlp_layer_para
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_layer_para0[iil], mlp_layer_para0[iil+1], bias=True),
                                    torch.nn.LayerNorm(mlp_layer_para0[iil+1]),torch.nn.ReLU())
                for iil in range(len(mlp_layer_para0)-1)])
            self.h2o = torch.nn.Linear(mlp_layer_para0[-1], self.output_size)
        else:
            self.i2o = torch.nn.Linear(self.input_size, self.output_size)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.save_para = {
            "model_para": model_para,
            "type": "FF_MLP",
            "misc_para": para
        }

    def para(self,para):
        self.misc_para=para
        self.dropout_rate = para.get("dropout_rate", 0.0)

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 64,
        #     "output_size":1,
        #     "mlp_layer_para": [32,16,8]
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.output_size = model_para["output_size"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0, hidden1, ...]

    def forward(self, inputx, hidden1=None, add_logit=None, logit_mode=True, schedule=None, context_set=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        self.cuda_device=inputx.device

        # print(torch.mean(inputx), torch.var(inputx))
        inputx = (inputx - 0.5) / np.sqrt(0.25)

        hidden = inputx
        if len(self.mlp_layer_para) > 0:
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
                # hidden = self.dropout(hidden)
            output=self.h2o(hidden)
        else:
            output = self.i2o(inputx)

        return output

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self, device, batch):
        return None

class FF_MLP_VIB(torch.nn.Module):
    """
    Feed forward multi-layer perceotron
    """
    def __init__(self, input_size, mlp_hidden, context_size, output_size, mlp_num_layers=[2,2] ,para=None):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.mlp_hidden=mlp_hidden
        self.mlp_num_layers = mlp_num_layers[0]
        self.mlp_num_layers_2 = mlp_num_layers[1]

        if para is None:
            para = dict([])
        self.para(para)

        if self.mlp_num_layers>0:
            self.i2h = torch.nn.Linear(input_size, mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True), torch.nn.LayerNorm(mlp_hidden),torch.nn.ReLU())
                for _ in range(self.mlp_num_layers-1)])
            self.h2mu = torch.nn.Linear(mlp_hidden, context_size)
            self.h2t = torch.nn.Linear(mlp_hidden, context_size)
        else:
            self.i2mu = torch.nn.Linear(input_size, context_size)
            self.i2t = torch.nn.Linear(input_size, context_size)

        if self.mlp_num_layers_2>0:
            self.c2h_2 = torch.nn.Linear(context_size, mlp_hidden)
            self.linear_layer_stack_2 = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True), torch.nn.LayerNorm(mlp_hidden),torch.nn.ReLU())
                for _ in range(self.mlp_num_layers-1)])
            self.h2o_2 = torch.nn.Linear(mlp_hidden, output_size)
        else:
            self.c2o_2 = torch.nn.Linear(context_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.softplus = torch.nn.Softplus()
        self.vagauss = VariationalGauss(multi_sample_flag=self.multi_sample_flag, sample_size=self.sample_size)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def para(self,para):
        self.dropout_rate = para.get("dropout_rate", 0.0)
        self.freeze_mode = para.get("freeze_mode", False)
        self.coop_mode = para.get("coop_mode", False)
        self.multi_sample_flag =  para.get("multi_sample_flag", False)
        self.sample_size = para.get("sample_size", 1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None, attention_sig=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        self.cuda_device=input.device

        if self.mlp_num_layers > 0:
            hidden=self.i2h(input)
            for fmd in self.linear_layer_stack:
                # hidden = self.dropout(hidden)
                hidden = fmd(hidden)
            cmu=self.h2mu(hidden)
            ctheta = self.h2t(hidden)
        else:
            cmu = self.h2mu(input)
            ctheta = self.h2t(input)

        ctheta = self.softplus(ctheta)
        context = self.vagauss(cmu, ctheta, cuda_device=self.cuda_device)

        self.cmu = cmu
        self.ctheta = ctheta
        self.loss_intf = [self.ctheta, self.cmu]
        self.context = context

        if self.coop_mode :
            return context, None

        if self.mlp_num_layers_2 > 0:
            hidden_c=self.c2h_2(context)
            # hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack_2:
                hidden_c = fmd(hidden_c)
            output=self.h2o_2(hidden_c)
        else:
            output = self.c2o_2(context)

        if not logit_mode:
            output = self.softmax(output)

        return output, None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self, device, batch):
        return None

class FF_MLP_GSVIB(torch.nn.Module):
    """
    Feed forward multi-layer perceptron
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        # self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)

        if len(self.mlp_layer_para)>0:
            mlp_layer_para0 = [self.input_size] + self.mlp_layer_para0
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_layer_para0[iil], mlp_layer_para0[iil + 1], bias=True),
                                    torch.nn.LayerNorm(mlp_layer_para0[iil + 1]), torch.nn.ReLU())
                for iil in range(len(mlp_layer_para0) - 1)])
            self.h2o = torch.nn.Linear(mlp_layer_para0[-1], self.gs_head_dim*self.gs_head_num)
        else:
            self.i2o = torch.nn.Linear(self.input_size, self.gs_head_dim*self.gs_head_num)

        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num, self.gs_head_dim)), requires_grad=True)

        if not self.coop_mode:
            if self.mlp_num_layers_2>0:
                self.c2h_2 = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.mlp_hidden)
                self.linear_layer_stack_2 = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=True), torch.nn.LayerNorm(self.mlp_hidden),torch.nn.ReLU())
                    for _ in range(self.mlp_num_layers-1)])
                self.h2o_2 = torch.nn.Linear(self.mlp_hidden, self.output_size)
                if self.num_sfm ==1:
                    self.c2o_2 = torch.nn.Linear(self.gs_head_dim * self.gs_head_num, self.output_size)
                else:
                    self.c2o_2 = HighRank_Softmax(self.gs_head_dim * self.gs_head_num, self.output_size,
                                                  num_sfm=self.num_sfm)
            else:
                if self.num_sfm == 1:
                    self.c2o_2 = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.output_size)
                else:
                    self.c2o_2 = HighRank_Softmax(self.gs_head_dim*self.gs_head_num, self.output_size, num_sfm=self.num_sfm)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size,sample_mode=True)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.layer_norm_coop = torch.nn.LayerNorm(self.gs_head_num * self.gs_head_dim)

        self.save_para = {
            "model_para": model_para,
            "type": "FF_MLP_GSVIB",
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
        self.output_size = model_para["output_size"]
        self.gs_head_dim = model_para["gs_head_dim"]
        self.gs_head_num = model_para["gs_head_num"]
        self.mlp_layer_para0 = model_para["mlp_layer_para0"] # [hidden0, hidden1 ...]
        self.mlp_layer_para1 = model_para["mlp_layer_para1"]  # [hidden0, hidden1 ...]

    def set_sample_size(self,sample_size):
        self.sample_size=sample_size
        sample_mode = False
        if sample_size>1:
            sample_mode = True
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size,sample_mode=sample_mode)

    def para(self,para):
        # misc_para = {
        #     "freeze_mode", False,
        #     "temp_scan_num", 4
        # }
        self.misc_para = para
        self.dropout_rate = para.get("dropout_rate", 0.0)
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.coop_mode = para.get("coop_mode", False)
        self.sample_size = para.get("sample_size",1)
        self.num_sfm = para.get("num_sfm",1)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None, attention_sig=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        self.cuda_device=input.device

        if schedule < 1.0:  # Multi-scanning trial
            schedule = schedule * self.temp_scan_num
            schedule = schedule - np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        if self.mlp_num_layers > 0:
            hidden=self.i2h(input)
            for fmd in self.linear_layer_stack:
                hidden = self.dropout(hidden)
                hidden = fmd(hidden)
            context=self.h2o(hidden)
        else:
            context = self.i2o(input)

        dw, batch, l_size = input.squeeze().shape
        context = context.view(dw, batch, self.gs_head_num, self.gs_head_dim)

        if attention_sig is not None:
            p_prior_exp = self.prior.expand(dw, batch, self.gs_head_num, self.gs_head_dim)
            # Sigmoid attention from POS
            context = context * attention_sig.view(dw, batch, self.gs_head_num, 1) + p_prior_exp * (
                    1 - attention_sig.view(dw, batch, self.gs_head_num, 1))
            context = self.softmax(context)

        context = self.softmax(context)
        if context_set is not None:
            context = context_set

        gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        p_prior = self.softmax(self.prior)
        self.context = context
        self.p_prior = p_prior
        self.gssample = gssample
        self.loss_intf = [self.context, p_prior]

        gssample = gssample.view(dw, batch, self.sample_size, self.gs_head_num * self.gs_head_dim)

        if self.coop_mode :
            coop_out = self.layer_norm_coop(gssample)
            return coop_out, None

        if self.mlp_num_layers_2 > 0:
            hidden_c=self.c2h_2(gssample)
            # hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack_2:
                hidden_c = fmd(hidden_c)
            output=self.h2o_2(hidden_c)
        else:
            output = self.c2o_2(gssample)

        # if not logit_mode:
        #     output = self.softmax(output)

        return output, None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self, device, batch):
        return None

class BiGRU_NLP_GSVIB(torch.nn.Module):
    """
    PyTorch BiGRU for NLP with Gumbel Softmax variational information bottleneck
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        # model_para = {"input_size": lsize_in,
        #               "output_size": vocab_out,
        #               "gs_head_dim": 2,
        #               "gs_head_num": 21,
        #               "bigru_hidden_size": 30,
        #               "mlp_num_layers": 0,
        #               "mlp_hidden": 80,
        #               "bigru_num_layers": 3}

        self.set_model_para(model_para)

        # self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)

        # self include by default
        # if num_layers>1:
        #     assert self.self_include_flag

        self.gru=torch.nn.GRU(self.input_size,self.hidden_size,num_layers=self.bigru_num_layers,bidirectional=True)
        self.h2c = torch.nn.Linear(self.hidden_size * 2, self.gs_head_dim*self.gs_head_num)
        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num,self.gs_head_dim)) , requires_grad=True)

        if not self.coop_mode:
            if self.mlp_num_layers>0:
                self.c2h = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.mlp_hidden)
                self.linear_layer_stack = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=True), torch.nn.LayerNorm(self.mlp_hidden),torch.nn.ReLU())
                    for _ in range(self.mlp_num_layers-1)])
                self.h2o = torch.nn.Linear(self.mlp_hidden, self.output_size*self.hsoftmax_depth)
            else:
                self.c2o = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.output_size*self.hsoftmax_depth)
        else:
            self.layer_norm_coop = torch.nn.LayerNorm(self.gs_head_num * self.gs_head_dim)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.save_para={
            "model_para":model_para,
            "type":"BiGRU_NLP_GSVIB",
            "misc_para":para
        }

    def set_model_para(self,model_para):
        self.model_para = model_para
        self.hidden_size = model_para.get("hidden_size", 30)
        self.input_size = model_para.get("input_size", None)
        self.output_size = model_para.get("output_size", None)
        self.gs_head_dim = model_para.get("gs_head_dim", 2)
        self.gs_head_num = model_para.get("gs_head_num", 21)
        self.bigru_num_layers = model_para.get("bigru_num_layers", 1)
        self.mlp_num_layers = model_para.get("mlp_num_layers", 0)
        self.mlp_hidden = model_para.get("mlp_hidden", 80)
        self.hsoftmax_depth = model_para.get("hsoftmax_depth", 1)

    def para(self,para):
        self.misc_para = para
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.dropout_rate = para.get("dropout_rate", 0.2)
        self.coop_mode = para.get("coop_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def set_sample_size(self,sample_size):
        self.sample_size=sample_size
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size)

    def forward(self, input, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None, attention_sig=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        self.cuda_device=input.device
        hout, hn = self.gru(input,hidden1)
        dw, batch, l_size = input.shape

        context = self.h2c(hout)
        context = context.view(dw, batch, self.gs_head_num, self.gs_head_dim)

        if attention_sig is not None:
            p_prior_exp = self.prior.expand(dw, batch, self.gs_head_num, self.gs_head_dim)
            # Sigmoid attention from POS
            context = context * attention_sig.view(dw, batch, self.gs_head_num, 1) + p_prior_exp * (
                        1 - attention_sig.view(dw, batch, self.gs_head_num, 1))
            context = self.softmax(context)  ## Bug unfixed for historical reason

        context = self.softmax(context)
        if context_set is not None:
            context=context_set

        gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        p_prior = self.softmax(self.prior)
        self.context = context
        self.p_prior=p_prior
        self.gssample = gssample
        self.loss_intf = [self.context, p_prior]

        if self.coop_mode:
            assert self.sample_size == 1
            coop_out = gssample.view(dw, batch, self.gs_head_num * self.gs_head_dim)
            coop_out = self.layer_norm_coop(coop_out)
            return coop_out, None
            # return torch.exp(context.view(dw, batch, self.gs_head_num*self.gs_head_dim)), None
            # return context.view(dw, batch, self.gs_head_num, self.gs_head_dim), None

        gssample = gssample.view(dw, batch, self.sample_size, self.gs_head_num * self.gs_head_dim)

        if self.mlp_num_layers > 0:
            hidden_c=self.c2h(gssample)
            hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            output=self.h2o(hidden_c)
        else:
            output = self.c2o(gssample)

        if not logit_mode: # CrossEntropy combines lsoftmax and nllloss
            if self.hsoftmax_depth==1:
                output = self.softmax(output)
            elif self.hsoftmax_depth==2:
                outputp0 = self.softmax(output[dw, batch, self.sample_size,:self.output_size])
                outputp1 = self.softmax(output[dw, batch, self.sample_size, self.output_size:])
                output = torch.cat((outputp0,outputp1),dim=-1)
            else:
                raise Exception("Hsoftmax_depth larger than 2, not implemented yet.")

        return output, hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.bigru_num_layers*2, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch):
        return Variable(torch.zeros(self.bigru_num_layers*2, batch, self.hidden_size), requires_grad=True).to(device)

class Linear_Multihead(torch.nn.Module):
    """
    A linear module with Multihead, designed to implement Hierachical Softmax
    """
    def __init__(self,input_size, output_size, head_num, bias=True):
        super(self.__class__, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.head_num=head_num
        self.weight=torch.nn.Parameter(torch.Tensor(output_size, head_num, input_size), requires_grad=True)
        # for ii in range(head_num):
        #     torch.nn.init.kaiming_uniform_(self.weight[ii,:,:], a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(head_num, output_size), requires_grad=True)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, input, cluster):
        """

        :param input: shape [wd, batch, smp,  input_size]
        :param cluster: shape [wd, batch] digit is head_id
        :return:
        """
        self.cuda_device=input.device
        cluster_oh = one_hot(cluster, self.head_num, cuda_device=self.cuda_device) # [wd, batch, self.head_num]
        weight_c = cluster_oh.unsqueeze(-3).matmul(self.weight) # [wd, output_size, batch, input_size]
        weight_c=weight_c.transpose(-2,-3).transpose(-1,-2) # [wd, batch, input_size, output_size]
        res = input.matmul(weight_c)
        bias = cluster_oh.matmul(self.bias).unsqueeze(-2)
        if self.bias is not None:
            res = res + bias
        return res
        # output=torch.matmul(input.permute(1,0,2),weight_mask)+self.bias.view(1,-1)
        # return output.permute(1,0,2)

class BiGRU_NLP_GSVIB_HSOFTMAX2(torch.nn.Module):
    """
    PyTorch BiGRU for NLP with Gumbel Softmax variational information bottleneck
    with Hierarchical Softmax level2
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        self.context_attn_size = self.gs_head_num * self.gs_head_dim

        # self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)
        self.misc_para=para

        self.gru=torch.nn.GRU(self.input_size,self.hidden_size,num_layers=self.bigru_num_layers,bidirectional=True)
        self.h2c = torch.nn.Linear(self.hidden_size * 2, self.gs_head_dim*self.gs_head_num)
        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num,self.gs_head_dim)) , requires_grad=True)

        if not self.coop_mode:
            if self.mlp_num_layers>0:
                self.c2h = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.mlp_hidden)
                self.linear_layer_stack = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=True), torch.nn.LayerNorm(self.mlp_hidden),torch.nn.ReLU())
                    for _ in range(self.mlp_num_layers-1)])
                self.h2oc = torch.nn.Linear(self.mlp_hidden, self.output_cluster)
                self.h2o = Linear_Multihead(self.mlp_hidden, self.output_size, self.output_cluster)
            else:
                self.c2oc = torch.nn.Linear(self.gs_head_dim*self.gs_head_num, self.output_cluster)
                self.c2o = Linear_Multihead(self.gs_head_dim*self.gs_head_num, self.output_size, self.output_cluster)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size)
        self.layer_norm = torch.nn.LayerNorm(self.gs_head_num * self.gs_head_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def set_model_para(self,model_para):
        # model_para = {"input_size": lsize_in,
        #               "output_size": 201,
        #                "output_cluster": 150,
        #               "gs_head_dim": 2,
        #               "gs_head_num": 21,
        #               "bigru_hidden_size": 30,
        #               "mlp_num_layers": 0,
        #               "mlp_hidden": 80,
        #               "bigru_num_layers": 3,
        #               "hsoftmax_depth": 2
        #               }
        self.model_para = model_para
        self.hidden_size = model_para.get("hidden_size", 30)
        self.input_size = model_para.get("input_size", None)
        self.output_size = model_para.get("output_size", None)
        self.output_cluster = model_para.get("output_cluster", None)
        self.gs_head_dim = model_para.get("gs_head_dim", 2)
        self.gs_head_num = model_para.get("gs_head_num", 21)
        self.bigru_num_layers = model_para.get("bigru_num_layers", 1)
        self.mlp_num_layers = model_para.get("mlp_num_layers", 0)
        self.mlp_hidden = model_para.get("mlp_hidden", 80)
        self.hsoftmax_depth = model_para.get("hsoftmax_depth", 2)

    def para(self,para):
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.dropout_rate = para.get("dropout_rate", 0.2)
        self.coop_mode = para.get("coop_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def set_sample_size(self,sample_size):
        self.sample_size=sample_size
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size)

    def forward(self, inputl, hidden1, schedule=None, context_set=None, attention_sig=None, logit_mode=True):
        """
        Forward
        :param inputl: [window batch l_size], cluster
        :param hidden:
        :return:
        """
        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        input , cluster = inputl
        cluster = cluster.transpose(1,0)
        self.cuda_device=input.device
        hout, hn = self.gru(input,hidden1)
        dw, batch, l_size = input.shape

        context = self.h2c(hout)
        context = context.view(dw, batch, self.gs_head_num, self.gs_head_dim)

        if attention_sig is not None:
            p_prior_exp = self.prior.expand(dw, batch, self.gs_head_num, self.gs_head_dim)
            # Sigmoid attention from POS
            context = context * attention_sig.view(dw, batch, self.gs_head_num, 1) + p_prior_exp * (
                        1 - attention_sig.view(dw, batch, self.gs_head_num, 1))
            context = self.softmax(context)  ## Bug unfixed for historical reason

        context = self.softmax(context)
        if context_set is not None:
            context=context_set

        gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        p_prior = self.softmax(self.prior)
        self.context = context
        self.p_prior=p_prior
        self.gssample = gssample
        self.loss_intf = [self.context, p_prior]

        if self.coop_mode:
            return gssample.view(dw, batch, self.sample_size,self.gs_head_num*self.gs_head_dim), None
            # return torch.exp(context.view(dw, batch, self.gs_head_num*self.gs_head_dim)), None
            # return context.view(dw, batch, self.gs_head_num, self.gs_head_dim), None

        gssample = gssample.view(dw, batch, self.sample_size, self.gs_head_num * self.gs_head_dim)
        gssample = self.layer_norm(gssample) # Very important

        # gssample.register_hook(lambda grad: print("gssample", grad))

        if self.mlp_num_layers > 0:
            hidden_c=self.c2h(gssample)
            hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            outputc=self.h2oc(hidden_c)
            outputh = self.h2o(hidden_c,cluster)
        else:
            outputc = self.c2oc(gssample)
            outputh = self.c2o(gssample, cluster)

        output = torch.cat((outputc, outputh), dim=-1) # (wd, b, s, outc+outh)

        return output, hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.bigru_num_layers*2, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch):
        return Variable(torch.zeros(self.bigru_num_layers*2, batch, self.hidden_size), requires_grad=True).to(device)


class BiGRU_NLP_GSVIB_ATTCOOP(torch.nn.Module):
    """
    PyTorch BiGRU for NLP with Gumbel Softmax variational information bottleneck
    with some label as attentional cooperator
    require a pre_trained ff network
    """
    def __init__(self, input_size, hidden_size, gs_head_num, mlp_hidden, output_size, ff_model, gs_head_dim=2, num_layers=1 ,para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.gs_head_num=gs_head_num
        self.gs_head_dim=gs_head_dim

        self.mlp_hidden=mlp_hidden
        self.num_layers=num_layers

        self.ff_model=ff_model

        # self.coop_mode=False

        if para is None:
            para = dict([])
        self.para(para)

        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers=num_layers,bidirectional=True)
        self.h2c = torch.nn.Linear(hidden_size * 2, gs_head_dim*gs_head_num)
        self.prior = torch.nn.Parameter(torch.zeros((gs_head_num,gs_head_dim)) , requires_grad=True)
        self.prior_att = None

        if self.mlp_num_layers>0:
            self.c2h = torch.nn.Linear(gs_head_dim*gs_head_num, mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True), torch.nn.LayerNorm(mlp_hidden),torch.nn.ReLU())
                for _ in range(self.mlp_num_layers-1)])
            self.h2o = torch.nn.Linear(mlp_hidden, output_size)
        else:
            self.c2o = torch.nn.Linear(gs_head_dim*gs_head_num, output_size)

        self.attn = torch.nn.Linear(ff_model.input_size, gs_head_num) # Attention from pos tag to representation, key to control hierachy/combinatory

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax()
        self.gsigmoid = Gumbel_Sigmoid()
        self.myhsig = myhsig

        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def para(self,para):
        self.mlp_num_layers = int(para.get("mlp_num_layers", 0))
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.dropout_rate = para.get("dropout_rate", 0.2)

    def forward(self, inputl, hidden1, add_logit=None, logit_mode=False, schedule=None, context_set=None):
        """
        Forward
        :param input: [input_wrd,input_pos]    [window batch l_size]
        :param hidden:
        :return:
        """
        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        self.cuda_device=inputl[0].device
        hout, hn = self.gru(inputl[0],hidden1)
        dw, batch, l_size = inputl[0].shape

        context = self.h2c(hout)
        context = context.view(dw, batch, self.gs_head_num, self.gs_head_dim)
        # context = self.softmax(context)

        # p_prior = self.softmax(self.prior).expand(dw, batch, self.gs_head_num, self.gs_head_dim)
        p_prior = self.prior.expand(dw, batch, self.gs_head_num, self.gs_head_dim)

        # Sigmoid attention from POS
        attention = self.attn(inputl[1])
        attention_sig = self.gsigmoid(attention, temperature=temperature, cuda_device=self.cuda_device)
        # attention_sig = self.sigmoid(attention/temperature)
        # attention_sig = self.myhsig(attention,temperature=noise_temp,cuda_device=self.cuda_device)

        context = context*attention_sig.view(dw, batch, self.gs_head_num, 1)+p_prior*(1-attention_sig.view(dw, batch, self.gs_head_num, 1))
        context = self.softmax(context)
        gssample = self.gsoftmax(context, temperature=temperature, cuda_device=self.cuda_device)

        # attention_prob=self.sigmoid(attention)
        self.context = context
        self.p_prior=self.softmax(p_prior)
        self.attention_prob=attention_sig
        self.gssample = gssample
        self.attention_sample=attention_sig

        gssample = gssample.view(dw, batch, self.gs_head_num*self.gs_head_dim)

        if self.mlp_num_layers > 0:
            hidden_c=self.c2h(gssample)
            hidden_c=self.dropout(hidden_c)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            output=self.h2o(hidden_c)
        else:
            output = self.c2o(gssample)

        logit_coop, _ = self.ff_model(inputl[1], None, logit_mode=True, schedule=1.0)
        output = logit_coop + output

        if not logit_mode:
            output = self.softmax(output)

        return output, hn

    def initHidden(self,batch):
        return Variable(torch.zeros(self.num_layers*2, batch,self.hidden_size), requires_grad=True)

    def initHidden_cuda(self, device, batch):
        return Variable(torch.zeros(self.num_layers*2, batch, self.hidden_size), requires_grad=True).to(device)

class BiTRF_NLP(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """
    def __init__(self, model_size, hidden_size, output_size, window_size, para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.model_size = model_size
        self.window_size = window_size

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


class BiTRF_NLP_VIB(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    With variational information bottleneck
    """
    def __init__(self, input_size, model_size, hidden_size, context_size, mlp_hidden, output_size, window_size, para=None):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.model_size = model_size
        self.mlp_hidden=mlp_hidden
        self.window_size = window_size

        if para is None:
            para = dict([])
        self.para(para)

        # self.slf_attn = MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout)
        # self.pos_ffn = PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)

        self.i2m = torch.nn.Linear(input_size, model_size)

        self.posi_sinmat = self.get_sinusoid_encoding_table(window_size, model_size)
        # self.posi_mat, self.n_posiemb = self.posi_embed(self.window_size,
        #                                                 switch="log")  # (self.input_len,self.n_posiemb)

        self.trf_layers = torch.nn.ModuleList([
            torch.nn.ModuleList(
                [MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                 PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
            for _ in range(self.num_layers)])

        self.h2mu = torch.nn.Linear(model_size, context_size)
        self.h2t = torch.nn.Linear(model_size, context_size)

        if self.mlp_num_layers > 0:
            self.c2h = torch.nn.Linear(context_size, mlp_hidden)
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_hidden, mlp_hidden, bias=True), torch.nn.Tanh())
                # torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size, bias=True), torch.nn.ReLU())
                for _ in range(self.mlp_num_layers)])
            self.h2o = torch.nn.Linear(mlp_hidden, output_size)
        else:
            self.c2o = torch.nn.Linear(context_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)
        self.softplus = torch.nn.Softplus()
        self.vagauss = VariationalGauss(cuda_device=self.cuda_device)

        # self.selfattn_mask=torch.eye(self.window_size)
        # self.selfattn_mask = self.selfattn_mask.type(torch.uint8)
        if torch.cuda.is_available():
            self.posi_sinmat = self.posi_sinmat.to(self.cuda_device)
            # self.selfattn_mask = self.selfattn_mask.to(self.cuda_device)

        self.input_mask = None

    def para(self, para):
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.dropout = para.get("dropout", 0.1)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.n_head = para.get("n_head", 2)
        self.d_k = para.get("d_k", 99)
        self.d_v = para.get("d_v", 101)
        self.num_layers = para.get("num_layers", 1)
        self.self_unmask_rate = para.get("self_unmask_rate", 0.1)
        self.multi_sample_flag = para.get("multi_sample_flag", False)
        self.mlp_num_layers = int(para.get("mlp_num_layers", 0))

    def get_sinusoid_encoding_table(self, n_position, model_size):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(1000, 2 * (hid_idx // 2) / model_size)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(model_size)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def posi_embed(self, emb_len, switch="onehot"):
        if switch == "log":
            n_posiemb = int(np.log(emb_len) / np.log(2)) + 1
            posi_mat = torch.ones((emb_len, n_posiemb))
            for ii in range(n_posiemb):
                step = 2 ** (ii + 1)
                for ii2 in range(2 ** ii):
                    posi_mat[ii2::step, ii] = 0
        elif switch == "onehot":
            n_posiemb = emb_len
            posi_mat = torch.eye(emb_len)
        else:
            raise Exception("Switch not known")
        print(posi_mat)
        if torch.cuda.is_available():
            posi_mat = posi_mat.to(self.cuda_device)
        return posi_mat, n_posiemb

    def forward(self, input, hidden, add_logit=None, logit_mode=False, schedule=None):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        assert len(input.shape) == 3
        length, batch, l_size = input.shape
        input = input.permute((1, 0, 2))  # Due to transformer, use [batch, seq_len, hd_size] convention

        if self.training:
            rnd = torch.rand((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask = torch.zeros((input.shape[0], input.shape[1]), device=self.cuda_device)
            self.input_mask[rnd > self.mask_rate] = 1
            # Self unmask sampling
            rnd2 = torch.rand((input.shape[0], input.shape[1]), device=self.cuda_device)
            self_unmask = torch.zeros((input.shape[0], input.shape[1]), device=self.cuda_device)
            self_unmask[rnd2 < self.self_unmask_rate] = 1
            comb_mask=self.input_mask+self_unmask
            comb_mask[comb_mask > 0.999] = 1
            input = input * comb_mask.view(input.shape[0], input.shape[1], 1).expand_as(input)

        modelin = self.i2m(input)
        modelin = self.tanh(modelin)
        enc_output = self.layer_norm(modelin) + self.layer_norm(self.posi_sinmat.view(1, -1, self.model_size))
        ### or
        # enc_output = torch.cat(
        #     (self.posi_mat.view(1, length , self.n_posiemb).expand(batch, length, self.n_posiemb), input), dim=-1)

        for trf_l in self.trf_layers:
            enc_output, attn = trf_l[0](enc_output, enc_output, enc_output)
            enc_output = trf_l[1](enc_output)

        cmu = self.h2mu(enc_output)
        ctheta = self.h2t(enc_output)
        ctheta = self.softplus(ctheta)
        context = self.vagauss(cmu, ctheta)
        self.cmu = cmu
        self.ctheta = ctheta
        self.loss_intf = [self.ctheta, self.cmu]
        self.context = context

        if self.mlp_num_layers > 0:
            hidden_c = self.c2h(context)
            for fmd in self.linear_layer_stack:
                hidden_c = fmd(hidden_c)
            output = self.h2o(hidden_c)
        else:
            output = self.c2o(context)

        if not logit_mode:
            output = self.softmax(output)

        output = output.permute((1, 0, 2))

        return output, None

    def initHidden(self, batch):
        return None

    def initHidden_cuda(self, device, batch):
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


class W2V_SkipGram(torch.nn.Module):
    """
    PyTorch SkipGram W2V with negtive sampling
    """
    def __init__(self, Ndim, Nvocab, window_size, ns_size, prior, para=None):
        """

        :param Ndim: Embedding dimension
        :param Nvocab: # of vocab
        :param window_size: context size
        :param ns_size: negtive sampling size
        :param prior: prior distribution
        """
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.Ndim = Ndim
        self.Nvocab = Nvocab
        self.window_size = window_size
        self.ns_size = ns_size
        self.prior = prior

        self.tid = int((window_size-1)/2)
        self.w2v=torch.nn.Embedding(Nvocab,Ndim)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available():
            self.w2v=self.w2v.to(self.cuda_device)

    def set_window_size(self,window_size):
        self.window_size=window_size
        self.tid = int((window_size - 1) / 2)

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")

    def forward(self, input, hidden=None, schedule=None):
        """
        Forward
        :param input: size[window_size,batch]
        :return:
        """
        batch=input.shape[1]
        tidi=input[self.tid,:]
        vi=self.w2v(tidi)
        tido=torch.cat((input[:self.tid,:],input[self.tid+1:,:]),dim=0)
        vo=self.w2v(tido)

        adjprior=(self.prior/np.sum(self.prior))**0.75
        adjprior=adjprior/np.sum(adjprior)
        nsi=sample_id(adjprior,(self.window_size-1,batch,self.ns_size))
        if torch.cuda.is_available():
            nsi=nsi.to(self.cuda_device)
        vns=self.w2v(nsi)
        # vo[window_size-1,batch,l_size] vi[batch,l_size]
        loss_l=torch.log(self.sigmoid(vo.permute(1,0,2).bmm(vi.view(batch,-1,1))))
        # vns[self.window_size-1,batch,self.ns_size,l_size]
        vnsmatl=vns.permute(1,0,2,3).contiguous().view(batch,(self.window_size-1)*self.ns_size,-1)
        vnsmatr=vi.view(batch,-1,1)
        vnsmat=-vnsmatl.bmm(vnsmatr)
        sgvnsmat=self.sigmoid(vnsmat)
        if (sgvnsmat == 0).any():
            sgvnsmat=sgvnsmat+1e-9*(self.sigmoid(vnsmat) == 0).type(torch.FloatTensor).to(self.cuda_device)
        loss_r=torch.log(sgvnsmat)
        output=torch.mean(loss_l)+torch.mean(loss_r)
        return -output,None

    def initHidden(self,batch):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True)

    def initHidden_cuda(self,device, batch):
        return Variable(torch.zeros(1, batch, 1), requires_grad=True).to(device)

class W2V_Softmax(torch.nn.Module):
    """
    PyTorch Simple Softmax W2V calculation
    """
    def __init__(self, Ndim, Nvocab, window_size, ns_size, para=None):
        """

        :param Ndim: Embedding dimension
        :param Nvocab: # of vocab
        :param window_size: context size
        :param ns_size: negtive sampling size
        :param prior: prior distribution
        """
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.Ndim = Ndim
        self.Nvocab = Nvocab
        self.window_size = window_size
        self.ns_size = ns_size

        self.w2v=torch.nn.Embedding(Nvocab,Ndim)

        self.conv1 = torch.nn.Conv1d(Ndim, Ndim, window_size, stride=1, padding=0, bias=False)
        ## A conv1d mimicking context sum
        # conv_tensor=torch.ones(Ndim, Ndim, window_size) / ((window_size-1))
        # conv_tensor[:,:,int((window_size-1)/2)]=0
        # self.conv1.weight = torch.nn.Parameter(conv_tensor, requires_grad=False)
        self.mask = torch.ones(window_size)
        self.mask[int((window_size-1)/2)] = 0

        self.hout=torch.nn.Linear(Ndim, Nvocab)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if torch.cuda.is_available():
            self.w2v=self.w2v.to(self.cuda_device)
            self.mask=self.mask.to(self.cuda_device)

        self.masked_conv1 = WeightMask(self.conv1, ["weight"], self.mask.view(1, 1, -1))

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")

    def forward(self, input, hidden=None, schedule=None):
        """
        Forward
        :param input: size[window_size_ext,batch]
        :return:
        """
        window_size_ext,batch = input.shape
        assert window_size_ext>=self.window_size

        input_mat = self.w2v(input) # window_size_ext,batch,Ndim
        input_mat = input_mat.permute(1, 2, 0) # batch,Ndim,window_size_ext
        # input_mat = input_mat * self.mask.view(1,1,-1)
        # conv_mat = self.conv1(input_mat)  # batch,Ndim,window_size_ext
        conv_mat = self.masked_conv1(input_mat) # batch,Ndim,window_size_ext
        output = self.hout(conv_mat.permute(2,0,1))

        output=self.softmax(output)

        # try:
        #     print(torch.max(self.w2v.weight.grad),torch.mean(self.w2v.weight.grad))
        # except:
        #     pass

        return output, None # [window_size_ext-self.window_size+1,batch,Nvcab]

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None

class bClustering(torch.nn.Module):
    """
    PyTorch simple balanced clustering method
    """
    def __init__(self, n_cluster, dim, para=None, cluster_init=None):
        """

        :param N: Number of points
        :param n_cluster: number of clusters
        :param dim: point dimension
        """
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.n_cluster = n_cluster
        self.dim = dim

        # self.assignMat=torch.nn.Parameter(torch.zeros((N,n_cluster)), requires_grad=True)
        self.assignMat = torch.nn.Linear(self.n_cluster, self.n_cluster)
        # centroidMat initialization should match data
        if cluster_init is not None:
            self.centroidMat = torch.nn.Parameter(torch.rand((n_cluster, dim))-0.5, requires_grad=True)
        else:
            self.centroidMat = torch.nn.Parameter(torch.from_numpy(cluster_init).type(torch.FloatTensor), requires_grad=True)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax()

        self.layer_norm = torch.nn.LayerNorm(n_cluster)
        # self.layer_norm = torch.nn.LayerNorm(n_cluster*dim)

    def para(self,para):
        self.cuda_device = para.get("cuda_device", "cuda:0")
        self.freeze_mode = para.get("freeze_mode", True)
        self.temp_scan_num = para.get("temp_scan_num", 4)
        self.beta = para.get("beta", 0) # Lagrangian multiplier

    def forward(self, input, hidden=None, schedule=None):
        """

        :param input: (N,dim)
        :param hidden:
        :param schedule:
        :return:
        """

        if schedule<1.0: # Multi-scanning trial
            schedule=schedule*self.temp_scan_num
            schedule=schedule-np.floor(schedule)

        if not self.freeze_mode:
            # temperature = np.exp(-schedule * 5)
            temperature = 1.005-schedule
        else:
            temperature = np.exp(-5)

        self.cuda_device = input.device

        N,dim = input.shape
        diff=input.view(N,1,dim)-self.centroidMat.view(1,self.n_cluster,dim)
        dist=torch.sqrt(torch.mean(diff**2,dim=-1))
        dist=self.layer_norm(dist) #Very important
        assignMat = self.assignMat(dist)
        assignMat=self.softmax(assignMat)
        assignMat_sample = self.gsoftmax(assignMat, temperature=temperature, cuda_device=self.cuda_device) # (N,n_cluster)

        diff2 = input - assignMat_sample.matmul(self.centroidMat)
        rms_dist=torch.sqrt(torch.mean(diff2**2))

        cltcnt = torch.sum(assignMat_sample, dim=0)
        # blcost=torch.sqrt(torch.mean((cltcnt-torch.mean(cltcnt))**2))

        cltcnt_p = torch.log(cltcnt / torch.sum(cltcnt)+1e-9)
        blcost = torch.sum(torch.exp(cltcnt_p) * (cltcnt_p + np.log(self.n_cluster)))

        self.gssample=assignMat_sample

        output=rms_dist+self.beta*blcost

        return output, None

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self,device, batch):
        return None