"""
PyTorch Modules of All kinds vol2 starting 2019/10/18
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.nn import functional as F
from ncautil.seqmultip2 import FF_MLP, Gumbel_Softmax
from ncautil.seqmultip3 import GSVIB_InfoBottleNeck, Softmax_Sample

class BiTRF_NLP(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.posi_sinmat = self.get_sinusoid_encoding_table(self.window_size, self.model_size)
        # self.posi_mat, self.n_posiemb = self.posi_embed(self.window_size,
        #                                                 switch="log")  # (self.input_len,self.n_posiemb)
        if not self.layer_share:
            self.trf_layers = torch.nn.ModuleList([
                torch.nn.ModuleList([MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                    PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
                for _ in range(self.num_layers)])
        else:
            print("ALBert layer shared version.")
            self.trf_layers = torch.nn.ModuleList([MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                     PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])

        self.h2o = torch.nn.Linear(self.model_size, self.output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)

        self.input_mask=None

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self,model_para):
        # model_para_bert = {
        #     "model_size": 300,
        #     "hidden_size": 768,
        #     "window_size": 256,
        #     "output_size": 30001,
        #     "n_head": 12,
        #     "d_k": 768,
        #     "d_v": 768,
        #     "num_layers": 12,
        # }
        self.model_para = model_para
        self.model_size = model_para["model_size"]
        self.hidden_size = model_para["hidden_size"]
        self.window_size = model_para["window_size"]
        self.output_size = model_para["output_size"]
        self.n_head = model_para.get("n_head", 12)
        self.d_k = model_para.get("d_k", 64)
        self.d_v = model_para.get("d_v", 768)
        self.num_layers = model_para.get("num_layers", 12)

    def para(self,para):
        self.misc_para = para
        self.dropout = para.get("dropout", 0.1)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.layer_share = para.get("layer_share", False)

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

    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx:
        :param hidden:
        :return:
        """
        self.cuda_device = inputx.device
        if self.posi_sinmat.device != inputx.device:
            self.posi_sinmat = self.posi_sinmat.to(inputx.device)

        rnd=torch.rand((inputx.shape[0],inputx.shape[1]),device=self.cuda_device)
        self.input_mask=torch.zeros((inputx.shape[0],inputx.shape[1]),device=self.cuda_device)
        self.input_mask[rnd>self.mask_rate]=1
        mask_input=inputx*self.input_mask.view(inputx.shape[0],inputx.shape[1],1).expand_as(inputx)
        enc_output= self.layer_norm(mask_input) + self.posi_sinmat.view(1, -1, self.model_size)
        ### or
        # enc_output = torch.cat(
        #     (self.posi_mat.view(1, length , self.n_posiemb).expand(batch, length, self.n_posiemb), input), dim=-1)

        if not self.layer_share:
            for trf_l in self.trf_layers:
                enc_output, attn = trf_l[0](enc_output, enc_output, enc_output)
                enc_output = trf_l[1](enc_output)
        else:
            for ii in range(self.num_layers):
                enc_output, attn = self.trf_layers[0](enc_output, enc_output, enc_output)
                enc_output = self.trf_layers[1](enc_output)

        output=self.h2o(enc_output)
        output = self.softmax(output)

        return output

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(self.__class__, self).__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # print(attn.shape,v.shape)
        # plot_mat(attn[0,:,:].cpu().detach())
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(self.__class__, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(0.2 / (d_model + d_k))) # 2 -> 0.2
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(0.2 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(0.2 / (d_model + d_v)))

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
            mask = mask.repeat(sz_b*n_head, 1, 1) # (n*b) x .. x ..
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
        super(self.__class__, self).__init__()
        self.w_1 = torch.nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = torch.nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(self.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)
        return output

class MultiHeadAttention_Discrete(torch.nn.Module):
    ''' Discrete version of multihead attention '''

    def __init__(self, model_para, para=None):
        """
        symmetry asumption, key = query
        Controlled position assumption
        :param n_head:
        :param d_model:
        :param d_k:
        :param d_v:
        """
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.set_model_para(model_para)

        self.w_kqs = torch.nn.Linear(self.model_size, self.n_head * self.d_kq)
        self.w_vs = torch.nn.Linear(self.model_size, self.n_head * self.d_v)
        torch.nn.init.normal_(self.w_kqs.weight, mean=0, std=np.sqrt(2.0 / (self.model_size + self.d_kq)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.model_size + self.d_v)))

        # self.layer_norm = torch.nn.LayerNorm(d_model)

        model_para_fftubes={
            "input_size": self.d_v,
            "n_head": self.n_head,
            "output_size": self.fftubes_output_size,
            "mlp_layer_para": self.fftubes_mlp_layer_para,
            "infobn_model": {"gs_head_dim":2,"gs_head_num":int(self.fftubes_output_size/2)}
        }
        self.multitubeff = Multitube_FF_MLP_TRF(model_para_fftubes, para=self.misc_para)

        self.fc = torch.nn.Linear(self.n_head * self.fftubes_output_size, self.model_size)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.psoftmax = torch.nn.Softmax(dim=-1)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=1, sample_mode=False)

        self.submodels = [self.multitubeff]

        self.fc_temp = torch.nn.Linear(self.d_v, self.model_size)

    def para(self, para):
        self.misc_para = para
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)

    def set_model_para(self, model_para):

        self.model_para = model_para
        self.window_size = model_para["window_size"]
        self.model_size = model_para["model_size"]
        self.n_head = model_para["n_head"]
        self.d_kq = model_para["d_kq"]
        self.d_v = model_para["d_v"]
        self.fftubes_output_size = model_para["fftubes_output_size"]
        self.fftubes_mlp_layer_para = model_para["fftubes_mlp_layer_para"]

    def forward(self, kq, v, schedule=1.0):

        if schedule < 1.0:  # Multi-scanning trial
            schedule = schedule * self.temp_scan_num
            schedule = schedule - np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        cuda_device = kq.device

        d_kq, d_v, n_head = self.d_kq, self.d_v, self.n_head

        batch, seq_len, _ = kq.size() # [batch, seq_len, hd_size]

        kq = self.w_kqs(kq).view(batch, seq_len, n_head, d_kq)
        v = self.w_vs(v).view(batch, seq_len, n_head, d_v)

        kq = kq.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_kq) # (n*b) x seq_len x dkq
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_v) # (n*b) x seq_len x dv

        attn = torch.bmm(kq, kq.transpose(1, 2)) # Symmetry

        mask = torch.eye(seq_len, device=cuda_device)
        mask = mask.repeat(n_head*batch, 1, 1)  # (n*b) x .. x ..
        attn = attn.masked_fill(mask==1, -np.inf)

        attn = self.softmax(attn)
        attn_sample = self.gsoftmax(attn, temperature=temperature)
        att_output = torch.bmm(attn_sample, v)
        att_output = att_output.view(n_head, batch, seq_len, d_v)
        outputsample = self.multitubeff(att_output,schedule=schedule)
        # print("outputsample",outputsample.shape)
        output = outputsample.permute(1, 2, 0, 3).contiguous().view(batch, seq_len, -1)  # b x lq x (n*dv)
        output = self.fc(output)

        # output = self.fc_temp(att_output[0])

        return output, attn

class Multitube_FF_MLP_TRF(torch.nn.Module):
    """
    Feed forward multi-tube perceptron (a multiple information flow tube, no mixing within tubes, each tube is an FF)
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        model_para_ff = {
            "input_size": self.input_size,
            "output_size":self.output_size,
            "mlp_layer_para": self.mlp_layer_para
        }

        self.ff_tubes = torch.nn.ModuleList([
            FF_MLP(model_para_ff)
            for iim in range(self.n_head)])

        self.infobnl = torch.nn.ModuleList([
            GSVIB_InfoBottleNeck(self.infobn_model,para=self.misc_para)
            for iim in range(self.n_head)])

        # self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        # self.layer_norm1 = torch.nn.LayerNorm(self.output_size)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para = para
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 300
        #     "n_head": 12
        #     "output_size": 128
        #     "mlp_layer_para": [256,128]
        #     "infobn_model": {"gs_head_dim":2,"gs_head_num":64}
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.n_head = model_para["n_head"]
        self.output_size = model_para["output_size"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0l, hidden1l, ...]
        self.infobn_model = model_para.get("infobn_model", None)


    def forward(self, datax, schedule=None):
        """
        :param datax: n_head, batch, seq_len, d_v
        :param schedule:
        :return:
        """

        dataml=[]
        for iim, fmd in enumerate(self.ff_tubes):
            datam = fmd(datax[iim,:,:,:])
            dataml.append(datam)

        gsamplel = []
        contextl = []
        for iim, infobn in enumerate(self.infobnl):
            gsample = infobn(dataml[iim], schedule=schedule)
            gsamplel.append(gsample)
            contextl.append(infobn.contextl)
        self.loss_reg = 0
        for iim in range(self.n_head):
            self.loss_reg = self.loss_reg + self.infobnl[iim].cal_regloss()

        self.context = torch.stack(contextl, dim=0)
        output = torch.stack(gsamplel, dim=0)
        self.output = output

        return output

class MultiHeadAttention_NoMask(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)

        self.set_model_para(model_para)

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

    def para(self, para):
        self.misc_para = para

    def set_model_para(self, model_para):

        self.model_para = model_para
        self.window_size = model_para["window_size"]
        self.model_size = model_para["model_size"]
        self.n_head = model_para["n_head"]
        self.d_kq = model_para["d_kq"]
        self.d_v = model_para["d_v"]

    def forward(self, q, k, v):
        """

        :param q: from value
        :param k: from posi + real vec
        :param v: from att * real vec
        :param mask:
        :return:
        """
        cuda_device = q.device
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

        mask = torch.eye(len_q, device=cuda_device)
        mask = mask.repeat(n_head * sz_b, 1, 1)  # (n*b) x .. x ..
        mask = (mask==1)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output+residual)
        # output = self.layer_norm(output)

        return output, attn


class BiTRF_NLP_PSAB(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """

    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.posi_sinmat = self.get_sinusoid_encoding_table(self.window_size, self.model_size)
        self.temperaturek = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.pt_emb = torch.nn.Embedding(self.output_size, self.model_size)
        self.pt_emb_bias = torch.nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        if self.pt_emb_ini is not None:
            self.pt_emb.load_state_dict(self.pt_emb_ini.state_dict())
        if self.emb_freeze:
            self.pt_emb.weight.requires_grad=False

        ### Old layers
        if not self.layer_share:
            self.trf_layers = torch.nn.ModuleList([
                torch.nn.ModuleList(
                    [MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                     PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
                for _ in range(self.num_layers)])
        else:
            print("ALBert layer shared version.")
            self.trf_layers = torch.nn.ModuleList(
                [MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                 PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])

        ### New layers
        if not self.layer_share:
            self.trf_layers_new = torch.nn.ModuleList([
                torch.nn.ModuleList(
                    [MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                     PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
                for _ in range(self.num_layers_new)])
        else:
            print("ALBert layer shared version.")
            self.trf_layers_new = torch.nn.ModuleList(
                [MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                 PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)

        self.input_mask = None
        self.v2id_output = None

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self, model_para):
        # model_para_bert = {
        #     "model_size": 300,
        #     "hidden_size": 768,
        #     "window_size": 256,
        #     "output_size": 30001,
        #     "n_head": 12,
        #     "d_k": 768,
        #     "d_v": 768,
        #     "num_layers": 12,
        # }
        self.model_para = model_para
        self.model_size = model_para["model_size"]
        self.hidden_size = model_para["hidden_size"]
        self.window_size = model_para["window_size"]
        self.output_size = model_para["output_size"]
        self.n_head = model_para.get("n_head", 12)
        self.d_k = model_para.get("d_k", 256)
        self.d_v = model_para.get("d_v", 256)
        self.num_layers = model_para.get("num_layers", 12)
        self.num_layers_new = model_para.get("num_layers_new", 1)
        self.pt_emb_ini = model_para.get("pt_emb_ini", None)

    def para(self, para):
        self.misc_para = para
        self.emb_freeze = para.get("emb_freeze", False)
        self.dropout = para.get("dropout", 0.1)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.layer_share = para.get("layer_share", False)
        self.v2id_out_flag = para.get("v2id_out_flag", False)

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

    def absorb(self, bert_psab):
        print("Absorbing new layers ...")
        old_nl = bert_psab.num_layers
        old_nlnew = bert_psab.num_layers_new

        for ii in range(old_nl):
            self.trf_layers[ii].load_state_dict(bert_psab.trf_layers[ii].state_dict())
        for ii in range(old_nlnew):
            self.trf_layers[ii+old_nl].load_state_dict(bert_psab.trf_layers_new[ii].state_dict())

        self.temperaturek = bert_psab.temperaturek
        self.pt_emb = bert_psab.pt_emb
        self.pt_emb_bias = bert_psab.pt_emb_bias

    def layer_stack(self, bert_psab):
        print("Stacking new layers ...")
        old_nl = bert_psab.num_layers

        for ii in range(old_nl):
            self.trf_layers[ii].load_state_dict(bert_psab.trf_layers[ii].state_dict())
            self.trf_layers[ii + old_nl].load_state_dict(bert_psab.trf_layers[ii].state_dict())

    def freeze_oldlayers(self, freeze=True):
        if freeze:
            print("Freezing old layers ...")
            for param in self.trf_layers.parameters():
                param.requires_grad = False
            self.pt_emb.weight.requires_grad = False
            self.temperaturek.requires_grad = False
            self.pt_emb_bias.requires_grad = False

        else:
            print("Unfreezing old layers ...")
            for param in self.trf_layers.parameters():
                param.requires_grad = True
            self.pt_emb.weight.requires_grad = True
            self.temperaturek.requires_grad = True
            self.pt_emb_bias.requires_grad = True

    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx:
        :param hidden:
        :return:
        """
        self.cuda_device = inputx.device
        if self.posi_sinmat.device != inputx.device:
            self.posi_sinmat = self.posi_sinmat.to(inputx.device)

        inputx = self.pt_emb(inputx)
        batch, seq_len, hd_size = inputx.shape

        rnd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        self.input_mask = torch.zeros((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        self.input_mask[rnd > self.mask_rate] = 1
        mask_input = inputx * self.input_mask.view(inputx.shape[0], inputx.shape[1], 1).expand_as(inputx)
        enc_output = self.layer_norm(mask_input) + self.posi_sinmat.view(1, -1, self.model_size)

        if not self.layer_share:
            for trf_l in self.trf_layers:
                enc_output, attn = trf_l[0](enc_output, enc_output, enc_output)
                enc_output = trf_l[1](enc_output)
        else:
            for ii in range(self.num_layers):
                enc_output, attn = self.trf_layers[0](enc_output, enc_output, enc_output)
                enc_output = self.trf_layers[1](enc_output)

        v2id_output = torch.nn.functional.linear(enc_output, self.pt_emb.weight, bias=self.pt_emb_bias)
        self.v2id_output = self.softmax(self.temperaturek*v2id_output.squeeze())
        if self.v2id_out_flag:
            return self.v2id_output

        if not self.layer_share:
            for trf_l in self.trf_layers_new:
                enc_output, attn = trf_l[0](enc_output, enc_output, enc_output)
                enc_output = trf_l[1](enc_output)
        else:
            for ii in range(self.num_layers_new):
                enc_output, attn = self.trf_layers_new[0](enc_output, enc_output, enc_output)
                enc_output = self.trf_layers_new[1](enc_output)

        return enc_output


class CAL_LOSS_PSAB(torch.nn.Module):
    """
    An abstract loss wrapper
    """

    def __init__(self, model, para=None):
        super(self.__class__, self).__init__()
        self.model = model
        if para is None:
            para = dict([])
        self.para(para)

        self.ssample = Softmax_Sample(para={"reduce_to_id": True})
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def para(self, para):
        self.misc_para = para
        self.pt_emb = para.get("pt_emb", None)
        self.nchoice = para.get("nchoice", 4)

    def forward(self, datax, labels, schedule=1.0):
        """
        datax: [batch, window, nmodel] h_vector
        labels: [batch, window] Real Label
        self.model.nvac_out : [batch, window, nVac] lSoftmax
        """
        cuda_device = datax.device

        datax = self.model(datax, schedule=schedule)

        input_mask = self.find_para_recursive(self.model, "input_mask")

        # id_samples=[]
        # for ii in range(self.nchoice):
        #     id_samples_p = self.ssample(self.model.v2id_output).detach().cpu()
        #     id_samples.append(id_samples_p)
        # id_samples=torch.stack(id_samples,dim=-1) # [batch, window, nchoice]

        probmat = self.model.v2id_output.detach()
        batch, window, nVac = probmat.shape
        probmat = torch.exp(probmat)
        probmat.scatter_(-1, labels.unsqueeze(-1), 0)
        id_samples = torch.multinomial(probmat.view(-1,nVac),self.nchoice).view(batch,window,self.nchoice)

        answer_samples = torch.floor(self.nchoice*torch.cuda.FloatTensor(labels.shape).uniform_()).type(torch.cuda.LongTensor) # [batch, window]

        # replace answer_samples of id_samples into real labels
        mctask = self.gen_mctask(id_samples, answer_samples, labels)  # [batch, window, nchoice]
        mctaskvec = self.model.pt_emb(torch.cuda.LongTensor(mctask))  # [batch, window, nchoice, nmodel]
        mctaskbias = self.model.pt_emb_bias[torch.cuda.LongTensor(mctask)]

        # softmax with hidden
        output = mctaskvec.matmul(datax.unsqueeze(-1)).squeeze() + mctaskbias # [batch, window, nchoice]
        lossc = torch.nn.CrossEntropyLoss()
        loss = lossc(self.model.temperaturek*output.permute(0, 2, 1), answer_samples)
        lossm = torch.sum(loss * (1 - input_mask)) / torch.sum(1 - input_mask)

        return lossm

    def gen_mctask(self, id_samples, answer_samples, labels):
        """
        id_samples: [batch, window, nchoice]
        answer_samples: [batch, window]
        labels: [batch, window]
        """
        return id_samples.scatter_(-1, answer_samples.unsqueeze(-1), labels.unsqueeze(-1))

    def find_para_recursive(self, model, para_str):
        if hasattr(model, para_str):
            return getattr(model,para_str)
        elif hasattr(model, "submodels"):
            # print("B")
            for submodel in model.submodels:
                return self.find_para_recursive(submodel, para_str)
        else:
            raise Exception("Attr not found!")

class BiTRF_NLP_Crystal(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """

    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.trf_layers = torch.nn.ModuleList(
            [MultiHeadAttention_firstlayer(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
             PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])

        ### Decouple position
        # 2:scaling para & distance para *2 & asymmetry para
        self.posi_para = torch.nn.ParameterList(
            [ torch.nn.Parameter(torch.ones(self.n_head), requires_grad=True),
              torch.nn.Parameter(torch.ones(self.n_head), requires_grad=True),
              torch.nn.Parameter(torch.ones(self.n_head), requires_grad=True),
              torch.nn.Parameter(torch.zeros(self.n_head), requires_grad=True)]
        )
        scale_attmat_np = np.zeros((self.window_size,self.window_size))
        for ii in range(self.window_size):
            for jj in range(self.window_size):
                scale_attmat_np[ii,jj]=np.abs(ii-jj)
        self.scale_attmat = torch.cuda.FloatTensor(scale_attmat_np)

        asyn_attmat_np = np.zeros((self.window_size,self.window_size))
        for ii in range(self.window_size):
            for jj in range(self.window_size):
                if ii<jj:
                    asyn_attmat_np[ii,jj]=1
        self.asyn_attmat = torch.cuda.FloatTensor(asyn_attmat_np)

        self.h2o = torch.nn.Linear(self.model_size, self.output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)
        self.softplus = torch.nn.Softplus()

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self, model_para):
        # model_para_bert = {
        #     "model_size": 300,
        #     "hidden_size": 768,
        #     "window_size": 256,
        #     "output_size": 30001,
        #     "n_head": 12,
        #     "d_k": 768,
        #     "d_v": 768,
        #     "num_layers": 12,
        # }
        self.model_para = model_para
        self.model_size = model_para["model_size"]
        self.hidden_size = model_para["hidden_size"]
        self.window_size = model_para["window_size"]
        self.output_size = model_para["output_size"]
        self.n_head = model_para.get("n_head", 12)
        self.d_k = model_para.get("d_k", 1)
        self.d_v = model_para.get("d_v", 256)
        self.num_layers = model_para.get("num_layers", 1)

    def para(self, para):
        self.misc_para = para
        self.first_layer_flag = para.get("first_layer_flag", True)
        self.dropout = para.get("dropout", 0.1)

    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx:
        :param hidden:
        :return:
        """
        self.cuda_device = inputx.device

        # (sz_b*n_head, len_k, len_k) self.posi_para [num_layers, n_head, 4]
        # test = torch.exp(self.posi_para[1].view(-1,1,1)*self.scale_attmat.unsqueeze(0))
        posi_att = self.posi_para[0].view(-1,1,1)*(torch.exp(-self.softplus(self.posi_para[1]).view(-1,1,1)*self.scale_attmat.unsqueeze(0))
                                      +torch.exp(-self.softplus(self.posi_para[2]).view(-1,1,1)*self.scale_attmat.unsqueeze(0))
                                      + self.posi_para[3].view(-1,1,1)*self.asyn_attmat.unsqueeze(0))
        self.posi_att = posi_att

        enc_output, attn = self.trf_layers[0](inputx, inputx, inputx, posi_att)
        enc_output = self.trf_layers[1](enc_output)

        output = self.h2o(enc_output)
        output = self.softmax(output)

        return output

class MultiHeadAttention_firstlayer(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(self.__class__, self).__init__()

        self.n_head = n_head
        self.d_v = d_v
        d_k = 1
        self.d_k = d_k

        self.w_ks = torch.nn.Linear(d_model, n_head)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)


    def forward(self, q, k, v, posi_att):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        cuda_device = q.device
        assert d_k == 1

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = torch.eye(len_k, device=cuda_device)
        mask = mask.repeat(n_head * sz_b, 1, 1)  # (n*b) x .. x ..
        mask = (mask == 1)
        # output, attn = self.attention(q, k, v, mask=mask)

        attn = k.squeeze()
        temperature = np.power(d_k, 0.5)
        attn = attn / temperature

        posi_att = posi_att.view(n_head, 1, len_k, len_k).expand(n_head, sz_b, len_k, len_k).contiguous().view(-1, len_k, len_k)
        attn = attn.view(sz_b*n_head, len_k, 1) + posi_att

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_k, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output)

        return output, attn

class MultiHeadAttention_dePosi(torch.nn.Module):
    ''' Multi-Head Attention module with decoupled parameterized attentions'''

    def __init__(self, n_head, d_model, d_k, d_v, window_size,dropout=0.1):
        super(self.__class__, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.window_size = window_size

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        ### Decouple position
        # 2:scaling para & distance para *2 & asymmetry para
        self.posi_para = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(self.n_head), requires_grad=True),
             torch.nn.Parameter(torch.ones(self.n_head), requires_grad=True),
             torch.nn.Parameter(torch.ones(self.n_head)/2, requires_grad=True),
             torch.nn.Parameter(torch.zeros(self.n_head), requires_grad=True)]
        )
        scale_attmat_np = np.zeros((self.window_size, self.window_size))
        for ii in range(self.window_size):
            for jj in range(self.window_size):
                scale_attmat_np[ii, jj] = np.abs(ii - jj)
        self.scale_attmat = torch.cuda.FloatTensor(scale_attmat_np)

        asyn_attmat_np = np.zeros((self.window_size, self.window_size))
        for ii in range(self.window_size):
            for jj in range(self.window_size):
                if ii < jj:
                    asyn_attmat_np[ii, jj] = 1
        self.asyn_attmat = torch.cuda.FloatTensor(asyn_attmat_np)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        torch.nn.init.xavier_normal_(self.fc.weight)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.temperature = np.power(d_k, 0.5)
        self.psoftmax = torch.nn.Softmax(dim=2)
        self.softplus = torch.nn.Softplus()


    def forward(self, q, k, v, mask=None):

        cuda_device = q.device

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

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(n_head * sz_b, 1, 1)  # (n*b) x .. x ..
            attn = attn.masked_fill(mask, -np.inf)

        posi_att = self.posi_para[0].view(-1, 1, 1) * (
                    torch.exp(-self.softplus(self.posi_para[1]).view(-1, 1, 1) * self.scale_attmat.unsqueeze(0))
                    + torch.exp(-self.softplus(self.posi_para[2]).view(-1, 1, 1) * self.scale_attmat.unsqueeze(0))
                    + self.posi_para[3].view(-1, 1, 1) * self.asyn_attmat.unsqueeze(0))
        self.posi_att = posi_att

        posi_att = posi_att.view(n_head, 1, len_k, len_k).expand(n_head, sz_b, len_k, len_k).contiguous().view(-1,len_k,len_k)
        attn = self.psoftmax(attn+posi_att)
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output+residual)

        return output, attn

class RoBerta(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.posi_sinmat = self.get_sinusoid_encoding_table(self.window_size, self.model_size)
        # self.posi_mat, self.n_posiemb = self.posi_embed(self.window_size,
        #                                                 switch="log")  # (self.input_len,self.n_posiemb)
        if not self.layer_share:
            self.trf_layers = torch.nn.ModuleList([
                torch.nn.ModuleList([MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                    PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])
                for _ in range(self.num_layers)])
        else:
            print("ALBert layer shared version.")
            self.trf_layers = torch.nn.ModuleList([MultiHeadAttention(self.n_head, self.model_size, self.d_k, self.d_v, dropout=self.dropout),
                     PositionwiseFeedForward(self.model_size, self.hidden_size, dropout=self.dropout)])

        self.pt_emb = torch.nn.Embedding.from_pretrained(self.pt_emb_ini, freeze=False)
        self.out_bias = torch.nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.layer_norm = torch.nn.LayerNorm(self.model_size)

        self.input_mask=None

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self,model_para):
        # model_para_bert = {
        #     "model_size": 300,
        #     "hidden_size": 768,
        #     "window_size": 256,
        #     "output_size": 30001,
        #     "n_head": 12,
        #     "d_k": 768,
        #     "d_v": 768,
        #     "num_layers": 12,
        # }
        self.model_para = model_para
        self.model_size = model_para["model_size"]
        self.hidden_size = model_para["hidden_size"]
        self.window_size = model_para["window_size"]
        self.output_size = model_para["output_size"]
        self.n_head = model_para.get("n_head", 12)
        self.d_k = model_para.get("d_k", 64)
        self.d_v = model_para.get("d_v", 768)
        self.num_layers = model_para.get("num_layers", 12)
        self.pt_emb_ini = model_para.get("pt_emb_ini", None)

    def para(self,para):
        self.misc_para = para
        self.dropout = para.get("dropout", 0.1)
        self.layer_share = para.get("layer_share", False)
        self.mask_rate = para.get("mask_rate", 0.15)
        self.replace_by_mask_rate = para.get("replace_by_mask_rate", 0.8)
        self.replace_by_rand_rate = para.get("replace_by_mask_rate", 0.1)
        self.unchange_rate = para.get("replace_by_mask_rate", 0.1)
        self.mask_id = para.get("mask_id", self.output_size-1) # 30002
        self.finetune = para.get("finetune", False)

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

    def input_masking(self,inputx):
        """
        never mask <s> and </s>
        replace_by_mask_rate -> <mask> 30003
        replace_by_rand_rate -> <rand> != <s>, </s>, </pad>, </mask>
        unchange_rate -> <rand>
        :return:
        """
        ## rnd for mask
        rnd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        self.input_mask = torch.zeros((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        unmaskbool = torch.logical_or(rnd > self.mask_rate,inputx == 0) # not mask for <s>, </s>
        unmaskbool = torch.logical_or(unmaskbool, inputx == 1)
        self.input_mask[unmaskbool] = 1 # 1 means not mask

        ## rnd for branch <mask>,<rand>,<unchanged>
        rnd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        maskid=rnd>(1-self.replace_by_mask_rate)
        inputx[torch.logical_and(maskid, torch.logical_not(unmaskbool))]=self.mask_id
        rndwrd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)*(self.mask_id-3)
        rndwrd = (torch.floor(rndwrd)+2).type(torch.cuda.LongTensor)
        randid=torch.logical_and(rnd < (1 - self.replace_by_mask_rate) , rnd>(1 - self.replace_by_mask_rate- self.replace_by_rand_rate))
        inputx[torch.logical_and(randid, torch.logical_not(unmaskbool))]=rndwrd[torch.logical_and(randid, torch.logical_not(unmaskbool))]

        return inputx

    @torch.cuda.amp.autocast()
    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx: [batch, seq_len] of IDs
        :param hidden:
        :return:
        """
        self.cuda_device = inputx.device
        if self.posi_sinmat.device != inputx.device:
            self.posi_sinmat = self.posi_sinmat.to(inputx.device)

        if not self.finetune:
            inputx = self.input_masking(inputx)
        inputx = self.pt_emb(inputx)

        enc_output= self.layer_norm(inputx) + self.posi_sinmat.view(1, -1, self.model_size)

        if not self.layer_share:
            for trf_l in self.trf_layers:
                enc_output, attn = trf_l[0](enc_output, enc_output, enc_output)
                enc_output = trf_l[1](enc_output)
        else:
            for ii in range(self.num_layers):
                enc_output, attn = self.trf_layers[0](enc_output, enc_output, enc_output)
                enc_output = self.trf_layers[1](enc_output)
        if self.finetune:
            return enc_output
        else:
            output =  F.linear(enc_output, self.pt_emb.weight, self.out_bias)
            return output

class GlueFintuning(torch.nn.Module):
    """
    Bi-directional simplified Transformer for NLP, based-on source code from "attention is all you need"
    from "Yu-Hsiang Huang"
    """
    def __init__(self, bert, model_para, para=None):
        super(self.__class__, self).__init__()
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.bert = bert
        self.w_out = torch.nn.Linear(self.bert.model_size, self.output_size)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self,model_para):
        # model_para_bert = {
        #     "output_size": 2
        # }
        self.model_para = model_para
        self.output_size = model_para["output_size"]

    def para(self,para):
        self.misc_para = para

    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx: [batch, seq_len] of IDs
        :param hidden:
        :return:
        """
        bertout = self.bert(inputx)
        output = self.w_out(bertout[:,0,:])
        return output


