"""
PyTorch Modules of All kinds vol2 starting 2019/10/18
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from ncautil.seqmultip2 import FF_MLP, Gumbel_Softmax


def cnn_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin+2*padding-dilation*(kernel_size-1)-1)/stride+1
    HWout = np.floor(HWout)
    return HWout

def deconv_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin-1)*stride-2*padding+kernel_size
    return HWout

class CAL_LOSS(torch.nn.Module):
    """
    An abstract loss wrapper
    """

    def __init__(self, model, para=None):
        super(self.__class__, self).__init__()
        self.model = model
        if para is None:
            para = dict([])
        self.para(para)

    def forward(self, datax, labels, schedule=1.0):

        output = self.model(datax, schedule=schedule)
        self.output = output
        if self.loss_flag == "cross_entropy":
            loss = self.CrossEntropyLoss(output, labels)
        elif self.loss_flag == "mse_loss":
            loss = self.MSELoss(output, labels[...,:output.shape[-2],:output.shape[-1]])*10000

        # print("Loss,",loss)

        if hasattr(self.model, "loss_reg"):
            loss = loss +self.model.loss_reg
            # print("A",self.model.loss_reg)
        if hasattr(self.model, "submodels"):
            for submodel in self.model.submodels:
                if hasattr(submodel, "loss_reg"):
                    loss = loss + submodel.loss_reg
                    # print("B", submodel.loss_reg)
        return loss

    def para(self,para):
        self.misc_para = para
        self.loss_flag = para.get("loss_flag", "cross_entropy")

    def CrossEntropyLoss(self, output, labels):
        device = labels.device
        lossc = torch.nn.CrossEntropyLoss()
        loss = lossc(output, labels.type(torch.LongTensor).to(device))
        return loss

    def MSELoss(self, output, labels):
        loss_mse = torch.nn.MSELoss()
        loss = loss_mse(output, labels)
        return loss

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
        self.submodels = [seq1_coop, seq2_train]

        if para is None:
            para = dict([])
        self.para(para)

        if not self.seq1_train_flag:
            for param in self.seq1_coop.parameters():
                param.requires_grad = False
        self.seq1_coop.coop_mode = True

        self.save_para = {
            "model_para":[seq1_coop.save_para, seq2_train.save_para],
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self,para):
        self.misc_para = para
        self.seq1_train_flag = para.get("seq1_train_flag", False)
        self.loss_flag = para.get("loss_flag", "cross_entropy")

    def forward(self, datax, schedule=1.0):
        self.device = datax.device
        if self.seq1_train_flag:
            out_seq1 = self.seq1_coop(datax, schedule=schedule)
        else:
            out_seq1 = self.seq1_coop(datax, schedule=1.0)
        self.out_seq1=out_seq1
        # self.auto_code_context=torch.cat((self.seq1_coop.cooprer.context,self.seq1_coop.trainer.context),dim=-1)
        # out_seq2 = self.seq2_train(out_seq1, labels, schedule=schedule)

        output = self.seq2_train(out_seq1, schedule=schedule)
        self.output = output

        return output

class ABS_CNN_COOP(torch.nn.Module):
    """
    An abstract cooperation container for Cooperation model
    """

    def __init__(self, cooprer, trainer, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.cooprer = cooprer
        self.trainer = trainer
        self.submodels = [cooprer, trainer]

        if para is None:
            para = dict([])
        self.para(para)

        if not self.cooprer_train_flag:
            for param in self.cooprer.parameters():
                param.requires_grad = False
        self.cooprer.coop_mode = True

        self.save_para = {
            "model_para": [cooprer.save_para, trainer.save_para],
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self, para):
        self.misc_para = para
        self.cooprer_train_flag = para.get("cooprer_train_flag", False)

    def forward(self, datax, schedule=1.0):
        self.device = datax.device
        if self.cooprer_train_flag:
            out_coop = self.cooprer(datax, schedule=schedule)
        else:
            out_coop = self.cooprer(datax, schedule=1.0)
        out_train = self.trainer(datax, schedule=schedule)
        output = torch.cat([out_coop,out_train],dim=-1)
        return output

class ConvNet(torch.nn.Module):
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

        if self.infobn_flag:
            self.infobn=GSVIB_InfoBottleNeck(self.infobn_model,para=self.infobn_para)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self,para):
        self.misc_para = para
        self.coop_mode = para.get("coop_mode", False)
        self.infobn_flag = para.get("infobn_flag", False)
        self.infobn_para = para.get("infobn_para", None)
        self.sfmsample_flag = para.get("sfmsample_flag", False)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.loss_flag = para.get("loss_flag", "colorshape")

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
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, datax, schedule=1.0):
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
            if self.infobn_flag:
                output = self.infobn(output, schedule=schedule)
                self.context = self.infobn.context.view(*self.infobn.context.shape[:-2],-1)
                self.output = output
                return output
            elif self.sfmsample_flag:
                if self.loss_flag == "colorshape":
                    output0s = self.softmax(output[:, :8])
                    output1s = self.softmax(output[:, 8:])
                    self.context = torch.cat([ output0s, output1s], dim=-1)
                    output0 = self.gsoftmax(output0s, temperature=np.exp(-5))
                    output1 = self.gsoftmax(output1s, temperature=np.exp(-5))
                    output = torch.cat([output0,output1],dim=-1)
                elif self.loss_flag == "posicolorshape":
                    output0s = self.softmax(output[:, 2:10])
                    output1s = self.softmax(output[:, 10:])
                    self.context = torch.cat([output[:,:2],output0s,output1s],dim=-1)
                    output0 = self.gsoftmax(output0s, temperature=np.exp(-5))
                    output1 = self.gsoftmax(output1s, temperature=np.exp(-5))
                    output = torch.cat([output[:,:2],output0,output1],dim=-1)
                self.output = output
                return output
            else:
                self.output = output
                return output


        ## Loss, posi
        self.output = output
        # device = datax.device
        # if self.loss_flag == "posicolorshape":
        #     loss_mse = torch.nn.MSELoss()
        #     lossposi = loss_mse(output[:,0:2],labels[:,0:2].type(torch.FloatTensor).to(device))
        #     ## Loss, color
        #     lossc = torch.nn.CrossEntropyLoss()
        #     losscolor = lossc(output[:, 2:10], labels[:, 2].type(torch.LongTensor).to(device))
        #     ## Loss, shape
        #     lossshape = lossc(output[:, 10:], labels[:, 3].type(torch.LongTensor).to(device))
        #     loss = losscolor+lossshape+lossposi
        # elif self.loss_flag == "colorshape":
        #     ## Loss, color
        #     lossc = torch.nn.CrossEntropyLoss()
        #     losscolor = lossc(output[:, :8], labels[:,0].type(torch.LongTensor).to(device))
        #     ## Loss, shape
        #     lossshape = lossc(output[:, 8:], labels[:, 1].type(torch.LongTensor).to(device))
        #     loss = losscolor+lossshape

        return output

class DeConvNet(torch.nn.Module):
    """
    Deconvolutional Net for auto-encoding
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)
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

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
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

    def para(self,para):
        self.misc_para = para
        self.coop_mode = para.get("coop_mode", False)

    def forward(self, datax, schedule=1.0):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """
        batch = datax.shape[0]

        datax = self.layer_norm0(datax)
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
        if self.coop_mode:
            return self.outimage

        return tconvin

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

class ConvFF_MLP_CLEVR(torch.nn.Module):
    """
    Convolutional FF for CLEVR multiple choice.
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.HiddenConvFF = FF_MLP(model_para["HiddenConvFF"], para=para) # [b, 4, objN, hd]-->[b, 4, objN, 1]
        self.ObjConvFF = FF_MLP(model_para["ObjConvFF"], para=para) # [b, 4, objN] --> [b, 4, 1]

        if para is None:
            para = dict([])
        self.para(para)

        self.save_para = {
            "model_para": model_para,
            "type": "ConvFF_MLP_CLEVR",
            "misc_para": para
        }

        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)
        self.layer_norm = torch.nn.LayerNorm(model_para["HiddenConvFF"]["input_size"])

    def para(self,para):
        self.misc_para=para
        self.dropout_rate = para.get("dropout_rate", 0.0)
        self.sfmsample_flag = para.get("sfmsample_flag", True)
        self.loss_flag = para.get("loss_flag", "posicolorshape")
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def forward(self, datax, schedule=None):
        """
        Forward
        :param datax: [b, 4, objN, hd]
        :param hidden:
        :return:
        """
        ### softmax sample
        if self.sfmsample_flag:
            if self.loss_flag == "posicolorshape":
                output0 = self.gsoftmax(datax[:,:,:, 2:10], temperature=np.exp(-5))
                output1 = self.gsoftmax(datax[:,:,:, 10:13], temperature=np.exp(-5))
                output2 = self.gsoftmax(datax[:,:,:, 13:].view([*datax.shape[0:3],-1,2]), temperature=np.exp(-5))
                output2 = output2.view([*output2.shape[0:3],-1])
                datax = torch.cat([datax[:,:,:, :2], output0, output1, output2], dim=-1)
                datax = self.layer_norm(datax)

        datax = self.HiddenConvFF(datax)[0].squeeze()
        output = self.ObjConvFF(datax)[0].squeeze()
        self.output = output

        return output

class Multitube_FF_MLP(torch.nn.Module):
    """
    Feed forward multi-tube perceptron (a multiple information flow tube, no mixing within tubes, each tube is an FF)
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        # self.coop_mode=False
        self.set_model_para(model_para)

        assert len(self.input_divide) == len(self.output_divide) == len(self.mlp_layer_para)
        assert  np.sum(self.input_divide) == self.input_size
        assert np.sum(self.output_divide) == self.output_size

        model_paral = []
        for iim in range(len(self.input_divide)):
            model_paral.append({
                    "input_size": self.input_divide[iim],
                    "output_size":self.output_divide[iim],
                    "mlp_layer_para": self.mlp_layer_para[iim]
            })

        self.ff_tubes = torch.nn.ModuleList([
            FF_MLP(model_paral[iim])
            for iim in range(len(self.input_divide))])

        if para is None:
            para = dict([])
        self.para(para)

        if self.infobn_flag:
            self.infobnl = torch.nn.ModuleList([
                GSVIB_InfoBottleNeck(self.infobn_model[iim],para=self.infobn_para)
                for iim in range(len(self.input_divide))])

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.output_size)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self,para):
        self.misc_para = para
        self.coop_mode = para.get("coop_mode", True)
        self.infobn_flag = para.get("infobn_flag", True)
        self.infobn_para = para.get("infobn_para", None)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.sfmsample_flag = para.get("sfmsample_flag", True)
        self.loss_flag = para.get("loss_flag", "posicolorshape")

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 2+8+3+64,
        #     "input_divide":[2,8,3,64]
        #     "output_size":12+12+12+32,
        #     "output_divide":[16,16,16,32]
        #     "mlp_layer_para": [[16,16],[16,16],[16,16],[32,32]]
        #     "infobn_model": [{"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":8},
        #                    {"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":16}]
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.input_divide = model_para["input_divide"]
        self.output_size = model_para["output_size"]
        self.output_divide = model_para["output_divide"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0l, hidden1l, ...]
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, datax, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        ### softmax sample
        if self.sfmsample_flag:
            if self.loss_flag == "posicolorshape":
                output0 = self.gsoftmax(datax[... , 2:10], temperature=np.exp(-5))
                output1 = self.gsoftmax(datax[... , 10:13], temperature=np.exp(-5))
                output2 = self.gsoftmax(datax[... , 13:].view([*datax.shape[0:3], -1, 2]), temperature=np.exp(-5))
                output2 = output2.view([*output2.shape[0:3], -1])
                datax = torch.cat([datax[... , :2], output0, output1, output2], dim=-1)
                datax = self.layer_norm0(datax)

        startp=0
        dataml=[]
        for iim, fmd in enumerate(self.ff_tubes):
            datam, _ = fmd(datax[...,startp:startp+self.input_divide[iim]])
            dataml.append(datam)
            startp+=self.input_divide[iim]

        gsamplel = []
        contextl = []
        if self.infobn_flag:
            for iim, infobn in enumerate(self.infobnl):
                gsample = infobn(dataml[iim], schedule=schedule)
                gsamplel.append(gsample)
                contextl.append(infobn.contextl)
            self.loss_reg = 0
            for iim in range(len(self.input_divide)):
                self.loss_reg = self.loss_reg + self.infobnl[iim].cal_regloss()


        self.contextl = torch.cat(contextl, dim=-1)
        output = torch.cat(gsamplel, dim=-1)
        output = self.layer_norm1(output)
        return output

class GSVIB_InfoBottleNeck(torch.nn.Module):
    """
    A GSVIB information bottleneck module
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num, self.gs_head_dim)), requires_grad=True)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": "GSVIB_InfoBottleNeck",
            "misc_para": para
        }

    def para(self,para):
        self.misc_para=para
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.reg_lamda = para.get("reg_lamda", 0.1) # weight on KL
        self.scale_factor = para.get("scale_factor", 1.0) # weight on entropy

    def set_model_para(self,model_para):
        # model_para = {
        #     "gs_head_dim": 2,
        #     "gs_head_num": 64
        # }
        self.model_para = model_para
        self.gs_head_dim = model_para["gs_head_dim"]
        self.gs_head_num = model_para["gs_head_num"]

    def forward(self, datax, schedule=1.0):

        assert datax.shape[-1] == self.gs_head_num * self.gs_head_dim

        if schedule < 1.0:  # Multi-scanning trial
            schedule = schedule * self.temp_scan_num
            schedule = schedule - np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        context = datax.view([*datax.shape[:-1], self.gs_head_num, self.gs_head_dim])
        context = self.softmax(context)
        self.context = context
        self.contextl = context.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])
        gssample = self.gsoftmax(context, temperature=temperature)

        self.loss_reg = self.cal_regloss()

        if self.sample_mode:
            return gssample.view([*datax.shape[:-1], self.sample_size,self.gs_head_num* self.gs_head_dim])
        else:
            return gssample.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])

    def cal_regloss(self):
        # context,prior should be log probability
        prior = self.softmax(self.prior)
        ent_prior = -torch.mean(torch.sum(torch.exp(prior) * prior, dim=-1))
        prior = prior.view(1, self.gs_head_num, self.gs_head_dim).expand_as(self.context)
        flatkl = torch.mean(torch.sum(torch.exp(self.context) * (self.context - prior), dim=-1))
        loss =  self.reg_lamda * (flatkl + self.scale_factor * ent_prior)
        return loss




