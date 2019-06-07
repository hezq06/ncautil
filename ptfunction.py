"""
A module to store manually added pytorch autograd function to avoid jupyter notebook autorelaod problem.
Author: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
from torch.nn import functional as F
import numpy as np


gl_cuda_device="cuda:0"

class MySign(torch.autograd.Function): # A straight through estimation of sign
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        gpuavail = torch.cuda.is_available()
        device = torch.device(gl_cuda_device if gpuavail else "cpu")

        ctx.save_for_backward(input)

        zeros = torch.zeros(input.shape)
        if gpuavail:
            zeros = zeros.to(device)
        input[input == zeros] = 1e-8
        output=input/torch.abs(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # input = ctx.saved_tensors
        # grad_output[input > 1] = 0
        # grad_output[input < -1] = 0

        return grad_output

def mysign(input, cuda_device="cuda:0"):
    global gl_cuda_device
    gl_cuda_device=cuda_device
    return MySign.apply(input)

class MyHardSig(torch.autograd.Function): # A straight through estimation of sign
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        gpuavail = torch.cuda.is_available()
        device = torch.device(gl_cuda_device if gpuavail else "cpu")

        ctx.save_for_backward(input)

        output = torch.zeros(input.shape)
        zeros = torch.zeros(input.shape)
        if gpuavail:
            output = output.to(device)
            zeros = zeros.to(device)
        output[input >= zeros] = 1.0

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # input = ctx.saved_tensors
        # grad_output[input > 1] = 0
        # grad_output[input < -1] = 0

        return grad_output

def myhsig(input, cuda_device="cuda:0"):
    global gl_cuda_device
    gl_cuda_device=cuda_device
    return MyHardSig.apply(input)

class MyHardSample(torch.autograd.Function): # A straight through estimation of sign
    """
    Hard sampling if input>0.5 then pick
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        gpuavail = torch.cuda.is_available()
        device = torch.device(gl_cuda_device if gpuavail else "cpu")

        ctx.save_for_backward(input)

        output = torch.zeros(input.shape)
        threds = torch.zeros(input.shape)+0.5
        if gpuavail:
            output = output.to(device)
            threds = threds.to(device)
        output[input >= threds] = 1.0

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # input = ctx.saved_tensors
        # grad_output[input > 1] = 0
        # grad_output[input < -1] = 0

        return grad_output

def myhsample(input, cuda_device="cuda:0"):
    global gl_cuda_device
    gl_cuda_device=cuda_device
    return MyHardSample.apply(input)

class MySampler(torch.autograd.Function): # a 0/1 sampler following straight through estimator
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        gpuavail = torch.cuda.is_available()
        device = torch.device(gl_cuda_device if gpuavail else "cpu")

        gate = torch.rand(input.shape)
        zeros = torch.zeros(input.shape)
        if gpuavail:
            gate = gate.to(device)
            zeros = zeros.to(device)
        gate = input - gate
        gate[gate == zeros] = 1e-8
        gate = (gate / torch.abs(gate) + 1.0) / 2
        if torch.isnan(gate).any():
            raise Exception("NaN Error")

        ctx.save_for_backward(input)

        return gate

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output

def mysampler(input, cuda_device="cuda:0"):
    global gl_cuda_device
    gl_cuda_device=cuda_device
    return MySampler.apply(input)


class Gumbel_Softmax(torch.nn.Module):
    """
    PyTorch Gumbel softmax function
    Categorical Reprarameteruzation with Gumbel-Softmax
    """
    def __init__(self,cuda_device="cuda:0"):
        super(self.__class__, self).__init__()

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.gpuavail else "cpu")

    def forward(self, input, temperature=1.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """

        gi = torch.rand(input.shape)
        lpii = torch.log(input)
        if self.gpuavail:
            gi = gi.to(self.device)

        yi=torch.exp((lpii+gi)/temperature)/torch.sum(torch.exp((lpii+gi)/temperature),axis=-1)

        return yi

class Gumbel_Sigmoid(torch.nn.Module):
    """
    PyTorch GRU for Gumbel Sigmoid
    "Towards Binary-Valued Gates for Robust LSTM Training"
    """
    def __init__(self,cuda_device="cuda:0"):
        super(self.__class__, self).__init__()

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.gpuavail else "cpu")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, temperature=1.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """

        U = torch.rand(input.shape)
        if self.gpuavail:
            U = U.to(self.device)

        G=self.sigmoid((input+torch.log(U)-torch.log(1-U))/temperature)

        return G

class Gumbel_Tanh(torch.nn.Module):
    """
    PyTorch GRU for Gumbel tanh (trial)
    "Towards Binary-Valued Gates for Robust LSTM Training"
    """
    def __init__(self,cuda_device="cuda:0"):
        super(self.__class__, self).__init__()

        self.gpuavail = torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.gpuavail else "cpu")
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, temperature=1.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """

        U = torch.rand(input.shape)
        if self.gpuavail:
            U = U.to(self.device)

        # G = 2 * self.sigmoid((input + (torch.log(U) - torch.log(1 - U))) / temperature) - 1
        # G = 2 * self.sigmoid((input + (1.1-temperature)*((torch.log(U) - torch.log(1 - U)))) / temperature) - 1
        G = self.tanh((input + (torch.log(U) - torch.log(1 - U))) / temperature)
        return G

class Linear_Mask(torch.nn.Module):
    """
    A linear module with mask
    """
    def __init__(self,input_size, output_size,bias=True, cuda_device="cuda:0"):
        super(self.__class__, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.weight=torch.nn.Parameter(torch.Tensor(output_size, input_size), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_size), requires_grad=True)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self.hard_mask=torch.ones(self.weight.shape)

        self.gpuavail = torch.cuda.is_available()
        if self.gpuavail:
            self.cuda_device=cuda_device
            self.hard_mask = self.hard_mask.to(cuda_device)

    def set_hard_mask(self,hard_mask):
        self.hard_mask = hard_mask
        if self.gpuavail:
            self.hard_mask = self.hard_mask.to(self.cuda_device)

    def forward(self, input, mask=None):
        # weight_mask=self.weight
        # if mask is not None:
        #     weight_mask=torch.mul(self.weight,mask)
        # output=torch.matmul(input,weight_mask)+self.bias
        # return output
        if mask is not None:
            weight_mask = torch.mul(self.weight, mask)*self.hard_mask
        else:
            weight_mask = self.weight*self.hard_mask
        return F.linear(input, weight_mask, self.bias)
        # output=torch.matmul(input.permute(1,0,2),weight_mask)+self.bias.view(1,-1)
        # return output.permute(1,0,2)

class Linear_Sparse(torch.nn.Module):
    """
    A linear module with mask
    """
    def __init__(self,input_size, output_size, bias=True, cuda_device="cuda:0"):
        super(self.__class__, self).__init__()
        self.input_size=input_size
        self.output_size=output_size

        self.weight = torch.nn.Parameter(torch.rand(input_size)/input_size, requires_grad=True)

        # posi = torch.LongTensor([np.array(range(output_size)), np.array(range(input_size))])
        # self.weight = torch.sparse.FloatTensor(posi, self.weight_para, torch.Size([output_size, input_size]))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size), requires_grad=True)
        else:
            self.bias = None
        self.hard_mask = torch.ones(input_size)

        self.gpuavail = torch.cuda.is_available()
        if self.gpuavail:
            self.cuda_device=cuda_device
            self.hard_mask = self.hard_mask.to(cuda_device)

    def set_hard_mask(self,hard_mask):
        self.hard_mask = hard_mask
        if self.gpuavail:
            self.hard_mask = self.hard_mask.to(self.cuda_device)

    def forward(self, input):
        matm=self.weight*input*self.hard_mask
        if self.bias is not None:
            matm = matm + self.bias
        return matm




