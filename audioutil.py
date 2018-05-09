"""
Python package for audio processing
Refernce:
https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.io import wavfile
from scipy import signal

import librosa
import librosa.display

from ncautil.seqgen import RNN_PDC_LSTM
from torch.autograd import Variable
import torch
import time

__author__ = "Harry He"

class AudioUtil(object):
    def __init__(self):
        self.file=None
        self.sample_rate=None
        self.samples = None
        self.nperseg = None
        self.noverlap = None

    def load(self,name):
        sample_rate, samples = wavfile.read(name)
        self.sample_rate =sample_rate
        self.samples=samples
        self.file = name
        return sample_rate,samples

    def log_specgram(self, audio, sample_rate, window_size=20,
                     step_size=10, eps=1e-10,nfft=None):
        """
        https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
        :param sample_rate:
        :param window_size:
        :param step_size:
        :param eps:
        :return:
        """
        nperseg = int(round(window_size * sample_rate / 1e3))
        self.nperseg=nperseg
        noverlap = int(round(step_size * sample_rate / 1e3))
        self.noverlap=noverlap
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=sample_rate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False,
                                                nfft=nfft)
        return freqs, times, np.log(spec.astype(np.float32) + eps)


    def inverse_log_specgram(self,log_specgram,nfft=None):
        """
        https://github.com/vadim-v-lebedev/audio_style_tranfer/blob/master/audio_style_transfer.ipynb
        Inverse a log_specgram to a wav file
        :param log_specgram:
        :return:
        """

        a=np.exp(log_specgram)-1
        print(a.shape)

        # This code is supposed to do phase reconstruction

        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi

        for i in range(500):
            S = a * np.exp(1j * p)
            t,x = signal.spectral.istft(S,fs=self.sample_rate,nperseg=self.nperseg, noverlap=self.noverlap,nfft=nfft)
            # print("x",x.shape)
            f,t,sx=signal.spectral.stft(x,fs=self.sample_rate,nperseg=self.nperseg,noverlap=self.noverlap,nfft=nfft)
            p = np.angle(sx)
            # print("p",p.shape)
        return x

    def mel_specgram(self,audio, sample_rate, window_size=20,
                     step_size=10, eps=1e-10, nfft=None, **kwargs):
        """
        https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
        n_mel=128 for **kwargs
        """

        nperseg = int(round(window_size * sample_rate / 1e3))
        self.nperseg = nperseg
        noverlap = int(round(step_size * sample_rate / 1e3))
        self.noverlap = noverlap
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=sample_rate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False,
                                                nfft=nfft)
        mel_basis = librosa.filters.mel(sample_rate, nfft, **kwargs)
        melspec=np.dot(mel_basis, spec)
        return np.log(melspec.astype(np.float32) + eps),mel_basis,np.log(spec.astype(np.float32) + eps)

    def inverse_mel_specgram(self,mel_specgram,mel_basis,nfft=None, eps=1e-10):
        """
        Inverse a mel_specgram to a wav file
        :param mel_specgram:
        :return:
        """
        specgram = np.exp(mel_specgram) - 1
        # mel_basis_inv = np.dot(np.linalg.pinv(np.dot(mel_basis.T, mel_basis)), mel_basis.T)
        # mel_basis_inv=mel_basis.T
        mel_basis_inv = np.linalg.pinv(mel_basis)
        spec = np.dot(mel_basis_inv, specgram)
        res=self.inverse_log_specgram(np.log(np.abs(spec).astype(np.float32) + eps),nfft=nfft)
        return res

    def mel_specgram_rosa(self, audio, sample_rate, n_mels=128, plot=False):
        """
        https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
        https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
        :return:
        """
        S = librosa.feature.melspectrogram(audio.astype(np.float32), sr=sample_rate, hop_length=self.nperseg,n_mels=n_mels)
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        if plot:
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(log_S, sr=self.sample_rate, x_axis='time', y_axis='mel')
            plt.title('Mel power spectrogram ')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()

        return log_S

    def plt_spec(self,spec):
        plt.imshow(spec.T, aspect='auto', origin='lower')
        plt.show()



    def plt_wav(self):
        """
        https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
        :return:
        """
        if type(self.samples) == type(None):
            file=input("File not loaded yet:")
            self.load(file)
        freqs, times, spectrogram = self.log_specgram(self.samples, self.sample_rate)

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Raw wave of ' + self.file)
        ax1.set_ylabel('Amplitude')
        ax1.plot(np.linspace(0, len(self.samples)/self.sample_rate, num=len(self.samples)),self.samples)

        ax2 = fig.add_subplot(212)
        ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax2.set_yticks(freqs[::16])
        ax2.set_xticks(times[::16])
        ax2.set_title('Spectrogram of ' + self.file)
        ax2.set_ylabel('Freqs in Hz')
        ax2.set_xlabel('Seconds')
        plt.show()

class PDC_Audio(object):
    """
    Main class for Audio PDC modeling
    """
    def __init__(self):
        """
        Data structure
        self.data_train() dict{"bed":databed,"bird":databird ...}
        databed dict{"file1":data,"file2":data ...}
        data is mel_specgram
        """
        self.adu=AudioUtil()
        self.data_train=None
        self.data_valid = None
        self.data_test = None
        self.pwd="/Users/zhengqihe/HezMain/MyWorkSpace/AudioLearn/train/"
        self.entry=["bed","bird","cat","dog","down","eight","five","four","go","happy","house","left","marvin",
                    "nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two",
                    "up","wow","yes","zero"]

        self.lsize=64

        self.mlost = 1.0e99
        self.model = None

        self.ampl=None

    def data_sep(self):
        """
        Seperation of data to validation and testing sets
        :return:
        """
        pwd=self.pwd
        with open(pwd+"validation_list.txt","r") as fp:
            content = fp.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for file in content:
                os.system("mv "+pwd+"audio/"+file+" "+pwd+"validation/"+file)

        with open(pwd+"testing_list.txt","r") as fp:
            content = fp.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for file in content:
                os.system("mv " + pwd + "audio/" + file + " " + pwd + "testing/"+file)

        print("Data seperation done.")

    def data_prep(self,flimit=100):
        """
        Preparation of data

        Data structure
        self.data_train() dict{"bed":databed,"bird":databird ...}
        databed dict{"file1":data,"file2":data ...}
        data is mel_specgram

        Build list for data file name
        self.data_train() dict{"bed":databed,"bird":databird ...}
        databed dict{"file1":data,"file2":data ...} --> databed dict{"file1":data,"file2":data, "list": hashlist...}
        hashlist ["file1","file2", ... ]

        :return:
        """
        ### Preparation of training data
        data_train=dict([])
        for key in self.entry:
            print("Building "+key)
            fcnt=flimit
            keydata=dict([])
            path=self.pwd+"audio/"+key
            files=os.listdir(path)
            flist = []
            for file in files:
                fcnt=fcnt-1
                if fcnt<0:
                    fcnt=flimit
                    break
                filepath=path+"/"+file
                # print("Building "+filepath)
                databuilt,mel_basis=self.data_build(filepath)
                keydata[file]=databuilt
                flist.append(file)
                keydata["list"] = flist
            data_train[key]=keydata
        self.data_train=data_train
        self.mel_basis=mel_basis


    def data_build(self,file):
        """
        Build data
        1, load wave file
        2, cal log_mel spectrun
        :param file:
        :return:
        """
        sample_rate, samples = self.adu.load(file)
        log_mel, mel_basis, log_spec = self.adu.mel_specgram(samples, sample_rate, n_mels=self.lsize, nfft=4096)
        self.ampl = np.amax(np.abs(log_mel))
        log_mel=log_mel/np.amax(np.abs(log_mel))
        return log_mel,mel_basis

    def do_eval(self,step):
        """
        Training evaluation
        :return:
        """
        rnn=self.model
        lsize=self.lsize
        hidden = rnn.initHidden(1)
        x = Variable(torch.zeros(1,1,lsize), requires_grad=True)
        y = Variable(torch.zeros(1,1,lsize), requires_grad=True)
        outputl=[]
        outputl.append(x.data.numpy().reshape(-1,))
        for iis in range(step):
            x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
            output, hidden = rnn(x, hidden, y, cps=0.0)
            # print(outputl.reshape(1,-1).shape,output.data.numpy().reshape(1,-1).shape)
            # outputl=np.stack((outputl.reshape(1,-1),output.data.numpy().reshape(1,-1)))
            outputl.append(output.data.numpy().reshape(-1,))
            x=output
            y=output
        return np.array(outputl)


    def run_training(self,step,learning_rate=1e-2,batch=10,save=None, seqtrain=False):
        """
        Entrance for training
        :param learning_rate:
                seqtrain: If whole sequence training is used or not
        :return:
        """
        startt=time.time()

        self.mlost = 1.0e9
        lsize=self.lsize

        if type(self.model)==type(None):
            # def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
            # rnn = RNN_PDC_LSTM_AU(lsize, 50, 3, 45, lsize)
            rnn = LSTM_AU(lsize, 50, lsize)
        else:
            rnn=self.model
        rnn = LSTM_AU(lsize, 50, lsize)

        gpuavail=torch.cuda.is_available()
        device = torch.device("cuda:0" if gpuavail else "cpu")
        # If we are on a CUDA machine, then this should print a CUDA device:
        print(device)
        if gpuavail:
            rnn.to(device)

        def customized_loss(xl, yl, model):
            # print(x,y)
            # l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            # for ii, W in enumerate(model.parameters()):
            #     l2_reg = l2_reg + W.norm(1)
            loss = 0
            for ii in range(len(xl)):
                loss = loss + torch.sqrt(torch.mean((xl[ii]-yl[ii])*(xl[ii]-yl[ii])))
            return loss  # + 0.01 * l2_reg


        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

        train_hist=[]
        his = 0

        for iis in range(step):

            rdata_b=[]
            for iib in range(batch):
                rkey=self.entry[int(np.random.rand()*len(self.entry))]
                rkey="bed"
                rlist=self.data_train[rkey]["list"]
                rfile=rlist[int(np.random.rand()*len(rlist))]
                rfile=rlist[0]
                rdata=self.data_train[rkey][rfile]
                rdata_b.append(rdata)
            rdata_b=np.array(rdata_b)

            assert rdata_b[0].shape[0]==lsize

            if gpuavail:
                hidden = rnn.initHidden_cuda(device, batch)
            else:
                hidden = rnn.initHidden(batch)

            seql=rdata_b[0].shape[-1]
            if not seqtrain:
                if gpuavail:
                    rdata_b = torch.from_numpy(rdata_b)
                    rdata_b = rdata_b.to(device)
                outputl = []
                yl = []
                # One by one training
                # vec1 = rdata_b[:, :, 0]
                # x = Variable(torch.from_numpy(vec1.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                for iiss in range(seql-1):
                    vec1 = rdata_b[:,:,iiss]
                    vec2 = rdata_b[:,:,iiss + 1]
                    if gpuavail:
                        x = Variable(vec1.reshape(-1, lsize).contiguous(), requires_grad=True)
                        y = Variable(vec2.reshape(1, batch, lsize).contiguous(), requires_grad=True)
                    else:
                        x = Variable(torch.from_numpy(vec1.reshape(-1, lsize)).contiguous(), requires_grad=True)
                        y = Variable(torch.from_numpy(vec2.reshape(1, batch,lsize)).contiguous(), requires_grad=True)
                    x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                    output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                    # x=output
                    outputl.append(output)
                    yl.append(y)
                loss = customized_loss(outputl, yl, rnn)

            else:
                # LSTM provided whole sequence training
                vec1 = rdata_b[:, :, 0:seql-1]
                vec2 = rdata_b[:, :, 1:seql]

                x = Variable(torch.from_numpy(np.transpose(vec1, (2,0,1))).contiguous(), requires_grad=True)
                y = Variable(torch.from_numpy(np.transpose(vec2, (2,0,1))).contiguous(), requires_grad=True)
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
                if gpuavail:
                    x, y = x.to(device), y.to(device)
                output, hidden = rnn(x, hidden, y, cps=0.0, batch=batch)
                loss = customized_loss(output, y, rnn)


            if int(iis / 100) != his:
                print(iis, loss.data[0])
                his=int(iis / 100)
            train_hist.append(loss.data[0])

            if loss.data[0]<self.mlost:
                self.mlost=loss.data[0]
                self.model = copy.deepcopy(rnn)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        endt = time.time()
        print("Time used in training:", endt - startt)

        x = []
        for ii in range(len(train_hist)):
            x.append([ii, train_hist[ii]])
        x = np.array(x)
        plt.plot(x[:, 0], x[:, 1])
        if type(save) != type(None):
            plt.savefig(save)
            plt.gcf().clear()
        else:
            plt.show()

class RNN_PDC_LSTM_AU(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, pipe_size, context_size, output_size):
        super(RNN_PDC_LSTM_AU, self).__init__()

        self.hidden_size = hidden_size
        self.pipe_size = pipe_size
        self.input_size = input_size
        self.context_size = context_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.err2c = torch.nn.Linear(input_size*pipe_size ,context_size, bias=False)
        self.c2r1h = torch.nn.Linear(context_size, hidden_size)
        # self.c2r2h = torch.nn.Linear(context_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, hidden, result, cps=1.0, gen=0.0, batch=1):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0][0].view(1, batch, self.hidden_size)
        c0 = hidden[0][1].view(1, batch, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(1, batch, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(batch, self.hidden_size))
        output = self.tanh(output)
        errin = result - output
        errpipe=hidden[1]
        errpipe=torch.cat((errpipe[:,:,1:], errin.view(self.input_size,batch,-1)),2)
        # context = self.hardtanh(hidden[2]) #2*self.sigmoid(hidden[2])-1
        context=hidden[2]
        # context = self.hardtanh(context+(2*self.sigmoid(self.err2c(errpipe.view(1,-1)))-1)+0.1*(2*self.sigmoid(self.c2c(context))-1))
        # context = self.tanh(context + (2 * self.sigmoid(self.err2c(errpipe.view(1, -1))) - 1) + (
        #         2 * self.sigmoid(self.c2c(context)) - 1))
        context = self.hardtanh(context + (1.0-gen)*self.tanh(self.err2c(errpipe.view(1, batch, -1))))
        # hidden1 = hidden1 * self.c2r1h(context)
        # hidden1 = hidden1 + cps*self.c2r1h(context)
        # hidden1 = hidden1*self.c2r2h(context)+ cps*self.c2r1h(context)
        hidden1 = hidden1
        return output, [(hidden1,c1), errpipe, context]

    def initHidden(self,batch):
        return [(Variable(torch.zeros(1, batch,self.hidden_size), requires_grad=True),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True),
                Variable(torch.zeros(1, batch, self.context_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [(Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device)),
                Variable(torch.zeros(self.input_size, batch, self.pipe_size), requires_grad=True).to(device),
                Variable(torch.zeros(1, batch, self.context_size), requires_grad=True).to(device)]

class LSTM_AU(torch.nn.Module):
    """
    PyTorch LSTM PDC for Audio
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_AU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        # self.c2r2h = torch.nn.Linear(context_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh=torch.nn.Hardtanh()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, hidden, result, cps=1.0, gen=0.0, batch=1):
        """

        :param input:
        :param hidden: [(lstm h0, c0),(errpipe),context]
        :param result:
        :return:
        """
        hidden0=hidden[0].view(1, batch, self.hidden_size)
        c0 = hidden[1].view(1, batch, self.hidden_size)
        hout, (hidden1,c1) = self.lstm(input.view(-1, batch, self.input_size), (hidden0,c0))
        output = self.h2o(hout.view(-1, batch, self.hidden_size))
        output = self.tanh(output)
        return output, (hidden1,c1)

    def initHidden(self,batch):
        return [Variable(torch.zeros(1, batch,self.hidden_size), requires_grad=True),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True)]

    def initHidden_cuda(self,device, batch):
        return [Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device),Variable(torch.zeros(1, batch, self.hidden_size), requires_grad=True).to(device)]










































