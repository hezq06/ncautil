"""
Python package for tensorflow based NLP system
Tensorflow MNIST example is referenced
Developer: Harry He
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TFNet(object):
    """
    Parent class for all TF Net object for nlp tasks
    """
    def __init__(self,option=None):
        self.lrate = option.get("lrate", 0.01)
        self.log_dir = option.get("log_dir", "log")
        self.max_steps = option.get("max_steps", 100000)
        self.batch_size = option.get("batch_size", 10)
        self.training_set = option.get("training_set", 0.6)
        self.valid_set = option.get("valid_set", 0.2)
        self.test_set = option.get("training_set", 0.2)
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

    def inference(self,invec):
        raise NotImplementedError

    def loss(self,outinf,outvec):
        raise NotImplementedError

    def evaluation(self,outinf, outvec):
        raise NotImplementedError

    def do_eval(self,sess,eval_correct,outinf,outvec):
        raise NotImplementedError

    def fill_feed_dict(self,invec,outvec,dataset,**kwargs):
        raise NotImplementedError

    def get_trainingDataSet(self,**kwargs):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def debug(self,sess,lsdebug,feed_dict):
        pass

    def training(self, loss, learning_rate):
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)
        # Create the gradient descent optimizer with the given learning rate.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        # train_op = optimizer.minimize(loss, global_step=global_step)
        train_op = optimizer.minimize(loss)
        return train_op

    def run_training(self,sess,invec,outvec,mode=None):
        loss_tab=[]
        outinf,debug_inf=self.inference(invec)
        loss,debug_loss=self.loss(outinf,outvec)
        train_op = self.training(loss, self.lrate)
        eval_correct = self.evaluation(outinf, outvec)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        sess.run(init)

        for ii in range(self.max_steps):
            start_time = time.time()
            dataset=self.get_trainingDataSet()
            feed_dict = self.fill_feed_dict(invec,outvec,dataset)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            loss_tab.append(loss_value)

            if mode == "debug":
                self.debug(sess, debug_inf+debug_loss,feed_dict)

            if ii % 1000 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (ii, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, ii)
                summary_writer.flush()

            if ii % 5000 == 1:
                checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=ii)
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.do_eval(sess,eval_correct,invec,outvec)

        plt.plot(loss_tab)
        plt.show()

        return True

    def resume_training(self,sess,save_dir,invec,outvec,mode=None):
        loss_tab = []
        outinf, debug_inf = self.inference(invec)
        loss, debug_loss = self.loss(outinf, outvec)
        train_op = self.training(loss, self.lrate)
        eval_correct = self.evaluation(outinf, outvec)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        saver.restore(sess=sess,save_path=tf.train.latest_checkpoint(save_dir))

        for ii in range(self.max_steps):
            start_time = time.time()
            dataset = self.get_trainingDataSet()
            feed_dict = self.fill_feed_dict(invec, outvec,dataset)

            if mode == "get_value":
                lsdebug = [debug_inf, debug_loss]
                debugres = sess.run(lsdebug, feed_dict=feed_dict)
                return debugres

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            loss_tab.append(loss_value)

            if ii % 1000 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (ii, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, ii)
                summary_writer.flush()

            if ii % 5000 == 1:
                checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=ii)
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.do_eval(sess,eval_correct,invec,outvec)

        plt.plot(loss_tab)
        plt.show()

        return True




class WindAppNet(TFNet):
    """
    TF network based upon window approach
    Trying to reproduce result "Natural Language Processing (Almost) from Scratch", by Ronan Collobert
    """
    def __init__(self,nlputil,option=None):
        if type(option)==type(None):
            option=dict([])
        super(WindAppNet, self).__init__(option=option)
        self.nlputil = nlputil
        self.win_size = option.get("win_size",5)
        self.hu1_size = option.get("hu1_size",100)
        self.Ntags = option.get("Ntags",46)
        self.emb_dim = option.get("emb_dim",200)
        self.goal_precision=option.get("goal_precision",0.9)

        self.featurelist=[',','.','$','``',"''",':','#','^PADDING']
        self.feature_dim = len(self.featurelist)

    def inference(self,wrd_in):
        with tf.name_scope("hidden1"):
            M1 = tf.Variable(tf.random_uniform([self.win_size*(self.emb_dim+self.feature_dim),self.hu1_size],-1.0,1.0,name="M1"))
            M1b = tf.Variable(tf.zeros([1,self.hu1_size],name="M1b"))
            nhu1=tf.nn.tanh(tf.matmul(wrd_in,M1)+M1b)
        with tf.name_scope("hidden2"):
            M2 = tf.Variable(tf.random_uniform([self.hu1_size,self.Ntags], -1.0, 1.0, name="M2"))
            M2b = tf.Variable(tf.zeros([1,self.Ntags], name="M2b"))
            nhu2 = tf.matmul(nhu1,M2)+M2b
        return nhu2

    def loss(self,nhu2,labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=nhu2, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def evaluation(self,logits, labels):
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def get_trainingDataSet(self):
        Nsents = len(self.nlputil.tagged_sents)
        trainNsents = int(self.training_set * Nsents)
        source_sents = self.nlputil.tagged_sents[:trainNsents]
        return source_sents

    def do_eval(self,
                sess,
                eval_correct,
                wrd_in,
                labels):
        # And run one epoch of eval.
        assert self.training_set+self.valid_set+self.test_set==1.0
        Nsents = len(self.nlputil.tagged_sents)

        true_count = 0  # Counts the number of correct predictions.
        wrd_count = 0
        trainNsents=int(self.training_set*Nsents)
        source_sents=self.nlputil.tagged_sents[:trainNsents]
        for ii_sents in range(trainNsents):
            for ii_wrd in range(len(source_sents[ii_sents])):
                feed_dict = self.fill_feed_dict(wrd_in,
                                           labels,
                                           source_sents,
                                           ii_sents=ii_sents,ii_wrd=ii_wrd)
                true_count += sess.run(eval_correct, feed_dict=feed_dict)
                wrd_count += 1
        precision1 = float(true_count) / wrd_count
        print('Training Set: Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (wrd_count, true_count, precision1))

        true_count = 0  # Counts the number of correct predictions.
        wrd_count = 0
        validNsents = int(self.valid_set * Nsents)
        source_sents = self.nlputil.tagged_sents[trainNsents:trainNsents+validNsents]
        for ii_sents in range(validNsents):
            for ii_wrd in range(len(source_sents[ii_sents])):
                feed_dict = self.fill_feed_dict(wrd_in,
                                           labels,
                                           source_sents,
                                           ii_sents=ii_sents,ii_wrd=ii_wrd)
                true_count += sess.run(eval_correct, feed_dict=feed_dict)
                wrd_count += 1
        precision2 = float(true_count) / wrd_count
        print('Validation Set: Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (wrd_count, true_count, precision2))

        true_count = 0  # Counts the number of correct predictions.
        wrd_count = 0
        testNsents = int(self.test_set * Nsents)
        source_sents = self.nlputil.tagged_sents[trainNsents+validNsents:trainNsents+validNsents+testNsents]
        for ii_sents in range(testNsents):
            for ii_wrd in range(len(source_sents[ii_sents])):
                feed_dict = self.fill_feed_dict(wrd_in,
                                           labels,
                                           source_sents,
                                           ii_sents=ii_sents,ii_wrd=ii_wrd)
                true_count += sess.run(eval_correct, feed_dict=feed_dict)
                wrd_count += 1
        precision3 = float(true_count) / wrd_count
        print('Test Set: Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (wrd_count, true_count, precision3))

        return precision1,precision2,precision3

    def fill_feed_dict(self,wrd_in,labels,source_sents,ii_sents=None,ii_wrd=None):
        if type(self.nlputil.w2v_dict) == type(None):
            self.nlputil.build_w2v()
        vecwrd_in = np.array([])
        veclabels = []
        if ii_sents!=None and ii_wrd!=None:
            # Given word training
            batch_size=1
        else:
            batch_size = self.batch_size
        for ii_batch in range(batch_size):
            Nsents=len(source_sents)
            rndsent=int(np.random.rand()*Nsents)
            if ii_sents != None:
                rndsent=ii_sents
            Nwrds=len(source_sents[rndsent])
            rndwrd=np.floor(np.random.rand()*Nwrds)
            if ii_wrd != None:
                rndwrd=ii_wrd
            assert self.win_size % 2 == 1
            sft=int((self.win_size-1)/2)
            # Creating vecwrd_in
            for ii in range(self.win_size):
                w2v = np.zeros(self.emb_dim)
                ExtV = np.zeros(len(self.featurelist))
                trg=rndwrd+ii-sft
                if trg<0 or trg>=Nwrds:
                    # Then we need Padding
                    ExtV[self.featurelist.index("^PADDING")]=1.0
                    tmp=np.concatenate((w2v,ExtV))
                    vecwrd_in=np.concatenate((vecwrd_in,tmp))
                else:
                    trg_word=source_sents[rndsent][int(trg)][0]
                    if trg_word in self.featurelist:
                        # Then we skip w2v and use the external feature
                        ExtV[self.featurelist.index(trg_word)] = 1.0
                        tmp = np.concatenate((w2v, ExtV))
                        vecwrd_in = np.concatenate((vecwrd_in, tmp))
                    else:
                        unkvec = self.nlputil.w2v_dict["UNK"]
                        w2v = self.nlputil.w2v_dict.get(trg_word.lower(),unkvec)
                        tmp = np.concatenate((w2v, ExtV))
                        vecwrd_in = np.concatenate((vecwrd_in, tmp))

            # Creating veclabels
            trg_type = source_sents[int(rndsent)][int(rndwrd)][1]
            ind=self.nlputil.labels.index(trg_type)
            veclabels.append(ind)

        feed_dict = {
            wrd_in: vecwrd_in.reshape((batch_size,-1)),
            labels: np.array(veclabels)
        }
        return feed_dict


    def print_test(self,save_dir):
        with tf.Graph().as_default(),tf.Session() as sess:
            loss_tab = []
            wrd_in = tf.placeholder(dtype=tf.float32, shape=(None, self.win_size * (self.emb_dim + self.feature_dim)))
            labels = tf.placeholder(dtype=tf.int32, shape=(None))
            nhu2 = self.inference(wrd_in)
            loss = self.loss(nhu2, labels)
            train_op = self.training(loss, self.lrate)
            eval_correct = self.evaluation(nhu2, labels)
            saver = tf.train.Saver()
            saver.restore(sess=sess,save_path=tf.train.latest_checkpoint(save_dir))

            Nsents = len(self.nlputil.tagged_sents)
            trainNsents = int(self.training_set * Nsents)
            validNsents = int(self.valid_set * Nsents)
            testNsents = int(self.test_set * Nsents)
            source_sents = self.nlputil.tagged_sents[trainNsents+validNsents:trainNsents+validNsents+testNsents]

            ii_sents = int(np.random.rand() * testNsents)
            wrds=[]
            labs_right=[]
            labs_inf=[]
            unk_list=[]
            for ii_wrd in range(len(source_sents[ii_sents])):
                feed_dict = self.fill_feed_dict(wrd_in,
                                                labels,
                                                source_sents,
                                                ii_sents=ii_sents, ii_wrd=ii_wrd)
                nhu2_inf = sess.run(nhu2, feed_dict=feed_dict)
                wrd=source_sents[int(ii_sents)][int(ii_wrd)][0]
                wrds.append(wrd)
                labs_right.append(source_sents[int(ii_sents)][int(ii_wrd)][1])
                labs_inf.append(self.nlputil.labels[np.argmax(nhu2_inf)])
                unk = "UNK"
                if wrd.lower() in self.nlputil.w2v_dict.keys():
                    unk = "IN"
                unk_list.append(unk)

        res=zip(wrds,labs_right,labs_inf,unk_list)
        print("Result print from setence "+str(ii_sents)+" of test dataset: (word/label right/label inf/Known)/n")
        print(list(res))

    def run(self,save_dir=None,mode=None):
        with tf.Graph().as_default(), tf.Session() as sess:
            wrd_in = tf.placeholder(dtype=tf.float32, shape=(None, self.win_size * (self.emb_dim + self.feature_dim)))
            labels = tf.placeholder(dtype=tf.int32, shape=(None))
            if type(save_dir)!=type(None):
                self.resume_training(sess,save_dir,wrd_in,labels)
            else:
                self.run_training(sess,wrd_in,labels,mode=None)



class WorMat_Syn2Gram(TFNet):
    """
    Training wordmat to learn syntax 2 gram model
    """
    def __init__(self,nlputil,option=None):
        super(WorMat_Syn2Gram, self).__init__(option=option)
        self.nlputil = nlputil
        self.in_size = option.get("in_size",48)
        self.out_size = option.get("out_size", 24)

    def inference(self,invec):
        with tf.name_scope("mat"):
            Mat = tf.Variable(tf.random_uniform([self.in_size,self.out_size],-1.0,1.0,name="Mat"))
            # out=tf.nn.tanh(tf.matmul(invec,Mat)+1.0)/2.0\
            # out=tf.matmul(invec,Mat)
            # Mb = tf.Variable(tf.zeros([1, self.out_size], name="Mb"))
            out = tf.nn.sigmoid(tf.matmul(invec,Mat))
            debug_list = [Mat]
        return out,debug_list

    def loss(self,outinf,outvec):
        EPS=1e-10
        vecdot=tf.diag_part(tf.tensordot(outinf, outvec,[1,1]))
        loutinf=tf.diag_part(tf.sqrt(tf.tensordot(outinf,outinf,[1,1])))
        loutvec=tf.diag_part(tf.sqrt(tf.tensordot(outvec,outvec,[1,1])))
        lprod=tf.multiply(loutinf,loutvec)
        zeroadj=tf.cast(tf.equal(lprod,0),tf.float32)*EPS
        loss=tf.div(vecdot,lprod+zeroadj)
        debug_list=[-tf.reduce_mean(loss),outinf,lprod,zeroadj]
        return -tf.reduce_mean(loss),[-tf.reduce_mean(loss)]

    def evaluation(self,outinf, outvec):
        return self.loss(outinf, outvec)

    def do_eval(self,sess,eval_correct,outinf,outvec):
        pass

    def fill_feed_dict(self,invec,outvec,source_sents,ii_sents=None,ii_wrd=None):
        vec_in = np.array([])
        vec_out = np.array([])
        if ii_sents!=None and ii_wrd!=None:
            # Given word training
            batch_size=1
        else:
            batch_size = self.batch_size
        for ii_batch in range(batch_size):
            Nsents=len(source_sents)
            while True:
                rndsent=int(np.random.rand()*Nsents)
                if ii_sents != None:
                    rndsent=ii_sents
                Nwrds=len(source_sents[rndsent])-2
                if Nwrds>0:
                    break
            rndwrd=np.floor(np.random.rand()*Nwrds)
            if ii_wrd != None:
                rndwrd=ii_wrd

            label1=source_sents[int(rndsent)][int(rndwrd)][1]
            label2 = source_sents[int(rndsent)][int(rndwrd)+1][1]
            label_pre = source_sents[int(rndsent)][int(rndwrd) + 2][1]
            vl1=self.nlputil.synmat.get_labvec(label1)
            vl2 = self.nlputil.synmat.get_labvec(label2)
            vlin=np.concatenate((vl1,vl2))
            vec_in = np.concatenate((vec_in, vlin))

            vl_pre = self.nlputil.synmat.get_labvec(label_pre)
            vec_out= np.concatenate((vec_out, vl_pre))

        feed_dict = {
            invec: vec_in.reshape((batch_size,-1)),
            outvec: vec_out.reshape((batch_size,-1))
        }
        return feed_dict

    def get_trainingDataSet(self):
        Nsents = len(self.nlputil.tagged_sents)
        trainNsents = int(self.training_set * Nsents)
        source_sents = self.nlputil.tagged_sents[:trainNsents]
        return source_sents

    def run(self,save_dir=None,mode=None):
        with tf.Graph().as_default(), tf.Session() as sess:
            invec = tf.placeholder(dtype=tf.float32, shape=(None, self.in_size))
            outvec = tf.placeholder(dtype=tf.float32, shape=(None, self.out_size))
            if type(save_dir)!=type(None):
                res=self.resume_training(sess,save_dir,invec,outvec,mode=mode)
            else:
                res =self.run_training(sess,invec,outvec,mode=mode)
        return res


    def debug(self,sess,lsdebug,feed_dict):
        debugres = sess.run(lsdebug, feed_dict=feed_dict)
        print("\n\n One debug print. \n\n")
        # print(debugres)
        #debugres[debug_inf+debug_loss]
        loss=debugres[-1]
        if math.isnan(loss):
            print(debugres)
            input("Input anything to continue...")




















