"""
Utility for remote server
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paramiko
import getpass
import time
import threading
import os
import numpy as np

import matplotlib.pyplot as plt

__author__ = "Harry He"

class remoteutil(object):
    """
    Main entrance for remote server utilization
    """
    def __init__(self):
        self.ncapath="/Users/zhengqihe/HezMain/MySourceCode/ncautil"
        self.localpath="/Users/zhengqihe/HezMain/MyWorkSpace/ContextLearn"
        self.remotepath="/home/hezq17/MyWorkSpace/test"
        self.filels=["__init__.py","ncalearn.py","ptext.py","rnnutil.py","tfnlp.py","wnetqa.py",
                     "cnutil.py","nlputil.py","remoteutil.py","seqgen.py","w2vutil.py"]
        self.server={"gpuserver":"172.16.96.245"}
        self.ssh=None
        self.sftp=None
        self.login()

    def login(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        hostin = input("Input hostname (gpuserver):")
        if len(hostin) == 0:
            hostin = "gpuserver"
        user = input("User (hezq17):")
        if len(user) == 0:
            user = "hezq17"
        pw = getpass.getpass("Password:")
        host = self.server.get(hostin, hostin)
        ssh.connect(host, username=user, password=pw)
        self.ssh=ssh

        t = paramiko.Transport((host, 22))
        t.connect(username=user, password=pw)
        sftp = paramiko.SFTPClient.from_transport(t)
        self.sftp=sftp

    def sync(self):
        ssh=self.ssh
        # ssh.exec_command("rm -r "+self.remotepath+"/python")
        # ssh.exec_command("mkdir "+self.remotepath+"/python")
        sftp=self.sftp
        try:
            stdin, stdout, stderr = ssh.exec_command("rm -r " + self.remotepath+"/ncautil")
            while not stdout.channel.exit_status_ready():
                time.sleep(0.1)
            # sftp.rmdir(self.remotepath+"/ncautil")
            sftp.mkdir(self.remotepath + "/ncautil")
        except:
            sftp.mkdir(self.remotepath + "/ncautil")
        for file in self.filels:
            sftp.put(self.ncapath+"/"+file, self.remotepath+"/ncautil/"+file)

    def singletask_multirun(self,exepy,num):
        """
        run same exepy.py for num times
        :return:
        """
        def worker(ii):
            stdin, stdout, stderr = ssh.exec_command("cd "+ self.remotepath +"/ws"+str(ii)+"; source /home/hezq17/apps/anaconda3/envs/tensorflow/bin/activate tensorflow; python " + exepy + " > output \&")
            print(stderr.readlines())
            print("end"+str(ii))

        threads = []
        ssh = self.ssh
        sftp = self.sftp
        for ii in range(num):
            try:
                stdin, stdout, stderr = ssh.exec_command("rm -r " + self.remotepath + "/ws"+str(ii))
                while not stdout.channel.exit_status_ready():
                    time.sleep(0.1)
                stdin, stdout, stderr = ssh.exec_command("mkdir " + self.remotepath + "/ws" + str(ii))
                while not stdout.channel.exit_status_ready():
                    time.sleep(0.1)
                # sftp.mkdir(self.remotepath + "/ws"+str(num))
            except:
                stdin, stdout, stderr = ssh.exec_command("mkdir " + self.remotepath + "/ws" + str(ii))
                while not stdout.channel.exit_status_ready():
                    time.sleep(0.1)
            sftp.put(self.localpath + "/" + exepy, self.remotepath+"/ws" + str(ii) + "/" + exepy)
        for ii in range(num):
            t = threading.Thread(target=worker, args=(ii,))
            threads.append(t)
            t.start()
        print("Exiting Main Thread")


if __name__ == '__main__':
    tst=remoteutil()
    tst.sync()
    tst.singletask_multirun("ContextLearn.py",5)
    print("Computation completed.")

