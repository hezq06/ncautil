#!/usr/bin/python

"""
A python script to help parameter scan on multi-GPU workstation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import os
import threading

class ScanUtil(object):

    def __init__(self,para_scan,python_main):
        self.para_scan=para_scan
        self.python_main=python_main
        self.pwd=os.getcwd()

    def run(self):
        for ii in range(len(self.para_scan)):
            dirname = "Workspace" + str(self.para_scan[ii])
            directory = os.path.join(self.pwd,dirname)
            try:
                subprocess.call(['rm', '-r', directory])
            except:
                pass
            subprocess.call(['mkdir', directory])
            f = open(os.path.join(directory, "para.txt"), "a")
            f.write("Parameter for this run")
            f.write(str(self.para_scan[ii]))
            # f.write("On device " + cuda_device_list[ii%3])
            f.close()
            subprocess.call(['cp', self.python_main, os.path.join(directory, self.python_main)])
            subprocess.call(['python', os.path.join(directory, self.python_main)] + self.para_scan[ii] + [directory])
            subprocess.call(['rm', os.path.join(directory, self.python_main)])

class ParallelUtil(object):
    def __init__(self,para_scan_l,python_main):
        """
        Parallel computing util
        :param para_scan_l: a list of para_scan [[para1_w1,para2_w1],[[para1_w2,para2_w2]],...]
        :param python_main:
        """
        self.num=len(para_scan_l)
        self.para_scan_l=para_scan_l
        self.python_main=python_main

    def worker(self,ii):
        sutil=ScanUtil(self.para_scan_l[ii],self.python_main)
        sutil.run()

    def run(self):
        threads = []
        print("Start "+str(self.num)+" thread.")
        for ii in range(self.num):
            t = threading.Thread(target=self.worker, args=(ii,))
            threads.append(t)
            t.start()
        print("Exiting Main Thread")

# if __name__ == "__main__":
#
#     para1 = 1
#     para2 = 2
#
#     para_scan_l = [[para1],[para2]]
#
#     python_main = "somefile.py"
#
#     plu=ParallelUtil(para_scan_l,python_main)
#     plu.run()


"""
somefile.py

import sys

if __name__ == "__main__":

    f = open("run.txt", "a")
    f.write("I get the parameter "+sys.argv[1])
    f.close()

"""
