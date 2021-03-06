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
import multiprocessing

class ScanUtil(object):

    def __init__(self,para_scan,python_main,thread_id=None):
        self.para_scan=para_scan
        self.python_main=python_main
        self.pwd=os.getcwd()
        self.thread_id=thread_id

    def run(self):
        """
        run
        :return:
        """
        for ii in range(len(self.para_scan)):
            # dirname = "Workspace" + str(self.para_scan[ii])
            dirname = "Workspace" + str(self.thread_id)
            directory = os.path.join(self.pwd,dirname)
            if not os.path.exists(directory):
                subprocess.call(['mkdir', directory])
            f = open(os.path.join(directory, "para.txt"), "a")
            f.write("Parameter for this run")
            f.write(str(self.para_scan[ii]))
            # f.write("On device " + cuda_device_list[ii%3])
            f.close()
            subprocess.call(['cp', self.python_main, os.path.join(directory, self.python_main)])
            assert type(self.para_scan[ii])== list
            subprocess.call(['python', os.path.join(directory, self.python_main)] + self.para_scan[ii] + [directory])
            # subprocess.call(['rm', os.path.join(directory, self.python_main)])

class ParallelUtil(object):
    def __init__(self,para_scan_l,python_main):
        """
        Parallel computing util
        :param para_scan_l: a list of para_scan [[para1_w1,para2_w1],[[para1_w2,para2_w2]],...]
        :param python_main:
        """
        print(para_scan_l)
        self.num=len(para_scan_l)
        self.para_scan_l=para_scan_l
        self.python_main=python_main

    def worker(self,ii):
        sutil=ScanUtil(self.para_scan_l[ii],self.python_main, thread_id=ii)
        sutil.run()

    def run(self):
        threads = []
        print("Start "+str(self.num)+" thread.")
        for ii in range(self.num):
            t = threading.Thread(target=self.worker, args=(ii,))
            threads.append(t)
            t.start()
        for ii in range(self.num):
            threads[ii].join()
        print("Exiting Main Thread")

# if __name__ == "__main__":
#
#     from ncautil.script_parascan import ParallelUtil
#
#     para1 = 1
#     para2 = 2
#
#     para_scan_l = [[[para11, para12],[para21, para22]]] # From outside to inside: multi-processing, trial case, paralist
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

### simple parallel example

# def Example_of_Parallel(procnum, return_dict):
#     """worker function"""
#     print(str(procnum) + " represent!")
#     return_dict[procnum] = procnum

def parallel_run(fun, pnum, return_flag=False):

    if return_flag:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()  # very slow if return_dict is big
        jobs = []
        for ii in range(pnum):
            p = multiprocessing.Process(target=fun, args=(ii, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        print(return_dict.values())

    else:
        jobs = []
        for ii in range(pnum):
            p = multiprocessing.Process(target=fun, args=(ii,))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
