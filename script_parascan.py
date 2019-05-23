#!/usr/bin/python

"""
A python script to help parameter scan on multi-GPU workstation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import os


cuda_device_list = ["cuda:0","cuda:1","cuda:2"]

para1 = 1
para2 = 2

para_scan = [para1,para2]

python_main = "somefile.py"

if __name__ == "__main__":

    for ii in range(len(para_scan)):
        dir="Workspace"+str(ii)
        subprocess.call(['rm','-r', dir])
        subprocess.call(['mkdir', dir])
        f = open(os.path.join(dir,"para.txt"), "a")
        f.write("Parameter for this run")
        f.write(str(para_scan[ii]))
        f.write("On device " + cuda_device_list[ii%3])
        f.close()
        subprocess.call(['cp', python_main, os.path.join(dir,python_main)])
        subprocess.call(['python',os.path.join(dir,python_main),cuda_device_list[ii%3]])


"""
somefile.py

import sys

if __name__ == "__main__":

    f = open("run.txt", "a")
    f.write("I get the parameter "+sys.argv[1])
    f.close()

"""
