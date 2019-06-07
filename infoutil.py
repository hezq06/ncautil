"""
Python package for information theory utility
Developer: Harry He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from ncautil.ncamath import *
from ncautil.ncalearn import *
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

def data_fetch(proc_evalmem,xdig=1,xposi=0,ysel="ht_enc",yneu=0,yposi=0):
    """
    proc_evalmem: data # x,label,enc(ht,zt,nt),dec(ht,zt,nt)
    xdig: neuron of input sensitive to xdig
    xposi: neuron of input in the position xposi
    ysel: selection of y kind
    yneu: # of neuron
    yposi: position of y
    """
    if ysel=="ht_enc":
        dataxtemp = np.array(proc_evalmem[0])[:, xposi]
        datax = np.array([1 if x == xdig else 0 for x in dataxtemp])
        datay=np.array(proc_evalmem[2])[:,yposi,yneu]
    elif ysel=="ht_dec":
        datax = np.array(proc_evalmem[5])[:, yposi, yneu]
        dataytemp = np.array(proc_evalmem[1])[:, xposi]
        datay = np.array([1 if x == xdig else 0 for x in dataytemp])
    else:
        raise Exception("Unknown ysel")
    return datax,datay

def parl(xdig,proc_evalmem,hdn,seql,ysel,yposi,xres=2,yres=20):
    resmatp = np.zeros((hdn, seql))
    for iis in tqdm(range(seql)):
        for iih in range(hdn):
            datax, datay = data_fetch(proc_evalmem, xdig=xdig, xposi=iis, ysel=ysel, yneu=iih, yposi=yposi)
            data_x_hist, _ = cal_hist(datax, bins=xres)
            data_y_hist, _ = cal_hist(datay, bins=yres)
            data_xy_hist, _, _ = cal_hist(np.array(list(zip(datax, datay))), bins=[xres, yres])
            muinfo = cal_mulinfo(data_x_hist, data_y_hist, data_xy_hist)
            resmatp[iih, iis] = muinfo
    return resmatp

def plot_infomat(proc_evalmem,ysel="ht_enc",yposi=0,clim=(-0.3,0.3),numdig=13):
    assert clim is not None
    seql=len(proc_evalmem[0][0])
    hdn=proc_evalmem[2][0].shape[-1]

    # Serial
    resmat=[]
    for xdig in range(numdig):
        print("Processing " ,xdig)
        resmat.append(parl(xdig,proc_evalmem,hdn,seql,ysel,yposi))

    # Parallel
    # resmat = Parallel(n_jobs=numdig)(delayed(parl)(xdig,proc_evalmem,hdn,seql,ysel,yposi) for xdig in range(numdig))
    # resmat = np.array(resmat)

    f, axes = plt.subplots(1,numdig, sharey=True)
    titlel=["0","1","2","3","4","5","6","7","8","9","+","=","#"]
    for ii in range(numdig):
        plt_t = plot_mat_ax(axes[ii], resmat[ii,:,:], title=titlel[ii], tick_step=1,clim=clim)
    plt.colorbar(plt_t, ax=axes[-1])
    plt.show()
    return resmat

def get_anim_infomat(proc_evalmem,ysel="ht_dec",xposi_range=9, yposi_range=9,numdig=10):

    hdn = proc_evalmem[2][0].shape[-1]

    parlist=[]
    for yposi in range(yposi_range):
        for xdig in range(numdig):
            parlist.append((yposi,xdig))
    num_cores = multiprocessing.cpu_count()
    resmat = Parallel(n_jobs=20)(delayed(parl)(xdig, proc_evalmem, hdn, xposi_range, ysel, yposi) for (yposi,xdig) in parlist)
    resmat_list=[]
    for yposi in range(yposi_range):
        resmat_list.append(np.array(resmat[yposi*numdig:yposi*numdig+numdig]))

    return resmat_list

def show_anim(resmat_list,numdig=10,clim=(-0.3,0.3),fps=5):

    fig, axes = plt.subplots(1, numdig, sharey=True, figsize=(200,100))
    titlel = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "#"]

    global lab_colorbar
    lab_colorbar=True
    def update(frame):
        for ii in range(numdig):
            plt_t = plot_mat_ax(axes[ii], resmat_list[frame][ii, :, :], title=titlel[ii], tick_step=1, clim=clim)
        global lab_colorbar
        if lab_colorbar:
            plt.colorbar(plt_t, ax=axes[-1])
            lab_colorbar=False

    ani = FuncAnimation(fig, update, frames=len(resmat_list), repeat=False, init_func=None, blit=False, interval=1000/fps)
    plt.show()

    return ani

def save_anim(ani,title,fps=1):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Harry'), bitrate=1800)
    ani.save(title, writer=writer)
