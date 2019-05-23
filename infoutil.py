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
    # Getting x data

    dataxtemp=np.array(proc_evalmem[0])[:,xposi]
    datax=np.array([1 if x==xdig else 0 for x in dataxtemp])
    # Getting y data
    if ysel=="ht_enc":
        datay=np.array(proc_evalmem[2])[:,yposi,yneu]
    else:
        raise Exception("Unknown ysel")
    return datax,datay

def parl(xdig,proc_evalmem,hdn,seql,ysel,yposi):
    resmatp = np.zeros((hdn, seql))
    for iis in tqdm(range(seql)):
        for iih in range(hdn):
            datax, datay = data_fetch(proc_evalmem, xdig=xdig, xposi=iis, ysel=ysel, yneu=iih, yposi=yposi)
            data_x_hist, _ = cal_hist(datax, bins=2)
            data_y_hist, _ = cal_hist(datay, bins=20)
            data_xy_hist, _, _ = cal_hist(np.array(list(zip(datax, datay))), bins=[2, 20])
            muinfo = cal_mulinfo(data_x_hist, data_y_hist, data_xy_hist)
            resmatp[iih, iis] = muinfo
    return resmatp

def plot_infomat(proc_evalmem,ysel="ht_enc",yposi=0,clim=(-0.3,0.3)):
    assert clim is not None
    numdig=10
    seql=len(proc_evalmem[0][0])
    hdn=proc_evalmem[2][0].shape[-1]

    # Serial
    # resmat=[]
    # for xdig in range(numdig):
    #     print("Processing " ,xdig)
    #     resmat.append(parl(xdig,proc_evalmem,hdn,seql,ysel,yposi))

    # Parallel
    resmat = Parallel(n_jobs=numdig)(delayed(parl)(xdig,proc_evalmem,hdn,seql,ysel,yposi) for xdig in range(numdig))
    resmat = np.array(resmat)

    f, axes = plt.subplots(1,numdig, sharey=True)
    for ii in range(numdig):
        plt_t = plot_mat_ax(axes[ii], resmat[ii,:,:], title=str(ii), tick_step=1,clim=clim)
    plt.colorbar(plt_t, ax=axes[-1])
    plt.show()
    return resmat

def get_anim_infomat(proc_evalmem,ysel="ht_enc",yposi_range=9,numdig=10):

    seql = len(proc_evalmem[0][0])
    hdn = proc_evalmem[2][0].shape[-1]

    parlist=[]
    for yposi in range(yposi_range):
        for xdig in range(numdig):
            parlist.append((yposi,xdig))
    num_cores = multiprocessing.cpu_count()
    resmat = Parallel(n_jobs=20)(delayed(parl)(xdig, proc_evalmem, hdn, seql, ysel, yposi) for (yposi,xdig) in parlist)
    resmat_list=[]
    for yposi in range(yposi_range):
        resmat_list.append(np.array(resmat[yposi*numdig:yposi*numdig+numdig]))

    return resmat_list

def show_anim(resmat_list,numdig=10,clim=(-0.3,0.3)):

    fig, axes = plt.subplots(1, numdig, sharey=True, figsize=(200,100))

    global lab_colorbar
    lab_colorbar=True
    def update(frame):
        for ii in range(numdig):
            plt_t = plot_mat_ax(axes[ii], resmat_list[frame][ii, :, :], title=str(ii), tick_step=1, clim=clim)
        global lab_colorbar
        if lab_colorbar:
            plt.colorbar(plt_t, ax=axes[-1])
            lab_colorbar=False

    ani = FuncAnimation(fig, update, frames=10, repeat=True, init_func=None, blit=False)
    plt.show()

    return ani
