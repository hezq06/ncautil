"""
Utility for computational neural science
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode

__author__ = "Harry He"

class Odeutil(object):
    """
    Use scipy to do ODE integration
    """
    def __init__(self,fun,y0,para=None,t0=0,t1=10,dt=1):
        """
        Scipy ode helper
        :param fun: callable f(t, y, *f_args)
        """
        self.fun=fun
        self.y0=y0
        self.t0=t0
        self.t1 = t1
        self.dt = dt
        self.para=para
        self.res = None

    def run(self):
        res=[]
        r = ode(self.fun).set_integrator('zvode', method='bdf', with_jacobian=False)
        r.set_initial_value(self.y0, self.t0).set_f_params(self.para)
        while r.successful() and r.t < self.t1:
            r.integrate(r.t + self.dt)
            res.append([r.t]+list(r.y))
        self.res=np.array(res)
        return res

    def plot(self,row=1):
        plt.plot(self.res[:,0],self.res[:,row])
        plt.show()




