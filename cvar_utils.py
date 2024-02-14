#import cvxpy as cp
import numpy as np
from scipy import stats
#from pathos import multiprocessing
#from models import PointNet
import torch 
from torch import nn 
from operator import attrgetter
from functools import partial

SOLVER = "MOSEK"
def cal_cvar(hist, alpha=0.995):
    sort_inds = np.argsort(hist)
    cvar_len = len(sort_inds) * (1 - 0.99)
    cvar_len = 10
    return np.sum(hist[sort_inds[-int(cvar_len):]]) / cvar_len

def get_cvar(risk, alpha=0.99):
    #ep = KDEProb(risk, bw=None)
    #exp_r, _= ep(risk) * risk 
    exp_r = risk
    return cal_cvar(exp_r, alpha)

def normalize(s, method='norm'):
    if method == 'norm':
        return  s / np.linalg.norm(s, axis=-1)[..., np.newaxis]
    elif method == 'max':
        return s / np.max(s, axis=-1)[..., np.newaxis]
    elif method=='sum':
        return s/ np.sum(s, axis=-1)[..., np.newaxis]
    else:
        raise AttributeError('No such normalization as \'{}\''.format(method))
    
    


class EmpericalProb:
    def __init__(self, r, bw=0.015, range=(0., 1.0)) -> None:
        if bw is None: bw = r.std()
        self.hist, self.bins = np.histogram(r, 
                                            range=(range[0], range[1]),
                                            bins=int((range[1]-range[0])/bw))
        self.size = len(r)
        self.hist = self.hist.astype(float) / float(self.size)
        self.x_pts = np.linspace(0., 0.5, 500)
    def __call__(self,r=None):
        if r is None: r = self.x_pts
        return self.hist[np.digitize(r, self.bins) - 1], r

class KDEProb:
    def __init__(self, r, bw=0.15, range=(0., 0.5)) -> None:
        self.gkde = stats.gaussian_kde(r, bw_method=bw)
        self.x_pts = np.linspace(range[0], range[1], 500)
        self.n = len(r)
    def __call__(self,r=None):
        if r is None: r = self.x_pts
        return self.gkde.evaluate(r)/self.n, r