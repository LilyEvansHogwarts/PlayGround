import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .NAR_GP import NAR_GP

class NAR_BO:
    def __init__(self, dataset, scale, bfgs_iter=[100], debug=True):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.scale = scale
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.low_x.shape[0]
        self.outdim = self.low_y.shape[0]
        self.num_low = self.low_x.shape[1]
        self.num_high = self.high_x.shape[1]

    def construct_model(self):
        dataset = {}
        dataset['low_x'] = self.low_x
        dataset['high_x'] = self.high_x
        self.model = []
        for i in range(self.outdim):
            dataset['low_y'] = self.low_y[i]
            dataset['high_y'] = self.high_y[i]
            self.model.append(NAR_GP(dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug))
            self.model[i].train(scale=self.scale[i])


