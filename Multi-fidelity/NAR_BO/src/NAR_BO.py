import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .NAR_Bagging import NAR_Bagging
from .activations import *

class NAR_BO:
    def __init__(self, num_models, dataset, scale, bfgs_iter, debug=True):
        self.num_models = num_models
        self.dataset = dataset
        self.scale = scale
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.standardization()
        self.dim = self.dataset['low_x'].shape[0]
        self.outdim = self.dataset['low_y'].shape[0]
        self.num_low = self.dataset['low_x'].shape[1]
        self.num_high = self.dataset['high_x'].shape[1]
        self.construct_model()

        self.best_constr = np.array([np.inf, np.inf])
        self.best_y = np.zeros((2, self.outdim))
        self.best_y[0,0] = np.inf
        self.best_y[1,0] = np.inf
        self.best_x = np.zeros((2, self.dim))
        self.get_best_y(self.dataset['low_x'], self.dataset['low_y'], is_high=0)
        self.get_best_y(self.dataset['high_x'], self.dataset['high_y'], is_high=1)
        
    def standardization(self):
        low_x = self.dataset['low_x']
        low_y = self.dataset['low_y']
        high_x = self.dataset['high_x']
        high_y = self.dataset['high_y']
        x = np.concatenate((low_x.T, high_x.T)).T
        self.in_mean = x.mean(axis=1)
        self.in_std = x.std(axis=1)
        low_x = ((low_x.T - self.in_mean)/self.in_std).T
        high_x = ((high_x.T - self.in_mean)/self.in_std).T
        self.low_mean = low_y.mean(axis=1)
        self.low_std = low_y.std(axis=1)
        low_y = ((low_y.T - self.low_mean)/self.low_std).T
        self.out_mean = high_y.mean(axis=1)
        self.out_std = high_y.std(axis=1)
        high_y = ((high_y.T - self.out_mean)/self.out_std).T

        self.dataset['low_x'] = low_x
        self.dataset['low_y'] = low_y
        self.dataset['high_x'] = high_x
        self.dataset['high_y'] = high_y

    def construct_model(self):
        dataset = {}
        dataset['low_x'] = self.dataset['low_x']
        dataset['high_x'] = self.dataset['high_x']
        self.models = []
        for i in range(self.outdim):
            dataset['low_y'] = self.dataset['low_y'][i]
            dataset['high_y'] = self.dataset['high_y'][i]
            self.models.append(NAR_Bagging(self.num_models, dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug))
            self.models[i].train(scale=self.scale[i])

    def get_best_y(self, x, y, is_high=1):
        for i in range(y.shape[1]):
            constr = np.maximum(y[1:,i], 0).sum()
            if constr < self.best_constr[is_high] and self.best_constr[is_high] > 0:
                self.best_constr[is_high] = constr
                self.best_y[is_high] = y[:,i]
                self.best_x[is_high] = x[:,i]
            elif constr <= 0 and self.best_constr[is_high] <= 0 and y[0,i] < self.best_y[is_high,0]:
                self.best_constr[is_high] = constr
                self.best_y[is_high] = y[:,i]
                self.best_x[is_high] = x[:,i]

    def EI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        if is_high:
            py, ps2 = self.models[0].predict(x)
        else:
            py, ps2 = self.models[0].predict_low(x)
        ps = np.sqrt(np.diag(ps2))
        EI = 1.0
        if self.best_constr[is_high] <= 0:
            tmp = -(py - self.best_y[is_high,0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        return EI

    def PI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        PI = 1.0
        for i in range(1,self.outdim):
            if is_high:
                py, ps2 = self.models[i].predict(x)
            else:
                py, ps2 = self.models[i].predict_low(x)
            ps = np.sqrt(np.diag(ps2))
            PI = PI*cdf(-py/ps)
        return PI

    def wEI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = self.EI(x, is_high=is_high)
        PI = self.PI(x, is_high=is_high)
        loss = -EI*PI
        if loss < self.loss:
            self.loss = loss
            self.x = np.copy(x)
        return loss

    def predict(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            tmp_py, tmp_ps2 = self.models[i].predict(test_x)
            py[i] = tmp_py
            ps2[i] = np.diag(tmp_ps2)
        py = (py.T*self.out_std + self.out_mean).T
        ps2 = ps2 * (self.out_std**2)
        return py, ps2

    def predict_low(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            tmp_py, tmp_ps2 = self.models[i].predict_low(test_x)
            py[i] = tmp_py
            ps2[i] = np.diag(tmp_ps2)
        py = (py.T*self.low_std + self.low_mean).T
        ps2 = ps2 * (self.low_std**2)
        return py, ps2





