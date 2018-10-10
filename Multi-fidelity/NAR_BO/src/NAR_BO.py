import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .NAR_Bagging import NAR_Bagging
from .activations import *
import random

class NAR_BO:
    def __init__(self, num_models, dataset, scale, bounds, bfgs_iter, debug=True):
        self.num_models = num_models
        self.dataset = dataset
        self.scale = scale
        self.bounds = np.copy(bounds)
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

    def rand_x(self):
        x = np.zeros((self.dim,1))
        for i in range(self.dim):
            x[i,0] = random.uniform(self.bounds[i,0], self.bounds[i,1])
        return x

    # x.size >= self.dim
    def EI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        if is_high:
            _, _, py, ps2 = self.models[0].predict(x)
        else:
            py, ps2 = self.models[0].predict_low(x)
        ps = np.sqrt(np.diag(ps2))
        EI = 1.0
        if self.best_constr[is_high] <= 0:
            tmp = -(py - self.best_y[is_high,0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        return EI

    # x.size >= self.dim
    def PI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        PI = 1.0
        for i in range(1,self.outdim):
            if is_high:
                _, _, py, ps2 = self.models[i].predict(x)
            else:
                py, ps2 = self.models[i].predict_low(x)
            ps = np.sqrt(np.diag(ps2))
            PI = PI*cdf(-py/ps)
        return PI
    
    # x.size >= self.dim
    def wEI(self, x, is_high=1):
        x = x.reshape(self.dim, int(x.size/self.dim))
        if is_high:
            _, _, py, ps2 = self.predict(x)
        else:
            py, ps2 = self.predict_low(x)
        ps = np.sqrt(ps2)
        EI = 1.0
        if self.best_constr[is_high] <= 0:
            tmp = -(py[0] - self.best_y[is_high,0])/ps[0]
            EI = ps[0]*(tmp*cdf(tmp)+pdf(tmp))
        PI = 1.0
        for i in range(1,self.outdim):
            PI = PI*cdf(-py[i]/ps[i])
        return EI*PI

    def predict(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        num_test = test_x.shape[1]
        py1 = np.zeros((self.outdim, num_test))
        ps21 = np.zeros((self.outdim, num_test))
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            tmp_py1, tmp_ps21, tmp_py, tmp_ps2 = self.models[i].predict(test_x)
            py1[i] = tmp_py1
            ps21[i] = np.diag(tmp_ps21)
            py[i] = tmp_py
            ps2[i] = np.diag(tmp_ps2)
        py1 = (py1.T*self.low_std + self.low_mean).T
        ps21 = ps21 * (self.low_std**2)
        py = (py.T*self.out_std + self.out_mean).T
        ps2 = ps2 * (self.out_std**2)
        return py1, ps21, py, ps2

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

