import numpy as np
from .GP import GP
from .activations import *
import random

class BO:
    def __init__(self, dataset, scale, bounds, bfgs_iter, debug=True):
        self.train_x = np.copy(dataset['train_x'])
        self.train_y = np.copy(dataset['train_y'])
        self.scale = scale
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.standard()
        self.dim = self.train_x.shape[0]
        self.outdim = self.train_y.shape[0]
        self.num_train = self.train_y.shape[1]
        self.construct_model()

        self.best_constr = np.inf
        self.best_y = np.zeros((self.outdim))
        self.best_y[0] = np.inf
        self.best_x = np.zeros((self.dim))
        self.get_best_y(self.train_x, self.train_y)

    def standard(self):
        self.out_mean = self.train_y.mean(axis=1)
        self.out_std = self.train_y.std(axis=1)
        self.train_y = ((self.train_y.T - self.out_mean)/self.out_std).T

    def re_standard(self, y):
        return (y.T * self.out_std + self.out_mean).T

    def construct_model(self):
        dataset = {}
        dataset['train_x'] = self.train_x
        self.models = []
        for i in range(self.outdim):
            dataset['train_y'] = self.train_y[i:i+1]
            self.models.append(GP(dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug))
            self.models[i].train(scale=self.scale[i])

    def get_best_y(self, x, y):
        for i in range(y.shape[1]):
            constr = np.maximum(y[1:,i],0).sum()
            if constr < self.best_constr and self.best_constr > 0:
                self.best_constr = constr
                self.best_y = y[:,i]
                self.best_x = x[:,i]
            elif constr <= 0 and self.best_constr <= 0 and y[0,i] < self.best_y[0]:
                self.best_constr = constr
                self.best_y = y[:,i]
                self.best_x = x[:,i]

    def rand_x(self,n=1):
        tmp = np.random.uniform(0,1,(n))
        idx = (tmp < 0.2)
        x = np.random.uniform(-0.5, 0.5, (self.dim,n))
        x[:,idx] = (0.05*np.random.uniform(-0.5,0.5,(self.dim,idx.sum())).T + self.best_x).T
        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))
        return x

    def EI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = np.ones((x.shape[1]))
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(np.diag(ps2))
            tmp = -(py - self.best_y[0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        return EI

    def PI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        PI = np.ones((x.shape[1]))
        for i in range(1,self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(np.diag(ps2))
            PI = PI*cdf(-py/ps)
        return PI

    def wEI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = np.ones((x.shape[1]))
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(np.diag(ps2))
            tmp = -(py - self.best_y[0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.ones((x.shape[1]))
        for i in range(1,self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(np.diag(ps2))
            PI = PI*cdf(-py/ps)
        return EI*PI

    def predict(self, test_x):
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            tmp_py, tmp_ps2 = self.models[i].predict(test_x)
            py[i] = tmp_py
            ps2[i] = np.diag(tmp_ps2)
        return py, ps2
        
        




