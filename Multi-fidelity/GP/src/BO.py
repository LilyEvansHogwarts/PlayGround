import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
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
        self.dim = self.train_x.shape[0]
        self.outdim = self.train_y.shape[0]
        self.num_train = self.train_y.shape[1]
        self.construct_model()

        self.best_constr = np.inf
        self.best_y = np.zeros((self.outdim))
        self.best_y[0] = np.inf
        self.best_x = np.zeros((self.dim))
        self.get_best_y(self.train_x, self.train_y)


    def construct_model(self):
        dataset = {}
        dataset['train_x'] = self.train_x
        self.models = []
        for i in range(self.outdim):
            dataset['train_y'] = self.train_y[i:i+1]
            self.models.append(GP(dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug))
            self.models[i].train(scale=self.scale[i])
        print('BO. Finish constructing model.')

    def get_best_y(self, x, y):
        for i in range(y.shape[1]):
            constr = np.maximum(y[1:,i],0).sum()
            if constr < self.best_constr and self.best_constr > 0:
                self.best_constr = constr
                self.best_y = np.copy(y[:,i])
                self.best_x = np.copy(x[:,i])
            elif constr <= 0 and self.best_constr <= 0 and y[0,i] < self.best_y[0]:
                self.best_constr = constr
                self.best_y = np.copy(y[:,i])
                self.best_x = np.copy(x[:,i])

    def rand_x(self,n=1):
        tmp = np.random.uniform(0,1,(n))
        idx = (tmp < 0.1)
        x = np.random.uniform(-0.5, 0.5, (self.dim,n))
        x[:,idx] = (0.1*np.random.uniform(-0.5,0.5,(self.dim,idx.sum())).T + self.best_x).T
        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))
        return x

    def wEI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = np.ones((x.shape[1]))
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(np.diag(ps2))
            tmp = -(py[0] - self.best_y[0])/ps[0]
            EI = ps[0]*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.ones((x.shape[1]))
        for i in range(1,self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(np.diag(ps2))
            PI = PI*cdf(-py[i]/ps[i])
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
        
        




