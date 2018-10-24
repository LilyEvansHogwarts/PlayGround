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
        idx = (tmp < 0.4)
        x = np.random.uniform(-0.5, 0.5, (self.dim,n))
        x[:,idx] = (0.1*np.random.uniform(-0.5,0.5,(self.dim,idx.sum())).T + self.best_x).T
        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))
        return x

    def wEI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = np.zeros((x.shape[1]))
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - self.best_y[0])/ps
            # tmp > -40
            # tmp1 = np.maximum(-40, tmp)
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            EI1 = np.log(np.maximum(0.000001, EI1))
            # tmp <= -40
            tmp2 = np.minimum(-40, tmp)**2
            EI2 = np.log(ps) - tmp2/2 - np.log(tmp2-1)
            # EI
            EI = EI1*(tmp > -40) + EI2*(tmp <= -40)
        PI = np.zeros((x.shape[1]))
        for i in range(1, self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py/ps)
        return EI+PI

    def predict(self, test_x):
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            py[i], ps2[i] = self.models[i].predict(test_x)
        return py, ps2
        
        




