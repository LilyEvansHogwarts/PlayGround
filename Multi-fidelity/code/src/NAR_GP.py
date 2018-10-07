import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .GP import GP

class NAR_GP:
    def __init__(self, dataset, scale=1.0, bfgs_iter=100, debug=True):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug

    def train(self, scale=1.0):
        dataset = {}
        dataset['train_x'] = self.low_x
        dataset['train_y'] = self.low_y
        model1 = GP(dataset, bfgs_iter=self.bfgs_iter, debug=self.debug)
        model1.train(scale=scale)
        self.model1 = model1
        
        mu, v = self.model1.predict(self.high_x)
        dataset['train_x'] = np.concatenate((self.high_x, mu.reshape(1,-1)))
        dataset['train_y'] = self.high_y
        model2 = GP(dataset, bfgs_iter=self.bfgs_iter, debug=self.debug, k=1)
        model2.train(scale=scale)
        self.model2 = model2

    def predict(self, test_x):
        nsamples = 100
        mu, v = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(mu, v, nsamples)
        tmp_m = np.zeros((nsamples, Nts))
        tmp_v = np.zeros((nsamples, Nts))
        for j in range(0, nsamples):
            mu, v = model2.predict(np.hstack
