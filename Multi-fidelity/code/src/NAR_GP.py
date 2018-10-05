import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .GP import GP

class NAR_GP:
    def __init__(self, dataset, bfgs_iter=100, debug=True):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.train()

    def train(self):
        self.model1 = GP(self.low_x, self.low_y, bfgs_iter=self.bfgs_iter, debug=self.debug)
        theta1 = self.model1.rand_theta(scale=0.4)
        self.model1.train(theta1)
        py, ps2 = self.model1.predict(self.high_x)

        x = np.vstack((self.high_x, py.T))
        self.model2 = GP(x, self.high_y, bfgs_iter=self.bfgs_iter, debug=self.debug)
        theta2 = self.model2.rand_theta(scale=0.4)
        self.model2.train(theta2)

    def predict(self, test_x):
        py, ps2 = self.model1.predict(test_x)
        x = np.vstack((test_x, py.T))
        return self.model2.predict(x)

    def predict_tmp(self, test_x):
        nsamples = 200
        py, ps2 = self.model1.predict(test_x)



