import autograd.numpy as np
from scipy.optimize import fmin_l_bfgs_b
from autograd import value_and_grad
import traceback
from .activations import *
from .NN import NN
from .scaled_NNGP import scaled_NNGP
from .Bagging import Bagging
from .GP import GP


class scaled_NN_NARGP:
    def __init__(self, num_model, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.num_model = num_model

    def train(self, scale=0.2):
        data = {}
        data['train_x'] = self.low_x
        data['train_y'] = self.low_y
        self.model1 = Bagging(scaled_NNGP, self.num_model, data, self.layer_sizes, self.activations, l1=self.l1, l2=self.l2, bfgs_iter=self.bfgs_iter, debug=self.debug)
        self.model1.train(scale=scale)
        mu, _ = self.model1.predict(self.high_x)
        tmp = np.concatenate((self.high_x, mu.reshape((1,-1))))
        data['train_x'] = tmp
        data['train_y'] = self.high_y
        self.model2 = GP(data, k=1, bfgs_iter=self.bfgs_iter, debug=self.debug)
        self.model2.train(scale=scale)
        print('scaled_NN_NARGP. Finish training')

    def predict(self, test_x):
        nsamples = 100
        py1, ps21 = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        pys = np.zeros((nsamples, test_x.shape[1]))
        ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(nsamples):
            pys[i], tmp = self.model2.predict(np.concatenate((test_x, py1.reshape((1,-1)))))
            ps2 += tmp/nsamples
        ps2 = ps2 + pys.var(axis=0)
        py = pys.mean(axis=0)
        return py, ps2



