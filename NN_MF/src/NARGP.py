import autograd.numpy as np
import traceback
from scipy.optimize import fmin_l_bfgs_b
from autograd import value_and_grad
from .GP import GP

class NARGP:
    def __init__(self, dataset, bfgs_iter=100, debug=False):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug

    def train(self, scale=0.2):
        data = {}
        data['train_x'] = self.low_x
        data['train_y'] = self.low_y
        self.model1 = GP(data, k=0, bfgs_iter=self.bfgs_iter, debug=self.debug)
        self.model1.train(scale=scale)

        mu, _ = self.model1.predict(self.high_x)
        v = np.concatenate((self.high_x, mu.reshape((1,-1))))
        data['train_x'] = v
        data['train_y'] = self.high_y
        self.model2 = GP(data, k=1, bfgs_iter=self.bfgs_iter, debug=self.debug)
        self.model2.train(scale=scale)
        print('NARGP. Finish training')

    def predict(self, test_x):
        nsamples = 100
        py1, ps21 = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        tmp_py = np.zeros((nsamples, test_x.shape[1]))
        tmp_ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(nsamples):
            tmp_x = np.concatenate((test_x, Z[i].reshape((1,-1))))
            tmp_py[i], tmp = self.model2.predict(tmp_x)
            tmp_ps2 += tmp/nsamples
        py = tmp_py.mean(axis=0)
        ps2 = tmp_ps2 + tmp_py.var(axis=0)
        ps2 = np.abs(ps2)
        return py, ps2

