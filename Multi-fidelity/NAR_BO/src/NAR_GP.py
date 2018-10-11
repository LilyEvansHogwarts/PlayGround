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
        num_test = test_x.shape[1]
        py1, ps21 = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        if self.debug:
            print('Z.shape',Z.shape)
            print('Z[0,:].shape', Z[0,:].shape)
            print('Z[0,:][None,:].shape', Z[0,:][None,:].shape)

        x = np.tile(test_x, nsamples)
        x = np.concatenate((x, Z.reshape(1,-1)))
        py, ps2 = self.model2.predict(x)
        py = py.reshape(-1,num_test)
        ps2 = np.diag(ps2).reshape(-1,num_test).mean(axis=0) + py.var(axis=0)
        ps2 = np.abs(ps2)
        py = py.mean(axis=0)
        return py1, ps21, py, ps2



        '''
        tmp_m = np.zeros((nsamples, num_test))
        tmp_v = np.zeros((num_test, num_test))
        
        for j in range(nsamples):
            py2, ps22 = self.model2.predict(np.concatenate((test_x, Z[j].reshape(1,-1))))
            tmp_m[j] = py2
            tmp_v += ps22/nsamples

        py = tmp_m.mean(axis=0)
        ps2 = tmp_v + tmp_m.var(axis=0)
        ps2 = np.abs(ps2)
        return py1, ps21, py, ps2
        '''

    def predict_low(self, test_x):
        return self.model1.predict(test_x)



