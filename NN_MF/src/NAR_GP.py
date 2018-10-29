import autograd.numpy as np
from autograd import grad
from .NN_GP import Bagging
from .GP import GP

class NAR_GP:
    def __init__(self, num_models, dataset, layer_sizes, activations, bfgs_iter, l1=0, l2=0, debug=True):
        self.num_models = num_models
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y'].reshape(-1)
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y'].reshape(-1)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bfgs_iter = bfgs_iter
        self.l1 = l1
        self.l2 = l2
        self.debug = debug

    def train(self, scale=1.0):
        dataset = {}
        dataset['train_x'] = self.low_x
        dataset['train_y'] = self.low_y
        model1 = Bagging(self.num_models, dataset, self.layer_sizes, self.activations, self.bfgs_iter, l1=self.l1, l2=self.l2, debug=self.debug)
        model1.train(scale=scale)
        self.model1 = model1

        mu, v = self.model1.predict(self.high_x)
        dataset['train_x'] = np.concatenate((self.high_x, mu.reshape(1,-1)))
        dataset['train_y'] = self.high_y
        model2 = GP(dataset, self.bfgs_iter, debug=self.debug, k=1)
        model2.train(scale=scale)
        self.model2 = model2
        print('NAR_GP. Finish training process')

    def predict_low(self, test_x):
        return self.model1.predict(test_x)

    def predict(self, test_x):
        py1, ps21 = self.model1.predict(test_x)
        x = np.concatenate((test_x, py1.reshape(1,-1)))
        py, ps2 = self.model2.predict(x)
        return py, ps2

    def predict_for_wEI(self, test_x):
        nsamples = 200
        num_test = test_x.shape[1]
        py1, ps21 = self.model1.predict(test_x, is_diag=0)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)

        x = np.tile(test_x, nsamples)
        x = np.concatenate((x, Z.reshape(1,-1)))
        py, ps2 = self.model2.predict(x)
        py = py.reshape(-1, num_test)
        ps2 = ps2.reshape(-1, num_test).mean(axis=0) + py.var(axis=0)
        py = py.mean(axis=0)
        return py, ps2
