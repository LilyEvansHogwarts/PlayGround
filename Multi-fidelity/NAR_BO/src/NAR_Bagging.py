from .Bagging import Bagging
import autograd.numpy as np

class NAR_Bagging:
    def __init__(self, num_models, dataset, bfgs_iter=100, debug=True):
        self.num_models = num_models
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
        self.model1 = Bagging(self.num_models, dataset, bfgs_iter=self.bfgs_iter, debug=self.debug)
        self.model1.train(scale=scale)

        mu, v = self.model1.predict(self.high_x)
        dataset['train_x'] = np.concatenate((self.high_x, mu.reshape(1,-1)))
        dataset['train_y'] = self.high_y
        self.model2 = Bagging(self.num_models, dataset, bfgs_iter=self.bfgs_iter, debug=self.debug, k=1)
        self.model2.train(scale=scale)

    def predict(self, test_x):
        nsamples = 100
        num_test = test_x.shape[1]
        py1, ps21 = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        if self.debug:
            print('Z.shape',Z.shape)
            print('Z[0,:].shape', Z[0,:].shape)
            print('Z[0,:][None,:].shape', Z[0,:][None,:].shape)
        
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

    def predict_low(self, test_x):
        return self.model1.predict(test_x)

