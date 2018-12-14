import autograd.numpy as np
import traceback
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b
from .NNGP import *
from .Bagging import Bagging

class GP:
    def __init__(self, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.NN = NN(layer_sizes, activations)
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim, self.num_train = self.train_x.shape
        self.nn_param = self.NN.num_param(self.dim)
        self.num_param = self.dim + 2 + 1 + self.nn_param + self.dim
        self.m = self.NN.layer_sizes[-1]
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y.reshape(-1) - self.mean)/self.std
        self.idx1 = np.array([self.dim-1])
        self.idx2 = np.arange(self.dim-1)

    def SE(self, x, xp, hyp):
        sf2 = np.exp(hyp[0])
        lengthscale = np.maximum(0.000001, np.exp(hyp[1:]))
        x = (x.T/lengthscale).T
        xp = (xp.T/lengthscale).T
        diff = 2*np.dot(x.T, xp) - (xp**2).sum(axis=0) - (x**2).sum(axis=0)[:,None]
        return sf2 * np.exp(0.5*diff)

    def NN_kernel(self, x, xp, lengthscale, hyp, same=1):
        sp2 = np.exp(hyp[0])
        w = hyp[1:]
        if same:
            x = (x.T/lengthscale).T
            phi = self.NN.predict(w, x)
            return np.dot(phi.T, phi) * sp2 / self.m
        else:
            x = (x.T/lengthscale).T
            xp = (xp.T/lengthscale).T
            phi1 = self.NN.predict(w, x)
            phi2 = self.NN.predict(w, xp)
            return np.dot(phi1.T, phi2) * sp2 / self.m

    def kernel(self, x, xp, lengthscale, hyp, same=1):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:3+self.nn_param]
        hyp3 = hyp[3+self.nn_param:]
        k1 = self.SE(x[self.idx1], xp[self.idx1], hyp1)
        k2 = self.NN_kernel(x[self.idx2], xp[self.idx2], lengthscale, hyp2, same=same)
        k3 = self.SE(x[self.idx2], xp[self.idx2], hyp3)
        return k1 * k2 + k3

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y)/2)
        for i in range(self.dim-1):
            theta[1+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[self.dim] = np.log(np.std(self.train_y))
        theta[1+self.dim] = np.maximum(-100, np.log(0.5*(self.train_x[-1].max() - self.train_x[-1].min())))
        theta[2+self.dim] = np.log(np.std(self.train_y))
        theta[2+self.dim+self.nn_param] = np.log(np.std(self.train_y))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(theta[0])
        lengthscale = np.maximum(0.000001, np.exp(theta[1:self.dim]))
        hyp = theta[self.dim:]
        return sn2, lengthscale, hyp

    def neg_likelihood(self, theta):
        sn2, lengthscale, hyp = self.split_theta(theta)
        K = self.kernel(self.train_x, self.train_x, lengthscale, hyp) + sn2 * np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = 2 * np.sum(np.log(np.diag(L)))
        NLML = 0.5*(np.dot(self.train_y, chol_inv(L, self.train_y)) + logDetK + self.num_train*np.log(2*np.pi))

        self.NLML = NLML
        return NLML

    def train(self, scale=0.2):
        theta0 = self.rand_theta(scale=scale)
        self.loss = np.inf
        self.theta = np.copy(theta0)

        def call_back_funct(theta):
            if self.NLML < self.loss:
                self.loss = self.NLML
                self.theta = np.copy(theta)

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = value_and_grad(loss)

        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=100, iprint=self.debug, callback=call_back_funct)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=call_back_funct)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        sn2, lengthscale, hyp = self.split_theta(self.theta)
        K = self.kernel(self.train_x, self.train_x, lengthscale, hyp) + sn2 * np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y)
        print('GP. Finish training')

    def predict(self, test_x):
        sn2, lengthscale, hyp = self.split_theta(self.theta)
        tmp = self.kernel(test_x, self.train_x, lengthscale, hyp, same=0)
        py = np.dot(tmp, self.alpha) * self.std + self.mean
        ps2 = sn2 + self.kernel(test_x, test_x, lengthscale, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        ps2 = ps2 * (self.std**2)
        return py, ps2

'''
class GP:
    def __init__(self, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.NN = NN(layer_sizes, activations)
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim, self.num_train = self.train_x.shape
        self.nn_param = self.NN.num_param(self.dim)
        self.num_param = self.dim + 2 + (1 + self.nn_param)*2
        self.m = self.NN.layer_sizes[-1]
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y.reshape(-1) - self.mean)/self.std
        self.idx1 = np.array([self.dim-1])
        self.idx2 = np.arange(self.dim-1)

    def SE(self, x, xp, hyp):
        sf2 = np.exp(hyp[0])
        lengthscale = np.maximum(0.000001, np.exp(hyp[1:]))
        x = (x.T/lengthscale).T
        xp = (xp.T/lengthscale).T
        diff = 2*np.dot(x.T, xp) - (xp**2).sum(axis=0) - (x**2).sum(axis=0)[:,None]
        return sf2 * np.exp(0.5*diff)

    def NN_kernel(self, x, xp, lengthscale, hyp, same=1):
        sp2 = np.exp(hyp[0])
        w = hyp[1:]
        if same:
            x = (x.T/lengthscale).T
            phi = self.NN.predict(w, x)
            return np.dot(phi.T, phi) * sp2 / self.m
        else:
            x = (x.T/lengthscale).T
            xp = (xp.T/lengthscale).T
            phi1 = self.NN.predict(w, x)
            phi2 = self.NN.predict(w, xp)
            return np.dot(phi1.T, phi2) * sp2 / self.m

    def kernel(self, x, xp, lengthscale, hyp, same=1):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:3+self.nn_param]
        hyp3 = hyp[3+self.nn_param:]
        k1 = self.SE(x[self.idx1], xp[self.idx1], hyp1)
        k2 = self.NN_kernel(x[self.idx2], xp[self.idx2], lengthscale, hyp2, same=same)
        k3 = self.NN_kernel(x[self.idx2], xp[self.idx2], lengthscale, hyp3, same=same)
        return k1 * k2 + k3

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y)/2)
        for i in range(self.dim-1):
            theta[1+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[self.dim] = np.log(np.std(self.train_y))
        theta[1+self.dim] = np.maximum(-100, np.log(0.5*(self.train_x[-1].max() - self.train_x[-1].min())))
        theta[2+self.dim] = np.log(np.std(self.train_y))
        theta[2+self.dim+self.nn_param] = np.log(np.std(self.train_y))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(theta[0])
        lengthscale = np.maximum(0.000001, np.exp(theta[1:self.dim]))
        hyp = theta[self.dim:]
        return sn2, lengthscale, hyp

    def neg_likelihood(self, theta):
        sn2, lengthscale, hyp = self.split_theta(theta)
        K = self.kernel(self.train_x, self.train_x, lengthscale, hyp) + sn2 * np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = 2 * np.sum(np.log(np.diag(L)))
        NLML = 0.5*(np.dot(self.train_y, chol_inv(L, self.train_y)) + logDetK + self.num_train*np.log(2*np.pi))

        self.NLML = NLML
        return NLML

    def train(self, scale=0.2):
        theta0 = self.rand_theta(scale=scale)
        self.loss = np.inf
        self.theta = np.copy(theta0)

        def call_back_funct(theta):
            if self.NLML < self.loss:
                self.loss = self.NLML
                self.theta = np.copy(theta)

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = value_and_grad(loss)

        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=100, iprint=self.debug, callback=call_back_funct)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=call_back_funct)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        sn2, lengthscale, hyp = self.split_theta(self.theta)
        K = self.kernel(self.train_x, self.train_x, lengthscale, hyp) + sn2 * np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y)
        print('GP. Finish training')

    def predict(self, test_x):
        sn2, lengthscale, hyp = self.split_theta(self.theta)
        tmp = self.kernel(test_x, self.train_x, lengthscale, hyp, same=0)
        py = np.dot(tmp, self.alpha) * self.std + self.mean
        ps2 = sn2 + self.kernel(test_x, test_x, lengthscale, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        ps2 = ps2 * (self.std**2)
        return py, ps2
'''

class new_NARGP:
    def __init__(self, num_model, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.num_model = num_model
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

    def train(self, scale=0.2):
        data = {}
        data['train_x'] = self.low_x
        data['train_y'] = self.low_y
        self.model1 = Bagging(NNGP, self.num_model, data, self.layer_sizes, self.activations, bfgs_iter=self.bfgs_iter)
        self.model1.train(scale=scale)
        mu, _ = self.model1.predict(self.high_x)

        data['train_x'] = np.concatenate((self.high_x, mu.reshape((1,-1))))
        data['train_y'] = self.high_y
        self.model2 = Bagging(GP, self.num_model, data, self.layer_sizes, self.activations, bfgs_iter=self.bfgs_iter)
        self.model2.train(scale=0.2)
        print('new_NARGP. Finish training')

    def predict(self, test_x):
        nsamples = 100
        mu, v = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(mu, v, nsamples)
        pys = np.zeros((nsamples, test_x.shape[1]))
        ps2 = np.zeros(test_x.shape[1])
        for i in range(nsamples):
            tmp_x = np.concatenate((test_x, Z[i].reshape((1,-1))))
            pys[i], tmp = self.model2.predict(tmp_x)
            ps2 += np.diag(tmp)/nsamples
        ps2 += pys.var(axis=0)
        py = pys.mean(axis=0)
        return py, ps2

