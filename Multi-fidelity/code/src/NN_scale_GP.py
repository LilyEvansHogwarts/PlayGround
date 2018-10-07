import autograd.numpy as np
from autograd import grad
from .NN import NN
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .activations import *

def scale_x(log_lscale, x):
    lscale = np.exp(log_lscale)
    return (x.T/lscale).T

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class NN_scale_GP:
    def __init__(self, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=True):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.nn = NN(layer_sizes, activations)
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.m = self.nn.layer_sizes[-1]
        self.num_param = 2 + self.dim + self.nn.num_param(self.dim)
        self.train_y = self.train_y.reshape(-1)

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y)/2)
        theta[1] = np.log(np.std(self.train_y))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(2 * theta[0])
        sp2 = np.exp(2 * theta[1])
        log_lscale = theta[2:2+self.dim]
        w = theta[2+self.dim:]
        return sn2, sp2, log_lscale, w

    def neg_likelihood(self, theta):
        sn2, sp2, log_lscale, w = self.split_theta(theta)
        scaled_x = scale_x(log_lscale, self.train_x)
        Phi = self.nn.predict(w, scaled_x)
        Phi_y = np.dot(Phi, self.train_y.T)
        A = np.dot(Phi, Phi.T) + self.m * sn2 / sp2 * np.eye(self.m)
        LA = np.linalg.cholesky(A)

        logDetA = 2*np.log(np.diag(LA)).sum()
        datafit = (np.dot(self.train_y, self.train_y.T) - np.dot(Phi_y.T, chol_inv(LA, Phi_y)))/sn2
        neg_likelihood = 0.5*(datafit + logDetA + self.num_train*np.log(2*np.pi*sn2) - self.m*np.log(self.m*sn2/sp2))
        neg_likelihood = neg_likelihood.sum()
        if np.isnan(neg_likelihood):
            neg_likelihood = np.inf

        w_nobias = self.nn.w_nobias(w, self.dim)
        l1_reg = self.l1 * np.sum(np.abs(w_nobias))
        l2_reg = self.l2 * np.dot(w_nobias, w_nobias.T)
        neg_likelihood += l1_reg + l2_reg

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = np.copy(theta)
            self.A = A.copy()
            self.LA = LA.copy()

        return neg_likelihood

    def train(self, scale=1.0):
        theta = self.rand_theta(scale)
        self.loss = np.inf
        theta0 = np.copy(theta)
        self.theta = np.copy(theta)

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=100, iprint=self.debug)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=10, iprint=self.debug)
            except:
                print('Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        if(np.isnan(self.loss) or np.isinf(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        sn2, sp2, log_lscale, w = self.split_theta(self.theta)
        Phi = self.nn.predict(w, scale_x(log_lscale, self.train_x))
        self.alpha = chol_inv(self.LA, np.dot(Phi, self.train_y.T))

    def predict(self, test_x):
        sn2, sp2, log_lscale, w = self.split_theta(self.theta)
        scaled_x = scale_x(log_lscale, test_x)
        phi = self.nn.predict(w, scaled_x)
        py = np.dot(phi.T, self.alpha)
        ps2 = sn2 + sn2 * np.dot(phi.T, chol_inv(self.LA, phi))
        return py, ps2



