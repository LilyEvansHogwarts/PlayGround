import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .NN import NN
from .activations import *

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class NN_GP:
    def __init__(self, train_x, train_y, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=True):
        self.train_x = train_x
        self.train_y = train_y
        self.standardization()
        self.nn = NN(layer_sizes, activations)
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.m = self.nn.layer_sizes[-1]
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.num_param = 2+self.nn.num_param(self.dim)
        self.loss = np.inf

    def standardization(self):
        self.in_mean = self.train_x.mean(axis=1)
        self.in_std = self.train_x.std(axis=1)
        self.train_x = ((self.train_x.T - self.in_mean)/self.in_std).T
        self.out_mean = self.train_y.mean()
        self.out_std = self.train_y.std()
        self.train_y = (self.train_y - self.out_mean)/self.out_std
    
    def rand_theta(self, scale=1.0):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y)/2)
        theta[1] = np.log(np.std(self.train_y))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(2 * theta[0])
        sp2 = np.exp(2 * theta[1])
        w = theta[2:]
        return sn2, sp2, w

    def neg_likelihood(self, theta):
        sn2, sp2, w = self.split_theta(theta)
        Phi = self.nn.predict(w, self.train_x)
        A = np.dot(Phi, Phi.T) + self.m * sn2 / sp2 * np.eye(self.m)
        LA = np.linalg.cholesky(A)

        Phi_y = np.dot(Phi, self.train_y.T)
        datafit = (np.dot(self.train_y, self.train_y.T) - np.dot(Phi_y.T, chol_inv(LA, Phi_y)))/sn2
        logDetA = np.sum(np.log(np.diag(LA)))
        neg_likelihood = 0.5*datafit + logDetA + 0.5 * self.num_train * np.log(2*np.pi*sn2) - 0.5 * self.m * np.log(self.m * sn2 / sp2)
        neg_likelihood = neg_likelihood.sum()
        if(np.isnan(neg_likelihood)):
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

    def train(self, theta):
        self.loss = np.inf
        theta0 = np.copy(theta)
        self.theta = np.copy(theta)

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=10, iprint=1)
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

        sn2, sp2, w = self.split_theta(self.theta)
        Phi = self.nn.predict(w, self.train_x)
        Phi_y = np.dot(Phi, self.train_y.T)
        self.alpha = chol_inv(self.LA, Phi_y)

    def predict(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        sn2, sp2, w = self.split_theta(self.theta)
        phi = self.nn.predict(w, test_x)
        py = self.out_mean + np.dot(phi.T, self.alpha) * self.out_std
        ps2 = sn2 + sn2 * np.dot(phi.T, chol_inv(self.LA, phi))
        ps2 = ps2 * (self.out_std**2)
        return py, ps2


