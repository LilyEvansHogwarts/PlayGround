import autograd.numpy as np
from scipy.optimize import fmin_l_bfgs_b
import traceback
from autograd import value_and_grad
from NN import NN
from activations import *

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class scaled_NNGP:
    def __init__(self, train_x, train_y, layer_sizes, activations, l1=0, l2=0, bfgs_iter=2000, debug=False):
        self.train_x = train_x
        self.train_y = train_y 
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.NN = NN(layer_sizes, activations)
        self.dim, self.num_train = self.train_x.shape
        self.num_param = 2 + self.dim + self.NN.num_param(self.dim)
        self.m = layer_sizes[-1]
        self.l1 = l1
        self.l2 = l2

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y)/2)
        theta[1] = np.log(np.std(self.train_y))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(2*theta[0])
        sp2 = np.exp(2*theta[1])
        lengthscale = np.exp(theta[2:2+self.dim])
        w = theta[2+self.dim:]
        return sn2, sp2, lengthscale, w

    def neg_likelihood(self, theta):
        sn2, sp2, lengthscale, w = self.split_theta(theta)
        x = (self.train_x.T/lengthscale).T
        Phi = self.NN.predict(w, x)
        Phi_y = np.dot(Phi, self.train_y)
        A = np.dot(Phi, Phi.T) + (self.m*sn2/sp2)*np.eye(self.m)
        L = np.linalg.cholesky(A)

        logDetA = 2*np.sum(np.log(np.diag(L)))
        datafit = ((self.train_y**2).sum() - np.dot(Phi_y, chol_inv(L, Phi_y)))/sn2
        NLML = 0.5*(datafit + logDetA + self.num_train*np.log(2*np.pi*sn2) - self.m*np.log(self.m*sn2/sp2))
        
        w_nobias = self.NN.w_nobias(w, self.dim)
        l1_reg = self.l1 * np.abs(w_nobias).sum()
        l2_reg = self.l2 * np.dot(w_nobias, w_nobias)
        NLML += l1_reg + l2_reg

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
            print('NN_GP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=call_back_funct)
            except:
                print('GP_NN. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP_NN. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        sn2, sp2, lengthscale, w = self.split_theta(self.theta)
        x = (self.train_x.T/lengthscale).T
        Phi = self.NN.predict(w, x)
        Phi_y = np.dot(Phi, self.train_y)
        A = np.dot(Phi, Phi.T) + (self.m*sn2/sp2)*np.eye(self.m)
        self.L = np.linalg.cholesky(A)
        self.alpha = chol_inv(self.L, Phi_y)
        print('GP_NN. Finish training model')

    def predict(self, test_x):
        sn2, sp2, lengthscale, w = self.split_theta(self.theta)
        test_x = (test_x.T/lengthscale).T
        phi = self.NN.predict(w, test_x)
        py = np.dot(phi.T, self.alpha)
        ps2 = sn2 + sn2 * np.dot(phi.T, chol_inv(self.L, phi))
        return py, ps2


class Bagging:
    def __init__(self, num_model, train_x, train_y, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.train_x = train_x
        self.train_y = train_y
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y.reshape(-1) - self.mean)/self.std
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.num_model = num_model

    def train(self, scale=0.2):
        self.models = []
        for i in range(self.num_model):
            model = scaled_NNGP(self.train_x, self.train_y, self.layer_sizes, self.activations, l1=self.l1, l2=self.l2, bfgs_iter=self.bfgs_iter, debug=self.debug)
            model.train(scale=scale)
            self.models.append(model)

    def predict(self, test_x):
        pys = np.zeros((self.num_model, test_x.shape[1]))
        ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(self.num_model):
            pys[i], tmp = self.models[i].predict(test_x)
            ps2 += tmp/self.num_model
        py = pys.mean(axis=0) * self.std + self.mean
        ps2 = ps2 + pys.var(axis=0)
        ps2 = ps2*(self.std**2)
        return py, ps2
        

