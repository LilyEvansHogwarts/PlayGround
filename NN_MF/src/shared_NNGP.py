import autograd.numpy as np
import traceback
from scipy.optimize import fmin_l_bfgs_b
from autograd import value_and_grad
from .NN import NN
from .activations import *

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class shared_NNGP:
    def __init__(self, dataset, shared_nn, nns, bfgs_iter=100, debug=False):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.shared_nn = shared_nn
        self.nns = nns
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim, self.num_train = self.train_x.shape
        self.outdim = self.train_y.shape[0]
        self.mean = self.train_y.mean(axis=1)
        self.std = self.train_y.std(axis=1)
        self.train_y = ((self.train_y.T - self.mean)/self.std).T
        # sn2 + sp2 + lengthscale + w
        # self.outdim + self.outdim + self.dim + self.nn[i].num_param
        self.num_param = 2*self.outdim + self.dim + self.shared_nn.num_param + np.sum([self.nns[i].num_param for i in range(self.outdim)])

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        for i in range(self.outdim):
            theta[2*i] = np.log(np.std(self.train_y[i])/2)
            theta[2*i+1] = np.log(np.std(self.train_y[i]))
        for i in range(self.dim):
            theta[2*self.outdim+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def split_theta(self, theta):
        sn2 = np.exp(theta[:self.outdim])
        sp2 = np.exp(theta[self.outdim:2*self.outdim])
        lengthscale = np.maximum(0.000001, np.exp(theta[2*self.outdim:2*self.outdim+self.dim]))
        shared_w = theta[2*self.outdim+self.dim:2*self.outdim+self.dim+self.shared_nn.num_param]
        start_idx = 2*self.outdim + self.dim + self.shared_nn.num_param
        ws = []
        for i in range(self.outdim):
            ws.append(theta[start_idx:start_idx+self.nns[i].num_param])
            start_idx += self.nns[i].num_param
        return sn2, sp2, lengthscale, shared_w, ws

    def calc_Phis(self, shared_w, ws, x):
        shared_Phi = self.shared_nn.predict(shared_w, x)
        Phis = [self.nns[i].predict(ws[i], shared_Phi) for i in range(self.outdim)]
        return Phis

    def neg_likelihood(self, theta):
        NLML = 0
        sn2, sp2, lengthscale, shared_w, ws = self.split_theta(theta)
        x = (self.train_x.T/lengthscale).T
        Phis = self.calc_Phis(shared_w, ws, x)
        for i in range(self.outdim):
            m = self.nns[i].layer_sizes[-1]
            Phi_y = np.dot(Phis[i], self.train_y[i])
            A = np.dot(Phis[i], Phis[i].T) + m*sn2[i]/sp2[i]*np.eye(m)
            L = np.linalg.cholesky(A)

            logDetA = 2*np.sum(np.log(np.diag(L)))
            datafit = ((self.train_y[i]**2).sum() - np.dot(Phi_y, chol_inv(L, Phi_y)))/sn2[i]
            NLML += (datafit + logDetA + self.num_train*np.log(2*np.pi*sn2[i]) - m*np.log(m*sn2[i]/sp2[i]))/2

            # w_nobias = self.nns[i].w_nobias(ws[i])
            # l1_reg = self.l1 * np.abs(w_nobias).sum()
            # l2_reg = self.l2 * np.dot(w_nobias, w_nobias)
            # NLML += l1_reg + l2_reg

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
            print('shared_NNGP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            for i in range(self.outdim):
                theta0[i] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=call_back_funct)
            except:
                print('shared_NNGP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('shared_NNGP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        sn2, sp2, lengthscale, shared_w, ws = self.split_theta(self.theta)
        x = (self.train_x.T/lengthscale).T
        Phis = self.calc_Phis(shared_w, ws, x)
        self.Ls = []
        self.alphas = []
        for i in range(self.outdim):
            m = self.nns[i].layer_sizes[-1]
            A = np.dot(Phis[i], Phis[i].T) + m*sn2[i]/sp2[i]*np.eye(m)
            L = np.linalg.cholesky(A)
            self.Ls.append(L)
            alpha = chol_inv(L, np.dot(Phis[i], self.train_y[i]))
            self.alphas.append(alpha)
        print('shared_NNGP. Finish training')

    def predict(self, test_x):
        sn2, sp2, lengthscale, shared_w, ws = self.split_theta(self.theta)
        x = (test_x.T/lengthscale).T
        phis = self.calc_Phis(shared_w, ws, x)
        pys = np.zeros((self.outdim, test_x.shape[1]))
        ps2s = np.zeros((self.outdim, test_x.shape[1]))
        for i in range(self.outdim):
            pys[i] = np.dot(phis[i].T, self.alphas[i])
            ps2s[i] = np.diag(sn2[i] + sn2[i] * np.dot(phis[i].T, chol_inv(self.Ls[i], phis[i])))
        pys = (pys.T * self.std + self.mean).T
        ps2s = (ps2s.T * (self.std**2)).T
        return pys, ps2s

