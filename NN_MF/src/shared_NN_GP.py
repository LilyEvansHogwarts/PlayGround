import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .NN import NN

def scale_x(log_lscale, x):
    lscale = np.exp(log_lscale)
    lscale = np.maximum(0.000001, lscale)
    return (x.T/lscale).T

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class shared_NN_GP:
    def __init__(self, dataset, shared_nn, non_shared_nns, bfgs_iter, l1=0, l2=0, debug=True):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.shared_nn = shared_nn
        self.non_shared_nns = non_shared_nns
        self.bfgs_iter = bfgs_iter
        self.l1 = l1
        self.l2 = l2
        self.debug = debug
        self.m = self.shared_nn.layer_sizes[-1]
        self.dim = self.train_x.shape[0]
        self.outdim = self.train_y.shape[0]
        self.num_train = self.train_x.shape[1]
        self.num_param = self.calc_num_params()

    def calc_num_params(self):
        '''
        lengthscales: self.dim
        noise: self.outdim
        self covariance: self.outdim
        shared_nn: self.shared_nn.num_param(self.dim)
        non_shared_nns
        '''
        num_param = self.dim + 2*self.outdim
        self.num_param1 = self.shared_nn.num_param(self.dim)
        self.num_param2 = []
        for i in range(self.outdim):
            self.num_param2.append(self.non_shared_nns[i].num_param(self.m))
        num_param = num_param + self.num_param1 + self.num_param2.sum()
        return num_param

    def rand_theta(self, scale):
        theta = scale * np.random.randn(self.num_param)
        # noise and self covariance
        for i in range(self.outdim):
            theta[i] = np.log(np.std(self.train_y[i])/2)
            theta[i+self.outdim] = np.log(np.std(self.train_y[i]))
        for i in range(self.dim):
            theta[2*self.outdim+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def split_theta(self, theta):
        log_sns = theta[:self.outdim]
        sn2 = np.exp(2*log_sns)
        log_sps = theta[self.outdim:2*self.outdim]
        sp2 = np.exp(2*log_sps)
        log_lscales = theta[2*self.outdim:2*self.outdim+self.dim]
        ws = theta[2*self.outdim+self.dim:]
        return sn2, sp2, log_lscales, ws

    def calc_Phi(self, ws, x):
        w_shared = ws[:self.num_param1]
        w_non_shared = ws[self.num_param1:]
        Phi_shared = self.shared_nn.predict(w_shared, x)
        Phis = []
        start_idx = 0
        for i in range(self.outdim):
            w_tmp = w_non_shared[start_idx:start_idx+self.num_param2[i]]
            Phi_tmp = self.non_shared_nns[i].predict(w_tmp, Phi_shared)
            start_idx += self.num_params[i]
            Phis.append(Phi_tmp)
        return Phis

    def neg_likelihood(self, sn2, sp2, Phi, train_y):
        m = Phi.shape[0]
        Phi_y = np.dot(Phi, train_y)
        A = np.dot(Phi, Phi.T) + (m * sn2 / sp2) * np.eye(m)
        LA = np.linalg.cholesky(A)

        logDetA = 2 * np.log(np.diag(LA)).sum()
        datafit = ((train_y**2).sum() - np.dot(Phi_y, chol_inv(LA, Phi_y)))/sn2
        neg_likelihood = 0.5*(datafit + logDetA + self.num_train * np.log(2*np.pi*sn2) - m * np.log(m * sn2 / sp2))
        neg_likelihood = neg_likelihood.sum()

        if np.isnan(neg_likelihood):
            neg_likelihood = np.inf

        return neg_likelihood

    def loss(self, theta):
        sn2, sp2, log_lscales, ws = self.split_theta(theta)
        scaled_x = scale_x(log_lscales, self.train_x)
        Phis = self.calc_Phi(ws, scaled_x)
        loss = 0.0
        for i in range(self.outdim):
            loss += self.neg_likelihood(sn2[i], sp2[i], Phis[i], self.train_y[i])
        return loss

    def train(self, scale=0.2):
        theta = self.rand_theta(scale=scale)
        theta0 = np.copy(theta)
        self.best_loss = np.inf
        self.theta = np.copy(theta)

        def lossfit(theta):
            tmp_loss = self.loss(theta)
            if tmp_loss < self.best_loss:
                self.best_loss = tmp_loss
                self.theta = np.copy(theta)
            return tmp_loss

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=100, iprint=self.debug)
        except np.linalg.LinAlgError:
            print('shared_NN_GP. Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0 += np.log(10)
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=10, iprint=self.debug)
            except:
                print('shared_NN_GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('shared_NN_GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        if(np.isnan(self.best_loss) or np.isinf(self.best_loss)):
            print('shared_NN_GP. Fail to build GP model')
            sys.exit(1)

        sn2, sp2, log_lscales, ws = self.split_theta(self.theta)
        self.Phis = self.calc_Phi(ws, scale_x(log_lscales, self.train_x))
        self.LAs = []
        self.alphas = []
        for i in range(self.outdim):
            m = self.Phis[i].shape[0]
            A = np.dot(self.Phis[i], self.Phis[i].T) + (m * sn2 / sp2) * np.eye(m)
            LA = np.linalg.cholesky(A)
            alpha = chol_inv(LA, np.dot(self.Phis[i], self.train_y[i]))
            self.LAs.append(LA)
            self.alphas.append(alpha)

    def predict(self, test_x, is_diag=1):
        sn2, sp2, log_lscales, ws = self.split_theta(self.theta)
        Phis_test = self.calc_Phi(log_lscales, scale_x(ws, test_x))
        py = np.zeros((self.outdim, num_test))
        if is_diag:
            ps2 = np.zeros((self.outdim, num_test))
            for i in range(self.outdim):
                py[i] = np.dot(Phi_test[i].T, self.alphas[i])
                ps2[i] = sn2 + sn2 * (Phi_test[i].T * chol_inv(self.LAs[i], Phi_test[i]).T).sum(axis=1)
        else:
            ps2 = np.zeros((self.outdim, num_test, num_test))
            for i in range(self.outdim):
                py[i] = np.dot(Phi_test[i].T, self.alphas[i])
                ps2[i] = sn2 + sn2 * np.dot(Phi_test[i].T, chol_inv(self.LAs[i], Phi_test[i]))
        return py, ps2

        

        

