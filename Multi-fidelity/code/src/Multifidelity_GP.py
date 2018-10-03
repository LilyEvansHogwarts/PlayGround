import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class Multifidelity_GP:
    def __init__(self, low_x, low_y, high_x, high_y, bfgs_iter=100, debug=True):
        self.low_x = low_x
        self.low_y = low_y
        self.high_x = high_x
        self.high_y = high_y
        self.standardization()
        self.y = np.concatenate((self.low_y.reshape(self.low_y.size), self.high_y.reshape(self.high_y.size)))
        self.y = self.y.reshape(1,self.y.size)
        self.dim = self.low_x.shape[0]
        self.num_low = self.low_x.shape[1]
        self.num_high = self.high_x.shape[1]
        self.num = self.num_low + self.num_high
        self.num_param = 5 + 2*self.dim

    def standardization(self):
        self.in_mean = np.concatenate((self.low_x.T, self.high_x.T)).mean(axis=0)
        self.in_std = np.concatenate((self.low_x.T, self.high_x.T)).std(axis=0)
        self.low_x = ((self.low_x.T - self.in_mean)/self.in_std).T
        self.high_x = ((self.high_x.T - self.in_mean)/self.in_std).T

        self.low_out_mean = self.low_y.mean()
        self.low_out_std = self.low_y.std()
        self.low_y = (self.low_y - self.low_out_mean)/self.low_out_std

        self.out_mean = self.high_y.mean()
        self.out_std = self.high_y.std()
        self.high_y = (self.high_y - self.out_mean)/self.out_std

    def rand_theta(self):
        theta = np.random.randn(self.num_param)
        theta[0] = 1.0
        theta[1] = np.log(np.std(self.low_y))
        theta[2] = np.log(np.std(self.high_y))
        for i in range(self.dim):
            theta[4+i] = np.maximum(-100, np.log(0.5*(self.low_x[i].max() - self.low_x[i].min())))
            theta[4+self.dim+i] = np.maximum(-100, np.log(0.5*(self.high_x[i].max() - self.high_x[i].min())))
        return theta

    def split_theta(self, theta):
        rho = theta[0]
        low_sn2 = np.exp(theta[1])
        high_sn2 = np.exp(theta[2])
        low_hyp = theta[3:4+self.dim]
        high_hyp = theta[4+self.dim:]
        return rho, low_sn2, high_sn2, low_hyp, high_hyp

    def kernel(self, x, xp, hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims((x.T/lengthscales).T,2) - np.expand_dims((xp.T/lengthscales).T,1)
        return output_scale * np.exp(-0.5*np.sum(diffs**2,axis=0))

    def neg_likelihood(self, theta):
        rho, low_sn2, high_sn2, low_hyp, high_hyp = self.split_theta(theta)
        K_LL = self.kernel(self.low_x, self.low_x, low_hyp) + low_sn2 * np.eye(self.num_low)
        K_LH = rho * self.kernel(self.low_x, self.high_x, low_hyp)
        K_HH = rho**2 * self.kernel(self.high_x, self.high_x, low_hyp) + self.kernel(self.high_x, self.high_x, high_hyp) + high_sn2 * np.eye(self.num_high)
        K = np.vstack((np.hstack((K_LL, K_LH)),np.hstack((K_LH.T, K_HH))))
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        datafit = np.dot(self.y, chol_inv(L, self.y.T))
        neg_likelihood = 0.5*datafit + logDetK + 0.5*self.num*np.log(2*np.pi)
        neg_likelihood = neg_likelihood.sum()
        if np.isnan(neg_likelihood):
            neg_likelihood = np.inf

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = np.copy(theta)
            self.K = np.copy(K)
            self.L = np.copy(L)

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
            theta0[1] += np.log(10)
            theta0[2] += np.log(10)
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

        self.alpha = chol_inv(self.L, self.y.T)

    def predict(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        rho, low_sn2, high_sn2, low_hyp, high_hyp = self.split_theta(self.theta)
        psi1 = rho * self.kernel(test_x, self.low_x, low_hyp)
        psi2 = rho**2 * self.kernel(test_x, self.high_x, low_hyp) + self.kernel(test_x, self.high_x, high_hyp)
        psi = np.hstack((psi1, psi2))
        py = np.dot(psi, self.alpha)
        py = self.out_mean + py * self.out_std
        beta = chol_inv(self.L, psi.T)
        ps2 = rho**2 * self.kernel(test_x, test_x, low_hyp) + self.kernel(test_x, test_x, high_hyp) - np.dot(psi, beta)
        ps2 = ps2 * (self.out_std**2)
        return py, ps2










