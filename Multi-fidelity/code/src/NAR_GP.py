import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .GP import *

class NAR_GP:
    def __init__(self, dataset, scale=1.0, bfgs_iter=100, debug=True):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.scale = scale
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.low_x.shape[0]
        self.num_param = 5 + 2*self.dim
        self.num_low = self.low_x.shape[1]
        self.num_high = self.high_x.shape[1]

    def rand_theta(self, scale=1.0):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.high_y))
        theta[2] = np.maximum(-100, np.log(0.5*(self.low_y.main() - self.low_y.min())))
        for i in range(self.dim):
            theta[4+i] = np.maximum(-100, np.log(0.5*(self.high_x[i].max() - self.high_x[i].min())))
            theta[5+self.dim+i] = np.maximum(-100, np.log(0.5*(self.high_x[i].max() - self.high_x[i].min())))
        return theta
    
    def kernel(self, x, xp, hyp):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:3+self.dim]
        hyp3 = hyp[3+self.dim:]
        return RBF(x, xp, hyp1, [self.dim])*RBF(x, xp, hyp2, np.arange(self.dim)) + RBF(x, xp, hyp3, np.arange(self.dim))
    
    def neg_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]

        x = np.concatenate((self.high_x, self.mu.T))
        K = self.kernel(x, x, hyp) + sn2*np.eye(self.num_high)
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.high_y.T)
        neg_likelihood = 0.5*(np.dot(self.high_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
        neg_likelihood = neg_likelihood.sum()
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = theta.copy()
            self.K = K.copy()
            self.L = L.copy()

        return neg_likelihood

    def train(self, theta):
        dataset = {}
        dataset['train_x'] = self.low_x
        dataset['train_y'] = self.low_y
        model1 = GP(dataset, bfgs_iter=self.bfgs_iter, debug=self.debug)
        theta1 = model1.rand_theta(scale=self.scale)
        model1.train(theta1)
        self.model1 = model1

        mu, v = self.model1.predict(self.high_x)
        self.mu = mu

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

        self.alpha = chol_inv(self.L, self.high_y.T)
        print('Finish training process')

    

        








