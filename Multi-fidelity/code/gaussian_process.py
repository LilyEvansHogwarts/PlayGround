import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys


def chol_inv(L, y):
    '''
    K = L * L.T
    return inv(K)*y
    '''
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP:
    def __init__(self, train_x, train_y, bfgs_iter=100, debug=True):
        self.train_x = train_x
        self.train_y = train_y
        self.debug = debug
        self.bfgs_iter = bfgs_iter
        self.standardization()
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.num_param = 2+self.dim # log_sn2, output_scale, lengthscales
        self.loss = np.inf

    def standardization(self):
        '''
        self.in_mean: (self.dim,) self.in_std: (self.dim,)
        self.out_mean: (self.outdim,) self.out_std: (self.outdim,)
        '''
        # standardize train_x
        self.in_mean = self.train_x.mean(axis=1)
        self.in_std = self.train_x.std(axis=1)
        self.train_x = ((self.train_x.T - self.in_mean)/self.in_std).T
        
        # standardize train_y
        self.out_mean = self.train_y.mean()
        self.out_std = self.train_y.std()
        self.train_y = (self.train_y - self.out_mean)/self.out_std
        if self.debug:
            print('self.in_mean',self.in_mean)
            print('self.in_std',self.in_std)
            print('self.out_mean',self.out_mean)
            print('self.out_std',self.out_std)
            print('train_x.shape',self.train_x.shape)
            print('train_y.shape',self.train_y.shape)

    def rand_theta(self, scale=1):
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def kernel(self, x, xp, hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims((x.T/lengthscales).T,2) - np.expand_dims((xp.T/lengthscales).T,1)
        if self.debug:
            print('x',x.shape)
            print('xp',xp.shape)
            print('expand x',np.expand_dims((x.T/lengthscales).T,2).shape)
            print('expand xp',np.expand_dims((xp.T/lengthscales).T,1).shape)
            print('diff.shape',diffs.shape)
        return output_scale * np.exp(-0.5*np.sum(diffs**2,axis=0))

    def neg_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]

        K = self.kernel(self.train_x, self.train_x, hyp) + sn2*np.eye(self.num_train)
        L = np.linalg.cholesky(K)
        if self.debug:
            print('K.shape',K.shape)
            print('L.shape',L.shape)

        neg_likelihood = np.inf
        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.train_y.T)
        neg_likelihood = 0.5*(np.dot(self.train_y, alpha).sum() + self.num_train * np.log(2*np.pi)) + logDetK
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = np.copy(theta)
            self.K = K.copy()
            self.L = L.copy()

        return neg_likelihood

    def train(self, theta):
        self.loss = np.inf
        theta0 = np.copy(theta)
        self.theta = theta0.copy()

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m = 100, iprint=1)
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

        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        self.alpha = chol_inv(self.L, self.train_y.T)
        print('Finished training process')

    def predict(self, test_x):
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        tmp = self.kernel(test_x, self.train_x, hyp)
        py = self.out_mean + np.dot(tmp, self.alpha) * self.out_std
        ps2 = sn2 + self.kernel(test_x, test_x, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        ps2 = ps2 * (self.out_std**2)
        return py, ps2

