import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b, minimize
import traceback
import sys
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad


def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP:
    def __init__(self, dataset, bfgs_iter=100, debug=True, k=0):
        self.k = k
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.train_y = self.train_y.reshape(-1)
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y - self.mean)/self.std
        self.idx1 = [self.dim-1]
        self.idx2 = np.arange(self.dim-1)


    def rand_theta(self, scale):
        if self.k: # kernel2
            # sn2 + (output_scale + lengthscale) + (output_scale + lengthscales) * 2
            # 1 + 2 + (1 + self.dim - 1)*2 = 3 + 2*self.dim
            theta = scale * np.random.randn(3 + 2*self.dim)
            theta[2] = np.maximum(-100, np.log(0.5*(self.train_x[self.dim-1].max() - self.train_x[self.dim-1].min())))
            for i in range(self.dim-1):
                tmp = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
                theta[4+i] = tmp
                theta[4+self.dim+i] = tmp
        else: # kernel1 RBF
            # sn2 + output_scale + lengthscales, 1 + 1 + self.dim
            theta = scale * np.random.randn(2 + self.dim)
            for i in range(self.dim):
                theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[0] = np.log(np.std(self.train_y)) # sn2
        # theta[1] = np.log(np.std(self.train_y))
        return theta
    
    def kernel1(self, x, xp, hyp):
        # if active_dims is None:
        #     active_dims = np.arange(self.dim)
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:]) + 0.000001
        # lengthscales = lengthscales + 0.000001
        # diffs = np.expand_dims((x[active_dims].T/lengthscales).T, 2) - np.expand_dims((xp[active_dims].T/lengthscales).T, 1)
        # return output_scale * np.exp(-0.5*np.sum(diffs**2, axis=0))
        x = (x.T/lengthscales).T
        xp = (xp.T/lengthscales).T
        diffs = (x**2).sum(axis=0)[:, None] + (xp**2).sum(axis=0) - 2*np.dot(x.T, xp)
        return output_scale * np.exp(-0.5*diffs)
    
    def kernel2(self, x, xp, hyp):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:2+self.dim]
        hyp3 = hyp[2+self.dim:]
        return self.kernel1(x[self.idx1], xp[self.idx1], hyp1) * self.kernel1(x[self.idx2], xp[self.idx2], hyp2) + self.kernel1(x[self.idx2], xp[self.idx2], hyp3)

    def kernel(self, x, xp, hyp):
        if self.k: 
            return self.kernel2(x, xp, hyp)
        else:
            return self.kernel1(x, xp, hyp)

    def neg_log_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]
         
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2 * np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.train_y.T)
        neg_likelihood = 0.5*(np.dot(self.train_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
        self.tmp = neg_likelihood
        return neg_likelihood

    def train(self, scale=0.2):
        theta0 = self.rand_theta(scale)
        self.loss = np.inf
        self.theta = np.copy(theta0)

        def loss(theta):
            nlz = self.neg_log_likelihood(theta)
            return nlz

        def callback(theta):
            if self.tmp < self.loss:
                self.loss = self.tmp
                self.theta = np.copy(theta)

        gloss = value_and_grad(loss)
        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=100, iprint=self.debug, callback=callback)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=callback)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())
        
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2 * np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y.T)
        if self.k:
            self.for_diag = np.exp(self.theta[1]) * np.exp(self.theta[3]) + np.exp(self.theta[3+self.dim])
        else:
            self.for_diag = np.exp(self.theta[1])
        print('GP. Finished training process.')

    def predict(self, test_x, is_diag=1):
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        tmp = self.kernel(test_x, self.train_x, hyp)
        py = np.dot(tmp, self.alpha)
        '''
        ps2 = sn2 + self.kernel(test_x, test_x, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        if is_diag:
            ps2 =np.diag(ps2)
        '''
        tmp1 = chol_inv(self.L, tmp.T)
        # ps2 = -np.dot(tmp, chol_inv(self.L, tmp.T)) 
        if is_diag:
            ps2 = self.for_diag + sn2 - (tmp*tmp1.T).sum(axis=1)
        else:
            ps2 = sn2 - np.dot(tmp, tmp1) + self.kernel(test_x, test_x, hyp)
        ps2 = np.abs(ps2)
        py = py * self.std + self.mean
        ps2 = ps2 * (self.std**2)
        return py, ps2

