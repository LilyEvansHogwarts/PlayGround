import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys

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
        return theta
    
    def kernel1(self, x, xp, hyp, active_dims=None):
        if active_dims is None:
            active_dims = np.arange(self.dim)
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims((x[active_dims].T/lengthscales).T, 2) - np.expand_dims((xp[active_dims].T/lengthscales).T, 1)
        return output_scale * np.exp(-0.5*np.sum(diffs**2, axis=0))
    
    def kernel2(self, x, xp, hyp):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:2+self.dim]
        hyp3 = hyp[2+self.dim:]
        return self.kernel1(x, xp, hyp1, active_dims=[self.dim-1]) * self.kernel1(x, xp, hyp2, active_dims=np.arange(self.dim-1)) + self.kernel1(x, xp, hyp3, active_dims=np.arange(self.dim-1))

    def kernel(self, x, xp, hyp):
        if self.k: 
            return self.kernel2(x, xp, hyp)
        else:
            return self.kernel1(x, xp, hyp)
        
    def neg_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]

        K = self.kernel(self.train_x, self.train_x, hyp) + sn2*np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.train_y.T)
        neg_likelihood = 0.5*(np.dot(self.train_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = theta.copy()
            self.K = K.copy()
            self.L = L.copy()

        return neg_likelihood

    def train(self, scale=1.0):
        theta = self.rand_theta(scale)
        self.loss = np.inf
        theta0 = np.copy(theta)
        self.theta = theta0.copy()

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            return nlz

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m = 100, iprint=self.debug)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=10, iprint=self.debug)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('GP. Fail to build GP model')
            sys.exit(1)

        self.alpha = chol_inv(self.L, self.train_y.T)
        print('GP. Finished training process')

    def predict(self, test_x):
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        tmp = self.kernel(test_x, self.train_x, hyp)
        py = np.dot(tmp, self.alpha)
        ps2 = sn2 + self.kernel(test_x, test_x, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        ps2 = np.abs(ps2)
        return py, ps2
    
