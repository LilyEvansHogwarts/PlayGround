import autograd.numpy as np
import traceback
from scipy.optimize import fmin_l_bfgs_b
from autograd import value_and_grad

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP:
    def __init__(self, train_x, train_y, bfgs_iter=100, k=0, debug=False):
        self.k = k
        self.train_x = train_x
        self.train_y = train_y
        self.dim, self.num_train = self.train_x.shape
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y.reshape(-1) - self.mean)/self.std
        self.idx1 = np.array([self.dim-1])
        self.idx2 = np.arange(self.dim-1)

    def kernel1(self, x, xp, hyp):
        # squred exponential kernel
        sf2 = np.exp(hyp[0])
        lengthscale = np.exp(hyp[1:]) + 0.000001
        x = (x.T/lengthscale).T
        xp = (xp.T/lengthscale).T
        diff = 2*np.dot(x.T, xp) - (xp**2).sum(axis=0) - (x**2).sum(axis=0)[:,None]
        return sf2 * np.exp(0.5 * diff)

    def kernel2(self, x, xp, hyp):
        hyp1 = hyp[:2]
        hyp2 = hyp[2:2+self.dim]
        hyp3 = hyp[2+self.dim:]
        k1 = self.kernel1(x[self.idx1], xp[self.idx1], hyp1)
        k2 = self.kernel1(x[self.idx2], xp[self.idx2], hyp2)
        k3 = self.kernel1(x[self.idx2], xp[self.idx2], hyp3)
        return k1*k2 + k3

    def kernel(self, x, xp, hyp):
        if self.k:
            return self.kernel2(x, xp, hyp)
        else:
            return self.kernel1(x, xp, hyp)

    def rand_theta(self, scale):
        if self.k:
            theta = scale * np.random.randn(3+2*self.dim)
            theta[2] = np.maximum(-100, np.log(0.5*(self.train_x[-1].max() - self.train_x[-1].min())))

            for i in range(self.dim-1):
                theta[4+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
                theta[4+self.dim+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        else:
            theta = scale * np.random.randn(2+self.dim)
            for i in range(self.dim):
                theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[0] = np.log(np.std(self.train_y))
        return theta

    def neg_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]

        K = self.kernel(self.train_x, self.train_x, hyp) + sn2*np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        alpha = chol_inv(L, self.train_y)
        logDetK = np.sum(np.log(np.diag(L)))
        NLML = 0.5*(np.dot(self.train_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
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
            print('GP. Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=call_back_funct)
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
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2*np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y)
        print('GP. Finish training')

    def predict(self, test_x):
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        tmp = self.kernel(test_x, self.train_x, hyp)
        py = np.dot(tmp, self.alpha) * self.std + self.mean
        ps2 = sn2 + self.kernel(test_x, test_x, hyp) - np.dot(tmp, chol_inv(self.L, tmp.T))
        ps2 = ps2*(self.std**2)
        return py, ps2

class NARGP:
    def __init__(self, dataset, bfgs_iter=100, debug=False):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug

    def train(self, scale=0.2):
        self.model1 = GP(self.low_x, self.low_y, bfgs_iter=self.bfgs_iter, k=0, debug=self.debug)
        self.model1.train(scale=scale)

        mu, _ = self.model1.predict(self.high_x)
        v = np.concatenate((self.high_x, mu.reshape((1,-1))))
        self.model2 = GP(v, self.high_y, bfgs_iter=self.bfgs_iter, k=1, debug=self.debug)
        self.model2.train(scale=scale)
        print('NAR_GP. Finish training NAR_GP.')

    def predict(self, test_x):
        nsamples = 100
        py1, ps21 = self.model1.predict(test_x)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        tmp_py = np.zeros((nsamples, test_x.shape[1]))
        tmp_ps2 = np.zeros((nsamples, test_x.shape[1]))
        for i in range(nsamples):
            tmp_x = np.concatenate((test_x, Z[i].reshape((1,-1))))
            tmp_py[i], tmp = self.model2.predict(tmp_x)
            tmp_ps2[i] = np.diag(tmp)
        py = tmp_py.mean(axis=0)
        ps2 = tmp_ps2.mean(axis=0) + tmp_py.var(axis=0)
        ps2 = np.abs(ps2)
        return py, ps2

