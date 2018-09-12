import autograd.numpy as np
from autograd import grad
import traceback
import sys
import pickle
from scipy.optimize import fmin_l_bfgs_b
from .activations import *
from .GP_model import GP_model
import random

class Constr_model:
    def __init__(self, main_f, dataset, dim, outdim, bounds, scale, num_layers, layer_size, act, max_iter, l1, l2, debug=True):
        self.dim = dim
        self.main_f = main_f
        self.outdim = outdim
        self.l1 = np.copy(l1)
        self.l2 = np.copy(l2)
        self.scale = np.copy(scale)
        self.num_layers = np.copy(num_layers)
        self.layer_size = np.copy(layer_size)
        self.act = np.copy(act)
        self.max_iter = np.copy(max_iter)
        self.bounds = np.copy(bounds)
        self.train_x = dataset['train_x'].copy()
        self.test_x = dataset['test_x'].copy()
        self.train_y = dataset['train_y'].copy()
        self.test_y = dataset['test_y'].copy()

        self.main_function = self.construct_model(0)
        self.constr_list = []
        for i in range(1, self.outdim):
            self.constr_list.append(self.construct_model(i))

    def construct_model(self, idx):
        layer_sizes = [self.layer_size[idx]]*self.num_layers[idx]
        activations = [get_act_f(self.act[idx])]*self.num_layers[idx]
        model = GP_model(self.train_x, self.train_y[idx], layer_sizes, activations, bfgs_iter=self.max_iter[idx], l1=self.l1[idx], l2 = self.l2[idx], debug=True)

        theta0 = model.rand_theta(scale=self.scale[idx])
        model.fit(theta0)
        return model

    def rand_x(self):
        x = np.zeros((self.dim, 1))
        for i in range(self.dim):
            x[i] = random.uniform(self.bounds[i, 0], self.bounds[i, 1])
        return x

    def fit(self, x):
        x0 = np.copy(x)
        self.x = np.copy(x)
        self.loss = np.inf
        # self.best_y = self.train_y[0].min()
        best_constr = np.inf
        self.best_y = np.inf
        for i in range(self.train_y.shape[1]):
            constr = np.maximum(self.train_y[1:, i], 0).sum()
            if best_constr > 0 and constr < best_constr:
                best_constr = constr
                self.best_y = self.train_y[0, i]
            elif best_constr <= 0 and constr <= 0 and self.train_y[0, i] < self.best_y:
                best_constr = constr
                self.best_y = self.train_y[0, i]

        def loss(x):
            x = x.reshape(self.dim, int(x.size/self.dim))
            py, ps2 = self.main_function.predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            tmp = (self.best_y - py)/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
            print('py', py, 'ps', ps, 'best_y', self.best_y, 'EI', EI)
            PI = 1.0
            for i in range(len(self.constr_list)):
                py, ps2 = self.constr_list[i].predict(x)
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                PI = PI*cdf(-py/ps)
                if py > 0:
                    EI = 1.0

            loss = - EI*PI
            if loss < self.loss:
                self.loss = loss
                self.x = np.copy(x)

            return loss

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=self.bounds, maxiter=200, m=100, iprint=0)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, gloss, bounds=self.bounds, maxiter=200, m=10, iprint=0)
            except:
                print('Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        print('Optimized loss is %g' % self.loss)
        if(np.isnan(self.loss) or np.isinf(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        print('best_y', self.best_y)
        print('predict', self.main_function.predict(self.x), 'loss', self.loss)
        print('x', self.x.T)
        print('true', self.main_f(self.x).T)

        return self.x

    def predict(self, x):
        pys = np.zeros((self.outdim, x.shape[1]))
        ps2s = np.zeros((self.outdim, x.shape[1]))
        pys[0], ps2 = self.main_function.predict(x)
        ps2s[0] = np.diagonal(ps2)
        for i in range(1, self.outdim):
            pys[i], ps2 = self.constr_list[i-1].predict(x)
            ps2s[i] = np.diagonal(ps2)
        return pys, ps2s






